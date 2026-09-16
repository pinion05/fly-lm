#!/usr/bin/env python3
"""Vast.ai 인스턴스 헌팅-런처: SSH 프록시가 안 떠서 실패하면 자동으로 다른 오퍼로 갈아탄다.

사용법: python3 vast_hunt.py <launcher.sh> <업로드파일1> [<업로드파일2> ...] [--min-ram 20] [--budget 3]
- launcher.sh: SSH 뚫리면 /root/launcher.sh로 올려 nohup 실행
- 업로드파일: /root/work/ 아래 원래 파일명으로 업로드
전략: 후보 오퍼를 DLP/$ 순으로 시도 → 인스턴스 생성 → 최대 N분 SSH 대기 →
  실패(프록시 안 뜸/인증거부/actual!=running)면 destroy하고 다음 오퍼로.
"""
import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

SSH_BASE = ["-o", "StrictHostKeyChecking=no", "-o", "ControlMaster=no",
            "-o", "ControlPath=none", "-i", str(Path.home() / ".ssh/id_vastai")]


def run(cmd, timeout=90, **kw):
    return subprocess.run(cmd, capture_output=True, text=True, timeout=timeout, **kw)


def search_offers(min_ram_gb, exclude):
    r = run(["vastai", "search", "offers",
             f"num_gpus=1 gpu_ram>={min_ram_gb} reliability>0.98",
             "--raw"], timeout=120)
    d = json.loads(r.stdout)
    offers = d if isinstance(d, list) else d.get("offers", d)
    ok = [o for o in offers
          if o.get("dph_total", 0) > 0 and not o.get("is_bid", False)
          and o.get("duration", 0) >= 86400 and o["id"] not in exclude]
    ok.sort(key=lambda x: x.get("dlperf", 0) / x["dph_total"], reverse=True)
    return ok[:8]


def try_instance(offer, launcher, files, wait_min):
    o_id = offer["id"]
    r = run(["vastai", "create", "instance", str(o_id),
             "--image", "nvidia/cuda:12.4.1-cudnn-devel-ubuntu22.04",
             "--disk", "80", "--ssh"], timeout=120)
    import re as _re
    m = _re.search(r"'new_contract': (\d+)", r.stdout)
    if not m:
        print(f"[{o_id}] create failed: {r.stdout.strip()[:150]}")
        return None
    contract = int(m.group(1))
    print(f"[{o_id}] created contract {contract} ({offer.get('gpu_name')} ${offer['dph_total']:.3f}/h)")

    # SSH 주소/포트 나올 때까지 대기
    host = port = None
    for _ in range(12):  # 최대 6분
        time.sleep(30)
        s = run(["vastai", "show", "instance", str(contract)], timeout=60).stdout
        import re
        m = re.search(r"(ssh\d+\.vast\.ai)\s+(\d+)", s)
        if m:
            host, port = m.group(1), m.group(2)
            break
        st = run(["vastai", "show", "instance", str(contract), "--raw"], timeout=60)
        try:
            actual = json.loads(st.stdout).get("actual_status")
        except Exception:
            actual = "?"
        if actual in ("error", "stopped"):
            print(f"[{o_id}] instance {actual} — abandoning")
            run(["vastai", "destroy", "instance", str(contract), "-y"], timeout=60)
            return None
    if not host:
        print(f"[{o_id}] no SSH endpoint — destroy & next")
        run(["vastai", "destroy", "instance", str(contract), "-y"], timeout=60)
        return None

    # SSH 접속 대기
    up = False
    for i in range(wait_min * 2):
        r = run(["ssh", *SSH_BASE, "-p", port, f"root@{host}", "echo UP"], timeout=30)
        if "UP" in r.stdout:
            up = True
            break
        if i % 6 == 5:
            run(["vastai", "attach", "ssh", str(contract),
                 Path.home().joinpath(".ssh/id_vastai.pub").read_text().strip()], timeout=60)
        time.sleep(30)
    if not up:
        print(f"[{o_id}] SSH never came up ({host}:{port}) — destroy & next")
        run(["vastai", "destroy", "instance", str(contract), "-y"], timeout=60)
        return None

    # 업로드 + 런치
    run(["scp", *SSH_BASE, "-P", port, launcher, f"root@{host}:/root/launcher.sh"], timeout=300)
    run(["ssh", *SSH_BASE, "-p", port, f"root@{host}", "mkdir -p /root/work"], timeout=30)
    for f in files:
        dest = f"/root/work/{Path(f).name}"
        r = run(["scp", *SSH_BASE, "-P", port, f, f"root@{host}:{dest}"], timeout=600)
        if r.returncode != 0:
            print(f"[{o_id}] upload failed {f}: {r.stderr[:100]}")
            run(["vastai", "destroy", "instance", str(contract), "-y"], timeout=60)
            return None
        print(f"[{o_id}] uploaded {Path(f).name}")
    r = run(["ssh", *SSH_BASE, "-p", port, f"root@{host}",
             "nohup bash /root/launcher.sh > /root/launch.log 2>&1 & sleep 2; echo LAUNCHED"], timeout=60)
    if "LAUNCHED" in r.stdout:
        print(f"[{o_id}] LAUNCHED — instance {contract} ({host}:{port})")
        return {"id": contract, "host": host, "port": port, "offer": o_id}
    print(f"[{o_id}] launch cmd failed: {r.stderr[:100]}")
    run(["vastai", "destroy", "instance", str(contract), "-y"], timeout=60)
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("launcher")
    ap.add_argument("files", nargs="*")
    ap.add_argument("--min-ram", type=int, default=20)
    ap.add_argument("--wait", type=int, default=8, help="SSH 대기 분")
    ap.add_argument("--exclude", type=str, default="", help="콤마 구분 offer id 제외목록")
    args = ap.parse_args()
    exclude = {int(x) for x in args.exclude.split(",") if x.strip()}

    for offer in search_offers(args.min_ram, exclude):
        res = try_instance(offer, args.launcher, args.files, args.wait)
        if res:
            (Path.home() / "fly-lm/.dash_state.json").write_text(json.dumps(res))
            print(json.dumps(res))
            return 0
    print("ALL OFFERS FAILED")
    return 1


if __name__ == "__main__":
    sys.exit(main())

# 대시보드 연동: 성공 시 host/port를 .dash_state.json에 기록
