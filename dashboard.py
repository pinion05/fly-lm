#!/usr/bin/env python3
"""fly-lm 통합 대시보드 v2 — 테일넷 전용 (8788).

- Stage 2 (8k, 3-way) 히스토리 + FLY 20k 진행 상황을 한 화면에
- 20k 인스턴스가 떠 있으면 SSH로 train_20k.log를 실시간 pull, 없으면 로컬 마지막 스냅샷
- 학습률(loss) 카드 + 곡선 + 최근 샘플 텍스트
"""
import http.server
import json
import re
import subprocess
import threading
import time
from pathlib import Path

HOME = Path.home()
PORT = 8788
STATE_FILE = HOME / "fly-lm/.dash_state.json"   # 인스턴스 host/port 기록용
LOG_CACHE = HOME / "fly-lm/.dash_20k.log"       # 마지막 pull 캐시

STEP_RE = re.compile(r"\[(FLY|RANDOM|MASLOV|FLY20k|MASLOV20k)\] step\s+(\d+) loss ([0-9.]+).*?\[(\d+)s\]")
SAMPLE_RE = re.compile(r"--- sample@(\d+) ---\n(.{0,300})", re.S)

state = {"updated": None, "points": [], "hist": [], "samples": {}, "phase": "loading"}


def pull_loop():
    while True:
        try:
            # 1) Stage2 히스토리 (정적)
            hist_log = (HOME / "fly-lm/stage2_train.log").read_text() if (HOME / "fly-lm/stage2_train.log").exists() else ""
            mlog = (HOME / "fly-lm/train_maslov.log").read_text() if (HOME / "fly-lm/train_maslov.log").exists() else ""
            hist = []
            for m in STEP_RE.finditer(hist_log):
                hist.append({"tag": m.group(1), "step": int(m.group(2)), "loss": float(m.group(3))})
            for m in STEP_RE.finditer(mlog):
                hist.append({"tag": "MASLOV", "step": int(m.group(2)), "loss": float(m.group(3))})

            # 2) 20k 진행 (원격 인스턴스들 pull: fly + maslov)
            eps = {}
            if STATE_FILE.exists():
                eps = json.loads(STATE_FILE.read_text())
            remote_log = ""
            cache2 = HOME / "fly-lm/.dash_maslov.log"
            for name, logname in [("fly", "train_20k.log"), ("maslov", "train_maslov20k.log")]:
                ep = eps.get(name) if isinstance(eps, dict) else None
                if ep and ep.get("host"):
                    r = subprocess.run(
                        ["ssh", "-o", "StrictHostKeyChecking=no", "-o", "ControlMaster=no",
                         "-o", "ControlPath=none", "-i", str(HOME / ".ssh/id_vastai"),
                         "-p", str(ep["port"]), f"root@{ep['host']}",
                         f"cat /root/work/{logname} 2>/dev/null || tail -100 /root/launch.log"],
                        capture_output=True, text=True, timeout=30)
                    if r.returncode == 0 and r.stdout:
                        remote_log += r.stdout
                        (LOG_CACHE if name == "fly" else cache2).write_text(r.stdout)
            if not remote_log and LOG_CACHE.exists():
                remote_log = LOG_CACHE.read_text()
                if cache2.exists():
                    remote_log += cache2.read_text()

            pts20 = [{"tag": m.group(1), "step": int(m.group(2)), "loss": float(m.group(3))}
                     for m in STEP_RE.finditer(remote_log)]
            samples = {m.group(1): m.group(2) for m in SAMPLE_RE.finditer(remote_log)}

            state["hist"] = hist
            state["points"] = pts20
            state["samples"] = samples
            state["updated"] = time.strftime("%H:%M:%S")
            state["phase"] = ("done" if "ALL-DONE" in remote_log
                              else "20k-running" if pts20
                              else "waiting-instance" if not ep else "launching")
        except Exception as e:
            state["phase"] = f"error: {type(e).__name__}"
        time.sleep(5)


PAGE = """<!DOCTYPE html>
<html lang="ko"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>fly-lm 20k 대시보드</title>
<style>
 body{font-family:system-ui,sans-serif;background:#0b0e14;color:#e6e9ef;margin:0;padding:16px}
 h1{font-size:18px;margin:0 0 4px}
 .sub{color:#8b93a7;font-size:12px;margin-bottom:12px}
 .card{background:#151a23;border-radius:12px;padding:12px;margin-bottom:12px}
 .row{display:flex;gap:12px;flex-wrap:wrap}
 .box{flex:1;min-width:130px}
 .big{font-size:32px;font-weight:700}
 .lbl{font-size:11px;color:#8b93a7;text-transform:uppercase;letter-spacing:.5px}
 canvas{width:100%;height:280px;background:#10141c;border-radius:8px}
 .mono{font-family:ui-monospace,monospace;font-size:12px;white-space:pre-wrap;word-break:break-all;color:#c8d0e0}
 .samp{border-left:3px solid #4cc38a;padding-left:8px;margin:6px 0}
 .status{font-size:12px;color:#8b93a7}
</style></head><body>
<h1>🪰 fly-lm — FLY 20k 수렴 추적 <span id="phase" style="font-size:12px;color:#8b93a7"></span></h1>
<div class="sub" id="updated"></div>
<div class="row">
 <div class="card box"><div class="lbl">현재 loss (20k)</div><div class="big" style="color:#4cc38a" id="cur">—</div><div class="status" id="cur-step"></div></div>
 <div class="card box"><div class="lbl">8k 기준</div><div class="big" style="color:#7aa2f7">2.629</div><div class="status">eval @ seed123</div></div>
 <div class="card box"><div class="lbl">unigram</div><div class="big" style="color:#8b93a7">3.080</div><div class="status">기준선</div></div>
</div>
<div class="card"><canvas id="cv"></canvas><div class="status">초록=FLY20k 진행 · 파랑=FLY 8k 히스토리 · 주황=RANDOM · 회점선=unigram</div></div>
<div class="card"><div class="lbl">생성 샘플 (2k스텝마다)</div><div id="samples"></div></div>
<script>
async function tick(){
 try{
  const d = await (await fetch('/data')).json();
  document.getElementById('phase').textContent = '· ' + d.phase;
  document.getElementById('updated').textContent = '갱신 ' + d.updated;
  const p20 = d.points.filter(p=>p.tag==='FLY20k');
  const last = p20.at(-1);
  if(last){document.getElementById('cur').textContent = last.loss.toFixed(3);
           document.getElementById('cur-step').textContent = 'step ' + last.step + ' / 20000';}
  draw(d);
  const el = document.getElementById('samples');
  el.innerHTML = Object.entries(d.samples).slice(-4).map(([s,t])=>
    `<div class="samp"><b style="color:#4cc38a">@${s}</b> <span class="mono">${t.slice(0,220)}</span></div>`).join('') || '<span class="status">아직 샘플 없음</span>';
 }catch(e){}
 setTimeout(tick,5000);
}
function draw(d){
 const cv=document.getElementById('cv'),ctx=cv.getContext('2d');
 const W=cv.width=cv.clientWidth*2,H=cv.height=560;
 const series=[['FLY','#7aa2f7',d.hist.filter(p=>p.tag==='FLY')],
               ['RANDOM','#f5a97f',d.hist.filter(p=>p.tag==='RANDOM')],
               ['MASLOV','#c0caf5',d.hist.filter(p=>p.tag==='MASLOV')],
               ['FLY20k','#4cc38a',d.points.filter(p=>p.tag==='FLY20k')]];
 const all=series.flatMap(s=>s[2]);
 const maxS=Math.max(20000,...all.map(p=>p.step)),minL=2.3,maxL=5;
 const X=s=>50+(s/maxS)*(W-70),Y=l=>20+((l-minL)/(maxL-minL))*(H-60);
 ctx.clearRect(0,0,W,H);
 ctx.strokeStyle='#3a4152';ctx.setLineDash([8,8]);
 ctx.beginPath();ctx.moveTo(50,Y(3.0802));ctx.lineTo(W-20,Y(3.0802));ctx.stroke();ctx.setLineDash([]);
 ctx.fillStyle='#8b93a7';ctx.font='20px system-ui';ctx.fillText('unigram 3.080',58,Y(3.0802)-8);
 for(const [name,color,pts] of series){
  if(!pts.length)continue;
  ctx.strokeStyle=color;ctx.lineWidth=3;ctx.beginPath();
  pts.forEach((p,i)=>{const x=X(p.step),y=Math.max(20,Math.min(H-40,Y(p.loss)));i?ctx.lineTo(x,y):ctx.moveTo(x,y);});
  ctx.stroke();
 }
 ctx.fillStyle='#8b93a7';ctx.font='18px system-ui';
 for(let s=0;s<=20000;s+=5000)ctx.fillText(s,X(s)-20,H-12);
 ctx.fillText('loss (nats/char)',10,30);
}
tick();
</script></body></html>"""


class H(http.server.BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path == "/data":
            body = json.dumps(state).encode()
            ctype = "application/json"
        else:
            body = PAGE.encode()
            ctype = "text/html; charset=utf-8"
        self.send_response(200)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *a):
        pass


if __name__ == "__main__":
    threading.Thread(target=pull_loop, daemon=True).start()
    ts_ip = subprocess.run(["tailscale", "ip", "-4"], capture_output=True, text=True
                           ).stdout.strip().split("\n")[0]
    srv = http.server.ThreadingHTTPServer((ts_ip, PORT), H)
    print(f"http://{ts_ip}:{PORT}")
    srv.serve_forever()
