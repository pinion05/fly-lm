#!/usr/bin/env python3
"""fly-lm 학습 실시간 대시보드 — 테일넷 전용.

Vast.ai 원격 train2.log를 5초마다 rsync로 당겨와서 파싱, FLY/RANDOM loss를
실시간 차트로 보여준다. 노드 미니멀·HTML모바일우선 규칙 반영.
"""
import http.server
import json
import re
import subprocess
import threading
import time
from pathlib import Path

SSH = ["ssh", "-o", "StrictHostKeyChecking=no", "-o", "ControlMaster=no",
       "-o", "ControlPath=none", "-i", str(Path.home() / ".ssh/id_vastai"),
       "-p", "29212", "root@ssh9.vast.ai"]
LOCAL_LOG = Path("/tmp/fly_dash_train.log")
PORT = 8788

STEP_RE = re.compile(r"\[(FLY|RANDOM)\] step\s+(\d+) loss ([0-9.]+).*?\[(\d+)s\]")

state = {"updated": None, "points": [], "phase": "loading"}


def pull_log():
    """5초마다 원격 로그 스냅샷."""
    while True:
        try:
            r = subprocess.run(SSH + ["cat /root/work/train2.log"],
                               capture_output=True, text=True, timeout=30)
            if r.returncode == 0 and r.stdout:
                LOCAL_LOG.write_text(r.stdout)
                pts = []
                for m in STEP_RE.finditer(r.stdout):
                    pts.append({"tag": m.group(1), "step": int(m.group(2)),
                                "loss": float(m.group(3)), "t": int(m.group(4))})
                state["points"] = pts
                state["updated"] = time.strftime("%H:%M:%S")
                state["phase"] = "done" if "ALL-DONE" in r.stdout else "training"
        except Exception as e:
            state["phase"] = f"error: {type(e).__name__}"
        time.sleep(5)


PAGE = """<!DOCTYPE html>
<html lang="ko"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>fly-lm 대시보드</title>
<style>
  body{font-family:system-ui,sans-serif;background:#0b0e14;color:#e6e9ef;margin:0;padding:16px}
  h1{font-size:18px;margin:0 0 4px}
  .sub{color:#8b93a7;font-size:12px;margin-bottom:12px}
  .card{background:#151a23;border-radius:12px;padding:12px;margin-bottom:12px}
  .big{font-size:34px;font-weight:700}
  .fly{color:#4cc38a}.rnd{color:#f5a97f}.uni{color:#8b93a7}
  .row{display:flex;gap:12px;flex-wrap:wrap}
  .box{flex:1;min-width:140px}
  .lbl{font-size:11px;color:#8b93a7;text-transform:uppercase;letter-spacing:.5px}
  canvas{width:100%;height:240px;background:#10141c;border-radius:8px}
  .status{font-size:12px;color:#8b93a7}
  .mono{font-family:ui-monospace,monospace;font-size:12px;white-space:pre-wrap;word-break:break-all}
</style></head><body>
<h1>🪰 fly-lm Stage 2 — MaleCNS 211k 뉴런</h1>
<div class="sub" id="updated"></div>
<div class="row">
  <div class="card box"><div class="lbl">FLY (파리 배선)</div><div class="big fly" id="fly-last">—</div><div class="status" id="fly-step"></div></div>
  <div class="card box"><div class="lbl">RANDOM (무작위 배선)</div><div class="big rnd" id="rnd-last">—</div><div class="status" id="rnd-step"></div></div>
  <div class="card box"><div class="lbl">unigram 기준선</div><div class="big uni">3.080</div><div class="status">PPL 21.8</div></div>
</div>
<div class="card"><canvas id="cv"></canvas><div class="status">초록=FLY · 주황=RANDOM · 회색 점선=unigram 3.080 (아래로 갈수록 좋음)</div></div>
<div class="card"><div class="lbl">최근 로그</div><pre class="mono" id="tail"></pre></div>
<script>
async function tick(){
  try{
    const r = await fetch('/data'); const d = await r.json();
    document.getElementById('updated').textContent =
      '갱신 ' + d.updated + ' · 상태: ' + d.phase + ' · Vast.ai RTX 5070 Ti';
    const fly = d.points.filter(p=>p.tag==='FLY');
    const rnd = d.points.filter(p=>p.tag==='RANDOM');
    const fl = fly.at(-1), rl = rnd.at(-1);
    if(fl){document.getElementById('fly-last').textContent=fl.loss.toFixed(3);
           document.getElementById('fly-step').textContent='step '+fl.step+' / 8000 · 완주';}
    if(rl){document.getElementById('rnd-last').textContent=rl.loss.toFixed(3);
           document.getElementById('rnd-step').textContent='step '+rl.step+' / 8000';}
    draw(fly,rnd);
  }catch(e){}
  setTimeout(tick,5000);
}
function draw(fly,rnd){
  const cv=document.getElementById('cv'),ctx=cv.getContext('2d');
  const W=cv.width=cv.clientWidth*2,H=cv.height=480;
  ctx.scale(1,1);
  const all=[...fly,...rnd];
  const maxS=Math.max(8000,...all.map(p=>p.step)),minL=2.3,maxL=5;
  const X=s=>40+(s/maxS)*(W-60), Y=l=>20+((l-minL)/(maxL-minL))*(H-50);
  ctx.clearRect(0,0,W,H);
  ctx.strokeStyle='#3a4152';ctx.setLineDash([8,8]);
  ctx.beginPath();ctx.moveTo(40,Y(3.0802));ctx.lineTo(W-20,Y(3.0802));ctx.stroke();
  ctx.setLineDash([]);
  ctx.fillStyle='#8b93a7';ctx.font='20px system-ui';ctx.fillText('unigram 3.080',48,Y(3.0802)-8);
  function line(pts,color){
    if(!pts.length)return;
    ctx.strokeStyle=color;ctx.lineWidth=3;ctx.beginPath();
    pts.forEach((p,i)=>{const x=X(p.step),y=Math.max(20,Math.min(H-30,Y(p.loss)));
      i?ctx.lineTo(x,y):ctx.moveTo(x,y);});
    ctx.stroke();
    const lp=pts.at(-1);
    ctx.fillStyle=color;ctx.beginPath();
    ctx.arc(X(lp.step),Math.max(20,Math.min(H-30,Y(lp.loss))),7,0,7);ctx.fill();
  }
  line(fly,'#4cc38a');line(rnd,'#f5a97f');
  ctx.fillStyle='#8b93a7';ctx.font='18px system-ui';
  for(let s=0;s<=8000;s+=2000){ctx.fillText(s,X(s)-15,H-8);}
}
tick();
</script></body></html>"""


class H(http.server.BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path == "/data":
            body = json.dumps(state).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
        else:
            body = PAGE.encode()
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *a):
        pass


if __name__ == "__main__":
    threading.Thread(target=pull_log, daemon=True).start()
    ts_ip = subprocess.run(["tailscale", "ip", "-4"], capture_output=True, text=True).stdout.strip().split("\n")[0]
    srv = http.server.ThreadingHTTPServer((ts_ip, PORT), H)
    print(f"http://{ts_ip}:{PORT}")
    srv.serve_forever()
