#!/usr/bin/env python3
"""fly-lm S4 실시간 대시보드 v3 — 10초 폴링, 서버에서 SSH로 로그 tail."""
import json, os, re, subprocess, threading, time
from http.server import HTTPServer, BaseHTTPRequestHandler

HOME = os.path.expanduser("~")
STATE = json.load(open(os.path.join(HOME, "fly-lm", ".dash_state.json")))
CACHE = {"t": 0, "data": None, "hist": []}  # hist: (ts, fly_loss, msl_loss)
LOCK = threading.Lock()

STEP_RE = re.compile(r"\[(FLYS5|MASLOVS5|FLYS4|MASLOVS4|FLY20k|MASLOV20k|FLY|RANDOM|MASLOV)\] step\s+(\d+) loss ([0-9.]+)")

def ssh_tail(host, port, logfile):
    cmd = ["ssh", "-o", "StrictHostKeyChecking=no", "-o", "ConnectTimeout=8",
           "-o", "ControlMaster=no", "-o", "ControlPath=none",
           "-i", os.path.join(HOME, ".ssh/id_vastai"), "-p", str(port),
           f"root@{host}", f"tail -80 /root/work/{logfile}"]
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=20)
        return r.stdout
    except Exception:
        return ""

def poll_loop():
    while True:
        out = {}
        for name, ep in STATE.items():
            logname = "train_s5_FLY.log" if name == "fly" else "train_s5_MASLOV.log"
            txt = ssh_tail(ep["host"], ep["port"], logname)
            pts = [(int(m[1]), float(m[2])) for m in STEP_RE.findall(txt) if m[0].endswith(("S4","S5"))]
            last_sample = ""
            if "--- s4sample@" in txt:
                seg = txt.split("--- s4sample@")[-1].split("\n", 1)
                if len(seg) > 1:
                    last_sample = seg[1].split("\n")[0][:160]
            out[name] = {
                "step": pts[-1][0] if pts else 0,
                "loss": pts[-1][1] if pts else None,
                "recent": pts[-10:],
                "sample": last_sample,
            }
        with LOCK:
            fl = out.get("fly", {}).get("loss")
            ml = out.get("maslov", {}).get("loss")
            if fl and ml:
                CACHE["hist"].append((time.time(), fl, ml))
                CACHE["hist"] = CACHE["hist"][-2000:]
            CACHE["data"] = out
            CACHE["t"] = time.time()
        time.sleep(10)

class H(BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path == "/data":
            with LOCK:
                body = json.dumps({"t": CACHE["t"], "data": CACHE["data"],
                                   "hist": CACHE["hist"][-400:]}).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Access-Control-Allow-Origin", "*")
            self.end_headers()
            self.wfile.write(body)
        else:
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.end_headers()
            self.wfile.write(HTML.encode())
    def log_message(self, *a):
        pass

HTML = """<!doctype html><html><head><meta charset="utf-8">
<title>fly-lm S4 live</title>
<style>
 body{background:#0b0e14;color:#e6e9ef;font-family:ui-monospace,monospace;margin:0;padding:16px}
 .grid{display:grid;grid-template-columns:1fr 1fr;gap:12px;max-width:1100px;margin:0 auto}
 .card{background:#10141c;border:1px solid #232a3a;border-radius:10px;padding:14px}
 h1{font-size:15px;margin:0 0 4px} .sub{color:#8b93a7;font-size:11px}
 .big{font-size:34px;font-weight:700;margin:6px 0}
 .fly .big{color:#4cc38a} .msl .big{color:#7aa2f7}
 .bar{height:8px;background:#1a2030;border-radius:4px;overflow:hidden;margin:8px 0}
 .bar>div{height:100%;transition:width .5s}
 .fly .bar>div{background:#4cc38a} .msl .bar>div{background:#7aa2f7}
 .sample{font-size:11px;color:#a9b1c6;background:#0d1119;border-radius:6px;padding:8px;margin-top:8px;min-height:34px;word-break:break-all}
 .spark{width:100%;height:60px;margin-top:6px}
 #gap{max-width:1100px;margin:12px auto;font-size:13px}
 #gap b{font-size:20px}
</style></head><body>
<div class="grid">
 <div class="card fly"><h1>FLY S4 <span class="sub">(실제 파리 배선)</span></h1>
  <div class="big" id="fl">—</div><div class="sub" id="fs">step —</div>
  <div class="bar"><div id="fb" style="width:0%"></div></div>
  <svg class="spark" id="fspark"></svg>
  <div class="sample" id="fsample">…</div></div>
 <div class="card msl"><h1>MASLOV S4 <span class="sub">(차수보존 대조)</span></h1>
  <div class="big" id="ml">—</div><div class="sub" id="ms">step —</div>
  <div class="bar"><div id="mb" style="width:0%"></div></div>
  <svg class="spark" id="mspark"></svg>
  <div class="sample" id="msample">…</div></div>
</div>
<div class="card" id="gap"><span class="sub">실시간 격차 (FLY − MASLOV, 낮을수록 FLY 우위)</span><br>
<b id="gapv">—</b> <span class="sub" id="age"></span></div>
<script>
async function tick(){
 try{
  const r = await fetch('/data'); const j = await(r.json());
  const d = j.data, T = 20000;
  const f = d.fly, m = d.maslov;
  fl.textContent = f.loss ? f.loss.toFixed(3) : '—';
  ml.textContent = m.loss ? m.loss.toFixed(3) : '—';
  fs.textContent = 'step ' + f.step + ' / ' + T;
  ms.textContent = 'step ' + m.step + ' / ' + T;
  fb.style.width = (f.step/T*100)+'%'; mb.style.width = (m.step/T*100)+'%';
  fsample.textContent = f.sample || '(샘플 대기 — 2k스텝부터)';
  msample.textContent = m.sample || '(샘플 대기 — 2k스텝부터)';
  if (f.loss && m.loss) gapv.textContent = (f.loss - m.loss).toFixed(3);
  age.textContent = '| 동기화 ' + Math.round((Date.now()/1000 - j.t)) + 's 전';
  spark(fspark, j.hist.map(h=>h[1]), '#4cc38a');
  spark(mspark, j.hist.map(h=>h[2]), '#7aa2f7');
 }catch(e){}
 setTimeout(tick, 10000);
}
function spark(el, ys, color){
 if(!ys||ys.length<2) return;
 const w=el.clientWidth||300, hgt=60, pad=2;
 const lo=Math.min(...ys), hi=Math.max(...ys), rg=hi-lo||1;
 const pts = ys.map((y,i)=>`${pad+i*(w-2*pad)/(ys.length-1)},${hgt-pad-(y-lo)/rg*(hgt-2*pad)}`).join(' ');
 el.innerHTML = `<polyline points="${pts}" fill="none" stroke="${color}" stroke-width="2"/>`;
}
tick();
</script></body></html>"""

if __name__ == "__main__":
    threading.Thread(target=poll_loop, daemon=True).start()
    HTTPServer(("100.106.12.22", 8788), H).serve_forever()
