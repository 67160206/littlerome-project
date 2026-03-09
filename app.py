"""
Littlelome AI Vision — Streamlit App
YOLOv11 · Flame Detection · NO cv2 dependency

requirements.txt:
  streamlit
  ultralytics
  pillow
  pandas
  numpy
"""

import streamlit as st
import time, os, json, tempfile
import numpy as np
import pandas as pd
from datetime import datetime
from PIL import Image, ImageDraw, ImageFont

# ── PAGE CONFIG ──────────────────────────────
st.set_page_config(
    page_title="Littlelome AI Vision",
    page_icon="👁",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;600;700;800&family=JetBrains+Mono:wght@400;500&display=swap');
html,body,[class*="css"],.stMarkdown{font-family:'Syne',sans-serif!important}
.stApp{background:#0d1117!important;color:#e6edf3!important}
[data-testid="stSidebar"]{background:#161b22!important;border-right:1px solid #30363d!important}
[data-testid="stSidebar"] *{color:#e6edf3!important}
[data-testid="stSidebar"] .stButton>button{
  background:transparent!important;border:1px solid #30363d!important;color:#8b949e!important;
  text-align:left!important;width:100%!important;border-radius:8px!important;
  padding:8px 14px!important;font-size:13px!important;font-weight:600!important;
  margin-bottom:2px!important}
[data-testid="stSidebar"] .stButton>button:hover{background:#1c2230!important;color:#e6edf3!important}
div[data-testid="metric-container"]{background:#161b22!important;border:1px solid #30363d!important;border-radius:10px!important;padding:16px!important}
div[data-testid="metric-container"]>label{color:#8b949e!important;font-size:11px!important;text-transform:uppercase!important;letter-spacing:.5px!important;font-weight:600!important}
div[data-testid="metric-container"] div[data-testid="stMetricValue"]>div{font-family:'JetBrains Mono',monospace!important;font-size:26px!important;font-weight:700!important}
.stButton>button{background:#21262d!important;border:1px solid #30363d!important;color:#e6edf3!important;border-radius:8px!important;font-family:'Syne',sans-serif!important;font-weight:700!important}
.stButton>button:hover{background:#2d333b!important}
.stTabs [data-baseweb="tab-list"]{background:#161b22!important;border-bottom:1px solid #30363d!important;gap:0!important}
.stTabs [data-baseweb="tab"]{background:transparent!important;color:#8b949e!important;border:none!important;border-bottom:2px solid transparent!important;font-family:'Syne',sans-serif!important;font-weight:600!important;font-size:13px!important;padding:10px 20px!important}
.stTabs [aria-selected="true"]{color:#e6edf3!important;border-bottom:2px solid #2188ff!important;background:transparent!important}
[data-testid="stExpander"]{background:#161b22!important;border:1px solid #30363d!important;border-radius:10px!important;margin-bottom:8px!important}
#MainMenu,footer,header{visibility:hidden}
.fault-box{background:#2d1515;border:2px solid #f85149;border-radius:10px;padding:16px 20px;margin:10px 0}
.ok-box{background:#0e2a1a;border:2px solid #3fb950;border-radius:10px;padding:16px 20px;margin:10px 0}
.detect-card{background:#1c2230;border:1px solid #30363d;border-radius:8px;padding:12px 16px;margin:6px 0}
.section-card{background:#161b22;border:1px solid #30363d;border-radius:10px;padding:20px;margin-bottom:16px}
.status-row{display:flex;align-items:center;justify-content:space-between;padding:8px 0;border-bottom:1px solid #21262d;font-size:13px}
.status-row:last-child{border:none}
.mono{font-family:'JetBrains Mono',monospace}
.dot-green{display:inline-block;width:8px;height:8px;border-radius:50%;background:#3fb950;box-shadow:0 0 6px #3fb950;margin-right:6px}
.dot-orange{display:inline-block;width:8px;height:8px;border-radius:50%;background:#e3b341;box-shadow:0 0 6px #e3b341;margin-right:6px}
.dot-gray{display:inline-block;width:8px;height:8px;border-radius:50%;background:#484f58;margin-right:6px}
</style>
""", unsafe_allow_html=True)

# ── SESSION STATE ────────────────────────────
for k, v in {
    "history": [], "model": None, "page": "dashboard",
    "threshold": 0.40, "show_boxes": True, "autosave": True,
}.items():
    if k not in st.session_state:
        st.session_state[k] = v

# ── DEFECT CONFIG ────────────────────────────
DEFECT_MAP = {
    "flame":     {"label": "🔥 Flame (ไฟ)",         "hex": "#f85149", "rgb": (248, 81,  73)},
    "fire":      {"label": "🔥 Fire (ไฟ)",           "hex": "#f85149", "rgb": (248, 81,  73)},
    "smoke":     {"label": "💨 Smoke (ควัน)",         "hex": "#8b949e", "rgb": (139,148, 158)},
    "rust":      {"label": "🟠 Rust (สนิม)",          "hex": "#e3b341", "rgb": (227,179,  65)},
    "dent":      {"label": "🔵 Dent (รอยบุบ)",         "hex": "#2188ff", "rgb": ( 33,136, 255)},
    "corrosion": {"label": "🟠 Corrosion (สนิม)",     "hex": "#e3b341", "rgb": (227,179,  65)},
    "scratch":   {"label": "🔵 Scratch (รอยขีด)",     "hex": "#2188ff", "rgb": ( 33,136, 255)},
}
def get_info(cls):
    return DEFECT_MAP.get(cls.lower(), {
        "label": f"⚠️ {cls.capitalize()}",
        "hex": "#f85149", "rgb": (248, 81, 73)
    })
def get_severity(c):
    return "🔴 High" if c >= .80 else ("🟡 Medium" if c >= .55 else "🟢 Low")

# ── DRAW BOXES (PIL only, no cv2) ────────────
def draw_boxes(img: Image.Image, dets: list) -> Image.Image:
    out = img.copy().convert("RGBA")
    draw = ImageDraw.Draw(out)
    W, H = out.size
    for d in dets:
        x1, y1, x2, y2 = d["x1"], d["y1"], d["x2"], d["y2"]
        rgb = d["rgb"]
        # Box
        for t in range(3):
            draw.rectangle([x1-t, y1-t, x2+t, y2+t], outline=rgb)
        # Label background
        label = f"{d['cls'].capitalize()} {d['conf']:.0%}"
        lw = len(label) * 8 + 10
        lh = 22
        draw.rectangle([x1, y1 - lh, x1 + lw, y1], fill=rgb + (200,))
        draw.text((x1 + 5, y1 - lh + 3), label, fill=(255, 255, 255))
    return out.convert("RGB")

# ── INFERENCE ────────────────────────────────
def run_inference(img: Image.Image, thr: float):
    t0 = time.time()
    results = st.session_state.model(img, conf=thr, verbose=False)[0]
    elapsed = time.time() - t0
    dets = []
    for box in results.boxes:
        cls_id   = int(box.cls[0])
        cls_name = st.session_state.model.names[cls_id]
        conf     = float(box.conf[0])
        x1, y1, x2, y2 = [float(v) for v in box.xyxy[0]]
        info = get_info(cls_name)
        dets.append({
            "cls": cls_name, "label": info["label"], "conf": conf,
            "severity": get_severity(conf),
            "hex": info["hex"], "rgb": info["rgb"],
            "x1": x1, "y1": y1, "x2": x2, "y2": y2,
        })
    dets.sort(key=lambda d: d["conf"], reverse=True)
    return dets, elapsed

def add_history(src, status, dets, thumb=None):
    if not st.session_state.autosave:
        return
    st.session_state.history.insert(0, {
        "id": int(time.time() * 1000),
        "time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "source": src, "status": status,
        "detections": dets, "thumb": thumb,
    })
    if len(st.session_state.history) > 200:
        st.session_state.history.pop()

def fault_count():
    return sum(1 for h in st.session_state.history if h["status"] == "FAULT")
def avg_conf():
    cs = [d["conf"] for h in st.session_state.history for d in h["detections"]]
    return f"{sum(cs)/len(cs):.0%}" if cs else "—"

# ── LOAD MODEL ───────────────────────────────
MODEL_FILE = "Flame_Best_Model.pt"

@st.cache_resource
def load_model(path):
    from ultralytics import YOLO
    return YOLO(path)

if st.session_state.model is None and os.path.exists(MODEL_FILE):
    with st.spinner(f"⏳ Loading {MODEL_FILE}…"):
        st.session_state.model = load_model(MODEL_FILE)

# ── SIDEBAR ──────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style="display:flex;align-items:center;gap:10px;padding:4px 0 16px">
      <div style="background:linear-gradient(135deg,#f85149,#7c3aed);border-radius:8px;
        width:32px;height:32px;display:flex;align-items:center;justify-content:center;font-size:16px">👁</div>
      <div>
        <div style="font-size:14px;font-weight:800">Littlelome<span style="color:#f85149">AI</span></div>
        <div style="font-size:10px;color:#8b949e">Vision · YOLOv11</div>
      </div>
    </div>""", unsafe_allow_html=True)

    if st.session_state.model:
        classes = list(st.session_state.model.names.values())
        st.markdown(f'<div style="font-size:11px;color:#3fb950;margin-bottom:12px"><span class="dot-green"></span>Loaded · {", ".join(classes)}</div>', unsafe_allow_html=True)
    else:
        st.markdown('<div style="font-size:12px;color:#f85149;margin-bottom:8px">⚠️ Model not found</div>', unsafe_allow_html=True)
        up = st.file_uploader("Upload .pt file", type=["pt"])
        if up:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pt") as tmp:
                tmp.write(up.read())
            st.session_state.model = load_model(tmp.name)
            st.rerun()

    st.divider()
    st.markdown('<div style="font-size:10px;font-weight:700;color:#484f58;letter-spacing:1px;margin-bottom:6px">MAIN</div>', unsafe_allow_html=True)

    for pid, plabel in [
        ("dashboard",  "📊  Dashboard"),
        ("upload_img", "🖼️  Upload Image"),
        ("upload_vid", "🎬  Upload Video"),
        ("history",    "📋  History"),
        ("settings",   "⚙️  Settings"),
    ]:
        if st.session_state.page == pid:
            st.markdown(f'<div style="background:#2188ff20;border:1px solid #2188ff40;border-radius:8px;padding:8px 14px;margin-bottom:2px;font-size:13px;font-weight:700;color:#2188ff">{plabel}</div>', unsafe_allow_html=True)
        else:
            if st.button(plabel, key=f"nav_{pid}", use_container_width=True):
                st.session_state.page = pid
                st.rerun()

    st.divider()
    total  = len(st.session_state.history)
    faults = fault_count()
    st.markdown(f"""
    <div style="display:flex;gap:8px">
      <div style="flex:1;background:#21262d;border-radius:8px;padding:10px;text-align:center">
        <div style="font-family:'JetBrains Mono';font-size:18px;font-weight:700">{total}</div>
        <div style="font-size:10px;color:#8b949e">Scanned</div>
      </div>
      <div style="flex:1;background:#21262d;border-radius:8px;padding:10px;text-align:center">
        <div style="font-family:'JetBrains Mono';font-size:18px;font-weight:700;color:#f85149">{faults}</div>
        <div style="font-size:10px;color:#8b949e">Faults</div>
      </div>
    </div>""", unsafe_allow_html=True)

# ── HEADER ───────────────────────────────────
h1, h2, h3 = st.columns([4, 1, 1])
with h1:
    titles = {
        "dashboard":  "📊 Dashboard",
        "upload_img": "🖼️ Upload Image",
        "upload_vid": "🎬 Upload Video",
        "history":    "📋 History",
        "settings":   "⚙️ Settings",
    }
    st.markdown(f'<div style="font-size:22px;font-weight:800;padding:4px 0 16px">{titles[st.session_state.page]}</div>', unsafe_allow_html=True)
with h2:
    st.markdown('<div style="display:flex;align-items:center;gap:6px;padding-top:8px"><span style="width:8px;height:8px;border-radius:50%;background:#3fb950;box-shadow:0 0 6px #3fb950;display:inline-block"></span><span style="font-size:11px;color:#8b949e;font-family:\'JetBrains Mono\'">ONLINE</span></div>', unsafe_allow_html=True)
with h3:
    st.markdown(f'<div style="font-size:12px;color:#8b949e;font-family:\'JetBrains Mono\';text-align:right;padding-top:8px">{datetime.now().strftime("%H:%M:%S")}</div>', unsafe_allow_html=True)

# ════════════════════════════════════════════
# DASHBOARD
# ════════════════════════════════════════════
if st.session_state.page == "dashboard":
    total = len(st.session_state.history)
    faults = fault_count()
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("🔍 Total Inspected", total)
    k2.metric("⚠️ Defects Found",   faults)
    k3.metric("✅ Passed",           total - faults)
    k4.metric("📊 Avg Confidence",  avg_conf())
    st.markdown("<br>", unsafe_allow_html=True)

    ca, cb = st.columns(2)
    with ca:
        st.markdown('<div class="section-card">', unsafe_allow_html=True)
        st.markdown("**📈 Defect Trend**")
        if len(st.session_state.history) >= 2:
            df = pd.DataFrame([{
                "Time":  h["time"],
                "Fault": 1 if h["status"] == "FAULT" else 0
            } for h in reversed(st.session_state.history)])
            df["Time"] = pd.to_datetime(df["Time"])
            st.line_chart(df.set_index("Time"), color="#f85149", height=150)
        else:
            st.markdown('<div style="text-align:center;color:#484f58;padding:40px 0;font-size:13px">No data yet</div>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

    with cb:
        st.markdown('<div class="section-card">', unsafe_allow_html=True)
        st.markdown("**🖥️ System Status**")
        model_ok = st.session_state.model is not None
        cls_txt  = ', '.join(st.session_state.model.names.values()) if model_ok else '—'
        st.markdown(f"""
        <div class="status-row"><span>AI Engine</span><span><span class="dot-green"></span><span style="color:#3fb950;font-size:12px;font-family:'JetBrains Mono'">Online</span></span></div>
        <div class="status-row"><span>YOLOv11 Model</span><span><span class="{'dot-green' if model_ok else 'dot-gray'}"></span><span style="color:{'#3fb950' if model_ok else '#484f58'};font-size:12px;font-family:'JetBrains Mono'">{'Loaded' if model_ok else 'Not loaded'}</span></span></div>
        <div class="status-row"><span>Classes</span><span style="font-size:11px;font-family:'JetBrains Mono';color:#8b949e">{cls_txt}</span></div>
        <div class="status-row"><span>Alert System</span><span><span class="dot-orange"></span><span style="color:#e3b341;font-size:12px;font-family:'JetBrains Mono'">Armed</span></span></div>
        <div class="status-row"><span>Auto-Save</span><span style="font-size:12px;font-family:'JetBrains Mono';color:#8b949e">{'On' if st.session_state.autosave else 'Off'}</span></div>
        """, unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    st.markdown("**📋 Recent Inspections**")
    if st.session_state.history:
        rows = [{"Status": "⚠️ FAULT" if h["status"] == "FAULT" else "✅ OK",
                 "Defect": ", ".join(d["label"] for d in h["detections"]) or "—",
                 "Conf":   f"{h['detections'][0]['conf']:.0%}" if h["detections"] else "—",
                 "Source": h["source"], "Time": h["time"]}
                for h in st.session_state.history[:5]]
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
    else:
        st.markdown('<div style="text-align:center;color:#484f58;padding:32px 0;font-size:13px">No inspections yet.</div>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

# ════════════════════════════════════════════
# UPLOAD IMAGE
# ════════════════════════════════════════════
elif st.session_state.page == "upload_img":
    if not st.session_state.model:
        st.warning("⚠️ Load model first (sidebar)"); st.stop()

    col1, col2 = st.columns(2)
    with col1:
        st.markdown('<div class="section-card">', unsafe_allow_html=True)
        st.markdown("**📤 Upload Image**")
        uimg = st.file_uploader("Drop image (JPG PNG WEBP)", type=["jpg","jpeg","png","webp","bmp"], key="img_up")
        thr  = st.slider("Confidence Threshold", 0.10, 0.95, st.session_state.threshold, 0.05, format="%.0f%%", key="img_thr")
        abtn = st.button("🔍 Detect Now", key="btn_img", use_container_width=True, disabled=(uimg is None))
        st.markdown('</div>', unsafe_allow_html=True)

    with col2:
        st.markdown('<div class="section-card">', unsafe_allow_html=True)
        st.markdown("**👁 Result**")
        if uimg:
            img   = Image.open(uimg).convert("RGB")
            rslot = st.empty()
            rslot.image(img, use_container_width=True)
            if abtn:
                with st.spinner("Running YOLOv11…"):
                    dets, elapsed = run_inference(img, thr)
                if dets:
                    if st.session_state.show_boxes:
                        rslot.image(draw_boxes(img, dets), use_container_width=True)
                    add_history("Image", "FAULT", dets, img.copy())
                    st.markdown('</div>', unsafe_allow_html=True)
                    st.markdown(f'<div class="fault-box"><div style="font-size:20px;font-weight:800;color:#f85149">🔥 DETECTED — {len(dets)} defect(s)</div><div style="font-size:12px;color:#8b949e;margin-top:4px">{elapsed*1000:.0f}ms inference</div></div>', unsafe_allow_html=True)
                    for d in dets:
                        st.markdown(f'<div class="detect-card"><div style="display:flex;justify-content:space-between;align-items:center"><span style="color:{d["hex"]};font-weight:800;font-size:15px">{d["label"]}</span><span class="mono" style="font-size:16px;font-weight:700">{d["conf"]:.1%}</span></div><div style="color:#8b949e;font-size:12px;margin-top:4px">{d["severity"]}</div><div style="height:4px;background:#21262d;border-radius:2px;margin-top:8px"><div style="width:{d["conf"]*100:.0f}%;height:100%;background:{d["hex"]};border-radius:2px"></div></div></div>', unsafe_allow_html=True)
                else:
                    add_history("Image", "OK", [], img.copy())
                    st.markdown('</div>', unsafe_allow_html=True)
                    st.markdown('<div class="ok-box"><div style="font-size:20px;font-weight:800;color:#3fb950">✅ NO DEFECT DETECTED</div><div style="font-size:12px;color:#8b949e;margin-top:4px">Image is clear</div></div>', unsafe_allow_html=True)
            else:
                st.markdown('</div>', unsafe_allow_html=True)
        else:
            st.markdown('<div style="text-align:center;color:#484f58;padding:60px 0;font-size:13px">Upload an image to begin</div>', unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)

# ════════════════════════════════════════════
# UPLOAD VIDEO
# ════════════════════════════════════════════
elif st.session_state.page == "upload_vid":
    if not st.session_state.model:
        st.warning("⚠️ Load model first (sidebar)"); st.stop()

    col1, col2 = st.columns(2)
    with col1:
        st.markdown('<div class="section-card">', unsafe_allow_html=True)
        st.markdown("**📤 Upload Video**")
        uvid = st.file_uploader("Drop video (MP4 MOV AVI)", type=["mp4","mov","avi","mkv","webm"], key="vid_up")
        va, vb = st.columns(2)
        with va: fint = st.slider("Every (s)", 1, 10, 2, key="vid_int")
        with vb: mxf  = st.slider("Max frames", 5, 30, 10, key="vid_max")
        thr_v = st.slider("Threshold", 0.10, 0.95, st.session_state.threshold, 0.05, format="%.0f%%", key="vid_thr")
        pbtn  = st.button("▶ Process Video", key="btn_vid", use_container_width=True, disabled=(uvid is None))
        st.markdown('</div>', unsafe_allow_html=True)

    with col2:
        st.markdown('<div class="section-card">', unsafe_allow_html=True)
        st.markdown("**🎞 Preview**")
        if uvid:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp:
                tmp.write(uvid.read()); vpath = tmp.name
            st.video(vpath)
        else:
            st.markdown('<div style="text-align:center;color:#484f58;padding:60px 0;font-size:13px">Upload a video to begin</div>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

    if uvid and pbtn:
        # Extract frames using PIL via imageio (no cv2)
        try:
            import imageio.v3 as iio
            reader = iio.imiter(vpath, plugin="pyav")
            fps_v  = 25  # default
            try:
                meta  = iio.immeta(vpath, plugin="pyav")
                fps_v = float(meta.get("fps", 25))
            except Exception:
                pass
        except ImportError:
            st.error("❌ imageio[pyav] not installed. Add `imageio[pyav]` to requirements.txt")
            st.stop()

        step   = max(1, int(fps_v * fint))
        prog   = st.progress(0, "Reading frames…")
        buf    = []
        fi     = 0
        for frame_arr in reader:
            if len(buf) >= mxf:
                break
            if fi % step == 0:
                buf.append((fi, fi / fps_v, Image.fromarray(frame_arr)))
            fi += 1

        fres = []
        st.markdown("**🔍 Frame-by-Frame Results**")
        tcols = st.columns(min(4, max(len(buf), 1)))
        for i, (fidx, tsec, pil) in enumerate(buf):
            prog.progress((i + 1) / len(buf), f"Frame {i+1}/{len(buf)}…")
            dets, _ = run_inference(pil, thr_v)
            disp    = draw_boxes(pil.copy(), dets) if (st.session_state.show_boxes and dets) else pil
            with tcols[i % 4]:
                st.image(disp, caption=f"{'🔥' if dets else '✅'} @{tsec:.1f}s", use_container_width=True)
            fres.append({"time": tsec, "status": "FAULT" if dets else "OK", "dets": dets})

        prog.empty(); st.divider()
        ff   = [r for r in fres if r["status"] == "FAULT"]
        adets= [d for r in fres for d in r["dets"]]
        m1,m2,m3,m4 = st.columns(4)
        m1.metric("Frames",      len(fres))
        m2.metric("🔥 Fault",    len(ff))
        m3.metric("✅ Clear",    len(fres) - len(ff))
        m4.metric("Clear Rate",  f"{(len(fres)-len(ff))/max(len(fres),1):.0%}")
        if ff:
            st.error(f"🔥 Flame at: {', '.join(f'{r[chr(116)]:.1f}s' for r in ff)}")
        else:
            st.success("✅ No flame detected")
        add_history("Video", "FAULT" if ff else "OK", adets[:5])

# ════════════════════════════════════════════
# HISTORY
# ════════════════════════════════════════════
elif st.session_state.page == "history":
    t1, t2, t3 = st.columns([4,1,1])
    with t2:
        if st.button("📥 Export JSON", use_container_width=True):
            d = json.dumps([{k:v for k,v in h.items() if k!="thumb"} for h in st.session_state.history], indent=2)
            st.download_button("Download", d, "history.json", "application/json")
    with t3:
        if st.button("🗑 Clear All", use_container_width=True):
            st.session_state.history = []; st.rerun()

    if not st.session_state.history:
        st.markdown('<div style="text-align:center;padding:80px 20px;color:#484f58"><div style="font-size:48px;margin-bottom:16px">📋</div><div style="font-size:16px;font-weight:700;margin-bottom:8px">No history yet</div></div>', unsafe_allow_html=True)
    else:
        total = len(st.session_state.history); faults = fault_count()
        k1,k2,k3,k4 = st.columns(4)
        k1.metric("Total", total); k2.metric("Faults", faults)
        k3.metric("Passed", total-faults); k4.metric("Pass Rate", f"{(total-faults)/max(total,1):.0%}")
        st.markdown("<br>", unsafe_allow_html=True)
        for i, h in enumerate(st.session_state.history):
            isf  = h["status"] == "FAULT"
            icon = "🔥 FAULT" if isf else "✅ OK"
            dstr = ", ".join(d["label"] for d in h["detections"]) or "No detections"
            with st.expander(f"{icon}  ·  {h['source']}  ·  {dstr}  ·  {h['time']}", expanded=(i==0 and isf)):
                ec1, ec2 = st.columns([2,1])
                with ec1:
                    if h.get("thumb"):
                        try:
                            disp = draw_boxes(h["thumb"].copy(), h["detections"]) if (st.session_state.show_boxes and h["detections"]) else h["thumb"]
                            st.image(disp, use_container_width=True)
                        except Exception: pass
                with ec2:
                    color = "#f85149" if isf else "#3fb950"
                    st.markdown(f'<div style="font-size:13px;line-height:2.2"><b>Status:</b> <span style="color:{color};font-weight:800">{icon}</span><br><b>Source:</b> <span class="mono">{h["source"]}</span><br><b>Time:</b> <span class="mono" style="font-size:11px">{h["time"]}</span></div>', unsafe_allow_html=True)
                    for d in h["detections"]:
                        st.markdown(f'<div class="detect-card" style="padding:8px 12px"><span style="color:{d["hex"]};font-weight:700">{d["label"]}</span> <span class="mono" style="float:right">{d["conf"]:.0%}</span><br><span style="color:#8b949e;font-size:11px">{d["severity"]}</span></div>', unsafe_allow_html=True)

# ════════════════════════════════════════════
# SETTINGS
# ════════════════════════════════════════════
elif st.session_state.page == "settings":
    col1, col2 = st.columns(2)
    with col1:
        st.markdown('<div class="section-card">', unsafe_allow_html=True)
        st.markdown("**🤖 AI Configuration**")
        nthr = st.slider("Confidence Threshold", 0.10, 0.95, st.session_state.threshold, 0.05, format="%.0f%%", key="set_thr")
        if nthr != st.session_state.threshold:
            st.session_state.threshold = nthr
        if st.session_state.model:
            st.markdown("**Active Classes:**")
            for cls in st.session_state.model.names.values():
                info = get_info(cls)
                st.markdown(f'<div class="detect-card" style="padding:8px 12px"><span style="color:{info["hex"]};font-weight:700">{info["label"]}</span></div>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

    with col2:
        st.markdown('<div class="section-card">', unsafe_allow_html=True)
        st.markdown("**🔔 Preferences**")
        st.session_state.show_boxes = st.toggle("Show Bounding Boxes", value=st.session_state.show_boxes, key="set_bbox")
        st.session_state.autosave   = st.toggle("Auto-save History",   value=st.session_state.autosave,   key="set_save")
        st.divider()
        st.markdown("**ℹ️ Model Info**")
        if st.session_state.model:
            cls_str = ', '.join(st.session_state.model.names.values())
            st.markdown(f'<div style="font-size:13px;line-height:2.2"><b>File:</b> <span class="mono">Flame_Best_Model.pt</span><br><b>Architecture:</b> <span class="mono">YOLOv11n</span><br><b>Classes:</b> <span class="mono">{cls_str}</span><br><b>Total Inspected:</b> <span class="mono">{len(st.session_state.history)}</span></div>', unsafe_allow_html=True)
        st.divider()
        if st.button("🗑 Clear History", use_container_width=True):
            st.session_state.history = []; st.success("Cleared"); st.rerun()
        if st.session_state.history:
            d = json.dumps([{k:v for k,v in h.items() if k!="thumb"} for h in st.session_state.history], indent=2)
            st.download_button("📥 Export JSON", d, "history.json", "application/json", use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)
