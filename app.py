"""
Littlelome AI Vision — Streamlit App
YOLOv11n · Flame Detection Model

วิธีรัน:
  pip install streamlit ultralytics pillow opencv-python-headless
  streamlit run app.py

โฟลเดอร์:
  app.py
  Flame_Best_Model.pt   ← โมเดลของคุณ
"""

import streamlit as st
from ultralytics import YOLO
from PIL import Image
import numpy as np
import cv2
import tempfile
import time
import os
from datetime import datetime

# ─────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="Littlelome AI Vision",
    page_icon="🔥",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;700;800&family=JetBrains+Mono:wght@400;500&display=swap');
html, body, [class*="css"] { font-family: 'Syne', sans-serif !important; }
.stApp { background: #0d1117; color: #e6edf3; }
div[data-testid="metric-container"] {
    background: #161b22; border: 1px solid #30363d; border-radius: 10px; padding: 16px;
}
div[data-testid="metric-container"] label { color: #8b949e !important; font-size: 12px !important; }
div[data-testid="metric-container"] div[data-testid="stMetricValue"] {
    font-family: 'JetBrains Mono' !important; font-size: 28px !important;
}
[data-testid="stSidebar"] { background: #161b22 !important; border-right: 1px solid #30363d; }
[data-testid="stSidebar"] * { color: #e6edf3 !important; }
.stButton > button {
    background: #f85149 !important; color: white !important; border: none !important;
    border-radius: 8px !important; font-family: 'Syne', sans-serif !important;
    font-weight: 700 !important; padding: 10px 24px !important;
}
.stButton > button:hover { background: #c1292b !important; }
.fault-box {
    background: #2d1515; border: 2px solid #f85149; border-radius: 10px;
    padding: 16px 20px; margin: 12px 0;
}
.ok-box {
    background: #0e2a1a; border: 2px solid #3fb950; border-radius: 10px;
    padding: 16px 20px; margin: 12px 0;
}
.detect-card {
    background: #1c2230; border: 1px solid #30363d;
    border-radius: 8px; padding: 12px 16px; margin: 6px 0;
}
#MainMenu, footer, header { visibility: hidden; }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# SESSION STATE
# ─────────────────────────────────────────────
if "history" not in st.session_state:
    st.session_state.history = []
if "model" not in st.session_state:
    st.session_state.model = None

# ─────────────────────────────────────────────
# CLASS CONFIG — ตรงกับ Flame_Best_Model.pt (class: Flame)
# ─────────────────────────────────────────────
DEFECT_MAP = {
    "flame":      {"label": "🔥 Flame (ไฟ/เปลวไฟ)", "color_bgr": (73, 81, 248),  "color_hex": "#f85149"},
    "fire":       {"label": "🔥 Fire (ไฟ)",           "color_bgr": (73, 81, 248),  "color_hex": "#f85149"},
    "smoke":      {"label": "💨 Smoke (ควัน)",         "color_bgr": (158,148,139),  "color_hex": "#8b949e"},
    "rust":       {"label": "🟠 Rust (สนิม)",          "color_bgr": (65, 179, 227), "color_hex": "#e3b341"},
    "dent":       {"label": "🔵 Dent (รอยบุบ)",         "color_bgr": (255,136,  33), "color_hex": "#2188ff"},
    "corrosion":  {"label": "🟠 Rust (สนิม)",          "color_bgr": (65, 179, 227), "color_hex": "#e3b341"},
    "scratch":    {"label": "🔵 Scratch (รอยขีด)",     "color_bgr": (255,136,  33), "color_hex": "#2188ff"},
}

def get_info(cls_name: str) -> dict:
    return DEFECT_MAP.get(cls_name.lower(), {
        "label":     f"⚠️ {cls_name.capitalize()}",
        "color_bgr": (73, 81, 248),
        "color_hex": "#f85149",
    })

def get_severity(conf: float) -> str:
    if conf >= 0.80: return "🔴 High"
    if conf >= 0.55: return "🟡 Medium"
    return "🟢 Low"

# ─────────────────────────────────────────────
# DRAW BOUNDING BOXES
# ─────────────────────────────────────────────
def draw_boxes(img: Image.Image, detections: list) -> Image.Image:
    frame = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
    for d in detections:
        x1,y1,x2,y2 = int(d["x1"]),int(d["y1"]),int(d["x2"]),int(d["y2"])
        c = d["color_bgr"]
        cv2.rectangle(frame, (x1,y1),(x2,y2), c, 3)
        txt = f"{d['cls'].capitalize()}  {d['conf']:.0%}"
        (tw,th),_ = cv2.getTextSize(txt, cv2.FONT_HERSHEY_DUPLEX, 0.6, 1)
        cv2.rectangle(frame,(x1,y1-th-12),(x1+tw+12,y1),c,-1)
        cv2.putText(frame, txt, (x1+6,y1-7),
                    cv2.FONT_HERSHEY_DUPLEX, 0.6, (255,255,255), 1, cv2.LINE_AA)
    return Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

# ─────────────────────────────────────────────
# INFERENCE
# ─────────────────────────────────────────────
def run_inference(model, img: Image.Image, threshold: float):
    t0 = time.time()
    results = model(img, conf=threshold, verbose=False)[0]
    elapsed = time.time() - t0
    detections = []
    for box in results.boxes:
        cls_id   = int(box.cls[0])
        cls_name = model.names[cls_id]
        conf     = float(box.conf[0])
        x1,y1,x2,y2 = box.xyxy[0].tolist()
        info = get_info(cls_name)
        detections.append({
            "cls":       cls_name,
            "label":     info["label"],
            "conf":      conf,
            "severity":  get_severity(conf),
            "color_bgr": info["color_bgr"],
            "color_hex": info["color_hex"],
            "x1":x1,"y1":y1,"x2":x2,"y2":y2,
        })
    detections.sort(key=lambda d: d["conf"], reverse=True)
    return detections, elapsed

def add_history(source, status, dets, thumb=None):
    st.session_state.history.insert(0, {
        "time":       datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "source":     source,
        "status":     status,
        "detections": dets,
        "thumb":      thumb,
    })

# ─────────────────────────────────────────────
# AUTO LOAD MODEL
# ─────────────────────────────────────────────
MODEL_FILE = "Flame_Best_Model.pt"

@st.cache_resource
def load_model_cached(path):
    return YOLO(path)

if st.session_state.model is None and os.path.exists(MODEL_FILE):
    with st.spinner(f"⏳ Loading {MODEL_FILE}…"):
        st.session_state.model = load_model_cached(MODEL_FILE)

# ─────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🔥 Littlelome AI Vision")
    st.markdown("*Flame Detection · YOLOv11*")
    st.divider()

    if st.session_state.model:
        classes = list(st.session_state.model.names.values())
        st.success(f"✅ {MODEL_FILE} loaded")
        st.markdown("**Detected Classes:**")
        for c in classes:
            info = get_info(c)
            st.markdown(f"- {info['label']}")
    else:
        st.error("❌ Model not found")
        st.markdown(f"Place `{MODEL_FILE}` in the same folder, or upload:")
        up = st.file_uploader("Upload .pt file", type=["pt"])
        if up:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pt") as tmp:
                tmp.write(up.read())
            st.session_state.model = load_model_cached(tmp.name)
            st.rerun()

    st.divider()
    st.markdown("### ⚙️ Settings")
    threshold  = st.slider("Confidence Threshold", 0.10, 0.95, 0.40, 0.05, format="%.0f%%")
    show_boxes = st.toggle("Bounding Boxes", value=True)

    st.divider()
    st.markdown("### 📊 Session")
    total  = len(st.session_state.history)
    faults = sum(1 for h in st.session_state.history if h["status"]=="FAULT")
    c1,c2 = st.columns(2)
    c1.metric("Inspected", total)
    c2.metric("Faults",    faults)
    if total:
        st.metric("Clear Rate", f"{(total-faults)/total:.0%}")

    if st.button("🗑 Clear History", use_container_width=True):
        st.session_state.history = []
        st.rerun()

# ─────────────────────────────────────────────
# HEADER
# ─────────────────────────────────────────────
st.markdown("""
<div style="display:flex;align-items:center;gap:14px;padding:18px 24px;
  background:linear-gradient(135deg,#1c0a0a,#2d1515);
  border:1px solid #f8514940;border-radius:12px;margin-bottom:20px">
  <div style="font-size:38px">🔥</div>
  <div>
    <div style="font-size:20px;font-weight:800">
      Littlelome<span style="color:#f85149">AI</span> Vision
      <span style="font-size:13px;font-weight:400;color:#8b949e;margin-left:8px">
        Flame Detection · YOLOv11n
      </span>
    </div>
    <div style="font-size:12px;color:#8b949e;margin-top:2px">
      ตรวจจับไฟ / เปลวไฟ · Real-time inference
    </div>
  </div>
</div>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# TABS
# ─────────────────────────────────────────────
tab1, tab2, tab3, tab4 = st.tabs([
    "🖼️  Upload Image",
    "🎬  Upload Video",
    "📹  Live Camera",
    "📋  History"
])

# ══════════════════════════════════════════════
# TAB 1 — IMAGE
# ══════════════════════════════════════════════
with tab1:
    st.markdown("### Analyze Image")
    uploaded_img = st.file_uploader(
        "Drop image here (JPG, PNG, WEBP)",
        type=["jpg","jpeg","png","webp","bmp"],
        key="img_up"
    )
    if uploaded_img:
        img = Image.open(uploaded_img).convert("RGB")
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("**Original**")
            st.image(img, use_container_width=True)
        with col2:
            st.markdown("**Result**")
            if not st.session_state.model:
                st.warning("Load model first (sidebar)")
            else:
                if st.button("🔍 Detect Flame", key="btn_img", use_container_width=True):
                    with st.spinner("Running YOLOv11…"):
                        dets, elapsed = run_inference(st.session_state.model, img, threshold)
                    if dets:
                        out = draw_boxes(img.copy(), dets) if show_boxes else img
                        st.image(out, use_container_width=True)
                        st.markdown(f"""
                        <div class="fault-box">
                          <div style="font-size:20px;font-weight:800;color:#f85149">
                            🔥 FLAME DETECTED
                          </div>
                          <div style="font-size:12px;color:#8b949e;margin-top:4px">
                            {len(dets)} detection(s) · {elapsed*1000:.0f}ms
                          </div>
                        </div>""", unsafe_allow_html=True)
                        for d in dets:
                            st.markdown(f"""
                            <div class="detect-card">
                              <div style="display:flex;justify-content:space-between;align-items:center">
                                <span style="color:{d['color_hex']};font-weight:800;font-size:15px">{d['label']}</span>
                                <span style="font-family:'JetBrains Mono';font-size:16px;font-weight:700">{d['conf']:.1%}</span>
                              </div>
                              <div style="color:#8b949e;font-size:12px;margin-top:4px">{d['severity']}</div>
                              <div style="height:4px;background:#30363d;border-radius:2px;margin-top:8px">
                                <div style="width:{d['conf']*100:.0f}%;height:100%;background:{d['color_hex']};border-radius:2px"></div>
                              </div>
                            </div>""", unsafe_allow_html=True)
                        add_history("Image", "FAULT", dets, img.copy())
                    else:
                        st.image(img, use_container_width=True)
                        st.markdown("""
                        <div class="ok-box">
                          <div style="font-size:20px;font-weight:800;color:#3fb950">
                            ✅ NO FLAME DETECTED
                          </div>
                          <div style="font-size:12px;color:#8b949e;margin-top:4px">Image is clear</div>
                        </div>""", unsafe_allow_html=True)
                        add_history("Image", "OK", [], img.copy())

# ══════════════════════════════════════════════
# TAB 2 — VIDEO
# ══════════════════════════════════════════════
with tab2:
    st.markdown("### Analyze Video")
    uploaded_vid = st.file_uploader(
        "Drop video file (MP4, MOV, AVI)",
        type=["mp4","mov","avi","mkv","webm"],
        key="vid_up"
    )
    col_va, col_vb = st.columns(2)
    with col_va:
        frame_interval = st.slider("Analyze every (s)", 1, 10, 2, key="vid_int")
    with col_vb:
        max_frames = st.slider("Max frames", 5, 30, 12, key="vid_max")

    if uploaded_vid:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp:
            tmp.write(uploaded_vid.read())
            vid_path = tmp.name
        st.video(vid_path)

        if not st.session_state.model:
            st.warning("Load model first")
        elif st.button("▶ Analyze Video", key="btn_vid", use_container_width=True):
            cap    = cv2.VideoCapture(vid_path)
            fps_v  = cap.get(cv2.CAP_PROP_FPS) or 25
            step_f = max(1, int(fps_v * frame_interval))
            prog   = st.progress(0, "Reading frames…")

            frames_buf = []
            fi = 0
            while len(frames_buf) < max_frames:
                ret, frm = cap.read()
                if not ret: break
                if fi % step_f == 0:
                    frames_buf.append((fi, fi/fps_v, frm))
                fi += 1
            cap.release()

            frame_results = []
            thumb_cols = st.columns(min(4, len(frames_buf)))
            for i, (fidx, tsec, frm) in enumerate(frames_buf):
                prog.progress((i+1)/len(frames_buf), f"Frame {i+1}/{len(frames_buf)}…")
                pil = Image.fromarray(cv2.cvtColor(frm, cv2.COLOR_BGR2RGB))
                dets, _ = run_inference(st.session_state.model, pil, threshold)
                disp = draw_boxes(pil.copy(), dets) if (show_boxes and dets) else pil
                cap_txt = f"🔥 @{tsec:.1f}s" if dets else f"✅ @{tsec:.1f}s"
                with thumb_cols[i % 4]:
                    st.image(disp, caption=cap_txt, use_container_width=True)
                frame_results.append({"time":tsec,"status":"FAULT" if dets else "OK","detections":dets})

            prog.empty()
            st.divider()

            fault_f  = [r for r in frame_results if r["status"]=="FAULT"]
            all_dets = [d for r in frame_results for d in r["detections"]]
            m1,m2,m3,m4 = st.columns(4)
            m1.metric("Frames",       len(frame_results))
            m2.metric("🔥 Flame",     len(fault_f))
            m3.metric("✅ Clear",     len(frame_results)-len(fault_f))
            m4.metric("Clear Rate",   f"{(len(frame_results)-len(fault_f))/max(len(frame_results),1):.0%}")
            if fault_f:
                ts = ", ".join(f"{r['time']:.1f}s" for r in fault_f)
                st.error(f"🔥 Flame detected at: **{ts}**")
            else:
                st.success("✅ No flame detected in any frame")
            add_history("Video", "FAULT" if fault_f else "OK", all_dets[:5])

# ══════════════════════════════════════════════
# TAB 3 — LIVE CAMERA
# ══════════════════════════════════════════════
with tab3:
    st.markdown("### Live Camera")
    cam_src = st.selectbox("Camera source", [0, 1, 2, "RTSP/HTTP URL"], key="cam_src_sel")
    if cam_src == "RTSP/HTTP URL":
        cam_src = st.text_input("Enter URL", "rtsp://")

    col_feed, col_ctrl = st.columns([3,1])
    with col_ctrl:
        run_cam      = st.toggle("▶ Start Camera", key="cam_on")
        scan_every_s = st.slider("Scan every (s)", 1, 10, 2, key="cam_scan_s")
        st.divider()
        slot_st  = st.empty()
        slot_res = st.empty()
        slot_ms  = st.empty()
    with col_feed:
        slot_fr = st.empty()

    if run_cam:
        if not st.session_state.model:
            st.warning("Load model first")
        else:
            src_val = int(cam_src) if isinstance(cam_src, int) else cam_src
            cap = cv2.VideoCapture(src_val)
            if not cap.isOpened():
                st.error("❌ Cannot open camera. Check device or URL.")
            else:
                slot_st.success("🟢 Camera running")
                last_t    = 0
                last_dets = []
                for _ in range(600):
                    if not st.session_state.get("cam_on", False): break
                    ret, frm = cap.read()
                    if not ret: break
                    now      = time.time()
                    annotated = frm.copy()
                    if now - last_t >= scan_every_s:
                        pil = Image.fromarray(cv2.cvtColor(frm, cv2.COLOR_BGR2RGB))
                        last_dets, el = run_inference(st.session_state.model, pil, threshold)
                        last_t = now
                        slot_ms.markdown(f"`{el*1000:.0f}ms` inference")
                        if last_dets:
                            slot_res.error(f"🔥 FLAME — {last_dets[0]['conf']:.0%}")
                            add_history("Camera", "FAULT", last_dets)
                        else:
                            slot_res.success("✅ Clear")
                    for d in last_dets:
                        x1,y1,x2,y2 = int(d["x1"]),int(d["y1"]),int(d["x2"]),int(d["y2"])
                        c = d["color_bgr"]
                        cv2.rectangle(annotated,(x1,y1),(x2,y2),c,3)
                        cv2.putText(annotated,f"Flame {d['conf']:.0%}",
                                    (x1+4,y1-8),cv2.FONT_HERSHEY_DUPLEX,0.6,(255,255,255),1)
                    slot_fr.image(
                        cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB),
                        channels="RGB", use_container_width=True
                    )
                cap.release()
                slot_st.info("⏹ Stopped")

# ══════════════════════════════════════════════
# TAB 4 — HISTORY
# ══════════════════════════════════════════════
with tab4:
    st.markdown("### Inspection History")
    if not st.session_state.history:
        st.info("No inspections yet.")
    else:
        for i, h in enumerate(st.session_state.history):
            is_fault = h["status"] == "FAULT"
            icon     = "🔥 FAULT" if is_fault else "✅ OK"
            dets_str = ", ".join(d["label"] for d in h["detections"]) or "No detections"
            with st.expander(f"{icon}  ·  {h['source']}  ·  {dets_str}  ·  {h['time']}",
                             expanded=(i==0 and is_fault)):
                c1, c2 = st.columns([2,1])
                with c1:
                    if h.get("thumb"):
                        try:
                            disp = draw_boxes(h["thumb"].copy(), h["detections"]) \
                                   if (show_boxes and h["detections"]) else h["thumb"]
                            st.image(disp, use_container_width=True)
                        except Exception:
                            pass
                with c2:
                    color = "#f85149" if is_fault else "#3fb950"
                    st.markdown(f"**Status:** <span style='color:{color};font-weight:800'>{icon}</span>",
                                unsafe_allow_html=True)
                    st.markdown(f"**Source:** `{h['source']}`")
                    st.markdown(f"**Time:** `{h['time']}`")
                    for d in h["detections"]:
                        st.markdown(f"- {d['label']} `{d['conf']:.1%}` {d['severity']}")
