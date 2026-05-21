"""
PCB Solder Joint Defect Analyzer
=================================
Streamlit app that uses:
  - YOLOv11 (custom best.pt) for defect detection
  - TinyLlama LLM for explanation generation
  - SentenceTransformer + FAISS for RAG knowledge retrieval
  - gTTS for text-to-speech audio output
"""

import streamlit as st
import torch
import numpy as np
import cv2
import faiss
import tempfile
import os
import base64
import time
from PIL import Image
from io import BytesIO
from gtts import gTTS
from ultralytics import YOLO
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline

# ──────────────────────────────────────────────
# PAGE CONFIG
# ──────────────────────────────────────────────
st.set_page_config(
    page_title="PCB Defect Analyzer",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ──────────────────────────────────────────────
# GLOBAL CSS  (dark industrial theme)
# ──────────────────────────────────────────────
st.markdown("""
<style>
/* ── Base ─────────────────────────────────── */
@import url('https://fonts.googleapis.com/css2?family=Share+Tech+Mono&family=Rajdhani:wght@400;600;700&family=Exo+2:wght@300;400;600&display=swap');

html, body, [data-testid="stAppViewContainer"] {
    background: #0a0d12 !important;
    color: #c8d6e8 !important;
    font-family: 'Exo 2', sans-serif;
}

[data-testid="stHeader"] { background: transparent !important; }

/* ── Hero Banner ──────────────────────────── */
.hero-banner {
    background: linear-gradient(135deg, #0f1923 0%, #111d2e 50%, #0a1520 100%);
    border: 1px solid #1e3a5f;
    border-radius: 16px;
    padding: 2.5rem 3rem;
    margin-bottom: 2rem;
    position: relative;
    overflow: hidden;
}
.hero-banner::before {
    content: '';
    position: absolute;
    top: -50%; left: -50%;
    width: 200%; height: 200%;
    background: radial-gradient(ellipse at center, rgba(0,180,255,0.04) 0%, transparent 60%);
    pointer-events: none;
}
.hero-title {
    font-family: 'Rajdhani', sans-serif;
    font-size: 2.8rem;
    font-weight: 700;
    letter-spacing: 3px;
    color: #00b4ff;
    text-transform: uppercase;
    margin: 0;
    text-shadow: 0 0 30px rgba(0,180,255,0.4);
}
.hero-sub {
    font-family: 'Share Tech Mono', monospace;
    color: #4a8ab5;
    font-size: 0.85rem;
    letter-spacing: 2px;
    margin-top: 0.4rem;
}

/* ── Upload Zone ──────────────────────────── */
[data-testid="stFileUploader"] {
    background: #0d1621 !important;
    border: 2px dashed #1e4060 !important;
    border-radius: 12px !important;
    padding: 1rem !important;
    transition: border-color 0.3s;
}
[data-testid="stFileUploader"]:hover {
    border-color: #00b4ff !important;
}

/* ── Cards ────────────────────────────────── */
.defect-card {
    background: linear-gradient(145deg, #0d1a2a, #0f2035);
    border: 1px solid #1a3a5c;
    border-left: 4px solid #00b4ff;
    border-radius: 12px;
    padding: 1.6rem 1.8rem;
    margin-bottom: 1.5rem;
    box-shadow: 0 4px 24px rgba(0,0,0,0.5), inset 0 1px 0 rgba(255,255,255,0.03);
    animation: slideIn 0.4s ease-out;
}
@keyframes slideIn {
    from { opacity: 0; transform: translateY(16px); }
    to   { opacity: 1; transform: translateY(0); }
}
.defect-card.bad  { border-left-color: #ff4b4b; }
.defect-card.warn { border-left-color: #ffa500; }
.defect-card.good { border-left-color: #00d96b; }

.defect-title {
    font-family: 'Rajdhani', sans-serif;
    font-size: 1.5rem;
    font-weight: 700;
    letter-spacing: 2px;
    text-transform: uppercase;
    margin-bottom: 0.3rem;
}

.section-label {
    font-family: 'Share Tech Mono', monospace;
    font-size: 0.72rem;
    letter-spacing: 2px;
    color: #4a8ab5;
    text-transform: uppercase;
    margin-top: 1rem;
    margin-bottom: 0.2rem;
    border-bottom: 1px solid #1a3a5c;
    padding-bottom: 0.2rem;
}
.section-body {
    font-size: 0.93rem;
    color: #a8c0d8;
    line-height: 1.7;
}

/* ── Confidence Badge ─────────────────────── */
.conf-badge {
    display: inline-block;
    padding: 3px 12px;
    border-radius: 20px;
    font-family: 'Share Tech Mono', monospace;
    font-size: 0.78rem;
    letter-spacing: 1px;
    font-weight: 600;
    margin-left: 10px;
}
.conf-high   { background: rgba(0,217,107,0.15); color: #00d96b; border: 1px solid #00d96b44; }
.conf-medium { background: rgba(255,165,0,0.15);  color: #ffa500; border: 1px solid #ffa50044; }
.conf-low    { background: rgba(255,75,75,0.15);   color: #ff4b4b; border: 1px solid #ff4b4b44; }

/* ── Progress bar ─────────────────────────── */
.conf-bar-bg {
    background: #0d1621;
    border-radius: 6px;
    height: 8px;
    width: 100%;
    margin-top: 6px;
    overflow: hidden;
    border: 1px solid #1a3a5c;
}
.conf-bar-fill {
    height: 100%;
    border-radius: 6px;
    transition: width 0.8s ease;
}

/* ── Audio Card ───────────────────────────── */
.audio-card {
    background: #080f18;
    border: 1px solid #1a3a5c;
    border-radius: 10px;
    padding: 1rem 1.4rem;
    margin-top: 1rem;
}
.audio-label {
    font-family: 'Share Tech Mono', monospace;
    font-size: 0.72rem;
    color: #4a8ab5;
    letter-spacing: 2px;
    text-transform: uppercase;
    margin-bottom: 0.6rem;
}

/* ── Divider ──────────────────────────────── */
.scan-divider {
    height: 1px;
    background: linear-gradient(90deg, transparent, #1e4060, transparent);
    margin: 2rem 0;
}

/* ── Status Pills ─────────────────────────── */
.status-pill {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    padding: 4px 14px;
    border-radius: 20px;
    font-family: 'Share Tech Mono', monospace;
    font-size: 0.78rem;
    letter-spacing: 1px;
}
.pill-ok   { background: rgba(0,217,107,0.12); color: #00d96b; border: 1px solid #00d96b33; }
.pill-warn { background: rgba(255,75,75,0.12);  color: #ff4b4b; border: 1px solid #ff4b4b33; }

/* ── No-defect warning ────────────────────── */
.no-defect-box {
    background: linear-gradient(135deg, #1a0a0a, #200d0d);
    border: 1px solid #5c1a1a;
    border-left: 4px solid #ff4b4b;
    border-radius: 12px;
    padding: 2rem;
    text-align: center;
    font-family: 'Rajdhani', sans-serif;
    font-size: 1.2rem;
    color: #ff7070;
    letter-spacing: 1px;
}

/* ── Streamlit widget overrides ───────────── */
.stButton > button {
    background: linear-gradient(135deg, #0f3460, #1a4a8a) !important;
    color: #a0d4ff !important;
    border: 1px solid #1e5080 !important;
    border-radius: 8px !important;
    font-family: 'Rajdhani', sans-serif !important;
    font-weight: 600 !important;
    letter-spacing: 1px !important;
    padding: 0.4rem 1.2rem !important;
    transition: all 0.2s !important;
}
.stButton > button:hover {
    background: linear-gradient(135deg, #1a4a8a, #2060aa) !important;
    box-shadow: 0 0 16px rgba(0,100,200,0.4) !important;
}
div[data-testid="stImage"] img {
    border-radius: 10px;
    border: 1px solid #1a3a5c;
}
</style>
""", unsafe_allow_html=True)


# ──────────────────────────────────────────────
# CONSTANTS
# ──────────────────────────────────────────────
MODEL_PATH = "best.pt"          # path to your custom YOLO weights
EMBED_MODEL = "all-MiniLM-L6-v2"
LLM_MODEL   = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
MAX_NEW_TOKENS = 150

CLASS_COLORS = {
    "excessive": (255, 80,  80),   # red
    "poor":      (255, 165,  0),   # orange
    "spike":     (255, 255,  0),   # yellow
    "good":      (0,   217, 107),  # green
    "no good":   (200,   0,   0),  # dark red
}

# ──────────────────────────────────────────────
# PCB KNOWLEDGE BASE  (used by RAG)
# ──────────────────────────────────────────────
PCB_KNOWLEDGE = [
    "Excessive solder joints occur when too much solder is applied, creating bulges or bridges between pads, often caused by improper paste printing or reflow settings.",
    "Poor solder joints are characterized by incomplete wetting, dull or grainy surfaces, and weak mechanical bonds, typically from insufficient heat or contaminated surfaces.",
    "Solder spikes are sharp protrusions from the joint surface caused by rapid cooling, insufficient flux activity, or improper reflow temperature profiles.",
    "A good solder joint has a smooth, shiny, concave meniscus with complete wetting on both the component lead and the PCB pad.",
    "No-good solder joints represent critical defects requiring rework — they may involve cold joints, open circuits, or bridging that will cause field failures.",
    "Reflow temperature profiles must be carefully controlled: preheat at 150–180 °C, soak for flux activation, peak at 220–260 °C (lead-free), then controlled cooling.",
    "Cold solder joints result from movement during solidification or insufficient temperature, appearing dull and grainy rather than shiny and smooth.",
    "Solder bridging between adjacent pads causes short circuits and is detected visually or by automated optical inspection (AOI) systems.",
    "Flux residues left on PCBs can cause corrosion and electrical leakage over time; proper cleaning is essential for high-reliability assemblies.",
    "Component tombstoning occurs when one end of a chip component lifts during reflow due to uneven heating or pad imbalance.",
    "Automated optical inspection (AOI) uses cameras and algorithms to detect solder defects at speeds impossible for manual inspection.",
    "Solder paste stencil aperture design directly impacts solder volume; incorrect apertures lead to excessive or insufficient solder deposition.",
    "PCB surface finish (HASL, ENIG, OSP) affects solderability; oxidized or contaminated surfaces prevent proper wetting.",
    "Nitrogen atmosphere reflow reduces oxidation and improves solder joint quality, especially for fine-pitch components.",
    "Lead-free solder alloys (SAC305) have higher melting points and require tighter process control than traditional tin-lead solders.",
    "Voiding inside solder joints, detected by X-ray inspection, reduces thermal and electrical performance and can cause premature failure.",
    "Solder joint reliability depends on intermetallic compound (IMC) thickness; excessive IMC layers from high-temperature exposure reduce joint strength.",
    "Wettability is critical for reliable solder joints; inadequate flux or oxidized surfaces prevent proper metallurgical bonding.",
    "Print-to-board registration accuracy in stencil printing determines whether solder paste is deposited on the correct pads.",
    "Visual inspection criteria for solder joints are defined by IPC-A-610, the industry standard for acceptability of electronic assemblies.",
]

# ──────────────────────────────────────────────
# MODEL LOADERS  (cached so they load only once)
# ──────────────────────────────────────────────

@st.cache_resource(show_spinner=False)
def load_yolo():
    """Load the custom YOLOv11 model."""
    model = YOLO(MODEL_PATH)
    return model


@st.cache_resource(show_spinner=False)
def load_embed_model():
    """Load the SentenceTransformer for RAG embeddings."""
    return SentenceTransformer(EMBED_MODEL)


@st.cache_resource(show_spinner=False)
def build_faiss_index(embed_model):
    """Embed PCB knowledge and build a FAISS index."""
    embeddings = embed_model.encode(PCB_KNOWLEDGE, convert_to_numpy=True)
    dim = embeddings.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(embeddings.astype(np.float32))
    return index, PCB_KNOWLEDGE


@st.cache_resource(show_spinner=False)
def load_llm():
    """Load TinyLlama tokenizer and pipeline (CPU mode)."""
    tokenizer = AutoTokenizer.from_pretrained(LLM_MODEL)
    llm_pipe = pipeline(
        "text-generation",
        model=LLM_MODEL,
        tokenizer=tokenizer,
        torch_dtype=torch.float32,
        device_map="cpu",
        max_new_tokens=MAX_NEW_TOKENS,
        do_sample=False,       # greedy = faster on CPU
        temperature=1.0,
        repetition_penalty=1.15,
    )
    return llm_pipe


# ──────────────────────────────────────────────
# RAG RETRIEVAL
# ──────────────────────────────────────────────

def retrieve_context(query: str, embed_model, faiss_index, knowledge: list, top_k: int = 3) -> str:
    """Retrieve top-k relevant knowledge chunks for the defect query."""
    q_emb = embed_model.encode([query], convert_to_numpy=True).astype(np.float32)
    _, indices = faiss_index.search(q_emb, top_k)
    return " ".join([knowledge[i] for i in indices[0]])


# ──────────────────────────────────────────────
# LLM EXPLANATION GENERATION
# ──────────────────────────────────────────────

def generate_explanation(defect: str, confidence: float, context: str, llm_pipe) -> dict:
    """
    Use TinyLlama to generate a structured explanation for a detected defect.
    Returns a dict with keys: description, root_cause, impact, solution, prevention.
    """
    prompt = f"""<|system|>
You are a PCB quality-control expert. Given a solder joint defect, produce a concise technical analysis.
Context knowledge: {context}
</s>
<|user|>
Defect detected: "{defect}" (confidence {confidence:.0%}).
Write exactly 5 short sections:
DESCRIPTION: (2-3 sentences about what this defect looks like)
ROOT CAUSE: (2-3 sentences about why it happens)
IMPACT: (2-3 sentences on electrical/mechanical effects)
SOLUTION: (2-3 sentences on how to fix it)
PREVENTION: (2-3 sentences on how to prevent it)
</s>
<|assistant|>"""

    result = llm_pipe(prompt)[0]["generated_text"]
    # Extract only the assistant reply
    reply = result.split("<|assistant|>")[-1].strip()

    # Parse sections robustly
    sections = {"description": "", "root_cause": "", "impact": "", "solution": "", "prevention": ""}
    keys_map = {
        "DESCRIPTION": "description",
        "ROOT CAUSE":  "root_cause",
        "IMPACT":      "impact",
        "SOLUTION":    "solution",
        "PREVENTION":  "prevention",
    }
    for label, key in keys_map.items():
        if label + ":" in reply:
            start = reply.index(label + ":") + len(label) + 1
            # Find next label to determine end
            end = len(reply)
            for other_label in keys_map:
                if other_label != label and other_label + ":" in reply:
                    pos = reply.index(other_label + ":")
                    if pos > start:
                        end = min(end, pos)
            sections[key] = reply[start:end].strip()

    # Fallback: if parsing failed, put the full reply in description
    if not any(sections.values()):
        sections["description"] = reply[:400]

    return sections


# ──────────────────────────────────────────────
# AUDIO GENERATION
# ──────────────────────────────────────────────

def generate_audio(text: str) -> bytes:
    """Convert text to speech using gTTS and return MP3 bytes."""
    tts = gTTS(text=text, lang="en", slow=False)
    buf = BytesIO()
    tts.write_to_fp(buf)
    return buf.getvalue()


def audio_player_html(audio_bytes: bytes) -> str:
    """Return an HTML audio element with inline base64 audio."""
    b64 = base64.b64encode(audio_bytes).decode()
    return f"""
    <audio controls style="width:100%;margin-top:6px;border-radius:8px;
           filter:invert(0.85) hue-rotate(180deg) brightness(0.9);">
      <source src="data:audio/mp3;base64,{b64}" type="audio/mp3">
    </audio>"""


# ──────────────────────────────────────────────
# YOLO DETECTION + ANNOTATION
# ──────────────────────────────────────────────

def detect_defects(image: np.ndarray, yolo_model) -> tuple[np.ndarray, list]:
    """
    Run YOLO inference on the image.
    Returns annotated image and list of detection dicts.
    """
    results = yolo_model(image, conf=0.25, iou=0.45, verbose=False)[0]
    detections = []
    annotated = image.copy()

    for box in results.boxes:
        cls_id   = int(box.cls[0])
        conf     = float(box.conf[0])
        cls_name = yolo_model.names[cls_id]
        x1, y1, x2, y2 = map(int, box.xyxy[0])

        color = CLASS_COLORS.get(cls_name, (200, 200, 200))
        # Draw bounding box
        cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
        # Label background
        label = f"{cls_name}  {conf:.0%}"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
        cv2.rectangle(annotated, (x1, y1 - th - 10), (x1 + tw + 6, y1), color, -1)
        cv2.putText(annotated, label, (x1 + 3, y1 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 1, cv2.LINE_AA)

        detections.append({"class": cls_name, "confidence": conf,
                            "bbox": (x1, y1, x2, y2)})

    return annotated, detections


# ──────────────────────────────────────────────
# CONFIDENCE HELPERS
# ──────────────────────────────────────────────

def conf_level(conf: float) -> tuple[str, str]:
    """Return (label, css_class) for confidence score."""
    if conf >= 0.75:
        return "HIGH",   "conf-high"
    elif conf >= 0.50:
        return "MEDIUM", "conf-medium"
    else:
        return "LOW",    "conf-low"


def card_class(defect: str) -> str:
    if defect in ("good",):
        return "good"
    elif defect in ("excessive", "spike", "no good"):
        return "bad"
    else:
        return "warn"


def bar_color(defect: str) -> str:
    if defect == "good":
        return "#00d96b"
    elif defect in ("excessive", "spike", "no good"):
        return "#ff4b4b"
    else:
        return "#ffa500"


# ──────────────────────────────────────────────
# SIDEBAR
# ──────────────────────────────────────────────

with st.sidebar:
    st.markdown("""
    <div style='font-family:Rajdhani,sans-serif;font-size:1.1rem;
                color:#00b4ff;letter-spacing:2px;text-transform:uppercase;
                border-bottom:1px solid #1a3a5c;padding-bottom:0.6rem;margin-bottom:1rem;'>
        ⚙ Configuration
    </div>""", unsafe_allow_html=True)

    conf_threshold = st.slider("YOLO Confidence Threshold", 0.1, 0.95, 0.25, 0.05)
    iou_threshold  = st.slider("IoU Threshold",             0.1, 0.95, 0.45, 0.05)
    max_tokens     = st.slider("Max LLM Tokens",             50,  300,  150,   10)
    enable_audio   = st.toggle("Enable Audio Generation", value=True)

    st.markdown("<div class='scan-divider'></div>", unsafe_allow_html=True)
    st.markdown("""
    <div style='font-family:Share Tech Mono,monospace;font-size:0.72rem;
                color:#4a8ab5;letter-spacing:1px;'>
    DEFECT CLASSES<br><br>
    🔴 excessive<br>
    🟠 poor<br>
    🟡 spike<br>
    🟢 good<br>
    ❌ no good
    </div>""", unsafe_allow_html=True)


# ──────────────────────────────────────────────
# HERO HEADER
# ──────────────────────────────────────────────

st.markdown("""
<div class='hero-banner'>
  <div class='hero-title'>🔬 PCB Defect Analyzer</div>
  <div class='hero-sub'>▸ YOLOv11 · TinyLlama · RAG · gTTS · Real-time Analysis</div>
</div>
""", unsafe_allow_html=True)


# ──────────────────────────────────────────────
# MODEL LOADING  (with progress indicators)
# ──────────────────────────────────────────────

col_load1, col_load2, col_load3 = st.columns(3)

with col_load1:
    with st.spinner("Loading YOLO model…"):
        try:
            yolo_model = load_yolo()
            st.success("✅ YOLO Loaded")
        except Exception as e:
            st.error(f"❌ YOLO: {e}")
            yolo_model = None

with col_load2:
    with st.spinner("Loading embeddings & FAISS…"):
        embed_model = load_embed_model()
        faiss_index, knowledge = build_faiss_index(embed_model)
        st.success("✅ RAG System Ready")

with col_load3:
    with st.spinner("Loading TinyLlama…"):
        llm_pipe = load_llm()
        st.success("✅ LLM Loaded")

st.markdown("<div class='scan-divider'></div>", unsafe_allow_html=True)


# ──────────────────────────────────────────────
# IMAGE UPLOAD
# ──────────────────────────────────────────────

st.markdown("""
<div style='font-family:Rajdhani,sans-serif;font-size:1.2rem;color:#a0c8e8;
            letter-spacing:2px;text-transform:uppercase;margin-bottom:0.8rem;'>
📁 Upload PCB Image
</div>""", unsafe_allow_html=True)

uploaded_file = st.file_uploader(
    "Drag & drop or click to upload a PCB image",
    type=["jpg", "jpeg", "png", "bmp", "webp"],
    label_visibility="collapsed",
)

if uploaded_file is None:
    st.markdown("""
    <div style='text-align:center;padding:3rem 0;font-family:Share Tech Mono,monospace;
                color:#2a4a6a;letter-spacing:2px;font-size:0.9rem;'>
        ⬆ UPLOAD A PCB IMAGE TO BEGIN ANALYSIS
    </div>""", unsafe_allow_html=True)
    st.stop()


# ──────────────────────────────────────────────
# PROCESS UPLOADED IMAGE
# ──────────────────────────────────────────────

file_bytes = np.frombuffer(uploaded_file.read(), np.uint8)
image_bgr  = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
image_rgb  = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

col_orig, col_anno = st.columns(2, gap="medium")

with col_orig:
    st.markdown("<div class='section-label'>Original Image</div>", unsafe_allow_html=True)
    st.image(image_rgb, use_container_width=True)

# ── Run YOLO ──────────────────────────────────
if yolo_model is None:
    st.error("YOLO model not loaded. Cannot run detection.")
    st.stop()

with st.spinner("🔍 Running YOLO detection…"):
    annotated_bgr, detections = detect_defects(image_bgr, yolo_model)
    annotated_rgb = cv2.cvtColor(annotated_bgr, cv2.COLOR_BGR2RGB)

with col_anno:
    st.markdown("<div class='section-label'>Annotated Detection</div>", unsafe_allow_html=True)
    st.image(annotated_rgb, use_container_width=True)

st.markdown("<div class='scan-divider'></div>", unsafe_allow_html=True)


# ──────────────────────────────────────────────
# DETECTION SUMMARY
# ──────────────────────────────────────────────

n_defects = len(detections)

if n_defects == 0:
    st.markdown("""
    <div class='no-defect-box'>
        ⚠ NO DEFECTS DETECTED<br>
        <span style='font-size:0.9rem;color:#c04040;'>
        Try lowering the confidence threshold or check image quality.</span>
    </div>""", unsafe_allow_html=True)
    st.stop()

pill = "pill-warn" if any(d["class"] != "good" for d in detections) else "pill-ok"
pill_icon = "⚠" if pill == "pill-warn" else "✅"

st.markdown(f"""
<div style='display:flex;align-items:center;gap:1rem;margin-bottom:1.5rem;'>
  <div style='font-family:Rajdhani,sans-serif;font-size:1.4rem;color:#a0c8e8;
              letter-spacing:2px;text-transform:uppercase;'>
    🧪 Analysis Results
  </div>
  <span class='status-pill {pill}'>{pill_icon} {n_defects} DEFECT{"S" if n_defects>1 else ""} DETECTED</span>
</div>
""", unsafe_allow_html=True)


# ──────────────────────────────────────────────
# PER-DEFECT CARDS
# ──────────────────────────────────────────────

for idx, det in enumerate(detections):
    defect_name = det["class"]
    conf        = det["confidence"]
    bbox        = det["bbox"]

    conf_label, conf_css = conf_level(conf)
    card_css = card_class(defect_name)
    fill_color = bar_color(defect_name)

    # ── Retrieve RAG context ──────────────────
    query   = f"{defect_name} solder joint defect PCB"
    context = retrieve_context(query, embed_model, faiss_index, knowledge)

    # ── Generate LLM explanation ──────────────
    with st.spinner(f"🤖 Generating explanation for defect {idx+1}/{n_defects}…"):
        explanation = generate_explanation(defect_name, conf, context, llm_pipe)

    # ── Build readable audio text ─────────────
    audio_text = (
        f"Defect detected: {defect_name}. Confidence: {conf:.0%}. "
        f"Description: {explanation['description']} "
        f"Root Cause: {explanation['root_cause']} "
        f"Impact: {explanation['impact']} "
        f"Solution: {explanation['solution']} "
        f"Prevention: {explanation['prevention']}"
    )

    # ── Render card ───────────────────────────
    st.markdown(f"""
    <div class='defect-card {card_css}'>
      <div class='defect-title'>
        #{idx+1} — {defect_name.upper()}
        <span class='conf-badge {conf_css}'>{conf_label} · {conf:.1%}</span>
      </div>

      <div style='font-family:Share Tech Mono,monospace;font-size:0.75rem;color:#3a6a8a;
                  margin-bottom:0.5rem;'>
        BBOX [{bbox[0]}, {bbox[1]}, {bbox[2]}, {bbox[3]}]
      </div>

      <div class='conf-bar-bg'>
        <div class='conf-bar-fill'
             style='width:{conf*100:.1f}%; background:{fill_color};'></div>
      </div>

      <div class='section-label'>📋 Description</div>
      <div class='section-body'>{explanation['description'] or "—"}</div>

      <div class='section-label'>🔩 Root Cause</div>
      <div class='section-body'>{explanation['root_cause'] or "—"}</div>

      <div class='section-label'>⚡ Impact</div>
      <div class='section-body'>{explanation['impact'] or "—"}</div>

      <div class='section-label'>🔧 Solution</div>
      <div class='section-body'>{explanation['solution'] or "—"}</div>

      <div class='section-label'>🛡 Prevention</div>
      <div class='section-body'>{explanation['prevention'] or "—"}</div>
    </div>
    """, unsafe_allow_html=True)

    # ── Audio section ─────────────────────────
    if enable_audio:
        with st.spinner(f"🔊 Generating audio for defect {idx+1}…"):
            try:
                audio_bytes = generate_audio(audio_text)

                st.markdown(f"""
                <div class='audio-card'>
                  <div class='audio-label'>🔊 Audio Explanation — {defect_name.upper()}</div>
                """, unsafe_allow_html=True)
                st.markdown(audio_player_html(audio_bytes), unsafe_allow_html=True)
                st.markdown("</div>", unsafe_allow_html=True)

                # Download button
                st.download_button(
                    label=f"⬇ Download Audio — {defect_name} #{idx+1}",
                    data=audio_bytes,
                    file_name=f"defect_{idx+1}_{defect_name}.mp3",
                    mime="audio/mp3",
                    key=f"dl_audio_{idx}",
                )
            except Exception as e:
                st.warning(f"Audio generation failed: {e}")

    st.markdown("<div class='scan-divider'></div>", unsafe_allow_html=True)


# ──────────────────────────────────────────────
# FOOTER
# ──────────────────────────────────────────────

st.markdown("""
<div style='text-align:center;padding:2rem 0 1rem;
            font-family:Share Tech Mono,monospace;font-size:0.72rem;
            color:#2a4a6a;letter-spacing:2px;'>
  PCB DEFECT ANALYZER  ·  YOLOv11 + TinyLlama + FAISS + gTTS  ·  BUILT WITH STREAMLIT
</div>""", unsafe_allow_html=True)
