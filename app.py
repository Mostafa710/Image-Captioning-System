import io
import math
import requests
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
import torchvision.models as models
import streamlit as st
from PIL import Image


# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Image Captioning",
    page_icon="🖼️",
    layout="wide",
)

# ── Custom CSS — Dark gradient abstract theme ──────────────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }

    /* ── Main background — deep dark with subtle gradient ── */
    .stApp {
        background: radial-gradient(ellipse at 60% 50%,
                    #1c1c1c 0%,
                    #111111 50%,
                    #0a0a0a 100%);
        min-height: 100vh;
    }

    /* ── Animated wave lines overlay (CSS only) ── */
    .stApp::before {
        content: "";
        position: fixed;
        top: 0; left: 0;
        width: 100%; height: 100%;
        background:
            repeating-linear-gradient(
                135deg,
                transparent,
                transparent 80px,
                rgba(255,255,255,0.015) 80px,
                rgba(255,255,255,0.015) 81px
            );
        pointer-events: none;
        z-index: 0;
    }

    /* ── Sidebar ── */
    [data-testid="stSidebar"] {
        background-color: #141414 !important;
        border-right: 1px solid #2a2a2a;
    }
    [data-testid="stSidebar"] * {
        color: #CCCCCC !important;
    }

    /* ── Title ── */
    h1 {
        color: #FFFFFF !important;
        font-weight: 700 !important;
        font-size: 2.2rem !important;
        letter-spacing: -0.5px;
    }

    /* ── Subheaders ── */
    h2, h3 {
        color: #FFFFFF !important;
        font-weight: 600 !important;
    }

    /* ── Paragraph / body text ── */
    p, label, div {
        color: #DDDDDD;
    }

    /* ── Upload box ── */
    [data-testid="stFileUploader"] {
        background-color: #1A1A1A !important;
        border: 1px solid #2E2E2E !important;
        border-radius: 14px !important;
        padding: 16px !important;
    }

    /* ── Radio ── */
    [data-testid="stRadio"] label {
        color: #BBBBBB !important;
    }

    /* ── Text input ── */
    input[type="text"] {
        background-color: #1A1A1A !important;
        border: 1px solid #2E2E2E !important;
        border-radius: 10px !important;
        color: #FFFFFF !important;
    }

    /* ── Generate button ── */
    .stButton > button {
        background: linear-gradient(135deg, #2E2E2E 0%, #1A1A1A 100%);
        color: #FFFFFF !important;
        border: 1px solid #3A3A3A !important;
        border-radius: 10px;
        padding: 0.65rem 2rem;
        font-size: 1rem;
        font-weight: 600;
        width: 100%;
        letter-spacing: 0.3px;
        transition: all 0.25s ease;
    }
    .stButton > button:hover {
        background: linear-gradient(135deg, #3A3A3A 0%, #222222 100%);
        border-color: #555555 !important;
        color: #FFFFFF !important;
    }
    .stButton > button:disabled {
        opacity: 0.35;
        cursor: not-allowed;
    }

    /* ── Caption result card ── */
    .caption-card {
        background: linear-gradient(145deg, #1A1A1A, #141414);
        border: 1px solid #2E2E2E;
        border-radius: 16px;
        padding: 26px 30px;
        margin-top: 16px;
        box-shadow: 0 4px 30px rgba(0, 0, 0, 0.5);
    }
    .caption-text {
        font-size: 1.4rem;
        font-weight: 600;
        color: #FFFFFF;
        line-height: 1.6;
        letter-spacing: 0.2px;
    }
    .caption-meta {
        font-size: 0.82rem;
        color: #999999;
        margin-top: 10px;
        text-transform: uppercase;
        letter-spacing: 1px;
    }
    .word-chip {
        display: inline-block;
        background-color: #222222;
        color: #CCCCCC;
        border: 1px solid #333333;
        border-radius: 20px;
        padding: 4px 13px;
        margin: 4px 3px;
        font-size: 0.82rem;
        font-weight: 500;
        letter-spacing: 0.3px;
    }

    /* ── Image preview placeholder ── */
    .img-placeholder {
        background: linear-gradient(145deg, #1A1A1A, #111111);
        border: 1px dashed #2E2E2E;
        border-radius: 16px;
        height: 320px;
        display: flex;
        align-items: center;
        justify-content: center;
        color: #333333;
        font-size: 0.95rem;
    }

    /* ── Hint text ── */
    .hint-text {
        color: #666666;
        font-size: 0.9rem;
        margin-top: 18px;
        line-height: 1.5;
    }

    /* ── Divider ── */
    hr {
        border-color: #2A2A2A !important;
    }

    /* ── Spinner ── */
    .stSpinner > div {
        border-top-color: #FFFFFF !important;
    }

    /* ── Error / warning boxes ── */
    [data-testid="stAlert"] {
        background-color: #1A1A1A !important;
        border: 1px solid #3A3A3A !important;
        border-radius: 10px !important;
    }
</style>
""", unsafe_allow_html=True)

# ── Constants ─────────────────────────────────────────────────────────────────
MODEL_PATH = Path("./modelling results/results resnet101/best_model.pt")
MAX_LEN    = 20
DEVICE     = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ══════════════════════════════════════════════════════════════════════════════
# Model definitions
# ══════════════════════════════════════════════════════════════════════════════

class Vocabulary:
    PAD, START, END, UNK = "<PAD>", "<START>", "<END>", "<UNK>"

    def __init__(self, word2idx):
        self.word2idx = word2idx
        self.idx2word = {v: k for k, v in word2idx.items()}

    def __len__(self):
        return len(self.word2idx)


class FeatureProjection(nn.Module):
    def __init__(self, cnn_dim, hidden_dim):
        super().__init__()
        self.proj = nn.Linear(cnn_dim, hidden_dim)
        self.relu = nn.ReLU()

    def forward(self, x):
        return self.relu(self.proj(x))


class SoftAttention(nn.Module):
    def __init__(self, hidden_dim, attention_dim):
        super().__init__()
        self.W_a = nn.Linear(hidden_dim, attention_dim)
        self.W_h = nn.Linear(hidden_dim, attention_dim)
        self.W_e = nn.Linear(attention_dim, 1)

    def forward(self, A, h):
        score = self.W_e(
            torch.tanh(self.W_a(A) + self.W_h(h).unsqueeze(1))
        ).squeeze(2)
        alpha = F.softmax(score, dim=1)
        z     = (alpha.unsqueeze(2) * A).sum(dim=1)
        return z, alpha


class CaptionerAttention(nn.Module):
    def __init__(self, vocab_size, cnn_dim, embed_dim,
                 hidden_dim, attention_dim, dropout):
        super().__init__()
        self.embedding  = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.proj       = FeatureProjection(cnn_dim, hidden_dim)
        self.attention  = SoftAttention(hidden_dim, attention_dim)
        self.init_h     = nn.Linear(hidden_dim, hidden_dim)
        self.init_c     = nn.Linear(hidden_dim, hidden_dim)
        self.lstm       = nn.LSTMCell(embed_dim + hidden_dim, hidden_dim)
        self.dropout    = nn.Dropout(dropout)
        self.fc_out     = nn.Linear(hidden_dim, vocab_size)

    def init_hidden(self, A):
        mean_A = A.mean(dim=1)
        return (torch.tanh(self.init_h(mean_A)),
                torch.tanh(self.init_c(mean_A)))

    @torch.no_grad()
    def generate(self, features, vocab, max_len=20):
        A       = self.proj(features)
        h, c    = self.init_hidden(A)
        word_id = torch.tensor(
            [vocab.word2idx[vocab.START]]).to(features.device)
        result  = []
        for _ in range(max_len):
            embed    = self.embedding(word_id)
            z, alpha = self.attention(A, h)
            lstm_in  = torch.cat([embed, z], dim=1)
            h, c     = self.lstm(lstm_in, (h, c))
            logits   = self.fc_out(h)
            logits[:, vocab.word2idx[vocab.UNK]] = float("-inf")
            word_id  = logits.argmax(dim=1)
            word     = vocab.idx2word[word_id.item()]
            if word == vocab.END:
                break
            result.append(word)
        return " ".join(result)


class CaptionerNoAttention(nn.Module):
    def __init__(self, vocab_size, cnn_dim, embed_dim, hidden_dim, dropout):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.proj      = FeatureProjection(cnn_dim, hidden_dim)
        self.init_h    = nn.Linear(hidden_dim, hidden_dim)
        self.init_c    = nn.Linear(hidden_dim, hidden_dim)
        self.lstm      = nn.LSTMCell(embed_dim, hidden_dim)
        self.dropout   = nn.Dropout(dropout)
        self.fc_out    = nn.Linear(hidden_dim, vocab_size)

    def init_hidden(self, features):
        A      = self.proj(features)
        mean_A = A.mean(dim=1)
        return (torch.tanh(self.init_h(mean_A)),
                torch.tanh(self.init_c(mean_A)))

    @torch.no_grad()
    def generate(self, features, vocab, max_len=20):
        h, c    = self.init_hidden(features)
        word_id = torch.tensor(
            [vocab.word2idx[vocab.START]]).to(features.device)
        result  = []
        for _ in range(max_len):
            embed   = self.embedding(word_id)
            h, c    = self.lstm(embed, (h, c))
            logits  = self.fc_out(h)
            logits[:, vocab.word2idx[vocab.UNK]] = float("-inf")
            word_id = logits.argmax(dim=1)
            word    = vocab.idx2word[word_id.item()]
            if word == vocab.END:
                break
            result.append(word)
        return " ".join(result)


# ══════════════════════════════════════════════════════════════════════════════
# CNN feature extractor
# ══════════════════════════════════════════════════════════════════════════════

@st.cache_resource
def load_cnn_extractor(backbone_name):
    if backbone_name == "resnet101":
        base      = models.resnet101(
            weights=models.ResNet101_Weights.IMAGENET1K_V1)
        extractor = nn.Sequential(*list(base.children())[:-2])
    elif backbone_name == "vgg16":
        base      = models.vgg16(
            weights=models.VGG16_Weights.IMAGENET1K_V1)
        extractor = base.features
    else:
        raise ValueError(f"Unknown backbone: {backbone_name}")

    extractor.eval()
    for p in extractor.parameters():
        p.requires_grad = False
    return extractor.to(DEVICE)


IMAGE_TRANSFORM = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std =[0.229, 0.224, 0.225]),
])


def extract_features(pil_image, extractor):
    tensor = IMAGE_TRANSFORM(pil_image).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        feat = extractor(tensor)
    feat = feat.squeeze(0).permute(1, 2, 0).reshape(49, -1)
    return feat.unsqueeze(0)


# ══════════════════════════════════════════════════════════════════════════════
# Model loader
# ══════════════════════════════════════════════════════════════════════════════

@st.cache_resource
def load_model():
    ckpt       = torch.load(MODEL_PATH, map_location=DEVICE,
                            weights_only=False)
    word2idx   = ckpt["vocab"]
    vocab      = Vocabulary(word2idx)
    model_name = ckpt["model_name"]
    backbone   = ckpt["backbone"]
    cnn_dim    = {"vgg16": 512, "resnet101": 2048}[backbone]

    if "NoAttention" in model_name:
        model = CaptionerNoAttention(
            vocab_size = len(vocab),
            cnn_dim    = cnn_dim,
            embed_dim  = ckpt["embed_dim"],
            hidden_dim = ckpt["hidden_dim"],
            dropout    = ckpt["dropout"],
        )
    else:
        model = CaptionerAttention(
            vocab_size    = len(vocab),
            cnn_dim       = cnn_dim,
            embed_dim     = ckpt["embed_dim"],
            hidden_dim    = ckpt["hidden_dim"],
            attention_dim = ckpt["attention_dim"],
            dropout       = ckpt["dropout"],
        )

    model.load_state_dict(ckpt["model_state"])
    model = model.to(DEVICE).eval()
    return model, vocab, backbone, model_name


# ══════════════════════════════════════════════════════════════════════════════
# Image loading helpers
# ══════════════════════════════════════════════════════════════════════════════

def load_image_from_url(url):
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        return Image.open(io.BytesIO(response.content)).convert("RGB")
    except Exception as e:
        st.error(f"Could not load image from URL: {e}")
        return None


# ══════════════════════════════════════════════════════════════════════════════
# UI
# ══════════════════════════════════════════════════════════════════════════════

# ── Header ────────────────────────────────────────────────────────────────────
st.markdown(
    "<h1>IMAGE CAPTIONING</h1>"
    "<p style='color:#444444; margin-top:-12px; margin-bottom:28px; "
    "font-size:0.9rem; letter-spacing:2px; text-transform:uppercase;'>"
    "AI-Powered Image Description</p>",
    unsafe_allow_html=True
)

# ── Load model ────────────────────────────────────────────────────────────────
try:
    with st.spinner("Loading model..."):
        model, vocab, backbone, model_name = load_model()
        cnn_extractor = load_cnn_extractor(backbone)
    model_loaded = True
except Exception as e:
    st.error(f"Failed to load model: {e}")
    model_loaded = False

# ── Sidebar ───────────────────────────────────────────────────────────────────
st.sidebar.markdown(
    "<p style='font-size:0.7rem; letter-spacing:2px; "
    "text-transform:uppercase; color:#444444;'>MODEL INFO</p>",
    unsafe_allow_html=True
)
if model_loaded:
    st.sidebar.markdown(f"**Model:** `{model_name}`")
    st.sidebar.markdown(f"**Backbone:** `{backbone}`")
    st.sidebar.markdown(f"**Device:** `{str(DEVICE).upper()}`")
    st.sidebar.markdown(f"**Max length:** `{MAX_LEN}` words")
else:
    st.sidebar.warning("Model not loaded.")

# ── Input method ──────────────────────────────────────────────────────────────
input_method = st.radio(
    "Input method",
    ["Upload an image", "Paste an image URL"],
    horizontal=True,
    label_visibility="collapsed"
)

pil_image = None

if input_method == "Upload an image":
    uploaded = st.file_uploader(
        "Choose an image",
        type=["jpg", "jpeg", "png", "webp"],
        label_visibility="collapsed"
    )
    if uploaded is not None:
        pil_image = Image.open(uploaded).convert("RGB")
else:
    url = st.text_input(
        "Image URL",
        placeholder="https://example.com/image.jpg",
        label_visibility="collapsed"
    )
    if url.strip():
        pil_image = load_image_from_url(url.strip())

st.markdown("<div style='height:12px'></div>", unsafe_allow_html=True)

# ── Layout ────────────────────────────────────────────────────────────────────
col1, col2 = st.columns([1, 1], gap="large")

with col1:
    if pil_image is not None:
        st.image(pil_image, use_container_width=True)
    else:
        st.markdown(
            "<div class='img-placeholder'>"
            "Image preview will appear here"
            "</div>",
            unsafe_allow_html=True
        )

with col2:
    generate_clicked = st.button(
        "GENERATE CAPTION",
        disabled=(pil_image is None or not model_loaded)
    )

    if generate_clicked and pil_image is not None and model_loaded:
        with st.spinner("Generating..."):
            features = extract_features(pil_image, cnn_extractor)
            caption  = model.generate(features, vocab, max_len=MAX_LEN)

        words      = caption.split()
        chips_html = "".join(
            [f"<span class='word-chip'>{w}</span>" for w in words]
        )

        st.markdown(
            f"<div class='caption-card'>"
            f"<div class='caption-text'>{caption.capitalize()}.</div>"
            f"<div class='caption-meta'>{len(words)} words</div>"
            f"</div>",
            unsafe_allow_html=True
        )

    elif not generate_clicked and pil_image is not None:
        st.markdown(
            "<div class='hint-text'>"
            "Click <b style='color:#888'>GENERATE CAPTION</b> "
            "to describe your image."
            "</div>",
            unsafe_allow_html=True
        )
    else:
        st.markdown(
            "<div class='hint-text'>"
            "Provide an image to enable caption generation."
            "</div>",
            unsafe_allow_html=True
        )
