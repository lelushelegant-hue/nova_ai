import json
from pathlib import Path
import streamlit as st
import torch
from model import NOVA, NOVAConfig

ROOT = Path(__file__).resolve().parent
st.set_page_config(page_title="NOVA AI Terminal", page_icon="🧠", layout="wide")
st.markdown("""
<style>
.stApp { background: #050907; color: #b8ffd0; }
h1,h2,h3,p,label { color: #a6ffbd !important; }
.stTextInput input, .stTextArea textarea { background:#07130c; color:#b8ffd0; border:1px solid #1e7940; }
.stButton button { background:#0b2917; color:#a6ffbd; border:1px solid #28a653; }
.terminal { background:#020604; border:1px solid #1e7940; padding:16px; border-radius:8px; font-family:monospace; }
</style>
""", unsafe_allow_html=True)
st.title("🧠 NOVA AI // TERMINAL")
st.caption("LOCAL TRANSFORMER • NO THIRD-PARTY AI API")
st.markdown('<div class="terminal">[SYSTEM] NOVA interface initialized<br>[MODE] Local text generation<br>[NOTICE] Prototype model; outputs may be incoherent.</div>', unsafe_allow_html=True)

@st.cache_resource
def load_model():
    ckpt, vocab_file = ROOT / "checkpoint.pt", ROOT / "vocab.json"
    if not ckpt.exists() or not vocab_file.exists():
        return None
    device = "cuda" if torch.cuda.is_available() else "cpu"
    saved = torch.load(ckpt, map_location=device, weights_only=False)
    config = NOVAConfig(**saved["config"])
    model = NOVA(config).to(device)
    model.load_state_dict(saved["model"])
    model.eval()
    vocab = json.loads(vocab_file.read_text(encoding="utf-8"))
    return model, device, vocab

loaded = load_model()
if loaded is None:
    st.error("Model files not found. Train first with: python train.py --steps 1500")
else:
    model, device, vocab = loaded
    prompt = st.text_area("INPUT >", value="", placeholder="Введите начало текста…", height=110)
    col1, col2 = st.columns(2)
    with col1:
        max_tokens = st.slider("MAX TOKENS", 20, 400, 180, 20)
    with col2:
        temperature = st.slider("TEMPERATURE", 0.2, 1.5, 0.8, 0.1)
    if st.button("▶ GENERATE"):
        stoi = vocab["stoi"]
        itos = {int(k): v for k, v in vocab["itos"].items()}
        known = [c for c in prompt if c in stoi]
        if not known:
            st.warning("Enter text using characters present in the training dataset.")
        else:
            ids = torch.tensor([[stoi[c] for c in known]], dtype=torch.long, device=device)
            with torch.no_grad():
                out = model.generate(ids, max_new_tokens=max_tokens, temperature=temperature, top_k=40)[0].tolist()
            result = "".join(itos[i] for i in out)
            st.markdown("### NOVA OUTPUT")
            st.code(result[len(known):].strip(), language="text")
st.markdown("---")
st.caption("NOVA AI is an experimental model trained by you. It does not call ChatGPT, Gemini, Claude, or any hosted AI API.")
