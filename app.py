import json
from pathlib import Path
import streamlit as st
import torch
from model import NOVA, NOVAConfig

ROOT = Path(__file__).resolve().parent

st.set_page_config(
    page_title="NOVA AI",
    page_icon="¶",
    layout="centered",
    initial_sidebar_state="collapsed",
)

st.markdown("""
<style>
:root {
  --bg: #06131b;
  --panel: rgba(10, 34, 48, .88);
  --line: rgba(96, 192, 236, .32);
  --cyan: #63d8ff;
  --blue: #3c9dff;
  --text: #e8f7ff;
  --muted: #91b7c9;
}
.stApp {
  background:
    radial-gradient(ellipse at 50% 18%, rgba(17, 132, 190, .34), transparent 32%),
    radial-gradient(ellipse at 50% 80%, rgba(14, 76, 125, .32), transparent 45%),
    linear-gradient(160deg, #061017 0%, #092330 54%, #06131c 100%);
  color: var(--text);
}
[data-testid="stHeader"] { background: transparent; }
[data-testid="stToolbar"] { right: 1rem; }
.block-container { max-width: 760px; padding-top: 1.4rem; padding-bottom: 2rem; }
h1, h2, h3, p, label, span { color: var(--text); }
.nova-top {
  display:flex; align-items:center; justify-content:space-between;
  padding: 12px 4px 18px; color: #b9eaff; font-size: 13px;
  letter-spacing: .12em; text-transform: uppercase;
}
.nova-brand { font-weight: 800; font-size: 18px; letter-spacing: .16em; }
.nova-brand span { color: var(--cyan); }
.nova-status { color: #8fb8cb; font-size: 11px; letter-spacing: .08em; }
.hero { text-align:center; padding: 16px 0 20px; }
.orb {
  width: 116px; height: 116px; margin: 8px auto 22px; border-radius: 50%;
  background:
    radial-gradient(circle at 32% 25%, rgba(255,255,255,.92) 0%, rgba(156,239,255,.75) 4%, transparent 16%),
    radial-gradient(circle at 36% 30%, #4fd6ff 0%, #1688ec 35%, #1550b5 67%, #081c52 100%);
  box-shadow: 0 0 18px rgba(65,196,255,.8), 0 0 46px rgba(29,139,255,.55),
              inset -10px -12px 25px rgba(1,12,54,.7), inset 8px 8px 22px rgba(255,255,255,.3);
  border: 1px solid rgba(156,231,255,.7);
}
.hero h1 { font-size: 28px; margin: 0 0 5px; letter-spacing: -.03em; }
.hero p { color: var(--muted); margin: 0; font-size: 14px; }
.glass {
  border: 1px solid var(--line); border-radius: 22px; padding: 18px;
  background: linear-gradient(145deg, rgba(13,48,65,.88), rgba(5,23,37,.94));
  box-shadow: 0 10px 35px rgba(0,0,0,.24), inset 0 1px rgba(255,255,255,.04);
}
.stTextArea textarea, .stTextInput input {
  background: rgba(4, 20, 31, .88) !important; color: #e8f7ff !important;
  border: 1px solid rgba(92,196,242,.5) !important; border-radius: 14px !important;
}
.stTextArea textarea:focus, .stTextInput input:focus {
  border-color: #6edaff !important; box-shadow: 0 0 0 1px #6edaff !important;
}
.stButton button {
  border-radius: 999px !important; border: 1px solid rgba(125,220,255,.58) !important;
  background: linear-gradient(135deg, #56d9ff, #238cff) !important;
  color: #041724 !important; font-weight: 800 !important; min-height: 44px;
  box-shadow: 0 0 18px rgba(40,157,255,.22);
}
.stButton button:hover { filter: brightness(1.08); border-color: #b9f0ff !important; }
.stSlider [data-baseweb="slider"] div { background-color: #55caff; }
div[data-testid="stAlert"] { background: rgba(9,36,52,.85); border-radius: 14px; }
.output {
  border: 1px solid rgba(88,192,239,.38); border-radius: 18px;
  background: rgba(4,18,29,.82); padding: 16px 18px; color: #e8f7ff;
  white-space: pre-wrap; line-height: 1.6; overflow-wrap: anywhere;
}
.small-note { text-align:center; color:#769caf; font-size:11px; margin-top:18px; }
div[data-testid="stMarkdownContainer"] code { color: #c8f1ff; }
hr { border-color: rgba(107,194,231,.18); }
@media (max-width: 600px) {
  .block-container { padding-left: 1rem; padding-right: 1rem; }
  .hero h1 { font-size: 24px; }
  .orb { width: 96px; height: 96px; }
  .glass { padding: 14px; }
}
</style>
""", unsafe_allow_html=True)

st.markdown("""
<div class="nova-top">
  <div class="nova-brand">✦ NOVA <span>AI</span></div>
  <div class="nova-status">● LOCAL SYSTEM</div>
</div>
<div class="hero">
  <div class="orb"></div>
  <h1>How can I help you today?</h1>
  <p>Your creative space. Powered locally.</p>
</div>
""", unsafe_allow_html=True)

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

st.markdown('<div class="glass">', unsafe_allow_html=True)
if loaded is None:
    st.error("Model files not found. Train first with: python train.py --steps 1500")
else:
    model, device, vocab = loaded
    prompt = st.text_area("ASK NOVA", value="", placeholder="Ask me anything…", height=110)
    col1, col2 = st.columns(2)
    with col1:
        max_tokens = st.slider("Response length", 20, 400, 180, 20)
    with col2:
        temperature = st.slider("Creativity", 0.2, 1.5, 0.8, 0.1)
    if st.button("✦  Generate response", use_container_width=True):
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
            st.markdown('<div class="output">' + result[len(known):].strip().replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;").replace("\n", "<br>") + '</div>', unsafe_allow_html=True)
st.markdown('</div>', unsafe_allow_html=True)
st.markdown('<div class="small-note">NOVA AI · Experimental local Transformer · Outputs may be incoherent</div>', unsafe_allow_html=True)