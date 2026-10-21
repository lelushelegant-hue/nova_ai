import json
from pathlib import Path
import torch
from model import NOVA, NOVAConfig

ROOT = Path(__file__).resolve().parent
CHECKPOINT, VOCAB = ROOT / "checkpoint.pt", ROOT / "vocab.json"

def main():
    if not CHECKPOINT.exists() or not VOCAB.exists():
        raise SystemExit("Model not trained yet. Run: python train.py --steps 1500")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    saved = torch.load(CHECKPOINT, map_location=device, weights_only=False)
    config = NOVAConfig(**saved["config"])
    model = NOVA(config).to(device)
    model.load_state_dict(saved["model"])
    model.eval()
    vocab = json.loads(VOCAB.read_text(encoding="utf-8"))
    stoi = vocab["stoi"]
    itos = {int(k): v for k, v in vocab["itos"].items()}
    print("\033[92m")
    print("╔══════════════════════════════════════╗")
    print("║            NOVA AI TERMINAL          ║")
    print("║   local Transformer | no AI API      ║")
    print("╚══════════════════════════════════════╝")
    print("Commands: /help, /quit")
    print("Note: this is a small text-generation model; it may produce nonsense.")
    print("\033[0m")
    while True:
        try:
            prompt = input("\033[92mYOU > \033[0m")
        except (EOFError, KeyboardInterrupt):
            print("\nNOVA: session closed.")
            break
        if prompt.strip() == "/quit":
            break
        if prompt.strip() == "/help":
            print("Type text to continue generation. /quit exits.")
            continue
        known = [c for c in prompt if c in stoi]
        if not known:
            print("NOVA: This prompt contains characters not present in the training data.")
            continue
        ids = torch.tensor([[stoi[c] for c in known]], dtype=torch.long, device=device)
        with torch.no_grad():
            output = model.generate(ids, max_new_tokens=250, temperature=0.8, top_k=40)[0].tolist()
        generated = "".join(itos[i] for i in output)
        print("\033[96mNOVA > \033[0m" + generated[len(known):].strip())

if __name__ == "__main__":
    main()
