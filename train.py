import argparse, json, random
from pathlib import Path
import torch
from model import NOVA, NOVAConfig

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data" / "train.txt"
OUT = ROOT / "checkpoint.pt"
VOCAB = ROOT / "vocab.json"

def main():
    p = argparse.ArgumentParser(description="Train the NOVA AI character-level Transformer.")
    p.add_argument("--steps", type=int, default=1500)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--block-size", type=int, default=128)
    p.add_argument("--learning-rate", type=float, default=3e-4)
    p.add_argument("--eval-interval", type=int, default=150)
    args = p.parse_args()

    if not DATA.exists():
        raise SystemExit(f"Dataset not found: {DATA}")
    text = DATA.read_text(encoding="utf-8")
    if len(text) < 100:
        raise SystemExit("Add at least a few paragraphs to data/train.txt before training.")

    # Character tokenizer: intentionally simple and fully local.
    chars = sorted(set(text))
    stoi = {ch: i for i, ch in enumerate(chars)}
    itos = {str(i): ch for ch, i in stoi.items()}
    VOCAB.write_text(json.dumps({"stoi": stoi, "itos": itos}, ensure_ascii=False, indent=2), encoding="utf-8")
    data = torch.tensor([stoi[c] for c in text], dtype=torch.long)
    if len(data) < args.block_size + 2:
        raise SystemExit("Dataset is shorter than block_size. Add more text or reduce --block-size.")
    split = int(0.9 * len(data))
    train_data, val_data = data[:split], data[split:]
    if len(val_data) < args.block_size + 2:
        val_data = train_data

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"NOVA AI | device={device} | characters={len(text):,} | vocab={len(chars)}")
    print("Training a small character-level Transformer. This is a learning prototype, not a ChatGPT-scale model.")
    config = NOVAConfig(vocab_size=len(chars), block_size=args.block_size)
    model = NOVA(config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)

    def batch(source):
        max_start = len(source) - args.block_size - 1
        starts = torch.randint(0, max_start + 1, (args.batch_size,))
        x = torch.stack([source[i:i+args.block_size] for i in starts]).to(device)
        y = torch.stack([source[i+1:i+args.block_size+1] for i in starts]).to(device)
        return x, y

    for step in range(1, args.steps + 1):
        model.train()
        xb, yb = batch(train_data)
        _, loss = model(xb, yb)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        if step == 1 or step % args.eval_interval == 0 or step == args.steps:
            model.eval()
            with torch.no_grad():
                vx, vy = batch(val_data)
                _, val_loss = model(vx, vy)
            print(f"step {step:5d}/{args.steps} | train_loss={loss.item():.3f} | val_loss={val_loss.item():.3f}")
            torch.save({"model": model.state_dict(), "config": vars(config)}, OUT)
    print(f"\nSaved model: {OUT}")
    print(f"Saved tokenizer: {VOCAB}")
    print("Next: python chat.py")

if __name__ == "__main__":
    main()
