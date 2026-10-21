# NOVA AI — small Transformer from scratch

NOVA is a learning project: a decoder-only Transformer written in PyTorch, a character-level tokenizer, a training script, a local terminal chat, and a Streamlit terminal UI. It does **not** call a hosted AI API.

## 1. Project files

- `model.py` — causal self-attention, feed-forward blocks, decoder-only Transformer
- `train.py` — trains on `data/train.txt` and saves `checkpoint.pt` + `vocab.json`
- `chat.py` — local terminal text generation
- `app.py` — web terminal UI
- `data/train.txt` — tiny starter corpus; replace/add text you have permission to use

## 2. Train in a free Google Colab session

1. Open https://colab.research.google.com/
2. Create a notebook and choose **Runtime → Change runtime type → T4 GPU** if available.
3. Upload this ZIP and extract it, or clone your own GitHub repository.
4. In a notebook cell, run:

```bash
!unzip -o nova_ai.zip -d /content/nova_ai
%cd /content/nova_ai
!pip install -r requirements.txt
!python train.py --steps 1500 --batch-size 32
```

If your ZIP has a different name, adjust the `unzip` command. If GPU is unavailable, training may be slow; reduce to `--steps 300 --batch-size 8`.

5. Download and keep both `checkpoint.pt` and `vocab.json`. They must stay together with `model.py`.

**Important:** Colab sessions are temporary. Save model files to your computer or Google Drive before the session ends. Free runtimes can disconnect and availability varies.

## 3. Run the terminal locally

Install Python 3.10+ and dependencies:

```bash
pip install -r requirements.txt
python train.py --steps 1500
python chat.py
```

Commands in the terminal: `/help`, `/quit`.

## 4. Run the web terminal

After training and keeping `checkpoint.pt` and `vocab.json` beside `app.py`:

```bash
pip install -r requirements.txt
streamlit run app.py
```

Streamlit opens a local web address. For a public deployment, create a Streamlit Community Cloud app from your GitHub repository and include the trained checkpoint and vocabulary files. Check the host's current storage, memory, file-size and sleep limits before deploying. The model will not work until the checkpoint and vocabulary are present.

## 5. Add training text

Edit `data/train.txt`. Use plain UTF-8 text, preferably a large, clean corpus with permission to use it. Keep the dataset consistent with the kinds of text you want NOVA to generate. Re-run training after changing the data.

The included corpus is only a tiny smoke-test example. It is not enough to make a useful assistant. Character-level generation learns spelling and local patterns, but it does not understand instructions like a large language model. Expect repetitive or incoherent outputs until you train on much more data.

## 6. What this model is and isn't

- It is a real, small decoder-only Transformer trained with next-token prediction.
- The architecture and training loop are implemented in this project.
- PyTorch is a third-party general-purpose machine-learning library, not an AI API.
- It is not comparable to ChatGPT. A useful conversational model requires much more data, compute, and careful evaluation.
- Do not put passwords, API keys, private messages, or sensitive information in the training corpus.
