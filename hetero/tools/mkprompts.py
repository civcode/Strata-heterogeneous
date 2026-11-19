#!/usr/bin/env python3
"""Token prompts for engine_drive.py:  mkprompts.py MODEL_GGUF OUT_JSON [--fresh N]

short, short2: one-turn chats.  long: about 4,900 tokens of this repository's docs to summarise (as in report 0068).
long2: the first 4,600 tokens of `long` and another question (resumes from a checkpoint at 4,096 with
--prompt-cache-every 2048, a layer split's mid-prompt checkpoint).  A1 / B: the first turns of two conversations.  A2live: A1, its answer as generated, and a second user turn (resumes from
the live session).  code: the coding request of reports 0073/0074.  long16k: about 16,000 tokens of this
repository's docs/*.md to summarise; long32k: the same with about 32,000 tokens.  --fresh N: also long@i and long16k@i for i < N, each starting with its own
"[read i]" tag, so a read neither reuses the previous one's prefix nor a checkpoint (as tools/calibrate.py reads).  A2ck: A1, a different assistant reply, and the same second turn (resumes from A1's turn checkpoint).  A2ck5: the same with a longer answer asked for (50 tokens and more).
"""
import json, os, sys

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(REPO, "terminal", "tools"))
from strata_tokenizer import Tokenizer  # noqa: E402

tk = Tokenizer.from_gguf(sys.argv[1])


def enc(t):
    return tk.encode(t, parse_special=True)


def chat(u):
    return enc("<|im_start|>user\n" + u + "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n")


TURN2 = "<|im_end|>\n<|im_start|>user\nAnd why is it red at sunset? Answer in two sentences.<|im_end|>\n" \
        "<|im_start|>assistant\n<think>\n\n</think>\n\n"
doc = open(os.path.join(REPO, "docs/MULTI_GPU.md")).read() + open(os.path.join(REPO, "README.md")).read()
body = tk.decode(tk.encode(doc)[:4900])
docs = "".join(open(os.path.join(REPO, "docs", f)).read() for f in sorted(os.listdir(os.path.join(REPO, "docs")))
               if f.endswith(".md"))
docs_ids = tk.encode(docs)
body16 = tk.decode(docs_ids[:16000])
body32 = tk.decode(docs_ids[:32000])
CODE = ("Write a complete Python implementation of a thread-safe LRU cache, including type hints, documentation, and "
        "example usage. Explain the design and important edge cases.")
out = {
    "code": chat(CODE),
    "long16k": chat("Summarise the following document in five bullet points.\n\n" + body16),
    "long32k": chat("Summarise the following document in five bullet points.\n\n" + body32),
    "short": chat("Explain in three sentences why the sky is blue."),
    "short2": chat("Write a haiku about a lighthouse in winter."),
    "long": chat("Summarise the following document in five bullet points.\n\n" + body),
    "long2": chat("Summarise the following document in five bullet points.\n\n" + body)[:4600] +
             enc("\n\nWhich hardware does the text mention? One sentence.<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"),
    "A1": chat("Explain in three sentences why the sky is blue."),
    "B": chat("List four uses of copper, one line each."),
    "A2live": [{"prompt": "A1"}, {"out": "A1"}, enc(TURN2)],
    "A2ck": [{"prompt": "A1"}, enc("The sky looks blue because air scatters short wavelengths most."), enc(TURN2)],
    "A2ck5": [{"prompt": "A1"}, enc("The sky looks blue because air scatters short wavelengths most."),
              enc(TURN2.replace("Answer in two sentences.", "Answer in five detailed sentences."))],
}
if "--fresh" in sys.argv:
    for i in range(int(sys.argv[sys.argv.index("--fresh") + 1])):
        out["long@%d" % i] = chat("[read %d] Summarise the following document in five bullet points.\n\n" % i + body)
        out["long16k@%d" % i] = chat("[read %d] Summarise the following document in five bullet points.\n\n" % i + body16)
json.dump(out, open(sys.argv[2], "w"))
print({k: (len(v) if v and isinstance(v[0], int) else "parts") for k, v in out.items()})
