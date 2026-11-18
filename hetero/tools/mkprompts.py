#!/usr/bin/env python3
"""Token prompts for engine_drive.py:  mkprompts.py MODEL_GGUF OUT_JSON

short, short2: one-turn chats.  long: about 4,900 tokens of this repository's docs to summarise (as in report 0068).
A1 / B: the first turns of two conversations.  A2live: A1, its answer as generated, and a second user turn (resumes from
the live session).  A2ck: A1, a different assistant reply, and the same second turn (resumes from A1's turn checkpoint).
"""
import json, os, sys

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(REPO, "tools"))
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
out = {
    "short": chat("Explain in three sentences why the sky is blue."),
    "short2": chat("Write a haiku about a lighthouse in winter."),
    "long": chat("Summarise the following document in five bullet points.\n\n" + body),
    "A1": chat("Explain in three sentences why the sky is blue."),
    "B": chat("List four uses of copper, one line each."),
    "A2live": [{"prompt": "A1"}, {"out": "A1"}, enc(TURN2)],
    "A2ck": [{"prompt": "A1"}, enc("The sky looks blue because air scatters short wavelengths most."), enc(TURN2)],
}
json.dump(out, open(sys.argv[2], "w"))
print({k: (len(v) if v and isinstance(v[0], int) else "parts") for k, v in out.items()})
