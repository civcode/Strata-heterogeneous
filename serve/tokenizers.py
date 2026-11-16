"""Native pack tokenizer construction and end-of-reply policy."""
import json
from pathlib import Path
from serve.frontend import IM_END


def load_pack_tokenizer(tpath: Path):
    """Construct the ordinary Strata pack tokenizer from its exported files."""
    from tools import strata_tokenizer as ST
    vocab = json.loads((tpath / "vocab.json").read_text(encoding="utf-8"))
    tokens = [None] * len(vocab)
    for t, i in vocab.items():
        tokens[i] = t
    merges = (tpath / "merges.txt").read_text(encoding="utf-8").split("\n")
    types = json.loads((tpath / "token_type.json").read_text())
    tcfg = {}
    try:
        tcfg = json.loads((tpath / "tokenizer.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        pass
    tok = ST.Tokenizer(tokens, merges, types, tcfg.get("pre") or "qwen35", tcfg.get("special_ids") or None)
    return tok, tcfg


def end_token_ids(tokenizer) -> set[int]:
    """The ids that end a reply for a tokenizer whose family was not named (Service.set_family sets them for DeepSeek).

    Qwen's <|im_end|> and <|endoftext|> end a reply only where each is ONE token.  A tokenizer without them (DeepSeek-V4.1's
    splits "<|im_end|>" into '<', '|', 'im', '_end', '|', '>') would make every piece of them a stop id: the reply ended in front
    of the first "|", ">", "end", "of", "text" or "im" it wrote, so "Nerazim" came out as "Neraz" (#4576, and a table's first
    "|" in #4137).  Those ids are left out, and the end token the tokenizer names (special_ids "eos") ends the reply instead."""
    ids: set[int] = set()
    for marker in (IM_END, "<|endoftext|>"):
        got = tokenizer.encode(marker, parse_special=True)
        if len(got) == 1:
            ids.add(int(got[0]))
    if not ids:
        eos = (getattr(tokenizer, "special_ids", None) or {}).get("eos")
        if eos is not None:
            ids.add(int(eos))
    return ids


def verify_pack_tokenizer(pack, gguf):
    # Both implementations load the same native class. Verify the exact
    # BPE tables and token types before using the ordinary pack path.
    tokenizer, _ = load_pack_tokenizer(Path(pack) / "tokenizer")
    if (tokenizer.tokens != gguf.tokens or tokenizer.ranks != gguf.ranks or
            tokenizer.token_types != gguf.token_types or tokenizer.pre != gguf.pre or
            end_token_ids(tokenizer) != end_token_ids(gguf)):
        raise ValueError("native pack tokenizer differs from the primary GGUF")
    return tokenizer
