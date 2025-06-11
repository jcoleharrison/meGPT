"""
prep.py

Usage:
    python3 prep.py \
      --input all_messages.jsonl \
      --output train.jsonl \
      [--model_name unsloth/Meta-Llama-3.1-8B-bnb-4bit]

Arguments:
    --input       Path to input JSONL (e.g. all_messages.jsonl) [required]
    --output      Path to output JSONL (default: train.jsonl)
    --model_name  Tokenizer model name to use (default: unsloth/Meta-Llama-3.1-8B-bnb-4bit)

Each line in the output is a JSON object:
{
    "conversation_id": "<phone_or_email>",
    "text": "<|DT_SHORT|> <|ME|> ...",
    "num_tokens": <int>
}
"""

import json
import argparse
from pathlib import Path
from tqdm import tqdm
from transformers import AutoTokenizer

SAME_USER_THRESHOLD = 300      # 5 minutes in seconds
CONVO_BREAK_THRESHOLD = 3600 * 24   # 24 hours in seconds
HISTORY_MAX_TOKENS = 2048      # adjust as needed
STRIDE = 512                   # overlap for long convos
SPECIAL_TOKENS = ["<|DT_SHORT|>", "<|DT_LONG|>", "<|ME|>", "<|OTHER|>"]


def parse_messages(input_path):
    """Read JSONL and return dict of sorted message lists per conversation."""
    conversations = {}
    with input_path.open("r", encoding="utf-8") as f:
        for line in f:
            msg = json.loads(line)
            cid = msg["conversation_id"]
            if cid == "UNKNOWN":
                continue
            conversations.setdefault(cid, []).append({
                "sender": cid,
                "ts": float(msg["date"]) / 1e9,  # assuming nanoseconds
                "text": msg["text"],
                "is_from_me": msg["is_from_me"]
            })
    # Sort each conversation by timestamp
    for msgs in conversations.values():
        msgs.sort(key=lambda m: m["ts"])
    return conversations


def assign_roles(messages):
    """Tag each message with ME or OTHER role."""
    for m in messages:
        m["role"] = "<|ME|>" if m["is_from_me"] else "<|OTHER|>"
    return messages


def collapse_bursts(messages):
    collapsed = []
    prev_role = None
    prev_ts = None
    burst = []
    for m in messages:
        delta = m["ts"] - prev_ts if prev_ts is not None else 9999
        if (m["role"] != prev_role) or (delta > SAME_USER_THRESHOLD):
            if burst:
                # Collapse previous burst
                collapsed.append({
                    "role": prev_role,
                    "ts": burst[0]["ts"],
                    "delta": burst[0]["ts"] - (collapsed[-1]["ts"] if collapsed else 0),
                    "text": "\n".join(b["text"].strip() for b in burst)
                })
            burst = [m]
        else:
            burst.append(m)
        prev_role = m["role"]
        prev_ts = m["ts"]
    if burst:
        collapsed.append({
            "role": prev_role,
            "ts": burst[0]["ts"],
            "delta": burst[0]["ts"] - (collapsed[-1]["ts"] if collapsed else 0),
            "text": "\n".join(b["text"].strip() for b in burst)
        })
    return collapsed


def build_turns(collapsed):
    turns = []
    for c in collapsed:
        dt = "<|DT_SHORT|>" if c["delta"] < 300 else "<|DT_LONG|>"
        turn = f"{dt} {c['role']} {c['text']}"
        turns.append({"turn": turn, "ts": c["ts"]})
    return turns


def chunk_conversations(turns):
    chunks = []
    chunk = []
    prev_ts = None
    for t in turns:
        # Conversation break if delta > 1hr
        if prev_ts is not None and (t["ts"] - prev_ts > CONVO_BREAK_THRESHOLD):
            if chunk:
                chunks.append("\n".join(x["turn"] for x in chunk))
            chunk = []
        chunk.append(t)
        prev_ts = t["ts"]
    if chunk:
        chunks.append("\n".join(x["turn"] for x in chunk))
    return chunks


def load_tokenizer(model_name: str):
    """Load pretrained tokenizer and register special tokens."""
    tok = AutoTokenizer.from_pretrained(model_name)
    tok.add_special_tokens({"additional_special_tokens": SPECIAL_TOKENS})
    return tok


def tokenize_text(text, tokenizer):
    """Return list of token IDs for given text."""
    return tokenizer(text, add_special_tokens=False).input_ids


def split_long_conversations(input_ids):
    """Split list of token IDs into overlapping chunks of max HISTORY_MAX_TOKENS."""
    num_tokens = len(input_ids)
    if num_tokens <= HISTORY_MAX_TOKENS:
        return [input_ids]
    subs = []
    start = 0
    while start < num_tokens:
        end = min(start + HISTORY_MAX_TOKENS, num_tokens)
        subs.append(input_ids[start:end])
        if end == num_tokens:
            break
        start += STRIDE
    return subs


def main():
    parser = argparse.ArgumentParser(
        description="Assemble messages into conversation threads and chunk them by token count."
    )
    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="Path to input JSONL (e.g. all_messages.jsonl)",
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default="unsloth/Llama-3.2-3B-bnb-4bit",
        help="Tokenizer model name to use (default: unsloth/Llama-3.2-3B-bnb-4bit)",
    )
    parser.add_argument("--output", type=str, default="train.jsonl")
    args = parser.parse_args()
    
    args.input = Path(args.input)
    args.output = Path(args.output)
    print(f"Loading tokenizer: {args.model_name}")
    tokenizer = load_tokenizer(args.model_name)

    print(f"Parsing messages from {args.input} ...")
    conversations = parse_messages(args.input)

    total_chunks = 0
    with args.output.open("w", encoding="utf-8") as f:
        for cid, messages in tqdm(conversations.items(), desc="Conversations"):
            messages = assign_roles(messages)
            collapsed = collapse_bursts(messages)
            turns = build_turns(collapsed)
            chunks = chunk_conversations(turns)
            for chunk in chunks:
                # tokenize once
                ids = tokenize_text(chunk, tokenizer)
                # split into ID-subsequences
                for sub_ids in split_long_conversations(ids):
                    # Fix: Decode the tokens back to text for Trainer
                    text = tokenizer.decode(sub_ids, skip_special_tokens=False)
                    f.write(json.dumps({
                        "conversation_id": cid,
                        "text": text,
                        "num_tokens": len(sub_ids)  # Fix: Use num_tokens instead of input_ids
                    }, ensure_ascii=False) + "\n")
                    total_chunks += 1
    print(f"Wrote {total_chunks} conversation chunks to {args.output}")


if __name__ == "__main__":
    main()