#!/usr/bin/env python3
"""
prepare_conversations.py

Reads a JSONL file of individual iMessage/SMS messages (with fields: conversation_id, is_from_me, text, date)
and assembles them into full conversation threads. Each message is wrapped with <|im start> and <|im end> tokens.
The resulting conversations are then sharded into multiple JSONL files for easy loading by a dataloader.

Usage:
    python3 prep.py \
      --input all_messages.jsonl \
      --output_dir shards/ \
      --num_shards 10

Outputs:
    shards/shard_0.jsonl
    shards/shard_1.jsonl
    ...
    shards/shard_{num_shards-1}.jsonl

Each line in each shard is a JSON object:
{
    "conversation_id": "<phone_or_email>",
    "text": "<|im start> Message1 <|im end><|im start> Message2 <|im end>..."
}
"""

import os
import json
import argparse
from transformers import AutoTokenizer

SAME_USER_THRESHOLD = 300      # 5 minutes in seconds
CONVO_BREAK_THRESHOLD = 3600 * 24   # 24 hours in seconds
HISTORY_MAX_TOKENS = 2048      # adjust as needed
STRIDE = 512                   # overlap for long convos

tokenizer = AutoTokenizer.from_pretrained("unsloth/Meta-Llama-3.1-8B-bnb-4bit")

def count_tokens(text):
    """
    Count the number of tokens in a text using the Llama 3 tokenizer.
    """
    tokens = tokenizer.encode(text)
    return len(tokens)


def parse_messages(input_path):
    # Read and filter messages
    conversations = {}
    with open(input_path, "r", encoding="utf-8") as f:
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
        default="meta-llama/Llama-3-8b",
        help="Tokenizer model name to use (default: meta-llama/Llama-3-8b)",
    )
    parser.add_argument("--output", type=str, default="train.jsonl")
    args = parser.parse_args()
    
    # Update tokenizer if a different model is specified
    global tokenizer
    if args.model_name != "meta-llama/Llama-3-8b":
        tokenizer = AutoTokenizer.from_pretrained(args.model_name)

    # New pipeline: parse, assign roles, collapse, build, chunk, write
    conversations = parse_messages(args.input)
    total_chunks = 0
    with open(args.output, "w", encoding="utf-8") as f:
        for cid, messages in conversations.items():
            messages = assign_roles(messages)
            collapsed = collapse_bursts(messages)
            turns = build_turns(collapsed)
            chunks = chunk_conversations(turns)
            for chunk in chunks:
                f.write(json.dumps({"conversation_id": cid, "text": chunk}, ensure_ascii=False) + "\n")
                total_chunks += 1
    print(f"Wrote {total_chunks} conversation chunks to {args.output}")


if __name__ == "__main__":
    main()
