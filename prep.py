#!/usr/bin/env python3
"""
prepare_conversations.py

Reads a JSONL file of individual iMessage/SMS messages (with fields: conversation_id, is_from_me, text, date)
and assembles them into full conversation threads. Each message is wrapped with <|im start> and <|im end> tokens.
The resulting conversations are then sharded into multiple JSONL files for easy loading by a dataloader.

Usage:
    python3 prepare_conversations.py \
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
from collections import defaultdict
from math import ceil
from transformers import AutoTokenizer

# Initialize the Llama 3 tokenizer
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3-8b")

def count_tokens(text):
    """
    Count the number of tokens in a text using the Llama 3 tokenizer.
    """
    tokens = tokenizer.encode(text)
    return len(tokens)


def load_messages(input_path):
    """
    Load messages from a JSONL file. Each line should be a JSON object with keys:
      - conversation_id (string)
      - is_from_me (bool)           # not used in ordering, but kept if needed
      - text (string)
      - date (integer or string)    # raw date field, used for sorting
    Returns a dict mapping conversation_id -> list of message dicts.
    """
    conv_dict = defaultdict(list)
    with open(input_path, "r", encoding="utf-8") as fin:
        for line in fin:
            line = line.strip()
            if not line:
                continue
            msg = json.loads(line)
            # Expect keys: "conversation_id", "is_from_me", "text", "date"
            cid = msg["conversation_id"]
            conv_dict[cid].append(msg)
    return conv_dict


def build_conversation_text(messages):
    """
    Given a list of message dicts (each with "text" and "date"), sort by date (ascending)
    and wrap each text in <|im start> and <|im end> tokens. Return the concatenated string.
    """
    # Sort by raw date value; assumes date is comparable (int or ISO string)
    sorted_msgs = sorted(messages, key=lambda m: m["date"])
    pieces = []
    for msg in sorted_msgs:
        text = msg["text"].replace("\n", " ")  # replace newlines to keep one line
        # wrap with tokens
        pieces.append(f"<|im start> {text} <|im end>")
    # Join without extra spaces to form a continuous sequence
    return "".join(pieces)


def shard_conversations(conversation_texts, num_shards, output_dir):
    """
    Given a list of (conversation_id, text) tuples, shard them into num_shards based on token counts.
    Writes out JSONL files under output_dir: shard_0.jsonl, shard_1.jsonl, ...
    """
    # Calculate token counts for each conversation
    conversation_tokens = [(cid, text, count_tokens(text)) for cid, text in conversation_texts]
    
    # Sort conversations by token count (descending) for better distribution
    conversation_tokens.sort(key=lambda x: x[2], reverse=True)
    
    # Initialize shards with token counts
    shards = [[] for _ in range(num_shards)]
    shard_token_counts = [0] * num_shards
    
    # Distribute conversations to balance token counts
    for cid, text, token_count in conversation_tokens:
        # Find the shard with the minimum token count
        min_idx = shard_token_counts.index(min(shard_token_counts))
        shards[min_idx].append((cid, text))
        shard_token_counts[min_idx] += token_count
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Write each shard to a file
    for shard_idx, shard_items in enumerate(shards):
        shard_path = os.path.join(output_dir, f"shard_{shard_idx}.jsonl")
        with open(shard_path, "w", encoding="utf-8") as fout:
            for cid, text in shard_items:
                rec = {"conversation_id": cid, "text": text}
                fout.write(json.dumps(rec, ensure_ascii=False) + "\n")
        print(f"Wrote {len(shard_items)} conversations ({shard_token_counts[shard_idx]} tokens) to {shard_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Assemble messages into conversation threads and shard them by token count."
    )
    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="Path to input JSONL (e.g. all_messages.jsonl)",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Directory where shard files will be written",
    )
    parser.add_argument(
        "--num_shards",
        type=int,
        default=10,
        help="Number of shards to split into (default: 10)",
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default="meta-llama/Llama-3-8b",
        help="Tokenizer model name to use (default: meta-llama/Llama-3-8b)",
    )
    args = parser.parse_args()
    
    # Update tokenizer if a different model is specified
    global tokenizer
    if args.model_name != "meta-llama/Llama-3-8b":
        tokenizer = AutoTokenizer.from_pretrained(args.model_name)

    # 1) Load and group messages by conversation_id
    conv_dict = load_messages(args.input)

    # 2) Build full conversation text for each conversation_id
    conversation_texts = []
    for cid, msgs in conv_dict.items():
        if len(msgs) < 1:
            continue
        convo_text = build_conversation_text(msgs)
        conversation_texts.append((cid, convo_text))

    # 3) Shard and write out JSONL files using token-based distribution
    shard_conversations(conversation_texts, args.num_shards, args.output_dir)


if __name__ == "__main__":
    main()
