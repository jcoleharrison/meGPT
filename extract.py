#!/usr/bin/env python3
"""
extract_imessages.py

Extracts all one‐on‐one iMessage/SMS text pairs from your Mac’s Messages SQLite DB
and writes them out as JSONL for supervised LoRA fine-tuning.

Output format (one JSON object per line):
{
  "conversation_id": "<phone_or_email>",
  "prompt": "<incoming_text>",
  "response": "<outgoing_text>",
  "timestamp": "<ISO8601_UTC>"
}

Usage:
    python3 extract.py [--db ~/Library/Messages/chat.db] [--out output.jsonl]
"""

import sqlite3
import json
import os
import argparse
from datetime import datetime, timezone


def mac_epoch_to_unix(mac_ts: int) -> float:
    """
    iMessage ‘date’ is stored as seconds since 2001-01-01 00:00:00 UTC (the “Mac epoch”).
    To convert to a Unix timestamp, add 978307200 seconds.
    """
    return mac_ts + 978307200


def main():
    parser = argparse.ArgumentParser(
        description="Dump all one-on-one iMessage/SMS text pairs into JSONL for LoRA FT."
    )
    parser.add_argument(
        "--db",
        type=str,
        default="~/Library/Messages/chat.db",
        help="Path to chat.db (default: ~/Library/Messages/chat.db)",
    )
    parser.add_argument(
        "--out",
        type=str,
        default="imessages_dataset.jsonl",
        help="Output JSONL file (default: imessages_dataset.jsonl)",
    )
    args = parser.parse_args()

    db_path = os.path.expanduser(args.db)
    if not os.path.exists(db_path):
        print(f"Error: cannot find database at {db_path}")
        print(
            "Make sure you’ve enabled Messages → Preferences → “Enable Messages in iCloud” OFF, or that you have a local copy of chat.db."
        )
        return

    # Connect to the SQLite database
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()

    # 1) Load handle table into a dict: handle_id → handle_string (phone number or email)
    cursor.execute("SELECT ROWID AS handle_rowid, id AS handle_string FROM handle")
    handles = {row["handle_rowid"]: row["handle_string"] for row in cursor.fetchall()}

    # 2) Query every message that has non‐NULL text, ordered by handle_id then date
    cursor.execute("""
        SELECT
            ROWID AS message_rowid,
            text,
            handle_id,
            date,
            is_from_me
        FROM message
        WHERE text IS NOT NULL
        ORDER BY handle_id, date
    """)
    all_messages = cursor.fetchall()

    # 3) Group messages by handle_id (one‐on‐one chats appear as a single handle_id).
    #    If handle_id is NULL, it’s usually a group chat or some special thread — we skip for now.
    conversations = {}
    for msg in all_messages:
        h_id = msg["handle_id"]
        if h_id is None:
            # skip group chats / non‐handle messages
            continue

        handle_str = handles.get(h_id, "UNKNOWN")
        conversations.setdefault(handle_str, []).append(msg)

    # 4) Walk each conversation’s message list in chronological order.
    #    Whenever we see an incoming (is_from_me=0) followed immediately by outgoing (is_from_me=1),
    #    we emit one JSON object: {"prompt": incoming_text, "response": outgoing_text, ...}
    output_path = args.out
    with open(output_path, "w", encoding="utf-8") as fout:
        for handle_str, msgs in conversations.items():
            for i in range(len(msgs) - 1):
                cur_msg = msgs[i]
                next_msg = msgs[i + 1]

                # Only emit if current is incoming AND next is outgoing
                if cur_msg["is_from_me"] == 0 and next_msg["is_from_me"] == 1:
                    # Convert Mac‐epoch → Unix → ISO8601 UTC for the incoming message’s date
                    mac_ts = cur_msg["date"]
                    unix_ts = mac_epoch_to_unix(mac_ts)
                    iso_ts = datetime.fromtimestamp(
                        unix_ts, tz=timezone.utc
                    ).isoformat()

                    rec = {
                        "conversation_id": handle_str,
                        "prompt": cur_msg["text"],
                        "response": next_msg["text"],
                        "timestamp": iso_ts,
                    }
                    fout.write(json.dumps(rec, ensure_ascii=False) + "\n")

    conn.close()
    print(f"Done. Extracted conversation pairs written to: {output_path}")


if __name__ == "__main__":
    main()
