#!/usr/bin/env python3
"""
Extracts all one‐on‐one iMessage/SMS text messages from ~/Library/Messages/chat.db
and writes them out line‐by‐line as JSON, including the raw `date` field.

Example output line:
{
  "conversation_id": "+1-555-123-4567",
  "is_from_me": false,
  "text": "Hey, are you free tomorrow?",
  "date": 637512345678901234
}

Usage:
    python3 extract.py [--db ~/Library/Messages/chat.db] [--out all_messages.jsonl]
"""

import sqlite3
import json
import os
import argparse


def main():
    parser = argparse.ArgumentParser(
        description="Dump all one-on-one iMessage/SMS text messages as JSONL (including raw date)."
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
        default="messages/all_messages.jsonl",
        help="Output JSONL file (default: all_messages.jsonl)",
    )
    args = parser.parse_args()

    db_path = os.path.expanduser(args.db)
    if not os.path.exists(db_path):
        print(f"Error: cannot find database at {db_path}")
        print(
            "→ Make sure you have a local copy of chat.db (not synced away), and that Messages is closed."
        )
        return

    # Connect to chat.db
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()

    # Load `handle` table: handle_rowid → handle_string (phone or email)
    cursor.execute("SELECT ROWID AS handle_rowid, id AS handle_string FROM handle")
    handles = {row["handle_rowid"]: row["handle_string"] for row in cursor.fetchall()}

    # Query every non-null text message with a non-null handle_id (skip group chats)
    cursor.execute("""
        SELECT
            m.text,
            m.handle_id,
            m.is_from_me,
            m.date,
            -- For outgoing messages, get the handle_id from chat_handle_join
            (
                SELECT h.id
                FROM chat_message_join cmj
                JOIN chat_handle_join chj ON cmj.chat_id = chj.chat_id
                JOIN handle h ON chj.handle_id = h.ROWID
                WHERE cmj.message_id = m.ROWID
                LIMIT 1
            ) AS outgoing_handle_string
        FROM message m
        WHERE m.text IS NOT NULL
          AND (
                m.handle_id IS NOT NULL
                OR m.is_from_me = 1
              )
        ORDER BY COALESCE(m.handle_id, outgoing_handle_string), m.date
    """)
    all_messages = cursor.fetchall()

    # Write each message as a JSON line (including raw date)
    output_path = args.out
    with open(output_path, "w", encoding="utf-8") as fout:
        for msg in all_messages:
            if msg["is_from_me"]:
                # Use outgoing_handle_string for sent messages
                handle_str = msg["outgoing_handle_string"] or "UNKNOWN"
            else:
                # Use handle_id for received messages
                h_id = msg["handle_id"]
                handle_str = handles.get(h_id, "UNKNOWN")

            rec = {
                "conversation_id": handle_str,
                "is_from_me": bool(msg["is_from_me"]),
                "text": msg["text"],
                "date": msg["date"],
            }
            fout.write(json.dumps(rec, ensure_ascii=False) + "\n")

    conn.close()
    print(f"Done. Exported {len(all_messages)} messages to {output_path}")


if __name__ == "__main__":
    main()
