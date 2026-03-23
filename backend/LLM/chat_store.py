import sqlite3
from pathlib import Path

DB_PATH = Path("chat_history.db")


def init_chat_db():
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS chat_messages (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER NOT NULL,
            ticker TEXT NOT NULL,
            role TEXT NOT NULL,
            content TEXT NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
        """
    )
    conn.commit()
    conn.close()


def save_message(user_id, ticker, role, content):
    """ Save a single chat message into the database."""
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute(
        """
        INSERT INTO chat_messages (user_id, ticker, role, content)
        VALUES (?, ?, ?, ?)
        """,
        (user_id, ticker, role, content),
    )
    conn.commit()
    conn.close()


def load_chat_history(user_id, ticker, limit=50):
    """
    Load the most recent messages first, then return them in display order.
    """
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    cur = conn.cursor()

    cur.execute(
        """
        SELECT id, user_id, ticker, role, content, created_at
        FROM (
            SELECT *
            FROM chat_messages
            WHERE user_id = ? AND ticker = ?
            ORDER BY id DESC
            LIMIT ?
        )
        ORDER BY id ASC
        """,
        (user_id, ticker, limit),
    )

    rows = cur.fetchall()
    conn.close()
    return [dict(r) for r in rows]


def clear_chat_history(user_id, ticker):
    """
    Delete all chat history for a specific user and ticker.
    """
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute(
        """
        DELETE FROM chat_messages
        WHERE user_id = ? AND ticker = ?
        """,
        (user_id, ticker),
    )
    conn.commit()
    conn.close()


# Initialize on import
init_chat_db()