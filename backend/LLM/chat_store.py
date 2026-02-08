import sqlite3
from datetime import datetime
from pathlib import Path

# Store DB inside backend folder
DB_PATH = Path(__file__).resolve().parent / "chat_history.db"


def get_connection():
    return sqlite3.connect(DB_PATH)


def init_chat_db():
    """
    Create chat_messages table if it doesn't exist.
    """
    with get_connection() as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS chat_messages (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER NOT NULL,
                ticker TEXT,
                role TEXT CHECK(role IN ('user', 'assistant')) NOT NULL,
                content TEXT NOT NULL,
                created_at TEXT NOT NULL
            )
            """
        )
        conn.commit()


def save_message(user_id: int, ticker: str, role: str, content: str):
    """
    Persist a single chat message.
    """
    with get_connection() as conn:
        conn.execute(
            """
            INSERT INTO chat_messages (user_id, ticker, role, content, created_at)
            VALUES (?, ?, ?, ?, ?)
            """,
            (
                user_id,
                ticker,
                role,
                content,
                datetime.utcnow().isoformat(),
            ),
        )
        conn.commit()


def load_chat_history(user_id: int, ticker: str | None = None, limit: int = 50):
    """
    Load recent chat messages for a user (optionally filtered by ticker).
    Returns a list of dicts: [{role, content}, ...]
    """
    query = """
        SELECT role, content
        FROM chat_messages
        WHERE user_id = ?
    """
    params = [user_id]

    if ticker:
        query += " AND ticker = ?"
        params.append(ticker)

    query += " ORDER BY id ASC LIMIT ?"
    params.append(limit)

    with get_connection() as conn:
        rows = conn.execute(query, params).fetchall()

    return [{"role": r[0], "content": r[1]} for r in rows]

def clear_chat_history(user_id: int, ticker: str | None = None):
    """
    Delete chat messages for a user.
    If ticker is provided, only clear messages for that ticker.
    Otherwise, clear all messages for the user.
    """
    query = "DELETE FROM chat_messages WHERE user_id = ?"
    params = [user_id]

    if ticker:
        query += " AND ticker = ?"
        params.append(ticker)

    with get_connection() as conn:
        conn.execute(query, params)
        conn.commit()
