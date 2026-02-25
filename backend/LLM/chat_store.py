import sqlite3
from datetime import datetime
from pathlib import Path

# Store DB inside the same folder as this file
DB_PATH = Path(__file__).resolve().parent / "chat_history.db"


def get_connection():
    """Create a new SQLite connection to the chat history database."""
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


def save_message(user_id: int, ticker: str, role: str, content: str) -> None:
    """
    Persist a single chat message.

    Parameters
    ----------
    user_id : int
        ID of the logged-in user.
    ticker : str
        Current stock ticker for this conversation.
    role : str
        'user' or 'assistant'.
    content : str
        Message text.
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


def load_chat_history(
    user_id: int,
    ticker: str | None = None,
    limit: int = 50,
) -> list[dict]:
    """
    Load recent chat messages for a user (optionally filtered by ticker).

    Parameters
    ----------
    user_id : int
        The user ID whose messages we want.
    ticker : str | None
        If provided, only messages for this ticker are returned.
        If None, messages for all tickers are returned.
    limit : int
        Maximum number of messages to return.

    Returns
    -------
    list[dict]
        A list of dicts like: [{"role": "user"/"assistant", "content": "..."}]
    """
    query = """
        SELECT role, content
        FROM chat_messages
        WHERE user_id = ?
    """
    params: list = [user_id]

    if ticker:
        query += " AND ticker = ?"
        params.append(ticker)

    query += " ORDER BY id ASC LIMIT ?"
    params.append(limit)

    with get_connection() as conn:
        rows = conn.execute(query, params).fetchall()

    return [{"role": r[0], "content": r[1]} for r in rows]


def clear_chat_history(user_id: int, ticker: str | None = None) -> None:
    """
    Delete chat messages for a user.

    If ticker is provided, only clear messages for that ticker.
    Otherwise, clear all messages for the user.
    """
    query = "DELETE FROM chat_messages WHERE user_id = ?"
    params: list = [user_id]

    if ticker:
        query += " AND ticker = ?"
        params.append(ticker)

    with get_connection() as conn:
        conn.execute(query, params)
        conn.commit()