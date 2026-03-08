"""
Database Module
SQLite storage for chat history and session management.
"""

import sqlite3
import os
from typing import List, Tuple, Optional
from datetime import datetime


class ChatDatabase:
    """SQLite-backed chat history storage."""

    def __init__(self, db_path: str = "data/chat_history.db"):
        os.makedirs(os.path.dirname(db_path), exist_ok=True)
        self.db_path = db_path
        self._init_db()

    # ------------------------------------------------------------------
    def _init_db(self):
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS chat_history (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id TEXT NOT NULL,
                    role TEXT NOT NULL,
                    message TEXT NOT NULL,
                    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
                )
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS sessions (
                    session_id TEXT PRIMARY KEY,
                    file_name TEXT,
                    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                    last_active DATETIME DEFAULT CURRENT_TIMESTAMP
                )
                """
            )
            conn.commit()

    # ------------------------------------------------------------------
    def save_message(self, session_id: str, role: str, message: str):
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                "INSERT INTO chat_history (session_id, role, message) VALUES (?, ?, ?)",
                (session_id, role, message),
            )
            conn.execute(
                """
                INSERT INTO sessions (session_id, last_active)
                VALUES (?, CURRENT_TIMESTAMP)
                ON CONFLICT(session_id)
                DO UPDATE SET last_active = CURRENT_TIMESTAMP
                """,
                (session_id,),
            )
            conn.commit()

    # ------------------------------------------------------------------
    def get_history(self, session_id: str) -> List[Tuple[str, str]]:
        """Returns list of (role, message) tuples."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute(
                "SELECT role, message FROM chat_history "
                "WHERE session_id = ? ORDER BY timestamp",
                (session_id,),
            )
            return cursor.fetchall()

    # ------------------------------------------------------------------
    def clear_session(self, session_id: str):
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                "DELETE FROM chat_history WHERE session_id = ?", (session_id,)
            )
            conn.commit()

    # ------------------------------------------------------------------
    def get_all_sessions(self) -> List[Tuple[str, str]]:
        """Returns list of (session_id, last_active)."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute(
                "SELECT session_id, last_active FROM sessions ORDER BY last_active DESC"
            )
            return cursor.fetchall()

    # ------------------------------------------------------------------
    def delete_all(self):
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("DELETE FROM chat_history")
            conn.execute("DELETE FROM sessions")
            conn.commit()