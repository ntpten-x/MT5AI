from __future__ import annotations

import sqlite3

from modules.db import Database


def test_database_initialize_migrates_legacy_provider_columns(tmp_path):
    db_path = tmp_path / "legacy.db"
    connection = sqlite3.connect(db_path)
    try:
        connection.executescript(
            """
            CREATE TABLE trades (
                ticket INTEGER PRIMARY KEY,
                symbol TEXT NOT NULL,
                side TEXT NOT NULL,
                volume REAL NOT NULL,
                requested_price REAL,
                executed_price REAL,
                sl REAL,
                tp REAL,
                status TEXT NOT NULL,
                strategy TEXT,
                comment TEXT,
                opened_at TEXT NOT NULL,
                closed_at TEXT,
                profit REAL,
                request_payload TEXT,
                response_payload TEXT
            );

            CREATE TABLE equity_snapshots (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                ts_utc TEXT NOT NULL,
                balance REAL NOT NULL,
                equity REAL NOT NULL,
                profit REAL NOT NULL,
                margin_free REAL NOT NULL,
                margin_level REAL
            );
            """
        )
        connection.commit()
    finally:
        connection.close()

    database = Database(db_path)
    database.initialize()

    check = sqlite3.connect(db_path)
    try:
        trades_columns = {row[1] for row in check.execute("PRAGMA table_info(trades)")}
        equity_columns = {row[1] for row in check.execute("PRAGMA table_info(equity_snapshots)")}
    finally:
        check.close()

    assert "provider" in trades_columns
    assert "provider" in equity_columns
    assert "symbol" in equity_columns
