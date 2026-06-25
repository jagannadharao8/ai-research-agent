import sqlite3
import os
import datetime

# Database path (stored in the data folder or root)
DB_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "metrics.db")

def init_db():
    """Initializes the SQLite database with the required tables if they don't exist."""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS queries (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
            query TEXT,
            mode TEXT,
            hallucination_score REAL,
            confidence REAL,
            risk_level TEXT
        )
    ''')
    conn.commit()
    conn.close()

def log_query(query, mode, score, confidence, risk):
    """Logs a query run to the database."""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute('''
        INSERT INTO queries (query, mode, hallucination_score, confidence, risk_level)
        VALUES (?, ?, ?, ?, ?)
    ''', (query, mode, score, confidence, risk))
    conn.commit()
    conn.close()

def get_analytics():
    """Returns all query logs as a list of dictionaries for analytics."""
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
    cursor.execute('SELECT * FROM queries ORDER BY timestamp DESC')
    rows = cursor.fetchall()
    conn.close()
    return [dict(row) for row in rows]

# Ensure DB is initialized when this module is imported
init_db()
