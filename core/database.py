import sqlite3
import os

# Database path (stored in the data folder or root)
DB_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "metrics.db")

_db_initialized = False

def init_db():
    """Initializes the SQLite database with the required tables if they don't exist."""
    global _db_initialized
    if _db_initialized:
        return
        
    with sqlite3.connect(DB_PATH) as conn:
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
    _db_initialized = True

def log_query(query, mode, score, confidence, risk):
    """Logs a query run to the database."""
    init_db()
    with sqlite3.connect(DB_PATH) as conn:
        cursor = conn.cursor()
        cursor.execute('''
            INSERT INTO queries (query, mode, hallucination_score, confidence, risk_level)
            VALUES (?, ?, ?, ?, ?)
        ''', (query, mode, score, confidence, risk))
        conn.commit()

def get_analytics():
    """Returns all query logs as a list of dictionaries for analytics."""
    init_db()
    with sqlite3.connect(DB_PATH) as conn:
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        cursor.execute('SELECT * FROM queries ORDER BY timestamp DESC')
        rows = cursor.fetchall()
        return [dict(row) for row in rows]
