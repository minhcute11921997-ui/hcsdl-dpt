import sqlite3

conn = sqlite3.connect("ir_database.db")
cur = conn.cursor()

cur.executescript("""
    CREATE TABLE IF NOT EXISTS documents (
        id       INTEGER PRIMARY KEY AUTOINCREMENT,
        filename TEXT NOT NULL UNIQUE,
        content  TEXT NOT NULL
    );

    CREATE TABLE IF NOT EXISTS tfidf_features (
        doc_id  INTEGER REFERENCES documents(id),
        term    TEXT NOT NULL,
        tfidf   REAL NOT NULL
    );

    CREATE TABLE IF NOT EXISTS search_results (
        id           INTEGER PRIMARY KEY AUTOINCREMENT,
        query_file   TEXT,
        matched_file TEXT,
        similarity   REAL,
        rank         INTEGER,
        searched_at  TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    );
""")

conn.commit()
conn.close()
print("✅ Tạo database ir_database.db thành công!")