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

    -- Index giúp rebuild ma trận TF-IDF nhanh hơn (query theo doc_id)
    CREATE INDEX IF NOT EXISTS idx_tfidf_doc_id ON tfidf_features(doc_id);

    -- Index giúp tìm kiếm theo term nhanh hơn
    CREATE INDEX IF NOT EXISTS idx_tfidf_term ON tfidf_features(term);

    -- Index composite tối ưu khi lookup (doc_id, term) cùng lúc
    CREATE INDEX IF NOT EXISTS idx_tfidf_doc_term ON tfidf_features(doc_id, term);

    -- Index giúp truy vấn lịch sử tìm kiếm theo file query
    CREATE INDEX IF NOT EXISTS idx_search_query_file ON search_results(query_file);
""")

conn.commit()
conn.close()
print("✅ Tạo database ir_database.db thành công!")
