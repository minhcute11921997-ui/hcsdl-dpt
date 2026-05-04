import os
import sqlite3

conn = sqlite3.connect("ir_database.db")
cur = conn.cursor()

source_folder = "dataset"
count = 0

for filename in os.listdir(source_folder):
    if filename.endswith(".txt"):
        filepath = os.path.join(source_folder, filename)
        try:
            content = open(filepath, encoding="utf-8", errors="ignore").read()
            cur.execute(
                "INSERT OR IGNORE INTO documents (filename, content) VALUES (?, ?)",
                (filename, content)
            )
            count += 1
        except Exception as e:
            print(f"Lỗi file {filename}: {e}")

conn.commit()
conn.close()
print(f"✅ Đã import {count} files vào SQLite!")
