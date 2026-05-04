import os
import sqlite3
import pickle
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# 1. Doc du lieu tu SQLite
print("1. Dang doc du lieu tu SQLite...")
conn = sqlite3.connect(os.path.join(BASE_DIR, "ir_database.db"))
cur = conn.cursor()
cur.execute("SELECT id, filename, content FROM documents ORDER BY id")
rows = cur.fetchall()
doc_ids    = [row[0] for row in rows]
file_names = [row[1] for row in rows]
documents  = [row[2] for row in rows]
print(f"   => Da doc {len(documents)} documents.")

# 2. Tinh TF-IDF
print("\n2. Dang tinh toan TF-IDF...")
vectorizer = TfidfVectorizer(
    lowercase=True,
    stop_words='english',
    token_pattern=r'(?u)\b[a-zA-Z]+\b',
    max_features=5000
)
tfidf_matrix = vectorizer.fit_transform(documents)  # ✅ Sửa ở đây
feature_names = vectorizer.get_feature_names_out()
print(f"   => Trich xuat {len(feature_names)} tu khoa.")
print(f"   => Kich thuoc ma tran: {tfidf_matrix.shape}")

# 3. Luu dac trung TF-IDF vao SQLite
print("\n3. Dang luu dac trung vao SQLite...")
cur.execute("DELETE FROM tfidf_features")
batch = []
matrix_array = tfidf_matrix.toarray()
for i, doc_id in enumerate(doc_ids):
    for j, term in enumerate(feature_names):
        score = matrix_array[i][j]
        if score > 0:
            batch.append((doc_id, term, float(score)))
    if len(batch) >= 1000:
        cur.executemany(
            "INSERT INTO tfidf_features (doc_id, term, tfidf) VALUES (?, ?, ?)",
            batch
        )
        batch = []
if batch:
    cur.executemany(
        "INSERT INTO tfidf_features (doc_id, term, tfidf) VALUES (?, ?, ?)",
        batch
    )

# 4. Luu cache ra file
with open(os.path.join(BASE_DIR, "vectorizer.pkl"), "wb") as f:
    pickle.dump(vectorizer, f)

np.save(os.path.join(BASE_DIR, "tfidf_matrix.npy"), matrix_array)  # ✅ Sửa ở đây

with open(os.path.join(BASE_DIR, "filenames.pkl"), "wb") as f:
    pickle.dump(file_names, f)

conn.commit()
conn.close()
print(f"Cache luu tai: {BASE_DIR}")
print("Hoan thanh! CSDL dac trung da duoc luu vao ir_database.db")