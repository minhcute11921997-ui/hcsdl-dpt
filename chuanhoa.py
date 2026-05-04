import sqlite3
from sklearn.feature_extraction.text import TfidfVectorizer

# 1. Đọc dữ liệu từ SQLite
print("1. Đang đọc dữ liệu từ SQLite...")
conn = sqlite3.connect("ir_database.db")
cur = conn.cursor()

cur.execute("SELECT id, filename, content FROM documents ORDER BY id")
rows = cur.fetchall()

doc_ids    = [row[0] for row in rows]
file_names = [row[1] for row in rows]
documents  = [row[2] for row in rows]
print(f"   => Đã đọc {len(documents)} documents.")

# 2. Tính TF-IDF
print("\n2. Đang tính toán TF-IDF...")
vectorizer = TfidfVectorizer(
    lowercase=True,
    stop_words='english',
    token_pattern=r'(?u)\b[a-zA-Z]+\b',
    max_features=5000  # Giới hạn top 5000 từ khóa để tránh DB quá lớn
)
tfidf_matrix = vectorizer.fit_transform(documents)
feature_names = vectorizer.get_feature_names_out()

print(f"   => Trích xuất {len(feature_names)} từ khóa.")
print(f"   => Kích thước ma trận: {tfidf_matrix.shape}")

# 3. Lưu đặc trưng TF-IDF vào SQLite
print("\n3. Đang lưu đặc trưng vào SQLite...")
cur.execute("DELETE FROM tfidf_features")  # Xóa dữ liệu cũ nếu có

batch = []
matrix_array = tfidf_matrix.toarray()

for i, doc_id in enumerate(doc_ids):
    for j, term in enumerate(feature_names):
        score = matrix_array[i][j]
        if score > 0:  # Chỉ lưu giá trị khác 0
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

# 4. Lưu vectorizer để dùng lại khi tìm kiếm
import pickle
with open("vectorizer.pkl", "wb") as f:
    pickle.dump(vectorizer, f)

conn.commit()
conn.close()
print("🎉 Hoàn thành! CSDL đặc trưng đã được lưu vào ir_database.db")