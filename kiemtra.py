import sqlite3
import pickle
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
from translator import translate_vi_to_en

# 1. Đọc file query
query_file = "test.txt"
print(f"1. Đang đọc file query: {query_file}")
with open(query_file, encoding="utf-8", errors="ignore") as f:
    input_text_raw = f.read()
has_vietnamese = any("\u00C0" <= c <= "\u1EF9" for c in input_text_raw)
if has_vietnamese:
    print("   => Phát hiện tiếng Việt, đang dịch sang tiếng Anh...")
    input_text = translate_vi_to_en(input_text_raw)
    print(f"   => Đã dịch xong: {input_text[:100]}...")
else:
    print("   => Phát hiện tiếng Anh, bỏ qua bước dịch.")
    input_text = input_text_raw

query_text = input_text  
print(f"   => Độ dài query: {len(query_text.split())} từ")

# 2. Load vectorizer và ma trận từ SQLite
print("\n2. Đang load CSDL từ SQLite...")
conn = sqlite3.connect("ir_database.db")
cur = conn.cursor()

cur.execute("SELECT id, filename FROM documents ORDER BY id")
doc_rows   = cur.fetchall()
doc_ids    = [r[0] for r in doc_rows]
file_names = [r[1] for r in doc_rows]

with open("vectorizer.pkl", "rb") as f:
    vectorizer = pickle.load(f)

feature_names = vectorizer.get_feature_names_out()

# Rebuild ma trận từ bảng tfidf_features
print("   => Đang rebuild ma trận TF-IDF từ DB...")
n_docs  = len(doc_ids)
n_terms = len(feature_names)
term_index = {term: idx for idx, term in enumerate(feature_names)}

tfidf_matrix = np.zeros((n_docs, n_terms))
doc_index    = {doc_id: idx for idx, doc_id in enumerate(doc_ids)}

cur.execute("SELECT doc_id, term, tfidf FROM tfidf_features")
for doc_id, term, tfidf in cur.fetchall():
    if doc_id in doc_index and term in term_index:
        tfidf_matrix[doc_index[doc_id]][term_index[term]] = tfidf

print(f"   => Ma trận: {tfidf_matrix.shape}")

# 3. Vector hóa query
print("\n3. Đang vector hóa file query...")
query_vector = vectorizer.transform([query_text])
query_array  = query_vector.toarray()[0]

# In kết quả trung gian — top 10 từ khóa nổi bật nhất
top_terms = sorted(
    zip(feature_names, query_array),
    key=lambda x: x[1], reverse=True
)[:10]

print("\n=== KẾT QUẢ TRUNG GIAN ===")
print("Top 10 từ khóa đặc trưng của file query:")
for term, score in top_terms:
    print(f"   '{term}': {score:.4f}")

# 4. Tính Cosine Similarity
print("\n4. Đang tính Cosine Similarity...")
similarities = cosine_similarity(query_vector, tfidf_matrix).flatten()

top3_idx = similarities.argsort()[::-1][:3]

print("\n=== KẾT QUẢ TÌM KIẾM ===")
print(f"Top 3 file tương đồng nhất với '{query_file}':\n")
results = []
for rank, idx in enumerate(top3_idx, 1):
    sim = similarities[idx] * 100
    fname = file_names[idx]
    print(f"   #{rank}: {fname}")
    print(f"         Độ tương đồng: {sim:.2f}%\n")
    results.append((query_file, fname, float(similarities[idx]), rank))

# 5. Lưu kết quả vào bảng search_results
cur.executemany(
    "INSERT INTO search_results (query_file, matched_file, similarity, rank) VALUES (?, ?, ?, ?)",
    results
)
conn.commit()
conn.close()
print("✅ Kết quả đã được lưu vào bảng search_results trong ir_database.db")