import sqlite3
import pickle
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from translator import translate_vi_to_en

# 1. Đọc file query
query_file = "test.txt"
print(f"1. Đang đọc file query: {query_file}")
with open(query_file, encoding="utf-8", errors="ignore") as f:
    input_text_raw = f.read()

has_vietnamese = any("\u00C0" <= c <= "\u1EF9" for c in input_text_raw)
if has_vietnamese:
    print("   => Phát hiện tiếng Việt, đang dịch...")
    input_text = translate_vi_to_en(input_text_raw)
else:
    print("   => Tiếng Anh, bỏ qua bước dịch.")
    input_text = input_text_raw

# 2. Load vectorizer + ma trận từ file (nhanh hơn rebuild từ DB)
print("\n2. Đang load vectorizer và ma trận...")
with open("vectorizer.pkl", "rb") as f:
    vectorizer = pickle.load(f)

with open("filenames.pkl", "rb") as f:
    file_names = pickle.load(f)

tfidf_matrix = np.load("tfidf_matrix.npy")  # ~40MB, load <1 giây
feature_names = vectorizer.get_feature_names_out()
print(f"   => Ma trận: {tfidf_matrix.shape}")

# 3. Vector hóa query
print("\n3. Đang vector hóa query...")
query_vector = vectorizer.transform([input_text])
query_array  = query_vector.toarray()[0]

top_terms = sorted(zip(feature_names, query_array), key=lambda x: x[1], reverse=True)[:10]
print("\n=== TOP 10 TỪ KHÓA ===")
for term, score in top_terms:
    print(f"   '{term}': {score:.4f}")

# 4. Tính Cosine Similarity
print("\n4. Đang tính Cosine Similarity...")
similarities = cosine_similarity(query_vector, tfidf_matrix).flatten()
top3_idx = similarities.argsort()[::-1][:3]

print(f"\n=== KẾT QUẢ TÌM KIẾM ===")
results = []
for rank, idx in enumerate(top3_idx, 1):
    sim = similarities[idx] * 100
    fname = file_names[idx]
    print(f"   #{rank}: {fname}  —  {sim:.2f}%")
    results.append((query_file, fname, float(similarities[idx]), rank))

# 5. Lưu kết quả vào DB
conn = sqlite3.connect("ir_database.db")
cur = conn.cursor()
cur.executemany(
    "INSERT INTO search_results (query_file, matched_file, similarity, rank) VALUES (?, ?, ?, ?)",
    results
)
conn.commit()
conn.close()
print("✅ Đã lưu kết quả vào search_results")