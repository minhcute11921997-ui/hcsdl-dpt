# debug.py
import pickle, numpy as np, sqlite3
from sklearn.metrics.pairwise import cosine_similarity

# 1. Load cache
with open("vectorizer.pkl", "rb") as f:
    vectorizer = pickle.load(f)
with open("filenames.pkl", "rb") as f:
    file_names = pickle.load(f)
tfidf_matrix = np.load("tfidf_matrix.npy")

# 2. Lấy đúng 1 document từ DB làm query (ví dụ doc id=1)
conn = sqlite3.connect("ir_database.db")
cur = conn.cursor()
cur.execute("SELECT id, filename, content FROM documents WHERE id = 1")
doc_id, filename, content = cur.fetchone()
conn.close()

print(f"Query document: {filename}")
print(f"filenames.pkl có {len(file_names)} files")
print(f"tfidf_matrix shape: {tfidf_matrix.shape}")
print(f"filenames.pkl[0]: {file_names[0]}")  # Phải khớp với doc id=1

# 3. Vector hóa và tính similarity
query_vec = vectorizer.transform([content])
print(f"\nQuery vector non-zero terms: {query_vec.nnz}")  # Nếu = 0 → vấn đề token

sims = cosine_similarity(query_vec, tfidf_matrix).flatten()
top5_idx = sims.argsort()[::-1][:5]

print("\nTop 5 kết quả:")
for idx in top5_idx:
    marker = " ← ĐÚNG" if file_names[idx] == filename else ""
    print(f"  {file_names[idx]}: {sims[idx]*100:.2f}%{marker}")

print(f"\nSimilarity với chính nó ({filename}): ", end="")
self_idx = file_names.index(filename)
print(f"{sims[self_idx]*100:.2f}%")