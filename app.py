from flask import Flask, render_template, request, jsonify
import sqlite3, pickle, numpy as np, os, threading
from sklearn.metrics.pairwise import cosine_similarity

app = Flask(__name__)
log_messages = []
search_lock = threading.Lock()

# ✅ Cache toàn cục - chỉ load 1 lần khi start Flask
_vectorizer = None
_tfidf_matrix = None
_file_names = None

def load_cache():
    global _vectorizer, _tfidf_matrix, _file_names
    if _vectorizer is None:
        print("🔄 Đang load vectorizer.pkl...")
        with open("vectorizer.pkl", "rb") as f:
            _vectorizer = pickle.load(f)
        print("🔄 Đang load tfidf_matrix.npy (~40MB)...")
        _tfidf_matrix = np.load("tfidf_matrix.npy")
        with open("filenames.pkl", "rb") as f:
            _file_names = pickle.load(f)
        print("✅ Cache loaded xong!")

# Gọi ngay khi app khởi động
with app.app_context():
    if os.path.exists("vectorizer.pkl"):
        load_cache()

def add_log(msg):
    log_messages.append(msg)

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/search", methods=["POST"])
def search():
    global log_messages
    with search_lock:
        log_messages = []

        if "file" not in request.files or request.files["file"].filename == "":
            return jsonify({"error": "Vui long chon file .txt!"}), 400

        uploaded_file = request.files["file"]
        if not uploaded_file.filename.endswith(".txt"):
            return jsonify({"error": "Chi chap nhan file .txt!"}), 400

        input_text_raw = uploaded_file.read().decode("utf-8", errors="ignore").strip()
        word_count = len(input_text_raw.split())
        add_log(f"Doc file: {uploaded_file.filename} ({word_count} tu)")

        # Phát hiện ngôn ngữ
        has_vietnamese = any("\u00C0" <= c <= "\u1EF9" for c in input_text_raw)
        if has_vietnamese:
            add_log("Phat hien tieng Viet -> Dang dich sang tieng Anh...")
            from translator import translate_vi_to_en
            input_text = translate_vi_to_en(input_text_raw)
            add_log(f"Dich xong: \"{input_text[:80]}...\"")
        else:
            add_log("Phat hien tieng Anh -> Bo qua buoc dich")
            input_text = input_text_raw

        # ✅ Dùng cache thay vì load lại
        if _vectorizer is None:
            return jsonify({"error": "Chua co vectorizer.pkl, hay chay chuanhoa.py truoc!"}), 500

        vectorizer   = _vectorizer
        tfidf_matrix = _tfidf_matrix
        file_names   = _file_names
        feature_names = vectorizer.get_feature_names_out()
        add_log(f"Vectorizer (cache): {len(feature_names)} tu khoa | Ma tran: {tfidf_matrix.shape}")

        # Vector hóa query
        add_log("Dang vector hoa van ban dau vao...")
        query_vector = vectorizer.transform([input_text])
        query_array  = query_vector.toarray()[0]

        top_terms = sorted(zip(feature_names, query_array), key=lambda x: x[1], reverse=True)[:10]
        add_log("Top 10 tu khoa dac trung:")
        for term, score in top_terms:
            if score > 0:
                add_log(f"&nbsp;&nbsp;&nbsp;- <b>{term}</b>: {score:.4f}")

        # Cosine Similarity
        add_log("Dang tinh Cosine Similarity...")
        similarities = cosine_similarity(query_vector, tfidf_matrix).flatten()
        add_log(f"Tinh xong {len(similarities)} diem tuong dong")

        top3_idx = similarities.argsort()[::-1][:3]
        results = []
        for rank, idx in enumerate(top3_idx, 1):
            sim = float(similarities[idx]) * 100
            fname = file_names[idx]
            results.append({"rank": rank, "filename": fname, "similarity": round(sim, 2)})
            add_log(f"#{rank}: {fname} - {sim:.2f}%")

        # Lưu kết quả
        add_log("Dang luu ket qua vao search_results...")
        conn = sqlite3.connect("ir_database.db")
        cur = conn.cursor()
        cur.executemany(
            "INSERT INTO search_results (query_file, matched_file, similarity, rank) VALUES (?, ?, ?, ?)",
            [(uploaded_file.filename, r["filename"], r["similarity"] / 100, r["rank"]) for r in results]
        )
        conn.commit()
        conn.close()
        add_log("Hoan tat!")

        return jsonify({"logs": log_messages, "results": results})

if __name__ == "__main__":
    app.run(debug=True)