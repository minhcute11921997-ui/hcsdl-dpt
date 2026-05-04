from flask import Flask, render_template, request, jsonify
import sqlite3, pickle, numpy as np, os, threading
from sklearn.metrics.pairwise import cosine_similarity

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

app = Flask(__name__)
log_messages = []
search_lock = threading.Lock()

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

        # Buoc 1: Nhan file upload
        if "file" not in request.files or request.files["file"].filename == "":
            return jsonify({"error": "Vui long chon file .txt!"}), 400

        uploaded_file = request.files["file"]
        if not uploaded_file.filename.endswith(".txt"):
            return jsonify({"error": "Chi chap nhan file .txt!"}), 400

        input_text_raw = uploaded_file.read().decode("utf-8", errors="ignore").strip()
        word_count = len(input_text_raw.split())
        add_log(f"Doc file: {uploaded_file.filename} ({word_count} tu)")

        # Buoc 2: Phat hien ngon ngu
        has_vietnamese = any("\u00C0" <= c <= "\u1EF9" for c in input_text_raw)
        if has_vietnamese:
            add_log("Phat hien tieng Viet -> Dang dich sang tieng Anh...")
            from translator import translate_vi_to_en
            input_text = translate_vi_to_en(input_text_raw)
            add_log(f"Dich xong: \"{input_text[:80]}...\"")
        else:
            add_log("Phat hien tieng Anh -> Bo qua buoc dich")
            input_text = input_text_raw

        # Buoc 3: Load vectorizer
        vectorizer_path = os.path.join(BASE_DIR, "vectorizer.pkl")
        add_log("Dang load vectorizer.pkl...")
        if not os.path.exists(vectorizer_path):
            return jsonify({"error": "Chua co vectorizer.pkl, hay chay chuanhoa.py truoc!"}), 500
        with open(vectorizer_path, "rb") as f:
            vectorizer = pickle.load(f)
        feature_names = vectorizer.get_feature_names_out()
        add_log(f"Vectorizer loaded: {len(feature_names)} tu khoa")

        # Buoc 4: Load ma tran TF-IDF
        matrix_path   = os.path.join(BASE_DIR, "tfidf_matrix.npy")
        filenames_path = os.path.join(BASE_DIR, "filenames.pkl")

        if os.path.exists(matrix_path) and os.path.exists(filenames_path):
            add_log("Dang load ma tran tu cache (tfidf_matrix.npy)...")
            tfidf_matrix = np.load(matrix_path)
            with open(filenames_path, "rb") as f:
                file_names = pickle.load(f)
            add_log(f"Load cache xong: ma tran {tfidf_matrix.shape}")
        else:
            add_log("Khong co cache -> Dang rebuild ma tran tu SQLite...")
            db_path = os.path.join(BASE_DIR, "ir_database.db")
            conn = sqlite3.connect(db_path)
            cur = conn.cursor()
            cur.execute("SELECT id, filename FROM documents ORDER BY id")
            doc_rows   = cur.fetchall()
            doc_ids    = [r[0] for r in doc_rows]
            file_names = [r[1] for r in doc_rows]
            term_index = {term: idx for idx, term in enumerate(feature_names)}
            n_docs, n_terms = len(doc_ids), len(feature_names)
            tfidf_matrix = np.zeros((n_docs, n_terms))
            doc_index = {doc_id: idx for idx, doc_id in enumerate(doc_ids)}
            cur.execute("SELECT doc_id, term, tfidf FROM tfidf_features")
            for doc_id, term, tfidf in cur.fetchall():
                if doc_id in doc_index and term in term_index:
                    tfidf_matrix[doc_index[doc_id]][term_index[term]] = tfidf
            conn.close()
            add_log(f"Rebuild xong: ma tran {tfidf_matrix.shape}")

        # Buoc 5: Vector hoa query
        add_log("Dang vector hoa van ban dau vao...")
        query_vector = vectorizer.transform([input_text])
        query_array  = query_vector.toarray()[0]

        top_terms = sorted(zip(feature_names, query_array), key=lambda x: x[1], reverse=True)[:10]
        add_log("Top 10 tu khoa dac trung:")
        for term, score in top_terms:
            if score > 0:
                add_log(f"&nbsp;&nbsp;&nbsp;- <b>{term}</b>: {score:.4f}")

        # Buoc 6: Tinh Cosine Similarity
        add_log("Dang tinh Cosine Similarity voi toan bo corpus...")
        similarities = cosine_similarity(query_vector, tfidf_matrix).flatten()
        add_log(f"Tinh xong {len(similarities)} diem tuong dong")

        top3_idx = similarities.argsort()[::-1][:3]
        results = []
        for rank, idx in enumerate(top3_idx, 1):
            sim = float(similarities[idx]) * 100
            fname = file_names[idx]
            results.append({"rank": rank, "filename": fname, "similarity": round(sim, 2)})
            add_log(f"#{rank}: {fname} - {sim:.2f}%")

        # Buoc 7: Luu ket qua vao DB
        add_log("Dang luu ket qua vao search_results...")
        db_path = os.path.join(BASE_DIR, "ir_database.db")
        conn = sqlite3.connect(db_path)
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