# hcsdl-dpt# 📚 Hệ thống Lưu trữ và Tìm kiếm Văn bản

Hệ thống tìm kiếm văn bản sử dụng thuật toán **TF-IDF + Cosine Similarity**, lưu trữ dữ liệu bằng **SQLite**, hỗ trợ đầu vào tiếng Việt và tiếng Anh.

---

## 📂 Cấu trúc thư mục

```text
📦 Thu_Muc_Do_An
 ┣ 📂 dataset/              # Thư mục chứa 1000 file .txt dữ liệu gốc
 ┣ 📜 setup_db.py           # Bước 1: Tạo database và các bảng
 ┣ 📜 import_data.py        # Bước 2: Import 1000 file .txt vào SQLite
 ┣ 📜 chuanhoa.py           # Bước 3: Tính TF-IDF và lưu đặc trưng vào SQLite
 ┣ 📜 kiemtra.py            # Bước 4: Tìm kiếm văn bản bằng Cosine Similarity
 ┣ 📜 translator.py         # Hỗ trợ dịch tiếng Việt → tiếng Anh
 ┣ 📜 ir_database.db        # SQLite Database (tự sinh sau bước 1)
 ┣ 📜 vectorizer.pkl        # TF-IDF Vectorizer đã train (tự sinh sau bước 3)
 ┣ 📜 test.txt              # File văn bản đầu vào để tìm kiếm
 ┗ 📜 README.md
```

---

## ⚙️ Cài đặt

Yêu cầu **Python 3.x**, sau đó cài các thư viện:

```bash
pip install scikit-learn numpy transformers sentencepiece
```

Tải dataset tại:
👉 https://drive.google.com/drive/folders/1oZgVpBQY-hkwOR_wzwVZvtIc4H-XZ_rX?usp=drive_link

Giải nén và đặt toàn bộ file vào thư mục `dataset/`.

---

## 🚀 Cách chạy

> Bước 1 và 2 chỉ cần chạy **một lần duy nhất**.

```bash
# Bước 1: Tạo database
python setup_db.py

# Bước 2: Import dữ liệu vào SQLite
python import_data.py

# Bước 3: Xây dựng CSDL đặc trưng TF-IDF
python chuanhoa.py

# Bước 4: Tìm kiếm (chạy mỗi khi cần tìm kiếm)
# Đặt nội dung cần tìm vào test.txt (tối thiểu 1000 từ), sau đó:
python kiemtra.py
```
