from transformers import MarianMTModel, MarianTokenizer

MODEL_NAME = "Helsinki-NLP/opus-mt-vi-en"

print("Đang tải model dịch (lần đầu ~300MB)...")
tokenizer = MarianTokenizer.from_pretrained(MODEL_NAME)
model = MarianMTModel.from_pretrained(MODEL_NAME)
print("Tải xong!")

def translate_vi_to_en(text: str, chunk_size: int = 400) -> str:
    """Dịch văn bản tiếng Việt sang tiếng Anh, tự chia nhỏ nếu text dài."""
    # Chia theo câu trước, rồi nhóm thành chunks ~400 ký tự
    sentences = text.replace("\n", " ").split(". ")
    chunks = []
    current = ""
    for s in sentences:
        if len(current) + len(s) < chunk_size:
            current += s + ". "
        else:
            if current:
                chunks.append(current.strip())
            current = s + ". "
    if current:
        chunks.append(current.strip())

    translated_parts = []
    for chunk in chunks:
        inputs = tokenizer(
            [chunk],
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512
        )
        outputs = model.generate(**inputs, num_beams=4)
        translated = tokenizer.decode(outputs[0], skip_special_tokens=True)
        translated_parts.append(translated)

    return " ".join(translated_parts)