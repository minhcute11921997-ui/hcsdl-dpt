# translator.py
MODEL_NAME = "Helsinki-NLP/opus-mt-vi-en"
_tokenizer = None
_model = None

def translate_vi_to_en(text: str, chunk_size: int = 3000) -> str:
    global _tokenizer, _model
    if _model is None:
        from transformers import MarianMTModel, MarianTokenizer  
        print("Dang tai model dich (lan dau ~300MB)...")
        _tokenizer = MarianTokenizer.from_pretrained(MODEL_NAME)
        _model = MarianMTModel.from_pretrained(MODEL_NAME)
        print("Tai xong!")

    sentences = text.replace("\n", " ").split(". ")
    chunks, current = [], ""
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
        inputs = _tokenizer(
            [chunk],
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512
        )
        outputs = _model.generate(**inputs, num_beams=4)
        translated_parts.append(_tokenizer.decode(outputs[0], skip_special_tokens=True))

    return " ".join(translated_parts)