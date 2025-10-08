from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch
import re
from typing import List

MODEL_NAME = "IlyaGusev/rut5_base_sum_gazeta"

device = (
    "mps" if torch.backends.mps.is_available()
    else "cuda" if torch.cuda.is_available()
    else "cpu"
)

torch_dtype = torch.float16 if device in ("cuda", "mps") else torch.float32

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME, dtype=torch_dtype)
model.to(device)
model.eval()
model.generation_config.length_penalty = 0.8

# Опционально (CUDA + PyTorch 2.x):
# try:
#     if device == "cuda":
#         model = torch.compile(model)  # может дать небольшой буст
# except Exception:
#     pass

def _split_sentences(text: str, k: int = 3) -> str:
    sents = re.split(r'(?<=[.!?])\s+', text.strip())
    return " ".join(sents[:k]).strip()

def chunk_text(text: str, chunk_chars: int = 1500, overlap: int = 100) -> List[str]:
    """
    Режем по символам (приблизительно под 1024 токена).
    overlap даёт небольшой перехлёст, чтобы не терять смысл на границах.
    """
    if not text:
        return []
    chunks = []
    i = 0
    step = max(1, chunk_chars - max(0, overlap))
    while i < len(text):
        chunks.append(text[i:i + chunk_chars])
        i += step
    return chunks

@torch.inference_mode()
def _generate_batch(prompts: List[str], max_new_tokens: int = 120, num_beams: int = 1) -> List[str]:
    """
    Один батч генерации (без градиентов).
    """
    inputs = tokenizer(
        prompts,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=1024
    ).to(device)

    output_ids = model.generate(
        **inputs,
        do_sample=False,
        num_beams=num_beams,        # 1 = самый быстрый
        length_penalty=0.8,
        max_new_tokens=max_new_tokens,
        no_repeat_ngram_size=3,
        eos_token_id=tokenizer.eos_token_id,
        use_cache=True,
    )
    return [tokenizer.decode(ids, skip_special_tokens=True).strip() for ids in output_ids]

def summarize(text: str, max_new_tokens: int = 120, batch_size: int = 8, num_beams: int = 1) -> List[str]:
    """
    Совместимо с твоим API: возвращает список кусочков-резюме для одного текста.
    Внутри работает батчами по чанкам.
    """
    chunks = chunk_text(text)
    if not chunks:
        return []

    prompts = [f"summarize: {c}\nКратко, 2–3 предложения." for c in chunks]
    out: List[str] = []

    for i in range(0, len(prompts), batch_size):
        batch_prompts = prompts[i:i+batch_size]
        preds = _generate_batch(batch_prompts, max_new_tokens=max_new_tokens, num_beams=num_beams)
        # нормализуем каждый ответ до 2–3 предложений
        out.extend(_split_sentences(p, 3) for p in preds)

    return out

def summarize_many(texts: List[str], max_new_tokens: int = 120, batch_size: int = 8, num_beams: int = 1) -> List[str]:
    """
    Батч-суммаризация НА НЕСКОЛЬКО СТАТЕЙ сразу.
    Возвращает список из N строк (по одной суммаризации на статью),
    где каждая строка — склейка суммаризаций её чанков.
    """
    chunk_map: List[int] = []   # индекс статьи для каждого чанка
    prompts: List[str] = []
    for idx, t in enumerate(texts):
        chunks = chunk_text(t)
        if not chunks:
            # чтобы размер совпадал
            chunk_map.append(idx)
            prompts.append("summarize: \nКратко, 2–3 предложения.")
            continue
        for c in chunks:
            chunk_map.append(idx)
            prompts.append(f"summarize: {c}\nКратко, 2–3 предложения.")

    if not prompts:
        return [""] * len(texts)

    # Генерим батчами
    all_preds: List[str] = []
    for i in range(0, len(prompts), batch_size):
        batch_prompts = prompts[i:i+batch_size]
        preds = _generate_batch(batch_prompts, max_new_tokens=max_new_tokens, num_beams=num_beams)
        all_preds.extend(_split_sentences(p, 3) for p in preds)

    # Склеиваем по статьям
    out = [""] * len(texts)
    acc: List[List[str]] = [[] for _ in range(len(texts))]
    for pred, idx in zip(all_preds, chunk_map):
        acc[idx].append(pred)

    for i in range(len(texts)):
        out[i] = "\n".join([p for p in acc[i] if p]).strip()

    return out
