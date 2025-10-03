from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch
import re


MODEL_NAME = "IlyaGusev/rut5_base_sum_gazeta"  # ~ 2 GB

device = (
    "mps" if torch.backends.mps.is_available()
    else "cuda" if torch.cuda.is_available()
    else "cpu"
)

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME)
model.to(device)
model.eval()

def chunk_text(text: str):
    chunks = []
    if len(text) > 1000:
        for i in range(int(len(text)/1000) + 1, len(text), 1000):
            chunks.append(text[i:i+1000])
    else:
        chunks.append(text)
    return chunks

def summarize(text: str, max_new_tokens: int = 120) -> list[str]:
    """
    Краткое резюме текста на 2–3 предложения.
    Модель — ruT5.
    """
    chunks = chunk_text(text)
    summaries = []
    for chunk in chunks:
        prompt = f"summarize: {text}\nКратко, 2–3 предложения."
        inputs = tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=1024, # Срезаем слишком длинные входы
        ).to(device)

        with torch.no_grad():
            output_ids = model.generate(
                **inputs,
                do_sample=False,
                num_beams=4,
                length_penalty=0.8,
                max_new_tokens=max_new_tokens,
                no_repeat_ngram_size=3,
                eos_token_id=tokenizer.eos_token_id,
            )

        summary = tokenizer.decode(output_ids[0], skip_special_tokens=True).strip()

        # Нормализуем до 2–3 предложений на всякий случай
        sentences = re.split(r'(?<=[.!?])\s+', summary)
        summary_3 = " ".join(sentences[:3]).strip()

        summaries.append(summary_3)
    return summaries

# Input prompt here
article = """
Сорта Селекционерами выведены самые различные по форме, размерам и цвету сорта баклажанов. Наиболее распространены
мелкие и цветные формы на дальнем востоке. Их форма варьирует от цилиндрической и грушевидной до почти шарообразной.
Масса баклажанов тоже варьирует существенно — от 30 г до 2 кг. Цвет баклажанов может быть не только синим или
фиолетовым. Плоды сорта «Белое яйцо» своей формой и цветом действительно напоминают куриные яйца. Англичане называют
такие баклажаны «пасхальными яйцами». Плоды сорта «Золотое яйцо» овальные и жёлтые. Они отличаются скороспелостью,
массой и количеством плодов и, что немаловажно, — вкусовыми качествами. Тайцы предпочитают мелкоплодные
светло-фиолетовые баклажанчики, в СНГ и России традиционно ценятся тёмно-фиолетовые плоды среднего размера.
"""


if __name__ == "__main__":
    print(summarize(article))

