import google.generativeai as genai
import json
import os
import time
from dotenv import load_dotenv
from static import BASE_SCENARIOS, GLOBAL_PROMPT

# конфігурація
load_dotenv()
API_KEY = os.getenv("GEMINI_API_KEY")

if not API_KEY:
    raise ValueError("GEMINI_API_KEY ERROR: ключ не заданий в конфігурації середовища.")

genai.configure(api_key=API_KEY)
model = genai.GenerativeModel("gemini-2.5-flash")

batch_size = 10
max_retries = 3

scenarios = []
for i in range(2):
    for s_id, s_type, s_desc in BASE_SCENARIOS:
        scenarios.append((f"{s_id}_v{i + 1}", s_type, s_desc))


def generate_batch(batch_scenarios, batch_num):
    prompt = GLOBAL_PROMPT.format(batch_count=batch_scenarios)
    for i, (s_id, s_type, s_desc) in enumerate(batch_scenarios):
        prompt += f"\nДіалог {i + 1}:\n- id: {s_id}\n- Опис: {s_desc}\n"

    for attempt in range(max_retries):
        try:
            print(f"Пакет [{batch_num}] Запит до API (спроба {attempt + 1}/{max_retries})...")
            response = model.generate_content(
                prompt,
                generation_config=genai.GenerationConfig(
                    temperature=0.85,
                    response_mime_type="application/json"
                )
            )
            return json.loads(response.text.strip())

        except Exception as e:
            print(f"  [Пакет #{batch_num}] Помилка генерації або парсингу JSON: {e}")
            if attempt < max_retries- 1:
                time.sleep(5)
            else:
                raise Exception(f" Помилка генерації пакету #{batch_num} після {max_retries} спроб.")


def main():
    print(f"Початок генерації {len(scenarios)} діалогів (Довжина пакету: {batch_size})...")

    all_chats_dict = {}
    batches = [scenarios[i:i + batch_size] for i in range(0, len(scenarios), batch_size)]

    for idx, batch in enumerate(batches):
        print(f"\nОбробка пакету {idx + 1}/{len(batches)}...")
        try:
            chats = generate_batch(batch, idx + 1)

            # Форматуємо результат у словник { "dialogue_id": [messages] }
            # для повної сумісності з модулем аналітики (analyze.py)
            for chat in chats:
                dialogue_id = chat.get("id", "unknown_id")
                messages = chat.get("messages", [])
                all_chats_dict[dialogue_id] = messages

            print(f"  Пакет {idx + 1} успішно згенеровано.")

            if idx < len(batches) - 1:
                time.sleep(10)
        except Exception as e:
            print(f"Критична помилка в пакеті {idx + 1}: {e}")

    with open("dataset.json", "w", encoding="utf-8") as f:
        json.dump(all_chats_dict, f, ensure_ascii=False, indent=4)

    print(f"\nГенерація завершена. Збережено {len(all_chats_dict)} діалогів у dataset.json")


if __name__ == "__main__":
    main()