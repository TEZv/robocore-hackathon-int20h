import os
import json
import time
import google.generativeai as genai
from dotenv import load_dotenv

# Завантажуємо змінні оточення (API ключ)
load_dotenv()
API_KEY = os.getenv("GEMINI_API_KEY")

if not API_KEY:
    raise ValueError("❌ Помилка: Не знайдено GEMINI_API_KEY у файлі .env!")

# Налаштовуємо API
genai.configure(api_key=API_KEY)

# Використовуємо 2.5 Flash, бо він швидкий і підтримує JSON
MODEL_ID = "gemini-2.5-flash"

# Налаштування генерації (жорстко вимагаємо JSON)
generation_config = genai.GenerationConfig(
    temperature=0.1,  # Низька температура: менше креативу, більше сухої аналітики
    response_mime_type="application/json"
)

model = genai.GenerativeModel(
    model_name=MODEL_ID,
    generation_config=generation_config
)

# Промпт для ролі аналітика
SYSTEM_PROMPT = """
Ти — Senior Data Analyst у відділі контролю якості (QA) служби підтримки.
Твоє завдання: проаналізувати масив діалогів і повернути виключно валідний JSON.

Для кожного діалогу визнач:
1. "intent" (категорія проблеми): payment_issue, tech_error, account_access, tariff_question, refund, other.
2. "satisfaction" (задоволеність клієнта в кінці): satisfied, neutral, unsatisfied.
3. "score" (оцінка роботи агента від 1 до 5).
4. "agent_errors" (масив помилок агента, якщо є): rude_tone, ignored_question, slow_response, false_info, none.
5. "summary" (коротке пояснення оцінки 1-2 реченнями).

Формат виводу — масив об'єктів:
[
  {
    "dialogue_id": "ID_діалогу",
    "intent": "...",
    "satisfaction": "...",
    "score": 5,
    "agent_errors": ["none"],
    "summary": "..."
  }
]
"""


def analyze_batch_with_retry(batch_dialogues, retries=3):
    """Відправляє пакет діалогів на аналіз з механізмом повторних спроб"""
    prompt = SYSTEM_PROMPT + f"\n\nОсь масив діалогів для аналізу (у форматі JSON):\n{json.dumps(batch_dialogues, ensure_ascii=False, indent=2)}"

    for attempt in range(retries):
        try:
            print(f"  Відправляю запит до Gemini (спроба {attempt + 1})...")
            response = model.generate_content(prompt)

            # Парсимо відповідь, щоб переконатися, що це валідний JSON
            result_json = json.loads(response.text)
            return result_json

        except Exception as e:
            print(f"  ⚠️ Помилка на спробі {attempt + 1}: {e}")
            if attempt < retries - 1:
                print("  ⏳ Чекаю 5 секунд перед повторною спробою...")
                time.sleep(5)
            else:
                print("  ❌ Не вдалося проаналізувати пакет після всіх спроб.")
                return None


def main():
    input_file = "dataset.json"
    output_file = "results.json"

    # Для аналізу можна залишити BATCH_SIZE = 20,
    # бо модель генерує мало тексту у відповідь (лише оцінки)
    batch_size = 20

    print(f"📂 Читаю {input_file}...")
    try:
        with open(input_file, "r", encoding="utf-8") as f:
            dataset = json.load(f)
    except FileNotFoundError:
        print(f"❌ Помилка: Файл {input_file} не знайдено!")
        return

    # Перетворюємо dict на list для зручної пакетної обробки
    dialogues_list = [{"dialogue_id": k, "messages": v} for k, v in dataset.items()]
    total_dialogues = len(dialogues_list)

    print(f"🔍 Починаємо ПАКЕТНИЙ аналіз {total_dialogues} діалогів...\n")

    all_results = {}

    for i in range(0, total_dialogues, batch_size):
        batch = dialogues_list[i:i + batch_size]
        current_batch_num = (i // batch_size) + 1
        total_batches = (total_dialogues + batch_size - 1) // batch_size

        print(
            f"📦 Обробка пакету {current_batch_num}/{total_batches} (Діалоги {i + 1} - {min(i + batch_size, total_dialogues)})...")

        batch_result = analyze_batch_with_retry(batch)

        if batch_result:
            for item in batch_result:
                # Збираємо результати назад у словник по dialogue_id
                d_id = item.pop("dialogue_id", "unknown_id")
                all_results[d_id] = item

                # Красивий висновок у консоль
                icon = "✅" if item.get("score", 0) >= 4 else ("⚠️" if item.get("score", 0) == 3 else "❌")
                print(
                    f"  {icon} {d_id} -> Intent: {item.get('intent')} | Задоволеність: {item.get('satisfaction')} | Оцінка: {item.get('score')}/5")

        # Пауза між пакетами для обходу Rate Limits
        if i + batch_size < total_dialogues:
            print("  ⏳ Пауза 10 секунд перед наступним пакетом...")
            time.sleep(10)

    # Зберігаємо фінальний результат
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(all_results, f, ensure_ascii=False, indent=4)

    print(f"\n✅ Готово! Результати збережено → {output_file}")


if __name__ == "__main__":
    main()