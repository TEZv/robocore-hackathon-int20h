import google.generativeai as genai
import json
import os
import time
from dotenv import load_dotenv

load_dotenv()
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

# Використовуємо 2.5-flash, бо робимо всього 2 запити
model = genai.GenerativeModel("gemini-2.5-flash")

BATCH_SIZE = 20


def format_dialogue(messages: list) -> str:
    """Перетворює список повідомлень в читабельний текст."""
    lines = []
    for msg in messages:
        role = "Клієнт" if msg["role"] == "client" else "Агент"
        lines.append(f"{role}: {msg['text']}")
    return "\n".join(lines)


def analyze_batch(batch: list, batch_num: int) -> list:
    prompt = """Ти — система аналізу якості служби підтримки.
Проаналізуй наступні діалоги між клієнтом та агентом.

Поверни результат ВИКЛЮЧНО у форматі валідного JSON-масиву об'єктів. Нічого крім JSON!
Структура масиву має бути такою:
[
  {
    "chat_id": "id_діалогу",
    "intent": "<payment_issue | tech_error | account_access | tariff_question | refund | other>",
    "satisfaction": "<satisfied | neutral | unsatisfied>",
    "quality_score": <число від 1 до 5>,
    "agent_mistakes": ["<список помилок або порожній масив []>"],
    "reasoning": "<одне речення>"
  }
]

Можливі помилки агента: ignored_question, incorrect_info, rude_tone, no_resolution, unnecessary_escalation
ВАЖЛИВО: Якщо клієнт формально дякує але проблема не вирішена — це unsatisfied.

ОСЬ ДІАЛОГИ ДЛЯ АНАЛІЗУ:
"""
    for chat in batch:
        prompt += f"\n\n--- ДІАЛОГ (ID: {chat['id']}) ---\n"
        prompt += format_dialogue(chat["messages"])

    max_retries = 3
    for attempt in range(max_retries):
        try:
            print(f"  Відправляю запит до Gemini (Пакет {batch_num}, спроба {attempt + 1})...")
            response = model.generate_content(
                prompt,
                generation_config=genai.GenerationConfig(
                    temperature=0.1,  # Низька температура, щоб аналітик був об'єктивним роботом
                    response_mime_type="application/json"  # Жорсткий формат JSON
                )
            )

            raw = response.text.strip()
            if raw.startswith("```"):
                raw = raw.split("```")[1]
                if raw.startswith("json"):
                    raw = raw[4:]

            parsed = json.loads(raw.strip())
            return parsed
        except Exception as e:
            print(f"  ⚠️ Помилка: {e}. Пробуємо ще раз...")
            time.sleep(5)

    raise Exception("Не вдалося проаналізувати пакет після 3 спроб.")


def main():
    print("📂 Читаю dataset.json...")
    with open("dataset.json", "r", encoding="utf-8") as f:
        dataset = json.load(f)

    print(f"🔍 Починаємо ПАКЕТНИЙ аналіз {len(dataset)} діалогів...\n")

    results = []
    # Розбиваємо на пакети по 20
    batches = [dataset[i:i + BATCH_SIZE] for i in range(0, len(dataset), BATCH_SIZE)]

    for idx, batch in enumerate(batches):
        print(
            f"📦 Обробка пакету {idx + 1}/{len(batches)} (Діалоги {idx * BATCH_SIZE + 1} - {idx * BATCH_SIZE + len(batch)})...")
        try:
            analyzed_batch = analyze_batch(batch, idx + 1)

            # Збираємо результати і друкуємо
            for analysis in analyzed_batch:
                # Знаходимо оригінальний тип чату для повноти
                original_chat = next((c for c in batch if c["id"] == analysis["chat_id"]), None)
                chat_type = original_chat["type"] if original_chat else "unknown"

                results.append({
                    "chat_id": analysis["chat_id"],
                    "chat_type": chat_type,
                    "analysis": analysis
                })
                print(
                    f"  ✅ {analysis['chat_id']} -> Intent: {analysis['intent']} | Задоволеність: {analysis['satisfaction']} | Оцінка: {analysis['quality_score']}/5")

            if idx < len(batches) - 1:
                print("  ⏳ Пауза 10 секунд перед наступним пакетом...")
                time.sleep(10)
        except Exception as e:
            print(f"❌ Помилка пакетного аналізу: {e}")

    with open("results.json", "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print(f"\n✅ Готово! Результати збережено → results.json")


if __name__ == "__main__":
    main()