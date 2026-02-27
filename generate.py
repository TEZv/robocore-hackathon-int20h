import google.generativeai as genai
import json
import os
import time
from dotenv import load_dotenv

# Configuration
load_dotenv()
API_KEY = os.getenv("GEMINI_API_KEY")

if not API_KEY:
    raise ValueError("GEMINI_API_KEY is not set in the environment variables.")

genai.configure(api_key=API_KEY)
model = genai.GenerativeModel("gemini-2.5-flash")

BATCH_SIZE = 10
MAX_RETRIES = 3

BASE_SCENARIOS = [
    ("payment_success", "normal",
     "КЛІЄНТ: Стривожений, пише короткими реченнями, робить дрібні описки ('не проходить оплата!!', 'шо робить?'). АГЕНТ: Професійний, швидко скидає нове посилання."),
    ("tech_success", "normal",
     "КЛІЄНТ: Технічно неграмотний (бумер), пояснює 'на пальцях' ('крутиться кружечок і все', 'ваша програма зламалась'). АГЕНТ: Терплячий, дає покрокову інструкцію."),
    ("access_success", "normal",
     "КЛІЄНТ: Дуже поспішає, пише без розділових знаків і з малої літери ('швидше дайте доступ горить проект'). АГЕНТ: Блискавично скидає код відновлення."),
    ("tariff_success", "normal",
     "КЛІЄНТ: Дуже економний, прискіпливо питає за кожну копійку, боїться прихованих платежів. АГЕНТ: Детально все пояснює."),
    ("refund_success", "normal",
     "КЛІЄНТ: Ввічливий, але дуже засмучений і розчарований якістю товару. АГЕНТ: Співчуває, швидко оформлює повернення."),
    ("other_success", "normal",
     "КЛІЄНТ: Пише з купою смайликів, питає графік роботи на свята. АГЕНТ: Радо надає інформацію."),
    ("tech_success_pro", "normal",
     "КЛІЄНТ: Програміст, спілкується суто технічними термінами, скидає логи помилки (Error 500, CORS policy). АГЕНТ: Спілкується на одному рівні."),
    ("payment_conflict", "conflict",
     "КЛІЄНТ: Дуже злий, пише КАПСОМ, використовує знаки оклику, погрожує судом за подвійне списання. АГЕНТ: Сухо ігнорує питання 'хто поверне комісію?' (помилка: ignored_question)."),
    ("tech_problem", "problem",
     "КЛІЄНТ: Роздратований, додаток вилітає третій день поспіль. АГЕНТ: Відповідає зверхньо, прямо звинувачує дешевий телефон клієнта (помилка: rude_tone)."),
    ("tariff_problem", "problem",
     "КЛІЄНТ: Питає про вартість річної підписки. АГЕНТ: Впевнено називає неправильну, стару ціну (помилка: incorrect_info). Клієнт вірить."),
    ("access_problem", "problem",
     "КЛІЄНТ: Не приходить SMS. АГЕНТ: Одразу каже 'пишіть листа директору', замість перевірки номера (помилка: unnecessary_escalation)."),
    ("refund_conflict", "conflict",
     "КЛІЄНТ: Вимагає повернення (пройшло 10 днів). АГЕНТ: Грубо відмовляє без жодних пояснень причини (помилка: rude_tone)."),
    ("payment_problem_2", "problem",
     "КЛІЄНТ: Гроші списало, а послуги нема, панікує. АГЕНТ: Відписує шаблонно 'чекайте' і миттєво закриває чат (помилка: no_resolution)."),
    ("hidden_dissatisfaction_1", "easter_egg",
     "КЛІЄНТ: Скаржиться на повільний інтернет. АГЕНТ: Скидає лінк на 100-сторінкову інструкцію. КЛІЄНТ: 'Ясно, сам розберусь. Дякую.' (проблема НЕ вирішена)."),
    ("hidden_dissatisfaction_2", "easter_egg",
     "КЛІЄНТ: Чому не спрацював промокод? АГЕНТ: Каже акція закінчилась (брехня, incorrect_info). КЛІЄНТ: 'Ну окей, хай буде без знижки, до побачення' (прихована образа)."),
    ("hidden_dissatisfaction_3", "easter_egg",
     "КЛІЄНТ: Хоче змінити пошту. АГЕНТ: 'Я створю тікет на адмінів, чекайте 5 днів' (unnecessary_escalation). КЛІЄНТ: 'Добре, буду чекати, дякую'."),
    ("mixed_tariff", "normal",
     "КЛІЄНТ: Студент, шукає халяву, використовує молодіжний сленг ('крінж', 'чілити'). АГЕНТ: Підбирає молодіжний тариф."),
    ("mixed_other", "problem",
     "КЛІЄНТ: Питає, як повністю видалити свій акаунт. АГЕНТ: Повністю ігнорує питання і нав'язує знижку 50% (помилка: ignored_question)."),
    ("mixed_refund", "easter_egg",
     "КЛІЄНТ: Гроші за повернення не прийшли на 5 день. АГЕНТ: 'Це проблема вашого банку' (rude_tone + no_resolution). КЛІЄНТ: 'Зрозуміло, піду сваритись з банком, до побачення'."),
    ("mixed_tech", "conflict",
     "КЛІЄНТ: Агресивний, через ваш баг втратив гроші. АГЕНТ: Замість вибачень починає агресивно сперечатися (помилка: rude_tone).")
]

SCENARIOS = []
for i in range(2):
    for s_id, s_type, s_desc in BASE_SCENARIOS:
        SCENARIOS.append((f"{s_id}_v{i + 1}", s_type, s_desc))


def generate_batch(batch_scenarios, batch_num):
    prompt = f"""Ти - AI-генератор реалістичних діалогів для тренування моделей служби підтримки.
Твоє завдання: згенерувати {len(batch_scenarios)} різних чатів українською мовою.

КРИТИЧНІ ВИМОГИ:
1. РЕАЛІСТИЧНІСТЬ: Клієнти не повинні бути ідеальними. Використовуй емоції, капслок для злих клієнтів, відсутність розділових знаків для тих, хто поспішає, дрібні одруки, сленг. Агент спілкується згідно зі своїм описом.
2. РІЗНОМАНІТНА ДОВЖИНА: Для кожного з {len(batch_scenarios)} діалогів випадково обери довжину від 5 до 40 реплік (сумарно клієнт+агент). 
   - Частина діалогів має бути короткою (5-8 реплік) - швидке вирішення проблеми.
   - Частина середньою (15-20 реплік) - стандартна консультація.
   - Мінімум 1-2 діалоги мають бути дуже довгими (30-40 реплік) - клієнт не розуміє, багато уточнень або затяжний конфлікт.

Поверни результат ВИКЛЮЧНО у форматі валідного JSON-масиву. Структура:
[
  {{
    "id": "назва_сценарію",
    "messages": [
      {{"role": "client", "text": "жива репліка"}},
      {{"role": "agent", "text": "репліка"}}
    ]
  }}
]

Сценарії для поточного пакету:
"""
    for i, (s_id, s_type, s_desc) in enumerate(batch_scenarios):
        prompt += f"\nДіалог {i + 1}:\n- id: {s_id}\n- Опис: {s_desc}\n"

    for attempt in range(MAX_RETRIES):
        try:
            print(f"  [Batch {batch_num}] Запит до API (спроба {attempt + 1}/{MAX_RETRIES})...")
            response = model.generate_content(
                prompt,
                generation_config=genai.GenerationConfig(
                    temperature=0.85,
                    response_mime_type="application/json"
                )
            )
            return json.loads(response.text.strip())

        except Exception as e:
            print(f"  [Batch {batch_num}] Помилка генерації або парсингу JSON: {e}")
            if attempt < MAX_RETRIES - 1:
                time.sleep(5)
            else:
                raise Exception(f"Failed to generate batch {batch_num} after {MAX_RETRIES} attempts.")


def main():
    print(f"Початок генерації {len(SCENARIOS)} діалогів (Batch size: {BATCH_SIZE})...")

    all_chats_dict = {}
    batches = [SCENARIOS[i:i + BATCH_SIZE] for i in range(0, len(SCENARIOS), BATCH_SIZE)]

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