import google.generativeai as genai
import json
import os
from dotenv import load_dotenv

load_dotenv()
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

# temperature=0 — ДЕТЕРМІНОВАНІСТЬ (завжди однакові результати)
model = genai.GenerativeModel(
    "gemini-2.5-flash",
    generation_config=genai.GenerationConfig(temperature=0)
)

ANALYSIS_PROMPT = """
Ти — система аналізу якості служби підтримки.
Проаналізуй наступний діалог між клієнтом та агентом.

ДІАЛОГ:
{dialogue}

Поверни результат ВИКЛЮЧНО у форматі JSON (без пояснень, без markdown, тільки JSON):
{{
  "intent": "<одне з: payment_issue | tech_error | account_access | tariff_question | refund | other>",
  "satisfaction": "<одне з: satisfied | neutral | unsatisfied>",
  "quality_score": <число від 1 до 5>,
  "agent_mistakes": ["<список помилок або порожній масив []>"],
  "reasoning": "<одне речення — чому такий висновок>"
}}

Можливі помилки агента: ignored_question, incorrect_info, rude_tone, no_resolution, unnecessary_escalation

ВАЖЛИВО: Якщо клієнт формально дякує але проблема не вирішена — це unsatisfied, не satisfied.
"""


def format_dialogue(messages: list) -> str:
    """Перетворює список повідомлень в читабельний текст."""
    lines = []
    for msg in messages:
        role = "Клієнт" if msg["role"] == "client" else "Агент"
        lines.append(f"{role}: {msg['text']}")
    return "\n".join(lines)


def analyze_chat(chat: dict) -> dict:
    """Аналізує один діалог через Gemini API."""
    dialogue_text = format_dialogue(chat["messages"])
    prompt = ANALYSIS_PROMPT.format(dialogue=dialogue_text)
    
    response = model.generate_content(prompt)
    
    raw = response.text.strip()
    # Очищаємо від markdown якщо є
    if raw.startswith("```"):
        raw = raw.split("```")[1]
        if raw.startswith("json"):
            raw = raw[4:]
    raw = raw.strip()
    
    analysis = json.loads(raw)
    
    return {
        "chat_id": chat["id"],
        "chat_type": chat["type"],
        "analysis": analysis
    }


def main():
    # Читаємо датасет
    print("📂 Читаю dataset.json...")
    with open("dataset.json", "r", encoding="utf-8") as f:
        dataset = json.load(f)
    
    print(f"🔍 Аналізую {len(dataset)} діалогів...\n")
    
    results = []
    for chat in dataset:
        try:
            result = analyze_chat(chat)
            results.append(result)
            
            analysis = result["analysis"]
            print(f"✅ {chat['id']}")
            print(f"   intent: {analysis['intent']}")
            print(f"   satisfaction: {analysis['satisfaction']}")
            print(f"   quality: {analysis['quality_score']}/5")
            if analysis["agent_mistakes"]:
                print(f"   mistakes: {analysis['agent_mistakes']}")
            print()
        except Exception as e:
            print(f"❌ {chat['id']} — ERROR: {e}\n")
    
    # Зберігаємо результати
    with open("results.json", "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    
    print(f"✅ Готово! Результати збережено → results.json")


if __name__ == "__main__":
    main()
