FROM python:3.10-slim

WORKDIR /app

# Копіюємо файл залежностей, якщо він є
COPY requirements.txt .
# Встановлюємо залежності + anthropic
RUN pip install --no-cache-dir anthropic google-generativeai python-dotenv

# Копіюємо решту коду
COPY . .

CMD ["python", "analyze.py"]