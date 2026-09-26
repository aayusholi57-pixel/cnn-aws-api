FROM python:3.10-slim-bookworm

WORKDIR /app

# requirements.txt is UTF-16 LE; convert it before pip reads it.
COPY requirements.txt .
RUN python -c "from pathlib import Path; p = Path('requirements.txt'); p.write_text(p.read_text(encoding='utf-16'), encoding='utf-8')" \
    && pip install --no-cache-dir --disable-pip-version-check -r requirements.txt

# Copy only the application and its trained weights (see .dockerignore).
COPY . .

EXPOSE 9000
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "9000"]
