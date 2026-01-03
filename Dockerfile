FROM python:3.9-slim

# Empêcher Python de créer des fichiers .pyc
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    git \
    && rm -rf /var/lib/apt/lists/*
 

WORKDIR /app

# Installer les dépendances en cache séparé
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
# Copier le reste du code
COPY . .

CMD ["uvicorn", "agent:app", "--host", "0.0.0.0", "--port", "80"]