# ==================== Étape 1 : Builder ====================
FROM python:3.9-slim AS builder

# Installer les outils de compilation nécessaires
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    gcc \
    git \
    && rm -rf /var/lib/apt/lists/*

# Empêcher la création de .pyc + output non bufférisé
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

# Copier seulement requirements pour profiter du cache
COPY requirements.txt .
RUN pip install --user --no-cache-dir -r requirements.txt

# ==================== Étape 2 : Runtime final (très léger) ====================
FROM python:3.9-slim

# Créer un utilisateur non-root pour sécurité
RUN adduser --disabled-password --gecos '' appuser

# Installer seulement les dépendances runtime nécessaires (ffmpeg pour torchaudio/speechbrain)
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

# Copier les packages installés depuis le builder
COPY --from=builder /root/.local /home/appuser/.local

# Rendre les binaires pip accessibles
ENV PATH=/home/appuser/.local/bin:$PATH

# Copier le code source
COPY . .

# Changer propriétaire des fichiers
RUN chown -R appuser:appuser /app
USER appuser

# Exposer le port
EXPOSE 80

# Lancer avec uvicorn (optimisé : workers auto si possible)
CMD ["uvicorn", "agent:app", "--host", "0.0.0.0", "--port", "80", "--workers", "2"]