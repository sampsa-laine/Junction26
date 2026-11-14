FROM python:3.12-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
    bash \
    ca-certificates \
    curl \
    git \
    nodejs \
    npm \
    procps \
    && rm -rf /var/lib/apt/lists/*

COPY --from=ghcr.io/astral-sh/uv:0.9.7 /uv /usr/local/bin/uv

RUN uv pip install --system --no-cache \
    "numpy==2.5.4" \
    "scipy==1.18.1" \
    "shapely==2.2.0"

# Pre-install nvm + Node 22 + opencode-ai so Harbor agent setup is near-instant
ENV NVM_DIR=/root/.nvm
RUN curl -fsSL https://raw.githubusercontent.com/nvm-sh/nvm/v0.40.2/install.sh | bash \
    && . "$NVM_DIR/nvm.sh" \
    && nvm install 22 \
    && nvm alias default 22 \
    && npm i -g opencode-ai@latest
ENV NPM_CONFIG_PREFER_OFFLINE=true

COPY cad_env /opt/cad_env
COPY cad-sim /usr/local/bin/cad-sim
RUN chmod +x /usr/local/bin/cad-sim

WORKDIR /app
COPY problem.json /app/problem.json
COPY design.json /app/design.json

ENV PYTHONPATH=/opt
ENV CAD_PROBLEM_PATH=/app/problem.json
ENV CAD_DESIGN_PATH=/app/design.json
