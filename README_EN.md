<p align="center">
  <h1 align="center">Actus</h1>
  <p align="center">
    A self-hosted AI Agent platform for planning, reasoning, execution, and human takeover
  </p>
  <p align="center">
    <a href="LICENSE"><img src="https://img.shields.io/badge/License-Apache_2.0-blue.svg" alt="License"></a>
    <img src="https://img.shields.io/badge/python-3.12-blue.svg" alt="Python">
    <img src="https://img.shields.io/badge/Next.js-16-black.svg" alt="Next.js">
  </p>
  <p align="center">
    English · <a href="README.md">中文</a>
  </p>
</p>

---

## Overview

Actus is organized around three runtimes:

- `api/`: FastAPI backend for sessions, agents, auth, files, settings, skills, and sandbox orchestration
- `ui/`: Next.js 16 frontend for chat, task progress, workbench, settings, and admin views
- `sandbox/`: per-session Docker sandbox with Shell, filesystem access, Chromium, and VNC/noVNC

The default execution model is a `Planner + ReAct` flow: Actus first builds a plan, then executes it step by step while streaming plan updates, tool calls, messages, and takeover events back to the client.

## Core Capabilities

- **Planner + ReAct agent flow**
- **Built-in tools** for file, shell, browser, search, and messaging
- **MCP / A2A / Skill integrations** managed as first-class agent tools
- **Skill v2 filesystem storage** under `/app/data/skills`
- **Human takeover** for both `shell` and `browser` scopes
- **Workbench UI** with terminal preview, browser preview, VNC, timeline scrubbing, and file preview
- **Streaming transport** via SSE and WebSocket
- **Containerized sandbox** with Chromium, Xvfb, x11vnc, and websockify
- **Attachment storage** through MinIO / S3-compatible object storage
- **JWT auth, admin tools, tool preferences, and runtime settings**

## Architecture

```text
┌──────────────┐     ┌──────────────┐     ┌──────────────┐
│ UI (Next.js) │────▶│ API (FastAPI)│────▶│ PostgreSQL   │
└──────────────┘     │              │     └──────────────┘
                     │ Agent / Auth │────▶│ Redis        │
                     │ Files / Skill│     └──────────────┘
                     │ Settings     │────▶│ MinIO / S3   │
                     │ Sandbox Ctrl │     └──────────────┘
                     └──────┬───────┘
                            │ Docker
                            ▼
                     ┌──────────────┐
                     │ Sandbox      │
                     │ Shell/File   │
                     │ Chromium/VNC │
                     └──────────────┘
```

For backend layering and runtime composition, see [项目架构.md](项目架构.md).

## Docker Compose Quick Start

### Requirements

- Docker Engine + Docker Compose v2
- At least 6 GB RAM available to Docker
- A reachable MinIO / S3-compatible bucket that already exists
- A valid LLM API key

### Start the stack

```bash
git clone https://github.com/hahaliu1029/Actus.git
cd Actus

cp .env.example .env
# Edit .env and provide at least:
# POSTGRES_PASSWORD
# JWT_SECRET_KEY
# MINIO_ENDPOINT
# MINIO_ACCESS_KEY
# MINIO_SECRET_KEY
# MINIO_BUCKET_NAME
# NEXT_PUBLIC_API_BASE_URL

docker compose --env-file .env up -d --build

# Optional: create a super admin
docker compose exec api python scripts/create_super_admin.py
```

After startup:

- UI: `http://localhost`
- API docs: `http://localhost:8000/docs`

### Runtime config notes

- In container mode, the backend uses `/app/data/config.yaml` inside the `api-data` volume
- If the file does not exist, the backend creates one from code defaults
- The recommended way to configure LLM, MCP, A2A, and Skill policies is through the frontend settings UI
- `api/config.yaml.example` is mainly for **local backend development** or manual pre-seeding

### Rebuild sandbox code correctly

The Compose service name is `sandbox-image`, not `sandbox`. After changing files under `sandbox/`, use:

```bash
docker compose --env-file .env build sandbox-image api
docker compose --env-file .env up -d --force-recreate api

# Optional: remove old temporary sandbox containers
docker ps --format '{{.Names}}' | grep '^actus-sb-' | xargs -r docker rm -f
```

## Local Development

### Frontend

```bash
cd ui
npm install
npm run dev
```

### Backend

Local backend development does **not** use the same variables as the root Compose `.env`. `api/core/config.py` reads runtime settings from `api/.env`, for example:

```bash
cd api
cp config.yaml.example config.yaml

cat > .env <<'EOF'
ENV=development
LOG_LEVEL=INFO
APP_CONFIG_FILEPATH=config.yaml
SQLALCHEMY_DATABASE_URL=postgresql+asyncpg://postgres:postgres@127.0.0.1:5432/manus
REDIS_HOST=127.0.0.1
REDIS_PORT=6379
REDIS_DB=0
MINIO_ENDPOINT=s3.example.com
MINIO_ACCESS_KEY=replace-me
MINIO_SECRET_KEY=replace-me
MINIO_SECURE=true
MINIO_BUCKET_NAME=replace-me
JWT_SECRET_KEY=replace-with-a-strong-random-string
SANDBOX_IMAGE=actus-sandbox:latest
SANDBOX_NAME_PREFIX=actus-sb
EOF

python3.12 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
bash dev.sh
```

You will also need PostgreSQL, Redis, a built `sandbox-image`, and an accessible MinIO/S3 bucket.

## Tests

```bash
# Backend
cd api
pytest

# Frontend
cd ui
npm run test
```

## Repository Layout

```text
Actus/
├── api/                  # FastAPI backend
├── ui/                   # Next.js frontend
├── sandbox/              # Docker sandbox source
├── docker-compose.yml    # Compose stack
├── DEPLOY.md             # Deployment guide
├── api.md                # English API reference
├── api_zhcn.md           # Chinese API reference
└── 项目架构.md             # Architecture notes
```

## Documentation

- [Deployment Guide](DEPLOY.md)
- [English API Reference](api.md)
- [中文 API 文档](api_zhcn.md)
- [Architecture Notes](项目架构.md)
- [API README](api/README.md)
- [UI README](ui/README.md)
- [Sandbox README](sandbox/README.md)
- [Contributing](CONTRIBUTING.md)

## License

Actus is released under the [Apache License 2.0](LICENSE).
