# 贡献指南

本文档基于当前仓库结构和运行方式编写，适用于 [Actus](https://github.com/hahaliu1029/Actus)。

## 你可以如何贡献

- 报告 Bug
- 提交功能建议
- 修复代码问题
- 完善测试
- 更新文档

## 开始前

建议先阅读：

- [行为准则](CODE_OF_CONDUCT.md)
- [部署指南](DEPLOY.md)
- [项目架构](项目架构.md)

同时请先查看现有 Issue：

- <https://github.com/hahaliu1029/Actus/issues>

## 环境要求

- Python 3.12（后端）
- Node.js 22+（前端）
- Docker + Docker Compose v2
- PostgreSQL、Redis
- 可访问的 MinIO / S3 兼容对象存储

## 开发方式建议

### 方式一：优先使用 Docker Compose 验证完整链路

适合联调、部署验证、回归测试。

```bash
cp .env.example .env
docker compose --env-file .env up -d --build
```

### 方式二：本地运行前端或后端

适合快速迭代某个子项目，但要注意前后端与 Compose 使用的配置来源不同。

## 后端开发

### 1. 启动依赖

你至少需要 PostgreSQL、Redis，以及一个已经构建好的 `sandbox-image`。最简单做法是：

```bash
docker compose up -d postgres redis
docker compose build sandbox-image
```

MinIO/S3 仍需自行准备，Compose 不会启动它。

### 2. 配置本地后端环境

`api/core/config.py` 读取的是 `api/.env`，不是根目录 Compose 用的 `.env`。建议在 `api/` 下创建：

```dotenv
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
MINIO_BUCKET_NAME=a2a-mcp
JWT_SECRET_KEY=replace-with-a-strong-random-string
SANDBOX_IMAGE=actus-sandbox:latest
SANDBOX_NAME_PREFIX=actus-sb
```

同时创建本地运行时配置：

```bash
cd api
cp config.yaml.example config.yaml
```

### 3. 安装依赖并启动

```bash
cd api
python3.12 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
bash dev.sh
```

### 4. 后端测试

```bash
cd api
pytest
```

## 前端开发

```bash
cd ui
npm install
npm run dev
```

常用命令：

```bash
cd ui
npm run lint
npm run test
npm run build
```

前端使用的主要环境变量：

- `NEXT_PUBLIC_API_BASE_URL`

该变量是**构建时注入**的；如果改了部署地址，需要重新构建前端。

## 文档更新

如果你的改动影响了以下任意内容，请同步更新文档：

- API 路由或响应格式
- 会话状态机 / 接管流程
- Skill / MCP / A2A 配置方式
- Docker Compose 服务名与部署步骤
- 本地开发命令

## 分支与提交

推荐从 `main` 创建功能分支：

```bash
git checkout -b feature/<short-description>
```

提交信息建议使用 Conventional Commits：

```text
feat(api): add takeover reopen endpoint
fix(ui): handle image proxy failures
docs: refresh deployment and API docs
```

常见类型：

- `feat`
- `fix`
- `docs`
- `refactor`
- `test`
- `chore`

## Pull Request 建议

提交 PR 前请尽量完成以下检查：

- 后端相关改动已运行 `pytest`
- 前端相关改动已运行 `npm run test`
- 如涉及构建链路，已运行 `npm run build`
- 文档与代码一致
- 未提交敏感信息、`.env` 或临时文件

## Bug 与安全问题

- 普通问题请使用 Issue：<https://github.com/hahaliu1029/Actus/issues>
- 安全问题请不要公开提交，请参考 [SECURITY.md](SECURITY.md)
