# Actus API

`api/` 是 Actus 的 FastAPI 后端，负责会话编排、Agent 执行、认证授权、文件管理、运行时配置、Skill 生态以及 Docker 沙箱调度。

## 当前能力

- 用户注册、登录、刷新令牌、个人资料维护
- 超级管理员用户管理
- 会话创建、SSE 对话流、任务停止、文件读取
- `shell` / `browser` 接管、续期、结束、补救
- LLM / MCP / A2A / Skill 风险策略配置
- Skill v2 安装、启用、删除、详情查看、AI 创建
- 文件上传、下载、删除
- 健康检查与 MinIO 自检

## 运行依赖

- Python 3.12
- PostgreSQL
- Redis
- MinIO / S3 兼容对象存储
- Docker（用于创建会话沙箱）

容器镜像额外内置：

- Docker CLI（支持 stdio 类 MCP 服务）
- Node.js 22（满足部分工具运行需求）

## 本地开发

### 1. 启动基础依赖

```bash
docker compose up -d postgres redis
docker compose build sandbox-image
```

说明：

- Compose 不会启动 MinIO，你需要单独准备对象存储
- `sandbox-image` 只是构建镜像；真正的会话沙箱由 API 运行时动态创建

### 2. 准备本地配置

在 `api/` 下创建本地运行时配置文件：

```bash
cd api
cp config.yaml.example config.yaml
```

再创建 `api/.env`。后端本地运行读取的是这里，而不是根目录 Compose 的 `.env`：

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
MINIO_BUCKET_NAME=replace-me
JWT_SECRET_KEY=replace-with-a-strong-random-string
SANDBOX_IMAGE=actus-sandbox:latest
SANDBOX_NAME_PREFIX=actus-sb
```

### 3. 安装依赖并启动

```bash
cd api
python3.12 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
bash dev.sh
```

启动后：

- API 文档：`http://localhost:8000/docs`
- OpenAPI：`http://localhost:8000/openapi.json`

## 配置来源

### 本地运行

- 环境变量：`api/.env`
- 业务运行时配置：`api/config.yaml`

### Docker Compose 运行

- 环境变量：仓库根目录 `.env`
- 业务运行时配置：`/app/data/config.yaml`
- Skill 目录：`/app/data/skills`

如果 `/app/data/config.yaml` 不存在，后端会按默认值自动创建。

## 测试

```bash
cd api
pytest
```

## 目录结构

```text
api/
├── app/
│   ├── application/      # 用例编排服务
│   ├── domain/           # 领域模型、工具、流程、Prompt
│   ├── infrastructure/   # 仓储实现、外部服务、存储客户端
│   └── interfaces/       # FastAPI 路由、Schema、依赖注入
├── core/                 # 环境变量与安全配置
├── alembic/              # 数据库迁移
├── scripts/              # 管理脚本
├── tests/                # 后端测试
├── config.yaml.example   # 本地运行时配置模板
├── dev.sh                # 本地开发启动脚本
├── run.sh                # 生产启动脚本
└── requirements.txt      # Python 依赖
```

## 关键路由模块

- `auth_routes.py`
- `session_routes.py`
- `app_config_routes.py`
- `file_routes.py`
- `skill_v2_routes.py`
- `user_routes.py`
- `user_tools_v2_routes.py`
- `admin_routes.py`
- `status_routes.py`

完整路由清单见：

- [../api_zhcn.md](../api_zhcn.md)
- [../api.md](../api.md)

## 管理脚本

- `scripts/create_super_admin.py`
- `scripts/reset_admin_password.py`
- `scripts/minio_smoke_test.py`
- `scripts/minio_upload_file.py`
- `scripts/migrate_skills_db_to_fs.py`
- `scripts/rollback_skills_fs_to_db.py`

## 开发注意事项

- 会话聊天和会话列表流使用 SSE，不是普通轮询
- 接管终端和 VNC 使用 WebSocket
- `Skill` 旧接口保留为 `410` 迁移提示，新接口统一在 `/api/v2/skills/*`
- 修改 `sandbox/` 后，记得重建 `sandbox-image` 和 `api`
