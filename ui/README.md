# Actus UI

`ui/` 是 Actus 的 Next.js 16 前端，负责聊天交互、会话列表、任务摘要、工作台、设置页和管理员界面。

## 当前界面能力

- 登录 / 注册
- 首页快捷提问与会话入口
- 左侧会话列表、删除会话、主题切换
- 会话详情页：
  - 流式消息展示
  - 计划 / 步骤状态展示
  - 任务摘要与文件面板
  - 终端预览
  - 浏览器预览
  - VNC 画面
  - 时间线回放
- 设置弹窗：
  - Agent 通用配置
  - 模型提供商配置
  - MCP 服务器
  - A2A Agent 配置
  - Skill 生态
  - 用户管理
- 图片代理路由：`/api/image-proxy`

## 技术栈

- Next.js 16（App Router）
- React 19
- Tailwind CSS 4
- Zustand
- Radix UI
- `markdown-it`
- noVNC
- Vitest + Testing Library

## 本地开发

```bash
cd ui
npm install
npm run dev
```

常用命令：

```bash
npm run lint
npm run test
npm run build
```

默认开发地址：`http://localhost:3000`

## 环境变量

| 变量 | 说明 | 默认值 |
|------|------|--------|
| `NEXT_PUBLIC_API_BASE_URL` | 浏览器可访问的后端 API 地址 | `http://localhost:8000/api` |

注意：

- 这是**构建时注入**变量
- 容器部署时由 `ui-app` 镜像构建参数传入
- 修改后端域名或端口后，需要重新执行前端构建

## 部署形态

容器模式下前端分成两层：

- `ui-app`：Next.js standalone 运行时
- `ui`：nginx 网关，对外暴露 80 端口

对应文件：

- `Dockerfile`
- `Dockerfile.nginx`
- `next.config.ts`
- `nginx.conf`

## 目录结构

```text
ui/
├── src/
│   ├── app/                   # App Router 页面与 API route
│   │   ├── page.tsx           # 首页
│   │   ├── login/             # 登录页
│   │   ├── register/          # 注册页
│   │   ├── sessions/[id]/     # 会话详情页
│   │   └── api/image-proxy/   # 图片代理
│   ├── components/            # UI 组件
│   ├── hooks/                 # 自定义 hooks
│   ├── lib/                   # API 客户端、store、状态文案、工具函数
│   └── test/                  # 测试初始化
├── public/
├── package.json
└── next.config.ts
```

## 关键组件

- `components/left-panel.tsx`
- `components/chat-input.tsx`
- `components/session-task-dock.tsx`
- `components/workbench-panel.tsx`
- `components/workbench-interactive-terminal.tsx`
- `components/workbench-browser-preview.tsx`
- `components/vnc-viewer.tsx`
- `components/manus-settings.tsx`

## 状态管理

Zustand store 位于：

- `src/lib/store/auth-store.ts`
- `src/lib/store/session-store.ts`
- `src/lib/store/settings-store.ts`
- `src/lib/store/ui-store.ts`

## 测试

测试文件与源码并列，例如：

- `src/components/*.test.tsx`
- `src/lib/**/*.test.ts`
- `src/app/sessions/[id]/page.test.tsx`

运行方式：

```bash
cd ui
npm run test
```
