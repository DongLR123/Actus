# 前端状态文案统一与图标规范设计

- 日期：2026-03-03
- 作者：Codex
- 类型：Design
- 关联问题：前端会话状态直接显示原始状态码（如 `running`、`completed`），未做统一中文化与视觉语义表达。

## 1. 背景与目标

当前 UI 存在状态裸值渲染：

1. 会话列表 [ui/src/components/left-panel.tsx](/Users/liuyixuan/Desktop/code/opensource/Actus/ui/src/components/left-panel.tsx) 直接显示 `session.status`。
2. 会话页顶部 [ui/src/app/sessions/[id]/page.tsx](/Users/liuyixuan/Desktop/code/opensource/Actus/ui/src/app/sessions/[id]/page.tsx) 直接显示 `visibleSession?.status`。

问题：

- 用户看到英文状态码，不符合中文界面一致性。
- 状态文案、图标、颜色逻辑分散，新增状态时容易漏改。

目标：

- 统一会话状态与步骤状态的人类可读文案。
- 增加线性单色图标（Lucide）与语义色（成功/进行中/失败等）。
- 建立可复用的状态元数据层，避免重复判断逻辑。

非目标：

- 本次不引入完整多语言框架（如 next-intl）。
- 不调整后端状态流转，不修改 API 类型定义。

## 2. 备选方案与取舍

### 方案 A：组件内局部映射

在每个组件写 `switch(status)` 返回中文。

- 优点：改动快。
- 缺点：重复逻辑高，后续维护成本高，状态扩展风险大。

### 方案 B：统一状态元数据层（选型）

新增 `status-copy` + `StatusIndicator`，组件统一消费。

- 优点：单一事实来源；文案、图标、颜色统一；测试集中。
- 缺点：需要替换现有组件调用点。

### 方案 C：直接上完整 i18n 基建

- 优点：长期多语言能力最好。
- 缺点：对当前问题改动过大，收益不匹配。

结论：采用方案 B。

## 3. 状态规范（文案 + 图标 + 语义色）

### 3.1 会话状态 `SessionStatus`

| 状态值 | 文案 | 图标 | 语义色 | 备注 |
| --- | --- | --- | --- | --- |
| `pending` | 待执行 | `CircleDashed` | muted/gray | 默认兜底态 |
| `running` | 执行中 | `Loader2` | warning/amber | 旋转动画 |
| `waiting` | 等待中 | `Clock3` | warning/amber | 等待用户或外部输入 |
| `completed` | 已完成 | `CheckCircle2` | success/emerald | 终态 |
| `failed` | 失败 | `XCircle` | danger/red | 终态 |
| `takeover_pending` | 待接管 | `AlertCircle` | warning/orange | 接管请求处理中 |
| `takeover` | 接管中 | `Hand` | info/blue | 人工接管态 |

### 3.2 步骤状态 `ExecutionStatus`

| 状态值 | 文案 | 图标 | 语义色 | 备注 |
| --- | --- | --- | --- | --- |
| `pending` | 待执行 | `CircleDashed` | muted/gray | - |
| `started` | 执行中 | `Loader2` | warning/amber | 归并为 running 视觉 |
| `running` | 执行中 | `Loader2` | warning/amber | 旋转动画 |
| `completed` | 已完成 | `CheckCircle2` | success/emerald | - |
| `failed` | 失败 | `XCircle` | danger/red | - |

### 3.3 未知状态兜底

- 文案：`未知状态`
- 图标：`CircleHelp`
- 语义色：`muted/gray`
- 行为：不抛错，确保 UI 可渲染；可附带原始值用于日志排查。

## 4. 架构设计

### 4.1 新增统一状态元数据模块

新增文件：`ui/src/lib/status-copy.ts`

职责：

1. 定义状态到元数据的映射字典。
2. 导出统一方法：
   - `getSessionStatusMeta(status)`
   - `getStepStatusMeta(status)`
3. 元数据结构统一：
   - `text`
   - `tone`（用于样式）
   - `icon`（字符串 key）
   - `spinning`（是否旋转）
   - `rawStatus`（可选，调试用）

### 4.2 新增通用渲染组件

新增文件：`ui/src/components/status-indicator.tsx`

职责：

1. 接收 `meta` 统一渲染图标+文案。
2. 内部完成 icon key -> Lucide 组件映射。
3. 基于 `tone` 输出一致 className（色彩与对比度）。

### 4.3 接入点（本期）

1. [ui/src/components/left-panel.tsx](/Users/liuyixuan/Desktop/code/opensource/Actus/ui/src/components/left-panel.tsx)
   - 当前：`{session.status}`
   - 目标：`<StatusIndicator meta={getSessionStatusMeta(session.status)} />`

2. [ui/src/app/sessions/[id]/page.tsx](/Users/liuyixuan/Desktop/code/opensource/Actus/ui/src/app/sessions/[id]/page.tsx)
   - 当前：`当前状态：{visibleSession?.status || "pending"}`
   - 目标：`当前状态：<StatusIndicator ... />`

3. 步骤状态（分阶段）
   - 当前 `SessionTaskDock` 内已有中文+图标逻辑，可先保持。
   - 下一步可迁移到 `getStepStatusMeta` 复用。

## 5. 数据流

1. 页面拿到后端原始状态码。
2. 调用 `getSessionStatusMeta` 或 `getStepStatusMeta` 归一化。
3. `StatusIndicator` 根据 meta 渲染图标、颜色、文案。
4. 未知值走 fallback，不阻塞界面。

该设计不改变状态值本身，仅改变展示层。

## 6. 错误处理与边界

1. `status` 为 `null/undefined/""`：按 `pending` 渲染。
2. `status` 为非预期字符串：按未知状态渲染。
3. icon key 不匹配：退回 `CircleHelp`。
4. UI 保持无异常渲染，不因状态字段异常崩溃。

## 7. 测试设计

### 7.1 单元测试

新增：`ui/src/lib/status-copy.test.ts`

覆盖：

1. 所有已知会话状态映射正确（文案、icon、tone）。
2. 所有已知步骤状态映射正确。
3. 未知状态与空值回退正确。

### 7.2 组件测试

更新：`ui/src/components/left-panel.test.tsx`

覆盖：

1. `running` 时显示中文（执行中）。
2. 不出现原始状态英文裸值。

更新：`ui/src/app/sessions/[id]/page.test.tsx`

覆盖：

1. 顶部“当前状态”显示中文文案。
2. 状态为 `completed`、`running` 的文案正确。

## 8. 实施计划（高层）

1. 新建 `status-copy.ts` 定义映射。
2. 新建 `StatusIndicator` 通用组件。
3. 替换会话列表与会话页顶部状态展示。
4. 补充/更新测试。
5. 验证测试通过。

## 9. 风险与回滚

风险：

1. 漏掉某个状态值导致显示 fallback。
2. className 颜色不一致带来视觉偏差。

回滚：

- 变更集中在新增模块与 2 个接入组件，可按文件快速回滚。

## 10. 验收标准

1. 页面不再出现 `running/completed/pending/waiting/takeover...` 等原始英文状态码。
2. 会话状态统一显示中文文案与线性图标。
3. 未知状态有稳定兜底。
4. 相关测试通过。
