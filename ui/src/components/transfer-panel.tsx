"use client";

import { useState } from "react";
import { ArrowDown, ArrowUp, ChevronDown, ChevronUp, RefreshCw, X } from "lucide-react";

import {
  useTransferStore,
  type TransferTask,
} from "@/lib/store/transfer-store";
import { TransferProgress } from "@/components/transfer-progress";

export function TransferPanel() {
  const [collapsed, setCollapsed] = useState(false);

  const tasks = useTransferStore((s) => s.tasks);
  const cancelTask = useTransferStore((s) => s.cancelTask);
  const retryTask = useTransferStore((s) => s.retryTask);
  const removeTask = useTransferStore((s) => s.removeTask);
  const clearCompleted = useTransferStore((s) => s.clearCompleted);

  const taskList: TransferTask[] = Object.values(tasks).sort(
    (a, b) => a.createdAt - b.createdAt,
  );

  if (taskList.length === 0) return null;

  const activeCount = taskList.filter(
    (t) => t.status === "pending" || t.status === "transferring",
  ).length;
  const completedCount = taskList.filter(
    (t) => t.status === "completed",
  ).length;
  const totalCount = taskList.length;

  return (
    <div className="fixed bottom-4 right-4 z-50 w-80 rounded-xl border border-border bg-card shadow-lg">
      {/* Header */}
      <div className="flex items-center gap-2 px-3 py-2.5">
        <span className="flex-1 text-sm font-semibold text-foreground">
          传输 {completedCount}/{totalCount}
        </span>
        {activeCount > 0 && (
          <span className="rounded-full bg-primary px-1.5 py-0.5 text-xs font-medium text-primary-foreground">
            {activeCount}
          </span>
        )}
        <button
          type="button"
          aria-label={collapsed ? "展开" : "折叠"}
          onClick={() => setCollapsed((c) => !c)}
          className="rounded p-0.5 text-muted-foreground transition-colors hover:bg-muted"
        >
          {collapsed ? <ChevronDown size={14} /> : <ChevronUp size={14} />}
        </button>
      </div>

      {/* Task list */}
      {!collapsed && (
        <div className="flex flex-col gap-1.5 px-3 pb-2">
          {taskList.map((task) => {
            const dirIcon =
              task.type === "upload" ? (
                <ArrowUp size={12} className="shrink-0 text-muted-foreground" />
              ) : (
                <ArrowDown
                  size={12}
                  className="shrink-0 text-muted-foreground"
                />
              );

            if (
              task.status === "pending" ||
              task.status === "transferring"
            ) {
              return (
                <div key={task.id} className="flex items-start gap-1.5">
                  <div className="mt-2.5">{dirIcon}</div>
                  <TransferProgress
                    task={task}
                    onCancel={cancelTask}
                    className="flex-1"
                  />
                </div>
              );
            }

            if (task.status === "completed") {
              return (
                <div
                  key={task.id}
                  className="flex items-center gap-1.5 rounded-lg px-1 py-1"
                >
                  {dirIcon}
                  <span className="min-w-0 flex-1 truncate text-xs text-foreground">
                    {task.filename}
                  </span>
                  <span className="shrink-0 text-xs text-muted-foreground">
                    完成
                  </span>
                  <button
                    type="button"
                    aria-label="移除"
                    onClick={() => removeTask(task.id)}
                    className="shrink-0 rounded p-0.5 text-muted-foreground transition-colors hover:bg-muted hover:text-foreground"
                  >
                    <X size={12} />
                  </button>
                </div>
              );
            }

            // failed or cancelled
            return (
              <div
                key={task.id}
                className="flex flex-col gap-1 rounded-lg border border-destructive/20 bg-destructive/5 px-2 py-1.5"
              >
                <div className="flex items-center gap-1.5">
                  {dirIcon}
                  <span className="min-w-0 flex-1 truncate text-xs font-medium text-foreground">
                    {task.filename}
                  </span>
                  {task.type === "download" && (
                    <button
                      type="button"
                      aria-label="重试"
                      onClick={() => retryTask(task.id)}
                      className="shrink-0 rounded p-0.5 text-muted-foreground transition-colors hover:bg-primary/10 hover:text-primary"
                    >
                      <RefreshCw size={12} />
                    </button>
                  )}
                  <button
                    type="button"
                    aria-label="移除"
                    onClick={() => removeTask(task.id)}
                    className="shrink-0 rounded p-0.5 text-muted-foreground transition-colors hover:bg-muted hover:text-foreground"
                  >
                    <X size={12} />
                  </button>
                </div>
                {task.error && (
                  <span className="text-xs text-destructive">{task.error}</span>
                )}
                {task.type === "upload" && (
                  <span className="text-xs text-muted-foreground">
                    重新选择文件以重试
                  </span>
                )}
              </div>
            );
          })}

          {/* Clear completed button */}
          {completedCount > 0 && (
            <button
              type="button"
              onClick={clearCompleted}
              className="mt-0.5 w-full rounded-lg py-1.5 text-xs text-muted-foreground transition-colors hover:bg-muted hover:text-foreground"
            >
              清除已完成
            </button>
          )}
        </div>
      )}
    </div>
  );
}
