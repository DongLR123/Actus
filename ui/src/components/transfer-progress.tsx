import { RefreshCw, X } from "lucide-react";

import type { TransferTask } from "@/lib/store/transfer-store";
import { formatFileSize } from "@/lib/session-ui";
import { cn } from "@/lib/utils";

interface TransferProgressProps {
  task: TransferTask;
  onCancel: (taskId: string) => void;
  onRetry?: (taskId: string) => void;
  className?: string;
}

const isActive = (status: TransferTask["status"]) =>
  status === "pending" || status === "transferring";

const isTerminal = (status: TransferTask["status"]) =>
  status === "failed" || status === "cancelled";

export function TransferProgress({
  task,
  onCancel,
  onRetry,
  className,
}: Readonly<TransferProgressProps>) {
  const { id, filename, progress, speed, status, error, totalBytes } = task;
  const indeterminate = totalBytes === 0;

  return (
    <div
      className={cn(
        "flex flex-col gap-1.5 rounded-xl border border-border bg-muted/80 px-3 py-2.5",
        className
      )}
    >
      {/* Row 1: filename + percentage + action buttons */}
      <div className="flex items-center gap-2">
        <span className="min-w-0 flex-1 truncate text-xs font-medium text-foreground">
          {filename}
        </span>
        <span className="shrink-0 text-xs text-muted-foreground">
          {Math.round(progress)}%
        </span>
        {isActive(status) && (
          <button
            type="button"
            aria-label="取消传输"
            onClick={() => onCancel(id)}
            className="shrink-0 rounded p-0.5 text-muted-foreground transition-colors hover:bg-destructive/10 hover:text-destructive"
          >
            <X size={12} />
          </button>
        )}
        {isTerminal(status) && onRetry && (
          <button
            type="button"
            aria-label="重试"
            onClick={() => onRetry(id)}
            className="shrink-0 rounded p-0.5 text-muted-foreground transition-colors hover:bg-primary/10 hover:text-primary"
          >
            <RefreshCw size={12} />
          </button>
        )}
      </div>

      {/* Row 2: progress bar */}
      <div
        role="progressbar"
        aria-valuenow={Math.round(progress)}
        aria-valuemin={0}
        aria-valuemax={100}
        aria-label={filename}
        className="h-1.5 w-full overflow-hidden rounded-full bg-muted"
      >
        <div
          className={cn(
            "h-full rounded-full bg-primary transition-[width] duration-300",
            indeterminate && "w-full animate-pulse"
          )}
          style={indeterminate ? undefined : { width: `${progress}%` }}
        />
      </div>

      {/* Row 3: speed or error */}
      {isTerminal(status) && error ? (
        <span className="text-xs text-destructive">{error}</span>
      ) : isActive(status) && speed > 0 ? (
        <span className="text-xs text-muted-foreground">
          {formatFileSize(speed)}/s
        </span>
      ) : null}
    </div>
  );
}
