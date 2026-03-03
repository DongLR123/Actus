import {
  AlertCircle,
  CheckCircle2,
  CircleDashed,
  CircleHelp,
  Clock3,
  Hand,
  Loader2,
  XCircle,
} from "lucide-react";

import type { StatusMeta } from "@/lib/status-copy";
import { cn } from "@/lib/utils";

const iconMap = {
  "circle-dashed": CircleDashed,
  loader: Loader2,
  clock: Clock3,
  "check-circle": CheckCircle2,
  "x-circle": XCircle,
  "alert-circle": AlertCircle,
  hand: Hand,
  "circle-help": CircleHelp,
} as const;

const toneClassMap = {
  muted: "text-muted-foreground",
  warning: "text-amber-600 dark:text-amber-400",
  success: "text-emerald-600 dark:text-emerald-400",
  danger: "text-red-600 dark:text-red-400",
  info: "text-blue-600 dark:text-blue-400",
} as const;

type StatusIndicatorProps = {
  meta: StatusMeta;
  className?: string;
};

export function StatusIndicator({ meta, className }: Readonly<StatusIndicatorProps>) {
  const Icon = iconMap[meta.icon] ?? CircleHelp;

  return (
    <span className={cn("inline-flex items-center gap-1", toneClassMap[meta.tone], className)}>
      <Icon
        size={12}
        data-testid={`status-icon-${meta.icon}`}
        className={cn(meta.spinning ? "animate-spin" : undefined)}
      />
      <span>{meta.text}</span>
    </span>
  );
}
