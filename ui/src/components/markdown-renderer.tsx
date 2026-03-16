"use client";

import MarkdownIt from "markdown-it";
import { useMemo } from "react";

import { cn } from "@/lib/utils";

const md = new MarkdownIt({
  html: false,
  linkify: true,
  breaks: true,
});

/**
 * 预处理 XML 标签：
 * 1. <think>/<thinking> → Markdown 引用块（保留展示思考过程）
 * 2. <tool_code> → Markdown 代码块
 * 3. <tool ...> → 紧凑提示文本（避免大段 XML 和空白）
 */
function preprocessXmlTags(content: string): string {
  let result = content;

  // 将完整的 <think>...</think> 转为引用块
  result = result.replace(
    /<think(?:ing)?>([\s\S]*?)<\/think(?:ing)?>/gi,
    (_match, inner: string) => {
      const quoted = inner
        .trim()
        .split("\n")
        .map((line: string) => `> ${line}`)
        .join("\n");
      return `\n> **💭 思考过程**\n>\n${quoted}\n`;
    }
  );

  // 处理未闭合的 <think>（流式截断场景）
  result = result.replace(
    /<think(?:ing)?>(?![\s\S]*<\/think)([\s\S]*)$/gi,
    (_match, inner: string) => {
      const quoted = inner
        .trim()
        .split("\n")
        .map((line: string) => `> ${line}`)
        .join("\n");
      return `\n> **💭 思考中…**\n>\n${quoted}\n`;
    }
  );

  // <tool_code>...</tool_code> → 代码块
  result = result.replace(
    /<tool_code>([\s\S]*?)<\/tool_code>/g,
    "\n```xml\n$1\n```\n"
  );

  // <tool ...>...</tool> → 紧凑提示（工具调用已通过独立事件卡片展示）
  result = result.replace(/<tool\b([^>]*)>[\s\S]*?<\/tool>/gi, (_match, attrs: string) => {
    const nameMatch = attrs.match(/name\s*=\s*["']([^"']+)["']/i);
    const toolName = nameMatch?.[1]?.trim();
    const displayName = toolName || "未知工具";
    return `\n> 🔧 工具调用：${displayName}\n`;
  });

  // 折叠连续空白行，避免出现大面积留白
  result = result.replace(/\n{3,}/g, "\n\n");

  return result.trim();
}

const defaultLinkOpen = md.renderer.rules.link_open;
md.renderer.rules.link_open = (tokens, index, options, env, self) => {
  tokens[index]?.attrSet("target", "_blank");
  tokens[index]?.attrSet("rel", "noopener noreferrer nofollow");
  if (defaultLinkOpen) {
    return defaultLinkOpen(tokens, index, options, env, self);
  }
  return self.renderToken(tokens, index, options);
};

type MarkdownRendererProps = {
  content: string;
  className?: string;
};

export function MarkdownRenderer({ content, className }: Readonly<MarkdownRendererProps>) {
  const html = useMemo(() => {
    const preprocessed = preprocessXmlTags(content || "（空消息）");
    return md.render(preprocessed || "（空消息）");
  }, [content]);

  return (
    <div
      className={cn(
        "break-words text-sm leading-7 text-foreground/85",
        "[&_p]:my-2 [&_p:first-child]:mt-0 [&_p:last-child]:mb-0",
        "[&_a]:text-blue-600 [&_a]:dark:text-blue-400 [&_a:hover]:underline",
        "[&_h1]:my-2 [&_h1]:text-xl [&_h1]:font-semibold",
        "[&_h2]:my-2 [&_h2]:text-lg [&_h2]:font-semibold",
        "[&_h3]:my-2 [&_h3]:text-base [&_h3]:font-semibold",
        "[&_ul]:my-2 [&_ul]:list-disc [&_ul]:pl-6",
        "[&_ol]:my-2 [&_ol]:list-decimal [&_ol]:pl-6",
        "[&_li]:my-1",
        "[&_blockquote]:my-2 [&_blockquote]:border-l-2 [&_blockquote]:border-border [&_blockquote]:pl-3 [&_blockquote]:text-muted-foreground",
        "[&_code]:rounded [&_code]:bg-muted [&_code]:px-1 [&_code]:py-0.5 [&_code]:font-mono [&_code]:text-[13px]",
        "[&_pre]:my-2 [&_pre]:overflow-x-auto [&_pre]:rounded-lg [&_pre]:bg-[oklch(0.16_0_0)] [&_pre]:p-3 [&_pre]:text-[oklch(0.92_0_0)]",
        "[&_pre_code]:bg-transparent [&_pre_code]:p-0 [&_pre_code]:text-[13px]",
        "[&_table]:my-3 [&_table]:w-full [&_table]:border-collapse [&_table]:text-sm",
        "[&_th]:border [&_th]:border-border [&_th]:bg-muted [&_th]:px-3 [&_th]:py-1.5 [&_th]:text-left [&_th]:font-semibold",
        "[&_td]:border [&_td]:border-border [&_td]:px-3 [&_td]:py-1.5",
        "[&_hr]:my-4 [&_hr]:border-border",
        className
      )}
      dangerouslySetInnerHTML={{ __html: html }}
    />
  );
}

