import { render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { describe, expect, it, vi } from "vitest";
import { TransferProgress } from "./transfer-progress";
import type { TransferTask } from "@/lib/store/transfer-store";

function buildTask(overrides?: Partial<TransferTask>): TransferTask {
  return {
    id: "t1", type: "upload", status: "transferring", filename: "report.pdf",
    totalBytes: 1000, transferredBytes: 450, progress: 45, speed: 2100000,
    createdAt: Date.now(), hasSourceFile: false, _lastProgressTime: 0, _lastProgressBytes: 0,
    ...overrides,
  };
}

describe("TransferProgress", () => {
  it("renders filename and progress", () => {
    render(<TransferProgress task={buildTask()} onCancel={vi.fn()} />);
    expect(screen.getByText("report.pdf")).toBeTruthy();
    expect(screen.getByText("45%")).toBeTruthy();
  });

  it("has ARIA progressbar attributes", () => {
    render(<TransferProgress task={buildTask()} onCancel={vi.fn()} />);
    const bar = screen.getByRole("progressbar");
    expect(bar.getAttribute("aria-valuenow")).toBe("45");
    expect(bar.getAttribute("aria-valuemax")).toBe("100");
  });

  it("calls onCancel when cancel clicked", async () => {
    const onCancel = vi.fn();
    render(<TransferProgress task={buildTask()} onCancel={onCancel} />);
    await userEvent.click(screen.getByLabelText("取消传输"));
    expect(onCancel).toHaveBeenCalledWith("t1");
  });

  it("shows retry for failed download", async () => {
    const onRetry = vi.fn();
    render(<TransferProgress task={buildTask({ status: "failed", type: "download", error: "网络错误" })} onCancel={vi.fn()} onRetry={onRetry} />);
    expect(screen.getByText("网络错误")).toBeTruthy();
    await userEvent.click(screen.getByLabelText("重试"));
    expect(onRetry).toHaveBeenCalledWith("t1");
  });

  it("shows indeterminate when totalBytes=0", () => {
    render(<TransferProgress task={buildTask({ totalBytes: 0, progress: 0 })} onCancel={vi.fn()} />);
    expect(screen.getByRole("progressbar").getAttribute("aria-valuenow")).toBe("0");
  });
});
