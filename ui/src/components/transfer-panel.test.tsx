import { render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { beforeEach, describe, expect, it, vi } from "vitest";

vi.mock("@/lib/store/reset", () => ({ registerStoreResetter: vi.fn() }));

import { useTransferStore } from "@/lib/store/transfer-store";
import { TransferPanel } from "./transfer-panel";

function getStore() { return useTransferStore.getState(); }

describe("TransferPanel", () => {
  beforeEach(() => { getStore()._reset(); });

  it("renders nothing when no tasks", () => {
    const { container } = render(<TransferPanel />);
    expect(container.firstChild).toBeNull();
  });

  it("renders panel with task count when tasks exist", () => {
    getStore().addTask({ type: "upload", filename: "a.txt", totalBytes: 100 });
    render(<TransferPanel />);
    expect(screen.getByText(/传输/)).toBeTruthy();
  });

  it("shows re-select hint for failed uploads (no retry button)", () => {
    const { taskId } = getStore().addTask({ type: "upload", filename: "a.txt", totalBytes: 100 });
    getStore().failTask(taskId, "err");
    render(<TransferPanel />);
    expect(screen.getByText(/重新选择/)).toBeTruthy();
  });

  it("shows retry button for failed downloads", () => {
    const { taskId } = getStore().addTask({ type: "download", filename: "b.csv", totalBytes: 200, sourceRef: "f1" });
    getStore().failTask(taskId, "err");
    render(<TransferPanel />);
    expect(screen.getByLabelText("重试")).toBeTruthy();
  });

  it("clears completed tasks", async () => {
    const { taskId } = getStore().addTask({ type: "upload", filename: "a.txt", totalBytes: 100 });
    getStore().completeTask(taskId);
    render(<TransferPanel />);
    await userEvent.click(screen.getByText("清除已完成"));
    expect(Object.keys(getStore().tasks)).toHaveLength(0);
  });
});
