import { beforeEach, describe, expect, it, vi } from "vitest";

vi.mock("@/lib/store/reset", () => ({ registerStoreResetter: vi.fn() }));

import {
  selectActiveTasks,
  selectCompletedUploadResults,
  selectHasActiveUploads,
  useTransferStore,
} from "@/lib/store/transfer-store";

function getStore() {
  return useTransferStore.getState();
}

describe("transfer-store", () => {
  beforeEach(() => {
    getStore()._reset();
  });

  // ===================== addTask =====================

  describe("addTask", () => {
    it("creates upload task with correct fields, hasSourceFile=true, EMA fields=0", () => {
      const file = new File(["hello"], "test.txt", { type: "text/plain" });
      const { taskId } = getStore().addTask({
        type: "upload",
        filename: "test.txt",
        totalBytes: 1024,
        sourceFile: file,
        sessionId: "sess-1",
      });

      const task = getStore().tasks[taskId];
      expect(task).toBeDefined();
      expect(task.type).toBe("upload");
      expect(task.status).toBe("pending");
      expect(task.filename).toBe("test.txt");
      expect(task.totalBytes).toBe(1024);
      expect(task.transferredBytes).toBe(0);
      expect(task.progress).toBe(0);
      expect(task.speed).toBe(0);
      expect(task.sessionId).toBe("sess-1");
      expect(task.hasSourceFile).toBe(true);
      expect(task._lastProgressTime).toBe(0);
      expect(task._lastProgressBytes).toBe(0);
      expect(task.createdAt).toBeGreaterThan(0);
    });

    it("creates download task with sourceRef, hasSourceFile=false", () => {
      const { taskId } = getStore().addTask({
        type: "download",
        filename: "output.pdf",
        totalBytes: 2048,
        sourceRef: "file-abc-123",
      });

      const task = getStore().tasks[taskId];
      expect(task.type).toBe("download");
      expect(task.sourceRef).toBe("file-abc-123");
      expect(task.hasSourceFile).toBe(false);
      expect(getStore().getSourceFile(taskId)).toBeUndefined();
    });

    it("returns valid taskId and non-aborted signal", () => {
      const { taskId, signal } = getStore().addTask({
        type: "download",
        filename: "file.bin",
        totalBytes: 0,
      });

      expect(typeof taskId).toBe("string");
      expect(taskId.length).toBeGreaterThan(0);
      expect(signal).toBeInstanceOf(AbortSignal);
      expect(signal.aborted).toBe(false);
    });
  });

  // ===================== updateProgress =====================

  describe("updateProgress", () => {
    it("updates transferredBytes, progress, sets status=transferring", () => {
      const { taskId } = getStore().addTask({
        type: "upload",
        filename: "big.zip",
        totalBytes: 1000,
      });

      getStore().updateProgress(taskId, 500, 1000);

      const task = getStore().tasks[taskId];
      expect(task.transferredBytes).toBe(500);
      expect(task.progress).toBe(50);
      expect(task.status).toBe("transferring");
    });

    it("updates totalBytes if initially 0 (sandbox download)", () => {
      const { taskId } = getStore().addTask({
        type: "download",
        filename: "stream.bin",
        totalBytes: 0,
      });

      getStore().updateProgress(taskId, 300, 600);

      const task = getStore().tasks[taskId];
      expect(task.totalBytes).toBe(600);
      expect(task.transferredBytes).toBe(300);
    });
  });

  // ===================== completeTask =====================

  describe("completeTask", () => {
    it("sets status=completed and stores result", () => {
      const { taskId } = getStore().addTask({
        type: "upload",
        filename: "doc.pdf",
        totalBytes: 500,
      });

      const fileInfo = {
        id: "fi-1",
        filename: "doc.pdf",
        filepath: "/uploads/doc.pdf",
        key: "uploads/doc.pdf",
        extension: "pdf",
        mime_type: "application/pdf",
        size: 500,
      };

      getStore().completeTask(taskId, fileInfo);

      const task = getStore().tasks[taskId];
      expect(task.status).toBe("completed");
      expect(task.result).toEqual(fileInfo);
    });

    it("deletes sourceFile from sidecar (hasSourceFile→false)", () => {
      const file = new File(["data"], "img.png", { type: "image/png" });
      const { taskId } = getStore().addTask({
        type: "upload",
        filename: "img.png",
        totalBytes: 4,
        sourceFile: file,
      });

      expect(getStore().getSourceFile(taskId)).toBe(file);

      getStore().completeTask(taskId);

      expect(getStore().tasks[taskId].hasSourceFile).toBe(false);
    });

    it("getSourceFile returns undefined after complete", () => {
      const file = new File(["x"], "x.txt");
      const { taskId } = getStore().addTask({
        type: "upload",
        filename: "x.txt",
        totalBytes: 1,
        sourceFile: file,
      });

      getStore().completeTask(taskId);
      expect(getStore().getSourceFile(taskId)).toBeUndefined();
    });
  });

  // ===================== failTask =====================

  describe("failTask", () => {
    it("sets status=failed with error", () => {
      const { taskId } = getStore().addTask({
        type: "upload",
        filename: "fail.txt",
        totalBytes: 10,
      });

      getStore().failTask(taskId, "network error");

      const task = getStore().tasks[taskId];
      expect(task.status).toBe("failed");
      expect(task.error).toBe("network error");
    });

    it("retains sourceFile (hasSourceFile stays true)", () => {
      const file = new File(["retry me"], "retry.txt");
      const { taskId } = getStore().addTask({
        type: "upload",
        filename: "retry.txt",
        totalBytes: 8,
        sourceFile: file,
      });

      getStore().failTask(taskId, "server error");

      expect(getStore().tasks[taskId].hasSourceFile).toBe(true);
      expect(getStore().getSourceFile(taskId)).toBe(file);
    });
  });

  // ===================== cancelTask =====================

  describe("cancelTask", () => {
    it("aborts signal, sets status=cancelled", () => {
      const { taskId, signal } = getStore().addTask({
        type: "download",
        filename: "cancel.bin",
        totalBytes: 100,
      });

      expect(signal.aborted).toBe(false);

      getStore().cancelTask(taskId);

      expect(signal.aborted).toBe(true);
      expect(getStore().tasks[taskId].status).toBe("cancelled");
    });

    it("retains sourceFile after cancel", () => {
      const file = new File(["keep me"], "keep.txt");
      const { taskId } = getStore().addTask({
        type: "upload",
        filename: "keep.txt",
        totalBytes: 7,
        sourceFile: file,
      });

      getStore().cancelTask(taskId);

      expect(getStore().tasks[taskId].hasSourceFile).toBe(true);
      expect(getStore().getSourceFile(taskId)).toBe(file);
    });
  });

  // ===================== retryTask =====================

  describe("retryTask", () => {
    it("resets failed task: status→pending, progress/speed/_lastProgressTime/_lastProgressBytes all →0, error→undefined", () => {
      const { taskId } = getStore().addTask({
        type: "upload",
        filename: "retry.txt",
        totalBytes: 100,
      });

      getStore().updateProgress(taskId, 50, 100);
      getStore().failTask(taskId, "oops");

      const before = getStore().tasks[taskId];
      const createdAtBefore = before.createdAt;

      getStore().retryTask(taskId);

      const task = getStore().tasks[taskId];
      expect(task.status).toBe("pending");
      expect(task.progress).toBe(0);
      expect(task.speed).toBe(0);
      expect(task._lastProgressTime).toBe(0);
      expect(task._lastProgressBytes).toBe(0);
      expect(task.error).toBeUndefined();
      expect(task.createdAt).toBe(createdAtBefore);
    });

    it("returns new signal (different from old)", () => {
      const { taskId, signal: oldSignal } = getStore().addTask({
        type: "upload",
        filename: "signal-test.txt",
        totalBytes: 100,
      });

      getStore().failTask(taskId, "fail");
      const { signal: newSignal } = getStore().retryTask(taskId);

      expect(newSignal).toBeInstanceOf(AbortSignal);
      expect(newSignal).not.toBe(oldSignal);
      expect(newSignal.aborted).toBe(false);
    });

    it("preserves createdAt across retry", () => {
      const { taskId } = getStore().addTask({
        type: "upload",
        filename: "preserve.txt",
        totalBytes: 50,
      });

      const originalCreatedAt = getStore().tasks[taskId].createdAt;
      getStore().cancelTask(taskId);
      getStore().retryTask(taskId);

      expect(getStore().tasks[taskId].createdAt).toBe(originalCreatedAt);
    });

    it("throws on non-terminal status task", () => {
      const { taskId } = getStore().addTask({
        type: "upload",
        filename: "active.txt",
        totalBytes: 100,
      });

      // status is "pending", not failed/cancelled
      expect(() => getStore().retryTask(taskId)).toThrow();
    });
  });

  // ===================== bindTaskSession =====================

  describe("bindTaskSession", () => {
    it("rebinds tasks with matching sessionId", () => {
      const { taskId: id1 } = getStore().addTask({
        type: "upload",
        filename: "a.txt",
        totalBytes: 10,
        sessionId: "old-sess",
      });
      const { taskId: id2 } = getStore().addTask({
        type: "upload",
        filename: "b.txt",
        totalBytes: 10,
        sessionId: "old-sess",
      });

      getStore().bindTaskSession("old-sess", "new-sess");

      expect(getStore().tasks[id1].sessionId).toBe("new-sess");
      expect(getStore().tasks[id2].sessionId).toBe("new-sess");
    });

    it("does not affect tasks with different sessionId", () => {
      const { taskId } = getStore().addTask({
        type: "upload",
        filename: "other.txt",
        totalBytes: 10,
        sessionId: "different-sess",
      });

      getStore().bindTaskSession("old-sess", "new-sess");

      expect(getStore().tasks[taskId].sessionId).toBe("different-sess");
    });
  });

  // ===================== removeTask =====================

  describe("removeTask", () => {
    it("removes task from store", () => {
      const { taskId } = getStore().addTask({
        type: "download",
        filename: "remove.bin",
        totalBytes: 0,
      });

      getStore().removeTask(taskId);

      expect(getStore().tasks[taskId]).toBeUndefined();
    });

    it("cleans both sidecar maps", () => {
      const file = new File(["data"], "clean.txt");
      const { taskId } = getStore().addTask({
        type: "upload",
        filename: "clean.txt",
        totalBytes: 4,
        sourceFile: file,
      });

      getStore().removeTask(taskId);

      expect(getStore().getSourceFile(taskId)).toBeUndefined();
      expect(getStore().getSignal(taskId)).toBeUndefined();
    });
  });

  // ===================== clearCompleted =====================

  describe("clearCompleted", () => {
    it("removes only completed tasks", () => {
      const { taskId: completedId } = getStore().addTask({
        type: "upload",
        filename: "done.txt",
        totalBytes: 10,
      });
      const { taskId: pendingId } = getStore().addTask({
        type: "upload",
        filename: "pending.txt",
        totalBytes: 10,
      });
      const { taskId: failedId } = getStore().addTask({
        type: "upload",
        filename: "failed.txt",
        totalBytes: 10,
      });

      getStore().completeTask(completedId);
      getStore().failTask(failedId, "error");

      getStore().clearCompleted();

      expect(getStore().tasks[completedId]).toBeUndefined();
      expect(getStore().tasks[pendingId]).toBeDefined();
      expect(getStore().tasks[failedId]).toBeDefined();
    });
  });

  // ===================== _reset =====================

  describe("_reset", () => {
    it("clears all tasks", () => {
      getStore().addTask({ type: "upload", filename: "a.txt", totalBytes: 1 });
      getStore().addTask({ type: "download", filename: "b.bin", totalBytes: 2 });

      getStore()._reset();

      expect(Object.keys(getStore().tasks)).toHaveLength(0);
    });

    it("aborts active controllers", () => {
      const { signal } = getStore().addTask({
        type: "upload",
        filename: "abort-me.txt",
        totalBytes: 10,
      });

      expect(signal.aborted).toBe(false);

      getStore()._reset();

      expect(signal.aborted).toBe(true);
    });

    it("clears sidecar (getSourceFile returns undefined)", () => {
      const file = new File(["data"], "sidecar.txt");
      const { taskId } = getStore().addTask({
        type: "upload",
        filename: "sidecar.txt",
        totalBytes: 4,
        sourceFile: file,
      });

      getStore()._reset();

      expect(getStore().getSourceFile(taskId)).toBeUndefined();
    });
  });

  // ===================== Selectors =====================

  describe("selectors", () => {
    it("selectActiveTasks: filters pending+transferring", () => {
      const { taskId: pendingId } = getStore().addTask({
        type: "upload",
        filename: "p.txt",
        totalBytes: 10,
      });
      const { taskId: transferringId } = getStore().addTask({
        type: "download",
        filename: "t.bin",
        totalBytes: 100,
      });
      const { taskId: doneId } = getStore().addTask({
        type: "upload",
        filename: "d.txt",
        totalBytes: 5,
      });

      getStore().updateProgress(transferringId, 10, 100);
      getStore().completeTask(doneId);

      const active = selectActiveTasks(getStore());
      expect(active.map((t) => t.id)).toContain(pendingId);
      expect(active.map((t) => t.id)).toContain(transferringId);
      expect(active.map((t) => t.id)).not.toContain(doneId);
    });

    it("selectHasActiveUploads(sessionId): filters by type+sessionId", () => {
      getStore().addTask({
        type: "upload",
        filename: "up.txt",
        totalBytes: 10,
        sessionId: "sess-A",
      });
      getStore().addTask({
        type: "download",
        filename: "down.bin",
        totalBytes: 10,
        sessionId: "sess-A",
      });

      const hasUploads = selectHasActiveUploads("sess-A")(getStore());
      const hasUploadsB = selectHasActiveUploads("sess-B")(getStore());

      expect(hasUploads).toBe(true);
      expect(hasUploadsB).toBe(false);
    });

    it("selectCompletedUploadResults(sessionId): returns {taskId, fileInfo}[], different session returns empty", () => {
      const fileInfo = {
        id: "fi-99",
        filename: "result.pdf",
        filepath: "/files/result.pdf",
        key: "files/result.pdf",
        extension: "pdf",
        mime_type: "application/pdf",
        size: 1024,
      };

      const { taskId } = getStore().addTask({
        type: "upload",
        filename: "result.pdf",
        totalBytes: 1024,
        sessionId: "sess-X",
      });
      getStore().completeTask(taskId, fileInfo);

      const results = selectCompletedUploadResults("sess-X")(getStore());
      expect(results).toHaveLength(1);
      expect(results[0].taskId).toBe(taskId);
      expect(results[0].fileInfo).toEqual(fileInfo);

      const otherResults = selectCompletedUploadResults("sess-Y")(getStore());
      expect(otherResults).toHaveLength(0);
    });
  });
});
