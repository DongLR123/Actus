import { beforeEach, describe, expect, it, vi } from "vitest";

const mockGetAccessToken = vi.fn(() => "mock-token");
const mockHandleLogout = vi.fn();
const mockMaybeRefreshToken = vi.fn(async () => true);

vi.mock("./auth-utils", () => ({
  API_BASE_URL: "http://localhost:8000/api",
  ApiError: class ApiError extends Error {
    code: number;
    httpStatus: number;
    data: unknown;
    constructor(params: { code: number; httpStatus: number; msg: string; data?: unknown }) {
      super(params.msg);
      this.name = "ApiError";
      this.code = params.code;
      this.httpStatus = params.httpStatus;
      this.data = params.data ?? null;
    }
  },
  getAccessToken: mockGetAccessToken,
  handleLogout: mockHandleLogout,
  maybeRefreshToken: mockMaybeRefreshToken,
}));

// Import after mocks are set up
const { fileTransferClient } = await import("./axios-client");

describe("fileTransferClient", () => {
  describe("defaults", () => {
    it("has baseURL set", () => {
      expect(fileTransferClient.defaults.baseURL).toBe("http://localhost:8000/api");
    });

    it("has timeout 0", () => {
      expect(fileTransferClient.defaults.timeout).toBe(0);
    });
  });

  describe("request interceptor", () => {
    it("injects Authorization header on requests", async () => {
      mockGetAccessToken.mockReturnValue("mock-token");

      let capturedAuth: string | undefined;
      const originalAdapter = fileTransferClient.defaults.adapter;
      fileTransferClient.defaults.adapter = (config) => {
        capturedAuth = config.headers?.Authorization as string | undefined;
        return Promise.reject({ config, response: { status: 0 } });
      };
      try {
        await fileTransferClient.get("/test").catch(() => {});
      } finally {
        fileTransferClient.defaults.adapter = originalAdapter;
      }
      expect(capturedAuth).toBe("Bearer mock-token");
    });

    it("does not inject Authorization header when no token", async () => {
      mockGetAccessToken.mockReturnValue(null);

      let capturedAuth: string | undefined;
      const originalAdapter = fileTransferClient.defaults.adapter;
      fileTransferClient.defaults.adapter = (config) => {
        capturedAuth = config.headers?.Authorization as string | undefined;
        return Promise.reject({ config, response: { status: 0 } });
      };
      try {
        await fileTransferClient.get("/test").catch(() => {});
      } finally {
        fileTransferClient.defaults.adapter = originalAdapter;
        mockGetAccessToken.mockReturnValue("mock-token");
      }
      expect(capturedAuth).toBeUndefined();
    });
  });

  describe("response interceptor: 401 handling", () => {
    beforeEach(() => {
      mockGetAccessToken.mockReturnValue("mock-token");
      mockHandleLogout.mockReset();
      mockMaybeRefreshToken.mockReset().mockResolvedValue(true);
    });

    it("retries original request with fresh token after successful refresh", async () => {
      let callCount = 0;
      const originalAdapter = fileTransferClient.defaults.adapter;
      fileTransferClient.defaults.adapter = (config) => {
        callCount++;
        if (callCount === 1) {
          // First call: simulate 401
          return Promise.reject({
            config,
            response: { status: 401, data: { msg: "Unauthorized" } },
            isAxiosError: true,
            message: "Request failed with status code 401",
          });
        }
        // Second call (retry): succeed
        return Promise.resolve({
          data: { success: true },
          status: 200,
          statusText: "OK",
          headers: {},
          config,
        });
      };

      try {
        const result = await fileTransferClient.get("/protected");
        expect(result.data).toEqual({ success: true });
        expect(mockMaybeRefreshToken).toHaveBeenCalledTimes(1);
        expect(callCount).toBe(2);
      } finally {
        fileTransferClient.defaults.adapter = originalAdapter;
      }
    });

    it("calls handleLogout when refresh fails", async () => {
      mockMaybeRefreshToken.mockResolvedValue(false);

      const originalAdapter = fileTransferClient.defaults.adapter;
      fileTransferClient.defaults.adapter = (config) => {
        return Promise.reject({
          config,
          response: { status: 401, data: { msg: "Unauthorized" } },
          isAxiosError: true,
          message: "Request failed with status code 401",
        });
      };

      try {
        await expect(fileTransferClient.get("/protected")).rejects.toBeDefined();
        expect(mockHandleLogout).toHaveBeenCalledTimes(1);
      } finally {
        fileTransferClient.defaults.adapter = originalAdapter;
      }
    });

    it("does not retry /auth/ endpoints on 401", async () => {
      const originalAdapter = fileTransferClient.defaults.adapter;
      fileTransferClient.defaults.adapter = (config) => {
        return Promise.reject({
          config,
          response: { status: 401, data: { msg: "Unauthorized" } },
          isAxiosError: true,
          message: "Request failed with status code 401",
        });
      };

      try {
        await expect(fileTransferClient.get("/auth/login")).rejects.toBeDefined();
        expect(mockMaybeRefreshToken).not.toHaveBeenCalled();
      } finally {
        fileTransferClient.defaults.adapter = originalAdapter;
      }
    });
  });
});
