import { describe, expect, it, vi } from "vitest";

import { API_BASE_URL, ApiError, getAccessToken, handleLogout } from "./auth-utils";

const mockLogout = vi.fn();
const mockRefresh = vi.fn(async () => true);
const stableState = { accessToken: "test-token", refresh: mockRefresh, logout: mockLogout };
vi.mock("@/lib/store/auth-store", () => ({ useAuthStore: { getState: () => stableState } }));

describe("auth-utils", () => {
  it("API_BASE_URL defaults to http://localhost:8000/api", () => {
    expect(API_BASE_URL).toBe("http://localhost:8000/api");
  });

  it("ApiError constructs correctly with all fields", () => {
    const err = new ApiError({
      code: 429,
      httpStatus: 429,
      msg: "Too many requests",
      data: { detail: "rate limited" },
      retryAfter: 60,
      limit: 100,
      bucket: "default",
      windowSeconds: 3600,
    });

    expect(err.code).toBe(429);
    expect(err.httpStatus).toBe(429);
    expect(err.message).toBe("Too many requests");
    expect(err.data).toEqual({ detail: "rate limited" });
    expect(err.retryAfter).toBe(60);
    expect(err.limit).toBe(100);
    expect(err.bucket).toBe("default");
    expect(err.windowSeconds).toBe(3600);
    expect(err.name).toBe("ApiError");
  });

  it("ApiError is instanceof Error", () => {
    const err = new ApiError({ code: 500, httpStatus: 500, msg: "Internal error" });
    expect(err).toBeInstanceOf(Error);
    expect(err).toBeInstanceOf(ApiError);
  });

  it("getAccessToken returns token from mock store", () => {
    const token = getAccessToken();
    expect(token).toBe("test-token");
  });

  it("handleLogout calls mock logout", () => {
    handleLogout();
    expect(mockLogout).toHaveBeenCalledTimes(1);
  });
});
