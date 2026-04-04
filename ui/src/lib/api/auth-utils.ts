import { useAuthStore } from "@/lib/store/auth-store";

export const API_BASE_URL =
  process.env.NEXT_PUBLIC_API_BASE_URL || "http://localhost:8000/api";

export class ApiError extends Error {
  code: number;
  httpStatus: number;
  data: unknown;
  retryAfter?: number;
  limit?: number;
  bucket?: string;
  windowSeconds?: number;

  constructor(params: {
    code: number;
    httpStatus: number;
    msg: string;
    data?: unknown;
    retryAfter?: number;
    limit?: number;
    bucket?: string;
    windowSeconds?: number;
  }) {
    super(params.msg);
    this.name = "ApiError";
    this.code = params.code;
    this.httpStatus = params.httpStatus;
    this.data = params.data ?? null;
    this.retryAfter = params.retryAfter;
    this.limit = params.limit;
    this.bucket = params.bucket;
    this.windowSeconds = params.windowSeconds;
  }
}

export function getAccessToken(): string | null {
  return useAuthStore.getState().accessToken;
}

export function handleLogout(): void {
  useAuthStore.getState().logout();
}

let refreshPromise: Promise<boolean> | null = null;

export async function maybeRefreshToken(): Promise<boolean> {
  if (!refreshPromise) {
    refreshPromise = useAuthStore
      .getState()
      .refresh()
      .finally(() => {
        refreshPromise = null;
      });
  }
  return refreshPromise;
}
