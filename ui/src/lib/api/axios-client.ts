import axios, { type AxiosError, type InternalAxiosRequestConfig } from "axios";
import { API_BASE_URL, ApiError, getAccessToken, handleLogout, maybeRefreshToken } from "./auth-utils";

export const fileTransferClient = axios.create({
  baseURL: API_BASE_URL,
  timeout: 0, // no timeout for large files
});

// Request interceptor: inject auth token
fileTransferClient.interceptors.request.use((config) => {
  const token = getAccessToken();
  if (token) {
    config.headers.Authorization = `Bearer ${token}`;
  }
  return config;
});

// Response interceptor: 401 refresh + error normalization
// CRITICAL: The pending queue stores EACH request's own config, not the original request.
// This is required because concurrent uploads/downloads may all hit 401 simultaneously.

let isRefreshing = false;
let pendingRequests: Array<{
  config: InternalAxiosRequestConfig;
  resolve: (value: unknown) => void;
  reject: (error: unknown) => void;
}> = [];

fileTransferClient.interceptors.response.use(
  (response) => response,
  async (error: AxiosError) => {
    // Propagate CanceledError as-is before any other handling
    if (axios.isCancel(error)) {
      return Promise.reject(error);
    }

    const originalRequest = error.config;
    if (!originalRequest || error.response?.status !== 401) {
      return Promise.reject(toApiError(error));
    }
    if (originalRequest.url?.startsWith("/auth/")) {
      return Promise.reject(toApiError(error));
    }
    if ((originalRequest as InternalAxiosRequestConfig & { _retried?: boolean })._retried) {
      handleLogout();
      return Promise.reject(toApiError(error));
    }

    if (isRefreshing) {
      // Queue this request WITH ITS OWN config
      return new Promise((resolve, reject) => {
        pendingRequests.push({ config: originalRequest, resolve, reject });
      });
    }

    isRefreshing = true;
    try {
      const refreshed = await maybeRefreshToken();
      if (!refreshed) {
        handleLogout();
        pendingRequests.forEach(({ reject }) => reject(toApiError(error)));
        pendingRequests = [];
        return Promise.reject(toApiError(error));
      }

      const token = getAccessToken();
      // Retry each queued request with its own config + fresh token
      pendingRequests.forEach(({ config, resolve, reject }) => {
        (config as InternalAxiosRequestConfig & { _retried?: boolean })._retried = true;
        if (token) config.headers.Authorization = `Bearer ${token}`;
        fileTransferClient(config).then(resolve, reject);
      });
      pendingRequests = [];

      // Retry the original request
      (originalRequest as InternalAxiosRequestConfig & { _retried?: boolean })._retried = true;
      if (token) originalRequest.headers.Authorization = `Bearer ${token}`;
      return fileTransferClient(originalRequest);
    } finally {
      isRefreshing = false;
    }
  }
);

function toApiError(error: AxiosError): ApiError {
  const status = error.response?.status ?? 500;
  const data = error.response?.data;
  let msg = "网络连接失败";
  if (typeof data === "object" && data !== null) {
    const record = data as Record<string, unknown>;
    if (typeof record.msg === "string") msg = record.msg;
    else if (typeof record.detail === "string") msg = record.detail;
  } else if (error.message) {
    msg = error.message;
  }
  return new ApiError({ code: status, httpStatus: status, msg, data });
}
