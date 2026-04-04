import { API_BASE_URL } from "./auth-utils";
import { fileTransferClient } from "./axios-client";
import type { ApiResponse, FileInfo, FileUploadParams } from "./types";

interface TransferOptions {
  onProgress?: (loaded: number, total: number) => void;
  signal?: AbortSignal;
}

export const fileApi = {
  uploadFile: async (
    params: FileUploadParams & TransferOptions
  ): Promise<FileInfo> => {
    const { file, session_id, onProgress, signal } = params;
    const formData = new FormData();
    formData.append("file", file);

    if (session_id) {
      formData.append("session_id", session_id);
    }

    const response = await fileTransferClient.post<ApiResponse<FileInfo>>(
      "/files",
      formData,
      {
        signal,
        onUploadProgress: onProgress
          ? (event) => {
              const loaded = event.loaded;
              const total = event.total ?? 0;
              onProgress(loaded, total);
            }
          : undefined,
      }
    );

    return response.data.data as FileInfo;
  },

  getFileInfo: async (fileId: string): Promise<FileInfo> => {
    const response = await fileTransferClient.get<ApiResponse<FileInfo>>(
      `/files/${fileId}`,
      { timeout: 30_000 }
    );
    return response.data.data as FileInfo;
  },

  downloadFile: async (
    fileId: string,
    options?: TransferOptions
  ): Promise<Blob> => {
    const response = await fileTransferClient.get(
      `/files/${fileId}/download`,
      {
        responseType: "blob",
        signal: options?.signal,
        onDownloadProgress: options?.onProgress
          ? (event) => {
              const loaded = event.loaded;
              const total = event.total ?? 0;
              options.onProgress!(loaded, total);
            }
          : undefined,
      }
    );
    return response.data as Blob;
  },

  getFileDownloadUrl: (fileId: string): string => {
    return `${API_BASE_URL}/files/${fileId}/download`;
  },
};
