export const SUPPORTED_UPLOAD_ACCEPT =
  ".pdf,.docx,application/pdf,application/vnd.openxmlformats-officedocument.wordprocessingml.document";
export const MAX_UPLOAD_FILE_SIZE_BYTES = 50 * 1024 * 1024;
export const UPLOAD_FILE_SIZE_ERROR = "Tệp vượt quá kích thước tối đa 50MB.";
export const UPLOAD_FILE_TYPE_ERROR = "Định dạng không được hỗ trợ. Chỉ chấp nhận PDF hoặc DOCX.";

export function isSupportedUploadFile(file: File): boolean {
  const filename = file.name.trim().toLowerCase();
  return filename.endsWith(".pdf") || filename.endsWith(".docx");
}

export function getUploadFileValidationError(file: File): string | null {
  if (!isSupportedUploadFile(file)) {
    return UPLOAD_FILE_TYPE_ERROR;
  }
  if (file.size > MAX_UPLOAD_FILE_SIZE_BYTES) {
    return UPLOAD_FILE_SIZE_ERROR;
  }
  return null;
}

export function splitValidUploadFiles(files: File[]): {
  accepted: File[];
  rejected: Array<{ file: File; message: string }>;
} {
  return files.reduce(
    (result, file) => {
      const error = getUploadFileValidationError(file);
      if (!error) {
        result.accepted.push(file);
      } else {
        result.rejected.push({ file, message: error });
      }
      return result;
    },
    { accepted: [] as File[], rejected: [] as Array<{ file: File; message: string }> },
  );
}

export function formatUploadValidationMessages(
  rejected: Array<{ file: File; message: string }>,
): string[] {
  return rejected.map(({ file, message }) => {
    const filename = file.name || "Tệp không tên";
    return `${filename}: ${message}`;
  });
}
