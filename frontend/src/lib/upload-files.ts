export const SUPPORTED_UPLOAD_ACCEPT =
  ".pdf,.docx,application/pdf,application/vnd.openxmlformats-officedocument.wordprocessingml.document";

export function isSupportedUploadFile(file: File): boolean {
  const filename = file.name.trim().toLowerCase();
  return filename.endsWith(".pdf") || filename.endsWith(".docx");
}

export function splitSupportedUploadFiles(files: File[]): {
  accepted: File[];
  rejected: File[];
} {
  return files.reduce(
    (result, file) => {
      if (isSupportedUploadFile(file)) {
        result.accepted.push(file);
      } else {
        result.rejected.push(file);
      }
      return result;
    },
    { accepted: [] as File[], rejected: [] as File[] },
  );
}

export function unsupportedUploadFilesMessage(files: File[]): string {
  if (files.length === 0) {
    return "";
  }

  const names = files.map((file) => file.name || "tệp không tên").join(", ");
  return `Chỉ hỗ trợ tệp PDF hoặc DOCX. Đã bỏ qua: ${names}`;
}
