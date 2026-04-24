import type { ApiQueryRequest, ApiQueryResponse } from "@/api/types";

const API_BASE_URL = (import.meta.env.VITE_API_BASE_URL || "/api/v1").replace(/\/$/, "");

async function parseError(response: Response): Promise<string> {
  try {
    const payload = (await response.json()) as { detail?: string };
    if (payload.detail) {
      return payload.detail;
    }
  } catch {
    return response.statusText || "Request failed";
  }
  return response.statusText || "Request failed";
}

export async function postQuery(request: ApiQueryRequest): Promise<ApiQueryResponse> {
  const response = await fetch(`${API_BASE_URL}/query`, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
    },
    body: JSON.stringify(request),
  });

  if (!response.ok) {
    const message = await parseError(response);
    throw new Error(`API error (${response.status}): ${message}`);
  }

  return (await response.json()) as ApiQueryResponse;
}
