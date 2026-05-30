/// <reference lib="webworker" />
import { parseCsv, type ParsedCsv } from './csvParse';

export interface WorkerRequest {
  text: string;
  opts: { lat: number; lon: number; name: string };
}

export interface WorkerResponse {
  ok: boolean;
  parsed?: ParsedCsv;
  error?: string;
}

self.onmessage = (e: MessageEvent<WorkerRequest>) => {
  try {
    const parsed = parseCsv(e.data.text, e.data.opts);
    (self as DedicatedWorkerGlobalScope).postMessage({ ok: true, parsed } satisfies WorkerResponse);
  } catch (err) {
    (self as DedicatedWorkerGlobalScope).postMessage({
      ok: false,
      error: err instanceof Error ? err.message : String(err),
    } satisfies WorkerResponse);
  }
};
