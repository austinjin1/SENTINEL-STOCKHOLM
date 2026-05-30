import { useEffect, useRef, useState, useCallback } from 'react';
import { UploadCloud } from 'lucide-react';
import { buildRecordFromCsv } from '../engine/importCsv';
import type { ParsedCsv } from '../engine/csvParse';
import CsvWorker from '../engine/csvParser.worker.ts?worker';
import type { WorkerRequest, WorkerResponse } from '../engine/csvParser.worker';
import type { Action } from '../state';
import { useToasts } from './Toasts';
import { clientToLngLat } from './mapRef';

interface Props {
  dispatch: React.Dispatch<Action>;
}

export function ImportDropZone({ dispatch }: Props) {
  const [hover, setHover] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const dragDepth = useRef(0);
  const workerRef = useRef<Worker | null>(null);
  const { push: pushToast } = useToasts();

  // Lazy-instantiate the worker on first import; tear it down on unmount.
  useEffect(() => {
    return () => {
      workerRef.current?.terminate();
      workerRef.current = null;
    };
  }, []);

  const parseInWorker = useCallback((req: WorkerRequest): Promise<ParsedCsv> => {
    if (!workerRef.current) workerRef.current = new CsvWorker();
    const worker = workerRef.current;
    return new Promise((resolve, reject) => {
      const onMessage = (ev: MessageEvent<WorkerResponse>) => {
        worker.removeEventListener('message', onMessage);
        if (ev.data.ok && ev.data.parsed) resolve(ev.data.parsed);
        else reject(new Error(ev.data.error ?? 'CSV parse failed.'));
      };
      worker.addEventListener('message', onMessage);
      worker.postMessage(req);
    });
  }, []);

  const handleFile = useCallback(
    async (file: File, point?: { lat: number; lon: number }) => {
      try {
        const text = await file.text();
        // Default to a point in middle CONUS for click-imports; the user can move it later.
        const lat = point?.lat ?? 39.5;
        const lon = point?.lon ?? -98;
        const parsed = await parseInWorker({
          text,
          opts: { lat, lon, name: file.name.replace(/\.csv$/i, '') },
        });
        const rec = buildRecordFromCsv(parsed);
        dispatch({ type: 'SELECT_SITE', record: rec });
        setError(null);
        pushToast('success', `Imported ${parsed.rows} rows from ${file.name}`, 2400);
      } catch (e) {
        const msg = e instanceof Error ? e.message : String(e);
        setError(msg);
        pushToast('error', `Import failed: ${msg}`);
      }
    },
    [dispatch, parseInWorker, pushToast],
  );

  return (
    <div
      onDragEnter={(e) => {
        e.preventDefault();
        dragDepth.current += 1;
        if (e.dataTransfer?.types?.includes('Files')) setHover(true);
      }}
      onDragOver={(e) => e.preventDefault()}
      onDragLeave={() => {
        dragDepth.current -= 1;
        if (dragDepth.current <= 0) {
          dragDepth.current = 0;
          setHover(false);
        }
      }}
      onDrop={(e) => {
        e.preventDefault();
        dragDepth.current = 0;
        setHover(false);
        const file = e.dataTransfer.files?.[0];
        if (file) {
          // If the drop landed over the map, use that location instead of the CONUS default.
          const point = clientToLngLat(e.clientX, e.clientY);
          void handleFile(file, point ?? undefined);
        }
      }}
      style={{
        position: 'fixed',
        inset: 0,
        zIndex: hover ? 1000 : -1,
        pointerEvents: hover ? 'auto' : 'none',
      }}
    >
      {hover && (
        <div
          style={{
            position: 'absolute',
            inset: 0,
            background: 'rgba(0,113,227,0.08)',
            backdropFilter: 'blur(6px)',
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'center',
            animation: 'fade-in var(--dur-fast)',
          }}
        >
          <div
            style={{
              padding: '32px 40px',
              background: 'rgba(255,255,255,0.96)',
              borderRadius: 'var(--r-lg)',
              border: '2px dashed var(--accent)',
              boxShadow: 'var(--shadow-lg)',
              textAlign: 'center',
              maxWidth: 420,
            }}
          >
            <UploadCloud
              size={36}
              strokeWidth={1.5}
              color="var(--accent)"
              style={{ marginBottom: 12 }}
            />
            <div style={{ fontSize: 18, fontWeight: 600, marginBottom: 4 }}>
              Drop a sensor CSV
            </div>
            <div style={{ fontSize: 13, color: 'var(--text-secondary)' }}>
              Columns: timestamp, DO, pH, turbidity, conductivity, temperature, ORP. Missing
              columns are inferred from regional baseline.
            </div>
          </div>
        </div>
      )}
      {error && (
        <div
          style={{
            position: 'fixed',
            top: 80,
            left: '50%',
            transform: 'translateX(-50%)',
            padding: '10px 16px',
            background: 'var(--alert-dim)',
            color: 'var(--alert)',
            border: '1px solid var(--alert)',
            borderRadius: 'var(--r-md)',
            fontSize: 13,
            zIndex: 1100,
            maxWidth: 480,
            pointerEvents: 'auto',
            animation: 'fade-in-up var(--dur-base)',
          }}
        >
          <strong>Import failed.</strong> {error}{' '}
          <button onClick={() => setError(null)} style={{ marginLeft: 8, textDecoration: 'underline' }}>
            Dismiss
          </button>
        </div>
      )}
    </div>
  );
}
