import { createContext, useCallback, useContext, useRef, useState } from 'react';

export type ToastKind = 'info' | 'error' | 'success';
export interface Toast {
  id: number;
  kind: ToastKind;
  text: string;
}

interface Ctx {
  toasts: Toast[];
  push: (kind: ToastKind, text: string, ttlMs?: number) => void;
  dismiss: (id: number) => void;
}

const ToastCtx = createContext<Ctx | null>(null);

export function ToastProvider({ children }: { children: React.ReactNode }) {
  const [toasts, setToasts] = useState<Toast[]>([]);
  const nextId = useRef(1);

  const dismiss = useCallback((id: number) => {
    setToasts((t) => t.filter((x) => x.id !== id));
  }, []);

  const push = useCallback(
    (kind: ToastKind, text: string, ttlMs = 4000) => {
      const id = nextId.current++;
      setToasts((t) => [...t, { id, kind, text }]);
      window.setTimeout(() => dismiss(id), ttlMs);
    },
    [dismiss],
  );

  return (
    <ToastCtx.Provider value={{ toasts, push, dismiss }}>
      {children}
      <ToastLayer />
    </ToastCtx.Provider>
  );
}

export function useToasts() {
  const ctx = useContext(ToastCtx);
  if (!ctx) throw new Error('useToasts must be used inside <ToastProvider>');
  return ctx;
}

const KIND_COLORS: Record<ToastKind, { bg: string; fg: string; border: string }> = {
  info: { bg: 'rgba(255,255,255,0.96)', fg: 'var(--text-primary)', border: 'var(--border-subtle)' },
  success: { bg: 'rgba(235,250,238,0.96)', fg: 'var(--nominal, #2e8540)', border: 'var(--nominal, #2e8540)' },
  error: { bg: 'rgba(255,235,235,0.96)', fg: 'var(--alert, #c0362c)', border: 'var(--alert, #c0362c)' },
};

function ToastLayer() {
  const { toasts, dismiss } = useToasts();
  return (
    <div
      style={{
        position: 'fixed',
        top: 80,
        left: '50%',
        transform: 'translateX(-50%)',
        display: 'flex',
        flexDirection: 'column',
        gap: 8,
        zIndex: 9999,
        pointerEvents: 'none',
      }}
    >
      {toasts.map((t) => {
        const c = KIND_COLORS[t.kind];
        return (
          <div
            key={t.id}
            onClick={() => dismiss(t.id)}
            style={{
              padding: '8px 16px',
              background: c.bg,
              border: `1px solid ${c.border}`,
              borderRadius: 'var(--r-pill)',
              boxShadow: 'var(--shadow-md)',
              fontSize: 13,
              color: c.fg,
              pointerEvents: 'auto',
              cursor: 'pointer',
              animation: 'fade-in var(--dur-fast)',
              maxWidth: 480,
            }}
          >
            {t.text}
          </div>
        );
      })}
    </div>
  );
}
