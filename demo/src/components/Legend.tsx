const TIERS: { label: string; sub: string; color: string }[] = [
  { label: 'Critical', sub: '> 0.70', color: '#c0362c' },
  { label: 'High', sub: '0.55 – 0.70', color: '#b8430f' },
  { label: 'Elevated', sub: '0.40 – 0.55', color: '#b06e1a' },
  { label: 'Moderate', sub: '0.25 – 0.40', color: '#86868b' },
  { label: 'Low', sub: '≤ 0.25', color: '#2e8540' },
];

export function Legend() {
  return (
    <div
      style={{
        position: 'absolute',
        bottom: 20,
        right: 20,
        width: 240,
        background: 'rgba(255,255,255,0.82)',
        backdropFilter: 'blur(20px) saturate(180%)',
        WebkitBackdropFilter: 'blur(20px) saturate(180%)',
        borderRadius: 'var(--r-md)',
        padding: 16,
        boxShadow: 'var(--shadow-md)',
        animation: 'fade-in-up var(--dur-base) var(--ease-entrance)',
        zIndex: 'var(--z-map-overlay)' as any,
      }}
    >
      <div style={{ fontSize: 12, fontWeight: 600, color: 'var(--text-primary)', marginBottom: 12 }}>Risk tier</div>
      {TIERS.map((t) => (
        <div
          key={t.label}
          style={{
            display: 'flex',
            alignItems: 'center',
            gap: 10,
            fontSize: 13,
            color: 'var(--text-secondary)',
            marginBottom: 8,
          }}
        >
          <span style={{ width: 10, height: 10, background: t.color, borderRadius: 5, border: '1px solid white' }} />
          <span style={{ color: 'var(--text-primary)', flex: 1 }}>{t.label}</span>
          <span className="mono" style={{ color: 'var(--text-tertiary)', fontSize: 12 }}>{t.sub}</span>
        </div>
      ))}
    </div>
  );
}
