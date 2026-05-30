export function ClassificationBanner() {
  return (
    <div
      style={{
        height: 22,
        background: 'rgba(255,255,255,0.04)',
        borderBottom: '1px solid var(--border-subtle)',
        color: 'var(--text-tertiary)',
        fontSize: 11,
        lineHeight: '22px',
        textAlign: 'center',
        flexShrink: 0,
        letterSpacing: 0,
      }}
    >
      Demonstration Environment · Cached and Modeled Data · Not for Operational Use
    </div>
  );
}
