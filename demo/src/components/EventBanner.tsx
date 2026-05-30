import { AlertTriangle, ShieldCheck, Clock } from 'lucide-react';
import type { EventTimeline } from '../data/eventTimelines';
import { daysBetween } from '../engine/timeSample';

interface Props {
  timeline: EventTimeline;
  currentDate: string;
}

// Slim ribbon at the top of PlaceCard. Has three states:
//   - pre-alert: muted gray, "no SENTINEL signal yet"
//   - alert window: green, "SENTINEL alert active · N days before advisory"
//   - post-advisory: red, "Official advisory issued · {date}"
export function EventBanner({ timeline, currentDate }: Props) {
  const beforeAlert = currentDate < timeline.sentinelAlertDate;
  const inAlertWindow =
    currentDate >= timeline.sentinelAlertDate && currentDate < timeline.advisoryDate;

  const daysToAdvisory = daysBetween(currentDate, timeline.advisoryDate);
  const daysFromAdvisory = daysBetween(timeline.advisoryDate, currentDate);

  const palette = beforeAlert
    ? { bg: 'rgba(134,134,139,0.15)', fg: 'var(--text-secondary)', icon: <Clock size={13} strokeWidth={2} /> }
    : inAlertWindow
      ? { bg: 'rgba(46,133,64,0.18)', fg: 'var(--nominal, #2e8540)', icon: <ShieldCheck size={13} strokeWidth={2} /> }
      : { bg: 'rgba(192,54,44,0.18)', fg: 'var(--alert, #c0362c)', icon: <AlertTriangle size={13} strokeWidth={2} /> };

  const label = beforeAlert
    ? 'Baseline · no SENTINEL signal yet'
    : inAlertWindow
      ? `SENTINEL alert active · ${daysToAdvisory}d before official advisory`
      : `Official advisory issued · T+${daysFromAdvisory}d`;

  return (
    <div
      style={{
        display: 'flex',
        alignItems: 'center',
        gap: 8,
        padding: '8px 16px',
        background: palette.bg,
        color: palette.fg,
        fontSize: 12,
        fontWeight: 600,
        letterSpacing: '0.02em',
        borderBottom: `1px solid ${palette.fg}`,
        transition: 'background 220ms, color 220ms',
        animation: inAlertWindow
          ? 'alert-pulse 2.4s ease-in-out infinite'
          : !beforeAlert
            ? 'advisory-pulse 2.4s ease-in-out infinite'
            : 'none',
      }}
    >
      {palette.icon}
      <span>{label}</span>
    </div>
  );
}
