import { useState, useMemo } from 'react'
import { MapContainer, TileLayer, CircleMarker, Popup } from 'react-leaflet'
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, BarChart, Bar } from 'recharts'
import './App.css'

import animasData from './data/demo_animas_river.json'
import erieData from './data/demo_lake_erie.json'
import sewageData from './data/demo_sewage_overflow.json'
import type { CaseStudy, AlertLevel } from './utils/types'
import { ALERT_COLORS } from './utils/types'

const CASE_STUDIES: CaseStudy[] = [animasData, erieData, sewageData] as CaseStudy[]

const MODALITIES = ['sensor', 'satellite', 'microbial', 'molecular', 'behavioral'] as const

function getHealthClass(score: number): string {
  if (score >= 70) return 'good'
  if (score >= 40) return 'warning'
  return 'danger'
}

function getAlertColor(level: AlertLevel): string {
  return ALERT_COLORS[level] || '#94a3b8'
}

function Header({ view, setView }: { view: string; setView: (v: string) => void }) {
  return (
    <header className="header">
      <div className="header-brand">
        <h1>SENTINEL</h1>
        <span>Multimodal Water Quality Intelligence</span>
      </div>
      <nav className="header-nav">
        <button className={`nav-btn ${view === 'map' ? 'active' : ''}`} onClick={() => setView('map')}>
          Map
        </button>
        <button className={`nav-btn ${view === 'cases' ? 'active' : ''}`} onClick={() => setView('cases')}>
          Case Studies
        </button>
        <button className={`nav-btn ${view === 'models' ? 'active' : ''}`} onClick={() => setView('models')}>
          Models
        </button>
      </nav>
    </header>
  )
}

function Sidebar({ cases, selected, onSelect }: {
  cases: CaseStudy[]
  selected: string | null
  onSelect: (id: string) => void
}) {
  return (
    <aside className="sidebar">
      <h2>Monitoring Events</h2>
      {cases.map(c => (
        <div
          key={c.id}
          className={`case-card ${selected === c.id ? 'selected' : ''}`}
          onClick={() => onSelect(c.id)}
        >
          <h3>{c.name}</h3>
          <p>{c.description?.slice(0, 100)}...</p>
          <div>
            <span className={`badge ${c.eventType?.replace(/\s+/g, '-').toLowerCase()}`}>
              {c.eventType}
            </span>
            {c.leadTimeDays > 0 && (
              <span className="badge detected-early" style={{ marginLeft: '0.3rem' }}>
                {c.leadTimeDays}d early detection
              </span>
            )}
          </div>
        </div>
      ))}

      <h2 style={{ marginTop: '1.5rem' }}>System Status</h2>
      <div className="stats-grid">
        <div className="stat-card">
          <div className="label">Active Sites</div>
          <div className="value blue">3,247</div>
        </div>
        <div className="stat-card">
          <div className="label">Alerts (24h)</div>
          <div className="value yellow">12</div>
        </div>
        <div className="stat-card">
          <div className="label">Satellite Passes</div>
          <div className="value green">847</div>
        </div>
        <div className="stat-card">
          <div className="label">Citizen Reports</div>
          <div className="value blue">156</div>
        </div>
      </div>

      <h2>Active Modalities</h2>
      <div className="modality-pills">
        {MODALITIES.map(m => (
          <span key={m} className={`modality-pill ${m}`}>{m}</span>
        ))}
      </div>
    </aside>
  )
}

function DetailPanel({ study }: { study: CaseStudy }) {
  const sensorData = useMemo(() =>
    (study.sensorData || []).slice(0, 48).map((r, i) => ({
      time: i,
      do: r.dissolvedOxygen,
      ph: r.pH,
      turb: r.turbidity,
      temp: r.temperature,
    })),
    [study]
  )

  const anomalyData = useMemo(() =>
    (study.anomalyScores || []).slice(0, 48).map((a, i) => ({
      time: i,
      sensor: a.sensorAnomaly,
      satellite: a.satelliteAnomaly,
      microbial: a.microbialAnomaly,
      fused: a.fusedScore,
    })),
    [study]
  )

  const sourceData = useMemo(() =>
    (study.sourceAttributions || []).map(s => ({
      name: s.source,
      probability: Math.round(s.probability * 100),
    })),
    [study]
  )

  return (
    <div className="detail-panel">
      <h2>{study.name}</h2>

      {/* Health Score */}
      <div className="health-score">
        <div className={`health-ring ${getHealthClass(study.communityHealthScore || 50)}`}>
          {study.communityHealthScore || '--'}
        </div>
        <div className="health-label">
          <h3>Water Health Score</h3>
          <p>Composite index from all 5 modalities</p>
        </div>
      </div>

      {/* Key Metrics */}
      <div className="stats-grid">
        <div className="stat-card">
          <div className="label">SENTINEL Detection</div>
          <div className="value green">{study.sentinelDetectionDate?.slice(0, 10) || 'N/A'}</div>
        </div>
        <div className="stat-card">
          <div className="label">Official Response</div>
          <div className="value red">{study.authorityResponseDate?.slice(0, 10) || 'N/A'}</div>
        </div>
        <div className="stat-card">
          <div className="label">Lead Time</div>
          <div className="value green">{study.leadTimeDays || 0} days</div>
        </div>
        <div className="stat-card">
          <div className="label">Event Type</div>
          <div className="value yellow">{study.eventType || 'Unknown'}</div>
        </div>
      </div>

      {/* Anomaly Score Chart */}
      {anomalyData.length > 0 && (
        <div className="chart-container">
          <h4>Fusion Anomaly Scores</h4>
          <ResponsiveContainer width="100%" height={160}>
            <LineChart data={anomalyData}>
              <CartesianGrid strokeDasharray="3 3" />
              <XAxis dataKey="time" tick={{ fontSize: 10 }} />
              <YAxis domain={[0, 1]} tick={{ fontSize: 10 }} />
              <Tooltip contentStyle={{ background: '#1e293b', border: '1px solid #334155', borderRadius: '0.375rem', fontSize: '0.75rem' }} />
              <Line type="monotone" dataKey="fused" stroke="#ef4444" strokeWidth={2} dot={false} name="Fused" />
              <Line type="monotone" dataKey="sensor" stroke="#3b82f6" strokeWidth={1} dot={false} name="Sensor" />
              <Line type="monotone" dataKey="satellite" stroke="#8b5cf6" strokeWidth={1} dot={false} name="Satellite" />
              <Line type="monotone" dataKey="microbial" stroke="#22c55e" strokeWidth={1} dot={false} name="Microbial" />
            </LineChart>
          </ResponsiveContainer>
        </div>
      )}

      {/* Sensor Data Chart */}
      {sensorData.length > 0 && (
        <div className="chart-container">
          <h4>Sensor Time Series (AquaSSM)</h4>
          <ResponsiveContainer width="100%" height={160}>
            <LineChart data={sensorData}>
              <CartesianGrid strokeDasharray="3 3" />
              <XAxis dataKey="time" tick={{ fontSize: 10 }} />
              <YAxis tick={{ fontSize: 10 }} />
              <Tooltip contentStyle={{ background: '#1e293b', border: '1px solid #334155', borderRadius: '0.375rem', fontSize: '0.75rem' }} />
              <Line type="monotone" dataKey="do" stroke="#3b82f6" strokeWidth={1.5} dot={false} name="DO (mg/L)" />
              <Line type="monotone" dataKey="ph" stroke="#8b5cf6" strokeWidth={1.5} dot={false} name="pH" />
              <Line type="monotone" dataKey="turb" stroke="#6b7280" strokeWidth={1.5} dot={false} name="Turbidity" />
            </LineChart>
          </ResponsiveContainer>
        </div>
      )}

      {/* Source Attribution */}
      {sourceData.length > 0 && (
        <div className="chart-container">
          <h4>Source Attribution (SENTINEL-Fusion)</h4>
          <ResponsiveContainer width="100%" height={120}>
            <BarChart data={sourceData} layout="vertical">
              <CartesianGrid strokeDasharray="3 3" />
              <XAxis type="number" domain={[0, 100]} tick={{ fontSize: 10 }} />
              <YAxis type="category" dataKey="name" tick={{ fontSize: 10 }} width={100} />
              <Tooltip contentStyle={{ background: '#1e293b', border: '1px solid #334155', borderRadius: '0.375rem', fontSize: '0.75rem' }} />
              <Bar dataKey="probability" fill="#3b82f6" radius={[0, 4, 4, 0]} />
            </BarChart>
          </ResponsiveContainer>
        </div>
      )}

      {/* Escalation Timeline */}
      {(study.escalationHistory || []).length > 0 && (
        <>
          <h3>Escalation Timeline</h3>
          <div className="timeline">
            {study.escalationHistory.slice(0, 6).map((e, i) => (
              <div key={i} className={`timeline-event ${e.tier >= 3 ? 'alert' : ''}`}>
                <div className="time">{e.timestamp?.slice(0, 16)} - Tier {e.tier}</div>
                <div className="desc">{e.action}</div>
              </div>
            ))}
          </div>
        </>
      )}
    </div>
  )
}

function ModelsView() {
  const models = [
    { name: 'AquaSSM', type: 'Sensor Encoder', params: '4.6M', status: 'Training', metric: 'AUROC', value: '0.661', target: '0.85' },
    { name: 'HydroViT', type: 'Satellite Encoder', params: '40.8M', status: 'Training', metric: 'R\u00B2', value: '--', target: '0.55' },
    { name: 'MicroBiomeNet', type: 'Microbial Encoder', params: '128.9M', status: 'Training', metric: 'F1', value: '--', target: '0.70' },
    { name: 'ToxiGene', type: 'Molecular Encoder', params: '420K', status: 'Done', metric: 'F1', value: '0.894', target: '0.80' },
    { name: 'BioMotion', type: 'Behavioral Encoder', params: '3.0M', status: 'Done', metric: 'AUROC', value: '1.000', target: '0.80' },
    { name: 'SENTINEL-Fusion', type: 'Perceiver IO', params: '~10M', status: 'Pending', metric: 'AUROC', value: '--', target: '0.90' },
  ]

  return (
    <div style={{ flex: 1, padding: '2rem', overflow: 'auto' }}>
      <h2 style={{ marginBottom: '1rem' }}>Model Training Status</h2>
      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fill, minmax(300px, 1fr))', gap: '1rem' }}>
        {models.map(m => (
          <div key={m.name} className="stat-card" style={{ padding: '1rem' }}>
            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '0.5rem' }}>
              <div>
                <div style={{ fontSize: '1rem', fontWeight: 700 }}>{m.name}</div>
                <div style={{ fontSize: '0.75rem', color: 'var(--text-secondary)' }}>{m.type} | {m.params} params</div>
              </div>
              <span className={`badge ${m.status === 'Done' ? 'detected-early' : m.status === 'Training' ? 'nutrient' : 'sewage'}`}>
                {m.status}
              </span>
            </div>
            <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '0.8rem' }}>
              <span>{m.metric}: <strong>{m.value}</strong></span>
              <span style={{ color: 'var(--text-secondary)' }}>Target: {m.target}</span>
            </div>
          </div>
        ))}
      </div>

      <h2 style={{ marginTop: '2rem', marginBottom: '1rem' }}>Architecture Overview</h2>
      <div className="chart-container" style={{ padding: '1.5rem' }}>
        <p style={{ fontSize: '0.85rem', lineHeight: 1.6 }}>
          <strong>189.5M total parameters</strong> across 5 modality-specific encoders feeding into
          a Perceiver IO cross-modal temporal attention framework. Each encoder produces a 256-dimensional
          embedding that is fused through learned temporal decay attention with confidence-weighted gating.
        </p>
        <div style={{ display: 'flex', gap: '0.5rem', marginTop: '1rem', flexWrap: 'wrap' }}>
          {['AquaSSM (SSM)', 'HydroViT (ViT)', 'MicroBiomeNet (Aitchison)', 'ToxiGene (P-NET)', 'BioMotion (Diffusion)'].map(m => (
            <span key={m} style={{ padding: '0.3rem 0.6rem', background: 'var(--bg-primary)', borderRadius: '0.25rem', fontSize: '0.7rem', border: '1px solid var(--border)' }}>
              {m}
            </span>
          ))}
          <span style={{ padding: '0.3rem 0.6rem', fontSize: '0.7rem' }}>{'-->'}</span>
          <span style={{ padding: '0.3rem 0.6rem', background: 'rgba(59,130,246,0.2)', borderRadius: '0.25rem', fontSize: '0.7rem', border: '1px solid var(--accent-blue)', color: '#60a5fa' }}>
            Perceiver IO Fusion
          </span>
        </div>
      </div>
    </div>
  )
}

function App() {
  const [view, setView] = useState('cases')
  const [selectedCase, setSelectedCase] = useState<string | null>(CASE_STUDIES[0]?.id || null)

  const activeStudy = CASE_STUDIES.find(c => c.id === selectedCase) || CASE_STUDIES[0]

  const mapCenter: [number, number] = activeStudy?.site?.location
    ? [activeStudy.site.location.lat, activeStudy.site.location.lng]
    : [39.5, -98.0]

  return (
    <div className="app">
      <Header view={view} setView={setView} />
      <div className="main">
        {view === 'models' ? (
          <ModelsView />
        ) : (
          <>
            <Sidebar cases={CASE_STUDIES} selected={selectedCase} onSelect={setSelectedCase} />
            <div className="map-container">
              <MapContainer center={mapCenter} zoom={5} style={{ height: '100%', width: '100%' }}>
                <TileLayer
                  url="https://{s}.basemaps.cartocdn.com/dark_all/{z}/{x}/{y}{r}.png"
                  attribution='&copy; CARTO'
                />
                {CASE_STUDIES.map(c => c.site?.location ? (
                  <CircleMarker
                    key={c.id}
                    center={[c.site.location.lat, c.site.location.lng]}
                    radius={selectedCase === c.id ? 12 : 8}
                    fillColor={getAlertColor(c.site?.currentAlertLevel || 'normal')}
                    color={selectedCase === c.id ? '#fff' : 'transparent'}
                    weight={2}
                    fillOpacity={0.8}
                    eventHandlers={{ click: () => setSelectedCase(c.id) }}
                  >
                    <Popup>
                      <strong>{c.name}</strong><br />
                      {c.eventType} | {c.site?.waterBodyType}
                    </Popup>
                  </CircleMarker>
                ) : null)}
              </MapContainer>
            </div>
            {activeStudy && <DetailPanel study={activeStudy} />}
          </>
        )}
      </div>
    </div>
  )
}

export default App
