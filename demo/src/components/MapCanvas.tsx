import { useEffect, useMemo, useRef, useState } from 'react';
import maplibregl, { Map as MLMap, Marker, Popup, GeoJSONSource } from 'maplibre-gl';
import 'maplibre-gl/dist/maplibre-gl.css';
import type { CanonicalRecord } from '../types';
import { EVENTS, CLEAN_SITES, NEON_SITES, USGS_STATIONS } from '../data/realData';
import type { Action, Layers } from '../state';
import { CoverageToggles } from './CoverageToggles';
import { StoryChips } from './StoryChips';
import { Legend } from './Legend';
import { resolve } from '../engine/resolve';
import { fetchLiveSensor, patchFromLive } from '../engine/liveUsgs';
import { useToasts } from './Toasts';
import { setMapInstance } from './mapRef';

interface Props {
  selected: CanonicalRecord | null;
  layers: Layers;
  legendOpen: boolean;
  dispatch: React.Dispatch<Action>;
  date: string;
  colorBlind: boolean;
}

// Default + color-blind-safe (viridis-ish) tier palettes. Same hue order so the
// "high tier = warmer" semantics survive, but with luminance separation that
// reads correctly under deuteranopia / protanopia.
const TIER_COLORS: Record<number, string> = {
  5: '#c0362c',
  4: '#b8430f',
  3: '#b06e1a',
  2: '#86868b',
  1: '#2e8540',
};
const TIER_COLORS_CB: Record<number, string> = {
  5: '#fde725', // bright yellow
  4: '#5ec962',
  3: '#21918c',
  2: '#3b528b',
  1: '#440154',
};
function tierColor(tier: number, cb: boolean) {
  return (cb ? TIER_COLORS_CB : TIER_COLORS)[tier] ?? '#aaa';
}

// GIBS WMTS endpoint for daily true-color imagery.
function gibsUrl(layer: string, date: string) {
  return `https://gibs.earthdata.nasa.gov/wmts/epsg3857/best/${layer}/default/${date}/GoogleMapsCompatible_Level9/{z}/{y}/{x}.jpg`;
}

const BASE_STYLE: maplibregl.StyleSpecification = {
  version: 8,
  glyphs: 'https://demotiles.maplibre.org/font/{fontstack}/{range}.pbf',
  sources: {
    // Carto Voyager — clean Apple-Maps-style light basemap. Free, attribution required.
    carto: {
      type: 'raster',
      tiles: [
        'https://a.basemaps.cartocdn.com/rastertiles/voyager/{z}/{x}/{y}.png',
        'https://b.basemaps.cartocdn.com/rastertiles/voyager/{z}/{x}/{y}.png',
        'https://c.basemaps.cartocdn.com/rastertiles/voyager/{z}/{x}/{y}.png',
        'https://d.basemaps.cartocdn.com/rastertiles/voyager/{z}/{x}/{y}.png',
      ],
      tileSize: 256,
      attribution: '© OpenStreetMap contributors · © CARTO',
      maxzoom: 19,
    },
    gibs: {
      type: 'raster',
      tiles: [gibsUrl('MODIS_Terra_CorrectedReflectance_TrueColor', '2024-07-15')],
      tileSize: 256,
      attribution: 'NASA EOSDIS GIBS',
      maxzoom: 9,
    },
    chlorophyll: {
      type: 'raster',
      tiles: [gibsUrl('MODIS_Aqua_Chlorophyll_A', '2024-07-15')],
      tileSize: 256,
      attribution: 'NASA EOSDIS GIBS · MODIS Aqua Chl-a',
      maxzoom: 7,
    },
    drought: {
      // USDM (US Drought Monitor) published weekly map as an image overlay.
      // Coordinates roughly CONUS bounds.
      type: 'image',
      url: 'https://droughtmonitor.unl.edu/data/png/current/current_west_text.png',
      coordinates: [
        [-127, 51],
        [-65, 51],
        [-65, 22],
        [-127, 22],
      ],
    },
  },
  layers: [
    {
      id: 'background',
      type: 'background',
      paint: { 'background-color': '#eaf3f7' },
    },
    {
      id: 'carto-base',
      type: 'raster',
      source: 'carto',
      paint: { 'raster-opacity': 1, 'raster-fade-duration': 200 },
    },
    {
      id: 'gibs-satellite',
      type: 'raster',
      source: 'gibs',
      layout: { visibility: 'none' },
      paint: { 'raster-opacity': 0.85, 'raster-fade-duration': 200 },
      maxzoom: 9,
    },
    {
      id: 'chlorophyll-overlay',
      type: 'raster',
      source: 'chlorophyll',
      layout: { visibility: 'none' },
      paint: { 'raster-opacity': 0.7 },
    },
    {
      id: 'drought-overlay',
      type: 'raster',
      source: 'drought',
      layout: { visibility: 'none' },
      paint: { 'raster-opacity': 0.55 },
    },
  ],
};

export function MapCanvas({ selected, layers, legendOpen, dispatch, date, colorBlind }: Props) {
  const containerRef = useRef<HTMLDivElement>(null);
  const mapRef = useRef<MLMap | null>(null);
  const ready = useRef(false);
  const [liveFetching, setLiveFetching] = useState(false);
  const { push: pushToast } = useToasts();
  const [hover, setHover] = useState<{
    name: string;
    sub: string;
    tier?: number;
    lead?: number | null;
    x: number;
    y: number;
  } | null>(null);
  const popupRef = useRef<Popup | null>(null);

  const ALL_SITES = useMemo<CanonicalRecord[]>(() => [...EVENTS, ...CLEAN_SITES], []);

  // Initialize the map once.
  useEffect(() => {
    if (!containerRef.current || mapRef.current) return;
    const map = new maplibregl.Map({
      container: containerRef.current,
      style: BASE_STYLE,
      center: [-96, 38],
      zoom: 3.4,
      minZoom: 2,
      maxZoom: 9,
      attributionControl: { compact: true },
      cooperativeGestures: false,
    });
    mapRef.current = map;
    setMapInstance(map);

    map.on('load', () => {
      ready.current = true;
      // Add data sources for sites
      map.addSource('usgs', {
        type: 'geojson',
        data: { type: 'FeatureCollection', features: USGS_STATIONS.map((s) => ({
          type: 'Feature',
          geometry: { type: 'Point', coordinates: [s.lon, s.lat] },
          properties: { id: s.id, kind: 'usgs' },
        })) },
      });
      map.addSource('neon', {
        type: 'geojson',
        data: { type: 'FeatureCollection', features: NEON_SITES.map((s) => ({
          type: 'Feature',
          geometry: { type: 'Point', coordinates: [s.lon, s.lat] },
          properties: { id: s.id, name: s.name, tier: s.tier, kind: 'neon' },
        })) },
      });
      map.addSource('events', {
        type: 'geojson',
        cluster: true,
        clusterMaxZoom: 5,
        clusterRadius: 36,
        data: { type: 'FeatureCollection', features: ALL_SITES.map((s) => ({
          type: 'Feature',
          geometry: { type: 'Point', coordinates: [s.lon, s.lat] },
          properties: {
            id: s.id,
            name: s.name,
            tier: s.tier ?? 1,
            bookmarked: !!s.bookmarked,
            leadDays: s.leadDays ?? null,
          },
        })) },
      });

      // USGS small dots
      map.addLayer({
        id: 'usgs-dots',
        type: 'circle',
        source: 'usgs',
        paint: {
          'circle-radius': 3,
          'circle-color': '#5ff0ea',
          'circle-opacity': 0.55,
          'circle-stroke-width': 0,
        },
      });

      // NEON sites
      map.addLayer({
        id: 'neon-dots',
        type: 'circle',
        source: 'neon',
        paint: {
          'circle-radius': ['interpolate', ['linear'], ['get', 'tier'], 1, 5, 5, 9],
          'circle-color': [
            'match',
            ['get', 'tier'],
            5, tierColor(5, colorBlind),
            4, tierColor(4, colorBlind),
            3, tierColor(3, colorBlind),
            2, tierColor(2, colorBlind),
            1, tierColor(1, colorBlind),
            '#aaa',
          ],
          'circle-stroke-color': '#ffffff',
          'circle-stroke-width': 1.5,
        },
      });

      // Cluster bubbles (only visible at low zoom where points clump)
      map.addLayer({
        id: 'events-clusters',
        type: 'circle',
        source: 'events',
        filter: ['has', 'point_count'],
        paint: {
          'circle-radius': ['step', ['get', 'point_count'], 14, 5, 18, 15, 24],
          'circle-color': 'rgba(46,133,64,0.85)',
          'circle-stroke-color': '#ffffff',
          'circle-stroke-width': 2,
        },
      });
      map.addLayer({
        id: 'events-cluster-count',
        type: 'symbol',
        source: 'events',
        filter: ['has', 'point_count'],
        layout: {
          'text-field': '{point_count_abbreviated}',
          'text-size': 12,
          'text-font': ['Open Sans Semibold', 'Arial Unicode MS Bold'],
        },
        paint: { 'text-color': '#ffffff' },
      });
      // Individual event markers (unclustered only)
      map.addLayer({
        id: 'events-dots',
        type: 'circle',
        source: 'events',
        filter: ['!', ['has', 'point_count']],
        paint: {
          'circle-radius': [
            'case',
            ['get', 'bookmarked'],
            10,
            8,
          ],
          'circle-color': [
            'match',
            ['get', 'tier'],
            5, tierColor(5, colorBlind),
            4, tierColor(4, colorBlind),
            3, tierColor(3, colorBlind),
            2, tierColor(2, colorBlind),
            1, tierColor(1, colorBlind),
            '#aaa',
          ],
          'circle-stroke-color': '#ffffff',
          'circle-stroke-width': 2,
        },
      });
      // Zoom into a cluster on click.
      map.on('click', 'events-clusters', (e) => {
        const feats = map.queryRenderedFeatures(e.point, { layers: ['events-clusters'] });
        const cluster = feats[0];
        if (!cluster) return;
        const center = (cluster.geometry as GeoJSON.Point).coordinates as [number, number];
        map.easeTo({ center, zoom: Math.min(7, map.getZoom() + 2) });
      });

      // Click handlers
      map.on('click', async (e) => {
        const features = map.queryRenderedFeatures(e.point, {
          layers: ['events-dots', 'neon-dots'],
        });
        const f = features[0];
        if (f) {
          const id = f.properties?.id as string;
          const rec = ALL_SITES.find((s) => s.id === id);
          if (rec) {
            dispatch({ type: 'SELECT_SITE', record: rec });
            return;
          }
        }
        // Open the place card immediately with a synthesized record (instant feedback),
        // then patch in real NWIS data when it returns.
        const { lat, lng } = e.lngLat;
        const synth = resolve(lat, lng);
        dispatch({ type: 'SELECT_SITE', record: synth });
        setLiveFetching(true);
        dispatch({ type: 'LIVE_FETCH_BEGIN' });
        try {
          const live = await fetchLiveSensor(lat, lng);
          const patch = patchFromLive(synth, live);
          if (Object.keys(patch).length > 0) {
            dispatch({ type: 'PATCH_RECORD', patch });
            pushToast('success', `Live USGS data from ${live.stationName}`, 2400);
          } else if (!live.ok) {
            pushToast('info', live.reason ?? 'No live USGS data near this location.');
          }
        } catch (err) {
          console.error('USGS fetch failed', err);
          pushToast('error', 'Could not reach USGS NWIS — showing synthesized data.');
        } finally {
          setLiveFetching(false);
          dispatch({ type: 'LIVE_FETCH_END' });
        }
      });

      // Hover preview — positioned floating card
      map.on('mousemove', 'events-dots', (e) => {
        map.getCanvas().style.cursor = 'pointer';
        const f = e.features?.[0];
        if (!f) return;
        const p = f.properties as any;
        setHover({
          name: p.name,
          sub: p.bookmarked ? 'Validation event' : 'Bookmarked',
          tier: p.tier,
          lead: p.leadDays,
          x: e.point.x,
          y: e.point.y,
        });
      });
      map.on('mouseleave', 'events-dots', () => {
        map.getCanvas().style.cursor = '';
        setHover(null);
      });
      map.on('mousemove', 'neon-dots', (e) => {
        map.getCanvas().style.cursor = 'pointer';
        const f = e.features?.[0];
        if (!f) return;
        const p = f.properties as any;
        setHover({
          name: p.name,
          sub: `NEON site · Tier ${p.tier}`,
          tier: p.tier,
          lead: null,
          x: e.point.x,
          y: e.point.y,
        });
      });
      map.on('mouseleave', 'neon-dots', () => {
        map.getCanvas().style.cursor = '';
        setHover(null);
      });
    });

    return () => {
      setMapInstance(null);
      map.remove();
      mapRef.current = null;
      ready.current = false;
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  // Update GIBS tile date (base true-color AND chlorophyll overlay).
  useEffect(() => {
    const map = mapRef.current;
    if (!map || !ready.current) return;
    const gibs = map.getSource('gibs') as any;
    const chl = map.getSource('chlorophyll') as any;
    gibs?.setTiles?.([gibsUrl('MODIS_Terra_CorrectedReflectance_TrueColor', date)]);
    chl?.setTiles?.([gibsUrl('MODIS_Aqua_Chlorophyll_A', date)]);
  }, [date]);

  // Live-swap tier palette when colorBlind toggles (no re-init needed).
  useEffect(() => {
    const map = mapRef.current;
    if (!map || !ready.current) return;
    const expr = (def: '#aaa') => [
      'match',
      ['get', 'tier'],
      5, tierColor(5, colorBlind),
      4, tierColor(4, colorBlind),
      3, tierColor(3, colorBlind),
      2, tierColor(2, colorBlind),
      1, tierColor(1, colorBlind),
      def,
    ];
    if (map.getLayer('events-dots'))
      map.setPaintProperty('events-dots', 'circle-color', expr('#aaa') as any);
    if (map.getLayer('neon-dots'))
      map.setPaintProperty('neon-dots', 'circle-color', expr('#aaa') as any);
  }, [colorBlind]);

  // Toggle layer visibility.
  useEffect(() => {
    const map = mapRef.current;
    if (!map || !ready.current) return;
    map.setLayoutProperty('usgs-dots', 'visibility', layers.usgs ? 'visible' : 'none');
    map.setLayoutProperty('neon-dots', 'visibility', layers.neon ? 'visible' : 'none');
    map.setLayoutProperty('gibs-satellite', 'visibility', layers.satellite ? 'visible' : 'none');
    map.setLayoutProperty(
      'chlorophyll-overlay',
      'visibility',
      layers.chlorophyll ? 'visible' : 'none',
    );
    map.setLayoutProperty('drought-overlay', 'visibility', layers.drought ? 'visible' : 'none');
  }, [layers]);

  // Fly to selection.
  useEffect(() => {
    const map = mapRef.current;
    if (!map || !ready.current || !selected) return;
    // Place card occupies right ~440px; offset center to the left.
    const offsetLon = selected.lon + 1.5;
    map.flyTo({
      center: [offsetLon, selected.lat],
      zoom: 6.5,
      duration: 1200,
      essential: true,
    });
  }, [selected]);

  // Selected marker pulse (separate layer).
  useEffect(() => {
    const map = mapRef.current;
    if (!map || !ready.current) return;
    const src = map.getSource('selected') as GeoJSONSource | undefined;
    const data = selected
      ? {
          type: 'FeatureCollection' as const,
          features: [
            {
              type: 'Feature' as const,
              geometry: { type: 'Point' as const, coordinates: [selected.lon, selected.lat] },
              properties: {},
            },
          ],
        }
      : { type: 'FeatureCollection' as const, features: [] };
    if (src) {
      src.setData(data);
    } else {
      map.addSource('selected', { type: 'geojson', data });
      map.addLayer({
        id: 'selected-halo',
        type: 'circle',
        source: 'selected',
        paint: {
          'circle-radius': 22,
          'circle-color': '#ffffff',
          'circle-opacity': 0.18,
          'circle-stroke-color': '#ffffff',
          'circle-stroke-width': 1.5,
          'circle-stroke-opacity': 0.6,
        },
      });
    }
  }, [selected]);

  // Breathe the selected halo: radius 22 → 32, opacity 0.18 → 0.04, period 2.4s.
  useEffect(() => {
    if (!selected) return;
    const map = mapRef.current;
    if (!map) return;
    let raf = 0;
    const t0 = performance.now();
    const tick = (t: number) => {
      const phase = ((t - t0) % 2400) / 2400;
      const eased = (Math.sin(phase * Math.PI * 2 - Math.PI / 2) + 1) / 2; // 0 → 1 → 0
      const radius = 22 + eased * 14;
      const opacity = 0.18 + eased * 0.1;
      const strokeOpacity = 0.6 - eased * 0.45;
      if (map.getLayer('selected-halo')) {
        map.setPaintProperty('selected-halo', 'circle-radius', radius);
        map.setPaintProperty('selected-halo', 'circle-opacity', opacity);
        map.setPaintProperty('selected-halo', 'circle-stroke-opacity', strokeOpacity);
      }
      raf = requestAnimationFrame(tick);
    };
    raf = requestAnimationFrame(tick);
    return () => cancelAnimationFrame(raf);
  }, [selected]);

  // Close any popup
  useEffect(() => {
    return () => {
      popupRef.current?.remove();
    };
  }, []);
  // suppress unused
  void popupRef;
  void Marker;

  return (
    <div
      style={{
        position: 'absolute',
        inset: 0,
        background: '#eaf3f7',
        overflow: 'hidden',
      }}
    >
      <div ref={containerRef} style={{ position: 'absolute', inset: 0 }} />
      <CoverageToggles layers={layers} dispatch={dispatch} />
      <StoryChips events={EVENTS} cleanSites={CLEAN_SITES} dispatch={dispatch} />
      {!selected && !liveFetching && (
        <div
          style={{
            position: 'absolute',
            bottom: 24,
            left: '50%',
            transform: 'translateX(-50%)',
            padding: '8px 14px',
            background: 'rgba(255,255,255,0.85)',
            backdropFilter: 'blur(20px) saturate(180%)',
            borderRadius: 'var(--r-pill)',
            boxShadow: 'var(--shadow-sm)',
            fontSize: 12,
            color: 'var(--text-secondary)',
            zIndex: 'var(--z-map-overlay)' as any,
            pointerEvents: 'none',
            animation: 'fade-in-up var(--dur-base) var(--ease-entrance)',
          }}
        >
          Click anywhere on the map to synthesize a record, or pick a bookmarked event above.
        </div>
      )}
      {legendOpen && <Legend />}
      {liveFetching && (
        <div
          style={{
            position: 'absolute',
            top: 80,
            left: '50%',
            transform: 'translateX(-50%)',
            padding: '8px 16px',
            background: 'rgba(255,255,255,0.92)',
            backdropFilter: 'blur(20px) saturate(180%)',
            borderRadius: 'var(--r-pill)',
            boxShadow: 'var(--shadow-md)',
            fontSize: 13,
            color: 'var(--text-secondary)',
            zIndex: 'var(--z-map-overlay)' as any,
            display: 'flex',
            alignItems: 'center',
            gap: 10,
            animation: 'fade-in var(--dur-fast)',
          }}
        >
          <span
            style={{
              width: 8,
              height: 8,
              borderRadius: 4,
              background: 'var(--accent)',
              animation: 'pulse 1s ease-in-out infinite',
            }}
          />
          Fetching live USGS data…
        </div>
      )}
      {hover && (
        <div
          style={{
            position: 'absolute',
            left: hover.x + 16,
            top: hover.y + 16,
            minWidth: 200,
            maxWidth: 260,
            background: 'rgba(255,255,255,0.92)',
            backdropFilter: 'blur(20px) saturate(180%)',
            WebkitBackdropFilter: 'blur(20px) saturate(180%)',
            color: 'var(--text-primary)',
            padding: '10px 14px',
            borderRadius: 'var(--r-md)',
            boxShadow: 'var(--shadow-lg)',
            fontSize: 13,
            pointerEvents: 'none',
            animation: 'fade-in var(--dur-fast)',
          }}
        >
          <div style={{ fontWeight: 600 }}>{hover.name}</div>
          <div style={{ fontSize: 11, color: 'var(--text-tertiary)', marginTop: 2 }}>{hover.sub}</div>
          {hover.lead != null && (
            <div
              style={{
                marginTop: 6,
                fontSize: 12,
                color: 'var(--accent)',
              }}
            >
              Detected <span className="mono">{hover.lead.toFixed(1)}d</span> before advisory
            </div>
          )}
        </div>
      )}
    </div>
  );
}
