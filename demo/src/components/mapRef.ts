// Module-level singleton so other components (e.g. ImportDropZone) can ask
// "what lat/lon is under this cursor?" without prop-drilling the maplibre instance.
import type { Map as MLMap } from 'maplibre-gl';

let currentMap: MLMap | null = null;

export function setMapInstance(m: MLMap | null) {
  currentMap = m;
}

export function getMapInstance(): MLMap | null {
  return currentMap;
}

export function clientToLngLat(clientX: number, clientY: number): { lat: number; lon: number } | null {
  const m = currentMap;
  if (!m) return null;
  const rect = m.getContainer().getBoundingClientRect();
  if (
    clientX < rect.left ||
    clientX > rect.right ||
    clientY < rect.top ||
    clientY > rect.bottom
  ) return null;
  const ll = m.unproject([clientX - rect.left, clientY - rect.top]);
  return { lat: ll.lat, lon: ll.lng };
}
