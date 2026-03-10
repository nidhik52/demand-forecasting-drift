/**
 * api.ts — typed fetch helpers for the FastAPI backend.
 *
 * In development, Vite proxies /forecast, /drift, /inventory to
 * http://localhost:8000, so BASE_URL can stay empty.
 * In production (Docker), set VITE_API_URL in the environment.
 */

import type { ForecastPoint, DriftEvent, InventoryRow } from "./types";

const BASE = import.meta.env.VITE_API_URL ?? "";

async function get<T>(url: string): Promise<T> {
  const res = await fetch(url);
  if (!res.ok) {
    const text = await res.text();
    throw new Error(`API error ${res.status}: ${text}`);
  }
  return res.json() as Promise<T>;
}

/** Return the list of all available SKU codes. */
export const fetchSkus = (): Promise<string[]> =>
  get<string[]>(`${BASE}/forecast/skus`);

/** Return 2026 forecast for a specific SKU (all 365 days). */
export const fetchForecast = (sku: string): Promise<ForecastPoint[]> =>
  get<ForecastPoint[]>(`${BASE}/forecast?sku=${encodeURIComponent(sku)}&limit=365`);

/** Return all drift events, optionally filtered by SKU. */
export const fetchDrift = (sku?: string): Promise<DriftEvent[]> =>
  get<DriftEvent[]>(
    sku ? `${BASE}/drift?sku=${encodeURIComponent(sku)}` : `${BASE}/drift`
  );

/** Return inventory recommendations (all SKUs or only those needing orders). */
export const fetchInventory = (needsOrder = false): Promise<InventoryRow[]> =>
  get<InventoryRow[]>(
    `${BASE}/inventory${needsOrder ? "?needs_order=true" : ""}`
  );
