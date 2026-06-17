const BASE = import.meta.env.VITE_API_URL ?? 'http://localhost:8000'

export interface SearchResult {
  image_path: string
  score: number
  dataset: string
  person_id?: string
}

export interface SearchTiming {
  encode_ms: number
  retrieve_ms: number
  rerank_ms: number
}

export interface SearchResponse {
  results: SearchResult[]
  latency_ms: number
  timing: SearchTiming
}

export interface StatsResponse {
  total: number
  collections: Record<string, number>
}

export interface IndexResponse {
  indexed: number
  dataset: string
  status: string
}

export async function searchText(
  query: string,
  topK: number,
  dataset: string,
): Promise<SearchResponse> {
  const res = await fetch(`${BASE}/search`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ query, top_k: topK, dataset }),
  })
  if (!res.ok) throw new Error(`Search failed: ${res.statusText}`)
  return res.json()
}

export async function getCollections(): Promise<string[]> {
  const res = await fetch(`${BASE}/collections`)
  if (!res.ok) throw new Error('Failed to fetch collections')
  return res.json()
}

export async function getStats(): Promise<StatsResponse> {
  const res = await fetch(`${BASE}/stats`)
  if (!res.ok) throw new Error('Failed to fetch stats')
  return res.json()
}

export async function indexDataset(
  folderPath: string,
  datasetName: string,
): Promise<IndexResponse> {
  const res = await fetch(`${BASE}/index/dataset`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ folder_path: folderPath, dataset_name: datasetName }),
  })
  if (!res.ok) throw new Error(`Indexing failed: ${res.statusText}`)
  return res.json()
}

export function imageUrl(path: string): string {
  return `${BASE}/image?path=${encodeURIComponent(path)}`
}
