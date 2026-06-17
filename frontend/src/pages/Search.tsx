import { useState, useEffect } from 'react'
import { Search as SearchIcon, Loader2 } from 'lucide-react'
import { searchText, getCollections, imageUrl, SearchResult, SearchTiming } from '../api'

export default function Search() {
  const [query, setQuery] = useState('')
  const [topK, setTopK] = useState(10)
  const [dataset, setDataset] = useState('all')
  const [collections, setCollections] = useState<string[]>([])
  const [results, setResults] = useState<SearchResult[]>([])
  const [latency, setLatency] = useState<number | null>(null)
  const [timing, setTiming] = useState<SearchTiming | null>(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)

  useEffect(() => {
    getCollections()
      .then(setCollections)
      .catch(() => {})
  }, [])

  async function handleSearch(e: React.FormEvent) {
    e.preventDefault()
    if (!query.trim()) return
    setLoading(true)
    setError(null)
    try {
      const resp = await searchText(query, topK, dataset)
      setResults(resp.results)
      setLatency(resp.latency_ms)
      setTiming(resp.timing ?? null)
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Search failed')
      setResults([])
    } finally {
      setLoading(false)
    }
  }

  return (
    <div className="max-w-6xl mx-auto px-4 py-8">
      <h1 className="text-2xl font-bold text-gray-800 mb-6">Text-based Person Search</h1>

      {/* Search form */}
      <form onSubmit={handleSearch} className="bg-white rounded-xl shadow p-6 mb-6 space-y-4">
        <div className="flex gap-3">
          <input
            type="text"
            value={query}
            onChange={e => setQuery(e.target.value)}
            placeholder="e.g. woman in red jacket carrying a bag"
            className="flex-1 border border-gray-300 rounded-lg px-4 py-2 text-sm focus:outline-none focus:ring-2 focus:ring-blue-500"
          />
          <button
            type="submit"
            disabled={loading || !query.trim()}
            className="flex items-center gap-2 bg-blue-600 hover:bg-blue-700 disabled:opacity-50 text-white px-5 py-2 rounded-lg text-sm font-medium transition-colors"
          >
            {loading ? <Loader2 size={16} className="animate-spin" /> : <SearchIcon size={16} />}
            Search
          </button>
        </div>

        <div className="flex flex-wrap gap-6 items-center text-sm">
          {/* Dataset filter */}
          <label className="flex items-center gap-2 text-gray-600">
            Dataset:
            <select
              value={dataset}
              onChange={e => setDataset(e.target.value)}
              className="border border-gray-300 rounded px-2 py-1 focus:outline-none focus:ring-1 focus:ring-blue-500"
            >
              <option value="all">All</option>
              {collections.map(c => (
                <option key={c} value={c}>{c}</option>
              ))}
            </select>
          </label>

          {/* Top-K slider */}
          <label className="flex items-center gap-3 text-gray-600">
            Top-K: <span className="font-semibold text-gray-800 w-4">{topK}</span>
            <input
              type="range"
              min={1}
              max={20}
              value={topK}
              onChange={e => setTopK(Number(e.target.value))}
              className="w-32 accent-blue-600"
            />
          </label>
        </div>
      </form>

      {/* Error */}
      {error && (
        <div className="bg-red-50 border border-red-200 text-red-700 rounded-lg px-4 py-3 mb-4 text-sm">
          {error}
        </div>
      )}

      {/* Results header */}
      {results.length > 0 && (
        <div className="mb-4 flex flex-wrap items-center gap-3 text-sm text-gray-500">
          <span>{results.length} results &mdash; {latency}ms total</span>
          {timing && (
            <span className="flex gap-2">
              <span className="bg-gray-100 rounded px-2 py-0.5">encode {timing.encode_ms}ms</span>
              <span className="bg-gray-100 rounded px-2 py-0.5">retrieve {timing.retrieve_ms}ms</span>
              <span className="bg-gray-100 rounded px-2 py-0.5">rerank {timing.rerank_ms}ms</span>
            </span>
          )}
        </div>
      )}

      {/* Image grid */}
      {results.length > 0 && (
        <div className="grid grid-cols-2 sm:grid-cols-3 md:grid-cols-4 lg:grid-cols-5 gap-4">
          {results.map((r, i) => (
            <div key={i} className="bg-white rounded-xl overflow-hidden shadow hover:shadow-md transition-shadow">
              <div className="aspect-[3/4] bg-gray-100 overflow-hidden">
                <img
                  src={imageUrl(r.image_path)}
                  alt={`result-${i}`}
                  className="w-full h-full object-cover"
                  onError={e => {
                    (e.target as HTMLImageElement).src =
                      'data:image/svg+xml,<svg xmlns="http://www.w3.org/2000/svg" width="100" height="133"><rect width="100%" height="100%" fill="%23e5e7eb"/><text x="50%" y="50%" text-anchor="middle" fill="%239ca3af" font-size="12">No image</text></svg>'
                  }}
                />
              </div>
              <div className="p-2">
                <div className="flex items-center justify-between">
                  <span className="text-xs font-semibold text-blue-600">
                    {(r.score * 100).toFixed(1)}%
                  </span>
                  <span className="text-xs text-gray-400 truncate ml-1">{r.dataset}</span>
                </div>
                {r.person_id && (
                  <p className="text-xs text-gray-400 truncate mt-0.5">{r.person_id}</p>
                )}
              </div>
            </div>
          ))}
        </div>
      )}

      {/* Empty state */}
      {!loading && results.length === 0 && !error && (
        <div className="text-center py-20 text-gray-400">
          <SearchIcon size={40} className="mx-auto mb-3 opacity-30" />
          <p>Type a description and hit Search</p>
        </div>
      )}
    </div>
  )
}
