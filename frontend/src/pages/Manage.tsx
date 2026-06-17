import { useState, useEffect, useRef } from 'react'
import { Database, Plus, RefreshCw, Loader2 } from 'lucide-react'
import { getStats, indexDataset, StatsResponse } from '../api'

export default function Manage() {
  const [stats, setStats] = useState<StatsResponse | null>(null)
  const [folderPath, setFolderPath] = useState('')
  const [datasetName, setDatasetName] = useState('')
  const [indexing, setIndexing] = useState(false)
  const [indexResult, setIndexResult] = useState<string | null>(null)
  const [error, setError] = useState<string | null>(null)
  const pollRef = useRef<ReturnType<typeof setInterval> | null>(null)

  async function fetchStats() {
    try {
      const s = await getStats()
      setStats(s)
    } catch {
      // silently ignore — may not be ready yet
    }
  }

  useEffect(() => {
    fetchStats()
  }, [])

  // Poll stats every 2s while indexing
  useEffect(() => {
    if (indexing) {
      pollRef.current = setInterval(fetchStats, 2000)
    } else {
      if (pollRef.current) clearInterval(pollRef.current)
    }
    return () => {
      if (pollRef.current) clearInterval(pollRef.current)
    }
  }, [indexing])

  async function handleIndex(e: React.FormEvent) {
    e.preventDefault()
    if (!folderPath.trim() || !datasetName.trim()) return
    setIndexing(true)
    setError(null)
    setIndexResult(null)
    try {
      const resp = await indexDataset(folderPath.trim(), datasetName.trim())
      setIndexResult(`Indexed ${resp.indexed} images into "${resp.dataset}"`)
      await fetchStats()
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Indexing failed')
    } finally {
      setIndexing(false)
    }
  }

  return (
    <div className="max-w-4xl mx-auto px-4 py-8 space-y-8">
      <h1 className="text-2xl font-bold text-gray-800">Manage Datasets</h1>

      {/* Stats table */}
      <div className="bg-white rounded-xl shadow p-6">
        <div className="flex items-center justify-between mb-4">
          <h2 className="text-lg font-semibold text-gray-700 flex items-center gap-2">
            <Database size={18} /> Indexed Datasets
          </h2>
          <button
            onClick={fetchStats}
            className="flex items-center gap-1 text-sm text-gray-500 hover:text-gray-700 transition-colors"
          >
            <RefreshCw size={14} /> Refresh
          </button>
        </div>

        {stats === null ? (
          <p className="text-sm text-gray-400">Loading...</p>
        ) : Object.keys(stats.collections).length === 0 ? (
          <p className="text-sm text-gray-400">No datasets indexed yet.</p>
        ) : (
          <table className="w-full text-sm">
            <thead>
              <tr className="text-left text-gray-500 border-b">
                <th className="pb-2 font-medium">Dataset</th>
                <th className="pb-2 font-medium text-right">Images</th>
              </tr>
            </thead>
            <tbody>
              {Object.entries(stats.collections).map(([name, count]) => (
                <tr key={name} className="border-b last:border-0">
                  <td className="py-2 font-mono text-gray-800">{name}</td>
                  <td className="py-2 text-right text-gray-600">{count.toLocaleString()}</td>
                </tr>
              ))}
            </tbody>
            <tfoot>
              <tr>
                <td className="pt-3 font-semibold text-gray-700">Total</td>
                <td className="pt-3 text-right font-semibold text-gray-700">
                  {stats.total.toLocaleString()}
                </td>
              </tr>
            </tfoot>
          </table>
        )}
      </div>

      {/* Index new dataset form */}
      <div className="bg-white rounded-xl shadow p-6">
        <h2 className="text-lg font-semibold text-gray-700 flex items-center gap-2 mb-4">
          <Plus size={18} /> Index New Dataset
        </h2>

        <form onSubmit={handleIndex} className="space-y-4">
          <div>
            <label className="block text-sm text-gray-600 mb-1">Dataset Name</label>
            <input
              type="text"
              value={datasetName}
              onChange={e => setDatasetName(e.target.value)}
              placeholder="e.g. cuhk_pedes"
              className="w-full border border-gray-300 rounded-lg px-3 py-2 text-sm focus:outline-none focus:ring-2 focus:ring-blue-500"
              disabled={indexing}
            />
          </div>

          <div>
            <label className="block text-sm text-gray-600 mb-1">
              Folder Path <span className="text-gray-400">(path on the Colab machine)</span>
            </label>
            <input
              type="text"
              value={folderPath}
              onChange={e => setFolderPath(e.target.value)}
              placeholder="e.g. /content/CUHK-PEDES/imgs"
              className="w-full border border-gray-300 rounded-lg px-3 py-2 text-sm focus:outline-none focus:ring-2 focus:ring-blue-500"
              disabled={indexing}
            />
            <p className="text-xs text-gray-400 mt-1">
              This path must exist on the Colab machine where the model service is running.
            </p>
          </div>

          <button
            type="submit"
            disabled={indexing || !folderPath.trim() || !datasetName.trim()}
            className="flex items-center gap-2 bg-blue-600 hover:bg-blue-700 disabled:opacity-50 text-white px-5 py-2 rounded-lg text-sm font-medium transition-colors"
          >
            {indexing ? (
              <>
                <Loader2 size={16} className="animate-spin" />
                Indexing... (this may take several minutes)
              </>
            ) : (
              <>
                <Plus size={16} />
                Start Indexing
              </>
            )}
          </button>
        </form>

        {indexResult && (
          <div className="mt-4 bg-green-50 border border-green-200 text-green-700 rounded-lg px-4 py-3 text-sm">
            {indexResult}
          </div>
        )}
        {error && (
          <div className="mt-4 bg-red-50 border border-red-200 text-red-700 rounded-lg px-4 py-3 text-sm">
            {error}
          </div>
        )}
      </div>
    </div>
  )
}
