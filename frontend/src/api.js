const API_BASE = import.meta.env.VITE_API_URL || 'http://localhost:8000'

export { API_BASE }

export async function checkSystemStatus() {
  const response = await fetch(`${API_BASE}/api/health`, { signal: AbortSignal.timeout(8000) })
  if (!response.ok) throw new Error('Health endpoint unavailable')
  return response.json()
}

export async function inspectImage(file) {
  const body = new FormData()
  body.append('file', file)
  const response = await fetch(`${API_BASE}/api/inspect`, { method: 'POST', body })
  if (!response.ok) throw new Error(`Inspection failed (${response.status})`)
  return response.json()
}

export async function fetchHistory() {
  const response = await fetch(`${API_BASE}/api/history`)
  if (!response.ok) throw new Error(`History unavailable (${response.status})`)
  return response.json()
}
