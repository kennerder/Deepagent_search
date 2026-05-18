import { inject, ref } from 'vue'

export function useApi() {
  const threadId = inject('threadId')
  const files = inject('files')
  const loading = ref(false)

  const base = '/api'

  async function submitTask(query) {
    loading.value = true
    try {
      const res = await fetch(`${base}/task`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ query, thread_id: threadId.value }),
      })
      return await res.json()
    } finally {
      loading.value = false
    }
  }

  async function uploadFiles(fileList) {
    const form = new FormData()
    for (const f of fileList) form.append('files', f)
    form.append('thread_id', threadId.value)

    const res = await fetch(`${base}/upload`, { method: 'POST', body: form })
    return await res.json()
  }

  async function fetchFiles() {
    const path = `output/session_${threadId.value}`
    try {
      const res = await fetch(`${base}/files?path=${encodeURIComponent(path)}`)
      const data = await res.json()
      files.value = data.files || []
    } catch { files.value = [] }
  }

  function downloadUrl(filePath) {
    return `${base}/download?path=${encodeURIComponent(filePath)}`
  }

  return { submitTask, uploadFiles, fetchFiles, downloadUrl, loading }
}
