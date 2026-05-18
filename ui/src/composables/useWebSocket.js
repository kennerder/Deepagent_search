import { ref, onUnmounted, inject } from 'vue'

export function useWebSocket() {
  const threadId = inject('threadId')
  const connected = inject('connected')
  const monitorEvents = inject('monitorEvents')
  const messages = inject('messages')

  let ws = null
  let pingTimer = null
  const eventId = ref(0)

  function connect() {
    const proto = location.protocol === 'https:' ? 'wss:' : 'ws:'
    const host = location.host || 'localhost:8000'
    const url = `${proto}//${host}/ws/${threadId.value}`

    ws = new WebSocket(url)

    ws.onopen = () => {
      connected.value = true
      pingTimer = setInterval(() => ws.send('ping'), 30000)
    }

    ws.onmessage = (e) => {
      try {
        const data = JSON.parse(e.data)
        if (data.type === 'monitor_event') {
          eventId.value++
          const entry = {
            id: eventId.value,
            event: data.event,
            message: data.message,
            data: data.data || {},
            time: new Date().toLocaleTimeString(),
          }
          monitorEvents.value.push(entry)

          if (data.event === 'task_result') {
            messages.value.push({
              id: Date.now(),
              role: 'agent',
              content: data.data?.result || data.message,
              time: new Date().toLocaleTimeString(),
            })
          }
        }
      } catch {}
    }

    ws.onclose = () => {
      connected.value = false
      clearInterval(pingTimer)
      setTimeout(connect, 3000)
    }

    ws.onerror = () => ws?.close()
  }

  function send(text) {
    if (ws?.readyState === WebSocket.OPEN) ws.send(text)
  }

  connect()
  onUnmounted(() => {
    clearInterval(pingTimer)
    ws?.close()
  })

  return { send }
}
