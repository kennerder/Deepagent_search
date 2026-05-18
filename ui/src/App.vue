<script setup>
import { ref, provide } from 'vue'
import Sidebar from './components/Sidebar.vue'
import ChatPanel from './components/ChatPanel.vue'
import InputBar from './components/InputBar.vue'

const threadId = ref(generateId())
const connected = ref(false)
const messages = ref([])
const monitorEvents = ref([])
const files = ref([])
const sidebarCollapsed = ref(false)

function generateId() {
  return 'session-' + Date.now() + '-' + Math.random().toString(36).slice(2, 8)
}

function addMessage(msg) {
  messages.value.push({ ...msg, id: Date.now(), time: new Date().toLocaleTimeString() })
}

provide('threadId', threadId)
provide('connected', connected)
provide('messages', messages)
provide('monitorEvents', monitorEvents)
provide('files', files)
</script>

<template>
  <div class="app-shell">
    <Sidebar :collapsed="sidebarCollapsed" @toggle="sidebarCollapsed = !sidebarCollapsed" />
    <main class="main-area" :class="{ expanded: sidebarCollapsed }">
      <ChatPanel />
      <InputBar @send="addMessage" />
    </main>
  </div>
</template>

<style>
:root {
  --bg-root: #0d0f11;
  --bg-layer1: #14171c;
  --bg-layer2: #1a1e24;
  --bg-layer3: #1f232b;
  --text-primary: #c8cdd4;
  --text-muted: #6b7280;
  --accent: #c8a45c;
  --accent-dim: rgba(200, 164, 92, 0.15);
  --border: #2a2d35;
  --violet: #a78bfa;
  --blue: #60a5fa;
  --amber: #f59e0b;
  --green: #4ade80;
  --red: #f87171;
  --radius: 8px;
  --font-mono: "SF Mono", "Cascadia Code", "Consolas", monospace;
}

.app-shell {
  display: flex;
  height: 100vh;
  width: 100vw;
}

.main-area {
  flex: 1;
  display: flex;
  flex-direction: column;
  min-width: 0;
  transition: margin-left 0.3s ease;
}

.main-area.expanded {
  margin-left: 0;
}
</style>
