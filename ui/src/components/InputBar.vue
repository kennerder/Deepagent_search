<script setup>
import { ref, inject } from 'vue'
import { useApi } from '../composables/useApi.js'
import { useWebSocket } from '../composables/useWebSocket.js'

const emit = defineEmits(['send'])
const messages = inject('messages')
const monitorEvents = inject('monitorEvents')

const { submitTask, uploadFiles, loading } = useApi()
const { send: wsSend } = useWebSocket()

const query = ref('')
const attachedFiles = ref([])
const fileInput = ref(null)

async function handleSend() {
  const text = query.value.trim()
  if (!text && !attachedFiles.value.length) return

  if (attachedFiles.value.length) {
    await uploadFiles(attachedFiles.value)
    attachedFiles.value = []
  }

  if (text) {
    // clear previous monitor events for this turn
    monitorEvents.value.length = 0
    messages.value.push({ id: Date.now(), role: 'user', content: text, time: new Date().toLocaleTimeString() })
    query.value = ''
    await submitTask(text)
  }
}

function handleKeydown(e) {
  if (e.key === 'Enter' && !e.shiftKey) {
    e.preventDefault()
    handleSend()
  }
}

function handleFileChange(e) {
  attachedFiles.value = [...e.target.files]
}

function removeFile(name) {
  attachedFiles.value = attachedFiles.value.filter(f => f.name !== name)
}
</script>

<template>
  <div class="input-bar">
    <div v-if="attachedFiles.length" class="file-chips">
      <span v-for="f in attachedFiles" :key="f.name" class="chip">
        {{ f.name }}
        <button @click="removeFile(f.name)">×</button>
      </span>
    </div>
    <div class="input-row">
      <button class="attach-btn" @click="fileInput?.click()" title="上传文件">+</button>
      <input ref="fileInput" type="file" multiple @change="handleFileChange" style="display:none" />
      <textarea
        v-model="query"
        placeholder="输入查询..."
        rows="1"
        @keydown="handleKeydown"
      ></textarea>
      <button class="send-btn" :disabled="loading || !query.trim()" @click="handleSend">
        {{ loading ? '...' : '→' }}
      </button>
    </div>
  </div>
</template>

<style scoped>
.input-bar {
  padding: 16px 32px 24px;
  border-top: 1px solid var(--border);
  background: var(--bg-root);
}

.file-chips {
  display: flex;
  gap: 8px;
  margin-bottom: 10px;
  flex-wrap: wrap;
}
.chip {
  display: flex;
  align-items: center;
  gap: 6px;
  padding: 4px 10px;
  background: var(--bg-layer2);
  border-radius: 4px;
  font-size: 12px;
  border: 1px solid var(--border);
}
.chip button {
  background: none; border: none;
  color: var(--text-muted); cursor: pointer; font-size: 14px;
}

.input-row {
  display: flex;
  gap: 10px;
  align-items: flex-end;
  background: var(--bg-layer1);
  border-radius: var(--radius);
  padding: 10px 14px;
  border: 1px solid var(--border);
}
.input-row:focus-within { border-color: var(--accent); }

.attach-btn {
  background: none; border: none;
  color: var(--text-muted); font-size: 18px; cursor: pointer;
  padding: 0 4px;
}
.attach-btn:hover { color: var(--accent); }

textarea {
  flex: 1;
  background: none; border: none; outline: none;
  color: var(--text-primary); font-size: 14px;
  font-family: inherit;
  resize: none;
  max-height: 120px;
}
textarea::placeholder { color: var(--text-muted); }

.send-btn {
  width: 36px; height: 36px;
  border-radius: 50%;
  background: var(--accent);
  border: none;
  color: #1a1e24;
  font-size: 18px;
  cursor: pointer;
  display: flex;
  align-items: center;
  justify-content: center;
  flex-shrink: 0;
  transition: opacity 0.2s;
}
.send-btn:disabled { opacity: 0.3; cursor: default; }
.send-btn:not(:disabled):hover { opacity: 0.85; }
</style>
