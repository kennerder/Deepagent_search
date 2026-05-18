<script setup>
import { inject, ref, onMounted } from 'vue'
import { useApi } from '../composables/useApi.js'

defineProps(['collapsed'])
const emit = defineEmits(['toggle'])

const connected = inject('connected')
const monitorEvents = inject('monitorEvents')
const files = inject('files')

const { fetchFiles, downloadUrl } = useApi()

const agents = [
  { name: '网络搜索助手', key: 'web_search', dot: 'var(--amber)', desc: 'Tavily 多角度深度搜索' },
  { name: '数据库查询助手', key: 'db_query', dot: 'var(--blue)', desc: 'LangGraph SQL 流水线' },
  { name: '知识库助手', key: 'kb_search', dot: 'var(--green)', desc: '本地 RAG 管线' },
]

function routeTag(key) {
  const map = { web_search: '网络', db_query: '数据库', kb_search: '知识库' }
  return map[key] || key
}

const lastRoute = ref(null)
</script>

<template>
  <aside class="sidebar" :class="{ collapsed }">
    <div class="sidebar-header">
      <span v-if="!collapsed" class="logo">DeepAgents</span>
      <button class="toggle-btn" @click="emit('toggle')">{{ collapsed ? '▶' : '◀' }}</button>
    </div>

    <div v-if="!collapsed" class="sidebar-body">
      <div class="status-bar">
        <span class="dot" :class="{ on: connected }"></span>
        <span>{{ connected ? '已连接' : '未连接' }}</span>
      </div>

      <section class="side-section">
        <h3>子代理</h3>
        <div v-for="a in agents" :key="a.key" class="agent-card">
          <span class="agent-dot" :style="{ background: a.dot }"></span>
          <div>
            <div class="agent-name">{{ a.name }}</div>
            <div class="agent-desc">{{ a.desc }}</div>
          </div>
        </div>
      </section>

      <section class="side-section">
        <h3>实时事件</h3>
        <div class="event-list">
          <div v-for="ev in monitorEvents.slice(-8).reverse()" :key="ev.id" class="event-item">
            <span class="event-dot" :class="ev.event"></span>
            <span class="event-msg">{{ ev.message }}</span>
          </div>
          <div v-if="!monitorEvents.length" class="text-muted">等待事件...</div>
        </div>
      </section>

      <section class="side-section">
        <h3>
          文件
          <button class="mini-btn" @click="fetchFiles">↻</button>
        </h3>
        <div class="file-list">
          <div v-for="f in files.slice(0, 10)" :key="f.name" class="file-item">
            <span class="file-name">{{ f.name }}</span>
            <a :href="downloadUrl(f.path)" class="file-dl">↓</a>
          </div>
          <div v-if="!files.length" class="text-muted">暂无文件</div>
        </div>
      </section>
    </div>
  </aside>
</template>

<style scoped>
.sidebar {
  width: 280px;
  background: var(--bg-layer1);
  border-right: 1px solid var(--border);
  display: flex;
  flex-direction: column;
  transition: width 0.3s ease;
  flex-shrink: 0;
}
.sidebar.collapsed { width: 40px; }

.sidebar-header {
  display: flex;
  justify-content: space-between;
  align-items: center;
  padding: 16px;
  border-bottom: 1px solid var(--border);
}
.logo {
  font-size: 16px;
  font-weight: 700;
  color: var(--accent);
  letter-spacing: 1px;
}
.toggle-btn {
  background: none;
  border: none;
  color: var(--text-muted);
  cursor: pointer;
  font-size: 12px;
}
.toggle-btn:hover { color: var(--text-primary); }

.sidebar-body {
  flex: 1;
  overflow-y: auto;
  padding: 12px;
}

.status-bar {
  display: flex;
  align-items: center;
  gap: 8px;
  padding: 8px 12px;
  background: var(--bg-layer2);
  border-radius: var(--radius);
  font-size: 13px;
  margin-bottom: 16px;
}
.dot {
  width: 8px; height: 8px;
  border-radius: 50%;
  background: var(--red);
  flex-shrink: 0;
}
.dot.on { background: var(--green); }

.side-section { margin-bottom: 20px; }
.side-section h3 {
  font-size: 11px;
  text-transform: uppercase;
  letter-spacing: 1px;
  color: var(--text-muted);
  margin-bottom: 8px;
  display: flex;
  justify-content: space-between;
  align-items: center;
}
.mini-btn {
  background: none; border: none; color: var(--accent);
  cursor: pointer; font-size: 14px;
}

.agent-card {
  display: flex;
  gap: 10px;
  padding: 10px;
  background: var(--bg-layer2);
  border-radius: var(--radius);
  margin-bottom: 6px;
}
.agent-dot { width: 8px; height: 8px; border-radius: 50%; margin-top: 4px; flex-shrink: 0; }
.agent-name { font-size: 13px; font-weight: 600; }
.agent-desc { font-size: 11px; color: var(--text-muted); margin-top: 2px; }

.event-list { max-height: 200px; overflow-y: auto; }
.event-item {
  display: flex;
  gap: 8px;
  padding: 6px 0;
  font-size: 12px;
  border-bottom: 1px solid var(--border);
}
.event-dot { width: 6px; height: 6px; border-radius: 50%; margin-top: 4px; flex-shrink: 0; }
.event-dot.router_decision { background: var(--violet); }
.event-dot.assistant_call { background: var(--blue); }
.event-dot.tool_start { background: var(--amber); }
.event-dot.task_result { background: var(--green); }
.event-dot.error { background: var(--red); }
.event-msg { line-height: 1.4; }

.file-list { max-height: 200px; overflow-y: auto; }
.file-item {
  display: flex;
  justify-content: space-between;
  align-items: center;
  padding: 6px 0;
  font-size: 12px;
  border-bottom: 1px solid var(--border);
}
.file-name { overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
.file-dl { color: var(--accent); text-decoration: none; font-size: 14px; }
.file-dl:hover { opacity: 0.8; }
.text-muted { font-size: 12px; color: var(--text-muted); }
</style>
