<script setup>
import { inject, ref, watch, nextTick } from 'vue'

const messages = inject('messages')
const monitorEvents = inject('monitorEvents')
const container = ref(null)

watch(
  () => messages.value.length + monitorEvents.value.length,
  () => nextTick(() => {
    const el = container.value
    if (el) el.scrollTop = el.scrollHeight
  }),
)

function eventIcon(event) {
  const map = {
    router_decision: { icon: '◎', cls: 'violet' },
    session_created: { icon: '◉', cls: 'blue' },
    assistant_call: { icon: '◆', cls: 'blue' },
    tool_start: { icon: '▲', cls: 'amber' },
    task_result: { icon: '✓', cls: 'green' },
    error: { icon: '✗', cls: 'red' },
  }
  return map[event] || { icon: '●', cls: 'muted' }
}
</script>

<template>
  <div ref="container" class="chat-container">
    <div v-if="!messages.length && !monitorEvents.length" class="empty-state">
      <div class="empty-icon">◇</div>
      <div class="empty-title">DeepAgents Search</div>
      <div class="empty-desc">输入查询，Agent 将协同工作</div>
    </div>

    <template v-for="msg in messages" :key="msg.id">
      <div class="message" :class="msg.role">
        <div class="msg-avatar">{{ msg.role === 'user' ? 'U' : 'A' }}</div>
        <div class="msg-body">
          <div class="msg-time">{{ msg.time }}</div>
          <div class="msg-content">{{ msg.content }}</div>
        </div>
      </div>

      <!-- thinking timeline between user message and agent response -->
      <div v-if="msg.role === 'user'" class="thinking-section">
        <div v-for="ev in monitorEvents" :key="'t-' + ev.id" class="thinking-item">
          <span class="t-dot" :class="ev.event"></span>
          <span class="t-msg">{{ ev.message }}</span>
          <span class="t-time">{{ ev.time }}</span>
        </div>
      </div>
    </template>
  </div>
</template>

<style scoped>
.chat-container {
  flex: 1;
  overflow-y: auto;
  padding: 24px 32px;
}

.empty-state {
  display: flex;
  flex-direction: column;
  align-items: center;
  justify-content: center;
  height: 100%;
  opacity: 0.6;
}
.empty-icon { font-size: 48px; color: var(--accent); margin-bottom: 16px; }
.empty-title { font-size: 20px; font-weight: 600; margin-bottom: 8px; }
.empty-desc { font-size: 14px; color: var(--text-muted); }

.message {
  display: flex;
  gap: 14px;
  margin-bottom: 28px;
  max-width: 85%;
}
.message.user { margin-left: auto; flex-direction: row-reverse; }
.msg-avatar {
  width: 34px; height: 34px;
  border-radius: 50%;
  display: flex;
  align-items: center;
  justify-content: center;
  font-size: 13px;
  font-weight: 700;
  flex-shrink: 0;
}
.message.user .msg-avatar { background: var(--accent-dim); color: var(--accent); }
.message.agent .msg-avatar { background: var(--bg-layer2); color: var(--text-primary); }

.msg-body {
  background: var(--bg-layer2);
  border-radius: var(--radius);
  padding: 14px 18px;
  border: 1px solid var(--border);
}
.message.user .msg-body { background: var(--accent-dim); border-color: transparent; }
.msg-time { font-size: 11px; color: var(--text-muted); margin-bottom: 6px; }
.msg-content { font-size: 14px; line-height: 1.7; white-space: pre-wrap; }

.thinking-section {
  margin: -12px 0 20px 48px;
  padding: 12px 16px;
  background: var(--bg-layer1);
  border-radius: var(--radius);
  border: 1px solid var(--border);
  max-width: 480px;
}
.thinking-item {
  display: flex;
  align-items: center;
  gap: 10px;
  padding: 3px 0;
  font-size: 12px;
}
.t-dot {
  width: 6px; height: 6px;
  border-radius: 50%;
  flex-shrink: 0;
}
.t-dot.router_decision { background: var(--violet); }
.t-dot.session_created { background: var(--blue); }
.t-dot.assistant_call { background: var(--blue); }
.t-dot.tool_start { background: var(--amber); }
.t-dot.task_result { background: var(--green); }
.t-dot.error { background: var(--red); }
.t-msg { flex: 1; color: var(--text-primary); }
.t-time { font-size: 10px; color: var(--text-muted); white-space: nowrap; }
</style>
