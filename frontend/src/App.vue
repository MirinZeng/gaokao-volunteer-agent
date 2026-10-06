<script setup>
import { computed, nextTick, onMounted, ref } from 'vue'

import { renderedContent } from './markdown'

const input = ref('')
const messages = ref([])
const conversations = ref([])
const currentConversation = ref(localStorage.getItem('gaokao-current-session') || '')
const activeMemory = ref({ metadata: {}, profile: {}, summary: '', window: [] })
const profileOpen = ref(false)
const isStreaming = ref(false)
const statusText = ref('正在检查服务')
const retrievalMode = ref('BM25')
const webEnabled = ref(true)
const messageList = ref(null)
let abortController = null

// 后端地址候选：优先同源（后端 8000 托管，或 Vite 开发服务器的 /api 代理），
// 同源不可用时回退到本地 FastAPI，这样前端无论被哪个本地服务托管都能连上后端。
const API_BASES = ['', 'http://127.0.0.1:8000']

async function apiFetch(path, options = {}) {
  let lastError = null
  let lastResponse = null
  for (const base of API_BASES) {
    try {
      const response = await fetch(`${base}${path}`, options)
      if (response.ok) return response
      lastResponse = response
    } catch (error) {
      if (error.name === 'AbortError') throw error
      lastError = error
    }
  }
  if (lastResponse) return lastResponse
  throw lastError ?? new Error('所有后端地址均不可用')
}

// ---- Embedding / 检索配置（来自 /api/health）----
const embedding = ref({
  backend: '',
  model: '',
  dimension: null,
  vectorEnabled: false,
  documentCount: null,
  matches: true,
  chunkChars: null,
  device: null,
  useFp16: false,
})
const weights = ref({ bm25: 0.45, vector: 0.55 })

const embeddingLabel = computed(() => {
  const info = embedding.value
  if (!info.vectorEnabled) return 'BM25 关键词检索'
  const parts = []
  if (info.model) parts.push(String(info.model).toUpperCase())
  if (info.dimension) parts.push(`${info.dimension} 维`)
  parts.push(info.backend === 'bge_m3' ? '本地 GPU' : '云端 API')
  return parts.join(' · ')
})

const embeddingShort = computed(() => {
  const info = embedding.value
  if (!info.vectorEnabled) return 'BM25'
  return info.model ? String(info.model).toUpperCase() : '向量'
})

// 每一条回答实际走的检索路由（由后端意图识别决定）
const ROUTE_LABELS = {
  bm25: '纯 BM25',
  entity_bm25: '院校检索 · 纯 BM25',
  city_bm25: '城市检索 · 纯 BM25',
  hybrid_bm25_cosine: '混合检索',
}
const routeLabel = (route) => ROUTE_LABELS[route] || '混合检索'
// 只有混合检索才有余弦分；实体/城市路径走纯 BM25，不显示向量列
const hasVector = (route) => route === 'hybrid_bm25_cosine'

const fmtScore = (value) =>
  value === null || value === undefined || value === '' ? '—' : Number(value).toFixed(3)

const scoreWidth = (value) => {
  const num = Number(value)
  if (!Number.isFinite(num) || num < 0) return '0%'
  return `${Math.round(Math.max(0, Math.min(1, num)) * 100)}%`
}

// 将模型的逐校说明拆成带标签的短行；旧回答没有标签时也能正常回退显示。
const schoolDetailParts = (text = '') => {
  const parts = String(text).split(/[；\n]+/).map((part) => part.trim()).filter(Boolean)
  return parts.map((part, index) => {
    const match = part.match(/^([^：:]{2,8})[：:]\s*(.+)$/)
    return match
      ? { label: match[1], value: match[2] }
      : { label: index === 0 ? '院校分析' : '', value: part }
  })
}

const canSend = computed(() => input.value.trim() && !isStreaming.value)
const hasMessages = computed(() => messages.value.length > 0)
const profileEntries = computed(() => {
  const labels = {
    province: '省份', year: '年份', batch: '批次', subject: '科类', score: '分数', rank: '位次',
    cities: '城市偏好', majors: '专业偏好', rejected_majors: '不接受专业', budget: '预算',
    risk_preference: '风险偏好', adjustment: '服从调剂', accept_private: '接受民办',
    accept_joint_program: '中外合作', priorities: '侧重方向',
  }
  return Object.entries(activeMemory.value?.profile || {})
    .filter(([, value]) => value !== '' && value !== null && value !== undefined)
    .map(([key, value]) => ({ key, label: labels[key] || key, value: Array.isArray(value) ? value.join('、') : value }))
})
const suggestions = [
  '江西物理类，位次 30000 左右，有哪些计算机专业？',
  '南昌有哪些软件工程专业可以参考？',
  '历史类想学法学，如何规划冲稳保？',
]

async function scrollToBottom() {
  await nextTick()
  if (messageList.value) messageList.value.scrollTop = messageList.value.scrollHeight
  // 思考面板是独立滚动容器，也需要跟着流式内容下滚
  document.querySelectorAll('.thinking-body').forEach((element) => {
    element.scrollTop = element.scrollHeight
  })
}

function emptyAssistant(payload = {}) {
  return {
    role: 'assistant', status: payload.status === 'failed' ? '生成失败' : '回答完成',
    route: payload.retrieval_mode || '', blocks: payload.blocks || [], content: payload.answer || payload.message || '',
    citations: payload.citations || [], evidence: payload.evidence || [], webSources: payload.web_sources || [],
    grounding: payload.grounding || '', validated: Boolean(payload.validated), warning: payload.warning || '',
    reasoning: '', reasoningBuffer: '', reasoningChars: 0, thinkingOpen: false,
    thinkingSeconds: 0, thinkingStartedAt: 0, answerStarted: true, evidenceOpen: false, retrieveMs: null,
  }
}

function hydrateMessages(rows = []) {
  return rows.map((row) => row.role === 'user'
    ? { role: 'user', content: row.content }
    : emptyAssistant(row.payload || {}))
}

async function refreshSessions() {
  const response = await apiFetch('/api/sessions')
  const data = await response.json()
  conversations.value = data.sessions || []
}

async function openConversation(id) {
  if (!id || isStreaming.value) return
  const response = await apiFetch(`/api/sessions/${id}`)
  const data = await response.json()
  if (!data.session) return
  currentConversation.value = id
  localStorage.setItem('gaokao-current-session', id)
  messages.value = hydrateMessages(data.messages)
  activeMemory.value = data.memory || { metadata: {}, profile: {}, summary: '', window: [] }
  await scrollToBottom()
}

async function newChat() {
  if (isStreaming.value) stopGeneration()
  const response = await apiFetch('/api/sessions', {
    method: 'POST', headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ title: '新的志愿咨询' }),
  })
  const session = await response.json()
  input.value = ''
  await refreshSessions()
  await openConversation(session.id)
}

async function removeConversation(id) {
  if (isStreaming.value) return
  await apiFetch(`/api/sessions/${id}`, { method: 'DELETE' })
  localStorage.removeItem('gaokao-current-session')
  await refreshSessions()
  if (conversations.value.length) await openConversation(conversations.value[0].id)
  else await newChat()
}

const sessionTime = (stamp) => stamp
  ? new Date(Number(stamp) * 1000).toLocaleDateString('zh-CN', { month: 'numeric', day: 'numeric' })
  : ''

function stopGeneration() {
  abortController?.abort()
  abortController = null
  isStreaming.value = false
  statusText.value = '已停止生成'
}

function useSuggestion(text) {
  input.value = text
  sendMessage()
}

function handleKeydown(event) {
  if (event.key === 'Enter' && !event.shiftKey) {
    event.preventDefault()
    sendMessage()
  }
}

async function sendMessage() {
  const question = input.value.trim()
  if (!question || isStreaming.value) return

  if (!currentConversation.value) await newChat()
  const active = conversations.value.find((item) => item.id === currentConversation.value)
  if (active?.title === '新的志愿咨询') active.title = question.slice(0, 20)
  messages.value.push({ role: 'user', content: question })
  const draft = {
    role: 'assistant', status: '正在检索 Excel 数据', route: '',
    blocks: [], content: '', citations: [], evidence: [], webSources: [], grounding: '', validated: false, warning: '',
    reasoning: '', reasoningBuffer: '', reasoningChars: 0, thinkingOpen: false,
    thinkingSeconds: 0, thinkingStartedAt: 0, answerStarted: false, evidenceOpen: false,
    retrieveMs: null,
  }
  messages.value.push(draft)
  // 关键：push 进去的是「原始对象」，直接改它不会触发重渲染。
  // Vue 只在读取数组元素时才返回响应式代理，所以必须把它取出来，后续改动全走这个代理。
  const assistant = messages.value[messages.value.length - 1]
  input.value = ''
  isStreaming.value = true
  statusText.value = '正在连接 Agent'
  abortController = new AbortController()
  await scrollToBottom()

  try {
    const response = await apiFetch('/api/chat', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        message: question, web_search: webEnabled.value, session_id: currentConversation.value,
      }),
      signal: abortController.signal,
    })
    if (!response.ok || !response.body) throw new Error(`服务请求失败：${response.status}`)

    const reader = response.body.getReader()
    const decoder = new TextDecoder('utf-8')
    let buffer = ''
    while (true) {
      const { value, done } = await reader.read()
      if (done) break
      buffer += decoder.decode(value, { stream: true })
      const lines = buffer.split('\n')
      buffer = lines.pop() || ''
      for (const line of lines) {
        if (line.trim()) handleEvent(JSON.parse(line), assistant)
      }
      await scrollToBottom()
    }
    if (buffer.trim()) handleEvent(JSON.parse(buffer), assistant)
  } catch (error) {
    if (error.name !== 'AbortError') {
      assistant.status = '请求失败'
      assistant.content ||= `无法连接后端：${error.message}`
      statusText.value = '服务异常'
    }
  } finally {
    flushBuffers(assistant)
    isStreaming.value = false
    abortController = null
    await refreshSessions()
    const memoryResponse = await apiFetch(`/api/sessions/${currentConversation.value}`)
    const memoryData = await memoryResponse.json()
    activeMemory.value = memoryData.memory || activeMemory.value
    await scrollToBottom()
  }
}

// 思考内容以极细粒度流式到达（实测 1.8 万字 ≈ 1 万个事件），逐个写响应式状态会触发
// 上万次重渲染，因此节流到 ~90ms 批量刷新。
// 正文不再需要节流：结构化块是"每凑齐一个完整 JSON 对象推一次"，天然低频。
const FLUSH_MS = 90
let flushTimer = null
const dirtyAssistants = new Set()

function applyBuffers(assistant) {
  if (!assistant?.reasoningBuffer) return
  assistant.reasoning += assistant.reasoningBuffer
  assistant.reasoningChars = assistant.reasoning.length
  assistant.reasoningBuffer = ''
}

function flushBuffers(assistant) {
  if (flushTimer) {
    clearTimeout(flushTimer)
    flushTimer = null
  }
  dirtyAssistants.delete(assistant)
  applyBuffers(assistant)
}

function scheduleFlush(assistant) {
  dirtyAssistants.add(assistant)
  if (flushTimer) return
  flushTimer = setTimeout(() => {
    flushTimer = null
    dirtyAssistants.forEach(applyBuffers)
    dirtyAssistants.clear()
    scrollToBottom()
  }, FLUSH_MS)
}

// ---- 思考计时：让"思考中"有可感知的进度 ----
let thinkingTimer = null

function startThinkingClock(assistant) {
  if (assistant.thinkingStartedAt) return
  assistant.thinkingStartedAt = Date.now()
  if (thinkingTimer) clearInterval(thinkingTimer)
  thinkingTimer = setInterval(() => {
    assistant.thinkingSeconds = Math.round((Date.now() - assistant.thinkingStartedAt) / 1000)
  }, 500)
}

function stopThinkingClock(assistant) {
  if (thinkingTimer) {
    clearInterval(thinkingTimer)
    thinkingTimer = null
  }
  if (assistant?.thinkingStartedAt) {
    assistant.thinkingSeconds = Math.round((Date.now() - assistant.thinkingStartedAt) / 1000)
  }
}

// 悬停在来源徽标上时显示该行的召回分数
function sourceTooltip(message, excelRow) {
  const item = (message.evidence || []).find((entry) => entry.excel_row === excelRow)
  if (!item) return `Excel 第 ${excelRow} 行`
  const parts = [`Excel 第 ${excelRow} 行`]
  if (item.bm25_score !== null && item.bm25_score !== undefined) {
    parts.push(`BM25 ${Number(item.bm25_score).toFixed(3)}`)
  }
  if (item.cosine_score !== null && item.cosine_score !== undefined && item.cosine_score >= 0) {
    parts.push(`余弦 ${Number(item.cosine_score).toFixed(3)}`)
  }
  return parts.join(' · ')
}

function handleEvent(event, assistant) {
  if (event.type === 'start') {
    retrievalMode.value = event.retrieval_mode === 'bm25' ? 'BM25' : '混合检索'
    assistant.route = event.retrieval_mode || ''
    assistant.retrieveMs = event.retrieve_ms ?? null
    const timing = event.retrieve_ms ? `（${(event.retrieve_ms / 1000).toFixed(1)}s）` : ''
    assistant.status =
      event.retrieval_mode === 'entity_bm25'
        ? `已锁定院校范围，命中 ${event.evidence_count} 条证据${timing}`
        : event.retrieval_mode === 'city_bm25'
          ? `已锁定城市范围，命中 ${event.evidence_count} 条证据${timing}`
          : `已找到 ${event.evidence_count} 条候选证据${timing}，正在生成`
  } else if (event.type === 'reasoning') {
    assistant.status = '深度思考中'
    startThinkingClock(assistant)
    if (!assistant.thinkingOpen) assistant.thinkingOpen = true
    assistant.reasoningBuffer += event.content || ''
    scheduleFlush(assistant)
  } else if (event.type === 'answer_start') {
    flushBuffers(assistant)
    stopThinkingClock(assistant)
    // 正文开始，自动收起思考面板，避免长思考链把回答挤到屏幕外
    if (assistant.reasoning) assistant.thinkingOpen = false
    assistant.answerStarted = true
    assistant.status = '正在组织答案'
  } else if (event.type === 'web_search_start') {
    assistant.status = '正在检索最新官方信息'
  } else if (event.type === 'web_search_result') {
    assistant.webSources = event.sources || []
    assistant.status = event.status === 'failed' ? '联网失败，继续使用本地数据' : '联网核验完成'
  } else if (event.type === 'block') {
    // 结构化块：每凑齐一个完整 JSON 对象就渲染一张卡片
    flushBuffers(assistant)
    assistant.blocks.push(event.block)
    assistant.status = `已生成 ${assistant.blocks.length} 个板块`
    scrollToBottom()
  } else if (event.type === 'result') {
    flushBuffers(assistant)
    stopThinkingClock(assistant)
    if (event.blocks?.length) {
      assistant.blocks = event.blocks
    } else {
      // 兜底：模型没按 NDJSON 输出时，把原文交给 Markdown 渲染器
      assistant.content = event.answer || ''
    }
    assistant.citations = event.citations || []
    assistant.evidence = event.evidence || []
    assistant.webSources = event.web_sources || assistant.webSources || []
    assistant.grounding = event.grounding || ''
    assistant.validated = Boolean(event.validated)
    assistant.warning = event.warning || ''
    assistant.status = event.status === 'fallback_retrieval'
      ? '余额不足，已切换纯检索'
      : (event.validated ? '回答完成' : '引用校验未通过')
  } else if (event.type === 'error') {
    flushBuffers(assistant)
    stopThinkingClock(assistant)
    assistant.status = '生成失败'
    assistant.content = event.message || '后端返回未知错误'
  }
  statusText.value = assistant.status
}

onMounted(async () => {
  try {
    const response = await apiFetch('/api/health')
    const data = await response.json()
    retrievalMode.value = data.retrieval_mode === 'bm25' ? 'BM25' : '混合检索'
    statusText.value = data.index_ready ? '服务已就绪' : '请先建立索引'
    embedding.value = {
      backend: data.embedding_backend || '',
      model: data.embedding_model || '',
      dimension: data.embedding_dimension ?? null,
      vectorEnabled: Boolean(data.vector_enabled),
      documentCount: data.document_count ?? null,
      matches: data.embedding_matches !== false,
      chunkChars: data.chunk_chars ?? null,
      device: data.bge_device ?? null,
      useFp16: Boolean(data.bge_use_fp16),
    }
    weights.value = { bm25: data.bm25_weight ?? 0.45, vector: data.vector_weight ?? 0.55 }
    if (!embedding.value.matches) {
      statusText.value = `索引模型与后端不符（索引=${embedding.value.model}）`
    }
    await refreshSessions()
    let target = currentConversation.value
    if (!target || !conversations.value.some((item) => item.id === target)) {
      target = conversations.value[0]?.id
    }
    if (target) await openConversation(target)
    else await newChat()
  } catch {
    statusText.value = '后端未启动'
  }
})
</script>

<template>
  <div class="app-shell">
    <aside class="sidebar">
      <div class="brand">
        <div class="brand-mark">志</div>
        <div><strong>志愿助手</strong><span>江西 2025 数据</span></div>
      </div>
      <button class="new-chat" @click="newChat"><span class="plus">＋</span>开启新对话</button>
      <div class="history-label">最近对话</div>
      <nav class="history-list">
        <button
          v-for="conversation in conversations" :key="conversation.id"
          :class="['history-item', { active: conversation.id === currentConversation }]"
          @click="openConversation(conversation.id)"
        >
          <svg viewBox="0 0 24 24"><path d="M5 5h14v11H9l-4 3V5Z" /></svg>
          <span class="history-copy">
            <b>{{ conversation.title }}</b>
            <small>{{ sessionTime(conversation.updated_at) }} · {{ conversation.message_count }} 条消息</small>
          </span>
          <span class="history-delete" title="删除会话" @click.stop="removeConversation(conversation.id)">×</span>
        </button>
      </nav>
      <div class="sidebar-footer">
        <span class="status-dot" :class="{ warn: !embedding.matches }"></span>
        <div>
          <strong>{{ retrievalMode }}</strong>
          <span>{{ statusText }}</span>
          <span class="embed-line">{{ embeddingLabel }}</span>
        </div>
      </div>
    </aside>

    <main class="main-panel">
      <header class="topbar">
        <div>
          <strong>高考志愿填报 Agent</strong>
          <span>基于 Excel 招生数据 · {{ embeddingLabel }}</span>
        </div>
        <div class="topbar-pills">
          <div class="model-pill"><span class="pulse"></span>DeepSeek</div>
          <div v-if="embedding.vectorEnabled" class="model-pill embed-pill">
            <span class="pulse"></span>{{ embeddingShort }}
          </div>
        </div>
      </header>

      <section v-if="profileEntries.length" class="profile-bar">
        <button class="profile-toggle" @click="profileOpen = !profileOpen">
          <span class="profile-icon">档</span>
          <span><b>考生档案</b><small>{{ profileEntries.slice(0, 3).map(item => `${item.label} ${item.value}`).join(' · ') }}</small></span>
          <i>{{ profileOpen ? '收起 ▲' : '展开 ▼' }}</i>
        </button>
        <div v-if="profileOpen" class="profile-grid">
          <div v-for="item in profileEntries" :key="item.key">
            <span>{{ item.label }}</span><strong>{{ item.value }}</strong>
          </div>
          <div v-if="activeMemory.summary" class="profile-summary">
            <span>近期摘要</span><p>{{ activeMemory.summary }}</p>
          </div>
        </div>
      </section>

      <section ref="messageList" class="conversation">
        <div v-if="!hasMessages" class="welcome">
          <div class="welcome-icon">志</div>
          <h1>你好，我是高考志愿助手</h1>
          <p>可以查询院校、专业、选科、历年分数与位次，也可以帮你梳理冲稳保策略。</p>
          <div class="stack-info">
            <span class="stack-item"><b>检索</b>{{ retrievalMode }}</span>
            <span class="stack-item"><b>向量</b>{{ embeddingLabel }}</span>
            <span v-if="embedding.documentCount" class="stack-item">
              <b>语料</b>{{ Number(embedding.documentCount).toLocaleString() }} 条
            </span>
            <span v-if="embedding.chunkChars" class="stack-item">
              <b>切块</b>{{ embedding.chunkChars }} 字符
            </span>
            <span v-if="embedding.useFp16" class="stack-item"><b>精度</b>FP16</span>
          </div>
          <div class="suggestion-grid">
            <button v-for="suggestion in suggestions" :key="suggestion" @click="useSuggestion(suggestion)">
              <span>{{ suggestion }}</span><span class="arrow">↗</span>
            </button>
          </div>
        </div>

        <div v-else class="messages">
          <article v-for="(message, index) in messages" :key="index" :class="['message', message.role]">
            <div v-if="message.role === 'assistant'" class="avatar">志</div>
            <div class="message-body">
              <div v-if="message.role === 'assistant'" class="message-meta">
                <strong>志愿助手</strong><span>{{ message.status }}</span>
              </div>
              <!-- 思考过程 -->
              <div v-if="message.reasoning || message.reasoningBuffer" class="thinking">
                <button class="thinking-head" @click="message.thinkingOpen = !message.thinkingOpen">
                  <span class="thinking-dot" :class="{ live: isStreaming && !message.answerStarted }"></span>
                  <span class="thinking-title">深度思考</span>
                  <span class="thinking-meta">
                    <template v-if="message.thinkingSeconds">已思考 {{ message.thinkingSeconds }} 秒 · </template>
                    {{ Number(message.reasoningChars || 0).toLocaleString() }} 字
                  </span>
                  <span class="thinking-toggle">{{ message.thinkingOpen ? '收起 ▲' : '展开 ▼' }}</span>
                </button>
                <div v-if="message.thinkingOpen" class="thinking-body">{{ message.reasoning }}</div>
              </div>

              <!-- 用户消息 -->
              <div v-if="message.role === 'user'" class="message-content">{{ message.content }}</div>

              <!-- 结构化卡片：模型按 NDJSON 输出的每个块渲染成一张卡片 -->
              <div v-else-if="message.blocks?.length" class="blocks">
                <template v-for="(block, bi) in message.blocks" :key="bi">
                  <section v-if="block.type === 'conclusion'" class="card card-conclusion">
                    <div class="card-label">结论</div>
                    <p class="card-lead">{{ block.text }}</p>
                  </section>

                  <section v-else-if="block.type === 'risks'" class="card card-risks">
                    <div class="card-label">风险提示</div>
                    <ul class="card-list">
                      <li v-for="(item, ii) in block.items" :key="ii">{{ item }}</li>
                    </ul>
                  </section>

                  <section v-else-if="block.layout === 'table'" class="card">
                    <div v-if="block.title" class="card-title">{{ block.title }}</div>
                    <div class="table-scroll">
                      <table class="data-table">
                        <thead>
                          <tr>
                            <th v-for="(column, ci) in block.columns" :key="ci">{{ column }}</th>
                            <th class="col-source">来源</th>
                          </tr>
                        </thead>
                        <tbody>
                          <tr v-for="(row, ri) in block.rows" :key="ri">
                            <td
                              v-for="(cell, ci) in row.cells"
                              :key="ci"
                              :class="{ 'school-cell': block.columns[ci] === '院校' }"
                            >{{ cell }}</td>
                            <td class="col-source">
                              <span
                                v-if="row.excel_row"
                                class="source-chip"
                                :title="sourceTooltip(message, row.excel_row)"
                              >{{ row.excel_row }}</span>
                              <span v-else class="source-empty">—</span>
                            </td>
                          </tr>
                        </tbody>
                      </table>
                    </div>
                  </section>

                  <section v-else-if="block.layout === 'list'" class="card">
                    <div v-if="block.title" class="card-title">{{ block.title }}</div>
                    <ul class="card-list">
                      <li v-for="(item, ii) in block.items" :key="ii">{{ item }}</li>
                    </ul>
                  </section>

                  <section v-else-if="block.layout === 'school_details'" class="card school-details">
                    <div v-if="block.title" class="card-title">{{ block.title }}</div>
                    <article v-for="(item, ii) in block.items" :key="ii" class="school-detail">
                      <div class="school-detail-head">
                        <h4>{{ item.school }}</h4>
                        <span
                          v-if="item.excel_row"
                          class="source-chip"
                          :title="sourceTooltip(message, item.excel_row)"
                        >{{ item.excel_row }}</span>
                      </div>
                      <div class="school-detail-content">
                        <div
                          v-for="(part, pi) in schoolDetailParts(item.text)"
                          :key="pi"
                          class="school-detail-line"
                        >
                          <span v-if="part.label" class="detail-label">{{ part.label }}</span>
                          <span class="detail-value">{{ part.value }}</span>
                        </div>
                      </div>
                    </article>
                  </section>

                  <section v-else class="card">
                    <div v-if="block.title" class="card-title">{{ block.title }}</div>
                    <p class="card-text">{{ block.text }}</p>
                  </section>
                </template>
              </div>

              <!-- 兜底：模型没按 NDJSON 输出时按 Markdown 渲染 -->
              <div
                v-else-if="message.content"
                class="message-content md"
                v-html="renderedContent(message, isStreaming && index === messages.length - 1)"
              ></div>

              <div
                v-else-if="isStreaming && index === messages.length - 1 && message.answerStarted"
                class="pending"
              >
                <span class="pending-dot"></span><span class="pending-dot"></span><span class="pending-dot"></span>
                正在组织结构化答案
              </div>

              <div v-if="message.warning" class="warning-box">{{ message.warning }}</div>

              <div v-if="message.webSources?.length" class="web-sources">
                <div class="web-sources-title">联网核验来源</div>
                <a
                  v-for="(source, sourceIndex) in message.webSources"
                  :key="source.url"
                  :href="source.url"
                  target="_blank"
                  rel="noopener noreferrer"
                >
                  <span>{{ sourceIndex + 1 }}. {{ source.title }}</span>
                  <b>{{ source.official ? '官方优先' : '待核验' }}</b>
                </a>
              </div>

              <!-- 召回依据：默认收起，避免刷屏 -->
              <div v-if="message.evidence?.length" class="evidence">
                <button class="evidence-head" @click="message.evidenceOpen = !message.evidenceOpen">
                  <span>
                    召回依据 · {{ routeLabel(message.route) }} · {{ message.evidence.length }} 条
                    <template v-if="message.retrieveMs"> · 检索 {{ (message.retrieveMs / 1000).toFixed(1) }}s</template>
                  </span>
                  <span class="evidence-weights">{{ message.evidenceOpen ? '收起 ▲' : '展开 ▼' }}</span>
                </button>
                <div v-if="message.evidenceOpen" class="evidence-body">
                  <div v-for="item in message.evidence" :key="item.excel_row" class="evidence-row">
                    <span class="ev-row">第 {{ item.excel_row }} 行</span>
                    <span
                      v-if="hasVector(message.route)"
                      class="ev-track"
                      :title="`余弦相似度 ${item.cosine_score}`"
                    >
                      <i :style="{ width: scoreWidth(item.cosine_score) }"></i>
                    </span>
                    <span class="ev-score">综合 {{ fmtScore(item.hybrid_score) }}</span>
                    <span
                      v-if="item.rerank_score !== null && item.rerank_score !== undefined"
                      class="ev-score rerank"
                    >重排 {{ fmtScore(item.rerank_score) }}</span>
                    <span class="ev-score muted">BM25 {{ fmtScore(item.bm25_score) }}</span>
                    <span v-if="hasVector(message.route)" class="ev-score muted">
                      余弦 {{ item.cosine_score >= 0 ? fmtScore(item.cosine_score) : '未入向量候选' }}
                    </span>
                  </div>
                </div>
              </div>
              <div v-else-if="message.citations?.length" class="citations">
                <span v-for="row in message.citations" :key="row">Excel 第 {{ row }} 行</span>
              </div>

              <div v-if="message.grounding" class="grounding">
                {{ message.grounding === 'excel_supported' ? 'Excel 数据支持' : '一般性建议' }}
              </div>
            </div>
          </article>
        </div>
      </section>

      <footer class="composer-wrap">
        <div class="composer">
          <textarea v-model="input" rows="1" placeholder="给志愿助手发送消息" :disabled="isStreaming" @keydown="handleKeydown"></textarea>
          <div class="composer-actions">
            <div class="mode-tools">
            <label class="web-toggle" title="仅在政策、章程和最新变化需要核实时联网">
              <input v-model="webEnabled" type="checkbox" :disabled="isStreaming" />
              <span>联网核验</span>
            </label>
            <div class="mode-chip">
              <span class="status-dot"></span>{{ retrievalMode }} 检索 · {{ embeddingShort }}
            </div>
            </div>
            <button v-if="isStreaming" class="send-button stop" title="停止生成" @click="stopGeneration">■</button>
            <button v-else class="send-button" :disabled="!canSend" title="发送" @click="sendMessage">
              <svg viewBox="0 0 24 24"><path d="m5 12 14-7-4 14-3-6-7-1Z" /></svg>
            </button>
          </div>
        </div>
        <p class="disclaimer">招生数据仅供志愿研究参考，正式填报请以考试院和院校官方信息为准。</p>
      </footer>
    </main>
  </div>
</template>

<style scoped>
/* ---------- Embedding / 混合检索 相关 ---------- */
.sidebar-footer .embed-line { color: #8fa1c2; font-size: 10px; }
.sidebar-footer .status-dot.warn { background: #f0b34a; box-shadow: 0 0 0 4px rgba(240, 179, 74, .12); }

.topbar-pills { display: flex; align-items: center; gap: 8px; }
.embed-pill { border-color: #cddbfa; background: #f2f6ff; color: #3b62c4; }
.embed-pill .pulse { background: #5b8dff; box-shadow: 0 0 0 4px rgba(91, 141, 255, .14); }

.stack-info { margin-top: 22px; display: flex; flex-wrap: wrap; justify-content: center; gap: 8px; }
.stack-item { padding: 7px 12px; border: 1px solid #e5e9f1; border-radius: 999px; background: rgba(255, 255, 255, .86); color: #5d6a80; font-size: 11.5px; display: inline-flex; align-items: center; gap: 6px; }
.stack-item b { color: #98a1b2; font-weight: 500; }

/* ---------- 四层记忆中的结构化考生档案 ---------- */
.profile-bar { position: relative; z-index: 3; margin: 10px 24px 0; border: 1px solid #e2e8f3; border-radius: 12px; background: rgba(255, 255, 255, .92); box-shadow: 0 4px 16px rgba(47, 66, 105, .05); }
.profile-toggle { width: 100%; padding: 9px 12px; border: 0; background: transparent; display: flex; align-items: center; gap: 9px; color: #53627b; text-align: left; }
.profile-icon { width: 27px; height: 27px; border-radius: 8px; background: #edf3ff; color: #4e70c5; display: grid; place-items: center; font-size: 11px; font-weight: 700; }
.profile-toggle > span:nth-child(2) { min-width: 0; display: flex; align-items: baseline; gap: 10px; }
.profile-toggle b { color: #31405b; font-size: 12.5px; }
.profile-toggle small { overflow: hidden; color: #8a96aa; font-size: 10.5px; text-overflow: ellipsis; white-space: nowrap; }
.profile-toggle i { margin-left: auto; color: #8793a7; font-size: 10px; font-style: normal; }
.profile-grid { padding: 4px 12px 12px 48px; display: flex; flex-wrap: wrap; gap: 8px; border-top: 1px solid #f0f2f7; }
.profile-grid > div { min-width: 120px; padding: 7px 9px; border-radius: 8px; background: #f7f9fd; display: flex; flex-direction: column; gap: 3px; }
.profile-grid span { color: #919caf; font-size: 9.5px; }
.profile-grid strong { color: #40506b; font-size: 11.5px; font-weight: 600; }
.profile-grid .profile-summary { flex: 1 1 100%; }
.profile-summary p { margin: 2px 0 0; max-height: 70px; overflow: auto; color: #5d6a80; font-size: 11px; line-height: 1.65; white-space: pre-wrap; }

/* ---------- 召回依据（可折叠） ---------- */
.evidence { margin-top: 14px; border: 1px solid #e5e9f1; border-radius: 12px; background: rgba(255, 255, 255, .72); overflow: hidden; }
.evidence-head { width: 100%; padding: 10px 13px; border: 0; background: transparent; color: #6d7b93; font-size: 11px; text-align: left; display: flex; justify-content: space-between; align-items: center; gap: 12px; }
.evidence-head:hover { background: rgba(91, 141, 255, .06); }
.evidence-weights { color: #9aa4b6; }
.evidence-body { padding: 4px 13px 10px; border-top: 1px solid #eef1f8; }
.evidence-row { padding: 6px 0; display: flex; align-items: center; gap: 10px; flex-wrap: wrap; border-top: 1px solid #f4f6fa; font-size: 10.5px; }
.evidence-row:first-child { border-top: 0; }
.ev-row { color: #4a63a8; font-variant-numeric: tabular-nums; white-space: nowrap; }
.ev-track { flex: 1 1 60px; min-width: 50px; height: 5px; border-radius: 999px; background: #eef2fa; overflow: hidden; }
.ev-track i { display: block; height: 100%; border-radius: 999px; background: linear-gradient(90deg, #5b8dff, #6c63ff); transition: width .4s ease; }
.ev-score { color: #56657e; font-variant-numeric: tabular-nums; white-space: nowrap; }
.ev-score.muted { color: #9aa4b6; }
.ev-score.rerank { color: #6c63ff; font-weight: 600; }

/* ---------- 思考过程面板 ---------- */
.thinking { margin-bottom: 14px; border: 1px solid #e3e8f2; border-radius: 12px; background: transparent; overflow: hidden; }
.thinking-head { width: 100%; padding: 10px 13px; border: 0; background: transparent; display: flex; align-items: center; gap: 9px; color: #5b6b86; font-size: 11.5px; text-align: left; }
.thinking-head:hover { background: transparent; }
.thinking-dot { flex: 0 0 auto; width: 7px; height: 7px; border-radius: 50%; background: #9aa8c0; }
.thinking-dot.live { background: #6c63ff; box-shadow: 0 0 0 4px rgba(108, 99, 255, .14); animation: thinkPulse 1.4s ease-in-out infinite; }
@keyframes thinkPulse { 0%, 100% { opacity: 1; transform: scale(1); } 50% { opacity: .35; transform: scale(.82); } }
.thinking-title { font-weight: 600; color: #4a5a76; }
.thinking-meta { color: #a3adc0; font-variant-numeric: tabular-nums; }
.thinking-toggle { margin-left: auto; color: #7d8aa3; }
.thinking-body {
  max-height: 340px; overflow-y: auto; padding: 12px 15px 14px; border-top: 1px solid #eef1f8;
  color: #5f6c81; font-size: 12.5px; line-height: 1.9; letter-spacing: .1px;
  white-space: pre-wrap; overflow-wrap: anywhere;
}

/* ---------- Markdown 正文排版 ---------- */
.message-content.md { white-space: normal; }
.message-content.md > *:first-child { margin-top: 0; }
.message-content.md > *:last-child { margin-bottom: 0; }
.message-content.md p { margin: 0 0 10px; }
.message-content.md h3,
.message-content.md h4,
.message-content.md h5 { margin: 18px 0 9px; color: #2b374a; font-weight: 650; line-height: 1.4; }
.message-content.md h3 { font-size: 15px; }
.message-content.md h4 { font-size: 14px; }
.message-content.md h5 { font-size: 13px; }
.message-content.md strong { color: #22304a; font-weight: 650; }
.message-content.md ul,
.message-content.md ol { margin: 0 0 10px; padding-left: 20px; }
.message-content.md li { margin: 3px 0; }
.message-content.md hr { margin: 16px 0; border: 0; border-top: 1px solid #e7eaf1; }
.message-content.md code { padding: 1px 5px; border-radius: 5px; background: #eef2fa; color: #4a5a76; font-size: 12px; }
.message-content.md .cite-ref { margin: 0 1px; padding: 1px 5px; border-radius: 5px; background: #eef2fa; color: #5a72b0; font-size: 10.5px; white-space: nowrap; font-variant-numeric: tabular-nums; }

/* 表格：横向可滚，不撑破气泡 */
.message-content.md .md-table { margin: 0 0 12px; overflow-x: auto; border: 1px solid #e5e9f1; border-radius: 10px; }
.message-content.md table { width: 100%; border-collapse: collapse; font-size: 12px; }
.message-content.md thead th { padding: 8px 10px; background: #f4f7fd; color: #4a5a76; font-weight: 600; text-align: left; white-space: nowrap; border-bottom: 1px solid #e5e9f1; }
.message-content.md tbody td { padding: 7px 10px; color: #46536a; vertical-align: top; border-bottom: 1px solid #f1f4f9; }
.message-content.md tbody tr:last-child td { border-bottom: 0; }
.message-content.md tbody tr:nth-child(even) { background: #fafbfe; }

/* ---------- 结构化卡片（NDJSON 块渲染） ---------- */
.blocks { display: flex; flex-direction: column; gap: 12px; }

.card { border: 1px solid #e6eaf2; border-radius: 13px; background: #fff; padding: 14px 16px; box-shadow: 0 1px 2px rgba(35, 55, 95, .04); }
.card-title { margin-bottom: 10px; color: #33415c; font-size: 13.5px; font-weight: 650; letter-spacing: .2px; display: flex; align-items: center; gap: 7px; }
.card-title::before { content: ""; width: 3px; height: 13px; border-radius: 2px; background: linear-gradient(180deg, #5b8dff, #6c63ff); }
.card-text { margin: 0; color: #46536a; font-size: 13.5px; line-height: 1.9; }
.card-list { margin: 0; padding-left: 18px; color: #46536a; font-size: 13.5px; line-height: 1.9; }
.card-list li { margin: 3px 0; }
.card-list li::marker { color: #8fa6d6; }

.card-conclusion { border-color: #d6e2fb; background: linear-gradient(180deg, #f6f9ff, #eef4ff); }
.card-label { margin-bottom: 7px; color: #4f74c8; font-size: 10.5px; font-weight: 600; letter-spacing: 1.2px; }
.card-lead { margin: 0; color: #24304a; font-size: 14.5px; line-height: 1.85; font-weight: 550; }

.card-risks { border-color: #f2e2c8; background: linear-gradient(180deg, #fffcf6, #fdf6ea); }
.school-details { padding: 14px 18px; }
.school-detail { position: relative; margin: 0 -2px; padding: 18px 16px 18px 20px; border-bottom: 1px solid #e8edf5; }
.school-detail::before { content: ""; position: absolute; left: 0; top: 20px; width: 4px; height: 20px; border-radius: 4px; background: linear-gradient(180deg, #4f7fe8, #7468ee); }
.school-detail:first-of-type { padding-top: 4px; }
.school-detail:first-of-type::before { top: 6px; }
.school-detail:last-child { padding-bottom: 0; border-bottom: 0; }
.school-detail-head { display: flex; align-items: center; gap: 10px; margin-bottom: 11px; }
.school-detail-head h4 { margin: 0; color: #182640; font-size: 16px; font-weight: 750; letter-spacing: .1px; }
.school-detail-content { display: flex; flex-direction: column; gap: 7px; }
.school-detail-line { display: grid; grid-template-columns: 72px minmax(0, 1fr); gap: 10px; align-items: start; color: #59677c; font-size: 13px; line-height: 1.75; }
.detail-label { margin-top: 1px; padding: 2px 7px; border-radius: 6px; background: #f0f4fc; color: #4d6494; font-size: 11px; font-weight: 650; text-align: center; white-space: nowrap; }
.detail-value { color: #4b5870; text-wrap: pretty; }
.card-risks .card-label { color: #b07d24; }
.card-risks .card-list li::marker { color: #d9a94e; }

/* 数据表：表头吸顶、斑马纹、悬停高亮、横向可滚 */
.table-scroll { margin: 0 -4px; overflow-x: auto; border-radius: 10px; }
.data-table { width: 100%; border-collapse: separate; border-spacing: 0; font-size: 12.5px; }
.data-table th { position: sticky; top: 0; z-index: 1; padding: 9px 11px; text-align: left; white-space: nowrap; color: #4c5a75; font-weight: 600; background: #f4f7fd; border-bottom: 1px solid #e3e9f4; }
.data-table th:first-child { border-top-left-radius: 9px; }
.data-table th:last-child { border-top-right-radius: 9px; }
.data-table td { padding: 8px 11px; color: #47546b; border-bottom: 1px solid #f1f4f9; vertical-align: top; line-height: 1.7; }
.data-table td.school-cell { color: #243653; font-weight: 700; }
.data-table tbody tr:last-child td { border-bottom: 0; }
.data-table tbody tr:nth-child(even) { background: #fafbfe; }
.data-table tbody tr:hover { background: #f3f7ff; }
.col-source { width: 1%; white-space: nowrap; text-align: right; }
.source-chip { display: inline-block; padding: 2px 8px; border-radius: 999px; background: #eef3ff; color: #4d6cbe; border: 1px solid #dde7fb; font-size: 10.5px; font-variant-numeric: tabular-nums; cursor: help; }
.source-chip:hover { background: #e2ebff; border-color: #c9daf8; }
.source-empty { color: #c3cad6; }

/* 可读性：表格需要更宽的容器，助手正文用满可用宽度 */
.messages { max-width: 968px; }
.message.assistant .message-body { max-width: 100%; }
.message-content, .card-text, .card-list, .thinking-body { text-wrap: pretty; }

/* 生成中提示 */
.pending { display: flex; align-items: center; gap: 6px; padding: 10px 2px; color: #98a1b2; font-size: 12.5px; }
.pending-dot { width: 6px; height: 6px; border-radius: 50%; background: #b8c4d8; animation: pendingBlink 1.2s ease-in-out infinite; }
.pending-dot:nth-child(2) { animation-delay: .18s; }
.pending-dot:nth-child(3) { animation-delay: .36s; }
@keyframes pendingBlink { 0%, 100% { opacity: .25; transform: translateY(0); } 50% { opacity: 1; transform: translateY(-2px); } }
</style>
