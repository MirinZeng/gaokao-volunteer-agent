/**
 * 轻量 Markdown 渲染器。
 *
 * 设计要点：**先转义 HTML，再套用白名单格式**。因此天然免疫 XSS ——
 * Excel 单元格里即便塞进 <script> 或 onclick，也只会以文本形式显示。
 * 只支持模型实际会产出的子集：标题、表格、粗体、行内代码、有序/无序列表、分隔线、段落。
 */

const HTML_ESCAPES = { '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }

export function escapeHtml(text) {
  return String(text).replace(/[&<>"']/g, (char) => HTML_ESCAPES[char])
}

export function renderInline(text) {
  let html = escapeHtml(text)
  html = html.replace(/\*\*(.+?)\*\*/g, '<strong>$1</strong>')
  html = html.replace(/`([^`]+)`/g, '<code>$1</code>')
  // [Excel第N行] 做成可辨识的引用徽标
  html = html.replace(/\[Excel第(\d+)行\]/g, '<span class="cite-ref">[Excel第$1行]</span>')
  return html
}

export function splitTableRow(line) {
  return line
    .replace(/^\s*\|/, '')
    .replace(/\|\s*$/, '')
    .split('|')
    .map((cell) => cell.trim())
}

const TABLE_DIVIDER = /^\s*\|?[\s:|-]+\|[\s:|-]*$/

export function renderTable(headers, rows) {
  const head = headers.map((cell) => `<th>${renderInline(cell)}</th>`).join('')
  const body = rows
    .map(
      (row) =>
        `<tr>${headers
          .map((_, index) => `<td>${renderInline(row[index] ?? '')}</td>`)
          .join('')}</tr>`,
    )
    .join('')
  return `<div class="md-table"><table><thead><tr>${head}</tr></thead><tbody>${body}</tbody></table></div>`
}

export function renderMarkdown(source) {
  if (!source) return ''
  const lines = String(source).replace(/\r\n?/g, '\n').split('\n')
  const blocks = []
  let listItems = []
  let listTag = 'ul'

  const flushList = () => {
    if (!listItems.length) return
    blocks.push(`<${listTag}>${listItems.map((item) => `<li>${item}</li>`).join('')}</${listTag}>`)
    listItems = []
  }

  for (let index = 0; index < lines.length; index += 1) {
    const line = lines[index].trim()
    if (!line) {
      flushList()
      continue
    }

    // 表格：当前行以 | 开头，且下一行是 |---|---| 这样的分隔行
    if (line.startsWith('|') && index + 1 < lines.length && TABLE_DIVIDER.test(lines[index + 1])) {
      flushList()
      const headers = splitTableRow(line)
      const rows = []
      index += 2
      while (index < lines.length && lines[index].trim().startsWith('|')) {
        rows.push(splitTableRow(lines[index].trim()))
        index += 1
      }
      index -= 1
      blocks.push(renderTable(headers, rows))
      continue
    }

    // 标题：降一级，避免与页面标题抢视觉层级（## → h3，### → h4）
    const heading = /^(#{1,6})\s+(.*)$/.exec(line)
    if (heading) {
      flushList()
      const level = Math.min(heading[1].length + 1, 6)
      blocks.push(`<h${level}>${renderInline(heading[2])}</h${level}>`)
      continue
    }

    if (/^(-{3,}|\*{3,}|_{3,})$/.test(line)) {
      flushList()
      blocks.push('<hr>')
      continue
    }

    const item = /^([-*+]|\d+[.)])\s+(.*)$/.exec(line)
    if (item) {
      const tag = /^\d/.test(item[1]) ? 'ol' : 'ul'
      if (listItems.length && tag !== listTag) flushList()
      listTag = tag
      listItems.push(renderInline(item[2]))
      continue
    }

    flushList()
    blocks.push(`<p>${renderInline(line)}</p>`)
  }

  flushList()
  return blocks.join('')
}

export function renderedContent(message, streaming) {
  return renderMarkdown(message.content) + (streaming ? '<span class="cursor"></span>' : '')
}
