import { defineConfig } from 'vite'
import vue from '@vitejs/plugin-vue'

// PyCharm / VS Code 的原子保存会在源码目录里短暂生成 .tmp / .*.tmpdir 临时文件。
// Windows 下 chokidar 去 watch 这些被锁定的文件会抛 EBUSY，而该错误是 FSWatcher 上的
// 未捕获 'error' 事件，会直接终止整个 Vite 进程（表现为 5173 突然打不开）。
// 显式忽略这些临时产物，避免"保存动作"把开发服务器打崩。
const WATCH_IGNORED = [
  '**/.*.tmpdir/**',
  '**/*.tmpdir/**',
  '**/*.tmp',
  '**/*.swp',
  '**/*~',
  '**/.~*',
  '**/node_modules/**',
  '**/dist/**',
]

export default defineConfig({
  plugins: [vue()],
  server: {
    host: '127.0.0.1',
    port: 5173,
    watch: { ignored: WATCH_IGNORED },
    proxy: { '/api': { target: 'http://127.0.0.1:8000', changeOrigin: true } },
  },
})
