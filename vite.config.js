import { defineConfig } from 'vite';
import react from '@vitejs/plugin-react';

export default defineConfig({
  plugins: [react()],
  server: {
    host: 'localhost',
    port: 5173,
    open: false,
    watch: {
      ignored: [
        '**/.venv/**',
        '**/backend/**',
        '**/.git/**',
        '**/dist/**',
        '**/benchmarks/**',
        '**/.pytest_cache/**',
        '**/*.pth',
        '**/*.safetensors',
        '**/*.zip',
      ],
    },
  },
  optimizeDeps: {
    include: ['react', 'react-dom', 'lucide-react'],
  },
});
