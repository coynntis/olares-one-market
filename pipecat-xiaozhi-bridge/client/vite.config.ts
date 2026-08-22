import react from "@vitejs/plugin-react";
import { defineConfig } from "vite";

// Dev: proxy WebSocket to local bridge so one origin + no mixed-content issues.
export default defineConfig({
  plugins: [react()],
  server: {
    port: 5173,
    proxy: {
      "/xiaozhi": {
        target: "http://127.0.0.1:8000",
        ws: true,
        changeOrigin: true,
      },
      "/api": {
        target: "http://127.0.0.1:8000",
        changeOrigin: true,
      },
    },
  },
});
