import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

export default defineConfig({
  plugins: [react()],
  server: {
    port: 5173,
    // Proxy API calls in dev so the browser never hits CORS issues
    proxy: {
      "/forecast": "http://localhost:8000",
      "/drift":    "http://localhost:8000",
      "/inventory":"http://localhost:8000",
    },
  },
});
