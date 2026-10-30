import react from "@vitejs/plugin-react";
import { defineConfig } from "vite";

export default defineConfig({
  plugins: [react()],
  server: {
    proxy: { "/api": process.env.MINUTES_API ?? "http://localhost:8000" },
  },
});
