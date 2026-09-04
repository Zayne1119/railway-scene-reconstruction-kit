import { defineConfig } from "vite";

export default defineConfig({
  server: {
    host: "0.0.0.0",
    port: 3010,
    fs: { allow: ["D:/Railway"] },
  },
  preview: {
    host: "0.0.0.0",
    port: 3010,
  },
});
