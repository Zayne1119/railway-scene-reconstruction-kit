import { defineConfig } from "vite";
import { fileURLToPath } from "node:url";
import { viewerServerConfig } from "./server-config.js";

export default defineConfig(viewerServerConfig(fileURLToPath(new URL(".", import.meta.url))));
