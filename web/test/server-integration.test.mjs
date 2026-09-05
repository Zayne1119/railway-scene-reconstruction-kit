import assert from "node:assert/strict";
import http from "node:http";
import net from "node:net";
import path from "node:path";
import test from "node:test";
import { fileURLToPath } from "node:url";
import { createServer } from "vite";
import { viewerServerConfig } from "../server-config.js";

const root = fileURLToPath(new URL("..", import.meta.url));

function testConfig(port) {
  const config = viewerServerConfig(root, {});
  return { ...config, configFile: false, logLevel: "silent", server: { ...config.server, port, watch: null }, optimizeDeps: { noDiscovery: true, include: [] } };
}

test("live server returns real missing-JSON 404 and blocks files outside the allowlist", async () => {
  const server = await createServer(testConfig(0));
  try {
    await server.listen();
    const address = server.httpServer.address();
    assert.equal(address.address, "127.0.0.1");
    const base = `http://127.0.0.1:${address.port}`;
    const missing = await fetch(`${base}/__onboarding_missing__.json`, { headers: { Accept: "application/json" } });
    assert.equal(missing.status, 404);
    const outside = path.resolve(root, "..", "LICENSE").replaceAll("\\", "/");
    const forbidden = await fetch(`${base}/@fs/${outside}`);
    assert.equal(forbidden.status, 403);
    const foreignHostStatus = await new Promise((resolve, reject) => {
      http.get(base, { headers: { Host: "untrusted.invalid" } }, (response) => {
        response.resume();
        resolve(response.statusCode);
      }).once("error", reject);
    });
    assert.equal(foreignHostStatus, 403);
    const local = await fetch(base, { headers: { Origin: "https://untrusted.invalid" } });
    assert.equal(local.headers.get("access-control-allow-origin"), null);
    assert.ok((await local.text()).includes('data-testid="onboarding"'));
  } finally {
    await server.close();
  }
});

test("occupied port fails without silently choosing another port", async () => {
  const blocker = net.createServer();
  await new Promise((resolve, reject) => { blocker.once("error", reject); blocker.listen(0, "127.0.0.1", resolve); });
  const server = await createServer(testConfig(blocker.address().port));
  try {
    await assert.rejects(server.listen(), /already in use/);
  } finally {
    await server.close();
    await new Promise((resolve, reject) => blocker.close((error) => error ? reject(error) : resolve()));
  }
});
