import assert from "node:assert/strict";
import { readFileSync, realpathSync } from "node:fs";
import { homedir } from "node:os";
import path from "node:path";
import test from "node:test";
import { fileURLToPath } from "node:url";
import { viewerServerConfig } from "../server-config.js";

const root = fileURLToPath(new URL("..", import.meta.url));

test("portable defaults bind only loopback with a fixed port and narrow filesystem root", () => {
  const config = viewerServerConfig(root, {});
  assert.equal(config.root, realpathSync(root));
  assert.deepEqual(config.server.fs, { strict: true, allow: [realpathSync(root)] });
  for (const options of [config.server, config.preview]) {
    assert.equal(options.host, "127.0.0.1");
    assert.equal(options.port, 3010);
    assert.equal(options.strictPort, true);
    assert.equal(options.cors, false);
    assert.notEqual(options.allowedHosts, true);
  }
});

test("external roots require explicit existing dedicated directories", () => {
  const approved = path.join(root, "public");
  assert.deepEqual(viewerServerConfig(root, { RECON_VIEWER_DATA_ROOT: approved }).server.fs.allow, [realpathSync(root), realpathSync(approved)]);
  for (const forbidden of ["../data", " /data ", path.parse(root).root, root, path.dirname(root), homedir(), path.join(root, "package.json"), path.join(root, "missing-not-a-directory")]) {
    assert.throws(() => viewerServerConfig(root, { RECON_VIEWER_DATA_ROOT: forbidden }), /RECON_VIEWER_DATA_ROOT/);
  }
});

test("checked-in entry files contain no fixed drive, LAN listener, or remote font request", () => {
  for (const name of ["vite.config.js", "server-config.js", "index.html", "src/main.js", "src/style.css", "public/project.example.json", "public/project.track-boundary.example.json"]) {
    const text = readFileSync(path.join(root, name), "utf8");
    assert.doesNotMatch(text, /[A-Za-z]:[\\/]|0\.0\.0\.0|fonts\.googleapis/);
  }
  const pkg = JSON.parse(readFileSync(path.join(root, "package.json"), "utf8"));
  assert.equal(pkg.engines.node, ">=22");
  assert.equal(pkg.scripts.dev, "vite");
  assert.equal(pkg.scripts.preview, "vite preview");
  const lock = JSON.parse(readFileSync(path.join(root, "package-lock.json"), "utf8"));
  assert.equal(lock.packages[""].engines.node, pkg.engines.node);
});

test("DOM exposes real model readiness, synthetic labeling, and persistent retry entry", () => {
  const html = readFileSync(path.join(root, "index.html"), "utf8");
  for (const id of ["viewer-status", "model-object-count", "synthetic-banner", "onboarding", "load-error", "open-demo"]) {
    assert.ok(html.includes(`data-testid="${id}"`));
  }
  assert.ok(html.includes("data-model-loaded=\"false\""));
  assert.ok(html.includes("data-retry"));
  assert.doesNotMatch(html, /4\.4|49 \/ 89|156 \/ 172|950|SITE B/);
});
