import assert from "node:assert/strict";
import test from "node:test";
import { fetchJson, missingProject, ResourceError, selectProject, validateProject } from "../src/project-loading.js";

test("default, explicit project, and opt-in demo selection remain separate", () => {
  assert.deepEqual(selectProject(new URLSearchParams()), { url: "/project.json", demo: false, explicit: false });
  assert.deepEqual(selectProject(new URLSearchParams("demo=1")), { url: "/demo/project.json", demo: true, explicit: true });
  assert.equal(selectProject(new URLSearchParams("demo=0")).url, "/project.json");
  assert.equal(selectProject(new URLSearchParams("project=project.track-boundary.json&demo=1")).url, "/project.track-boundary.json");
});

test("invalid explicit selections fail instead of falling back to private defaults", () => {
  for (const name of ["", "../project.json", "/demo/project.json", "https://example.test/x.json", "x\\y.json", "x.html", ".json"]) {
    assert.throws(() => selectProject(new URLSearchParams({ project: name })), /JSON 文件名/);
  }
});

test("only missing default or demo config enables onboarding", () => {
  for (const query of ["", "demo=1", "project=alternate.json"]) {
    const selected = selectProject(new URLSearchParams(query));
    assert.equal(missingProject(new ResourceError(selected.url, "not found", 404), selected), !query.startsWith("project="));
    for (const status of [401, 403, 500, null]) {
      assert.equal(missingProject(new ResourceError(selected.url, "blocked", status), selected), false);
    }
    assert.equal(missingProject(new ResourceError("/model.json", "missing", 404), selected), false);
  }
});

test("optional means absent or 404, never invalid JSON, network, or security failures", async () => {
  assert.equal(await fetchJson(undefined, true), null);
  await assert.rejects(fetchJson(undefined), /缺少必需/);
  assert.equal(await fetchJson("/optional.json", true, async () => new Response("", { status: 404 })), null);
  for (const status of [401, 403, 429, 500]) {
    await assert.rejects(fetchJson("/optional.json", true, async () => new Response("", { status })), (error) => error.status === status);
  }
  await assert.rejects(fetchJson("/optional.json", true, async () => { throw new Error("network unavailable"); }), /network unavailable/);
  await assert.rejects(fetchJson("/optional.json", true, async () => new Response("<html>fallback</html>")), /不是有效 JSON/);
});

test("JSON requests avoid HTML history fallback and preserve exact values", async () => {
  const value = await fetchJson("/project.json", false, async (url, options) => {
    assert.equal(url, "/project.json");
    assert.equal(options.headers.Accept, "application/json");
    return Response.json({ model_glb_url: "/scene.glb" });
  });
  assert.deepEqual(value, { model_glb_url: "/scene.glb" });
});

test("demo requires explicit synthetic marking and required model/audit fields", () => {
  const config = { synthetic_demo: true, model_glb_url: "/demo/scene.glb", mesh_audit_url: "/demo/mesh_audit.json" };
  const selection = selectProject(new URLSearchParams("demo=1"));
  assert.equal(validateProject(config, selection), config);
  assert.throws(() => validateProject({ ...config, synthetic_demo: false }, selection), /synthetic_demo/);
  assert.throws(() => validateProject({ ...config, mesh_audit_url: null }, selection), /mesh_audit_url/);
  assert.throws(() => validateProject({}, selection), /model_glb_url/);
  assert.throws(() => validateProject([], selection), /JSON 对象/);
});
