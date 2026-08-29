import assert from "node:assert/strict";
import test from "node:test";

import {
  acceptanceRequested,
  assetSetSha256,
  canonicalAssetIds,
  validateAcceptanceConfig,
} from "../src/acceptance.js";


const HASH = "a".repeat(64);

test("query or config can enable fail-closed acceptance mode", () => {
  assert.equal(acceptanceRequested(new URLSearchParams("acceptance=1")), true);
  assert.equal(
    acceptanceRequested(new URLSearchParams(), { acceptance_mode: true }),
    true,
  );
  assert.equal(acceptanceRequested(new URLSearchParams(), {}), false);
});

test("acceptance config requires release and content hashes", () => {
  assert.deepEqual(
    validateAcceptanceConfig({
      release_id: "release-001",
      model_url: "/scene.glb",
      model_sha256: HASH,
      registry_url: "/registry.json",
      registry_sha256: HASH,
      asset_set_sha256: HASH,
    }),
    [],
  );
  assert.ok(validateAcceptanceConfig({ model_url: "/scene.glb" }).length >= 5);
});

test("asset set hashing is stable across registry order", async () => {
  const left = [{ id: "TRACK-002" }, { id: "TRACK-001" }];
  const right = [{ id: "TRACK-001" }, { id: "TRACK-002" }];
  assert.equal(canonicalAssetIds(left), "TRACK-001\nTRACK-002");
  assert.equal(await assetSetSha256(left), await assetSetSha256(right));
});
