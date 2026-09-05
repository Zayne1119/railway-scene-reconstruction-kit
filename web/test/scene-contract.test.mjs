import assert from "node:assert/strict";
import { existsSync, readFileSync } from "node:fs";
import test from "node:test";
import * as THREE from "three";
import { GLTFLoader } from "three/addons/loaders/GLTFLoader.js";
import { perspectiveFitDistance } from "../src/scene-math.js";

test("automatic fit keeps an elongated model inside both landscape and portrait viewports", () => {
  const box = new THREE.Box3(new THREE.Vector3(0, -1.75, -0.4), new THREE.Vector3(40.025, 7.8, 5.46));
  const sphere = box.getBoundingSphere(new THREE.Sphere());
  for (const aspect of [16 / 9, 1, 9 / 16, 0.35]) {
    const camera = new THREE.PerspectiveCamera(42, aspect, 0.05, 5000);
    camera.up.set(0, 0, 1);
    const distance = perspectiveFitDistance(sphere.radius, camera.fov, camera.aspect);
    camera.position.copy(sphere.center).add(new THREE.Vector3(0.7, -0.55, 0.42).normalize().multiplyScalar(distance));
    camera.lookAt(sphere.center);
    camera.updateMatrixWorld(true);
    for (const x of [box.min.x, box.max.x]) for (const y of [box.min.y, box.max.y]) for (const z of [box.min.z, box.max.z]) {
      const ndc = new THREE.Vector3(x, y, z).project(camera);
      assert.ok(Math.abs(ndc.x) < 1 && Math.abs(ndc.y) < 1 && Math.abs(ndc.z) < 1, `Model clipped at aspect ${aspect}`);
    }
  }
});

test("fit preserves the landscape distance and rejects non-finite inputs", () => {
  assert.ok(Math.abs(perspectiveFitDistance(20, 42, 16 / 9) - 20 / Math.sin(21 * Math.PI / 180) * 1.28) < 1e-10);
  assert.ok(perspectiveFitDistance(20, 42, 9 / 16) > perspectiveFitDistance(20, 42, 16 / 9));
  for (const aspect of [0, -1, NaN, Infinity]) assert.throws(() => perspectiveFitDistance(20, 42, aspect));
});

const demoDirectory = new URL("../public/demo/", import.meta.url);
const demoExists = existsSync(new URL("scene.glb", demoDirectory));

test("generated synthetic demo loads through real GLTFLoader with matching registry and audit", { skip: !demoExists }, async () => {
  const json = name => JSON.parse(readFileSync(new URL(name, demoDirectory), "utf8"));
  const config = json("project.json");
  const registry = json("asset_registry.json");
  const audit = json("mesh_audit.json");
  const origin = json("model_origin.json");
  assert.equal(config.synthetic_demo, true);
  assert.equal(registry.synthetic_demo, true);
  assert.equal(origin.axis, "Z-up");
  assert.deepEqual(origin.origin_xyz, [0, 0, 0]);
  const bytes = readFileSync(new URL("scene.glb", demoDirectory));
  const model = (await new GLTFLoader().parseAsync(bytes.buffer.slice(bytes.byteOffset, bytes.byteOffset + bytes.byteLength), "")).scene;
  model.updateMatrixWorld(true);
  const meshes = [];
  model.traverse(object => { if (object.isMesh) meshes.push(object); });
  const names = new Set(meshes.map(object => object.name));
  assert.equal(names.size, meshes.length);
  assert.equal(meshes.length, audit.object_count);
  assert.equal(registry.assets.length, meshes.length);
  for (const asset of registry.assets) assert.ok(names.has(asset.geometry.node), `Unbound registry node ${asset.id}`);
  assert.equal(meshes.reduce((sum, object) => sum + (object.geometry.index?.count ?? object.geometry.attributes.position.count) / 3, 0), audit.triangle_count_after_fan_triangulation);
  const box = new THREE.Box3().setFromObject(model);
  for (let axis = 0; axis < 3; axis++) {
    assert.ok(Math.abs(box.min.getComponent(axis) - audit.bounds.minimum[axis]) < 1e-4);
    assert.ok(Math.abs(box.max.getComponent(axis) - audit.bounds.maximum[axis]) < 1e-4);
  }
  assert.ok(meshes.some(object => object.name.startsWith("TRACK--")));
  assert.ok(meshes.some(object => object.name.startsWith("STATION--")));
});
