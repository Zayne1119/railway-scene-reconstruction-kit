import * as THREE from "three";
import { OrbitControls } from "three/addons/controls/OrbitControls.js";
import { GLTFLoader } from "three/addons/loaders/GLTFLoader.js";
import {
  acceptanceRequested,
  assetSetSha256,
  sha256Hex,
  validateAcceptanceConfig,
} from "./acceptance.js";
import "./style.css";

const canvas = document.querySelector("#viewport");
const status = document.querySelector("#status");
const title = document.querySelector("#title");
const details = document.querySelector("#details");
const message = document.querySelector("#message");
const renderer = new THREE.WebGLRenderer({ canvas, antialias: true, alpha: true, logarithmicDepthBuffer: true });
renderer.setPixelRatio(Math.min(devicePixelRatio, 2));
renderer.outputColorSpace = THREE.SRGBColorSpace;
renderer.toneMapping = THREE.ACESFilmicToneMapping;
renderer.toneMappingExposure = 1.05;

const scene = new THREE.Scene();
const camera = new THREE.PerspectiveCamera(45, 1, 0.05, 100000);
camera.up.set(0, 0, 1);
camera.position.set(28, -34, 24);
const controls = new OrbitControls(camera, canvas);
controls.enableDamping = true;
controls.screenSpacePanning = true;
scene.add(new THREE.HemisphereLight(0xdff5ff, 0x23313a, 2.2));
const sun = new THREE.DirectionalLight(0xffffff, 3.1);
sun.position.set(-20, -15, 35);
scene.add(sun);
scene.add(new THREE.GridHelper(200, 40, 0x2b6472, 0x153b47).rotateX(Math.PI / 2));

let modelRoot = null;
let registry = new Map();
let selected = null;
let originalMaterial = null;
const raycaster = new THREE.Raycaster();
const pointer = new THREE.Vector2();

function showMessage(text) { message.textContent = text; message.hidden = false; }
function failClosed(text) {
  status.textContent = "验收失败";
  showMessage(text);
  canvas.dataset.acceptance = "failed";
}
function assetIdFor(object) {
  let current = object;
  while (current) {
    if (current.userData?.asset_id) return current.userData.asset_id;
    if (registry.has(current.name)) return current.name;
    current = current.parent;
  }
  return object.name || "UNREGISTERED";
}
function showAsset(id) {
  const asset = registry.get(id);
  if (!asset) {
    details.innerHTML = `<h2>${id}</h2><p>模型节点未在资产注册表中找到；交付前必须补齐映射。</p>`;
    return;
  }
  const limitations = (asset.limitations || []).join("；") || "无记录";
  details.innerHTML = `<h2>${asset.id}</h2><dl><dt>类型</dt><dd>${asset.type}</dd><dt>证据</dt><dd>${asset.evidence_level}</dd><dt>置信度</dt><dd>${asset.confidence}</dd><dt>状态</dt><dd>${asset.status}</dd><dt>里程</dt><dd>${asset.chainage_m ?? "未填写"}</dd><dt>限制</dt><dd>${limitations}</dd></dl>`;
}
function fitCamera(root) {
  const box = new THREE.Box3().setFromObject(root);
  if (box.isEmpty()) return;
  const sphere = box.getBoundingSphere(new THREE.Sphere());
  const radius = Math.max(sphere.radius, 1);
  controls.target.copy(sphere.center);
  camera.position.copy(sphere.center).add(new THREE.Vector3(radius * 1.25, -radius * 1.55, radius * 0.9));
  camera.near = Math.max(radius / 10000, 0.01);
  camera.far = radius * 100;
  camera.updateProjectionMatrix();
  controls.update();
}
function addSyntheticPlaceholder() {
  const group = new THREE.Group();
  const rail = new THREE.MeshStandardMaterial({ color: 0x6e8087, metalness: .65, roughness: .3 });
  const bed = new THREE.MeshStandardMaterial({ color: 0x504a42, roughness: .95 });
  for (const y of [-0.72, 0.72, 4.28, 5.72]) {
    const mesh = new THREE.Mesh(new THREE.BoxGeometry(40, .08, .14), rail);
    mesh.position.set(0, y, .25);
    group.add(mesh);
  }
  for (const y of [0, 5]) {
    const mesh = new THREE.Mesh(new THREE.BoxGeometry(40, 3.4, .35), bed);
    mesh.position.set(0, y, -.15);
    group.add(mesh);
  }
  scene.add(group);
  modelRoot = group;
  fitCamera(group);
}

async function load() {
  const query = new URLSearchParams(location.search);
  const configUrl = query.get("config") || "/project.json";
  let acceptanceMode = acceptanceRequested(query);
  let config;
  try {
    const response = await fetch(configUrl, { cache: "no-store" });
    if (!response.ok) throw new Error(`${response.status} ${response.statusText}`);
    config = await response.json();
  } catch (error) {
    if (acceptanceMode) {
      failClosed(`无法加载验收配置 ${configUrl}：${error.message || error}`);
      return;
    }
    status.textContent = "等待项目配置";
    showMessage(`未找到 ${configUrl}。已显示合成占位场景；复制 public/project.example.json 为 public/project.json 并填写模型与注册表地址。`);
    addSyntheticPlaceholder();
    return;
  }
  acceptanceMode = acceptanceRequested(query, config);
  if (acceptanceMode) {
    const configErrors = validateAcceptanceConfig(config);
    if (configErrors.length) {
      failClosed(configErrors.join("；"));
      return;
    }
    canvas.dataset.acceptance = "verifying";
  }
  title.textContent = config.title || "铁路场景重建验收";
  let registryValue;
  try {
    const registryResponse = await fetch(config.registry_url, { cache: "no-store" });
    if (!registryResponse.ok) {
      throw new Error(`${registryResponse.status} ${registryResponse.statusText}`);
    }
    const registryBytes = await registryResponse.arrayBuffer();
    if (acceptanceMode) {
      const actualHash = await sha256Hex(registryBytes);
      if (actualHash !== config.registry_sha256.toLowerCase()) {
        throw new Error(`注册表 SHA-256 不一致：${actualHash}`);
      }
    }
    registryValue = JSON.parse(new TextDecoder().decode(registryBytes));
    const assets = registryValue.assets || [];
    registry = new Map(assets.map((asset) => [asset.id, asset]));
    if (acceptanceMode) {
      if (registryValue.release_id !== config.release_id) {
        throw new Error("注册表 release_id 与项目配置不一致");
      }
      const actualAssetSet = await assetSetSha256(assets);
      if (actualAssetSet !== config.asset_set_sha256.toLowerCase()) {
        throw new Error(`资产集合 SHA-256 不一致：${actualAssetSet}`);
      }
      const blocking = assets.filter((asset) => asset.status !== "accepted");
      if (!assets.length || blocking.length) {
        throw new Error(`clean 注册表包含 ${blocking.length} 个非 accepted 资产`);
      }
    }
  } catch (error) {
    if (acceptanceMode) {
      failClosed(`资产注册表验收失败：${error.message || error}`);
      return;
    }
    showMessage(`资产注册表加载失败；只能浏览模型：${error.message || error}`);
  }
  status.textContent = "加载模型";
  try {
    const modelResponse = await fetch(config.model_url, { cache: "no-store" });
    if (!modelResponse.ok) throw new Error(`${modelResponse.status} ${modelResponse.statusText}`);
    const modelBytes = await modelResponse.arrayBuffer();
    if (acceptanceMode) {
      const actualHash = await sha256Hex(modelBytes);
      if (actualHash !== config.model_sha256.toLowerCase()) {
        throw new Error(`模型 SHA-256 不一致：${actualHash}`);
      }
    }
    const modelUrl = new URL(config.model_url, location.href);
    const basePath = modelUrl.href.slice(0, modelUrl.href.lastIndexOf("/") + 1);
    const gltf = await new Promise((resolve, reject) => {
      new GLTFLoader().parse(modelBytes, basePath, resolve, reject);
    });
    modelRoot = gltf.scene;
    if (acceptanceMode) {
      const missing = new Set();
      modelRoot.traverse((object) => {
        if (!object.isMesh) return;
        const id = assetIdFor(object);
        if (!registry.has(id)) missing.add(id);
      });
      if (missing.size) {
        throw new Error(`存在 ${missing.size} 个未注册 Mesh 节点：${[...missing].slice(0, 10).join(", ")}`);
      }
    }
    scene.add(modelRoot);
    fitCamera(modelRoot);
    status.textContent = acceptanceMode
      ? `${registry.size} 项资产 · 验收哈希匹配`
      : `${registry.size} 项资产 · 模型已加载`;
    if (acceptanceMode) canvas.dataset.acceptance = "passed";
  } catch (error) {
    modelRoot = null;
    if (acceptanceMode) {
      failClosed(`模型验收失败：${error.message || error}`);
      return;
    }
    status.textContent = "模型加载失败";
    showMessage(`无法加载模型：${error.message || error}`);
    addSyntheticPlaceholder();
  }
}

canvas.addEventListener("pointerdown", (event) => {
  if (!modelRoot) return;
  const rect = canvas.getBoundingClientRect();
  pointer.x = ((event.clientX - rect.left) / rect.width) * 2 - 1;
  pointer.y = -((event.clientY - rect.top) / rect.height) * 2 + 1;
  raycaster.setFromCamera(pointer, camera);
  const hit = raycaster.intersectObject(modelRoot, true).find((item) => item.object.isMesh);
  if (!hit) return;
  if (selected && originalMaterial) selected.material = originalMaterial;
  selected = hit.object;
  originalMaterial = selected.material;
  selected.material = new THREE.MeshStandardMaterial({ color: 0xb9ff48, emissive: 0x284500 });
  showAsset(assetIdFor(selected));
});

document.querySelector("#reset").addEventListener("click", () => modelRoot && fitCamera(modelRoot));
document.querySelector("#search").addEventListener("change", (event) => {
  const value = event.target.value.trim().toLowerCase();
  const asset = [...registry.values()].find((item) => item.id.toLowerCase().includes(value) || item.type.toLowerCase().includes(value));
  if (asset) showAsset(asset.id);
});
document.querySelector("#evidence").addEventListener("change", (event) => {
  const level = event.target.value;
  if (!modelRoot) return;
  modelRoot.traverse((object) => {
    if (!object.isMesh) return;
    const asset = registry.get(assetIdFor(object));
    object.visible = !level || !asset || asset.evidence_level === level;
  });
});

function resize() {
  const width = canvas.clientWidth;
  const height = canvas.clientHeight;
  renderer.setSize(width, height, false);
  camera.aspect = width / Math.max(height, 1);
  camera.updateProjectionMatrix();
}
function animate() { resize(); controls.update(); renderer.render(scene, camera); requestAnimationFrame(animate); }
load();
animate();
