import * as THREE from "three";
import { OrbitControls } from "three/addons/controls/OrbitControls.js";
import { GLTFLoader } from "three/addons/loaders/GLTFLoader.js";
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
  let config;
  try {
    const response = await fetch(configUrl, { cache: "no-store" });
    if (!response.ok) throw new Error(`${response.status} ${response.statusText}`);
    config = await response.json();
  } catch (error) {
    status.textContent = "等待项目配置";
    showMessage(`未找到 ${configUrl}。已显示合成占位场景；复制 public/project.example.json 为 public/project.json 并填写模型与注册表地址。`);
    addSyntheticPlaceholder();
    return;
  }
  title.textContent = config.title || "铁路场景重建验收";
  const registryResponse = await fetch(config.registry_url, { cache: "no-store" });
  if (registryResponse.ok) {
    const value = await registryResponse.json();
    registry = new Map((value.assets || []).map((asset) => [asset.id, asset]));
  } else {
    showMessage("资产注册表加载失败；模型可以浏览，但资产查询不能作为验收结果。 ");
  }
  status.textContent = "加载模型";
  new GLTFLoader().load(
    config.model_url,
    (gltf) => {
      modelRoot = gltf.scene;
      scene.add(modelRoot);
      fitCamera(modelRoot);
      status.textContent = `${registry.size} 项资产 · 模型已加载`;
    },
    (event) => { if (event.total) status.textContent = `加载 ${Math.round(event.loaded / event.total * 100)}%`; },
    (error) => { status.textContent = "模型加载失败"; showMessage(`无法加载模型：${error.message || error}`); addSyntheticPlaceholder(); },
  );
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

