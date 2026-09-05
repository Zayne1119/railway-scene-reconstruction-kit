import * as THREE from "three";
import { OrbitControls } from "three/addons/controls/OrbitControls.js";
import { MTLLoader } from "three/addons/loaders/MTLLoader.js";
import { OBJLoader } from "three/addons/loaders/OBJLoader.js";
import { GLTFLoader } from "three/addons/loaders/GLTFLoader.js";
import { MeshoptDecoder } from "three/addons/libs/meshopt_decoder.module.js";
import { fetchJson, missingProject, selectProject, validateProject } from "./project-loading.js";
import { perspectiveFitDistance } from "./scene-math.js";
import "./style.css";

const $ = (selector) => document.querySelector(selector);
const $$ = (selector) => [...document.querySelectorAll(selector)];
const canvas = $("#viewport");
const ui = {
  title: $("#title"),
  status: $("#status span"),
  loader: $("#loader"),
  loaderLabel: $("#loader-label"),
  loaderProgress: $("#loader-progress"),
  message: $("#message"),
  issueList: $("#issue-list"),
  issueCount: $("#issue-count"),
  details: $("#details"),
  modeNote: $("#mode-note"),
};

const state = {
  config: null,
  model: null,
  modelObjects: [],
  assets: new Map(),
  reports: {},
  issues: { wire: [], mast: [], gap: [] },
  markers: new THREE.Group(),
  mode: "evidence",
  issueTab: "wire",
  selection: null,
  selectionHelper: null,
};

let renderer;

function initializeRenderer() {
  renderer = new THREE.WebGLRenderer({ canvas, antialias: true, alpha: true, logarithmicDepthBuffer: true, powerPreference: "high-performance" });
  renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
  renderer.outputColorSpace = THREE.SRGBColorSpace;
  renderer.toneMapping = THREE.ACESFilmicToneMapping;
  renderer.toneMappingExposure = 1.08;
}

const scene = new THREE.Scene();
scene.background = new THREE.Color(0x071822);
scene.fog = new THREE.FogExp2(0x071822, 0.00012);
scene.add(new THREE.HemisphereLight(0xc8efff, 0x142229, 2.1));
const keyLight = new THREE.DirectionalLight(0xffffff, 2.7);
keyLight.position.set(1200, -800, 1600);
scene.add(keyLight);
const rimLight = new THREE.DirectionalLight(0x4fd9f4, 1.2);
rimLight.position.set(-1600, 1200, 700);
scene.add(rimLight);

const camera = new THREE.PerspectiveCamera(42, 1, 0.1, 100000);
camera.up.set(0, 0, 1);
const controls = new OrbitControls(camera, canvas);
controls.enableDamping = true;
controls.dampingFactor = 0.07;
controls.screenSpacePanning = false;
controls.maxDistance = 20000;

const raycaster = new THREE.Raycaster();
const pointer = new THREE.Vector2();
const clock = new THREE.Clock();
const namespacePattern = /^(TRACK|CATENARY|CONDUCTOR|STATION(?:_[A-Z0-9]+)*)--/;
const candidateRootPattern = /^(TRACKGRAPH--TRACK-|TRACK--|TRACK-\d|CATENARY--|CONDUCTOR--|STATION_|SEG\d+-|S\d+-|CORE-|SUPPLEMENTAL-|ADJACENT-)/;

function layerFromName(name = "") {
  if (/(CONDUCTOR|CONTACT-WIRE|MESSENGER-WIRE|WIRE-BRIDGE)/.test(name)) return "CONDUCTOR";
  if (/(CATENARY-MAST|CATENARY--|MAST-)/.test(name)) return "CATENARY";
  if (/^(TRACKGRAPH--TRACK-|TRACK--|TRACK-\d)/.test(name)) return "TRACK";
  return "STATION";
}

function setLoading(label, progress) {
  ui.loaderLabel.textContent = label;
  ui.loaderProgress.style.width = `${Math.max(3, Math.min(100, progress))}%`;
}

function showMessage(text) {
  ui.message.textContent = text;
  ui.message.hidden = false;
  window.clearTimeout(showMessage.timer);
  showMessage.timer = window.setTimeout(() => { ui.message.hidden = true; }, 3600);
}

function rootObjectFor(object) {
  let current = object;
  while (current && current !== state.model) {
    const name = current.name || "";
    if (state.assets.has(name) || namespacePattern.test(name) || candidateRootPattern.test(name)) return current;
    current = current.parent;
  }
  return object;
}

function namespaceFor(object) {
  const named = rootObjectFor(object);
  return (named.name.match(namespacePattern) || [])[1] || layerFromName(named.name);
}

function layerFor(object) {
  const namespace = namespaceFor(object);
  return namespace.startsWith("STATION_") ? "STATION" : namespace;
}

function focusConfiguredView(name) {
  const value = state.config?.fixed_views?.[name];
  if (!value?.target || !value?.eye) return false;
  camera.position.fromArray(value.eye);
  controls.target.fromArray(value.target);
  camera.near = Number(value.near) || 0.05;
  camera.far = Number(value.far) || 5000;
  camera.updateProjectionMatrix();
  controls.update();
  return true;
}

function findObjects(predicate) {
  return state.modelObjects.filter(predicate);
}

function boxFor(objects) {
  const box = new THREE.Box3();
  objects.filter(Boolean).forEach((object) => box.expandByObject(object));
  return box;
}

function fitObjects(objects, direction = new THREE.Vector3(0.7, -0.55, 0.42), padding = 1.28) {
  const box = boxFor(objects);
  if (box.isEmpty()) {
    showMessage("当前视图没有可定位的几何");
    return;
  }
  const sphere = box.getBoundingSphere(new THREE.Sphere());
  const radius = Math.max(sphere.radius, 2);
  const distance = perspectiveFitDistance(radius, camera.fov, Math.max(camera.aspect, 0.01), padding);
  camera.position.copy(sphere.center).add(direction.clone().normalize().multiplyScalar(distance));
  camera.near = Math.max(0.05, distance / 10000);
  camera.far = Math.max(5000, distance + radius * 14);
  camera.updateProjectionMatrix();
  controls.target.copy(sphere.center);
  controls.maxDistance = Math.max(200, distance * 4);
  controls.update();
}

function selectObject(object, asset = null) {
  const selected = rootObjectFor(object);
  state.selection = selected;
  if (state.selectionHelper) scene.remove(state.selectionHelper);
  state.selectionHelper = new THREE.BoxHelper(selected, 0xb5ff48);
  state.selectionHelper.material.depthTest = false;
  state.selectionHelper.renderOrder = 20;
  scene.add(state.selectionHelper);
  renderDetails(selected, asset || state.assets.get(selected.name));
}

function detailRow(label, value) {
  const row = document.createElement("div");
  row.className = "detail-row";
  const key = document.createElement("b");
  const content = document.createElement("span");
  key.textContent = label;
  content.textContent = value ?? "—";
  row.append(key, content);
  return row;
}

function renderDetails(object, asset) {
  ui.details.replaceChildren();
  const title = document.createElement("h3");
  title.textContent = asset?.id || object?.name || "资产信息";
  ui.details.append(title);
  ui.details.append(
    detailRow("专业", namespaceFor(object) || object?.userData?.issueType || "质检标记"),
    detailRow("类型", asset?.type || object?.userData?.label || "场景构件"),
    detailRow("证据等级", asset?.evidence_level || object?.userData?.evidence || "未登记"),
    detailRow("状态", asset?.status || object?.userData?.status || "候选"),
  );
  if (Number.isFinite(asset?.chainage_m)) ui.details.append(detailRow("里程", `K${(asset.chainage_m / 1000).toFixed(3)}`));
  if (Number.isFinite(asset?.confidence)) ui.details.append(detailRow("置信度", asset.confidence.toFixed(2)));
  const limitation = asset?.limitations?.[0] || object?.userData?.limitation;
  if (limitation) ui.details.append(detailRow("限制", limitation));
}

function updateMode(mode) {
  state.mode = mode;
  $$(".mode-button").forEach((button) => button.classList.toggle("active", button.dataset.mode === mode));
  state.modelObjects.forEach((object) => {
    if (object.name.includes("INFERRED")) object.visible = mode === "hypothesis";
  });
  ui.modeNote.textContent = mode === "evidence"
    ? `证据版：隐藏 ${state.issues.gap.length} 段报告中的推断轨道${state.config?.synthetic_demo ? "；当前仅为合成演示" : "；具体来源以资产登记为准"}`
    : "推断补全版：橙色轨道为规则补全假设，不等同于现场观测";
  if (mode === "hypothesis") {
    const length = state.issues.gap.reduce(
      (sum, gap) => sum + gap.raw.chainage_end_m - gap.raw.chainage_start_m,
      0,
    );
    showMessage(`已显示 ${state.issues.gap.length} 段推断轨道，共 ${Math.round(length)} 米；橙色表示规则假设`);
  }
}

function updateLayers() {
  const enabled = new Map($$("[data-layer]").map((input) => [input.dataset.layer, input.checked]));
  state.modelObjects.forEach((object) => {
    const namespace = layerFor(object);
    const modeVisible = !object.name.includes("INFERRED") || state.mode === "hypothesis";
    object.visible = (enabled.get(namespace) ?? true) && modeVisible;
  });
}

function issueButton(issue, index) {
  const button = document.createElement("button");
  button.className = "issue-item";
  const code = document.createElement("b");
  const label = document.createElement("span");
  const meta = document.createElement("em");
  code.textContent = String(index + 1).padStart(2, "0");
  label.textContent = issue.label;
  meta.textContent = issue.meta;
  button.append(code, label, meta);
  button.addEventListener("click", () => focusIssue(issue));
  return button;
}

function renderIssues() {
  const issues = state.issues[state.issueTab] || [];
  ui.issueCount.textContent = String(issues.length);
  ui.issueList.replaceChildren();
  if (!issues.length) {
    const empty = document.createElement("div");
    empty.className = "issue-empty";
    const reportKey = { wire: "seams", mast: "catenary", gap: "track" }[state.issueTab];
    empty.textContent = state.reports[reportKey] ? "当前报告未列出本类待处理问题" : "未提供本类报告，不能据此判定通过";
    ui.issueList.append(empty);
    return;
  }
  issues.forEach((issue, index) => ui.issueList.append(issueButton(issue, index)));
}

function focusIssue(issue) {
  let targets = [];
  if (issue.type === "wire") {
    targets = findObjects((object) => issue.objectNames.includes(object.name));
  } else if (issue.type === "mast") {
    targets = state.markers.children.filter((marker) => marker.userData.issueId === issue.id);
  } else if (issue.type === "gap") {
    targets = findObjects((object) =>
      object.name.startsWith(`TRACK--${issue.trackId}`) &&
      object.name.includes(issue.nodeToken)
    );
    if (state.mode !== "hypothesis") updateMode("hypothesis");
  }
  if (targets.length) {
    fitObjects(targets, new THREE.Vector3(0.5, -0.75, 0.32), 2.1);
    selectObject(targets[0]);
  } else {
    showMessage("该问题保留在报告中，但没有生成正式网格");
  }
}

function buildIssues() {
  const seamReport = state.reports.seams || {};
  state.issues.wire = (seamReport.seams || [])
    .filter((seam) => seam.adjacent && !seam.passed)
    .map((seam) => ({
      type: "wire",
      label: `${seam.track_id} · ${seam.wire_type}`,
      meta: seam.endpoint_z_error_m > 0.25 ? "Z" : "XY",
      objectNames: [
        `CONDUCTOR--${seam.left_span_id}`,
        `CONDUCTOR--${seam.right_span_id}`,
      ],
      raw: seam,
    }));

  const catenary = state.reports.catenary || {};
  state.issues.mast = (catenary.candidates || [])
    .filter((candidate) => candidate.candidate_mesh_decision === "withheld_track_clearance_conflict")
    .map((candidate) => ({
      type: "mast",
      id: candidate.id || candidate.candidate_id,
      label: `${candidate.id || candidate.candidate_id} · K${(candidate.chainage_m / 1000).toFixed(3)}`,
      meta: "暂缓",
      raw: candidate,
    }));

  state.issues.gap = [];
  (state.reports.track?.tracks || []).forEach((track) => {
    (track.inferred_gap_hypotheses || []).forEach((gap, index) => {
      state.issues.gap.push({
        type: "gap",
        trackId: track.track_id,
        nodeToken: `INFERRED-${String((index + 1) * 2).padStart(3, "0")}`,
        label: `${track.track_id} · ${gap.chainage_start_m}–${gap.chainage_end_m} m`,
        meta: `${Math.round(gap.chainage_end_m - gap.chainage_start_m)}m`,
        raw: gap,
      });
    });
  });
}

function addRiskMarkers(origin) {
  state.markers.name = "RISK-MARKERS";
  const material = new THREE.MeshBasicMaterial({
    color: 0xff6c23,
    transparent: true,
    opacity: 0.92,
    depthTest: false,
  });
  state.issues.mast.forEach((issue) => {
    const candidate = issue.raw;
    const height = Math.max(2, candidate.maximum_z - candidate.minimum_z);
    const geometry = new THREE.ConeGeometry(1.15, 3.2, 12);
    const marker = new THREE.Mesh(geometry, material);
    marker.name = `RISK--${issue.id}`;
    marker.position.set(
      candidate.center_x - origin[0],
      candidate.center_y - origin[1],
      candidate.maximum_z + 2.1 - origin[2],
    );
    marker.userData = {
      issueId: issue.id,
      issueType: "CATENARY",
      label: "暂缓入模的接触网支柱候选",
      evidence: "点云候选 / 周期规则",
      status: "轨道净空冲突，未进入正式网格",
      limitation: `候选高度 ${height.toFixed(2)} m；需人工或补充影像复核`,
    };
    marker.renderOrder = 18;
    state.markers.add(marker);
  });
  scene.add(state.markers);
}

async function loadRegistries(sources = []) {
  const results = await Promise.all(sources.map(async (source) => ({
    source,
    registry: await fetchJson(source.url, true),
  })));
  results.forEach(({ source, registry }) => {
    (registry?.assets || []).forEach((asset) => {
      state.assets.set(`${source.namespace}--${asset.id}`, asset);
      state.assets.set(asset.id, asset);
      const node = asset.geometry?.node;
      if (node) state.assets.set(node, asset);
    });
  });
}

function prepareModel(model) {
  state.model = model;
  model.name = state.config.synthetic_demo ? "SYNTHETIC-DEMO" : "LOCAL-PROJECT";
  model.traverse((object) => {
    if (object.isMesh) {
      object.castShadow = false;
      object.receiveShadow = false;
      object.frustumCulled = true;
      const materials = Array.isArray(object.material) ? object.material : [object.material];
      materials.filter(Boolean).forEach((material) => {
        material.side = THREE.FrontSide;
        material.depthWrite = true;
        material.depthTest = true;
      });
    }
    if (object.isMesh) state.modelObjects.push(object);
  });
  scene.add(model);
}

function updateMetrics() {
  const seams = state.reports.seams || {};
  const catenary = state.reports.catenary || {};
  const track = state.reports.track || {};
  const gaps = state.issues.gap;
  const gapLength = gaps.reduce((sum, gap) => sum + gap.raw.chainage_end_m - gap.raw.chainage_start_m, 0);
  $("#metric-masts").textContent = `${catenary.accepted_support_count ?? "—"} / ${catenary.periodic_support_count ?? "—"}`;
  $("#metric-seams").textContent = `${seams.passing_seam_count ?? "—"} / ${seams.adjacent_seam_count ?? "—"}`;
  $("#metric-gaps").innerHTML = `${Math.round(gapLength)} <em>m</em>`;
  $("#metric-mesh").textContent = state.reports.mesh?.passed ? "PASS" : "REVIEW";
  $("#metric-mesh").style.color = state.reports.mesh?.passed ? "#b5ff48" : "#ff7b29";
  if (!track.include_inferred_gap_hypotheses) $("#metric-gaps").textContent = state.reports.track ? "OFF" : "—";
  (state.config.metric_overrides || []).forEach((metric) => {
    const target = document.getElementById(metric.id);
    const container = target?.closest("div");
    if (!target || !container) return;
    if (metric.label) container.querySelector("small").textContent = metric.label;
    target.textContent = metric.value;
    if (metric.note) container.querySelector("span").textContent = metric.note;
  });
}

function configureEvents() {
  $$(".mode-button").forEach((button) => button.addEventListener("click", () => updateMode(button.dataset.mode)));
  $$("[data-layer]").forEach((input) => input.addEventListener("change", updateLayers));
  $("#risk-layer").addEventListener("change", (event) => { state.markers.visible = event.target.checked; });
  $$(".issue-tabs button").forEach((button) => button.addEventListener("click", () => {
    state.issueTab = button.dataset.issueTab;
    $$(".issue-tabs button").forEach((item) => item.classList.toggle("active", item === button));
    renderIssues();
  }));
  $$("[data-view]").forEach((button) => button.addEventListener("click", () => {
    const view = button.dataset.view;
    if (view === "overview") fitObjects([state.model]);
    if (view === "station") fitObjects(findObjects((object) => layerFor(object) === "STATION"), new THREE.Vector3(-0.35, 0.85, 0.28), 1.75);
    if (view === "track") fitObjects(findObjects((object) => namespaceFor(object) === "TRACK"), new THREE.Vector3(0.48, -0.78, 0.27), 1.7);
    if (view === "risk") fitObjects(state.markers.children, new THREE.Vector3(0.4, -0.7, 0.45), 1.5);
    if (view === "boundary" && !focusConfiguredView("track_boundary")) showMessage("当前项目没有轨道交界固定视角");
  }));
  $("#reset").addEventListener("click", () => {
    updateMode("evidence");
    $$("[data-layer]").forEach((input) => { input.checked = true; });
    updateLayers();
    fitObjects([state.model]);
  });
  $("#search-form").addEventListener("submit", (event) => {
    event.preventDefault();
    const query = $("#search").value.trim().toLowerCase();
    if (!query) return;
    const target = state.modelObjects.find((object) => {
      const asset = state.assets.get(object.name);
      return object.name.toLowerCase().includes(query) ||
        asset?.id?.toLowerCase().includes(query) ||
        asset?.type?.toLowerCase().includes(query);
    });
    if (!target) return showMessage("没有找到匹配的资产");
    if (target.name.includes("INFERRED") && state.mode !== "hypothesis") updateMode("hypothesis");
    fitObjects([target], new THREE.Vector3(0.6, -0.6, 0.38), 2.4);
    selectObject(target);
  });
  canvas.addEventListener("pointerup", (event) => {
    const rect = canvas.getBoundingClientRect();
    pointer.x = ((event.clientX - rect.left) / rect.width) * 2 - 1;
    pointer.y = -((event.clientY - rect.top) / rect.height) * 2 + 1;
    raycaster.setFromCamera(pointer, camera);
    const meshes = [];
    state.model?.traverse((object) => { if (object.isMesh && object.visible) meshes.push(object); });
    const hit = raycaster.intersectObjects([...meshes, ...state.markers.children], false)[0];
    if (hit) selectObject(hit.object);
  });
}

function resize() {
  const width = canvas.clientWidth;
  const height = canvas.clientHeight;
  renderer?.setSize(width, height, false);
  camera.aspect = width / Math.max(height, 1);
  camera.updateProjectionMatrix();
}

function animate() {
  requestAnimationFrame(animate);
  controls.update(clock.getDelta());
  state.markers.children.forEach((marker, index) => {
    marker.rotation.z += 0.006;
    marker.scale.setScalar(1 + Math.sin(performance.now() * 0.002 + index) * 0.08);
  });
  if (state.selectionHelper) state.selectionHelper.update();
  renderer?.render(scene, camera);
}

async function bootstrap() {
  let selection;
  try {
    configureEvents();
    selection = selectProject(new URLSearchParams(window.location.search));
    state.config = validateProject(await fetchJson(selection.url), selection);
    $("#synthetic-banner").hidden = !state.config.synthetic_demo;
    initializeRenderer();
    resize();
    ui.title.textContent = state.config.title || ui.title.textContent;
    let boundaryButton = document.querySelector('[data-view="boundary"]');
    if (!boundaryButton && state.config.fixed_views?.track_boundary) {
      boundaryButton = document.createElement("button");
      boundaryButton.dataset.view = "boundary";
      boundaryButton.textContent = "轨道交界";
      boundaryButton.addEventListener("click", () => focusConfiguredView("track_boundary"));
      document.querySelector(".view-grid")?.append(boundaryButton);
    }
    if (boundaryButton) boundaryButton.hidden = !state.config.fixed_views?.track_boundary;
    setLoading("正在读取质检报告与资产注册表", 10);

    const [origin, seams, catenary, track, mesh] = await Promise.all([
      fetchJson(state.config.origin_url, !state.config.origin_url),
      fetchJson(state.config.conductor_seam_url, true),
      fetchJson(state.config.catenary_selection_url, true),
      fetchJson(state.config.track_report_url, true),
      fetchJson(state.config.mesh_audit_url),
      loadRegistries(state.config.registry_sources),
    ]);
    state.reports = { origin, seams, catenary, track, mesh };
    buildIssues();
    addRiskMarkers(origin?.origin_xyz || [0, 0, 0]);
    updateMetrics();
    renderIssues();

    let model;
    if (state.config.model_glb_url) {
      setLoading("正在载入二进制场景", 18);
      const gltfLoader = new GLTFLoader();
      gltfLoader.setMeshoptDecoder(MeshoptDecoder);
      const gltf = await gltfLoader.loadAsync(state.config.model_glb_url, (event) => {
        if (!event.total) return;
        const ratio = event.loaded / event.total;
        const label = ratio > 0.98
          ? "下载完成，正在创建 GPU 场景"
          : "正在载入二进制场景";
        setLoading(label, 18 + ratio * 74);
      });
      model = gltf.scene;
    } else {
      setLoading("正在载入材质", 18);
      const materials = await new MTLLoader().loadAsync(state.config.material_url);
      materials.preload();
      const objLoader = new OBJLoader();
      objLoader.setMaterials(materials);
      setLoading("正在解析 OBJ 完整模型", 22);
      model = await objLoader.loadAsync(state.config.model_url, (event) => {
        if (event.total) setLoading("正在解析 OBJ 完整模型", 22 + (event.loaded / event.total) * 70);
      });
    }
    prepareModel(model);
    if (!state.modelObjects.length) throw new Error("模型文件已读取，但场景没有可显示的网格对象。请检查导出结果。");
    updateMode("evidence");
    updateLayers();
    fitObjects([model]);
    ui.loaderProgress.style.width = "100%";
    const triangleCount = mesh.triangle_count_after_fan_triangulation || mesh.face_count || 0;
    ui.status.textContent = `模型已加载 · ${state.modelObjects.length} 网格对象 · ${triangleCount.toLocaleString()} 报告面数 · QA ${mesh.passed ? "PASS" : "REVIEW"}`;
    ui.status.dataset.state = "ready";
    ui.status.dataset.modelLoaded = "true";
    ui.status.dataset.objectCount = String(state.modelObjects.length);
    $("#metric-objects").textContent = String(state.modelObjects.length);
    window.setTimeout(() => { ui.loader.hidden = true; }, 260);
  } catch (error) {
    ui.loader.hidden = true;
    if (missingProject(error, selection)) {
      ui.status.textContent = "等待生成演示或配置项目";
      ui.status.dataset.state = "onboarding";
      $("#onboarding-reason").textContent = selection.demo
        ? "合成演示尚未生成。请在工具包根目录运行下列命令，然后打开演示。"
        : "未找到 project.json。可以先打开已生成的演示，或运行下列命令生成；不会自动读取其他项目。";
      $("#onboarding").hidden = false;
    } else {
      console.error(error);
      ui.status.textContent = "加载失败 · 可重试";
      ui.status.dataset.state = "error";
      $("#load-error-detail").textContent = error.message || String(error);
      $("#load-error").hidden = false;
    }
  }
}

$$("[data-retry]").forEach((button) => button.addEventListener("click", () => window.location.reload()));
window.addEventListener("resize", resize);
bootstrap();
animate();
