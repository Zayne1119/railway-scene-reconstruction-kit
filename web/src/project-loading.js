export function selectProject(query) {
  if (query.has("project")) {
    const filename = query.get("project");
    if (!/^[A-Za-z0-9_-][A-Za-z0-9._-]*\.json$/.test(filename || "")) {
      throw new Error("project 只接受普通 JSON 文件名，不接受路径或外部网址；未切换到其他项目。");
    }
    return { url: `/${filename}`, demo: false, explicit: true };
  }
  if (query.get("demo") === "1") return { url: "/demo/project.json", demo: true, explicit: true };
  return { url: "/project.json", demo: false, explicit: false };
}

export class ResourceError extends Error {
  constructor(url, message, status = null) {
    super(`读取 ${url} 失败：${message}`);
    this.name = "ResourceError";
    this.url = url;
    this.status = status;
  }
}

export async function fetchJson(url, optional = false, request = fetch) {
  if (!url) {
    if (optional) return null;
    throw new ResourceError("未配置资源", "缺少必需的 JSON 地址");
  }
  let response;
  try {
    response = await request(url, { cache: "no-store", headers: { Accept: "application/json" } });
  } catch (error) {
    throw new ResourceError(url, error.message);
  }
  if (!response.ok) {
    if (optional && response.status === 404) return null;
    throw new ResourceError(url, `${response.status} ${response.statusText}`, response.status);
  }
  try {
    return await response.json();
  } catch {
    throw new ResourceError(url, "返回内容不是有效 JSON；请检查资源地址和服务器响应", response.status);
  }
}

export function missingProject(error, selection) {
  return error instanceof ResourceError && error.status === 404 && error.url === selection?.url &&
    (!selection.explicit || selection.demo);
}

export function validateProject(config, selection) {
  if (!config || typeof config !== "object" || Array.isArray(config)) throw new Error("项目配置必须是 JSON 对象。");
  if (!config.model_glb_url && !(config.model_url && config.material_url)) {
    throw new Error("项目配置缺少 model_glb_url，或完整的 model_url / material_url。");
  }
  if (!config.mesh_audit_url) throw new Error("项目配置缺少 mesh_audit_url；不能把未检查的网格显示为通过。");
  if (selection.demo && config.synthetic_demo !== true) {
    throw new Error("演示配置必须明确标记 synthetic_demo: true；未加载未标记的模型。");
  }
  if (config.registry_sources !== undefined && !Array.isArray(config.registry_sources)) {
    throw new Error("registry_sources 必须是数组。");
  }
  return config;
}
