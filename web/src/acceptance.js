export function acceptanceRequested(query, config = null) {
  return query.get("acceptance") === "1" || config?.acceptance_mode === true;
}

export function validateAcceptanceConfig(config) {
  const errors = [];
  for (const field of [
    "release_id",
    "model_url",
    "model_sha256",
    "registry_url",
    "registry_sha256",
    "asset_set_sha256",
  ]) {
    if (!config?.[field] || typeof config[field] !== "string") {
      errors.push(`验收配置缺少 ${field}`);
    }
  }
  for (const field of ["model_sha256", "registry_sha256", "asset_set_sha256"]) {
    const value = config?.[field];
    if (typeof value === "string" && !/^[a-f0-9]{64}$/i.test(value)) {
      errors.push(`${field} 必须是 64 位 SHA-256`);
    }
  }
  return errors;
}

export async function sha256Hex(value) {
  const bytes = value instanceof ArrayBuffer
    ? value
    : new TextEncoder().encode(String(value));
  const digest = await globalThis.crypto.subtle.digest("SHA-256", bytes);
  return [...new Uint8Array(digest)]
    .map((item) => item.toString(16).padStart(2, "0"))
    .join("");
}

export function canonicalAssetIds(assets) {
  return assets.map((asset) => String(asset.id)).sort().join("\n");
}

export async function assetSetSha256(assets) {
  return sha256Hex(canonicalAssetIds(assets));
}
