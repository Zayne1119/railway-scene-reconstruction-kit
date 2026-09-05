import { realpathSync, statSync } from "node:fs";
import { homedir } from "node:os";
import path from "node:path";

function contains(parent, child) {
  const relative = path.relative(parent, child);
  return relative === "" || (!relative.startsWith(`..${path.sep}`) && relative !== ".." && !path.isAbsolute(relative));
}

export function viewerServerConfig(webDirectory, environment = process.env) {
  const webRoot = realpathSync(webDirectory);
  const allow = [webRoot];
  const configured = environment.RECON_VIEWER_DATA_ROOT;
  if (configured !== undefined && configured !== "") {
    if (configured.trim() !== configured || !path.isAbsolute(configured)) {
      throw new Error("RECON_VIEWER_DATA_ROOT 必须是一个明确的绝对目录，不接受相对路径或首尾空白。");
    }
    let approved;
    try {
      approved = realpathSync(configured);
      if (!statSync(approved).isDirectory()) throw new Error("not a directory");
    } catch {
      throw new Error("RECON_VIEWER_DATA_ROOT 必须指向存在且可读取的专用模型目录。");
    }
    if (approved === path.parse(approved).root || contains(approved, webRoot) || contains(approved, homedir())) {
      throw new Error("RECON_VIEWER_DATA_ROOT 不得放行磁盘根目录、用户主目录、网页目录或其上级工作区；请指定专用模型子目录。");
    }
    allow.push(approved);
  }
  const local = { host: "127.0.0.1", port: 3010, strictPort: true, allowedHosts: ["127.0.0.1", "localhost"], cors: false };
  return {
    root: webRoot,
    server: { ...local, fs: { strict: true, allow } },
    preview: { ...local },
  };
}
