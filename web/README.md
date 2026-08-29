# Config-driven Web viewer

The viewer is intentionally model-agnostic. It loads `public/project.json`, one GLB and one asset registry. Production files stay outside this source repository and are copied into an approved deployment package only after review.

```powershell
cd web
npm install
Copy-Item public/project.example.json public/project.json
npm run dev
```

Open `http://localhost:3010`. For LAN review, the dev command already listens on all interfaces; allow the port through the firewall only on a trusted private network. Do not publish a site or model without explicit project-owner authorization.

GLB node names should match registry asset IDs, or each node should contain `userData.asset_id`. The page reports missing registry mappings instead of silently presenting unregistered geometry as a valid asset.

## Fail-closed acceptance mode

Generate `public/project.json` with `railway-recon web-release-config`, or fill every release/hash field in `project.example.json`. Then set `acceptance_mode` to `true` or open the page with `?acceptance=1`.

Acceptance mode stops instead of showing a placeholder when any of these checks fail:

- config, registry or model cannot be loaded;
- release ID, model hash, registry hash or asset-set hash differs;
- the release registry contains a non-accepted asset;
- a visible Mesh node has no registry mapping.

The normal non-acceptance viewer keeps its synthetic placeholder for first-run development only. A placeholder is never an acceptance result.
