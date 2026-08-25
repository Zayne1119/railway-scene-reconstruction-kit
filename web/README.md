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

