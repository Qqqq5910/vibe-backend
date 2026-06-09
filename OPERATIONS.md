# VibeLocator Backend Operations

## Prevent Data Loss On Deploys

Zeabur must have a persistent volume mounted before SQLite is used for user data.
This service is deployed from GitHub, so pushing `main` updates the code and triggers a redeploy, but it does not safely migrate existing container files by itself. Mount the volume and back up the current database before pushing any deploy-triggering commit.

Required service settings:

```text
Volume mount directory: /data
DB_PATH=/data/memories.db
UPLOAD_DIR=/data/uploads
ADMIN_TOKEN=<a long random secret>
REQUIRE_PERSISTENT_STORAGE=1
```

The app defaults to `/data/memories.db` and `/data/uploads`. On startup it will migrate from the legacy `memories.db` file only when the target database is empty. `REQUIRE_PERSISTENT_STORAGE=1` prevents accidental fallback to a non-`/data` path.

## Check Current User Counts

```bash
curl -sS \
  -H "Authorization: Bearer $ADMIN_TOKEN" \
  "https://vibe-backend.zeabur.app/api/admin/stats?include_device_ids=true&limit=1000"
```

## Search For Existing SQLite Databases In Zeabur Terminal

Run this inside the Zeabur service terminal:

```bash
python - <<'PY'
import json
import os
import sqlite3

roots = ["/app", "/data", "/tmp", "/root", "/var"]
seen = set()
results = []

for root in roots:
    if not os.path.exists(root):
        continue
    for current_root, dirs, files in os.walk(root):
        dirs[:] = [d for d in dirs if d not in {".git", "__pycache__", "node_modules"}]
        for name in files:
            lower = name.lower()
            if not (lower.endswith(".db") or lower.endswith(".sqlite") or lower.endswith(".sqlite3")):
                continue
            path = os.path.join(current_root, name)
            if path in seen:
                continue
            seen.add(path)
            item = {"path": path, "size": os.path.getsize(path)}
            try:
                conn = sqlite3.connect(path)
                cursor = conn.cursor()
                cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='memory_pool'")
                if cursor.fetchone():
                    item["total_places"] = cursor.execute("SELECT COUNT(*) FROM memory_pool").fetchone()[0]
                    item["users_with_data"] = cursor.execute(
                        "SELECT COUNT(DISTINCT device_id) FROM memory_pool WHERE device_id IS NOT NULL AND TRIM(device_id) != ''"
                    ).fetchone()[0]
                conn.close()
            except Exception as exc:
                item["error"] = str(exc)
            results.append(item)

print(json.dumps(results, ensure_ascii=False, indent=2))
PY
```

## Backup Current Database Before Any Redeploy

```bash
python - <<'PY'
import os
import shutil
from datetime import datetime

db = os.getenv("DB_PATH", "memories.db")
target = f"/tmp/memories-backup-{datetime.now().strftime('%Y%m%d-%H%M%S')}.db"
shutil.copy2(db, target)
print(target)
PY
```

Download that backup before changing Zeabur storage or triggering a deploy.
