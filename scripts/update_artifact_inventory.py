"""
Update artifact_inventory.json with fresh SHA-256 digests and file sizes.
"""
import os
import hashlib
import json
from pathlib import Path
from datetime import datetime, timezone

BASE_DIR = Path(r"D:\DL\reports\resegregated_evaluation")
MANIFEST_DIR = BASE_DIR / "manifests"
INVENTORY_FILE = MANIFEST_DIR / "artifact_inventory.json"

def sha256_file(filepath):
    h = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()

def main():
    print("Computing artifact inventory for:", BASE_DIR)
    artifacts = []
    
    # Files to track:
    # 1. Main documents
    # 2. Results CSVs and confusion matrices
    # 3. Figures (PNG, SVG)
    # 4. Splits and manifests (except inventory itself)
    # Exclude snapshot directories: audit_v1_snapshot, audit_v2_snapshot
    
    for root, dirs, files in os.walk(BASE_DIR):
        dirs[:] = [d for d in dirs if not d.startswith("audit_") and not d.startswith(".")]
        for f in files:
            full_path = Path(root) / f
            rel_path = full_path.relative_to(BASE_DIR).as_posix()
            
            # Skip inventory file itself to avoid circular hash
            if rel_path == "manifests/artifact_inventory.json":
                continue
                
            size = full_path.stat().st_size
            digest = sha256_file(full_path)
            ext = full_path.suffix.lstrip(".").upper()
            
            artifacts.append({
                "relative_path": rel_path,
                "size_bytes": size,
                "sha256": digest,
                "file_type": ext
            })
            
    artifacts.sort(key=lambda x: x["relative_path"])
    
    inventory_data = {
        "report_package": "TORGO Resegregated Evaluation Package (Reconciled Final)",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "total_artifacts": len(artifacts),
        "artifacts": artifacts
    }
    
    with open(INVENTORY_FILE, "w", encoding="utf-8") as f:
        json.dump(inventory_data, f, indent=2)
        
    print(f"Updated {INVENTORY_FILE} successfully with {len(artifacts)} verified artifacts.")

if __name__ == "__main__":
    main()
