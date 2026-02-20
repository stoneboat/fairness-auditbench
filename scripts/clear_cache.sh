#!/bin/bash
# Clear stale cached processed data so it gets regenerated with correct column types
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

echo "Removing cached processed data..."
rm -rf "$REPO_ROOT/data/processed/acs_public_coverage"
echo "✅ Cache cleared. Re-run the notebook to regenerate with correct column types."
