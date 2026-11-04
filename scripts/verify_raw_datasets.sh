#!/usr/bin/env bash
#
# Re-hash every raw dataset file and compare against the recorded manifest.
#
# download_raw_datasets.sh only hashes files it has just fetched, so this is
# the script that detects silent corruption or an upstream file changing under
# a path we already have. It reads tens of gigabytes, so run it through Slurm
# rather than on a login node:
#
#   sbatch --partition=kutem --account=kutem --cpus-per-task=1 --mem=4G \
#          --time=02:00:00 --job-name=domaino-verify \
#          --wrap "scripts/verify_raw_datasets.sh"

set -euo pipefail

REPO_ROOT="$(git rev-parse --show-toplevel 2>/dev/null || pwd)"
MANIFEST="${REPO_ROOT}/data/manifests/raw-datasets.sha256"

if [[ ! -f "${MANIFEST}" ]]; then
    echo "ERROR: no manifest at ${MANIFEST}; run scripts/download_raw_datasets.sh first."
    exit 1
fi

cd "${REPO_ROOT}"

echo "Verifying $(wc -l < "${MANIFEST}") recorded files against ${MANIFEST}"
echo

status=0
sha256sum --check --quiet "${MANIFEST}" || status=$?

# Files present on disk but absent from the manifest are reported too: an
# unrecorded file is not reproducible even if it is not corrupt.
echo
echo "Checking for files missing from the manifest..."
while IFS= read -r -d '' path; do
    if ! grep -qF "  ${path}" "${MANIFEST}"; then
        echo "  UNRECORDED: ${path}"
        status=1
    fi
done < <(find data/raw -type f ! -name "*.part" -print0 | sort -z)

echo
if [[ ${status} -eq 0 ]]; then
    echo "All raw datasets match the manifest."
else
    echo "Verification FAILED; see the lines above."
fi
exit ${status}
