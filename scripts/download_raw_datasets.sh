#!/usr/bin/env bash

set -euo pipefail

# Run relative to the Git repository root.
REPO_ROOT="$(git rev-parse --show-toplevel 2>/dev/null || pwd)"

RAW_ROOT="${REPO_ROOT}/data/raw"
MANIFEST_DIR="${REPO_ROOT}/data/manifests"

CATH_CLASS_DIR="${RAW_ROOT}/cath/v4.4.0/classification"
CATH_SEQUENCE_DIR="${RAW_ROOT}/cath/v4.4.0/sequences"
PFAM_DIR="${RAW_ROOT}/pfam/v38.2"
TED_DIR="${RAW_ROOT}/ted/v5"

mkdir -p "${CATH_CLASS_DIR}"
mkdir -p "${CATH_SEQUENCE_DIR}"
mkdir -p "${PFAM_DIR}"
mkdir -p "${TED_DIR}"
mkdir -p "${MANIFEST_DIR}"

download_file() {
    local url="$1"
    local destination="$2"
    local partial="${destination}.part"

    if [[ -f "${destination}" ]]; then
        echo "Already downloaded: ${destination}"
        return
    fi

    echo "Downloading: ${url}"

    if [[ -f "${partial}" ]]; then
        curl \
            --fail \
            --location \
            --retry 5 \
            --retry-delay 10 \
            --continue-at - \
            --output "${partial}" \
            "${url}"
    else
        curl \
            --fail \
            --location \
            --retry 5 \
            --retry-delay 10 \
            --output "${partial}" \
            "${url}"
    fi

    mv "${partial}" "${destination}"
}

###############################################################################
# CATH-Plus 4.4.0
###############################################################################

CATH_CLASS_BASE="https://download.cathdb.info/cath/releases/all-releases/v4_4_0/cath-classification-data"
CATH_SEQUENCE_BASE="https://download.cathdb.info/cath/releases/all-releases/v4_4_0/sequence-data"

CATH_CLASS_FILES=(
    "cath-domain-list-v4_4_0.txt"
    "cath-domain-boundaries-seqreschopping-v4_4_0.txt"
    "cath-chain-list-v4_4_0.txt"
    "cath-names-v4_4_0.txt"
    "cath-unclassified-list-v4_4_0.txt"
    "README-cath-domain-boundaries-file-format.txt"
    "README-cath-domain-desc-file-format.txt"
    "README-cath-list-file-format.txt"
    "README-cath-names-file-format.txt"
)

for filename in "${CATH_CLASS_FILES[@]}"; do
    download_file \
        "${CATH_CLASS_BASE}/${filename}" \
        "${CATH_CLASS_DIR}/${filename}"
done

download_file \
    "${CATH_SEQUENCE_BASE}/cath-domain-seqs-v4_4_0.fa" \
    "${CATH_SEQUENCE_DIR}/cath-domain-seqs-v4_4_0.fa"

###############################################################################
# Pfam 38.2
#
# Pfam does not currently expose an easily accessible versioned Pfam38.2
# directory, so these files come from current_release. Pfam.version.gz is
# checked below to ensure the downloaded release is still 38.2.
###############################################################################

PFAM_BASE="https://ftp.ebi.ac.uk/pub/databases/Pfam/current_release"

PFAM_FILES=(
    "Pfam.version.gz"
    "relnotes.txt"
    "md5_checksums"
    "Pfam-A.regions.tsv.gz"
    "Pfam-A.clans.tsv.gz"
    "pfamA.txt.gz"
    "Pfam-A.seed.gz"
)

for filename in "${PFAM_FILES[@]}"; do
    download_file \
        "${PFAM_BASE}/${filename}" \
        "${PFAM_DIR}/${filename}"
done

echo
echo "Checking the downloaded Pfam release..."

if ! gzip -cd "${PFAM_DIR}/Pfam.version.gz" | grep -q "38.2"; then
    echo "ERROR: The current Pfam release is no longer Pfam 38.2."
    echo "Do not continue until the dataset version has been reviewed."
    exit 1
fi

gzip -cd "${PFAM_DIR}/Pfam.version.gz"

###############################################################################
# TED v5 for AlphaFold Database v4
###############################################################################

TED_BASE="https://zenodo.org/records/13908086/files"

# Main chain-level segmentation source.
download_file \
    "${TED_BASE}/ted_214m_per_chain_segmentation.tsv.gz?download=1" \
    "${TED_DIR}/ted_214m_per_chain_segmentation.tsv.gz"

# Domain-level metadata and predicted CATH assignments.
download_file \
    "${TED_BASE}/ted_365m.domain_summary.cath.globularity.taxid.tsv.gz?download=1" \
    "${TED_DIR}/ted_365m.domain_summary.cath.globularity.taxid.tsv.gz"

###############################################################################
# Verify upstream checksums where available
###############################################################################

echo
echo "Verifying Pfam checksums..."

(
    cd "${PFAM_DIR}"
    md5sum --check --ignore-missing md5_checksums
)

echo
echo "Verifying TED checksums..."

(
    cd "${TED_DIR}"

    printf '%s  %s\n' \
        "99b6afad9d670aa51813397d58f8e7c6" \
        "ted_214m_per_chain_segmentation.tsv.gz" \
        | md5sum --check -

    printf '%s  %s\n' \
        "51468b8ec6c74388085ebe0f66f973c5" \
        "ted_365m.domain_summary.cath.globularity.taxid.tsv.gz" \
        | md5sum --check -
)

###############################################################################
# Generate project-level SHA-256 checksums
###############################################################################

echo
echo "Generating SHA-256 manifest..."

(
    cd "${REPO_ROOT}"
    find data/raw \
        -type f \
        ! -name "*.part" \
        -print0 \
        | sort -z \
        | xargs -0 sha256sum
) > "${MANIFEST_DIR}/raw-datasets.sha256"

###############################################################################
# Record source versions
###############################################################################

{
    printf "source\tversion\tupstream_url\trole\n"
    printf "CATH\t4.4.0\thttps://download.cathdb.info/cath/releases/all-releases/v4_4_0/\tprimary ground truth\n"
    printf "Pfam\t38.2\thttps://ftp.ebi.ac.uk/pub/databases/Pfam/current_release/\tauxiliary region supervision\n"
    printf "TED\tv5\thttps://doi.org/10.5281/zenodo.13908086\tlower-confidence pretraining\n"
} > "${MANIFEST_DIR}/source-versions.tsv"

echo
echo "Downloads completed successfully."
echo "Raw data: ${RAW_ROOT}"
echo "Checksums: ${MANIFEST_DIR}/raw-datasets.sha256"