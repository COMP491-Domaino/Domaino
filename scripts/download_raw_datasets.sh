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
PDB_DIR="${RAW_ROOT}/pdb/seqres"
CHAINSAW_DIR="${RAW_ROOT}/benchmarks/chainsaw"
CATHE2_DIR="${RAW_ROOT}/benchmarks/cathe2"

mkdir -p "${CATH_CLASS_DIR}"
mkdir -p "${CATH_SEQUENCE_DIR}"
mkdir -p "${PFAM_DIR}"
mkdir -p "${TED_DIR}"
mkdir -p "${PDB_DIR}"
mkdir -p "${CHAINSAW_DIR}"
mkdir -p "${CATHE2_DIR}"
mkdir -p "${MANIFEST_DIR}"

# Files fetched during this run. Only these are checksum-verified, so that
# re-running the script does not re-hash tens of gigabytes of existing data.
# Use scripts/verify_raw_datasets.sh (under Slurm) for a full re-verification.
NEW_FILES=()

# How many times to re-attempt a file after a dropped connection. The CATHe2
# archive is 36.8 GB and Zenodo closes the connection partway through, which
# curl reports as error 18 and does not retry; curl 7.61 here is also too old
# for --retry-all-errors. So each attempt resumes from the bytes already on
# disk, and the loop keeps going until curl reports success.
DOWNLOAD_MAX_ATTEMPTS="${DOWNLOAD_MAX_ATTEMPTS:-200}"
DOWNLOAD_RETRY_DELAY="${DOWNLOAD_RETRY_DELAY:-15}"

download_file() {
    local url="$1"
    local destination="$2"
    local partial="${destination}.part"

    if [[ -f "${destination}" ]]; then
        echo "Already downloaded: ${destination}"
        return
    fi

    echo "Downloading: ${url}"

    # Resuming needs a file to resume from, and --continue-at - on an empty
    # file simply starts at byte 0.
    touch "${partial}"

    local attempt=1
    until curl \
        --fail \
        --location \
        --retry 10 \
        --retry-delay "${DOWNLOAD_RETRY_DELAY}" \
        --remote-time \
        --continue-at - \
        --output "${partial}" \
        "${url}"
    do
        local status=$?
        # curl exits 33 when the server will not accept a Range request and 416
        # maps to 33 as well; either way there is nothing more to resume.
        if [[ ${status} -eq 33 ]]; then
            echo "Server refused to resume ${destination}; restarting from zero."
            : > "${partial}"
        fi

        if [[ ${attempt} -ge ${DOWNLOAD_MAX_ATTEMPTS} ]]; then
            echo "ERROR: gave up on ${url} after ${attempt} attempts."
            return 1
        fi

        echo "Attempt ${attempt} ended (curl exit ${status}) at $(stat -c%s "${partial}") bytes; resuming in ${DOWNLOAD_RETRY_DELAY}s."
        attempt=$((attempt + 1))
        sleep "${DOWNLOAD_RETRY_DELAY}"
    done

    mv "${partial}" "${destination}"
    NEW_FILES+=("${destination}")
}

# Verify one file against an expected checksum, but only if it was just
# downloaded. Already-present files are left to verify_raw_datasets.sh.
verify_if_new() {
    local algorithm="$1"
    local expected="$2"
    local path="$3"

    local file
    for file in ${NEW_FILES[@]+"${NEW_FILES[@]}"}; do
        if [[ "${file}" == "${path}" ]]; then
            echo "Verifying ${algorithm}: ${path}"
            printf '%s  %s\n' "${expected}" "${path}" | "${algorithm}sum" --check -
            return
        fi
    done

    echo "Skipping ${algorithm} check (already present): ${path}"
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
# PDB SEQRES sequences
#
# CATH ships per-domain sequences only, but Domaino is trained on whole chains:
# cath-domain-boundaries-seqreschopping indexes into the SEQRES record of each
# PDB chain, so the chain sequences themselves come from here. RCSB rebuilds
# this file weekly and publishes no version string or checksum, so the retrieval
# date and our own SHA-256 in the manifest are what pin it.
###############################################################################

download_file \
    "https://files.rcsb.org/pub/pdb/derived_data/pdb_seqres.txt.gz" \
    "${PDB_DIR}/pdb_seqres.txt.gz"

###############################################################################
# Benchmark hold-out definitions
#
# Work package 1 must keep these test examples, and their close homologues, out
# of the training partitions. The two benchmarks are independent: Chainsaw is
# the segmentation benchmark and CATHe2 the classification benchmark.
###############################################################################

# Chainsaw CATH1363. Pinned to the last commit that touched the benchmark
# directory so the hold-out never shifts under us.
CHAINSAW_COMMIT="c8837653d6cc52b41fa0dbed6086921e4b740a48"
CHAINSAW_BASE="https://raw.githubusercontent.com/JudeWells/Chainsaw/${CHAINSAW_COMMIT}/data_and_benchmarking"

for filename in \
    "chainsaw_cath1363_train_test_splits.json" \
    "chainsaw_model_v3_on_cath1363_test.csv" \
    "README.md"
do
    download_file \
        "${CHAINSAW_BASE}/${filename}" \
        "${CHAINSAW_DIR}/${filename}"
done

# CATHe2. Only the full Zenodo archive carries the test-domain identifiers that
# the exclusion needs; the GitHub repository publishes labels without them.
download_file \
    "https://zenodo.org/records/14970431/files/data.zip?download=1" \
    "${CATHE2_DIR}/data.zip"

# Superfamily labels of the CATHe2 test set, used by the audit to report how
# much of their label space our CATH label space covers.
CATHE2_COMMIT="b046d476ca55791e75f9d23d6d9fdad241709832"
download_file \
    "https://raw.githubusercontent.com/Mouret-Orfeu/CATHe2/${CATHE2_COMMIT}/src/cathe-predict/Y_Test_SF.csv" \
    "${CATHE2_DIR}/Y_Test_SF.csv"

###############################################################################
# Verify upstream checksums for newly downloaded files
###############################################################################

if [[ ${#NEW_FILES[@]} -eq 0 ]]; then
    echo
    echo "No new files downloaded; skipping checksum verification."
else
    echo
    echo "Verifying Pfam checksums..."

    (
        cd "${PFAM_DIR}"
        md5sum --check --ignore-missing md5_checksums
    )

    echo
    echo "Verifying TED and CATHe2 checksums..."

    verify_if_new md5 \
        "99b6afad9d670aa51813397d58f8e7c6" \
        "${TED_DIR}/ted_214m_per_chain_segmentation.tsv.gz"

    verify_if_new md5 \
        "51468b8ec6c74388085ebe0f66f973c5" \
        "${TED_DIR}/ted_365m.domain_summary.cath.globularity.taxid.tsv.gz"

    verify_if_new md5 \
        "5dfcb873b07dcc89b32ec3ed18d42417" \
        "${CATHE2_DIR}/data.zip"
fi

###############################################################################
# Generate project-level SHA-256 checksums
###############################################################################

# Hashing the whole tree costs tens of gigabytes of reads, so only files that
# are missing from the manifest are hashed here. Run
# scripts/verify_raw_datasets.sh to re-hash everything and detect drift.
MANIFEST="${MANIFEST_DIR}/raw-datasets.sha256"

echo
echo "Updating SHA-256 manifest..."

(
    cd "${REPO_ROOT}"

    touch "${MANIFEST}"

    while IFS= read -r -d '' path; do
        if ! grep -qF "  ${path}" "${MANIFEST}"; then
            echo "  hashing ${path}" >&2
            sha256sum "${path}"
        fi
    done < <(find data/raw -type f ! -name "*.part" -print0 | sort -z) \
        >> "${MANIFEST}"

    sort -k2 -o "${MANIFEST}" "${MANIFEST}"
)

###############################################################################
# Record source versions
###############################################################################

{
    printf "source\tversion\tupstream_url\trole\n"
    printf "CATH\t4.4.0\thttps://download.cathdb.info/cath/releases/all-releases/v4_4_0/\tprimary ground truth\n"
    printf "Pfam\t38.2\thttps://ftp.ebi.ac.uk/pub/databases/Pfam/current_release/\tauxiliary region supervision\n"
    printf "TED\tv5\thttps://doi.org/10.5281/zenodo.13908086\tlower-confidence pretraining\n"
    printf "PDB\tseqres-%s\thttps://files.rcsb.org/pub/pdb/derived_data/pdb_seqres.txt.gz\tchain sequences for CATH choppings\n" \
        "$(date -u -r "${PDB_DIR}/pdb_seqres.txt.gz" +%Y-%m-%d)"
    printf "Chainsaw\tgit-%s\thttps://github.com/JudeWells/Chainsaw\tsegmentation benchmark hold-out\n" \
        "${CHAINSAW_COMMIT:0:8}"
    printf "CATHe2\tzenodo-14970431\thttps://doi.org/10.5281/zenodo.14970431\tclassification benchmark hold-out\n"
} > "${MANIFEST_DIR}/source-versions.tsv"

echo
echo "Downloads completed successfully."
echo "Raw data: ${RAW_ROOT}"
echo "Checksums: ${MANIFEST}"