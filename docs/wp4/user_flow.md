# WP4 / T4.1 — User workflow, wireframes and API contract

Deliverable **D4.1**. Milestone **M4.1**: a user can submit a sequence to the
mocked system and inspect a simulated result.

Three artefacts make up this task, and they are consistent with each other by
construction:

| Artefact | Where | How it stays honest |
| --- | --- | --- |
| API contract | [`docs/api/domaino-api-v1.json`](../api/domaino-api-v1.json) | The backend loads this file at runtime and validates every request **and** every response against it. A drift between document and code fails a test. |
| Wireframes | this document, §3 | Each screen names the API fields it reads, so a redesign that needs a new field shows up as a contract change. |
| Interactive mock | `web/mock/index.html`, served by `python -m domaino.service --static web/mock` | Talks to the real backend over the real API. The only difference from the finished product is that the predictor is `MockPredictor`. |

---

## 1. Who uses it, and what for

The interface serves one task: *I have a protein sequence and no structure, and
I want to know how it breaks into domains and what those domains are.*

Two kinds of user, and the design follows the first:

- **A biologist with one sequence.** Pastes it, waits, reads the domain map.
  Needs to know how much to trust each answer and must never be shown a guess
  dressed as a fact.
- **A bioinformatician with a list.** Uses the same HTTP API from a script. This
  is why the API is the contract and the interface is a client of it, rather
  than the interface being the product and the API an afterthought.

## 2. The flow

```
   ┌──────────────┐
   │ 1. Submit    │  paste a sequence, or drop a FASTA file
   └──────┬───────┘
          │ POST /api/v1/predictions
          ▼
   ┌──────────────┐  invalid?  ──────────────────────────────┐
   │ 2. Validate  │                                          │
   └──────┬───────┘                                          ▼
          │ 202 Accepted + job_id              ┌─────────────────────────┐
          ▼                                    │ 2b. Fix it              │
   ┌──────────────┐                            │ the error names the     │
   │ 3. Wait      │  GET .../{job_id}          │ character and its       │
   │              │  every 1.5 s               │ position; the box keeps │
   │ embedding →  │                            │ what was typed          │
   │ segmenting → │                            └─────────────────────────┘
   │ classifying  │
   └──────┬───────┘
          │ status: succeeded
          ▼
   ┌──────────────┐
   │ 4. Read      │  domain map, per-domain CATH, confidence
   └──────┬───────┘
          │
          ├──→ 5a. inspect a domain      (click it: residues, label, evidence)
          ├──→ 5b. export                (JSON, FASTA of each domain, PNG) — T4.6
          ├──→ 5c. similar CATH domains  — T4.4
          └──→ 5d. 3-D view              — T4.5, optional, never blocking
```

### Stages the user sees while waiting

`validating → embedding → segmenting → classifying → done`, named rather than
a percentage bar. A percentage would have to be invented; a named stage that
stalls says which component stalled. The frontend polls
`GET /api/v1/predictions/{job_id}` and reads `progress.stage`.

### Why submission is asynchronous

Embedding a 4,000-residue chain with ESM-2 650M takes longer than a browser
will hold a request open. `POST` therefore returns `202` with a `job_id`
immediately, and the result is fetched by polling. This is also what makes a
queue possible when several users arrive at once.

## 3. Wireframes

### 3.1 Submit

```
┌────────────────────────────────────────────────────────────────────┐
│  Domaino                                   [ docs ]  [ about ]     │
├────────────────────────────────────────────────────────────────────┤
│                                                                    │
│   Protein sequence                                                 │
│  ┌──────────────────────────────────────────────────────────────┐  │
│  │ >sp|P0A7B8| example                                          │  │
│  │ MKTAYIAKQRQISFVKSHFSRQLEERLGLIEVQAPILSRVGDGTQDNLSGAEKAVQVKV  │  │
│  │ KALPDAQFEVVHSLAKWKRQTLGQHDFSAGEGLYTHMKALRPDEDRLSPLHSVYVDQWD  │  │
│  │                                                              │  │
│  └──────────────────────────────────────────────────────────────┘  │
│   118 residues · amino-acid letters only · one sequence            │
│                                                                    │
│   [ Drop a FASTA file here, or browse ]                            │
│                                                                    │
│   ▸ Options                                                        │
│      Model:  (•) pairwise   ( ) set-prediction                     │
│      [x] per-residue confidence    [ ] predict structure (slower)  │
│                                                                    │
│                                     [  Predict domains  ]          │
│                                                                    │
│   Your sequence is not stored after the prediction finishes.       │
└────────────────────────────────────────────────────────────────────┘
```

Reads: nothing. Writes: `PredictionRequest` — `sequence`, `name`, `options`.

The live residue count under the box is the cheap half of validation; it runs
in the browser and catches the common mistakes before a request is made. The
authoritative check is still the backend's, because a second implementation of
the rules is a second set of rules.

### 3.2 Waiting

```
┌────────────────────────────────────────────────────────────────────┐
│  example  ·  118 residues                            [ cancel ]   │
├────────────────────────────────────────────────────────────────────┤
│                                                                    │
│   ●───────●───────○───────○                                        │
│   valid  embed  segment classify                                   │
│                                                                    │
│   Embedding 118 residues with esm2_t33_650M_UR50D…                 │
│                                                                    │
│   ⓘ A FASTA header was found and used as the name.                 │
│                                                                    │
└────────────────────────────────────────────────────────────────────┘
```

Reads: `status`, `progress.stage`, `progress.message`.
`cancel` issues `DELETE /api/v1/predictions/{job_id}`.

### 3.3 Result

```
┌────────────────────────────────────────────────────────────────────┐
│  example  ·  118 residues  ·  2 domains      [ export ▾ ] [ new ] │
├────────────────────────────────────────────────────────────────────┤
│  ⚠ SIMULATED RESULT — no trained model is loaded                   │  ← only when model.mode == "mock"
├────────────────────────────────────────────────────────────────────┤
│                                                                    │
│  1       20        40        60        80       100       118      │
│  ├────────┼─────────┼─────────┼─────────┼─────────┼─────────┤      │
│  ▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓░░░░░▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒      │
│  └──── domain 1 ────────┘     └──────── domain 2 ───────────┘      │
│                                                                    │
│  in domain  ▁▂▅███████▇▅▃▁▁▁▁▂▅███████████████▇▅▃▁                 │  ← tracks.in_domain
│  boundary   ▁▁▁▁▁▁▂▅█▃▁▁▁▁▂▅█▃▁▁▁▁▁▁▁▁▁▁▁▁▁▁▅█▃▁                   │  ← tracks.boundary
│                                                                    │
├────────────────────────────────────────────────────────────────────┤
│  ▓ Domain 1   residues 1–45        45 aa      confidence 0.91      │
│     CATH  3  Alpha Beta                 0.94                       │
│           3.40  3-Layer(aba) Sandwich   0.88                       │
│           3.40.50  Rossmann fold        0.71                       │
│           H     — not assigned —              ⓘ below threshold    │
│                                                                    │
│  ▒ Domain 2   residues 46–118      73 aa      confidence 0.84      │
│     CATH  1  Mainly Alpha               0.90                       │
│           1.10  Orthogonal Bundle       0.77                       │
│           1.10.10                       0.52                       │
│           1.10.10.10                    0.61                       │
│                                                                    │
├────────────────────────────────────────────────────────────────────┤
│  Model  pairwise · run t32-pairwise-a65bdee1b5ab · val mean IoU …  │
│  esm2_t33_650M_UR50D · dataset_v1 · API v1.0                       │
└────────────────────────────────────────────────────────────────────┘
```

Reads: the whole `Prediction` object.

Four decisions in this screen are not cosmetic:

1. **One colour per domain, used everywhere.** The map, the per-domain cards,
   the export and later the 3-D view all use the same colour for the same
   domain. M4.3 is exactly this consistency.
2. **A discontinuous domain gets one colour in two places** and a visible join,
   rather than two domains sharing a name. It is the thing this project can do
   that a boundary-only tool cannot, so it has to be legible.
3. **Every residue is coloured.** Residues in no domain come from
   `unassigned`, which the API returns explicitly so that a gap can never be a
   rendering accident.
4. **An unresolved level says so, with a reason.** "— not assigned —" plus
   `reason` from `CathLevel`, never a dimmed guess. The model abstains by
   design when a superfamily is poorly supported or its probability is below
   the calibrated threshold, and hiding that would misrepresent it.

### 3.4 Error

```
┌────────────────────────────────────────────────────────────────────┐
│  ✕ That sequence cannot be used                                    │
│                                                                    │
│    The sequence contains characters that are not amino-acid         │
│    letters: '@' at residue 41                                      │
│                                                                    │
│    Residue 41 is highlighted below.                                │
│  ┌──────────────────────────────────────────────────────────────┐  │
│  │ MKTAYIAKQRQISFVKSHFSRQLEERLGLIEVQAPILSR[@]GDGTQDNLSGAEKAVQ   │  │
│  └──────────────────────────────────────────────────────────────┘  │
│                                                   [ edit ]         │
└────────────────────────────────────────────────────────────────────┘
```

Reads: `error.code`, `error.message`, `error.details.positions`.

The input is never cleared, and the message is shown verbatim — which is why
the backend's messages are written as sentences a user can act on rather than
as codes.

## 4. The contract in brief

Base path `/api/v1`. Full schema: [`docs/api/domaino-api-v1.json`](../api/domaino-api-v1.json),
also served live at `GET /api/v1/schema`.

| Method | Path | Purpose |
| --- | --- | --- |
| `GET` | `/api/v1/health` | Liveness, which model is loaded, job counts |
| `POST` | `/api/v1/predictions` | Submit a sequence → `202` + job |
| `GET` | `/api/v1/predictions/{job_id}` | Poll; same shape throughout the job's life |
| `DELETE` | `/api/v1/predictions/{job_id}` | Forget the job and its sequence |
| `GET` | `/api/v1/schema` | The contract itself |

### Conventions worth stating once

- **Residue coordinates are 1-based and inclusive**, matching the dataset, so a
  result can be compared with CATH without re-indexing.
- **`status` decides which optional field is present**: `succeeded` guarantees
  `result`, `failed` guarantees `error`.
- **`model.mode` is `trained` or `mock`.** The interface must label `mock`
  prominently. A simulated result is never presented as a prediction.
- **Warnings are not failures.** A normalised sequence, a chain past the
  encoder window, an abstained superfamily and an unavailable optional service
  are all warnings; the result still arrives.
- **Versioning**: additive fields do not bump the version. A breaking change
  becomes `/api/v2`, and `api_version` in every response says which contract
  produced it.

### Error codes

| Code | HTTP | Meaning |
| --- | --- | --- |
| `invalid_request` | 400 | The body does not match the contract |
| `invalid_sequence` | 400 | Bad characters, too short, or not a protein |
| `multiple_records` | 400 | More than one FASTA record was submitted |
| `sequence_too_long` | 413 | Over 100,000 residues |
| `payload_too_large` | 413 | The body itself is over 256 KiB |
| `not_found` | 404 | No such job, or it has expired |
| `method_not_allowed` | 405 | Wrong verb for that path |
| `model_unavailable` | 503 | No model could be loaded |
| `internal_error` | 500 | A fault on this side, including an off-contract result |

## 5. Privacy and provenance

Both are WP4 objectives, and both are decided here rather than retrofitted in
T4.6:

- A submitted sequence lives in its job and nowhere else. It is dropped the
  moment the prediction finishes, and `JobStore.as_json` cannot serialise it —
  asserted in `tests/test_service.py`.
- Results expire (one hour by default) and the store is bounded.
- `DELETE` removes a job immediately, which is what the interface's *cancel*
  and *forget this* controls call.
- Every result carries `provenance`: API version, creation time, code revision,
  and the retention promise in words.
- Every result carries `model`: architecture, mode, run id, encoder and the
  validation metric its checkpoint was selected on — enough to find the run in
  `reports/experiments/` and reproduce it.

## 6. Running the mocked system

```bash
# API plus the interactive mock interface
python -m domaino.service --static web/mock
# → http://127.0.0.1:8080/

# API only
python -m domaino.service

# with a trained model from WP3
python -m domaino.service \
    --checkpoint data/processed/dataset_v1/checkpoints/<run-id>-best.pt \
    --encoder-config configs/representation_v1.json \
    --static web/mock
```

`--checkpoint` is the only difference between the mocked system and the real
one; everything above this line is unchanged by it. If the checkpoint cannot
be loaded the service says so and falls back to the mock rather than refusing
to start, so the interface is always demonstrable.

## 7. What this task does not cover

T4.1 is the design and the mocked interface. Deliberately out of scope here,
and tracked in their own tasks: the real frontend (T4.3), similarity and
functional evidence (T4.4), structure prediction and the 3-D view (T4.5),
export, deployment and the containerised configuration (T4.6), usability
testing (T4.7).
