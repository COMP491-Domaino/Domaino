"use strict";
/* Domaino web interface.

   A client of the documented API (docs/api/domaino-api-v1.json) and nothing
   more: it holds no prediction logic, so a mocked backend and a trained one
   look the same here except for the label the API itself provides. */

const API = (location.protocol === "file:" ? "http://127.0.0.1:8080" : "") + "/api/v1";
const POLL_MS = 900;
const STRUCTURE_POLL_MS = 1000;
const STRUCTURE_POLL_LIMIT = 90;
const SEQUENCE_VIEW_LIMIT = 5000;
const CATH_RELEASE = "https://www.cathdb.info/version/v4_4_0/";

const STAGES = [
  {id: "validating", name: "Check"},
  {id: "embedding", name: "Embed"},
  {id: "segmenting", name: "Segment"},
  {id: "classifying", name: "Classify"},
];
const LEVELS = [
  {key: "C", name: "Class"},
  {key: "A", name: "Architecture"},
  {key: "T", name: "Topology"},
  {key: "H", name: "Homologous superfamily"},
];

/* Real UniProt multidomain proteins with AlphaFold DB models, so the examples
   exercise the structure view; T4 lysozyme has no model and shows the
   "no structure" state. */
const EXAMPLES = [
  {name: "β-galactosidase", note: "1,024 aa", header: "sp|P00722|BGAL_ECOLI Beta-galactosidase",
   sequence: "MTMITDSLAVVLQRRDWENPGVTQLNRLAAHPPFASWRNSEEARTDRPSQQLRSLNGEWRFAWFPAPEAVPESWLECDLPEADTVVVPSNWQMHGYDAPIYTNVTYPITVNPPFVPTENPTGCYSLTFNVDESWLQEGQTRIIFDGVNSAFHLWCNGRWVGYGQDSRLPSEFDLSAFLRAGENRLAVMVLRWSDGSYLEDQDMWRMSGIFRDVSLLHKPTTQISDFHVATRFNDDFSRAVLEAEVQMCGELRDYLRVTVSLWQGETQVASGTAPFGGEIIDERGGYADRVTLRLNVENPKLWSAEIPNLYRAVVELHTADGTLIEAEACDVGFREVRIENGLLLLNGKPLLIRGVNRHEHHPLHGQVMDEQTMVQDILLMKQNNFNAVRCSHYPNHPLWYTLCDRYGLYVVDEANIETHGMVPMNRLTDDPRWLPAMSERVTRMVQRDRNHPSVIIWSLGNESGHGANHDALYRWIKSVDPSRPVQYEGGGADTTATDIICPMYARVDEDQPFPAVPKWSIKKWLSLPGETRPLILCEYAHAMGNSLGGFAKYWQAFRQYPRLQGGFVWDWVDQSLIKYDENGNPWSAYGGDFGDTPNDRQFCMNGLVFADRTPHPALTEAKHQQQFFQFRLSGQTIEVTSEYLFRHSDNELLHWMVALDGKPLASGEVPLDVAPQGKQLIELPELPQPESAGQLWLTVRVVQPNATAWSEAGHISAWQQWRLAENLSVTLPAASHAIPHLTTSEMDFCIELGNKRWQFNRQSGFLSQMWIGDKKQLLTPLRDQFTRAPLDNDIGVSEATRIDPNAWVERWKAAGHYQAEAALLQCTADTLADAVLITTAHAWQHQGKTLFISRKTYRIDGSGQMAITVDVEVASDTPHPARIGLNCQLAQVAERVNWLGLGPQENYPDRLTAACFDRWDLPLSDMYTPYVFPSENGLRCGTRELNYGPHQWRGDFQFNISRYSQQQLMETSHRHLLHAEEGTWLNIDGFHMGIGGDDSWSPSVSAEFQLSAGRYHYQLVWCQK"},
  {name: "DNA polymerase I", note: "928 aa", header: "sp|P00582|DPO1_ECOLI DNA polymerase I",
   sequence: "MVQIPQNPLILVDGSSYLYRAYHAFPPLTNSAGEPTGAMYGVLNMLRSLIMQYKPTHAAVVFDAKGKTFRDELFEHYKSHRPPMPDDLRAQIEPLHAMVKAMGLPLLAVSGVEADDVIGTLAREAEKAGRPVLISTGDKDMAQLVTPNITLINTMTNTILGPEEVVNKYGVPPELIIDFLALMGDSSDNIPGVPGVGEKTAQALLQGLGGLDTLYAEPEKIAGLSFRGAKTMAAKLEQNKEVAYLSYQLATIKTDVELELTCEQLEVQQPAAEELLGLFKKYEFKRWTADVEAGKWLQAKGAKPAAKPQETSVADEAPEVTATVISYDNYVTILDEETLKAWIAKLEKAPVFAFDTETDSLDNISANLVGLSFAIEPGVAAYIPVAHDYLDAPDQISRERALELLKPLLEDEKALKVGQNLKYDRGILANYGIELRGIAFDTMLESYILNSVAGRHDMDSLAERWLKHKTITFEEIAGKGKNQLTFNQIALEEAGRYAAEDADVTLQLHLKMWPDLQKHKGPLNVFENIEMPLVPVLSRIERNGVKIDPKVLHNHSEELTLRLAELEKKAHEIAGEEFNLSSTKQLQTILFEKQGIKPLKKTPGGAPSTSEEVLEELALDYPLPKVILEYRGLAKLKSTYTDKLPLMINPKTGRVHTSYHQAVTATGRLSSTDPNLQNIPVRNEEGRRIRQAFIAPEDYVIVSADYSQIELRIMAHLSRDKGLLTAFAEGKDIHRATAAEVFGLPLETVTSEQRRSAKAINFGLIYGMSAFGLARQLNIPRKEAQKYMDLYFERYPGVLEYMERTRAQAKEQGYVETLDGRRLYLPDIKSSNGARRAAAERAAINAPMQGTAADIIKRAMIAVDAWLQAEQPRVRMIMQVHDELVFEVHKDDVDAVAKQIHQLMENCTRLDVPLLVEVGSGENWDQAH"},
  {name: "Src kinase", note: "536 aa", header: "sp|P12931|SRC_HUMAN Proto-oncogene tyrosine-protein kinase Src",
   sequence: "MGSNKSKPKDASQRRRSLEPAENVHGAGGGAFPASQTPSKPASADGHRGPSAAFAPAAAEPKLFGGFNSSDTVTSPQRAGPLAGGVTTFVALYDYESRTETDLSFKKGERLQIVNNTEGDWWLAHSLSTGQTGYIPSNYVAPSDSIQAEEWYFGKITRRESERLLLNAENPRGTFLVRESETTKGAYCLSVSDFDNAKGLNVKHYKIRKLDSGGFYITSRTQFNSLQQLVAYYSKHADGLCHRLTTVCPTSKPQTQGLAKDAWEIPRESLRLEVKLGQGCFGEVWMGTWNGTTRVAIKTLKPGTMSPEAFLQEAQVMKKLRHEKLVQLYAVVSEEPIYIVTEYMSKGSLLDFLKGETGKYLRLPQLVDMAAQIASGMAYVERMNYVHRDLRAANILVGENLVCKVADFGLARLIEDNEYTARQGAKFPIKWTAPEAALYGRFTIKSDVWSFGILLTELTTKGRVPYPGMVNREVLDQVERGYRMPCPPECPESLHDLMCQCWRKEPEERPTFEYLQAFLEDYFTSTEPQYQPGENL"},
  {name: "Maltose-binding protein", note: "396 aa", header: "sp|P0AEX9|MALE_ECOLI Maltose/maltodextrin-binding periplasmic protein",
   sequence: "MKIKTGARILALSALTTMMFSASALAKIEEGKLVIWINGDKGYNGLAEVGKKFEKDTGIKVTVEHPDKLEEKFPQVAATGDGPDIIFWAHDRFGGYAQSGLLAEITPDKAFQDKLYPFTWDAVRYNGKLIAYPIAVEALSLIYNKDLLPNPPKTWEEIPALDKELKAKGKSALMFNLQEPYFTWPLIAADGGYAFKYENGKYDIKDVGVDNAGAKAGLTFLVDLIKNKHMNADTDYSIAEAAFNKGETAMTINGPWAWSNIDTSKVNYGVTVLPTFKGQPSKPFVGVLSAGINAASPNKELAKEFLENYLLTDEGLEAVNKDKPLGAVALKSYEEELAKDPRIAATMENAQKGEIMPNIPQMSAFWYAVRTAVINAASGRQTVDEALKDAQTRITK"},
  {name: "Hsp70", note: "641 aa", header: "sp|P0DMV8|HS71A_HUMAN Heat shock 70 kDa protein 1A",
   sequence: "MAKAAAIGIDLGTTYSCVGVFQHGKVEIIANDQGNRTTPSYVAFTDTERLIGDAAKNQVALNPQNTVFDAKRLIGRKFGDPVVQSDMKHWPFQVINDGDKPKVQVSYKGETKAFYPEEISSMVLTKMKEIAEAYLGYPVTNAVITVPAYFNDSQRQATKDAGVIAGLNVLRIINEPTAAAIAYGLDRTGKGERNVLIFDLGGGTFDVSILTIDDGIFEVKATAGDTHLGGEDFDNRLVNHFVEEFKRKHKKDISQNKRAVRRLRTACERAKRTLSSSTQASLEIDSLFEGIDFYTSITRARFEELCSDLFRSTLEPVEKALRDAKLDKAQIHDLVLVGGSTRIPKVQKLLQDFFNGRDLNKSINPDEAVAYGAAVQAAILMGDKSENVQDLLLLDVAPLSLGLETAGGVMTALIKRNSTIPTKQTQIFTTYSDNQPGVLIQVYEGERAMTKDNNLLGRFELSGIPPAPRGVPQIEVTFDIDANGILNVTATDKSTGKANKITITNDKGRLSKEEIERMVQEAEKYKAEDEVQRERVSAKNALESYAFNMKSAVEDEGLKGKISEADKKKVLDKCQEVISWLDANTLAEKDEFEHKRKELEQVCNPIISGLYQGAGGPGPGGFGAQGPKGGSGSGPTIEEVD"},
  {name: "Calmodulin", note: "149 aa", header: "sp|P0DP23|CALM1_HUMAN Calmodulin-1",
   sequence: "MADQLTEEQIAEFKEAFSLFDKDGDGTITTKELGTVMRSLGQNPTEAELQDMINEVDADGNGTIDFPEFLTMMARKMKDTDSEEEIREAFRVFDKDGNGYISAAELRHVMTNLGEKLTDEEVDEMIREADIDGDGQVNYEEFVQMMTAK"},
  {name: "T4 lysozyme", note: "no 3-D model", header: "sp|P00720|ENLYS_BPT4 Endolysin",
   sequence: "MNIFEMLRIDERLRLKIYKDTEGYYTIGIGHLLTKSPSLNAAKSELDKAIGRNCNGVITKDEAEKLFNQDVDAAVRGILRNAKLKPVYDSLDAVRRCALINMVFQMGETGVAGFTNSLRMLQQKRWDEAAVNLAKSIWYNQTPNRAKRVITTFRTGTWDAYKNL"},
];

/* One pictogram per CATH class: helices, strands, both, few, special. */
const CLASS_ICONS = {
  "1": '<path d="M6 34C6 14 18 14 16 24S26 34 24 24 34 14 32 24 42 34 42 14"/>',
  "2": '<path d="M8 13h28M30 8l6 5-6 5"/><path d="M40 24H12M18 19l-6 5 6 5"/><path d="M8 35h28M30 30l6 5-6 5"/>',
  "3": '<path d="M6 17C6 6 14 6 13 12S20 18 19 12 26 6 25 12 32 18 32 7"/><path d="M8 29h26M28 24l6 5-6 5"/><path d="M40 40H14M20 35l-6 5 6 5"/>',
  "4": '<path d="M6 32c6-18 12 12 18-4s10-12 18 6"/><circle cx="6" cy="32" r="1.5"/><circle cx="42" cy="34" r="1.5"/>',
  "6": '<path d="M24 8v32M10 16l28 16M38 16 10 32"/>',
};

const $ = (id) => document.getElementById(id);
const state = {
  job: null,
  result: null,
  structure: null,
  model: null,
  selected: null,
  mode: "domain",
  submittedText: "",
  aborted: false,
  jobTimer: null,
  structureTimer: null,
  structurePolls: 0,
  structureSources: ["alphafold_db"],
};
let view = null;

/* ---------- helpers ---------------------------------------------------- */

function escapeHtml(text) {
  return String(text).replace(/[&<>"']/g, (c) =>
    ({"&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;"}[c]));
}

function domainColour(index) {
  return `var(${window.DomainoPalette.token(index)})`;
}

function percent(value) {
  return `${Math.round(Number(value) * 100)}%`;
}

function plural(n, word) {
  return `${n.toLocaleString()} ${word}${n === 1 ? "" : "s"}`;
}

/* The residues the backend will see, for the sequence view only. */
function residuesOf(text) {
  return text.split(/\r?\n/).filter((line) => !line.startsWith(">"))
    .join("").replace(/[^A-Za-z]/g, "").toUpperCase();
}

function show(which) {
  for (const id of ["submitView", "waitingView", "failureView", "resultView"]) {
    $(id).hidden = id !== which;
  }
  window.scrollTo({top: 0, behavior: "smooth"});
}

function download(filename, text, type) {
  const url = URL.createObjectURL(new Blob([text], {type}));
  const link = Object.assign(document.createElement("a"), {href: url, download: filename});
  document.body.appendChild(link);
  link.click();
  link.remove();
  setTimeout(() => URL.revokeObjectURL(url), 1000);
}

/* ---------- input ------------------------------------------------------ */

function updateInput() {
  const text = $("sequence").value;
  const residues = residuesOf(text);
  const headers = (text.match(/^>/gm) || []).length;
  const bad = [...new Set(residues.replace(/[ACDEFGHIKLMNPQRSTVWYBJOUXZ]/g, ""))];
  $("residueCount").textContent = plural(residues.length, "residue");
  const hint = $("inputHint");
  hint.classList.remove("bad");
  if (headers > 1) {
    hint.textContent = `${headers} FASTA records: submit one sequence at a time.`;
    hint.classList.add("bad");
  } else if (bad.length) {
    hint.textContent = `Not amino-acid letters: ${bad.join(" ")}`;
    hint.classList.add("bad");
  } else if (residues.length && residues.length < 16) {
    hint.textContent = "At least 16 residues are needed.";
    hint.classList.add("bad");
  } else {
    hint.textContent = "Amino-acid letters only, one sequence at a time. You can also drop a file here.";
  }
  $("predictButton").disabled = residues.length === 0;
}

function renderExamples() {
  $("examples").innerHTML = EXAMPLES.map((example, index) =>
    `<button class="chip" type="button" data-example="${index}">${escapeHtml(example.name)}` +
    ` <small>${escapeHtml(example.note)}</small></button>`).join("");
  $("examples").addEventListener("click", (event) => {
    const button = event.target.closest("[data-example]");
    if (!button) return;
    const example = EXAMPLES[Number(button.dataset.example)];
    const lines = example.sequence.match(/.{1,60}/g).join("\n");
    $("sequence").value = `>${example.header}\n${lines}`;
    updateInput();
  });
}

function readFile(file) {
  const reader = new FileReader();
  reader.onload = () => { $("sequence").value = String(reader.result); updateInput(); };
  reader.readAsText(file);
}

/* ---------- submission and polling ------------------------------------ */

async function submit() {
  const options = {
    include_structure: $("optStructure").checked && !$("optStructure").disabled,
    include_tracks: $("optTracks").checked,
  };
  if ($("optArchitecture").value) options.architecture = $("optArchitecture").value;
  state.submittedText = $("sequence").value;
  state.aborted = false;
  renderStages("validating", "submitting…");
  show("waitingView");

  let response, body;
  try {
    response = await fetch(`${API}/predictions`, {
      method: "POST",
      headers: {"Content-Type": "application/json"},
      body: JSON.stringify({sequence: state.submittedText, options}),
    });
    body = await response.json();
  } catch (problem) {
    unreachable();
    return;
  }
  if (!response.ok) { fail(body.error); return; }
  handleJob(body);
}

function handleJob(job) {
  if (state.aborted) return;
  state.job = job;
  if (job.status === "failed") { fail(job.error); return; }
  if (job.status === "succeeded") { renderResult(job); return; }
  renderStages((job.progress || {}).stage || "validating", (job.progress || {}).message || job.status);
  clearTimeout(state.jobTimer);
  state.jobTimer = setTimeout(pollJob, POLL_MS);
}

async function pollJob() {
  if (state.aborted || !state.job) return;
  try {
    const response = await fetch(`${API}/predictions/${state.job.job_id}`);
    const body = await response.json();
    if (!response.ok) { fail(body.error); return; }
    handleJob(body);
  } catch (problem) {
    unreachable();
  }
}

async function cancel() {
  state.aborted = true;
  stopTimers();
  if (state.job) {
    try { await fetch(`${API}/predictions/${state.job.job_id}`, {method: "DELETE"}); }
    catch (problem) { /* it expires on its own */ }
  }
  state.job = null;
  show("submitView");
}

function stopTimers() {
  clearTimeout(state.jobTimer);
  clearTimeout(state.structureTimer);
}

function renderStages(stage, message) {
  const at = Math.max(0, STAGES.findIndex((s) => s.id === stage));
  $("stepper").innerHTML = STAGES.map((s, index) => {
    const status = index < at ? "done" : index === at ? "now" : "";
    const bullet = index < at ? "✓" : index + 1;
    return `<li class="step ${status}"><span class="bullet">${bullet}</span>${s.name}</li>`;
  }).join("");
  $("stageMessage").textContent = message;
}

function fail(error) {
  error = error || {};
  stopTimers();
  $("errorTitle").textContent = error.code === "internal_error"
    ? "The prediction could not be completed"
    : "That sequence cannot be used";
  $("errorMessage").textContent = error.message || "No reason was given.";
  const positions = (error.details || {}).positions || [];
  const box = $("errorContext");
  if (positions.length) {
    const raw = state.submittedText;
    const spot = positions[0];
    box.innerHTML = escapeHtml(raw.slice(Math.max(0, spot - 40), spot - 1)) +
      "<mark>" + escapeHtml(raw.slice(spot - 1, spot)) + "</mark>" +
      escapeHtml(raw.slice(spot, spot + 40));
    box.hidden = false;
  } else {
    box.hidden = true;
  }
  show("failureView");
}

function unreachable() {
  stopTimers();
  $("errorTitle").textContent = "The Domaino service is not reachable";
  $("errorMessage").innerHTML = "Start it and reload this page: " +
    "<code>PYTHONPATH=src python -m domaino.service --static web/app</code>";
  $("errorContext").hidden = true;
  setStatus("off", "API offline");
  show("failureView");
}

/* ---------- result ----------------------------------------------------- */

function renderResult(job) {
  const result = job.result;
  state.result = result;
  state.selected = null;
  state.structure = job.structure || null;
  state.model = null;
  const simulated = result.model.mode === "mock";

  $("mockBanner").hidden = !simulated;
  $("resultName").textContent = result.name || job.name || `Job ${job.job_id}`;
  $("resultMeta").innerHTML = [
    plural(result.sequence_length, "residue"),
    plural(result.domains.length, "domain"),
    `${result.model.architecture} model`,
  ].map((text) => `<span class="meta-chip">${escapeHtml(text)}</span>`).join("") +
    (simulated ? '<span class="meta-chip simulated">simulated</span>' : "");

  renderStats();
  renderMap();
  renderTracks();
  renderSequence();
  renderDomainCards();
  renderInspector();
  renderNotes();
  renderProvenance();
  $("exportPdb").hidden = true;
  show("resultView");
  startStructure(job);
}

function renderStats() {
  const r = state.result;
  const discontinuous = r.domains.filter((d) => d.discontinuous).length;
  const meanConfidence = r.domains.length
    ? r.domains.reduce((sum, d) => sum + d.confidence, 0) / r.domains.length : null;
  const covered = r.domains.reduce((sum, d) => sum + d.length, 0);
  const s = state.structure;
  let structureValue = "Not requested", structureSub = "turn it on before predicting";
  if (s && s.status === "pending") { structureValue = "Looking…"; structureSub = "AlphaFold DB"; }
  if (s && s.status === "available") {
    structureValue = s.mean_plddt != null ? `pLDDT ${Number(s.mean_plddt).toFixed(1)}` : "Available";
    structureSub = `AlphaFold DB · ${s.model_id || ""}`;
  }
  if (s && s.status === "unavailable") { structureValue = "None found"; structureSub = "domains still shown below"; }
  const tiles = [
    ["Residues", r.sequence_length.toLocaleString(), `${percent(covered / r.sequence_length)} in a domain`],
    ["Domains", String(r.domains.length), discontinuous ? `${discontinuous} discontinuous` : "all continuous"],
    ["Segmentation confidence", meanConfidence == null ? "–" : percent(meanConfidence), "mean over domains"],
    ["3-D structure", structureValue, structureSub],
  ];
  $("stats").innerHTML = tiles.map(([label, value, sub]) =>
    `<div class="stat"><div class="label">${label}</div><div class="value">${escapeHtml(value)}</div>` +
    `<div class="sub">${escapeHtml(sub)}</div></div>`).join("");
}

/* Linear map: every residue is coloured, either by its domain or as linker. */
function renderMap() {
  const r = state.result;
  const length = r.sequence_length;
  const left = (residue) => ((residue - 1) / length) * 100;
  const width = (segment) => ((segment.end - segment.start + 1) / length) * 100;

  const step = length <= 150 ? 25 : length <= 400 ? 50 : length <= 1000 ? 100 : length <= 3000 ? 250 : 1000;
  const marks = [1];
  for (let v = step; v < length - step / 3; v += step) marks.push(v);
  if (length > 1) marks.push(length);

  const segments = [], labels = [], arcs = [];
  r.domains.forEach((domain, index) => {
    const colour = domainColour(index);
    domain.segments.forEach((segment, k) => {
      segments.push(`<div class="map-seg" data-domain="${index}" style="left:${left(segment.start)}%;` +
        `width:${width(segment)}%;background:${colour}" title="Domain ${domain.index}: ${segment.start}–${segment.end}"></div>`);
      labels.push(`<button type="button" data-domain="${index}" style="left:${left(segment.start) + width(segment) / 2}%;` +
        `color:${colour}">D${domain.index}</button>`);
      const next = domain.segments[k + 1];
      if (next) {
        // Discontinuous domains: an arc joins the pieces that belong together.
        const x1 = (segment.end / length) * 1000, x2 = ((next.start - 1) / length) * 1000;
        arcs.push(`<path d="M${x1} 32 Q${(x1 + x2) / 2} 2 ${x2} 32" style="stroke:${colour}" vector-effect="non-scaling-stroke"/>`);
      }
    });
  });

  $("domainMap").innerHTML =
    `<svg class="map-arcs" viewBox="0 0 1000 34" preserveAspectRatio="none" aria-hidden="true">${arcs.join("")}</svg>` +
    `<div class="map-track">${segments.join("")}</div>` +
    `<div class="map-labels">${labels.join("")}</div>` +
    `<div class="map-ruler">${marks.map((v) => `<span style="left:${left(v)}%">${v}</span>`).join("")}</div>`;
  const linker = r.unassigned ? r.unassigned.reduce((sum, s) => sum + s.end - s.start + 1, 0) : 0;
  $("mapCaption").textContent = linker
    ? `grey: ${plural(linker, "residue")} outside any domain`
    : "every residue is in a domain";
}

function renderTracks() {
  const tracks = state.result.tracks || {};
  const rows = [["in_domain", "In a domain"], ["boundary", "Boundary"]]
    .filter(([key]) => tracks[key])
    .map(([key, title]) => `<div class="track-row"><span>${title}</span><div class="track-bars">` +
      downsample(tracks[key], 220).map((v) =>
        `<i style="height:${Math.max(4, v * 100)}%;opacity:${0.3 + v * 0.7}"></i>`).join("") +
      "</div></div>");
  $("tracks").className = rows.length ? "tracks" : "";
  $("tracks").innerHTML = rows.join("");
}

function downsample(values, width) {
  if (values.length <= width) return values;
  const out = [];
  const size = values.length / width;
  for (let i = 0; i < width; i++) {
    const slice = values.slice(Math.floor(i * size), Math.floor((i + 1) * size));
    out.push(slice.length ? Math.max(...slice) : 0);
  }
  return out;
}

/* The sequence as submitted, residues tinted by domain, 60 per line. */
function renderSequence() {
  const r = state.result;
  const sequence = residuesOf(state.submittedText);
  const details = $("sequenceDetails");
  if (sequence.length !== r.sequence_length || sequence.length > SEQUENCE_VIEW_LIMIT) {
    details.hidden = true;
    return;
  }
  details.hidden = false;
  const owner = new Array(sequence.length + 1).fill(-1);
  r.domains.forEach((domain, index) => domain.segments.forEach((s) => {
    for (let i = s.start; i <= s.end; i++) owner[i] = index;
  }));
  const lines = [];
  for (let start = 1; start <= sequence.length; start += 60) {
    const runs = [];
    for (let block = start; block < Math.min(start + 60, sequence.length + 1); block += 10) {
      let i = block;
      const blockEnd = Math.min(block + 9, sequence.length);
      while (i <= blockEnd) {
        let j = i;
        while (j + 1 <= blockEnd && owner[j + 1] === owner[i]) j++;
        const text = sequence.slice(i - 1, j);
        const style = owner[i] >= 0
          ? ` style="background:color-mix(in srgb, ${domainColour(owner[i])} 26%, transparent)"` : "";
        runs.push(`<span${style}${j === blockEnd ? ' class="gap"' : ""}>${text}</span>`);
        i = j + 1;
      }
    }
    lines.push(`<div class="seq-line"><span class="num">${start}</span><span class="res">${runs.join("")}</span></div>`);
  }
  $("sequenceView").innerHTML = lines.join("");
}

function renderDomainCards() {
  const r = state.result;
  if (!r.domains.length) {
    $("domainGrid").innerHTML = '<p class="muted">No domains were predicted for this sequence.</p>';
    return;
  }
  $("domainGrid").innerHTML = r.domains.map((domain, index) => {
    const ranges = domain.segments.map((s) => `${s.start}–${s.end}`).join(", ");
    const cath = domain.cath;
    const cathLine = cath && cath.C && cath.C.status === "assigned"
      ? `<b>${escapeHtml(cath.C.name || `Class ${cath.C.code}`)}</b> · ${escapeHtml(cath.path || cath.C.code)}`
      : "No CATH label";
    return `<button type="button" class="domain-card${state.selected === index ? " active" : ""}"` +
      ` data-domain="${index}" style="--dom:${domainColour(index)}">` +
      `<span class="title"><span>Domain ${domain.index}</span><span class="muted">${percent(domain.confidence)}</span></span>` +
      `<span class="ranges">${ranges} · ${domain.length} aa${domain.discontinuous ? " · discontinuous" : ""}</span>` +
      `<span class="cath-line">${cathLine}</span></button>`;
  }).join("");
}

function renderNotes() {
  const notes = (state.result.warnings || [])
    .filter((w) => !/simulated result/.test(w.message))
    .map((w) => `<li>${escapeHtml(w.message)}</li>`);
  $("notesCard").hidden = !notes.length;
  $("notes").innerHTML = notes.join("");
}

function renderProvenance() {
  const r = state.result;
  const selection = r.model.selection || {};
  const s = state.structure;
  const lines = [
    `Model: ${escapeHtml(r.model.architecture)}${r.model.mode === "mock" ? " (simulated, no checkpoint loaded)" : ""}` +
      (r.model.run_id ? ` · run <code>${escapeHtml(r.model.run_id)}</code>` : "") +
      (selection.value != null ? ` · ${escapeHtml(selection.metric)} ${Number(selection.value).toFixed(4)}` : ""),
    [r.model.encoder, r.model.dataset_version, `API v${r.provenance.api_version}`, r.provenance.created_at]
      .filter(Boolean).map(escapeHtml).join(" · "),
    escapeHtml(r.provenance.retention || ""),
  ];
  if (s && s.status === "available") {
    lines.push(`Structure: AlphaFold DB model ${escapeHtml(s.model_id || "")}` +
      `${s.version ? ` (v${s.version})` : ""}, CC BY 4.0. Shown for context only; Domaino does not use it.`);
  }
  $("provenance").innerHTML = lines.filter(Boolean).map((line) => `<div>${line}</div>`).join("");
}

/* ---------- selection and the CATH inspector --------------------------- */

function select(index) {
  state.selected = index === state.selected ? null : index;
  if (view) view.setSelected(state.selected);
  renderInspector();
  renderDomainCards();
  renderLegend();
}

function renderInspector() {
  const r = state.result;
  const box = $("inspector");
  if (!r.domains.length) {
    box.innerHTML = '<div class="inspector-empty"><strong>No domains</strong>' +
      "<span>The model did not find a domain in this sequence.</span></div>";
    return;
  }
  if (state.selected === null) {
    box.style.removeProperty("--dom");
    box.innerHTML = `<div class="inspector-empty">${emptyIllustration()}
      <div><strong>Pick a domain</strong><br>Click a domain on the structure, the map or the list
      to see its predicted CATH classification.</div>
      <div class="chips">${r.domains.map((d, i) =>
        `<button class="domain-chip-button" type="button" data-domain="${i}">` +
        `<span class="sw" style="background:${domainColour(i)}"></span>Domain ${d.index}</button>`).join("")}</div>
    </div>`;
    return;
  }

  const index = state.selected;
  const domain = r.domains[index];
  const simulated = r.model.mode === "mock";
  const ranges = domain.segments.map((s) => `${s.start}–${s.end}`).join(", ");
  box.style.setProperty("--dom", domainColour(index));

  box.innerHTML = `
    <div class="inspector-head">
      <span class="swatch" style="background:var(--dom)"></span>
      <div><h3>Domain ${domain.index}</h3><div class="ranges">${ranges}</div></div>
      <button class="icon-button small close" type="button" id="clearSelection" aria-label="Show all domains">
        <svg viewBox="0 0 24 24" aria-hidden="true"><path d="M6 6l12 12M18 6 6 18"/></svg>
      </button>
    </div>
    <div class="tag-row">
      <span class="tag">${domain.length} residues</span>
      <span class="tag">${domain.discontinuous ? `discontinuous · ${domain.segments.length} segments` : "continuous"}</span>
      ${simulated ? '<span class="tag simulated">simulated</span>' : ""}
    </div>
    <div class="gauge-row">${gauge(domain.confidence)}
      <div class="text"><strong>Segmentation confidence</strong>How sure the model is that these residues form one domain.</div>
    </div>
    ${cathPanel(domain.cath, simulated)}`;
  $("clearSelection").addEventListener("click", () => select(null));
}

function cathPanel(cath, simulated) {
  if (!cath) {
    return '<p class="muted">No CATH label for this domain: the classifier is not loaded.</p>';
  }
  const top = cath.C || {};
  const icon = CLASS_ICONS[top.code] || '<circle cx="24" cy="24" r="14"/>';
  const hero = top.status === "assigned"
    ? `<div class="class-hero">
        <div class="class-icon"><svg viewBox="0 0 48 48" aria-hidden="true">${icon}</svg></div>
        <div><div class="kicker">Predicted class ${escapeHtml(top.code)}</div>
          <div class="name">${escapeHtml(top.name || `Class ${top.code}`)}</div>
          <div class="conf">${percent(top.confidence)} confidence</div></div>
      </div>`
    : "";
  const rungs = LEVELS.map((level) => {
    const entry = cath[level.key] || {status: "unresolved"};
    if (entry.status !== "assigned") {
      return `<li class="rung unresolved"><span class="node">${level.key}</span><div class="body">
        <div class="top"><span class="level">${level.name}</span></div>
        <span class="label">Not assigned</span>
        ${entry.reason ? `<span class="reason">${escapeHtml(entry.reason)}</span>` : ""}
      </div></li>`;
    }
    const page = level.key === "H" ? "superfamily" : "cathnode";
    const link = simulated ? "" :
      ` <a class="ext" href="${CATH_RELEASE}${page}/${encodeURIComponent(entry.code)}" target="_blank" rel="noopener">CATH ↗</a>`;
    return `<li class="rung"><span class="node">${level.key}</span><div class="body">
      <div class="top"><span class="level">${level.name}</span><span class="code">${escapeHtml(entry.code)}</span>${link}</div>
      ${entry.name ? `<span class="label">${escapeHtml(entry.name)}</span>` : ""}
      <div class="bar-row"><div class="bar"><span style="width:${percent(entry.confidence)}"></span></div>${percent(entry.confidence)}</div>
    </div></li>`;
  }).join("");
  return `${hero}
    <div><div class="kicker muted small-text" style="font-weight:700;margin-bottom:10px">CATH HIERARCHY</div>
    <ol class="ladder">${rungs}</ol></div>
    <p class="footnote">Each level carries the classifier's calibrated probability. It is separate from the
      segmentation confidence above, so a well-placed domain can still have an uncertain label.</p>`;
}

function gauge(value) {
  const circumference = 2 * Math.PI * 22;
  const filled = Math.max(0, Math.min(1, value)) * circumference;
  return `<svg class="gauge" viewBox="0 0 54 54" aria-hidden="true">
    <circle class="bg" cx="27" cy="27" r="22"/>
    <circle cx="27" cy="27" r="22" stroke="var(--dom)" stroke-linecap="round"
      stroke-dasharray="${filled} ${circumference}" transform="rotate(-90 27 27)"/>
    <text x="27" y="31.5" text-anchor="middle">${percent(value)}</text></svg>`;
}

function emptyIllustration() {
  return `<svg viewBox="0 0 92 92" aria-hidden="true" fill="none" stroke-width="7" stroke-linecap="round">
    <circle cx="46" cy="46" r="32" stroke="var(--surface-3)"/>
    <path d="M46 14a32 32 0 0 1 27.7 16" stroke="var(--d1)"/>
    <path d="M77 40a32 32 0 0 1-15 33.6" stroke="var(--d2)"/>
    <path d="M51 77.6A32 32 0 0 1 15 52" stroke="var(--d3)"/>
    <circle cx="46" cy="46" r="7" fill="var(--surface-3)" stroke="none"/></svg>`;
}

/* ---------- the structure ---------------------------------------------- */

function setViewerState(html) {
  $("viewerState").innerHTML = html ? `<div class="box">${html}</div>` : "";
}

const ICON_NO_STRUCTURE = '<svg viewBox="0 0 48 48"><path d="M8 34C8 18 20 18 18 26S28 36 26 26 36 14 34 22"/><path d="M6 6l36 36"/></svg>';

function startStructure(job) {
  clearTimeout(state.structureTimer);
  state.structurePolls = 0;
  if (view) view.clear();
  $("viewerReadout").hidden = true;
  $("viewerHint").hidden = true;
  $("structureToolbar").hidden = true;
  $("structureSource").textContent = "";
  renderLegend();
  if (!job.structure) {
    setViewerState(`${ICON_NO_STRUCTURE}<strong>No 3-D structure requested</strong>` +
      "<span>Turn on “Show the 3-D structure” before predicting to see the domains on the fold.</span>");
    return;
  }
  setViewerState('<div class="spinner"></div><strong>Looking for a structure</strong>' +
    "<span>Searching AlphaFold DB for a model of this exact sequence…</span>");
  fetchStructure();
}

async function fetchStructure() {
  const job = state.job;
  if (!job || state.aborted) return;
  let response, body;
  try {
    response = await fetch(`${API}/predictions/${job.job_id}/structure`);
    body = await response.json();
  } catch (problem) {
    structureUnavailable("the service could not be reached");
    return;
  }
  if (state.job !== job) return;
  if (!response.ok) { structureUnavailable((body.error || {}).message || "the structure could not be fetched"); return; }
  state.structure = body.structure;
  if (body.structure.status === "pending") {
    if (++state.structurePolls > STRUCTURE_POLL_LIMIT) {
      structureUnavailable("the lookup is taking too long; try again later");
      return;
    }
    state.structureTimer = setTimeout(fetchStructure, STRUCTURE_POLL_MS);
    return;
  }
  renderStats();
  renderProvenance();
  if (body.structure.status !== "available" || !body.model) {
    structureUnavailable(body.structure.message);
    return;
  }
  try {
    showModel(body.model.data);
  } catch (problem) {
    // Most often WebGL is switched off or unsupported in this browser.
    structureUnavailable(`the 3-D viewer could not start in this browser (${problem.message || problem})`);
  }
}

function structureUnavailable(message) {
  setViewerState(`${ICON_NO_STRUCTURE}<strong>No 3-D structure for this sequence</strong>` +
    `<span>${escapeHtml(message || "")}</span>` +
    "<span>The domains and CATH labels are unaffected: Domaino uses the sequence only.</span>");
  renderLegend();
}

function showModel(pdb) {
  if (!window.StructureView.libraryLoaded()) {
    structureUnavailable("the 3-D viewer library (3Dmol.js, from cdnjs.cloudflare.com) could not be loaded");
    return;
  }
  state.model = pdb;
  setViewerState("");
  if (!view) {
    view = new window.StructureView($("viewer"), {
      onPick: (index) => select(index === null ? state.selected : index),
      onHover: showReadout,
    });
  }
  view.setDomains(state.result.domains);
  view.setMode(state.mode);
  view.setLabels($("toggleLabels").classList.contains("active"));
  view.setBoundaries($("toggleBoundaries").classList.contains("active"));
  view.setSelected(state.selected);
  const residues = view.load(pdb);
  view.resize();
  $("structureToolbar").hidden = false;
  $("viewerHint").hidden = false;

  const s = state.structure;
  const parts = [
    `<strong>AlphaFold DB</strong> · ${escapeHtml(s.model_id || "")}${s.version ? ` (v${s.version})` : ""}`,
    s.description ? escapeHtml(s.description) : "",
    s.organism ? `<i>${escapeHtml(s.organism)}</i>` : "",
    s.mean_plddt != null ? `mean pLDDT ${Number(s.mean_plddt).toFixed(1)}` : "",
    s.entry_url ? `<a href="${escapeHtml(s.entry_url)}" target="_blank" rel="noopener">open entry ↗</a>` : "",
  ].filter(Boolean);
  $("structureSource").innerHTML = parts.join(" · ") +
    "<br>A predicted model of this exact sequence, shown for context. Domaino does not use it." +
    (residues !== state.result.sequence_length
      ? ` <strong>Note:</strong> the model has ${residues} residues; the sequence has ${state.result.sequence_length}.` : "");
  $("exportPdb").hidden = false;
  renderLegend();
}

function showReadout(info) {
  const box = $("viewerReadout");
  if (!info) { box.hidden = true; return; }
  const domain = info.domain === null ? null : state.result.domains[info.domain];
  const where = domain ? `Domain ${domain.index}` : "outside any domain";
  box.innerHTML = `<span class="sw" style="background:${info.colour}"></span>` +
    `${escapeHtml(info.resn)} ${info.resi} · ${where} · pLDDT ${Number(info.plddt).toFixed(1)}`;
  box.hidden = false;
}

function renderLegend() {
  const legend = $("legend");
  if (!state.result || !state.model) { legend.innerHTML = ""; return; }
  if (state.mode === "plddt") {
    legend.innerHTML = window.DomainoPalette.plddtBins.map((bin) =>
      `<span class="legend-item"><span class="sw" style="background:var(${bin.token})"></span>${bin.label}</span>`).join("") +
      '<span class="legend-item">AlphaFold model confidence, not Domaino\'s</span>';
    return;
  }
  legend.innerHTML = state.result.domains.map((d, i) =>
    `<button type="button" class="legend-item${state.selected === i ? " active" : ""}" data-domain="${i}">` +
    `<span class="sw" style="background:${domainColour(i)}"></span>Domain ${d.index}</button>`).join("") +
    '<span class="legend-item"><span class="sw" style="background:var(--unassigned)"></span>outside any domain</span>';
}

/* ---------- theme, health, wiring -------------------------------------- */

function currentTheme() {
  return document.documentElement.dataset.theme ||
    (matchMedia("(prefers-color-scheme: dark)").matches ? "dark" : "light");
}

function setTheme(theme) {
  document.documentElement.dataset.theme = theme;
  try { localStorage.setItem("domaino-theme", theme); } catch (problem) { /* not kept */ }
  if (view) view.refreshTheme();
}

function setStatus(kind, text) {
  $("apiStatus").className = `status-pill ${kind}`;
  $("apiText").textContent = text;
}

async function checkHealth() {
  try {
    const response = await fetch(`${API}/health`);
    const body = await response.json();
    const simulated = body.model && body.model.mode === "mock";
    setStatus(simulated ? "mock" : "ok",
      simulated ? "Connected · simulated model" : `Connected · ${body.model.architecture} model`);
    state.structureSources = body.structure_sources || [];
    if (!state.structureSources.length) {
      $("optStructure").checked = false;
      $("optStructure").disabled = true;
      $("structureNote").textContent = "3-D structure lookup is turned off on this server.";
    }
  } catch (problem) {
    setStatus("off", "API offline");
  }
}

function wire() {
  $("sequence").addEventListener("input", updateInput);
  $("predictButton").addEventListener("click", submit);
  $("cancelButton").addEventListener("click", cancel);
  $("editButton").addEventListener("click", () => { show("submitView"); $("sequence").focus(); });
  $("clearInput").addEventListener("click", () => { $("sequence").value = ""; updateInput(); $("sequence").focus(); });
  $("fileInput").addEventListener("change", (e) => { if (e.target.files[0]) readFile(e.target.files[0]); });
  $("newButton").addEventListener("click", () => {
    stopTimers();
    state.job = null;
    if (view) view.clear();
    show("submitView");
  });

  const drop = $("dropZone");
  for (const type of ["dragenter", "dragover"]) {
    drop.addEventListener(type, (e) => { e.preventDefault(); drop.classList.add("dragging"); });
  }
  for (const type of ["dragleave", "drop"]) {
    drop.addEventListener(type, () => drop.classList.remove("dragging"));
  }
  drop.addEventListener("drop", (e) => {
    e.preventDefault();
    if (e.dataTransfer.files[0]) readFile(e.dataTransfer.files[0]);
  });

  // Anything carrying data-domain selects that domain: map, legend, cards, chips.
  $("resultView").addEventListener("click", (event) => {
    const target = event.target.closest("[data-domain]");
    if (target && !target.closest(".viewer")) select(Number(target.dataset.domain));
  });

  document.querySelectorAll(".segmented button").forEach((button) => {
    button.addEventListener("click", () => {
      document.querySelectorAll(".segmented button").forEach((b) => b.classList.toggle("active", b === button));
      state.mode = button.dataset.mode;
      if (view) view.setMode(state.mode);
      renderLegend();
    });
  });
  for (const [id, setter] of [["toggleLabels", "setLabels"], ["toggleBoundaries", "setBoundaries"]]) {
    $(id).addEventListener("click", () => {
      const on = !$(id).classList.contains("active");
      $(id).classList.toggle("active", on);
      $(id).setAttribute("aria-pressed", String(on));
      if (view) view[setter](on);
    });
  }
  $("resetView").addEventListener("click", () => view && view.reset());
  $("fullscreen").addEventListener("click", () => {
    const wrap = $("viewerWrap");
    if (document.fullscreenElement) document.exitFullscreen();
    else if (wrap.requestFullscreen) wrap.requestFullscreen();
  });
  document.addEventListener("fullscreenchange", () => setTimeout(() => view && view.resize(), 60));
  let resizeTimer = null;
  window.addEventListener("resize", () => {
    clearTimeout(resizeTimer);
    resizeTimer = setTimeout(() => view && view.resize(), 120);
  });
  // The hint has done its job once someone touches the structure.
  $("viewerWrap").addEventListener("pointerdown", () => {
    if (state.model) $("viewerHint").hidden = true;
  });

  $("exportJson").addEventListener("click", () => {
    if (!state.job) return;
    const name = (state.result.name || state.job.job_id).replace(/[^\w.-]+/g, "_").slice(0, 60);
    const payload = {job_id: state.job.job_id, result: state.result, structure: state.structure};
    download(`${name}.domaino.json`, JSON.stringify(payload, null, 2), "application/json");
  });
  $("exportPdb").addEventListener("click", () => {
    if (!state.model) return;
    download(`${(state.structure && state.structure.model_id) || "structure"}.pdb`, state.model, "chemical/x-pdb");
  });

  $("themeToggle").addEventListener("click", () => setTheme(currentTheme() === "dark" ? "light" : "dark"));
  matchMedia("(prefers-color-scheme: dark)").addEventListener("change", () => view && view.refreshTheme());
}

(function start() {
  try {
    const saved = localStorage.getItem("domaino-theme");
    if (saved) document.documentElement.dataset.theme = saved;
  } catch (problem) { /* default theme */ }
  renderExamples();
  wire();
  updateInput();
  checkHealth();
})();
