"use strict";
/* The 3-D structure view (task T4.5), on 3Dmol.js.

   Residue numbers in the model are positions in the submitted sequence (the
   backend only accepts a model of the identical sequence), so a domain's
   segments select its residues directly and the colours here are the colours
   of the sequence map (milestone M4.3). Domain colours come from the same CSS
   tokens the rest of the page uses. */

(function () {
  const PALETTE_SIZE = 8;
  const PLDDT_BINS = [
    {min: 90, token: "--plddt-vh", label: "Very high (≥ 90)"},
    {min: 70, token: "--plddt-c", label: "Confident (70–90)"},
    {min: 50, token: "--plddt-l", label: "Low (50–70)"},
    {min: -Infinity, token: "--plddt-vl", label: "Very low (< 50)"},
  ];

  function cssVar(name) {
    return getComputedStyle(document.documentElement).getPropertyValue(name).trim();
  }

  function domainToken(index) {
    return `--d${(index % PALETTE_SIZE) + 1}`;
  }

  function hexToInt(hex) {
    return parseInt(hex.replace("#", ""), 16);
  }

  /* Blend two #rrggbb colours; t = 0 gives a, t = 1 gives b. */
  function blend(a, b, t) {
    const x = hexToInt(a), y = hexToInt(b);
    const channel = (shift) => Math.round(
      ((x >> shift) & 255) * (1 - t) + ((y >> shift) & 255) * t);
    return (channel(16) << 16) | (channel(8) << 8) | channel(0);
  }

  class StructureView {
    constructor(container, {onPick, onHover} = {}) {
      this.container = container;
      this.onPick = onPick || (() => {});
      this.onHover = onHover || (() => {});
      this.viewer = null;
      this.model = null;
      this.residues = new Map();   // resi -> {resn, b, x, y, z} of the C-alpha
      this.domainOf = new Map();   // resi -> index into this.domains
      this.domains = [];
      this.mode = "domain";
      this.selected = null;
      this.showLabels = true;
      this.showBoundaries = false;
    }

    static libraryLoaded() {
      return typeof window.$3Dmol !== "undefined";
    }

    /* Load a PDB model; returns the number of residues it contains. */
    load(pdbText) {
      if (!this.viewer) {
        // An opaque background in the theme's colour rather than a transparent
        // one: 3Dmol's offscreen canvas ignores alpha in Firefox and Safari,
        // and its depth fog blends towards the background colour anyway.
        this.viewer = window.$3Dmol.createViewer(this.container, {antialias: true});
        if (this.viewer.setHoverDuration) this.viewer.setHoverDuration(30);
      }
      this.viewer.clear();
      this.model = this.viewer.addModel(pdbText, "pdb");
      this.residues.clear();
      for (const atom of this.model.selectedAtoms({atom: "CA"})) {
        this.residues.set(atom.resi, {resn: atom.resn, b: atom.b, x: atom.x, y: atom.y, z: atom.z});
      }
      this.viewer.setHoverable({}, true,
        (atom) => this.onHover(this.describe(atom)),
        () => this.onHover(null));
      this.viewer.setClickable({}, true, (atom) => {
        const index = this.domainOf.get(atom.resi);
        this.onPick(index === undefined ? null : index);
      });
      this.apply();
      this.viewer.zoomTo();
      this.viewer.render();
      return this.residues.size;
    }

    setDomains(domains) {
      this.domains = domains;
      this.domainOf.clear();
      domains.forEach((domain, index) => {
        for (const segment of domain.segments) {
          for (let resi = segment.start; resi <= segment.end; resi++) {
            this.domainOf.set(resi, index);
          }
        }
      });
      this.apply();
    }

    setMode(mode) { this.mode = mode; this.apply(); }
    setSelected(index) { this.selected = index; this.apply(); }
    setLabels(on) { this.showLabels = on; this.apply(); }
    setBoundaries(on) { this.showBoundaries = on; this.apply(); }
    refreshTheme() { this.apply(); }

    reset() {
      if (!this.viewer) return;
      this.viewer.zoomTo();
      this.viewer.render();
    }

    resize() {
      if (!this.viewer) return;
      this.viewer.resize();
      this.viewer.render();
    }

    clear() {
      if (this.viewer) {
        this.viewer.clear();
        this.viewer.render();
      }
      this.model = null;
      this.residues.clear();
      this.selected = null;
    }

    /* Hover read-out: residue, its domain, and the model's own confidence. */
    describe(atom) {
      const index = this.domainOf.get(atom.resi);
      return {
        resi: atom.resi,
        resn: atom.resn,
        plddt: atom.b,
        domain: index === undefined ? null : index,
        colour: "#" + this.colourAt(atom).toString(16).padStart(6, "0"),
      };
    }

    // -- drawing -----------------------------------------------------------

    apply() {
      if (!this.model) return;
      const viewer = this.viewer;
      this.theme = {
        background: cssVar("--viewer-bg"),
        unassigned: cssVar("--unassigned"),
        muted: cssVar("--muted-structure"),
        ink: cssVar("--ink"),
        domains: Array.from({length: PALETTE_SIZE}, (_, i) => cssVar(domainToken(i))),
        plddt: PLDDT_BINS.map((bin) => ({min: bin.min, colour: cssVar(bin.token)})),
      };

      viewer.setBackgroundColor(hexToInt(this.theme.background), 1);
      viewer.setStyle({}, {cartoon: {colorfunc: (atom) => this.colourAt(atom)}});

      viewer.removeAllSurfaces();
      if (this.mode === "domain" && this.selected !== null && this.domains[this.selected]) {
        // A translucent envelope makes the picked domain stand out from any angle.
        viewer.addSurface(window.$3Dmol.SurfaceType.VDW,
          {opacity: 0.2, color: this.domainColour(this.selected)},
          {resi: this.ranges(this.selected)});
      }

      viewer.removeAllLabels();
      if (this.showLabels && this.mode === "domain") {
        this.domains.forEach((domain, index) => {
          const centre = this.centroid(index);
          if (!centre) return;
          const faded = this.selected !== null && this.selected !== index;
          viewer.addLabel(`D${domain.index}`, {
            position: centre,
            backgroundColor: this.domainColour(index),
            backgroundOpacity: faded ? 0.35 : 0.95,
            fontColor: "#ffffff",
            fontOpacity: faded ? 0.6 : 1,
            fontSize: 13,
            borderThickness: 0,
            inFront: true,
            showBackground: true,
            alignment: "center",
          });
        });
      }

      viewer.removeAllShapes();
      if (this.showBoundaries) {
        for (const domain of this.domains) {
          for (const segment of domain.segments) {
            for (const resi of [segment.start, segment.end]) {
              const residue = this.residues.get(resi);
              if (!residue) continue;
              viewer.addSphere({
                center: {x: residue.x, y: residue.y, z: residue.z},
                radius: 1.25,
                color: this.theme.ink,
              });
            }
          }
        }
      }
      viewer.render();
    }

    colourAt(atom) {
      if (this.mode === "plddt") {
        const bin = this.theme.plddt.find((entry) => atom.b >= entry.min);
        return hexToInt(bin.colour);
      }
      const index = this.domainOf.get(atom.resi);
      if (index === undefined) {
        return this.selected === null
          ? hexToInt(this.theme.unassigned)
          : blend(this.theme.unassigned, this.theme.muted, 0.5);
      }
      const colour = this.theme.domains[index % PALETTE_SIZE];
      if (this.selected !== null && this.selected !== index) {
        // Other domains stay faintly tinted, so the fold keeps its context.
        return blend(colour, this.theme.muted, 0.7);
      }
      return hexToInt(colour);
    }

    domainColour(index) {
      return this.theme.domains[index % PALETTE_SIZE];
    }

    ranges(index) {
      return this.domains[index].segments.map((s) => `${s.start}-${s.end}`);
    }

    centroid(index) {
      let x = 0, y = 0, z = 0, n = 0;
      for (const segment of this.domains[index].segments) {
        for (let resi = segment.start; resi <= segment.end; resi++) {
          const residue = this.residues.get(resi);
          if (!residue) continue;
          x += residue.x; y += residue.y; z += residue.z; n++;
        }
      }
      return n ? {x: x / n, y: y / n, z: z / n} : null;
    }
  }

  window.StructureView = StructureView;
  window.DomainoPalette = {size: PALETTE_SIZE, token: domainToken, plddtBins: PLDDT_BINS};
})();
