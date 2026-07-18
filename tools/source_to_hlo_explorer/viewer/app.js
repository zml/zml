"use strict";

const ARTIFACTS = {
  source: "/artifact/source.zig",
  stable: "/artifact/stablehlo.mlir",
  hlo: "/artifact/hlo.before_optimizations.txt",
  mapping: "/artifact/mapping.json",
};

const PANE_ORDER = ["source", "stable", "hlo"];
const PANE_LABELS = {
  source: "ZML source",
  stable: "StableHLO",
  hlo: "pre-optimization HLO",
};

const elements = {
  status: document.querySelector("#status"),
  reload: document.querySelector("#reload"),
  details: document.querySelector("#details-content"),
  template: document.querySelector("#line-template"),
  panes: {
    source: document.querySelector("#source-code"),
    stable: document.querySelector("#stable-code"),
    hlo: document.querySelector("#hlo-code"),
  },
  counts: {
    source: document.querySelector("#source-count"),
    stable: document.querySelector("#stable-count"),
    hlo: document.querySelector("#hlo-count"),
  },
};

const state = {
  graph: null,
  lines: { source: [], stable: [], hlo: [] },
  selected: null,
};

elements.reload.addEventListener("click", () => loadArtifacts());

for (const paneName of PANE_ORDER) {
  elements.panes[paneName].addEventListener("keydown", (event) => {
    handleLineKeydown(event, paneName);
  });
}

loadArtifacts();

async function loadArtifacts() {
  setStatus("Loading artifacts…");
  elements.reload.disabled = true;

  try {
    const [source, stable, hlo, mapping] = await Promise.all([
      fetchText(ARTIFACTS.source),
      fetchText(ARTIFACTS.stable),
      fetchText(ARTIFACTS.hlo),
      fetchJson(ARTIFACTS.mapping),
    ]);

    state.lines.source = splitLines(source);
    state.lines.stable = splitLines(stable);
    state.lines.hlo = splitLines(hlo);
    state.graph = normalizeMapping(mapping, state.lines);
    state.selected = null;

    for (const paneName of PANE_ORDER) {
      renderPane(paneName);
    }
    renderEmptyDetails();

    const mapped = state.graph.sources.size + state.graph.stable.size + state.graph.hlo.size;
    setStatus(`${mapped} provenance records loaded`);
  } catch (error) {
    console.error(error);
    setStatus(error instanceof Error ? error.message : String(error), true);
    renderLoadError(error);
  } finally {
    elements.reload.disabled = false;
  }
}

async function fetchText(url) {
  const response = await fetch(url, { cache: "no-store" });
  if (!response.ok) {
    throw new Error(`Could not load ${url} (${response.status})`);
  }
  return response.text();
}

async function fetchJson(url) {
  const response = await fetch(url, { cache: "no-store" });
  if (!response.ok) {
    throw new Error(`Could not load ${url} (${response.status})`);
  }
  try {
    return await response.json();
  } catch (error) {
    throw new Error(`Invalid JSON in ${url}: ${error.message}`);
  }
}

function splitLines(text) {
  const lines = text.replace(/\r\n?/g, "\n").split("\n");
  if (lines.length > 1 && lines.at(-1) === "") {
    lines.pop();
  }
  return lines;
}

function setStatus(message, isError = false) {
  elements.status.textContent = message;
  elements.status.classList.toggle("error", isError);
}

function renderPane(paneName) {
  const container = elements.panes[paneName];
  const fragment = document.createDocumentFragment();
  container.replaceChildren();

  state.lines[paneName].forEach((text, index) => {
    const line = index + 1;
    const row = elements.template.content.firstElementChild.cloneNode(true);
    const mappedIds = state.graph.lineIndex[paneName].get(line);

    row.dataset.pane = paneName;
    row.dataset.line = String(line);
    row.querySelector(".line-number").textContent = String(line);
    row.querySelector(".line-text").textContent = text || " ";
    row.classList.toggle("has-mapping", Boolean(mappedIds?.size));
    row.classList.toggle("is-unmapped", !mappedIds?.size);
    row.dataset.baseAriaLabel = `${PANE_LABELS[paneName]} line ${line}${mappedIds?.size ? ", mapped" : ", not mapped"}: ${text || "blank"}`;
    row.setAttribute("aria-label", row.dataset.baseAriaLabel);
    row.addEventListener("click", () => selectLine(paneName, line, true));
    fragment.append(row);
  });

  container.append(fragment);
  const count = state.lines[paneName].length;
  elements.counts[paneName].textContent = `${count} ${count === 1 ? "line" : "lines"}`;
}

function selectLine(paneName, line, shouldScroll) {
  if (!state.graph) return;

  const seedIds = state.graph.lineIndex[paneName].get(line) || new Set();
  const selection = expandSelection(paneName, seedIds);
  selection.activePane = paneName;
  selection.activeLine = line;
  state.selected = selection;

  updateHighlights();
  renderDetails(selection);
  if (shouldScroll) scrollRelatedIntoView(selection, paneName);
}

function expandSelection(seedPane, seedIds) {
  const selection = {
    source: new Set(),
    stable: new Set(),
    hlo: new Set(),
  };
  for (const id of seedIds) selection[seedPane].add(id);

  let changed = true;
  while (changed) {
    changed = false;
    for (const sourceId of [...selection.source]) {
      const source = state.graph.sources.get(sourceId);
      if (!source) continue;
      changed = addAll(selection.stable, source.stableIds) || changed;
      changed = addAll(selection.hlo, source.hloIds) || changed;
    }
    for (const stableId of [...selection.stable]) {
      const stable = state.graph.stable.get(stableId);
      if (!stable) continue;
      changed = addAll(selection.source, stable.sourceIds) || changed;
      changed = addAll(selection.hlo, stable.hloIds) || changed;
    }
    for (const hloId of [...selection.hlo]) {
      const hlo = state.graph.hlo.get(hloId);
      if (!hlo) continue;
      changed = addAll(selection.source, hlo.sourceIds) || changed;
      changed = addAll(selection.stable, hlo.stableIds) || changed;
    }
  }
  return selection;
}

function addAll(target, values) {
  let changed = false;
  for (const value of values) {
    if (!target.has(value)) {
      target.add(value);
      changed = true;
    }
  }
  return changed;
}

function updateHighlights() {
  for (const paneName of PANE_ORDER) {
    for (const row of elements.panes[paneName].querySelectorAll(".code-line")) {
      const line = Number(row.dataset.line);
      const lineIds = state.graph.lineIndex[paneName].get(line) || new Set();
      const related = intersects(lineIds, state.selected[paneName]);
      const active = state.selected.activePane === paneName && state.selected.activeLine === line;
      row.classList.toggle("is-related", related);
      row.classList.toggle("is-active", active);
      row.setAttribute("aria-pressed", active ? "true" : "false");
      const relation = related && !active ? ", related to the active selection" : active ? ", active selection" : "";
      row.setAttribute("aria-label", `${row.dataset.baseAriaLabel}${relation}`);
    }
  }
}

function intersects(left, right) {
  for (const value of left) {
    if (right.has(value)) return true;
  }
  return false;
}

function scrollRelatedIntoView(selection, originPane) {
  for (const paneName of PANE_ORDER) {
    if (paneName === originPane) continue;
    const lines = selectedLines(paneName, selection[paneName]);
    const firstLine = Math.min(...lines);
    if (Number.isFinite(firstLine)) {
      lineButton(paneName, firstLine)?.scrollIntoView({ block: "nearest", inline: "nearest" });
    }
  }
}

function selectedLines(paneName, ids) {
  const result = new Set();
  const records = graphRecords(paneName);
  for (const id of ids) {
    const record = records.get(id);
    if (!record) continue;
    if (paneName === "source") {
      for (let line = record.line; line <= record.endLine; line += 1) result.add(line);
    } else {
      for (const line of record.lines) result.add(line);
    }
  }
  return result;
}

function graphRecords(paneName) {
  if (paneName === "source") return state.graph.sources;
  return state.graph[paneName];
}

function handleLineKeydown(event, paneName) {
  const current = event.target.closest(".code-line");
  if (!current) return;

  const line = Number(current.dataset.line);
  let targetPane = paneName;
  let targetLine = line;

  if (event.key === "ArrowDown") targetLine += 1;
  else if (event.key === "ArrowUp") targetLine -= 1;
  else if (event.key === "Home") targetLine = 1;
  else if (event.key === "End") targetLine = state.lines[paneName].length;
  else if (event.key === "ArrowLeft") targetPane = PANE_ORDER[Math.max(0, PANE_ORDER.indexOf(paneName) - 1)];
  else if (event.key === "ArrowRight") targetPane = PANE_ORDER[Math.min(PANE_ORDER.length - 1, PANE_ORDER.indexOf(paneName) + 1)];
  else if (["1", "2", "3"].includes(event.key)) targetPane = PANE_ORDER[Number(event.key) - 1];
  else return;

  event.preventDefault();
  targetLine = Math.max(1, Math.min(targetLine, state.lines[targetPane].length));
  const target = lineButton(targetPane, targetLine);
  target?.focus();
  if (target) selectLine(targetPane, targetLine, true);
}

function lineButton(paneName, line) {
  return elements.panes[paneName].querySelector(`.code-line[data-line="${line}"]`);
}

function renderEmptyDetails() {
  const message = document.createElement("p");
  message.className = "empty-details";
  message.textContent = "Select a mapped source expression or IR operation to trace its lowering.";
  elements.details.replaceChildren(message);
}

function renderLoadError(error) {
  for (const paneName of PANE_ORDER) {
    elements.panes[paneName].replaceChildren();
    elements.counts[paneName].textContent = "Unavailable";
  }
  const message = document.createElement("p");
  message.className = "empty-details";
  message.textContent = `Artifacts could not be loaded. ${error instanceof Error ? error.message : String(error)}`;
  elements.details.replaceChildren(message);
}

function renderDetails(selection) {
  const total = selection.source.size + selection.stable.size + selection.hlo.size;
  if (total === 0) {
    const message = document.createElement("p");
    message.className = "empty-details";
    message.textContent = `${PANE_LABELS[selection.activePane]} line ${selection.activeLine} has no provenance mapping.`;
    elements.details.replaceChildren(message);
    return;
  }

  const grid = document.createElement("div");
  grid.className = "detail-grid";
  grid.append(
    detailCard("Source", detailRowsForSources(selection.source)),
    detailCard("StableHLO operations", detailRowsForStable(selection.stable)),
    detailCard("HLO instructions", detailRowsForHlo(selection.hlo)),
  );
  elements.details.replaceChildren(grid);
}

function detailCard(title, rows) {
  const card = document.createElement("section");
  card.className = "detail-card";
  const heading = document.createElement("h3");
  heading.textContent = title;
  const list = document.createElement("ul");

  if (rows.length === 0) rows.push(["No mapped records", ""]);
  for (const [primary, secondary] of rows) {
    const item = document.createElement("li");
    const first = document.createElement("span");
    first.className = "detail-primary";
    first.textContent = primary;
    item.append(first);
    if (secondary) {
      const second = document.createElement("span");
      second.className = "detail-secondary";
      second.textContent = secondary;
      item.append(second);
    }
    list.append(item);
  }
  card.append(heading, list);
  return card;
}

function detailRowsForSources(ids) {
  return [...ids].map((id) => {
    const record = state.graph.sources.get(id);
    if (!record) return [id, "Missing source record"];
    const start = `${record.file}:${record.line}:${record.column}`;
    const end = record.endLine !== record.line || record.endColumn !== record.column
      ? `–${record.endLine}:${record.endColumn}`
      : "";
    return [`${start}${end}`, record.label ? `${record.label} · ${id}` : id];
  });
}

function detailRowsForStable(ids) {
  return [...ids].map((id) => {
    const record = state.graph.stable.get(id);
    if (!record) return [id, "Missing StableHLO record"];
    const lines = formatLines(record.lines);
    return [record.operation || stableMarker(id), `${id}${lines ? ` · StableHLO ${lines}` : ""}`];
  });
}

function detailRowsForHlo(ids) {
  return [...ids].map((id) => {
    const record = state.graph.hlo.get(id);
    if (!record) return [id, "Missing HLO record"];
    const lines = formatLines(record.lines);
    const mapping = mappingLabel(record.mapping);
    const metadata = [record.opcode, lines ? `HLO ${lines}` : "", mapping, id].filter(Boolean).join(" · ");
    return [record.name || id, metadata];
  });
}

function mappingLabel(mapping) {
  if (mapping === "metadata") return "direct metadata";
  if (mapping === "dataflow_operand") return "operand dataflow";
  return mapping;
}

function formatLines(lines) {
  const values = [...lines].sort((left, right) => left - right);
  if (values.length === 0) return "";
  return `${values.length === 1 ? "line" : "lines"} ${values.join(", ")}`;
}

function normalizeMapping(raw, textLines) {
  if (!raw || typeof raw !== "object") {
    throw new Error("mapping.json must contain a JSON object");
  }

  const graph = {
    sources: new Map(),
    stable: new Map(),
    hlo: new Map(),
    lineIndex: { source: new Map(), stable: new Map(), hlo: new Map() },
  };

  const sourceCollection = raw.sources ?? raw.source_spans ?? raw.sourceSpans;
  for (const [entryKey, record] of collectionEntries(sourceCollection, "source")) {
    const location = parseLocation(entryKey);
    const file = String(firstDefined(record.file, record.filename, location?.file, "source.zig"));
    const line = positiveInteger(firstDefined(record.line, record.start_line, record.startLine, location?.line), 1);
    const column = positiveInteger(firstDefined(record.column, record.col, record.start_column, record.startColumn, location?.column), 1);
    const endLine = positiveInteger(firstDefined(record.end_line, record.endLine), line);
    const endColumn = positiveInteger(firstDefined(record.end_column, record.endColumn), column);
    const id = stringId(firstDefined(record.id, record.source_id, record.sourceId, entryKey, `${file}:${line}:${column}`));
    graph.sources.set(id, {
      id,
      file,
      line,
      column,
      endLine: Math.max(line, endLine),
      endColumn,
      label: optionalString(firstDefined(record.label, record.expression)),
      stableIds: idSet(firstDefined(record.stable_op_ids, record.stableOpIds, record.stable_ops)),
      hloIds: idSet(firstDefined(record.hlo_instruction_ids, record.hloInstructionIds)),
    });
  }

  const stableCollection = raw.stable_ops ?? raw.stableOps ?? raw.operations;
  for (const [entryKey, record] of collectionEntries(stableCollection, "stable")) {
    const id = stringId(firstDefined(record.id, record.stable_op_id, record.stableOpId, entryKey));
    if (!id) continue;
    graph.stable.set(id, {
      id,
      sourceIds: idSet(firstDefined(record.source_ids, record.sourceIds, record.source_id, record.sourceId, record.source)),
      hloIds: idSet(firstDefined(record.hlo_instruction_ids, record.hloInstructionIds, record.hlo_ids, record.hloIds)),
      lines: lineSet(record, ["stablehlo_lines", "stablehloLines", "lines", "line"]),
      operation: optionalString(firstDefined(record.operation, record.op, record.opcode, record.name)),
    });
  }

  const hloCollection = raw.hlo_instructions ?? raw.hloInstructions ?? raw.hlo_ops ?? raw.hloOps;
  for (const [entryKey, record] of collectionEntries(hloCollection, "hlo")) {
    const id = stringId(firstDefined(record.id, record.hlo_instruction_id, record.hloInstructionId, entryKey, record.name));
    if (!id) continue;
    graph.hlo.set(id, {
      id,
      stableIds: idSet(firstDefined(record.stable_op_ids, record.stableOpIds, record.stable_op_id, record.stableOpId)),
      sourceIds: idSet(firstDefined(record.source_ids, record.sourceIds, record.source_id, record.sourceId, record.source)),
      lines: lineSet(record, ["hlo_lines", "hloLines", "lines", "line"]),
      name: optionalString(firstDefined(record.name, record.instruction)),
      opcode: optionalString(firstDefined(record.opcode, record.operation, record.op)),
      mapping: optionalString(firstDefined(record.mapping, record.mapping_kind, record.mappingKind)),
    });
  }

  addProvenanceRecords(graph, raw.provenance_records ?? raw.provenanceRecords ?? raw.provenance ?? raw.records);
  connectGraph(graph);
  inferMissingIrLines(graph, textLines);
  indexGraphLines(graph, textLines);
  return graph;
}

function collectionEntries(collection, prefix) {
  if (!collection) return [];
  if (Array.isArray(collection)) {
    return collection
      .filter((value) => value && typeof value === "object")
      .map((value, index) => [stringId(firstDefined(value.id, value[`${prefix}_id`], `${prefix}.${index + 1}`)), value]);
  }
  if (typeof collection === "object") {
    return Object.entries(collection)
      .filter(([, value]) => value && typeof value === "object");
  }
  return [];
}

function addProvenanceRecords(graph, records) {
  if (!Array.isArray(records)) return;
  for (const record of records) {
    if (!record || typeof record !== "object") continue;
    const stableId = stringId(firstDefined(record.stable_op_id, record.stableOpId, record.id));
    if (!stableId) continue;
    const file = String(firstDefined(record.file, record.filename, "source.zig"));
    const line = positiveInteger(record.line, 1);
    const column = positiveInteger(firstDefined(record.column, record.col), 1);
    const sourceId = stringId(firstDefined(record.source_id, record.sourceId, `${file}:${line}:${column}`));

    if (!graph.sources.has(sourceId)) {
      graph.sources.set(sourceId, {
        id: sourceId,
        file,
        line,
        column,
        endLine: positiveInteger(firstDefined(record.end_line, record.endLine), line),
        endColumn: positiveInteger(firstDefined(record.end_column, record.endColumn), column),
        label: optionalString(record.label),
        stableIds: new Set(),
        hloIds: new Set(),
      });
    }
    if (!graph.stable.has(stableId)) {
      graph.stable.set(stableId, {
        id: stableId,
        sourceIds: new Set(),
        hloIds: new Set(),
        lines: lineSet(record, ["stablehlo_lines", "stablehloLines"]),
        operation: optionalString(firstDefined(record.operation, record.op)),
      });
    }
    graph.sources.get(sourceId).stableIds.add(stableId);
    graph.stable.get(stableId).sourceIds.add(sourceId);
  }
}

function connectGraph(graph) {
  for (const source of graph.sources.values()) {
    for (const stableId of source.stableIds) {
      const stable = ensureStable(graph, stableId);
      stable.sourceIds.add(source.id);
    }
    for (const hloId of source.hloIds) {
      const hlo = ensureHlo(graph, hloId);
      hlo.sourceIds.add(source.id);
    }
  }

  for (const stable of [...graph.stable.values()]) {
    for (const sourceId of stable.sourceIds) {
      const source = graph.sources.get(sourceId);
      if (source) source.stableIds.add(stable.id);
    }
    for (const hloId of stable.hloIds) {
      const hlo = ensureHlo(graph, hloId);
      hlo.stableIds.add(stable.id);
    }
  }

  for (const hlo of [...graph.hlo.values()]) {
    for (const stableId of hlo.stableIds) {
      const stable = ensureStable(graph, stableId);
      stable.hloIds.add(hlo.id);
      for (const sourceId of stable.sourceIds) hlo.sourceIds.add(sourceId);
    }
    for (const sourceId of hlo.sourceIds) {
      const source = graph.sources.get(sourceId);
      if (source) source.hloIds.add(hlo.id);
    }
  }
}

function ensureStable(graph, id) {
  if (!graph.stable.has(id)) {
    graph.stable.set(id, {
      id,
      sourceIds: new Set(),
      hloIds: new Set(),
      lines: new Set(),
      operation: "",
    });
  }
  return graph.stable.get(id);
}

function ensureHlo(graph, id) {
  if (!graph.hlo.has(id)) {
    graph.hlo.set(id, {
      id,
      stableIds: new Set(),
      sourceIds: new Set(),
      lines: new Set(),
      name: "",
      opcode: "",
      mapping: "",
    });
  }
  return graph.hlo.get(id);
}

function inferMissingIrLines(graph, textLines) {
  for (const stable of graph.stable.values()) {
    if (stable.lines.size === 0) {
      addMarkerLines(stable.lines, textLines.stable, stableMarker(stable.id));
    }
  }

  for (const hlo of graph.hlo.values()) {
    if (hlo.lines.size === 0 && hlo.name) {
      const namePattern = new RegExp(`(?:^|[ ,(])%?${escapeRegExp(hlo.name)}(?:\\s|=|,|\\)|$)`);
      textLines.hlo.forEach((line, index) => {
        if (namePattern.test(line)) hlo.lines.add(index + 1);
      });
    }
    if (hlo.lines.size === 0) {
      for (const stableId of hlo.stableIds) {
        addMarkerLines(hlo.lines, textLines.hlo, stableMarker(stableId));
      }
    }
  }
}

function addMarkerLines(target, lines, marker) {
  lines.forEach((line, index) => {
    if (line.includes(marker)) target.add(index + 1);
  });
}

function stableMarker(id) {
  return id.startsWith("zml.stable_op.") ? id : `zml.stable_op.${id}`;
}

function indexGraphLines(graph, textLines) {
  for (const source of graph.sources.values()) {
    const endLine = Math.min(source.endLine, textLines.source.length);
    for (let line = Math.max(1, source.line); line <= endLine; line += 1) {
      addLineIndex(graph.lineIndex.source, line, source.id);
    }
  }
  for (const stable of graph.stable.values()) {
    for (const line of stable.lines) {
      if (line <= textLines.stable.length) addLineIndex(graph.lineIndex.stable, line, stable.id);
    }
  }
  for (const hlo of graph.hlo.values()) {
    for (const line of hlo.lines) {
      if (line <= textLines.hlo.length) addLineIndex(graph.lineIndex.hlo, line, hlo.id);
    }
  }
}

function addLineIndex(index, line, id) {
  if (!index.has(line)) index.set(line, new Set());
  index.get(line).add(id);
}

function lineSet(record, fields) {
  const result = new Set();
  for (const field of fields) {
    if (!(field in record)) continue;
    for (const value of arrayValue(record[field])) {
      const line = positiveInteger(value, 0);
      if (line > 0) result.add(line);
    }
  }
  return result;
}

function idSet(value) {
  const result = new Set();
  for (const item of arrayValue(value)) {
    const id = typeof item === "object" && item !== null
      ? stringId(firstDefined(item.id, item.stable_op_id, item.hlo_instruction_id, item.source_id))
      : stringId(item);
    if (id) result.add(id);
  }
  return result;
}

function arrayValue(value) {
  if (value === undefined || value === null) return [];
  return Array.isArray(value) ? value : [value];
}

function firstDefined(...values) {
  return values.find((value) => value !== undefined && value !== null);
}

function stringId(value) {
  if (value === undefined || value === null) return "";
  return String(value);
}

function optionalString(value) {
  return value === undefined || value === null ? "" : String(value);
}

function positiveInteger(value, fallback) {
  const parsed = Number(value);
  return Number.isInteger(parsed) && parsed > 0 ? parsed : fallback;
}

function parseLocation(value) {
  if (!value) return null;
  const match = String(value).match(/^(.*):(\d+):(\d+)$/);
  if (!match) return null;
  return { file: match[1], line: Number(match[2]), column: Number(match[3]) };
}

function escapeRegExp(value) {
  return value.replace(/[.*+?^${}()|[\]\\]/g, "\\$&");
}
