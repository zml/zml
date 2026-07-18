"use strict";

const ARTIFACTS = {
  source: "/artifact/source.zig",
  stable: "/artifact/stablehlo.mlir",
  hlo: "/artifact/hlo.before_optimizations.txt",
  mapping: "/artifact/mapping.json",
};

const PANE_ORDER = ["source", "stable", "hlo"];
const IR_PANES = ["stable", "hlo"];
const PANE_LABELS = {
  source: "ZML source",
  stable: "StableHLO",
  hlo: "pre-optimization HLO",
};
const utf8Encoder = new TextEncoder();
const scrollBehavior = window.matchMedia("(prefers-reduced-motion: reduce)").matches ? "auto" : "smooth";

const elements = {
  status: document.querySelector("#status"),
  reload: document.querySelector("#reload"),
  details: document.querySelector("#details-content"),
  template: document.querySelector("#line-template"),
  irPane: document.querySelector("#ir-pane"),
  irCount: document.querySelector("#ir-count"),
  irToggles: [...document.querySelectorAll("[data-ir-pane]")],
  panes: {
    source: document.querySelector("#source-code"),
    stable: document.querySelector("#stable-code"),
    hlo: document.querySelector("#hlo-code"),
  },
  counts: {
    source: document.querySelector("#source-count"),
  },
};

const state = {
  graph: null,
  lines: { source: [], stable: [], hlo: [] },
  selected: null,
  activeIrPane: "stable",
};

elements.reload.addEventListener("click", () => loadArtifacts());

for (const toggle of elements.irToggles) {
  toggle.addEventListener("click", () => setActiveIrPane(toggle.dataset.irPane, true));
  toggle.addEventListener("keydown", handleIrToggleKeydown);
}

for (const paneName of PANE_ORDER) {
  elements.panes[paneName].addEventListener("keydown", (event) => {
    handleLineKeydown(event, paneName);
  });
}

loadArtifacts();

async function loadArtifacts() {
  setStatus("Loading artifacts…");
  elements.reload.disabled = true;
  setIrTogglesDisabled(true);

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
    setActiveIrPane(state.activeIrPane, false);
    renderEmptyDetails();

    const mapped = state.graph.sources.size + state.graph.stable.size + state.graph.hlo.size;
    setStatus(`${mapped} provenance records loaded`);
  } catch (error) {
    console.error(error);
    setStatus(error instanceof Error ? error.message : String(error), true);
    renderLoadError(error);
  } finally {
    elements.reload.disabled = false;
    setIrTogglesDisabled(false);
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
    const lineText = row.querySelector(".line-text");
    const hasExactRanges = paneName === "source" && renderSourceLineText(lineText, text, line);
    if (!hasExactRanges) lineText.textContent = text || " ";
    row.classList.toggle("has-mapping", Boolean(mappedIds?.size));
    row.classList.toggle("is-unmapped", !mappedIds?.size);
    row.classList.toggle("has-exact-ranges", hasExactRanges);
    row.dataset.baseAriaLabel = `${PANE_LABELS[paneName]} line ${line}${mappedIds?.size ? ", mapped" : ", not mapped"}: ${text || "blank"}`;
    row.setAttribute("aria-label", row.dataset.baseAriaLabel);
    row.addEventListener("click", (event) => {
      const sourceRange = event.target.closest?.(".source-range");
      if (paneName === "source" && sourceRange?.sourceIds) {
        selectIds(paneName, line, sourceRange.sourceIds, true);
      } else {
        selectLine(paneName, line, true);
      }
    });
    fragment.append(row);
  });

  container.append(fragment);
  updateLineCount(paneName);
}

function setActiveIrPane(paneName, shouldFocus) {
  if (!IR_PANES.includes(paneName)) return;
  state.activeIrPane = paneName;
  elements.irPane.dataset.activeIr = paneName;

  for (const irPaneName of IR_PANES) {
    const isActive = irPaneName === paneName;
    elements.panes[irPaneName].hidden = !isActive;
    const toggle = elements.irToggles.find((candidate) => candidate.dataset.irPane === irPaneName);
    toggle.setAttribute("aria-selected", String(isActive));
    toggle.tabIndex = isActive ? 0 : -1;
  }

  updateLineCount(paneName);
  if (state.selected) {
    updateHighlights();
    const sourceLine = firstSelectedLine("source", state.selected.source);
    alignRelatedPane(state.selected, "source", sourceLine);
  }
  if (shouldFocus) {
    elements.irToggles.find((toggle) => toggle.dataset.irPane === paneName)?.focus();
  }
}

function handleIrToggleKeydown(event) {
  let nextIndex = IR_PANES.indexOf(state.activeIrPane);
  if (event.key === "ArrowLeft" || event.key === "ArrowUp") nextIndex -= 1;
  else if (event.key === "ArrowRight" || event.key === "ArrowDown") nextIndex += 1;
  else if (event.key === "Home") nextIndex = 0;
  else if (event.key === "End") nextIndex = IR_PANES.length - 1;
  else return;

  event.preventDefault();
  nextIndex = (nextIndex + IR_PANES.length) % IR_PANES.length;
  setActiveIrPane(IR_PANES[nextIndex], true);
}

function setIrTogglesDisabled(disabled) {
  for (const toggle of elements.irToggles) toggle.disabled = disabled;
}

function updateLineCount(paneName) {
  const count = state.lines[paneName].length;
  const text = `${count} ${count === 1 ? "line" : "lines"}`;
  if (paneName === "source") elements.counts.source.textContent = text;
  else if (paneName === state.activeIrPane) elements.irCount.textContent = text;
}

function renderSourceLineText(container, text, line) {
  const intervals = [];
  for (const sourceId of state.graph.lineIndex.source.get(line) || []) {
    const source = state.graph.sources.get(sourceId);
    if (!source?.hasExactSpan) continue;

    const start = line === source.line ? columnToStringIndex(text, source.column) : 0;
    const end = line === source.endLine ? columnToStringIndex(text, source.endColumn) : text.length;
    if (end > start) intervals.push({ start, end, sourceId });
  }
  if (intervals.length === 0) return false;

  const boundaries = [...new Set([0, text.length, ...intervals.flatMap(({ start, end }) => [start, end])])]
    .sort((left, right) => left - right);
  for (let index = 0; index + 1 < boundaries.length; index += 1) {
    const start = boundaries[index];
    const end = boundaries[index + 1];
    if (end <= start) continue;
    const sourceIds = new Set(intervals
      .filter((interval) => interval.start <= start && interval.end >= end)
      .map((interval) => interval.sourceId));
    if (sourceIds.size === 0) {
      container.append(document.createTextNode(text.slice(start, end)));
      continue;
    }

    const range = document.createElement("span");
    range.className = "source-range";
    range.sourceIds = sourceIds;
    range.textContent = text.slice(start, end);
    range.title = [...sourceIds]
      .map((id) => state.graph.sources.get(id)?.method)
      .filter(Boolean)
      .join(", ") || "Mapped source expression";
    container.append(range);
  }
  return true;
}

// Zig and the sidecar count UTF-8 bytes; JavaScript string indexes count UTF-16
// code units. Convert explicitly so non-ASCII source still highlights exactly.
function columnToStringIndex(text, oneBasedByteColumn) {
  const targetBytes = Math.max(0, oneBasedByteColumn - 1);
  let bytes = 0;
  let stringIndex = 0;
  for (const codePoint of text) {
    const width = utf8Encoder.encode(codePoint).length;
    if (bytes + width > targetBytes) break;
    bytes += width;
    stringIndex += codePoint.length;
  }
  return Math.min(stringIndex, text.length);
}

function selectLine(paneName, line, shouldAlign) {
  if (!state.graph) return;

  const seedIds = state.graph.lineIndex[paneName].get(line) || new Set();
  selectIds(paneName, line, seedIds, shouldAlign);
}

function selectIds(paneName, line, seedIds, shouldAlign) {
  const selection = expandSelection(paneName, seedIds);
  selection.activePane = paneName;
  selection.activeLine = line;
  selection.activeIds = new Set(seedIds);
  state.selected = selection;

  updateHighlights();
  renderDetails(selection);
  if (shouldAlign) alignRelatedPane(selection, paneName, line);
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

      if (paneName === "source") {
        for (const range of row.querySelectorAll(".source-range")) {
          const rangeRelated = intersects(range.sourceIds, state.selected.source);
          const rangeActive = state.selected.activePane === "source" && intersects(range.sourceIds, state.selected.activeIds);
          range.classList.toggle("is-related", rangeRelated);
          range.classList.toggle("is-active", rangeActive);
        }
      }
    }
  }
}

function intersects(left, right) {
  for (const value of left) {
    if (right.has(value)) return true;
  }
  return false;
}

function alignRelatedPane(selection, originPane, originLine) {
  const targetPane = visiblePaneOrder().find((paneName) => paneName !== originPane);
  const targetLine = targetPane ? firstSelectedLine(targetPane, selection[targetPane]) : Infinity;
  const originRow = lineButton(originPane, originLine);
  const targetRow = targetPane ? lineButton(targetPane, targetLine) : null;
  if (!originRow || !targetRow) return;

  const verticalDelta = targetRow.getBoundingClientRect().top - originRow.getBoundingClientRect().top;
  if (Math.abs(verticalDelta) < 1) return;
  elements.panes[targetPane].scrollTo({
    top: elements.panes[targetPane].scrollTop + verticalDelta,
    behavior: scrollBehavior,
  });
}

function firstSelectedLine(paneName, ids) {
  return Math.min(...selectedLines(paneName, ids));
}

function visiblePaneOrder() {
  return ["source", state.activeIrPane];
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
  else if (event.key === "ArrowLeft") targetPane = visiblePaneOrder()[0];
  else if (event.key === "ArrowRight") targetPane = visiblePaneOrder()[1];
  else if (["1", "2"].includes(event.key)) targetPane = visiblePaneOrder()[Number(event.key) - 1];
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
  }
  elements.counts.source.textContent = "Unavailable";
  elements.irCount.textContent = "Unavailable";
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
    const byteRange = record.startByte !== null && record.endByte !== null
      ? `bytes ${record.startByte}–${record.endByte}`
      : "";
    const provenance = record.provenanceLine !== record.line || record.provenanceColumn !== record.column
      ? `@src ${record.provenanceLine}:${record.provenanceColumn}`
      : "";
    const metadata = [record.method || record.label, byteRange, provenance, id].filter(Boolean).join(" · ");
    return [`${start}${end}`, metadata];
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
    const rawEndLine = firstDefined(record.end_line, record.endLine);
    const rawEndColumn = firstDefined(record.end_column, record.endColumn);
    const endLine = positiveInteger(rawEndLine, line);
    const endColumn = positiveInteger(rawEndColumn, column);
    const startByte = optionalNonNegativeInteger(firstDefined(record.start_byte, record.startByte));
    const endByte = optionalNonNegativeInteger(firstDefined(record.end_byte, record.endByte));
    const hasExactSpan = rawEndLine !== undefined && rawEndColumn !== undefined &&
      (endLine > line || (endLine === line && endColumn > column));
    const id = stringId(firstDefined(record.id, record.source_id, record.sourceId, entryKey, `${file}:${line}:${column}`));
    graph.sources.set(id, {
      id,
      file,
      line,
      column,
      endLine: Math.max(line, endLine),
      endColumn,
      startByte,
      endByte,
      method: optionalString(firstDefined(record.method, record.instrumented_method, record.instrumentedMethod)),
      provenanceLine: positiveInteger(firstDefined(record.provenance_line, record.provenanceLine), line),
      provenanceColumn: positiveInteger(firstDefined(record.provenance_column, record.provenanceColumn), column),
      hasExactSpan,
      label: optionalString(firstDefined(record.label, record.expression, record.method)),
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
        startByte: optionalNonNegativeInteger(firstDefined(record.start_byte, record.startByte)),
        endByte: optionalNonNegativeInteger(firstDefined(record.end_byte, record.endByte)),
        method: optionalString(firstDefined(record.method, record.instrumented_method, record.instrumentedMethod)),
        provenanceLine: positiveInteger(firstDefined(record.provenance_line, record.provenanceLine), line),
        provenanceColumn: positiveInteger(firstDefined(record.provenance_column, record.provenanceColumn), column),
        hasExactSpan: firstDefined(record.end_line, record.endLine) !== undefined &&
          firstDefined(record.end_column, record.endColumn) !== undefined,
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

function optionalNonNegativeInteger(value) {
  const parsed = Number(value);
  return Number.isInteger(parsed) && parsed >= 0 ? parsed : null;
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
