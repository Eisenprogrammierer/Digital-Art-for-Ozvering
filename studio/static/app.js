"use strict";

const $ = (id) => document.getElementById(id);
let catalog = [];
let current = null;
let busy = false;
let lastResult = null;

const el = (tag, cls, text) => {
  const node = document.createElement(tag);
  if (cls) node.className = cls;
  if (text !== undefined) node.textContent = text;
  return node;
};

function formatBytes(n) {
  if (n < 1024) return `${n} B`;
  if (n < 1024 * 1024) return `${(n / 1024).toFixed(1)} KB`;
  return `${(n / 1024 / 1024).toFixed(2)} MB`;
}

function setStatus(text, kind) {
  const node = $("status");
  node.textContent = text || "";
  node.className = kind || "";
}

function fieldValue(field) {
  const node = document.querySelector(`[name="${field.name}"]`);
  if (!node) return field.default;
  if (field.type === "bool") return node.checked;
  return node.value;
}

function addField(field) {
  const wrap = document.createElement("label");
  wrap.className = "field";
  if (field.type === "bool") {
    const line = el("span", "checkline");
    const box = document.createElement("input");
    box.type = "checkbox";
    box.name = field.name;
    box.checked = !!field.default;
    line.appendChild(box);
    line.appendChild(el("span", "name", field.label));
    if (field.help) line.appendChild(el("span", "hint", field.help));
    wrap.appendChild(line);
    $("form").appendChild(wrap);
    return;
  }

  const top = el("span", "top");
  top.appendChild(el("span", "name", field.label));
  if (field.optional) top.appendChild(el("span", "opt", "optional"));
  wrap.appendChild(top);

  let node;
  if (field.type === "choice") {
    node = document.createElement("select");
    node.name = field.name;
    if (field.optional) {
      const any = document.createElement("option");
      any.value = "";
      any.textContent = "— any —";
      node.appendChild(any);
    }
    for (const choice of field.choices || []) {
      if (choice === "") continue;
      const option = document.createElement("option");
      option.value = choice;
      option.textContent = choice;
      node.appendChild(option);
    }
    node.value = String(field.default ?? "");
  } else if (field.type === "text" && field.repeat) {
    node = document.createElement("textarea");
    node.name = field.name;
    node.rows = 2;
    node.value = field.default || "";
  } else if (field.type === "text") {
    node = document.createElement("input");
    node.type = "text";
    node.name = field.name;
    node.value = field.default || "";
    node.placeholder = field.optional ? "leave empty for the default" : "";
  } else {
    node = document.createElement("input");
    node.type = "number";
    node.name = field.name;
    node.value = field.default;
    if (field.min !== undefined) node.min = field.min;
    if (field.max !== undefined) node.max = field.max;
    node.step = field.step !== undefined ? field.step
      : (field.type === "int" ? 1 : "any");
  }
  wrap.appendChild(node);
  if (field.help) wrap.appendChild(el("span", "hint", field.help));
  $("form").appendChild(wrap);
}

function selectGenerator(id) {
  current = catalog.find((g) => g.id === id) || catalog[0];
  if (!current) return;
  $("generator").value = current.id;
  $("blurb").textContent = current.blurb || "";
  $("form").replaceChildren();
  for (const field of current.fields) addField(field);
  $("preview-title").textContent = current.title;
  $("preview-meta").textContent = "ready";
  $("preview").replaceChildren(el("p", "empty",
    "The picture appears here. Set a seed and press Generate."));
  $("result").hidden = true;
  setStatus(current.slow ? "This one can take a few seconds." : "");
}

function buildCatalog() {
  const select = $("generator");
  const groups = new Map();
  for (const gen of catalog) {
    if (!groups.has(gen.group)) {
      const optgroup = document.createElement("optgroup");
      optgroup.label = gen.group;
      groups.set(gen.group, optgroup);
      select.appendChild(optgroup);
    }
    const option = document.createElement("option");
    option.value = gen.id;
    option.textContent = gen.title;
    groups.get(gen.group).appendChild(option);
  }
  select.addEventListener("change", () => selectGenerator(select.value));
  const wanted = new URLSearchParams(location.search).get("make");
  selectGenerator(wanted && catalog.some((g) => g.id === wanted) ? wanted : catalog[0].id);
}

function randomSeed() {
  const node = document.querySelector('[name="seed"]');
  if (node) {
    node.value = String(Math.floor(Math.random() * 1_000_000_000));
    node.focus();
    node.select();
  } else {
    setStatus("this generator has no seed field", "bad");
  }
}

function makePreview(result) {
  const box = $("preview");
  box.replaceChildren();
  const img = document.createElement("img");
  img.alt = result.title;
  img.src = `${result.primary_url}?t=${Date.now()}`;
  box.appendChild(img);
}

function showResult(result) {
  lastResult = result;
  $("preview-title").textContent = result.title;
  $("preview-meta").textContent = `${result.elapsed}s · ${result.files.length} file(s) · ${result.primary}`;
  makePreview(result);

  const list = $("files");
  list.replaceChildren();
  for (const file of result.files) {
    const row = el("div", "file" + (file.name === result.primary ? " primary" : ""));
    const link = document.createElement("a");
    link.href = file.url;
    link.textContent = file.name;
    link.target = "_blank";
    link.rel = "noopener";
    row.appendChild(link);
    row.appendChild(el("span", "bytes", formatBytes(file.bytes)));
    list.appendChild(row);
  }
  const bundle = el("div", "file");
  const zip = document.createElement("a");
  zip.href = result.bundle_url;
  zip.textContent = "bundle.zip";
  zip.target = "_blank";
  zip.rel = "noopener";
  bundle.appendChild(zip);
  bundle.appendChild(el("span", "bytes", "all files"));
  list.appendChild(bundle);

  $("folder-line").replaceChildren(
    document.createTextNode("files are in "),
    el("code", "", result.folder));
  $("command").textContent = result.command;
  $("sidecar").textContent = result.sidecar
    ? JSON.stringify(result.sidecar, null, 2)
    : "no sidecar for this asset";
  $("result").hidden = false;
  setStatus(`${result.elapsed}s — done`, "good");
}

async function generate() {
  if (busy || !current) return;
  busy = true;
  const button = $("go");
  button.disabled = true;
  const label = button.textContent;
  button.replaceChildren(el("span", "spinner"), document.createTextNode(" Working"));
  setStatus(`rendering ${current.title.toLowerCase()}…`);
  const values = {};
  for (const field of current.fields) values[field.name] = fieldValue(field);
  try {
    const response = await fetch("/api/render", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ generator: current.id, values }),
    });
    const data = await response.json();
    if (!response.ok) throw new Error(data.error || `HTTP ${response.status}`);
    showResult(data);
  } catch (err) {
    setStatus(err.message || "the render failed", "bad");
    $("preview-meta").textContent = "failed";
  } finally {
    busy = false;
    button.disabled = false;
    button.textContent = label;
  }
}

async function copyCommand() {
  const text = $("command").textContent;
  if (!text) return;
  try {
    await navigator.clipboard.writeText(text);
    $("copy").textContent = "Copied";
    setTimeout(() => { $("copy").textContent = "Copy"; }, 1200);
  } catch {
    const range = document.createRange();
    range.selectNodeContents($("command"));
    const selection = window.getSelection();
    selection.removeAllRanges();
    selection.addRange(range);
    setStatus("press Ctrl/Cmd-C to copy the selected command", "good");
  }
}

async function boot() {
  try {
    const response = await fetch("/api/generators");
    if (!response.ok) throw new Error(`HTTP ${response.status}`);
    catalog = await response.json();
    if (!catalog.length) throw new Error("the catalog is empty");
    buildCatalog();
  } catch (err) {
    setStatus(`could not load the catalog: ${err.message}`, "bad");
    $("blurb").textContent = "The studio server is not answering.";
    return;
  }
  try {
    const health = await fetch("/api/health").then((r) => r.json());
    if (health.root) $("root-note").textContent = health.root;
  } catch { /* the page still works without the path */ }

  $("go").addEventListener("click", generate);
  $("random").addEventListener("click", randomSeed);
  $("copy").addEventListener("click", copyCommand);
  document.addEventListener("keydown", (event) => {
    if ((event.ctrlKey || event.metaKey) && event.key === "Enter") {
      event.preventDefault();
      generate();
    }
  });
}

boot();
