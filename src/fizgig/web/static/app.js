// Fizgig web UI. Every form on the page is built from /api/schema: each family's description and its form
// (fizgig/families/form.py). Nothing here names a family, a setting or a preset - a new one appears by itself.
"use strict";

const TOKEN = document.querySelector('meta[name="fizgig-token"]').content;
const $ = (id) => document.getElementById(id);
// a colour of the theme (style.css :root), for what CSS can't colour itself: a canvas. themeColor("--accent", 0.5)
// is it half see-through. A hidden probe works it out, so a derived token (a color-mix) comes back as a colour too.
let themeProbe = null;
function themeColor(name, alpha = 1) {
  if (!themeProbe) {
    themeProbe = document.createElement("i");
    themeProbe.style.display = "none";
    document.documentElement.append(themeProbe);
  }
  themeProbe.style.color = alpha >= 1 ? `var(${name})` : `color-mix(in srgb, var(${name}) ${alpha * 100}%, transparent)`;
  const c = getComputedStyle(themeProbe).color;
  const m = /^color\(srgb ([\d.e-]+) ([\d.e-]+) ([\d.e-]+)(?: \/ ([\d.e-]+))?\)$/.exec(c);
  return m ? `rgba(${[m[1], m[2], m[3]].map((v) => Math.round(v * 255)).join(", ")}, ${m[4] ?? 1})` : c;
}
const el = (tag, attrs = {}, ...kids) => {
  const e = document.createElement(tag);
  for (const [k, v] of Object.entries(attrs)) {
    if (k === "class") e.className = v;
    else if (k === "text") e.textContent = v;
    else if (k.startsWith("on")) e.addEventListener(k.slice(2), v);
    else if (v !== undefined && v !== null && v !== false) e.setAttribute(k, v === true ? "" : v);
  }
  for (const c of kids.flat()) if (c != null) e.append(c);
  return e;
};

const S = {
  schema: null, fam: null, family: "", values: {}, all: {}, imageFolder: "", prefs: {},
  state: {hidden: [], disabled: [], notes: {}}, run: {}, logSince: 0, inputs: {},
};

// ---- server ------------------------------------------------------------------------------------------------
async function api(path, body) {
  const opt = {headers: {"X-Fizgig-Token": TOKEN}};
  if (body !== undefined) {
    opt.method = "POST";
    opt.headers["Content-Type"] = "application/json";
    opt.body = JSON.stringify(body);
  }
  const r = await fetch(path, opt);
  const data = await r.json().catch(() => ({}));
  if (r.status === 401 && data.login) { showLogin(); throw new Error("login"); }
  if (!r.ok && !data.problems) throw new Error(data.error || r.statusText);
  return data;
}
const debounce = (fn, ms) => { let t; return (...a) => { clearTimeout(t); t = setTimeout(() => fn(...a), ms); }; };

// A picture from this server that failed to load (a busy moment: a long page asks for dozens at once, or a file still
// arriving) tries again, three times, further apart each time - rather than broken images until a reload
document.addEventListener("error", (e) => {
  const img = e.target;
  if (!(img instanceof HTMLImageElement) || !img.src.includes("/api/")) return;
  const u = new URL(img.src), n = Number(u.searchParams.get("retry") || 0);
  if (n >= 3) return;
  u.searchParams.set("retry", n + 1);
  const failed = img.src;                            // a pane given a new picture meanwhile keeps it
  setTimeout(() => { if (img.src === failed) img.src = u.toString(); }, [1500, 4000, 10000][n]);
}, true);

// ---- start-up ----------------------------------------------------------------------------------------------
async function boot() {
  try { S.schema = await api("/api/schema"); } catch (e) { if (e.message === "login") return; throw e; }
  // Fizgig was updated while this web UI kept running the old server: some new parts won't work until it restarts
  $("outdatedBanner").style.display = S.schema.outdated ? "" : "none";
  const st = await api("/api/state");
  S.uploadRoot = await api("/api/upload/root").catch(() => null);   // before the forms: where Choose… starts
  const pr = await api("/api/prefs");
  S.prefs = pr.models || {};
  S.captioner = pr.captioner;              // the secondary captioning model's row
  S.webPort = pr.web_port;
  S.portNow = pr.port_now;
  S.onPod = pr.on_pod;
  S.hostName = pr.host_name || "";
  if (S.podLocal && S.podLocal.list.length) podLocalDraw();   // drawn before login: now it can say where it sends to
  S.passwordOn = pr.password_on;
  S.portFixed = pr.port_fixed;
  S.pw = {saved: pr.password_saved, env: pr.password_env, lan: pr.lan_ip || ""};
  S.pod = pr.pod;
  S.outputDefault = pr.output_default || "";
  S.outputPref = pr.lora_output_dir || "";
  S.set = pr.settings || {}; S.setAbs = pr.abs || {};
  S.gpu = {chosen: pr.cuda_device || "", list: pr.gpus || [], envLocked: pr.gpu_env_locked};
  S.all = st.values || {};
  S.imageFolder = st.image_folder || "";
  S.capSaved = st.captions || {};
  S.prepSaved = st.prep || {};
  $("imageFolder").value = S.imageFolder;
  await loadTrigger();
  for (const sel of document.querySelectorAll(".family-pick")) {     // Start's, Samples' and Training's: one choice
    sel.innerHTML = "";
    for (const f of S.schema.families) sel.append(el("option", {value: f.key, text: f.gui_label || f.display_name}));
  }
  await chooseFamily(st.family && S.schema.order.includes(st.family) ? st.family : S.schema.order[0]);
  showTab("start");                // Fizgig always opens on Start (its News card), never the last tab used
  // the forms are there: the page is usable. The cover stays at least 3.5 s from the page opening, so its tip can be
  // read - once a window: switching it between this computer and a pod (window.name stays across the switch, whatever
  // the address) shows the cover only while the page loads
  const switched = window.name === "fizgig-window";
  window.name = "fizgig-window";
  setTimeout(() => $("bootCover").classList.add("done"), switched ? 0 : Math.max(0, 3500 - performance.now()));
  newsCard();
  wire();
  $("podsTabBtn").style.display = S.onPod ? "none" : "";      // on a pod there's nothing to connect to
  $("sideDeploy").style.display = S.onPod ? "none" : "";      // ...nor a pod to deploy
  $("sideDeploy").addEventListener("click", (e) => {          // to Rent a pod: its disk steps come before Deploy
    e.preventDefault();
    showTab("pods");
    setTimeout(() => { const c = document.querySelector(".pods-rent"); if (c) c.scrollIntoView({behavior: "smooth"}); }, 50);
  });
  fileManagerLinks(pr.file_manager);
  phoneCard();
  podLocalInit();                                             // in the app window on a pod: this computer's folders
  welcomeShow(pr);
  pollRun(); pollStats(); followLog();
  setInterval(pollRun, 2000); setInterval(pollStats, 2000); setInterval(loadGallery, 15000);
  setInterval(loadCheckpoints, 15000);
}

async function chooseFamily(key) {
  try {                                     // the latest saved, in case another open page changed it
    const st = await api("/api/state");
    if (st && st.values && st.values[key]) S.all[key] = st.values[key];
  } catch (e) { /* keep what this page has */ }
  S.family = key;
  S.fam = S.schema.families.find((f) => f.key === key);
  for (const sel of document.querySelectorAll(".family-pick")) sel.value = key;
  if (S.all[key]) {
    S.values = {...defaults(), ...S.all[key]};
  } else {                              // a first visit takes the family's first preset, as the desktop does
    S.values = defaults();
    const first = S.fam.form.presets[0];
    if (first) {
      S.values = (await api("/api/preset", {family: key, name: first.name, values: S.values})).values;
      S.values.__preset = first.name;
    }
  }
  if (S.trigger !== undefined) S.values.CAPTION_TRIGGER_WORD = S.trigger;     // the folder's, not the model's
  // on whenever a form loads: a change made here while a run trains goes to its next previews (only a change is
  // sent, so a run started from the queue keeps its own prompts until you change something)
  S.values.__SAMPLE_LIVE = true;
  renderAll();
  await refreshState();
  saveState();
}

// the trigger word is the training folder's (both apps keep one per folder): the Captions tab's box and the
// Training tab's auto-recaption box show it; typing in either keeps it for this folder
async function loadTrigger() {
  const r = await api("/api/trigger?folder=" + encodeURIComponent(S.imageFolder || "")).catch(() => ({word: ""}));
  showTrigger(r.word || "");
}
function showTrigger(word) {
  S.trigger = word.replace(/,+$/, "").trim();
  $("capTrigger").value = word;
  if (S.values) { S.values.CAPTION_TRIGGER_WORD = S.trigger; syncInputs(["CAPTION_TRIGGER_WORD"]); }
}
const saveTriggerSoon = debounce((word) => api("/api/trigger", {folder: S.imageFolder || "", word}).catch(() => {}), 500);

function defaults() {
  const v = {};
  for (const s of S.fam.form.sections) for (const r of s.rows) if (r.kind !== "note" && r.kind !== "shown") {
    v[r.key] = r.default;
    for (const p of r.parts || []) v[p.key] = p.default;   // a control of several values (the preview cadence)
  }
  return v;
}

function renderAll() {
  $("familyNotes").textContent = S.fam.model_note || "";
  for (const st of document.querySelectorAll(".family-status")) {    // the model's status line (description status_note)
    st.textContent = S.fam.status_note || "";
    st.style.display = S.fam.status_note ? "" : "none";
  }
  renderSections("samples", $("samplesForm"));
  renderSections("training", $("trainingForm"));
  renderPresets();
  renderPrefs();
  watchFetch();
  checkModels();
  loadGallery();
  loadCheckpoints();
}

// ---- the generated forms -----------------------------------------------------------------------------------
// ---- Check a trigger word: under the Subjects card, collapsed. The training model's own text encoder reads the word
// the way Fizgig LoRA Isolation joins it ("zwxem, a woman" -> zwxemwoman); values kept across the card's re-draws
S.tc = {open: false, word: "", cls: "woman", others: ""};

function triggerCheckBlock() {
  const name = (S.fam && (S.fam.gui_label || S.fam.display_name)) || "this model";
  const word = el("input", {type: "text", class: "w-combo", placeholder: "e.g. zwxem", value: S.tc.word});
  const cls = el("select", {class: "w-combo"});
  for (const [v, t] of [["woman", "a woman"], ["man", "a man"], ["girl", "a girl"], ["boy", "a boy"],
                        ["person", "a person"], ["dog", "a dog"], ["cat", "a cat"], ["horse", "a horse"],
                        ["", "a style or look"]])
    cls.append(el("option", {value: v, text: t}));
  cls.value = S.tc.cls;
  const others = el("input", {type: "text", class: "w-combo-lg", placeholder: "e.g. dlor, gwelt", value: S.tc.others});
  const result = el("div");
  const run = async () => {
    S.tc.word = word.value.trim(); S.tc.cls = cls.value; S.tc.others = others.value;
    result.innerHTML = "";
    if (!S.tc.word) { result.append(el("div", {class: "field-hint", text: "Type a trigger word first."})); return; }
    result.append(el("div", {class: "field-hint", text: "Checking…"}));
    let r;
    try {
      r = await api("/api/captions/trigger_check", {family: S.family, trigger: S.tc.word, cls: cls.value,
        others: others.value.split(",").map((s) => s.trim()).filter(Boolean)});
    } catch (e) { result.innerHTML = ""; result.append(el("div", {class: "field-hint", text: String(e.message || e)})); return; }
    result.innerHTML = "";
    const clean = r.ok && !r.notes.length;
    result.append(el("div", {class: "tc-verdict " + (clean ? "tc-good" : r.ok ? "tc-warn" : "tc-bad"),
      text: (clean ? "✓ Good word. " : r.ok ? "Joins cleanly, with a note below. " : "✗ ") +
            (clean || r.ok ? `${r.joined} reads as ${r.pieces.join(" | ")}` : r.notes[0])}));
    for (const n of (r.ok ? r.notes : r.notes.slice(1))) result.append(el("div", {class: "field-hint", text: "• " + n}));
    if (r.encoder) result.append(el("div", {class: "field-hint", text: `Checked with ${r.encoder}.`}));
    if (r.suggestions && r.suggestions.length) {
      const row = el("div", {class: "field-row"},
        el("span", {class: "field-hint", text: "Made-up words that work here (click one to check it):"}));
      for (const s of r.suggestions)
        row.append(el("button", {class: "btn btn-sm", text: s, onclick: () => { word.value = s; run(); }}));
      result.append(row);
    }
  };
  word.addEventListener("keydown", (e) => { if (e.key === "Enter") run(); });
  const d = el("details", {class: "tc-box"},
    el("summary", {text: "Check a trigger word (optional)"}),
    el("div", {class: "field-hint", text:
      `Checks how a trigger word reads to ${name}'s text encoder, the one this LoRA trains with, before you ` +
      `caption. Fizgig LoRA Isolation joins it to what the subject is ("zwxem, a woman" trains as zwxemwoman), and ` +
      `some words join badly: fyxo + woman reads as f | yx | ow | oman, and the LoRA learns that subject less cleanly.`}),
    el("div", {class: "field-row"}, el("span", {class: "field-label", text: "Trigger word to check:"}), word),
    el("div", {class: "field-row"}, el("span", {class: "field-label", text: "The subject is:"}), cls),
    el("div", {class: "field-hint", text: "The word Fizgig joins on: the same trigger can join cleanly with one " +
      "(dlor + woman) and break with another (dlor + man). A style or look is never joined, so only the word itself " +
      "is checked."}),
    el("div", {class: "field-row"}, el("span", {class: "field-label", text: "Other triggers:"}), others),
    el("div", {class: "field-hint", text: "Optional: the trigger words of your other subjects in this same LoRA, separated by commas. Two " +
      "subjects whose words start with the same letter, or share a piece, can blend into each other, so the check " +
      "looks for that too. Leave it empty for a one-subject LoRA."}),
    el("div", {class: "field-row"}, el("button", {class: "btn btn-primary", text: "Check", onclick: run})),
    result);
  d.open = S.tc.open;
  d.addEventListener("toggle", () => { S.tc.open = d.open; });
  return d;
}

function renderSections(tab, host) {
  host.innerHTML = "";
  if (!S.secShut) S.secShut = {};
  for (const sec of S.fam.form.sections.filter((s) => s.tab === tab)) {
    // open or closed as the form says, or as you left it this session (a family switch redraws the cards)
    const key = tab + ":" + sec.id, shut = key in S.secShut ? S.secShut[key] : !!sec.collapsed;
    const body = el("div", {class: "card-content" + (shut ? " collapsed" : "")});
    const label = el("span");
    const head = el("div", {class: "card-title collapsible-header"}, label);
    const show = () => {
      const c = body.classList.contains("collapsed");
      label.textContent = (c ? "▶ " : "▼ ") + sec.title;
      card.classList.toggle("sec-open", !c);
    };
    head.onclick = () => { body.classList.toggle("collapsed"); S.secShut[key] = body.classList.contains("collapsed"); show(); };
    if (sec.help) {         // the card's video, a link across the top of the card
      body.append(el("a", {class: "sec-video", href: sec.help.href, target: "_blank", rel: "noopener"},
        el("span", {class: "sec-video-play", text: "▶"}),
        el("span", {class: "sec-video-text"}, el("b", {text: sec.help.text}),
          el("span", {class: "sec-video-sub", text: sec.help.sub || "Watch on YouTube"}))));
    }
    // a group's rows (form.py ROW_GROUPS) in one box where its first row is, side by side; their hints behind ?
    const groups = {};
    for (const r of sec.rows) {
      const row = renderRow(r);
      if (!r || !r.group) { body.append(row); continue; }
      let g = groups[r.group];
      if (!g) {
        const inner = el("div", {class: "form-group-rows"}), help = el("div", {class: "form-group-help"});
        const q = el("button", {class: "form-group-q", text: "?", title: "What these do",
          onclick: () => help.classList.toggle("open")});
        const box = el("div", {class: "form-group", "data-group": r.group},
          el("div", {class: "form-group-head"}, el("span", {text: r.group_label}), q), inner, help);
        g = groups[r.group] = {box, inner, help, q};
        body.append(box);
      }
      g.inner.append(row);
      if (r.hint) g.help.append(el("div", {}, el("b", {text: r.label.replace(/:$/, "") + ": "}), r.hint));
    }
    for (const g of Object.values(groups)) if (!g.help.children.length) g.q.remove();
    if (sec.id === "subjects" && !S.family.startsWith("acestep")) body.append(triggerCheckBlock());
    const card = el("div", {class: "section-card", id: "sec-" + sec.id, "data-section": sec.id}, head, body);
    show();
    if (sec.extras) host.append(el("div", {class: "extras-divider", "data-extras": "1"}, el("span", {text: "✦ Extras ✦"})));
    if (sec.extras) card.classList.add("sec-extra");
    else if (host.querySelector(".extras-divider")) card.classList.add("sec-extra");
    host.append(card);
  }
}

function renderRow(r) {
  // its kind sets its layout (style.css): boxes in the grid, switches as tiles, a choice of pills, files and notes
  // across the card - from the schema's row kind, so every model's form lays out the same way
  const kind = {check: "check", radio: "radio", note: "note", text: "wide", path: "wide", folder: "wide", shown: "wide",
    cadence: "cadence", split: "cadence"}[r.kind] || "field";
  const wrap = el("div", {class: "form-row kind-" + kind + (r.featured ? " row-featured" : "") +
    (r.boxed ? " row-boxed" : ""), "data-key": r.key});
  if (r.kind === "shown") {                       // a value from elsewhere (the Start tab's folder), read-only here
    const shownRow = el("div", {class: "field-row"}, el("span", {class: "field-label", text: r.label + ":"}),
      el("input", {type: "text", class: "w-path", id: "note_" + r.key, readonly: true, tabindex: -1}));
    if (r.sets_training_folder) {                 // the training folder itself (an edit's Edited folder, #217): choosing
      shownRow.append(el("button", {class: "btn", text: "Choose…",          // it here sets the Start tab's
        title: "Pick from what is on the machine running Fizgig - it becomes the training folder on the Start tab",
        onclick: () => pick(S.imageFolder, true, (p) => {
          $("imageFolder").value = p; $("imageFolder").dispatchEvent(new Event("change")); })}));
    }
    wrap.append(shownRow);
    if (r.hint) wrap.append(el("div", {class: "field-hint", text: r.hint}));
    return wrap;
  }
  if (r.kind === "note") {                        // its own text, and any the form's state writes (RefMod's tokens)
    if (r.label) wrap.append(el("div", {class: r.hint_labels ? "field-label" : "field-hint", text: r.label}));
    wrap.append(el("div", {class: "field-hint row-note" + (r.hint_labels ? " hint-lines note-lines" : ""),
      id: "note_" + r.key}));
    return wrap;
  }
  const v = S.values[r.key];
  let input;
  const set = (val) => onValue(r, val);
  if (r.kind === "blocks") return blockPicker(r, wrap);
  if (r.lines) return promptLines(r, wrap);
  if (r.kind === "cadence") return cadenceRow(r, wrap);
  if (r.kind === "split") return splitRow(r, wrap);
  if (r.kind === "check") {
    input = el("input", {type: "checkbox", id: "f_" + r.key, onchange: (e) => set(e.target.checked)});
    input.checked = !!v;
    wrap.append(el("label", {class: "checkbox-label"}, input, " " + r.label));
  } else if (r.kind === "radio") {
    const box = el("div", {class: "radio-row"});
    for (const c of r.choices) {
      const rb = el("input", {type: "radio", name: "f_" + r.key, value: c, onchange: () => set(c)});
      rb.checked = String(v) === c;
      box.append(el("label", {class: "checkbox-label"}, rb, " " + c));
    }
    wrap.append(el("span", {class: "field-label", text: r.label + ":"}), box);
    input = box;
  } else {
    if (r.kind === "choice") {
      input = el("select", {id: "f_" + r.key, onchange: (e) => set(e.target.value)});
      // a saved value in another case is the same choice ("auto" saved before the box read "Auto")
      const same = v != null && r.choices.find((c) => c.toLowerCase() === String(v).toLowerCase());
      const val = same || v;
      if (same && same !== v) S.values[r.key] = same;
      const opts = r.choices.includes(String(val)) || !val ? r.choices : [String(val), ...r.choices];
      for (const c of opts) input.append(el("option", {value: c, text: c}));
      input.value = String(val ?? "");
    } else if (r.kind === "text") {
      input = el("textarea", {id: "f_" + r.key, rows: 4, spellcheck: "false", oninput: (e) => set(e.target.value)});
      input.value = v ?? "";
    } else if (r.kind === "combo") {
      // a dropdown of the usual values, as the desktop's, with Other… for a value of your own (a browser's
      // suggestion list only shows what matches what's typed, so with 1.0 in the box it showed nothing)
      const other = el("input", {type: "text", class: "combo-other", placeholder: "Your value",
        oninput: (e) => set(e.target.value)});
      input = el("select", {id: "f_" + r.key, onchange: (e) => {
        const custom = e.target.value === "__other";
        other.style.display = custom ? "" : "none";
        if (custom) { other.value = ""; other.focus(); } else set(e.target.value);
      }});
      fillCombo(input, r.choices, v, r.labels);
      other.style.display = "none";
      input._other = other;
    } else {
      input = el("input", {type: "text", id: "f_" + r.key, oninput: (e) => set(e.target.value)});
      input.value = v ?? "";
    }
    const row = el("div", {class: "field-row"}, el("span", {class: "field-label", text: r.label + ":"}), input);
    if (input._other) row.append(input._other);
    let upStatus = null;
    if (r.kind === "path" || r.kind === "folder") {
      const startAt = r.start === "lora" ? (S.set.input_lora_dir || S.values.LORA_OUTPUT_DIR || "") :
        r.start === "output" ? (S.values.LORA_OUTPUT_DIR || "") :
        r.start === "data" ? ((S.uploadRoot && S.uploadRoot.root) || S.imageFolder) : S.imageFolder;
      row.append(el("button", {class: "btn", text: "Choose…", title: "Pick from what is on the machine running Fizgig", onclick: () =>
        pick(input.value || startAt, r.kind === "folder", (p) => { input.value = p; set(p); })}));
      if (r.upload) {
        const up = uploadButton(r, input, set);
        row.append(up.btn);
        upStatus = up.status;
      }
    }
    if (r.key === "LORA_OUTPUT_DIR") {
      // empty = the default from Preferences; a folder picked here is this model's own (kept per model)
      input.placeholder = "Default: " + (S.outputDefault || "output_loras inside Fizgig");
      row.append(el("button", {class: "btn", text: "Reset to default",
        title: "Back to the LoRA output folder in Preferences", onclick: () => { input.value = ""; set(""); }}));
    }
    if (r.suffix) row.append(el("span", {class: "field-hint", text: r.suffix}));
    let actMsg = null;
    if (r.action === "write_captions") {          // the typed line as every photo's caption (Edit, Slider pairs)
      actMsg = el("div", {class: "field-hint"});
      const write = async (overwrite) => {
        try {
          const res = await api("/api/captions/write_all", {text: input.value, folder: S.imageFolder,
            ext: S.values.DATASET_CAPTION_EXT || ".txt", overwrite});
          if (res.exists) {
            actMsg.replaceChildren(document.createTextNode(`${res.exists} of the ${res.total} photos already have a ` +
              "caption. "), el("button", {class: "btn btn-sm btn-danger", text: "Replace them", onclick: () => write(true)}));
            return;
          }
          actMsg.textContent = `Captions written for ${res.written} photos.`;
        } catch (e) { actMsg.textContent = e.message; }
      };
      row.append(el("button", {class: "btn", text: "Write captions", onclick: () => write(false)}));
    }
    wrap.append(row);
    if (actMsg) wrap.append(actMsg);
    if (upStatus) wrap.append(upStatus);
  }
  wrap.append(el("div", {class: "field-hint row-note", id: "note_" + r.key}));
  const link = r.link ? el("a", {class: "inline-link", target: "_blank", rel: "noopener", href: r.link.href,
    text: r.link.text + " →"}) : null;      // a row's guide on GitHub, under its hint
  if (r.hint && r.hint_labels) {                  // one spaced line per choice, "Name: what it does" with the name bold
    const h = el("div", {class: "field-hint hint-lines"});
    for (const line of r.hint.split("\n")) {
      const at = line.indexOf(": ");
      h.append(at > 0 ? el("div", {}, el("b", {text: line.slice(0, at) + ": "}), line.slice(at + 2))
                      : el("div", {text: line}));
    }
    wrap.append(h);
  } else if (r.hint) {                            // a hint of several lines keeps its line breaks
    const h = el("div", {class: "field-hint row-hint", text: r.hint, title: "Click to read it all",
      onclick: () => h.classList.toggle("open")});
    if (r.hint.includes("\n")) h.style.whiteSpace = "pre-line";
    wrap.append(h);
  }
  if (r.bold) wrap.append(el("div", {class: "field-hint"}, el("b", {text: r.bold})));   // a line the row stresses
  if (link) wrap.append(el("div", {class: "field-hint"}, link));
  if (r.warn) wrap.append(el("div", {class: "field-hint tc-warn"}, r.warn, r.warn_link ? el("a", {class: "inline-link",
    target: "_blank", rel: "noopener", href: r.warn_link, text: " Best values: the Reddit discussion →"}) : ""));
  return wrap;
}

// How often to preview: one sentence of boxes - "Every [N] epochs", then (ticked) "then from epoch [10], every
// [1] epochs" - each box its own value, greyed one by one (refreshState reads data-field / data-part). Ticking it
// fills in the row's tick_fills (every 5) while the first box is still at its default, and unticking puts that back
function cadenceRow(r, wrap) {
  const [later, until, after] = r.parts.map((p) => p.key);
  const num = (key) => {
    const i = el("input", {type: "text", inputmode: "numeric", class: "cad-num", id: "f_" + key, "data-field": key,
      oninput: (e) => onValue({key}, e.target.value)});
    i.value = S.values[key] ?? "";
    return i;
  };
  const tick = el("input", {type: "checkbox", id: "f_" + later, "data-field": later, onchange: (e) => {
    S._cadFilled = S._cadFilled || {};            // what the tick filled in, so unticking undoes only that
    for (const [k, v] of Object.entries(r.tick_fills || {})) {
      if (e.target.checked && String(S.values[k] ?? "") === String(r.default)) {
        S.values[k] = v; S._cadFilled[k] = v; syncInputs([k]);
      } else if (!e.target.checked && S._cadFilled[k] !== undefined && String(S.values[k]) === String(v)) {
        S.values[k] = r.default; syncInputs([k]);
      }
      if (!e.target.checked) delete S._cadFilled[k];
    }
    onValue({key: later}, e.target.checked);
  }});
  tick.checked = !!S.values[later];
  wrap.append(el("div", {class: "cad-title", text: r.label}),
    el("div", {class: "cad-line"}, el("span", {text: "Every"}), num(r.key), el("span", {text: "epochs"})),
    el("div", {class: "cad-line", "data-part": later},
      el("label", {class: "cad-tick"}, tick, el("span", {text: "then from epoch"})),
      el("span", {class: "cad-pair"}, num(until)),
      el("span", {class: "cad-pair"}, el("span", {text: "every"}), num(after), el("span", {text: "epochs"}))),
    el("div", {class: "field-hint row-note", id: "note_" + r.key}),
    el("div", {class: "field-hint", "data-part": later, text: r.hint}));
  return wrap;
}

// An entry with a Split mode (MOSS-Image's Clean-end share): Single is one value, Split a value for the first
// epochs and another for the rest, Slide (when the row has one) a value held to an epoch, then easing down to another
// by a later epoch. It stays one text - "30", "75:5:40" or "75:6:40:10" - so presets, Last Train and the queue carry
// it as any entry; each mode remembers its own numbers while you switch between them
function splitRow(r, wrap) {
  // a row without Split shows a split value (75:5:40) as a Slide that drops at once (75:5:40:5)
  const modeOf = () => { const n = String(S.values[r.key] ?? "").split(":").length;
    return (n === 4 || n === 5) && r.slide ? "Slide" : n === 3 ? (r.split ? "Split" : "Slide") : "Single"; };
  const partsOf = () => { const p = String(S.values[r.key] ?? "").split(":");
    return p.length === 3 && !r.split ? [...p, p[1]] : p; };
  S._split = S._split || {};
  const m0 = modeOf();
  const mem = S._split[r.key] = S._split[r.key] || {
    single: m0 === "Single" ? String(S.values[r.key] ?? "") : String(r.single ?? (String(r.default ?? "").includes(":") ? "" : r.default ?? "")),
    split: m0 === "Split" ? partsOf() : (r.split || []).slice(), slide: m0 === "Slide" ? partsOf() : (r.slide || []).slice()};
  if (mem.slide.length === 5) { mem.every = mem.slide[4]; mem.slide = mem.slide.slice(0, 4); }   // the tick's number apart
  const unit = (r.suffix || "").split(" ")[0];
  const write = () => {
    const m = modeOf();
    if (m === "Split") { mem.split = [first, until, rest].map((i) => i.value.trim()); onValue(r, mem.split.join(":")); }
    else if (m === "Slide") {          // the refresh tick adds its "every N" as a fifth number
      mem.slide = [sFirst, sHold, sRest, sBy].map((i) => i.value.trim());
      mem.every = sEvery.value.trim() || mem.every || "3";
      refreshEcho();
      sEvery.disabled = !sTick.checked;
      onValue(r, [...mem.slide, ...(sTick.checked ? [mem.every] : [])].join(":"));
    }
    else { mem.single = one.value.trim(); onValue(r, mem.single); }
  };
  const box = () => el("input", {type: "text", inputmode: "numeric", class: "cad-num", oninput: write});
  const one = box(), first = box(), until = box(), rest = box();
  const sFirst = box(), sHold = box(), sRest = box(), sBy = box(), sEvery = box();
  const sTick = el("input", {type: "checkbox", onchange: write});
  const sFirstEcho = el("span");                   // "epoch at 30%": the lower of the two shares, as typed
  const sNth = el("span");                         // "every 2nd epoch": the box's ordinal ending
  const refreshEcho = () => {
    const n = parseInt(sEvery.value, 10);
    sNth.textContent = isNaN(n) ? "th" : n % 100 >= 11 && n % 100 <= 13 ? "th" : ["th", "st", "nd", "rd"][n % 10] || "th";
    const a = parseFloat(sFirst.value), b = parseFloat(sRest.value);
    sFirstEcho.textContent = `${isNaN(a) || isNaN(b) ? sRest.value : a <= b ? sFirst.value : sRest.value}${unit}`;
  };
  const pills = el("div", {class: "radio-row"});
  for (const c of ["Single", ...(r.split ? ["Split"] : []), ...(r.slide ? ["Slide"] : [])]) {
    pills.append(el("label", {class: "checkbox-label"}, el("input", {type: "radio", name: "split_" + r.key, value: c,
      onchange: () => { onValue(r, c === "Split" ? mem.split.join(":") : c === "Slide" ? [...mem.slide,
          ...(mem.every && mem.slideTick !== false ? [mem.every] : [])].join(":") : mem.single);
        show(); }}), " " + c));
  }
  const lineOne = el("div", {class: "cad-line"}, one, el("span", {text: r.suffix || ""}));
  const lineSplit = el("div", {class: "cad-line"},
    el("span", {class: "cad-pair"}, first, el("span", {text: unit + " for the first"})),
    el("span", {class: "cad-pair"}, until, el("span", {text: "epochs, then"})),
    el("span", {class: "cad-pair"}, rest, el("span", {text: unit})));
  const lineSlide = el("div", {class: "cad-line"},
    el("span", {class: "cad-pair"}, sFirst, el("span", {text: unit + " until epoch"})),
    el("span", {class: "cad-pair"}, sHold, el("span", {text: "then slides to"})),
    el("span", {class: "cad-pair"}, sRest, el("span", {text: unit + " by epoch"})),
    el("span", {class: "cad-pair"}, sBy));
  const lineRefresh = (r.slide || []).length > 4 ? el("div", {class: "cad-line"},
    el("label", {class: "cad-tick"}, sTick, el("span", {text: "then every"})),
    el("span", {class: "cad-pair"}, sEvery, el("span", {}, sNth, " epoch at ", sFirstEcho))) : null;
  const hintOne = el("div", {class: "field-hint", text: r.hint});
  const hintSplit = el("div", {class: "field-hint", text: r.split_hint || r.hint});
  const hintSlide = el("div", {class: "field-hint", text: r.slide_hint || r.hint});
  function show() {
    const m = modeOf();
    pills.querySelectorAll("input").forEach((rb) => { rb.checked = rb.value === m; });
    if (m === "Split") { mem.split = partsOf(); [first.value, until.value, rest.value] = mem.split; }
    else if (m === "Slide") {
      const p = partsOf();
      mem.slide = p.slice(0, 4); [sFirst.value, sHold.value, sRest.value, sBy.value] = mem.slide;
      sTick.checked = p.length === 5;
      if (p.length === 5) mem.every = p[4];
      sEvery.value = mem.every || (r.slide || [])[4] || "3";
      sEvery.disabled = !sTick.checked;
      refreshEcho();
    }
    else { one.value = S.values[r.key] ?? ""; mem.single = one.value; }
    for (const [node, on] of [[lineOne, m === "Single"], [hintOne, m === "Single"], [lineSplit, m === "Split"],
                              [hintSplit, m === "Split"], [lineSlide, m === "Slide"], [lineRefresh, m === "Slide"],
                              [hintSlide, m === "Slide"]]) {
      if (node) node.style.display = on ? "" : "none";
    }
  }
  const sync = el("input", {type: "hidden", id: "f_" + r.key});    // syncInputs (a preset, Last Train) redraws it
  sync._sync = show;
  sTick.addEventListener("change", () => { mem.slideTick = sTick.checked; });
  wrap.append(el("div", {class: "cad-title", text: r.label}), pills, lineOne, lineSplit, lineSlide, ...(lineRefresh ? [lineRefresh] : []),
    hintOne, hintSplit,
    hintSlide, sync,
    el("div", {class: "field-hint row-note", id: "note_" + r.key}));
  if (r.warn) wrap.append(el("div", {class: "field-hint tc-warn"}, r.warn, r.warn_link ? el("a", {class: "inline-link",
    target: "_blank", rel: "noopener", href: r.warn_link, text: " Best values: the Reddit discussion →"}) : ""));
  show();
  return wrap;
}

// a list of prompts, one numbered box each (the gallery's Prompt 1, 2, 3…), saved as the one-per-line text it always was
function promptLines(r, wrap) {
  const list = el("div", {class: "prompt-lines"});
  const boxes = () => [...list.querySelectorAll("textarea")];
  const save = () => onValue(r, boxes().map((b) => b.value).join("\n"));
  const grow = (b) => {                     // to its text; a hidden tab measures 0, so it waits to be seen
    if (!b.offsetParent) return;
    b.style.height = "auto"; b.style.height = b.scrollHeight + 2 + "px";
  };
  new ResizeObserver(() => boxes().forEach(grow)).observe(list);     // shown at last, or a wider window
  const renumber = () => list.querySelectorAll(".prompt-num").forEach((n, i) => { n.textContent = i + 1; });
  function line(text, after) {
    const box = el("textarea", {rows: 1, spellcheck: "false", "aria-label": "Prompt"});
    box.value = text;
    const item = el("div", {class: "prompt-line"}, el("span", {class: "prompt-num"}), box,
      el("button", {class: "prompt-del", text: "×", title: "Remove this prompt", tabindex: -1, onclick: () => {
        if (boxes().length > 1) item.remove(); else box.value = "";
        renumber(); save();
      }}));
    box.addEventListener("input", () => { grow(box); save(); });
    box.addEventListener("keydown", (e) => {
      const at = box.selectionStart, flat = at === box.selectionEnd;
      if (e.key === "Enter" && !e.isComposing) {           // the rest of the line starts the next prompt
        e.preventDefault();
        const rest = box.value.slice(box.selectionEnd);
        box.value = box.value.slice(0, at); grow(box);
        const next = line(rest, item);
        next.setSelectionRange(0, 0); next.focus();
        renumber(); save();
      } else if (e.key === "Backspace" && flat && at === 0 && item.previousElementSibling) {   // join the one above
        e.preventDefault();
        const prev = item.previousElementSibling.querySelector("textarea"), end = prev.value.length;
        prev.value += box.value; grow(prev);
        item.remove(); prev.focus(); prev.setSelectionRange(end, end);
        renumber(); save();
      }
    });
    box.addEventListener("paste", (e) => {                 // pasted lines become prompts of their own
      const text = (e.clipboardData || window.clipboardData).getData("text");
      if (!/\r?\n/.test(text)) return;
      e.preventDefault();
      const parts = text.split(/\r?\n/);
      const head = box.value.slice(0, box.selectionStart), tail = box.value.slice(box.selectionEnd);
      box.value = head + parts[0]; grow(box);
      let last = item, lastBox = box;
      for (const p of parts.slice(1)) { lastBox = line(p, last); last = lastBox.parentNode; }
      const end = lastBox.value.length;
      lastBox.value += tail; grow(lastBox);
      lastBox.focus(); lastBox.setSelectionRange(end, end);
      renumber(); save();
    });
    if (after) after.after(item); else list.append(item);
    requestAnimationFrame(() => grow(box));
    return box;
  }
  function draw() {
    list.replaceChildren();
    const v = String(S.values[r.key] ?? "").split(/\r?\n/);
    for (const t of v) line(t);
    renumber();
  }
  const sync = el("input", {type: "hidden", id: "f_" + r.key});    // syncInputs (a preset, Last Train) redraws it
  sync._sync = draw;
  const add = el("button", {class: "btn btn-sm", text: "+ Add prompt", onclick: () => {
    const last = list.lastElementChild;
    line("", last).focus(); renumber(); save();
  }});
  wrap.append(el("span", {class: "field-label", text: r.label + ":"}), list, add, sync);
  if (r.key === "SAMPLE_PROMPT") wrap.append(promptSetsBar(r, draw));
  if (r.hint) wrap.append(el("div", {class: "field-hint", text: r.hint}));
  wrap.append(el("div", {class: "field-hint row-note", id: "note_" + r.key}));
  draw();
  return wrap;
}

// the Samples tab's prompt sets (#267): saved prompts, shared by every model, loaded in place of the ones here (a
// change goes to the run going now as any edit does, while Change the previews of the run going now is ticked)
function promptSetsBar(r, draw) {
  const pick = el("select", {"aria-label": "Prompt set"});
  const ask = el("div"), msg = el("span", {class: "field-hint"});
  const load = el("button", {class: "btn btn-sm", text: "Load", onclick: async () => {
    try {
      const d = await api("/api/prompt_set", {name: pick.value});
      onValue(r, d.prompts); draw();
      msg.textContent = `Loaded ${pick.value}: ${d.prompts.split("\n").length} prompt(s).`;
    } catch (e) { msg.textContent = e.message; }
  }});
  const del = el("button", {class: "btn btn-sm", text: "Delete", onclick: () => {
    const name = pick.value;
    ask.replaceChildren(el("div", {class: "field-row"}, el("span", {class: "field-hint", text: `Delete the set ${name}?`}),
      el("button", {class: "btn btn-sm btn-danger", text: "Delete", onclick: async () => {
        try { fill((await api("/api/prompt_set/delete", {name})).sets); msg.textContent = `Deleted ${name}.`; }
        catch (e) { msg.textContent = e.message; }
        ask.replaceChildren();
      }}),
      el("button", {class: "btn btn-sm", text: "Cancel", onclick: () => ask.replaceChildren()})));
  }});
  const fill = (sets, chosen) => {
    pick.replaceChildren(el("option", {value: "", text: sets.length ? "Choose a set…" : "No saved sets yet"}),
      ...sets.map((n) => el("option", {value: n, text: n})));
    pick.value = sets.includes(chosen) ? chosen : "";
    load.disabled = del.disabled = !pick.value;
  };
  pick.addEventListener("change", () => { load.disabled = del.disabled = !pick.value; ask.replaceChildren(); });
  const saveAs = el("button", {class: "btn btn-sm", text: "Save as set…", onclick: () => {
    if (ask.childNodes.length) { ask.replaceChildren(); return; }
    const name = el("input", {type: "text", class: "w-path", placeholder: "A name for these prompts", value: pick.value});
    const note = el("span", {class: "field-hint"});
    const save = async (overwrite) => {
      const n = name.value.trim();
      if (!n) { note.textContent = "Type a name first."; return; }
      try {
        const d = await api("/api/prompt_set/save", {name: n, prompts: S.values[r.key] || "", overwrite: !!overwrite});
        if (d.exists) {
          note.replaceChildren(document.createTextNode(`There's a set called ${n} already. `),
            el("button", {class: "btn btn-sm btn-danger", text: "Replace it", onclick: () => save(true)}));
          return;
        }
        fill(d.sets, n); ask.replaceChildren();
        msg.textContent = `Saved ${n}: every model's Samples tab can load it.`;
      } catch (e) { note.textContent = e.message; }
    };
    name.addEventListener("keydown", (e) => { if (e.key === "Enter") save(false); });
    ask.replaceChildren(el("div", {class: "field-row"}, name,
      el("button", {class: "btn btn-sm btn-primary", text: "Save", onclick: () => save(false)}),
      el("button", {class: "btn btn-sm", text: "Cancel", onclick: () => ask.replaceChildren()}), note));
    name.focus();
  }});
  api("/api/prompt_sets").then((d) => fill(d.sets)).catch(() => fill([]));
  return el("div", {}, el("div", {class: "field-row"}, el("span", {class: "field-label", text: "Prompt sets:"}), pick,
    load, saveAs, del, msg), ask);
}

// Model Area to Train > Custom: the model's blocks as ticks (All / None / an area's blocks), as the desktop's
function blockPicker(r, wrap) {
  const sel = () => ({...(S.values[r.key] || {})});
  const ticks = {};
  const put = (m) => { onValue(r, m); for (const [id, t] of Object.entries(ticks)) t.checked = !!m[id]; count(); };
  const n = el("span", {class: "field-hint"});
  const count = () => { const m = S.values[r.key] || {}; n.textContent = `${Object.values(m).filter(Boolean).length} chosen`; };
  const all = (on) => { const m = {}; for (const g of r.groups) for (const b of g.blocks) m[b.id] = on; put(m); };
  wrap.append(el("div", {class: "field-row block-head"}, el("b", {text: "Select blocks to train:"}),
    el("button", {class: "btn btn-sm", text: "All", onclick: () => all(true)}),
    el("button", {class: "btn btn-sm", text: "None", onclick: () => all(false)}),
    ...Object.entries(r.areas || {}).map(([name, ids]) => el("button", {class: "btn btn-sm", text: name, onclick: () => {
      const m = {}; for (const g of r.groups) for (const b of g.blocks) m[b.id] = ids.includes(b.id); put(m); }})), n));
  for (const g of r.groups) {
    const row = el("div", {class: "block-row"}, el("span", {class: "block-group", text: g.label + ":"}));
    for (const b of g.blocks) {
      const t = ticks[b.id] = el("input", {type: "checkbox", onchange: (e) => { const m = sel(); m[b.id] = e.target.checked; onValue(r, m); count(); }});
      t.checked = !!(S.values[r.key] || {})[b.id];
      row.append(el("label", {class: "block-tick"}, t, el("span", {text: b.label})));
    }
    wrap.append(row);
  }
  count();
  return wrap;
}

function onValue(r, val) {
  S.values[r.key] = val;
  if (r.key === "__SAMPLE_LIVE" || (S.values.__SAMPLE_LIVE &&
      ["SAMPLE_PROMPT", "SAMPLE_SEED", "SAMPLE_WIDTH", "SAMPLE_HEIGHT", "SAMPLE_STEPS", "SAMPLE_CFG_SCALE",
       "FAMILY_TURBO_STRENGTH", "SAMPLE_EVERY_N_EPOCHS", "SAMPLE_LATER", "SAMPLE_UNTIL_EPOCH",
       "SAMPLE_EVERY_AFTER", "FAMILY_PREVIEW_LOOK", "SAMPLE_CHECKPOINT"].includes(r.key))) sampleLiveSoon();
  if (r.key === "CAPTION_TRIGGER_WORD") { S.trigger = String(val).trim(); $("capTrigger").value = S.trigger; saveTriggerSoon(S.trigger); }
  const extra = r.sets && r.sets[val];            // a kind of training that also sets others (Fine-tune's LR)
  if (extra) { Object.assign(S.values, extra); syncInputs(Object.keys(extra)); }
  if (r.timesteps && r.timesteps[val]) {           // a Model Area with its own timestep window
    [S.values.MIN_TIMESTEP, S.values.MAX_TIMESTEP] = r.timesteps[val];
    syncInputs(["MIN_TIMESTEP", "MAX_TIMESTEP"]);
  }
  refreshStateSoon();
  saveStateSoon();
}

// a combo row's dropdown: the usual values, a value of your own kept as an option, and Other…
function fillCombo(sel, choices, v, labels) {
  sel.innerHTML = "";
  const val = v == null ? "" : String(v);
  const opts = val && !choices.includes(val) ? [...choices, val] : choices;
  opts.forEach((c, i) => sel.append(el("option", {value: c, text: (labels && labels[i]) || c})));
  sel.append(el("option", {value: "__other", text: "Other…"}));
  sel.value = val || (choices[0] ?? "");
}

function syncInputs(keys) {
  for (const k of keys) {
    const i = $("f_" + k);
    if (!i) {                                     // a choice of pills: its radio inputs share the name
      document.querySelectorAll(`input[type="radio"][name="f_${k}"]`).forEach((rb) => {
        rb.checked = rb.value === String(S.values[k] ?? ""); });
      continue;
    }
    if (i.tagName === "SELECT" && [...i.options].some((o) => o.value === "__other")) {
      const row = (S.fam.form.sections.flatMap((s) => s.rows).find((r) => r && r.key === k)) || {choices: []};
      fillCombo(i, row.choices, S.values[k], row.labels);
      continue;
    }
    if (i._sync) { i._sync(); continue; }         // a control that draws itself from the value (splitRow)
    if (i.type === "checkbox") i.checked = !!S.values[k]; else i.value = S.values[k] ?? "";
  }
}

async function refreshState() {
  S.state = await api("/api/form/state", {family: S.family, values: S.values, image_folder: S.imageFolder});
  if (S.state.poll) setTimeout(refreshStateSoon, 2000);     // a download under way: its note moves on
  const force = Object.keys(S.state.force || {});      // what a setting turns off (automagic3: Adaptive LR...)
  if (force.length) { Object.assign(S.values, S.state.force); syncInputs(force); saveStateSoon(); }
  const hidden = new Set(S.state.hidden), disabled = new Set(S.state.disabled);
  for (const [k, ph] of Object.entries(S.state.placeholders || {})) {   // what Fizgig found, greyed in the box
    const box = $("f_" + k);
    if (box) box.placeholder = ph;
  }
  document.querySelectorAll(".form-row[data-key]").forEach((row) => {
    const k = row.dataset.key;
    row.style.display = hidden.has(k) ? "none" : "";
    const grp = row.closest(".form-group");         // a box shows while any of its rows does
    if (grp) grp.style.display = [...grp.querySelectorAll(".form-row[data-key]")]
      .some((x) => !hidden.has(x.dataset.key)) ? "" : "none";
    // a label the state renumbers (the Edit pairs steps when an extra-pictures folder joins), else the form's own
    const lab = row.querySelector(".field-label"), rowDef = (S.fam.form.sections.flatMap((s) => s.rows)
      .find((r) => r && r.key === k));
    if (lab && rowDef) lab.textContent = ((S.state.labels || {})[k] || rowDef.label) + ":";
    // a control of several values greys and hides each of its parts by its own key
    row.querySelectorAll("input,select,textarea,button").forEach((i) => { i.disabled = disabled.has(i.dataset.field || k); });
    row.querySelectorAll("[data-part]").forEach((p) => { p.style.display = hidden.has(p.dataset.part) ? "none" : ""; });
    // single choices greyed in a row of pills (Fizgig LoRA Isolation with one subject)
    const off = ((S.state.disabled_choices || {})[k]) || {};       // {choice: why}, the why on hover
    row.querySelectorAll(`input[type="radio"][name="f_${k}"]`).forEach((rb) => {
      if (rb.value in off) rb.disabled = true;
      const lab = rb.closest("label");
      lab.classList.toggle("choice-off", rb.disabled);
      lab.title = off[rb.value] || "";
    });
    // a greyed box shows what that run will really use (the Distilled's steps / CFG), the form's own value kept
    const shown = (S.state.shown || {})[k], box = $("f_" + k);
    if (box && shown !== undefined && disabled.has(k)) { box.value = shown; box.dataset.shown = "1"; }
    else if (box && box.dataset.shown) { delete box.dataset.shown; syncInputs([k]); }
    const n = $("note_" + k);
    if (n && k === "__SAMPLE_LIVE") {          // its note follows the run (liveNote), not the form's state
      S._liveNoteFor = undefined;
      liveNote(!!(S.run && (S.run.running || S.run.state === "starting")));
      return;
    }
    if (n) {
      if (n.tagName === "INPUT") { n.value = S.state.notes[k] || ""; return; }
      if (n.classList.contains("note-lines")) {    // one line each, "Label: text" with the label bold (Subjects card)
        n.replaceChildren(...String(S.state.notes[k] || "").split("\n").filter(Boolean).map((line) => {
          const at = line.indexOf(": ");
          return at > 0 ? el("div", {}, el("b", {text: line.slice(0, at) + ": "}), line.slice(at + 2))
                        : el("div", {text: line});
        }));
      } else n.textContent = S.state.notes[k] || "";
      const act = (S.state.actions || {})[k];      // a hint's one-click change (the desktop's: Use 1e-4)
      if (act) n.append(" ", el("button", {class: "btn btn-sm", text: act.label, onclick: () => {
        Object.assign(S.values, act.values); syncInputs(Object.keys(act.values)); refreshStateSoon(); saveStateSoon(); }}));
    }
  });
  document.querySelectorAll(".section-card[data-section]").forEach((c) => {
    c.style.display = hidden.has(c.dataset.section) ? "none" : "";
  });
  document.querySelectorAll(".extras-divider").forEach((d) => {    // no divider over nothing
    const extras = [...d.parentNode.querySelectorAll(".section-card.sec-extra")];
    d.style.display = extras.some((c) => c.style.display !== "none") ? "" : "none";
  });
  turboFollow();
  if (S.state.pending) setTimeout(refreshStateSoon, 2500);   // a note still being worked out (the fine-tune plan)
}

// Steps and CFG follow the Turbo LoRA, as the desktop's Samples tab: its own (Qwen: 8 steps, CFG 1) while it renders -
// file set in Preferences AND strength above 0 - else the plain model's (Qwen: 25, CFG 3). Only an empty box or one on
// another of these defaults switches, so a value you typed is never replaced.
// A preview-checkpoint tick (Klein's / Flux 3's distilled) does the same with its row's follow: ticked with the file
// set, Steps shows the distilled's own count (Flux 3: 16), unticked the Base's (50).
// A model with both (Qwen Image 2.1, MOSS-Image) gets one answer for the boxes they share: a ticked preview checkpoint's
// own values, else a live Turbo's, else the plain model's. Each used to switch the boxes on its own, so an unticked
// checkpoint put the plain model's 25 steps, CFG 3 back under a live Turbo LoRA (noisy previews). A box still on any
// of the defaults is put right when the form loads, which also mends a form saved in that state; after that the boxes
// move only when a switch changes, so steps typed by hand stay (they used to snap back to the mode's default).
function turboFollow() {
  const rows = [];
  for (const s of S.fam.form.sections) for (const row of s.rows) if (row && (row.turbo || row.follow)) rows.push(row);
  if (!rows.length) return;
  const want = {}, known = {}, flags = {};
  const note = (side) => { for (const [k, v] of Object.entries(side)) (known[k] = known[k] || []).push(parseFloat(v)); };
  for (const row of rows) {
    const sides = row.turbo || row.follow;
    note(sides.on); note(sides.off);
    Object.assign(want, sides.off);
  }
  // the later pass wins the boxes they share: normally the preview checkpoint's own values, but a checkpoint that
  // takes the Turbo LoRA (Kromoss) renders a live Turbo's steps and CFG
  const order = S.state && S.state.checkpoint_turbo ? ["follow", "turbo"] : ["turbo", "follow"];
  for (const pass of order) for (const row of rows) {
    if (!row[pass]) continue;
    let live;
    if (pass === "turbo") {
      const raw = S.values.FAMILY_TURBO_STRENGTH;
      const strength = parseFloat(raw === undefined || raw === "" ? row.default : raw);
      live = !!S.state.turbo_file && strength > 0; flags.__turbo_live = live;
    } else {
      const v = S.values[row.key];
      live = !!S.state.checkpoint_file && !!(v === undefined ? row.default : v); flags["__follow_" + row.key] = live;
    }
    if (live) Object.assign(want, row[pass].on);
  }
  const moved = Object.keys(want).filter((k) => {
    const v = S.values[k];
    if (v === undefined || v === "") return true;
    return parseFloat(v) !== parseFloat(want[k]) && known[k].includes(parseFloat(v));   // "3.0" is the default 3
  });
  const flagsChanged = Object.entries(flags).some(([f, live]) => S.values[f] !== live);
  const loading = S._turboFollowFor !== S.fam;       // the first pass for this model's form
  S._turboFollowFor = S.fam;
  if (!flagsChanged && !loading) return;
  if (!moved.length && !flagsChanged) return;
  for (const k of moved) S.values[k] = want[k];
  Object.assign(S.values, flags);
  syncInputs(moved);
  saveStateSoon();
  if (S.values.__SAMPLE_LIVE && moved.length) sampleLiveSoon();
  // what hangs on the moved values (the negative box shows only above CFG 1) is worked out again from them; the next
  // pass finds nothing to move, so this settles after one round
  if (moved.length) refreshStateSoon();
}
const refreshStateSoon = debounce(refreshState, 250);

async function renderPresets() {
  const sel = $("presets");
  const fam = S.family;
  const list = await api("/api/presets?family=" + encodeURIComponent(fam)).catch(() =>
    ({builtin: S.fam.form.presets.map((p) => p.name), user: []}));
  if (fam !== S.family) return;
  sel.innerHTML = "";
  // the preset last picked shows (a first visit applies the first, as the desktop); "Choose a preset…" before any
  sel.append(el("option", {value: "", text: "Choose a preset…"}));
  for (const n of list.builtin) sel.append(el("option", {value: n, text: n}));
  if (list.user.length) sel.append(el("optgroup", {label: "Your presets"}, ...list.user.map((n) => el("option", {value: n, text: n}))));
  const names = [...list.builtin, ...list.user];
  sel.value = names.includes(S.values.__preset) ? S.values.__preset : "";
  // the note a model gives one of its presets (MiniMax H3's rank 16: for larger datasets), beside the box
  const note = () => { $("presetNote").textContent = (Object.fromEntries(S.fam.preset_notes || []))[sel.value] || ""; };
  note();
  sel.onchange = async () => {
    const name = sel.value;
    note();
    if (!name) return;
    S.values = (await api("/api/preset", {family: S.family, name, values: S.values})).values;
    S.values.__preset = name;
    renderSections("samples", $("samplesForm"));
    renderSections("training", $("trainingForm"));
    await refreshState();
    saveState();
  };
}

// ---- Preferences: every family's model files, from its description -----------------------------------------
function renderPrefs() {
  const host = $("prefsForm");
  host.innerHTML = "";
  if (S.pod) host.append(podCard());
  host.append(webPortCard());
  if (!S.pod) host.append(networkCard());
  host.append(dataFolderCard());
  host.append(outputFolderCard());
  host.append(foldersCard());
  host.append(inferenceCard());
  host.append(captionerCard());
  const gpuCard = gpuPickCard();
  if (gpuCard) host.append(gpuCard);
  S.fetchRows = {};
  for (const f of S.schema.families) {
    if (f.prefs_skip) continue;              // RefMod: MiniMax H3's files, on H3's card
    // a model's card stays as you left it, open or shut, when the tab redraws (a download finishing); they all start
    // shut - you open the one you want
    if (!S.prefsOpen) S.prefsOpen = new Set();
    const body = el("div", {class: "card-content" + (S.prefsOpen.has(f.key) ? "" : " collapsed")});
    if (f.prefs_intro) body.append(el("div", {class: "card-desc prefs-intro", text: f.prefs_intro}));
    const files = new Map(f.model_files.map((m) => [m.pref_key, m]));
    for (const key of f.model_keys) {
      const m = files.get(key) || {label: key, note: "", hint: ""};
      const input = el("input", {type: "text", class: "w-path", value: S.prefs[key] || ""});
      input.addEventListener("change", async () => {
        S.prefs[key] = input.value.trim();
        await api("/api/prefs", {changes: {[key]: S.prefs[key]}});
        checkModels();
        refreshStateSoon();              // a preview file set or cleared: Samples follows it
        captionModelsChanged();          // the Krea 2 text encoder: Captions' Qwen3-VL
      });
      const row = el("div", {class: "field-row"},
        el("span", {class: "field-label", text: (m.label || key) + (m.required ? " *" : "") + ":"}), input,
        el("button", {class: "btn", text: "Choose…", title: "Pick from what is on the machine running Fizgig", onclick: () =>
          pick(input.value, false, (p) => { input.value = p; input.dispatchEvent(new Event("change")); })}));
      if (m.repo && m.path) {
        row.append(el("a", {class: "btn", target: "_blank", rel: "noopener", text: m.download_label || "Download",
          href: `https://huggingface.co/${m.repo}/blob/main/${m.path}`}));
      }
      if (m.alt_repo && m.alt_path) {          // another build of the same file (the desktop's second link)
        row.append(el("a", {class: "btn", target: "_blank", rel: "noopener", text: m.alt_label || "Download",
          href: `https://huggingface.co/${m.alt_repo}/blob/main/${m.alt_path}`}));
      }
      const hint = m.hint || m.note;
      const dl = [m.repo && m.path ? (m.download_note || `~${m.size_gb || "?"} GB - ${m.repo} → ${m.path.split("/").pop()}`) : "",
        m.alt_repo && m.alt_path ? m.alt_note : ""].filter(Boolean).join(" · ");
      body.append(el("div", {class: "pref-file"}, row, hint ? el("div", {class: "field-hint", text: hint}) : null,
        dl ? el("div", {class: "field-hint", text: dl}) : null));
    }
    if (f.prefs_note) body.append(el("div", {class: "field-hint prefs-note", text: f.prefs_note}));
    const fetchBox = fetchRow(f);
    if (fetchBox) body.append(fetchBox);
    host.append(el("div", {class: "section-card"},
      el("div", {class: "card-title collapsible-header", onclick: () => {
        body.classList.toggle("collapsed");
        S.prefsOpen[body.classList.contains("collapsed") ? "delete" : "add"](f.key);
      }},
        el("span", {text: f.prefs_title || ("Model Paths (" + f.display_name + ")")})), body));
  }
  host.append(el("div", {class: "section-card"}, el("div", {class: "card-title"}, el("span", {text: "Reset"})),
    el("div", {class: "card-content"}, el("div", {class: "field-row"}, el("button", {class: "btn", text: "Reset to defaults",
      onclick: async (e) => {
        if (e.target.dataset.sure !== "1") { e.target.dataset.sure = "1"; e.target.textContent = "Reset? Click again"; return; }
        await api("/api/prefs", {reset: true});
        const pr = await api("/api/prefs");
        S.set = pr.settings || {}; S.setAbs = pr.abs || {}; S.outputPref = pr.lora_output_dir || ""; S.outputDefault = pr.output_default;
        renderPrefs();
      }})),
      el("div", {class: "field-hint", text: "Puts the folders and inference settings above back to Fizgig's defaults. " +
        "Model files, the web password and the RunPod key are kept."}))));
  paintFetch();
}

// the secondary captioning model (#270): set, it captions everything (the Captions tab and auto-recaption);
// empty, the Krea 2 text encoder does. Plain Hugging Face links to the builds Fizgig reads - nothing downloads itself
function captionerCard() {
  const c = S.captioner || {path: "", note: "", links: []};
  const input = el("input", {type: "text", class: "w-path", value: c.path || "", placeholder: "Empty = the Krea 2 text encoder"});
  const note = el("div", {class: "field-hint", text: c.note || ""});
  const save = async () => {
    try {
      await api("/api/prefs", {changes: {captioner_secondary: input.value.trim()}});
      S.captioner = (await api("/api/prefs")).captioner || c;
      note.textContent = S.captioner.note || "";
      captionModelsChanged();
    } catch (e) { note.textContent = e.message; }
  };
  input.addEventListener("change", save);
  const row = el("div", {class: "field-row"}, el("span", {class: "field-label", text: "Secondary captioning model:"}),
    input, el("button", {class: "btn", text: "Choose…", title: "Pick from what is on the machine running Fizgig",
      onclick: () => pick(input.value, false, (p) => { input.value = p; save(); })}),
    ...(c.links || []).map((l) => el("a", {class: "btn", target: "_blank", rel: "noopener", href: l.href,
      text: l.label.split(" (")[0], title: l.label})));
  return el("div", {class: "section-card"}, el("div", {class: "card-title"}, el("span", {text: "Captioning model"})),
    el("div", {class: "card-content"}, el("div", {class: "pref-file"}, row,
      el("div", {class: "field-hint", text: "Optional. Qwen3-VL 8B, the bigger model of the family the Krea 2 text " +
        "encoder belongs to (it's Qwen Image 2.1's text encoder): more detailed captions. Download a build from the " +
        "links and choose it here; from then on it does all captioning - the Captions tab and auto-recaption. Empty " +
        "the box to go back to the Krea 2 text encoder. " + (c.links || []).map((l) => l.label).join(" · ") +
        ". On a card with less VRAM than it needs, it keeps what fits and streams the rest from system RAM (slower)."}),
      note)));
}

// a folder setting the two apps share (prefs.json): a box, Choose…, saved when it changes
function settingRow(key, label, hint, placeholder) {
  const input = el("input", {type: "text", class: "w-path", value: S.set[key] || "", placeholder: placeholder || ""});
  const save = async () => {
    try { await api("/api/prefs", {changes: {[key]: input.value.trim()}}); S.set[key] = input.value.trim(); }
    catch (e) { note.textContent = e.message; }
  };
  const note = el("div", {class: "field-hint", text: hint});
  input.addEventListener("change", save);
  return el("div", {class: "pref-file"}, el("div", {class: "field-row"},
    el("span", {class: "field-label", text: label}), input,
    el("button", {class: "btn", text: "Choose…", title: "Pick from what is on the machine running Fizgig", onclick: () =>
      pick(input.value || S.setAbs[key] || "", true, (p) => { input.value = p; save(); })})), note);
}

function foldersCard() {
  return el("div", {class: "section-card"}, el("div", {class: "card-title"}, el("span", {text: "Other folders"})),
    el("div", {class: "card-content"},
      settingRow("profiles_dir", "Profiles:", "Where the Profiler's reports go (Studio finds a LoRA's profile there).", "profiles inside Fizgig"),
      settingRow("cache_dir", "Cache:", "Cached latents and text encodings: each training folder gets its own folder in here.", "cache inside Fizgig"),
      settingRow("input_lora_dir", "LoRA loading:", "Where the pickers for LoRAs start: Studio, Royale, Profiler, Extract, Context LoRA."),
      settingRow("input_ref_dir", "Reference images:", "Where the pickers for reference pictures start: Studio and Royale.")));
}

// the workbench tools' memory and speed (Studio, Royale, Profiler, Extract): the desktop's Inference Performance card
const INFERENCE_SWAP = ["Auto (detect from GPU)", "0  (Distilled fp8: 16GB / Base: 24GB)", "4  (Distilled fp8: 14GB / Base: 20GB)",
  "8  (Distilled fp8: 11GB / Base: 16GB)", "12 (Distilled fp8: 10GB / Base: 14GB)", "16 (Distilled fp8: 8GB / Base: 12GB)"];
function inferenceCard() {
  const cur = String(S.set.inference_blocks_to_swap || INFERENCE_SWAP[0]).trim();
  const lead = (cur.match(/^\d+/) || [""])[0];
  const swap = el("select", {class: "w-combo"});
  for (const o of INFERENCE_SWAP) swap.append(el("option", {value: o, text: o}));
  swap.value = INFERENCE_SWAP.find((o) => lead && o.startsWith(lead + " ")) || INFERENCE_SWAP[0];
  swap.addEventListener("change", () => api("/api/prefs", {changes: {inference_blocks_to_swap: swap.value}})
    .then(() => { S.set.inference_blocks_to_swap = swap.value; }));
  const int8 = el("input", {type: "checkbox"});
  int8.checked = String(S.set.inference_int8 ?? "1") !== "0";
  int8.addEventListener("change", () => api("/api/prefs", {changes: {inference_int8: int8.checked}})
    .then(() => { S.set.inference_int8 = int8.checked ? "1" : "0"; }));
  return el("div", {class: "section-card"}, el("div", {class: "card-title"}, el("span", {text: "Inference performance"})),
    el("div", {class: "card-desc", text: "For Studio, Royale, the Profiler and Extract. Block swap moves blocks to the CPU " +
      "between steps to save memory, at some speed. Training manages its own and ignores these."}),
    el("div", {class: "card-content"},
      el("div", {class: "field-row"}, el("span", {class: "field-label", text: "DiT block swap:"}), swap),
      el("label", {class: "checkbox-label"}, int8, " INT8 fast inference (recommended)"),
      el("div", {class: "field-hint", text: "On by default: same memory as fp8, near-identical pictures, previews only. " +
        "Turn it off to use fp8."})));
}

// where the LoRA and reference pickers start (Preferences > Other folders, else what they used before)
const loraStart = () => S.set.input_lora_dir || S.outputDefault;
const refStart = () => S.set.input_ref_dir || S.imageFolder;

// "Download models for me": the family's files + the helper models, paths filled in when they land
function fetchRow(f) {
  const files = f.model_files.filter((m) => m.repo && m.path);
  if (!files.length) return null;
  const total = files.reduce((a, m) => a + (m.size_gb || 0), 0);
  const gated = files.filter((m) => m.gated);
  const box = el("div", {class: "fetch-row"});
  const opt = f.fetch_optional_label ? el("input", {type: "checkbox"}) : null;
  const token = gated.length ? el("input", {type: "password", class: "w-path", placeholder: "hf_…"}) : null;
  // a file offered with another build (MOSS-Image Base or Kromoss): a pick, starting on the one its path holds now
  const picks = files.filter((m) => m.fetch_choice && m.fetch_choice.length === 2 && m.alt_repo && m.alt_path).map((m) => {
    const sel = el("select", {}, el("option", {value: "main", text: m.fetch_choice[0]}),
      el("option", {value: "alt", text: m.fetch_choice[1]}));
    const cur = String(S.prefs[m.pref_key] || "").split(/[\\/]/).pop();
    sel.value = cur && cur === m.alt_path.split("/").pop() ? "alt" : "main";
    return {key: m.pref_key, sel, label: m.label};
  });
  const btn = el("button", {class: "btn btn-accent", text: "⬇ Download models for me", onclick: async () => {
    try {
      await api("/api/fetch", {family: f.key, token: token ? token.value.trim() : "", optional: opt ? opt.checked : false,
        choices: Object.fromEntries(picks.map((p) => [p.key, p.sel.value]))});
      if (token) token.value = "";
      S.fetchError = null;
      watchFetch();
    } catch (e) { S.fetchError = {family: f.key, text: e.message}; paintFetch(); }
  }});
  const line = el("div", {class: "field-row"}, btn);
  for (const p of picks) line.append(el("label", {class: "fetch-opt"}, p.label + ": ", p.sel));
  if (opt) line.append(el("label", {class: "fetch-opt"}, opt, " " + f.fetch_optional_label));
  box.append(line);
  if (token) {
    const seen = new Set();
    const links = gated.filter((m) => !seen.has(m.repo) && seen.add(m.repo)).map((m) =>
      el("a", {target: "_blank", rel: "noopener", href: `https://huggingface.co/${m.repo}`, text: m.label + " licence"}));
    box.append(el("div", {class: "fetch-token"},
      el("div", {class: "field-hint", text: "These files are gated: sign in to Hugging Face, accept the licence on " +
        "each page below, then paste a read token. It is used for this download only, never saved. " +
        "Leave it empty when the machine or pod has HF_TOKEN set."}),
      el("div", {class: "fetch-links"}, links,
        el("a", {target: "_blank", rel: "noopener", href: "https://huggingface.co/settings/tokens", text: "Create a token"})),
      el("div", {class: "field-row"}, el("span", {class: "field-label", text: "Token:"}), token)));
  }
  box.append(el("div", {class: "field-hint", text: f.fetch_note ||
    `Fetches every file above (~${Math.round(total)} GB) and fills in these paths, plus the Qwen3-VL captioner ` +
    "the Captions tab uses and the small face and captioning models. Interrupted downloads resume where they left off."}));
  const status = el("div", {class: "fetch-status"});
  box.append(status);
  S.fetchRows[f.key] = {status, btn};
  return box;
}

function paintFetch() {
  const st = S.fetch || {};
  const busy = st.state === "running";
  const gb = (n) => (n / 1024 ** 3).toFixed(1);
  for (const [key, r] of Object.entries(S.fetchRows || {})) {
    r.btn.disabled = busy;
    r.status.innerHTML = "";
    if (S.fetchError && S.fetchError.family === key) {
      r.status.append(el("div", {class: "fetch-msg bad", text: S.fetchError.text}));
    }
    if (st.family !== key || !st.state || st.state === "idle") continue;
    if (busy) {
      const pct = st.total ? Math.min(100, (100 * st.done) / st.total) : 0;
      r.status.append(
        el("div", {class: "fetch-msg", text: st.file ? `${st.file}: ${gb(st.done)} of ${gb(st.total)} GB` :
          (st.lines[st.lines.length - 1] || "Starting…")}),
        el("div", {class: "fetch-bar"}, el("div", {style: `width:${pct.toFixed(1)}%`})),
        el("button", {class: "btn", text: "Cancel", onclick: () => api("/api/fetch/cancel", {}).then(watchFetch)}));
    } else {
      const text = {done: "Done: the paths above are filled in.", cancelled: "Cancelled. Downloading again resumes it.",
        failed: "Some files did not download; press the button again to retry just those."}[st.state] || "";
      r.status.append(el("div", {class: "fetch-msg" + (st.state === "done" ? "" : " bad"), text}));
      if (st.state === "failed") r.status.append(el("pre", {class: "fetch-log", text: st.lines.slice(-12).join("\n")}));
    }
  }
}

async function watchFetch() {
  clearTimeout(S.fetchTimer);
  const was = S.fetch && S.fetch.state;
  try { S.fetch = await api("/api/fetch"); } catch (e) { return; }
  if (was === "running" && S.fetch.state !== "running") {
    S.prefs = (await api("/api/prefs")).models || {};    // the fetcher wrote the new paths
    if (S.prefsOpen && S.fetch.family) S.prefsOpen.add(S.fetch.family);   // its card open, showing how it ended
    renderPrefs();
    checkModels();
    refreshState();                    // a Turbo LoRA that just landed: Samples' steps and CFG follow it now
    captionModelsChanged();            // a text encoder that just landed: Captions offers Qwen3-VL now
  }
  paintFetch();
  if (S.fetch.state === "running") S.fetchTimer = setTimeout(watchFetch, 1500);
}

// which GPU runs use (Preferences cuda_device, the desktop's GPU setting): only shown with more than one card
function gpuPickCard() {
  const g = S.gpu || {list: []};
  if (g.list.length < 2) return null;
  const sel = el("select", {class: "w-combo-lg"},
    el("option", {value: "", text: "System default (GPU 0)"}),
    g.list.map((c) => el("option", {value: c.uuid, text: `${c.index}: ${c.name} (${Math.round(c.total_gb)} GB)`})));
  const known = g.list.find((c) => c.uuid === g.chosen || String(c.index) === g.chosen);
  sel.value = known ? known.uuid : "";
  sel.disabled = !!g.envLocked;
  const hint = el("div", {class: "field-hint", text: g.envLocked ?
    "CUDA_VISIBLE_DEVICES is set where Fizgig was started, and it wins over this setting." :
    "The card every run trains on, and the one the memory meter shows. A run already going keeps its card."});
  sel.addEventListener("change", async () => {
    try { await api("/api/prefs", {changes: {cuda_device: sel.value}}); g.chosen = sel.value; }
    catch (e) { hint.textContent = e.message; }
  });
  return el("div", {class: "section-card"},
    el("div", {class: "card-title"}, el("span", {text: "GPU"})),
    el("div", {class: "card-content"},
      el("div", {class: "field-row"}, el("span", {class: "field-label", text: "Train on:"}), sel), hint));
}

// where uploads go: Preferences' training data folder (the desktop's Input Directories > Training images), else
// training_data inside Fizgig
function showUploadRoot() {
  // on the machine running Fizgig, Choose… picks the folder itself: uploading would only copy it
  $("uploadCard").style.display = pageIsElsewhere() ? "" : "none";
  api("/api/upload/root").then((d) => {
    S.uploadRoot = d;
    if (!S.onPod) $("imageFolder").placeholder = d.root + (d.root.includes("\\") ? "\\" : "/") + "my_subject";
    const box = $("dataFolder");
    if (box && document.activeElement !== box) box.value = d.set ? d.root : "";
    $("uploadWhere").textContent = `It goes into ${d.root}, in a folder with the name above. ` +
      (d.set ? "That folder is set in Preferences (Training data folder). " :
        "That is Fizgig's own training_data folder; choose another in Preferences (Training data folder). ") +
      "Uploading again with the same name adds to that folder.";
  }).catch(() => {});
}

// the default LoRA output folder: where a model's LoRAs go until you pick its own on the Training tab
function outputFolderCard() {
  const input = el("input", {type: "text", class: "w-path", value: S.outputPref || "",
    placeholder: "output_loras inside Fizgig"});
  const hint = el("div", {class: "field-hint", text: "Where a model's LoRAs and checkpoints go until you pick a " +
    "folder for it on the Training tab (that one is kept for that model; Reset to default comes back here). Leave " +
    "it empty for output_loras inside Fizgig."});
  const save = async () => {
    try {
      await api("/api/prefs", {changes: {lora_output_dir: input.value.trim()}});
      const pr = await api("/api/prefs");
      S.outputDefault = pr.output_default; S.outputPref = pr.lora_output_dir;
      const f = $("f_LORA_OUTPUT_DIR");
      if (f) f.placeholder = "Default: " + S.outputDefault;
      hint.dataset.saved = "1";
    } catch (e) { hint.textContent = e.message; }
  };
  input.addEventListener("change", save);
  return el("div", {class: "section-card"},
    el("div", {class: "card-title"}, el("span", {text: "LoRA output folder"})),
    el("div", {class: "card-content"},
      el("div", {class: "field-row"}, el("span", {class: "field-label", text: "Folder:"}), input,
        el("button", {class: "btn", text: "Choose…", title: "Pick from what is on the machine running Fizgig", onclick: () =>
          pick(input.value || S.outputDefault, true, (p) => { input.value = p; save(); })})), hint));
}

function dataFolderCard() {
  const input = el("input", {type: "text", class: "w-path", id: "dataFolder", value: (S.uploadRoot && S.uploadRoot.set) ?
    S.uploadRoot.root : "", placeholder: "Default: " + ((S.uploadRoot && S.uploadRoot.root) || "Fizgig's own training_data folder")});
  const save = async () => {
    try { await api("/api/prefs", {changes: {input_dataset_dir: input.value.trim()}}); showUploadRoot(); loadDatasets(); }
    catch (e) { hint.textContent = e.message; }
  };
  input.addEventListener("change", save);
  const hint = el("div", {class: "field-hint", text: "Where uploaded training data goes, each upload in its own folder " +
    "inside it, and where the folder pickers start. Leave it empty for Fizgig's own training_data folder."});
  return el("div", {class: "section-card"},
    el("div", {class: "card-title"}, el("span", {text: "Training data folder"})),
    el("div", {class: "card-content"},
      el("div", {class: "field-row"}, el("span", {class: "field-label", text: "Folder:"}), input,
        el("button", {class: "btn", text: "Choose…", title: "Pick from what is on the machine running Fizgig", onclick: () =>
          pick(input.value, true, (p) => { input.value = p; save(); })})), hint));
}

// on a rented pod: stop it once a run finishes (fizgig/web/pod_stop.py), and the account key that can
function podCard() {
  const P = S.pod;
  const on = el("input", {type: "checkbox"});
  on.checked = P.stop_when_done;
  on.addEventListener("change", () => api("/api/prefs", {changes: {runpod_stop_when_done: on.checked}}).then(() => { P.stop_when_done = on.checked; }));
  const key = el("input", {type: "password", class: "w-path", autocomplete: "off",
    placeholder: P.key_saved ? "saved - type a new one to replace it" : "rpa_…"});
  const keySay = el("span", {class: "field-hint"});
  const sayKey = () => { keySay.textContent = P.key_saved ? "✓ key saved" : P.key_env ? "✓ using RUNPOD_STOP_API_KEY from the template" :
    "no key yet - the pod can't stop itself without one"; };
  const saveKey = async (v) => { await api("/api/prefs", {changes: {runpod_api_key: v}}); P.key_saved = !!v; key.value = ""; sayKey(); };
  key.addEventListener("change", () => key.value.trim() && saveKey(key.value.trim()));
  sayKey();
  const vast = P.provider === "vast";
  const midRun = el("div", {class: "field-hint warn", text: "Stopping the " + (vast ? "instance" : "pod") +
    " yourself while a run trains loses what it trained since its last save: press Pause first (it saves and stops " +
    "at the end of the epoch), then stop it."});
  if (vast) return el("div", {class: "section-card"},
    el("div", {class: "card-title"}, el("span", {text: "Vast.ai"})),
    el("div", {class: "card-desc", text: "Fizgig is running on a rented GPU (a Vast.ai instance). These settings only appear here."}),
    el("div", {class: "card-content"},
      P.https ? el("div", {class: "field-hint"}, "Its https address (new each time the instance starts): ",
        el("a", {href: P.https, target: "_blank", rel: "noopener", text: P.https}),
        location.protocol === "http:" ? " - use it rather than this plain http one: your password and login travel " +
        "encrypted there." : "") : null,
      el("label", {class: "checkbox-label"}, on, " Stop this instance when a training run finishes"),
      el("div", {class: "field-hint", text: "Only after a run completes on its own - never after a pause, a stop or a " +
        "failure, and with a queue only after its last run. A two-minute countdown you can cancel comes first, on any " +
        "page. It stops the GPU billing; the instance's disk still bills until you destroy it on Vast's site, so bring " +
        "your LoRAs home and destroy it when you're done. No key needed: the instance stops itself."}),
      midRun,
      P.storage ? el("div", {class: "field-hint" + (P.storage[1] ? " warn" : ""), text: P.storage[0]}) : null,
      el("div", {class: "field-hint prefs-note", text: "Your files: datasets in /workspace/datasets, models in " +
        "/workspace/models, finished LoRAs in /workspace/output_loras. They survive stopping and starting the " +
        "instance, and go when you destroy it. A stopped instance may not start again while someone else rents its GPU."}),
      el("div", {class: "field-hint", text: "Closing this browser tab doesn't stop training: Fizgig runs on the instance, not in your browser."})));
  return el("div", {class: "section-card"},
    el("div", {class: "card-title"}, el("span", {text: "RunPod"})),
    el("div", {class: "card-desc", text: "Fizgig is running on a rented GPU" + (P.pod_id ? ` (pod ${P.pod_id})` : "") + ". These settings only appear here."}),
    el("div", {class: "card-content"},
      el("label", {class: "checkbox-label"}, on, " Stop this pod when a training run finishes"),
      el("div", {class: "field-hint", text: "Only after a run completes on its own - never after a pause, a stop or a failure, " +
        "and with a queue only after its last run. A two-minute countdown you can cancel comes first, on any page. It stops " +
        "the pod, never terminates it, so your files are still here when you start it again."}),
      midRun,
      el("div", {class: "field-row"}, el("span", {class: "field-label", text: "RunPod API key:"}), key,
        el("button", {class: "btn btn-sm", text: "Clear", onclick: () => saveKey("")}), keySay),
      el("div", {class: "field-hint", text: "Make one at RunPod > Settings > API Keys. The key RunPod gives a pod by itself " +
        "can't stop pods. Saved in prefs.json on your volume, so it carries over to your next pod and stays out of any shared template."}),
      P.storage ? el("div", {class: "field-hint" + (P.storage[1] ? " warn" : ""), text: P.storage[0]}) : null,
      el("div", {class: "field-hint prefs-note", text: "Your files: datasets in /workspace/datasets, models in " +
        "/workspace/models, finished LoRAs in /workspace/output_loras. Everything under /workspace survives stopping " +
        "and restarting the pod. On the default Volume Disk it goes when you TERMINATE the pod, so stop rather than " +
        "terminate between sessions; a Network Volume survives that too."}),
      el("div", {class: "field-hint", text: "Closing this browser tab doesn't stop training: Fizgig runs on the pod, not in your browser."})));
}

// Fizgig 8's welcome, on this computer's own Fizgig (not a pod's or another computer's page): once a launch until
// Don't show this again. The tour ships with Fizgig; if it won't play, the words stand on their own.
function welcomeShow(pr) {
  let seen = false;
  try { seen = sessionStorage.getItem("fizgig.welcomeSeen") === "1"; } catch (e) { /* no storage */ }
  if (pr.welcome_hidden || seen || S.onPod || pageIsElsewhere() || $("welcomeModal").classList.contains("active")) return;
  const v = $("welcomeVideo"), m = $("welcomeModal");
  v.onerror = () => { v.style.display = "none"; };
  const close = async () => {
    m.classList.remove("active"); v.pause();
    try { sessionStorage.setItem("fizgig.welcomeSeen", "1"); } catch (e) { /* no storage */ }
    if ($("welcomeNever").checked) api("/api/prefs", {changes: {welcome_v8_hidden: "1"}}).catch(() => {});
  };
  $("welcomeClose").onclick = close; $("welcomeX").onclick = close;
  m.onclick = (e) => { if (e.target === m) close(); };
  m.classList.add("active");
  v.play().catch(() => {});
}

// the web UI's password (prefs web_password, the desktop's): empty = this machine only; set = other devices log in
function webPasswordRows() {
  if (S.onPod) return [el("div", {class: "field-hint", text: "Password: this pod's FIZGIG_PASSWORD from the template, or the " +
    "one the pod made up and printed in its log on its first start."})];
  const P = S.pw;
  const input = el("input", {type: "password", class: "w-path", autocomplete: "new-password",
    placeholder: P.saved ? "saved - type a new one to replace it" : "8 characters or more"});
  const show = el("input", {type: "checkbox", onchange: () => { input.type = show.checked ? "text" : "password"; }});
  const say = el("div", {class: "field-hint"});
  const base = () => (P.env ? "A password given when Fizgig was started (FIZGIG_PASSWORD or --password) is used instead of this one. " : "") +
    "Empty: the web UI opens on this machine only. Set (8 characters or more): other devices can open it too, after " +
    "logging in. Whoever has it can use Fizgig and open files on this machine, so treat it like a remote-desktop " +
    "password. Saved in prefs.json as plain text.";
  // the running web UI still has the old password (or none) until it restarts
  const restart = el("button", {class: "btn btn-sm btn-primary", text: "Restart Fizgig now", style: "display:none",
    title: "Fizgig starts again with the new password (a run going stops it: wait for that first)", onclick: async () => {
      restart.disabled = true; say.textContent = "Restarting Fizgig...";
      try { await api("/api/restart", {}); } catch (e) { say.textContent = e.message; restart.disabled = false; return; }
      await new Promise((ok) => setTimeout(ok, 1500));
      for (let i = 0; i < 120; i++) {                // back when its page answers again
        try { if ((await fetch("/icon.png", {cache: "no-store"})).ok) break; } catch (e) { /* not yet */ }
        await new Promise((ok) => setTimeout(ok, 500));
      }
      const a = window.pywebview && window.pywebview.api;
      if (a && a.home && !pageIsElsewhere()) a.home(); else location.reload();   // the window logs itself in again
    }});
  const stale = () => !P.env && P.saved !== !!S.passwordOn || P.changed;
  const links = el("div", {class: "field-hint"});
  const remove = el("button", {class: "btn btn-sm", text: "Remove", onclick: () => save("")});
  const paint = () => {
    remove.style.display = P.saved ? "" : "none";
    restart.style.display = stale() ? "" : "none";
    const port = S.portNow || S.webPort;
    links.textContent = P.lan ? (P.saved ? `Other devices on your network: http://${P.lan}:${port}/` + (stale() ? " (once Fizgig has restarted)." : ".") +
      (navigator.platform.startsWith("Win") ? " The first time, Windows may ask to let Python through the firewall: allow it on private networks." : "")
      : "Set a password to open it from other devices.") : "";
  };
  const save = async (v) => {
    try {
      const r = await api("/api/prefs/password", {password: v});
      P.saved = r.saved; P.changed = true; input.value = "";
      input.placeholder = P.saved ? "saved - type a new one to replace it" : "8 characters or more";
      say.textContent = (r.saved ? "Saved: restart Fizgig for it to apply. " : "Removed: restart Fizgig for it to apply. ") + base(); paint();
    } catch (e) { say.textContent = e.message; }
  };
  input.addEventListener("change", () => input.value.trim() && save(input.value.trim()));
  say.textContent = base(); paint();
  return [el("div", {class: "field-row"}, el("span", {class: "field-label", text: "Password:"}), input,
    el("label", {class: "checkbox-label"}, show, " Show"), remove, restart), say, links, firewallRow()];
}

// Windows Firewall, on the computer Fizgig runs on: is Fizgig let through on this network, and a button that lets it
// (our two rules, private networks only; Windows asks for an administrator's OK). A page from elsewhere gets nothing.
function firewallRow() {
  const row = el("div", {class: "field-row", style: "display:none"});
  const say = el("div", {class: "field-hint", style: "display:none"});
  const box = el("div", {}, row, say);
  const btn = el("button", {class: "btn btn-sm", text: "Allow on my network"});
  const show = (d) => {
    if (!d || d.windows === false || !S.pw.saved) { row.style.display = say.style.display = "none"; return; }
    row.style.display = say.style.display = "";
    if (d.error) { say.textContent = d.error; say.className = "field-hint warn"; return; }
    const all = d.networks || [];
    const nets = all.some((n) => n.lan) ? all.filter((n) => n.lan) : all;   // the home network, not a VPN's adapter
    const pub = d.allowed ? [] : nets.filter((n) => n.category === "Public");   // allowed there too: nothing to change
    const ok = d.allowed && !d.blocked && nets.length;
    row.replaceChildren(el("span", {class: "field-label", text: "Network:"}), btn);
    btn.textContent = "Allow on my network";
    row.style.display = ok ? "none" : "";          // allowed: nothing to do (it comes back, with why, if that changes)
    const lines = [];
    if (pub.length) lines.push(`Windows calls ${pub.map((n) => `"${n.name}"`).join(", ")} a Public network and lets no ` +
      "other device in there. If it's your home network, make it Private: Settings > Network & internet > Wi-Fi or " +
      "Ethernet > your network > Network profile type > Private.");
    if (d.blocked) lines.push("Windows Firewall blocks Fizgig's Python (its question was cancelled once). Allow on my network takes that block off.");
    if (!d.allowed) lines.push("Windows Firewall doesn't let Fizgig in on this port yet. Allow on my network adds two rules " +
      `for private networks only: the web UI's port (and the few above it Fizgig moves to when it's taken) and the one other computers find Fizgig on. Windows asks for an administrator's OK.`);
    if (ok) lines.push("✓ Allowed on your network: your other computers find this one in their Pods tab and can open it.");
    say.textContent = lines.join(" ");
    say.className = "field-hint" + (ok ? "" : " warn");
  };
  btn.onclick = async () => {
    btn.disabled = true; say.textContent = "Waiting for Windows (it asks for an administrator's OK)..."; say.className = "field-hint";
    try { show(await api("/api/lan/firewall", {})); } catch (e) { say.textContent = e.message; say.className = "field-hint warn"; }
    btn.disabled = false;
  };
  api("/api/lan/firewall").then(show).catch(() => {});
  return box;
}

// what Fizgig does on the network by itself (issue #236): finding your other computers and being found by them (a
// broadcast on the local network while the Helpers or Pods tab is open, and an answer to theirs), and the release check
function networkCard() {
  const box = (key, label) => {
    const on = el("input", {type: "checkbox"});
    on.checked = String(S.set[key] ?? "1") !== "0";
    on.addEventListener("change", () => api("/api/prefs", {changes: {[key]: on.checked}})
      .then(() => { S.set[key] = on.checked ? "1" : "0"; }));
    return el("label", {class: "checkbox-label"}, on, " " + label);
  };
  return el("div", {class: "section-card"}, el("div", {class: "card-title"}, el("span", {text: "Network"})),
    el("div", {class: "card-desc", text: "What Fizgig does on the network without being asked. Pods, RunPod and " +
      "computers added by their address work either way."}),
    el("div", {class: "card-content"},
      box("lan_discovery", "Find Fizgig on my other computers"),
      el("div", {class: "field-hint", text: "While the Helpers or Pods tab is open, this computer asks the local network " +
        "which computers run Fizgig, and it answers theirs (with a web password set). Nothing leaves your network. " +
        "Off: no asking and no answering."}),
      box("update_check", "Check for new releases"),
      el("div", {class: "field-hint", text: "Once at start-up, Fizgig asks GitHub for its newest release tag (the About " +
        "box says when one is out). Off: no check unless you press Check for updates in the About box."})));
}

// the web UI's port: saved by the first launch (8081, or the next free port if 8081 was taken)
function webPortCard() {
  const input = el("input", {type: "number", min: "1024", max: "65535", class: "w-port", value: S.webPort || 8081});
  const hint = el("div", {class: "field-hint"});
  if (S.onPod || S.portFixed) {
    input.disabled = true;
    input.value = S.portNow;
    hint.textContent = S.onPod ? "A pod always uses 8081, the port its proxy forwards." :
      `Set by --port where Fizgig was started (${S.portNow}). Change it there.`;
    return el("div", {class: "section-card"},
      el("div", {class: "card-title"}, el("span", {text: "Web UI"})),
      el("div", {class: "card-content"},
        el("div", {class: "field-row"}, el("span", {class: "field-label", text: "Port:"}), input), hint,
        ...webPasswordRows(), logoutRow()));
  }
  const say = () => {
    const p = Number(input.value);
    hint.textContent = p === S.portNow ? "The port this page is on. Fizgig uses it every time it starts." :
      `Fizgig moves to port ${p} next time it starts (this page is on ${S.portNow} until then). ` +
      "If that port is taken, it uses the next free one above it.";
  };
  input.addEventListener("change", async () => {
    const p = Number(input.value);
    if (!Number.isInteger(p) || p < 1024 || p > 65535) { input.value = S.webPort; return say(); }
    await api("/api/prefs", {changes: {web_port: p}});
    S.webPort = p;
    say();
  });
  say();
  return el("div", {class: "section-card"},
    el("div", {class: "card-title"}, el("span", {text: "Web UI"})),
    el("div", {class: "card-content"},
      el("div", {class: "field-row"}, el("span", {class: "field-label", text: "Port:"}), input), hint,
      ...webPasswordRows(), logoutRow()));
}

// with a password: log this device out, or every device (a phone left logged in, a password you shared)
function logoutRow() {
  if (!S.passwordOn) return null;
  const done = () => { showLogin(); };
  return el("div", {class: "logout-row"},
    el("div", {class: "field-row"},
      el("button", {class: "btn", text: "Log out", onclick: () => api("/api/logout", {}).then(done)}),
      el("button", {class: "btn btn-warning", text: "Log out everywhere", onclick: () =>
        api("/api/logout_all", {}).then(done)})),
    el("div", {class: "field-hint", text: "Log out everywhere ends every device's login, this one too: a phone left " +
      "logged in, say. Each one asks for the password again. If someone else knows the password, change it too " +
      "(Password, above), or they can just log in again."}));
}

function checkModels() {
  const missing = S.fam.model_files.filter((m) => m.required && !S.prefs[m.pref_key]).map((m) => m.label);
  $("modelWarning").style.display = missing.length ? "" : "none";
  $("modelWarningText").textContent = missing.length ?
    `${S.fam.display_name} needs: ${missing.join(", ")}. Set them on the Preferences tab.` : "";
}

// ---- saved state -------------------------------------------------------------------------------------------
function saveState() {
  S.all[S.family] = S.values;
  // only the model on show: another open page (a phone, another tab) keeps the models it changed
  api("/api/state", {family: S.family, image_folder: S.imageFolder,
    model_values: {family: S.family, values: S.values}}).catch(() => {});
}
const saveStateSoonInner = debounce(() => { S.saveWaiting = false; saveState(); }, 600);
const saveStateSoon = () => { S.saveWaiting = true; saveStateSoonInner(); };

// ---- the run -----------------------------------------------------------------------------------------------
const runBody = () => ({family: S.family, values: S.values, image_folder: S.imageFolder});

function showProblems(list, okText) {
  const box = $("problems"), bad = (list || []).length;
  box.classList.toggle("ok", !bad && !!okText);
  box.textContent = bad ? "Can't start yet:\n• " + list.join("\n• ") : (okText || "");
}

async function check() {
  try {
    const p = await api("/api/plan", runBody());
    showProblems(p.problems, "✓ All good - nothing missing, Start Training will work. It will run: " +
      p.stages.map((s) => s.name).join(" → ") + "." +
      ((p.warnings || []).length ? "\n\nWorth a look before you start:\n• " + p.warnings.join("\n• ") : ""));
  } catch (e) { showProblems([e.message]); }
}

// the Start checks' warnings (launch.start_checks): shown with Start anyway, never a block
function showWarnings(list, body) {
  const box = $("problems");
  box.classList.remove("ok");
  box.replaceChildren(el("div", {text: "Worth a look before you start:\n• " + list.join("\n• ")}),
    el("div", {class: "field-row"},
      el("button", {class: "btn btn-primary btn-sm", text: "Start anyway", onclick: () => start("/api/start", {...body, anyway: true})}),
      el("button", {class: "btn btn-sm", text: "Cancel", onclick: () => box.replaceChildren()})));
}

async function start(path, body) {
  if (S.starting) return;                  // checking a run takes the server a few seconds: one press is one run
  S.starting = true;
  const btn = path === "/api/start" ? $("btnStart") : null;
  if (btn) { btn.disabled = true; btn.dataset.label = btn.dataset.label || btn.textContent; btn.textContent = "Starting…"; }
  try {
    const r = await api(path, path === "/api/start" ? {...runBody(), ...(body || {})} : (body || {}));
    if (r.warnings && r.warnings.length) { showWarnings(r.warnings, body || {}); return; }
    showProblems(r.problems || (r.error ? [r.error] : []));    // stays up: the run line below refreshes itself
    // the form went to a run that started: it isn't the queued run being edited any more (Save to queue would have
    // put this run's settings in its place)
    if (r.ok && S.editQueue) {
      queueEditing(null);
      $("runStatus").textContent = "Started. The queued run you were editing is left as it was.";
    }
  } catch (e) { showProblems([e.message]); }
  finally {
    S.starting = false;
    if (btn) btn.textContent = btn.dataset.label;
    await pollRun();                        // sets Start's state from the run
  }
}

// Resume: the paused run's own settings, unless the form now has another picture size for it (Target Megapixels,
// a clip family's clip size): then a choice - as it was, or at the new size (the cache is made again first, keeping
// what still fits). Training at a bigger size for the last epochs, say.
const SIZE_KEYS = {DATASET_MEGAPIXELS: "Target Megapixels", CLIP_MEGAPIXELS: "Clip Megapixels"};
// what Resume takes from the form as it is now (server.py _RESUME_SAMPLE_KEYS); the rest trains on as the run began
const RESUME_KEYS = ["SAMPLE_PROMPT", "SAMPLE_SEED", "SAMPLE_WIDTH", "SAMPLE_HEIGHT", "SAMPLE_STEPS", "SAMPLE_CFG_SCALE",
  "FAMILY_TURBO_STRENGTH", "SAMPLE_CHECKPOINT", "BLOCKS_SWAP", "FAMILY_PREVIEW_LOOK",
  "SAMPLE_EVERY_N_EPOCHS", "SAMPLE_LATER", "SAMPLE_UNTIL_EPOCH", "SAMPLE_EVERY_AFTER",
  "SAVE_EVERY_N_EPOCHS", "SAVE_LATER", "SAVE_UNTIL_EPOCH", "SAVE_EVERY_AFTER",
  "LEARNING_RATE", "ADAPTIVE_LR_MIN", "ADAPTIVE_LR_MAX"];

// while this model's run is paused, the rows Resume uses are marked "Applies on Resume"
function resumeMarks() {
  const r = S.run || {}, on = r.state === "paused" && r.family === S.family && r.resume_mode !== "ft";
  const keys = new Set([...RESUME_KEYS, ...Object.keys(SIZE_KEYS)]);
  document.querySelectorAll(".form-row[data-key]").forEach((row) =>
    row.classList.toggle("row-resume", on && keys.has(row.dataset.key)));
}

async function resumeClick() {
  const r = S.run || {}, was = r.values || {}, box = $("resumeChoice");
  // the previews as the Samples tab has them now (prompts, seed, size, steps, CFG, Turbo strength, the preview
  // checkpoint tick), block swap, the save schedule and the learning rate go with the resume
  const samples = r.family === S.family ? Object.fromEntries(RESUME_KEYS
    .filter((k) => k in S.values).map((k) => [k, S.values[k]])) : {};
  const diff = r.family === S.family ? Object.keys(SIZE_KEYS).filter((k) => S.values[k] !== undefined &&
    String(S.values[k]).trim() !== "" && Number(S.values[k]) !== Number(was[k])) : [];
  if (!diff.length) { box.style.display = "none"; return start("/api/resume", {samples}); }
  const say = diff.map((k) => `${SIZE_KEYS[k]} ${was[k] || "?"} → ${S.values[k]}`).join(", ");
  const go = (body) => async () => {
    box.style.display = "none";
    try { const x = await api("/api/resume", {...body, samples}); showProblems(x.problems || (x.error ? [x.error] : [])); }
    catch (e) { showProblems([e.message]); }
    pollRun();
  };
  box.replaceChildren(
    el("span", {text: `The form's size differs from the run's (${say}). Resume:`}),
    el("button", {class: "btn btn-sm btn-primary", text: "At the new size",
      title: "Caches the pictures again at the new size first (what already fits is kept), then trains on from where it paused",
      onclick: go(Object.fromEntries(diff.map((k) => [k, S.values[k]])))}),
    el("button", {class: "btn btn-sm", text: "As it was", onclick: go({})}),
    el("button", {class: "btn btn-sm", text: "Cancel", onclick: () => { box.style.display = "none"; }}));
  box.style.display = "";
}

const RUN_LABELS = {idle: "Idle", starting: "Starting", running: "Running", pausing: "Pausing", paused: "Paused",
                    completed: "Finished", failed: "Failed", stopped: "Stopped", moved: "Moved"};

async function pollRun() {
  try { S.run = await api("/api/run"); } catch (e) { return; }
  resumeMarks();
  runSummary();
  if (training() !== S.wasTraining) {             // the LoRA lists' Studio buttons show only between runs
    if (S.wasTraining !== undefined && $("tab-training").classList.contains("active")) loadCheckpoints();
    S.wasTraining = training();
  }
  const ps = S.run.pod_stop || {};
  $("podStopBar").style.display = ps.counting || ps.note ? "" : "none";
  $("podStopKeep").textContent = ps.counting ? "Keep the pod running" : "OK";      // a result note: dismissed
  $("podStopText").textContent = ps.counting ? `Stopping this pod in ${ps.left} s so it stops billing.` : ps.note || "";
  const r = S.run, live = r.running || r.state === "starting";
  liveNote(live);
  $("runPodNote").style.display = S.onPod && live ? "" : "none";     // a pod's run doesn't need this page open
  const label = RUN_LABELS[r.state] || r.state;
  $("runStatus").textContent = r.lora ? `${label}: ${r.lora}${r.stage ? " · " + r.stage : ""}` : label;
  $("runBadge").textContent = label;
  $("runBadge").dataset.state = r.state;
  $("runCardBadge").textContent = label;
  $("runCardBadge").dataset.state = r.state;
  showProgress(r.progress, r.state, r.lora, r.stage);
  const moveBusy = !!(r.move && ["moving", "back", "moved"].includes(r.move.state));
  $("btnStart").disabled = live || moveBusy || !!S.starting;
  $("btnPause").disabled = !r.running;
  // a model that can't pause (a RefMod is made in one go): no Pause button for its runs
  const runFam = S.schema.families.find((f) => f.key === (r.family || S.family)) || {};
  $("btnPause").style.display = runFam.no_pause ? "none" : "";
  $("btnStop").disabled = !live;
  $("btnResume").style.display = !live && r.resume && !moveBusy ? "" : "none";
  renderWebQueue(r.web_queue || {items: []});
  queueSummary(r.web_queue || {items: []}, r);
  runLossTick(live);
  // Per-image loss on the run card while a run with the loss watch trains (Runs & outputs has it after): any of its
  // ticks runs the watch - per-image LR, the look warm-up and auto-recaption track every image too
  // always there for a model with the loss watch: greyed (its hover says which ticks turn it on) until a run tracks it
  const ilOn = live && r.history_id && ["KREA2_LOSS_WATCH", "KREA2_PER_IMAGE_LR", "KREA2_WARMUP_LOOK",
    "KREA2_AUTO_RECAPTION"].some((k) => [true, "true", "True", "1", 1].includes((r.values || {})[k]));
  const ilDesc = S.schema.families.find((f) => f.key === (live ? r.family : S.family));
  const ilb = $("btnImageLoss");
  ilb.style.display = ilDesc && ilDesc.loss_watch !== false ? "" : "none";
  ilb.classList.toggle("is-off", !ilOn);
  ilb.setAttribute("aria-disabled", ilOn ? "false" : "true");
  ilb.title = ilOn ? IL_ON_TIP : IL_OFF_TIP;
  if (!ilOn) $("runImageLoss").style.display = "none";
  // Sample now / Save now: while a LoRA trains (a fine-tune saves and previews at whole rotations)
  const runDesc = S.schema.families.find((f) => f.key === r.family);
  const ftRun = r.values && r.values.FAMILY_KIND && /fine-tune/i.test(String(r.values.FAMILY_KIND));
  const nowOk = !!(r.running && r.stage === "Training" && runDesc && !runDesc.no_pause && !ftRun);
  for (const id of ["btnSampleNow", "btnSaveNow"]) $(id).style.display = nowOk ? "" : "none";
  if (!nowOk) $("runNowSay").textContent = "";
  // Move to…: a LoRA run training or paused here goes on on another computer, and comes back the same way
  const mv = r.move || null, moving = mv && ["moving", "back", "moved"].includes(mv.state);
  const moveOk = !!(runDesc && !runDesc.no_pause && !ftRun && !moving &&
                    (r.running && r.stage === "Training" || r.state === "paused"));
  $("btnMove").style.display = moveOk ? "" : "none";
  if (!moveOk) S.moveOpen = false;
  moveDraw(mv);
  sliderLiveDraw(r.slider_live);
}

// ---- the workbench's render choice (description.workbench_render_choice): while the Samples tab previews on the
// training model, Studio, LoRA Royale and the Profiler render on the preview checkpoint (the default) or follow the
// Samples tab. One choice per model (prefs workbench_render_<key>); every tab shows the same one
function renderChoiceDraw(id, L) {
  const box = $(id);
  if (!box) return;
  const rc = L && L.render_choice;
  box.style.display = rc ? "" : "none";
  if (!rc) return;
  const pick = (v) => async () => {
    try { await api("/api/prefs", {changes: {[`workbench_render_${L.key}`]: v}}); } catch (e) { showProblems([e.message]); }
    renderChoiceRefresh();
  };
  const opt = (v, text, title) => el("label", {class: "radio-label", title},
    el("input", {type: "radio", name: id, ...(rc.value === v ? {checked: true} : {}), onchange: pick(v)}), " " + text);
  box.replaceChildren(el("span", {class: "field-label", text: "Render with:"}),
    opt("checkpoint", rc.checkpoint, "Renders on the model most people generate with (the default). The Samples tab's " +
        "previews are on your Base"),
    opt("samples", "Follow Samples tab settings", "Renders as the Samples tab's previews do: your Base" +
        (rc.look ? ", with the SFT look LoRA" : "") + ", at the Samples tab's steps and CFG"),
    el("span", {class: "field-hint", text: "Applies from Studio's next Load, and the next Royale or Profiler render " +
      "(the model reloads, so that one takes longer)"}));
}

// every tab's box (and Studio's note) after a change, or a tab's model changing
async function renderChoiceRefresh() {
  for (const [id, sel] of [["stRender", "stFamily"], ["ryRenderChoice", "ryFamily"], ["pfRenderChoice", "pfFamily"]]) {
    if (!$(id) || !$(sel) || !$(sel).value) continue;
    const r = await api("/api/studio/render_choice?family=" + encodeURIComponent($(sel).value)).catch(() => null);
    if (!r) continue;
    renderChoiceDraw(id, r);
    if (id === "stRender" && S.st.layout && S.st.layout.key === r.key) {
      Object.assign(S.st.layout, r);
      const L = S.st.layout;
      $("stDitNote").textContent = L.follows_samples && L.follow_ck ? L.follow_ck : L.follows_samples ?
        `Previews ${L.follow_turbo ? "on the " + L.follow_turbo + ", " : ""}${L.follow_steps} steps, CFG ${L.follow_cfg}` +
        `${L.follow_look ? ", " + L.follow_look : ""} - from this model's Samples tab.` : $("stDitNote").textContent;
    }
  }
}

// ---- moving a run to another computer (web/helpers.py move_*): the chooser, then the move's own line
function moveDraw(mv) {
  const box = $("runMove");
  if (S.moveOpen) return;                          // the chooser is up: left alone until it's used or closed
  if (!mv) { box.style.display = "none"; box.replaceChildren(); return; }
  box.style.display = "";
  const go = (path, body) => async () => {
    try { await api(path, body || {}); } catch (e) { showProblems([e.message]); }
    pollRun();
  };
  const rm = mv.remote || {}, kids = [];
  let say = mv.text || "";
  if (mv.state === "moved") {
    say = rm.state === "completed" ? `Finished on ${mv.to}.` :
      rm.state === "paused" ? `Paused on ${mv.to}.` :
      rm.state === "failed" || rm.state === "stopped" ? `It ${rm.state === "failed" ? "failed" : "was stopped"} on ${mv.to} ` +
        "(its log there says why). Bring it back to carry on here from its last save." :
      rm.state === "away" ? `Training on ${mv.to} (it isn't answering just now).` :
      rm.state === "login" ? `On ${mv.to}: its login ran out, so this computer can't see it. Add it again on the Helpers tab.` :
      rm.state === "running" || rm.state === "starting" ?
        `Training on ${mv.to}` + (rm.epoch ? `: epoch ${rm.epoch} of ${rm.epochs}` : rm.stage ? `: ${rm.stage}` : "") +
        (rm.eta ? ` · ${rm.eta} left` : "") : `On ${mv.to}.`;
    if (mv.error) say += ` (${mv.error})`;
    if (mv.home_text) say += ` Coming home as they're saved: ${mv.home_text}.`;
    if (S.moveDelete === undefined) S.moveDelete = true;
    const del = () => !!(mv.pod && S.moveDelete);
    kids.push(el("button", {class: "btn btn-sm btn-primary", text: rm.state === "completed" ? "Bring its LoRAs here" : "Bring it back",
      title: "Paused there at the end of its epoch, its new checkpoints, previews and saved state brought here, and it carries on here",
      onclick: () => go("/api/helpers/move_back", {now: false, delete: del()})()}));
    if (rm.state === "running") kids.push(el("button", {class: "btn btn-sm", text: "Bring it back now",
      title: "Stopped there now and brought back from its last save (the epochs since it are trained again here)",
      onclick: () => go("/api/helpers/move_back", {now: true, delete: del()})()}));
    if (mv.pod) kids.push(el("label", {class: "checkbox-label", title: "Once the run and everything it made are here, the " +
      "pod is deleted on RunPod, so it stops billing. Untick it to keep the pod (its files stay on it)."},
      el("input", {type: "checkbox", ...(S.moveDelete ? {checked: true} : {}),
        onchange: (e) => { S.moveDelete = e.target.checked; }}), ` Delete the pod once it's all here`));
    // as the Pods tab's Open: in the app window it switches over there (a pod logs you in); in a browser, a new tab
    if (mv.url) kids.push(el("button", {class: "btn btn-sm", text: inApp() ? `Open ${mv.to}` : `Open ${mv.to} ↗`,
      onclick: () => podOpen({url: mv.url, name: mv.to})}));
    if (["away", "login", "gone", "failed", "stopped"].includes(rm.state) || mv.error)
      kids.push(el("button", {class: "btn btn-sm", text: "Carry on here",
        title: `For a ${mv.to} that won't come back (a pod deleted): carries on here from this computer's last save - ` +
               "the epochs it trained there are lost. It's stopped there if it still answers",
        onclick: go("/api/helpers/move_clear", {here: true})}));
  } else if (mv.state === "failed") {
    kids.push(el("button", {class: "btn btn-sm", text: "OK", onclick: go("/api/helpers/move_clear")}));
  }
  const bar = mv.progress != null ? el("div", {class: "run-bar"}, el("span", {style: `width:${Math.round(100 * mv.progress)}%`})) : null;
  box.replaceChildren(el("div", {class: "field-hint", text: say}), ...(bar ? [bar] : []),
    ...(kids.length ? [el("div", {class: "field-row"}, ...kids)] : []));
}

async function moveOpen() {
  const box = $("runMove");
  S.moveOpen = true;
  box.style.display = "";
  box.replaceChildren(el("div", {class: "field-hint", text: "Finding the computers it could move to…"}));
  let hs = [];
  try { hs = (await api("/api/helpers")).helpers || []; } catch (e) { /* none listed */ }
  const ready = hs.filter((h) => !h.inbound && h.state === "ready" && (h.build === "same" || !h.build));
  const close = () => { S.moveOpen = false; pollRun(); };
  if (!ready.length) {
    box.replaceChildren(el("div", {class: "field-hint", text: hs.length ?
      "None of your helpers can take it now: each needs to be on, not training, and on the same Fizgig version (Helpers tab)." :
      "Add the computer to move it to on the Helpers tab first: another computer running Fizgig, or a pod."}),
      el("div", {class: "field-row"}, el("button", {class: "btn btn-sm", text: "OK", onclick: close})));
    return;
  }
  const pick = el("select", {class: "w-combo"}, ...ready.map((h) => el("option", {value: h.id,
    text: `${h.name}${h.gpu ? ` (${h.gpu}${h.vram ? `, ${h.vram} GB` : ""})` : ""}`})));
  const home = el("input", {type: "checkbox", checked: true});
  const send = (now) => async () => {
    try {
      await api("/api/helpers/move", {id: pick.value, now, home: home.checked});
      S.moveOpen = false;
    } catch (e) { showProblems([e.message]); }
    pollRun();
  };
  const running = S.run && S.run.running;
  box.replaceChildren(
    el("div", {class: "field-hint", text: "It pauses here, its saved state and training folder go to the other computer, " +
      "and it carries on there on the same model files (it downloads any it's missing and makes its own cache). " +
      "Bring it back from here when you want it."}),
    el("div", {class: "field-row"}, el("span", {class: "field-label", text: "Move it to"}), pick,
      el("button", {class: "btn btn-sm btn-primary", text: running ? "At the end of this epoch" : "Move it",
        onclick: send(false)}),
      ...(running ? [el("button", {class: "btn btn-sm", text: "Now",
        title: "Stopped now and moved from its last save (the epochs since it are trained again there)",
        onclick: send(true)})] : []),
      el("button", {class: "btn btn-sm", text: "Cancel", onclick: close})),
    el("label", {class: "checkbox-label", title: "Each checkpoint and preview comes back to this run's folder here as " +
      "it's saved there, one after another (a slow link just queues them), so bringing the run back only has the " +
      "last few to fetch."}, home, " Bring its checkpoints and previews back here as they're saved"));
}

// ---- a prompt slider's live prompts: edited here while it trains, taken by the trainer at its next epoch boundary
const SL_FIELDS = {base: "slBase", pos: "slPos", neg: "slNeg", guidance: "slGuidance", anchors: "slAnchors"};
S.sl = {shown: null, edited: false};
function sliderLiveDraw(sl) {
  const box = $("sliderLive");
  box.style.display = sl ? "" : "none";
  if (!sl) { S.sl = {shown: null, edited: false}; return; }
  const now = JSON.stringify([sl.base, sl.pos, sl.neg, sl.guidance, sl.anchors]);
  if (!S.sl.edited && S.sl.shown !== now) {           // the run's prompts, unless you're part-way through an edit
    for (const [k, id] of Object.entries(SL_FIELDS)) $(id).value = sl[k];
    S.sl.shown = now;
  }
  $("slSay").textContent = sl.waiting ? "Waiting for the end of this epoch…" :
    sl.epoch != null ? `Training on these since epoch ${sl.epoch + 1}.` : "";
}

function wireSliderLive() {
  for (const id of Object.values(SL_FIELDS)) $(id).addEventListener("input", () => { S.sl.edited = true; });
  $("slReset").addEventListener("click", () => { S.sl = {shown: null, edited: false}; pollRun(); });
  $("slApply").addEventListener("click", async () => {
    const body = Object.fromEntries(Object.entries(SL_FIELDS).map(([k, id]) => [k, $(id).value.trim()]));
    try {
      await api("/api/run/slider_prompts", body);
      S.sl = {shown: null, edited: false};
      $("slSay").textContent = "Waiting for the end of this epoch…";
      pollRun();
    } catch (e) { $("slSay").textContent = e.message; }
  });
}

// A button whose job lasts as long as its request (a score, an export, a save): greyed from the press until it
// answers, so a second press can't start it again (#249). Jobs the page polls grey their buttons from the status.
function held(btn, fn) {
  return async (...a) => {
    if (btn.disabled) return;
    btn.disabled = true;
    try { return await fn(...a); } finally { btn.disabled = false; }
  };
}

// Per-image loss, on the run card and Runs & outputs: what it shows, and (greyed) how to turn it on
const IL_ON_TIP = "Each training image's loss by epoch, against the rest of the dataset: which images are still being learned, which are done, and what the loss watch decided about each";
const IL_OFF_TIP = "Each training image's loss by epoch, and what the loss watch decided about each. To use it, tick one of these on the Training tab before you press Start: Track how each image trains, per-image LR, the look warm-up or auto-recaption.";

// the queue at a glance, from every tab (the sidebar, under the run, always there) and at the top of the Run card:
// what's next, or that it's held and why, with Start queue right there; the queue itself is on Runs & outputs
function toQueue() {
  showTab("outputs");
  setTimeout(() => $("webQueueCard").scrollIntoView({behavior: "smooth", block: "start"}), 50);
}

function queueSummary(q, r) {
  const items = q.items || [], n = items.length;
  const side = $("sideQueue"), up = $("runUpNext");
  up.style.display = n ? "" : "none";
  if (!n) {
    if (S.queueSumSig === "empty") return;
    S.queueSumSig = "empty";
    side.classList.remove("held");
    $("sideQueueBadge").textContent = "Empty";
    $("sideQueueBadge").dataset.state = "idle";
    $("sideQueueList").replaceChildren(el("div", {class: "side-queue-empty", text: "Nothing queued"}));
    $("sideQueueHeld").textContent = "";
    $("sideQueueStart").style.display = "none";
    return;
  }
  const text = q.held ? `Queue held: ${q.reason || "press Start queue to carry on"}` :
    q.waiting ? `Queue waiting: ${q.waiting}` : "";
  const sig = JSON.stringify([text, q.held, q.reason, q.waiting, items.map((it) => [it.id, it.lora])]);
  if (sig === S.queueSumSig) return;              // as renderWebQueue: a rebuild mid-click drops Start queue's click
  S.queueSumSig = sig;
  side.classList.toggle("held", !!q.held);
  const badge = $("sideQueueBadge");
  badge.textContent = q.held ? "Held" : q.waiting ? "Waiting" : `${n} queued`;
  badge.dataset.state = q.held ? "paused" : q.waiting ? "pausing" : "running";
  $("sideQueueList").replaceChildren(...items.slice(0, 3).map((it, i) => el("div", {class: "side-queue-row"},
    el("span", {class: "n", text: String(i + 1)}), el("span", {class: "name", text: it.lora, title: it.lora}))),
    ...(n > 3 ? [el("div", {class: "side-queue-more", text: `+${n - 3} more`})] : []));
  $("sideQueueHeld").textContent = q.held ? q.reason || "Press Start queue to carry on." : q.waiting || "";
  $("sideQueueStart").style.display = q.held ? "" : "none";
  up.classList.toggle("held", !!q.held);
  up.replaceChildren(...[el("span", {text: q.held ? text : `Up next: ${items.slice(0, 3).map((it, i) => `${i + 1}. ${it.lora}`).join(" · ")}` +
      (n > 3 ? ` · and ${n - 3} more` : "")}),
    q.held ? el("button", {class: "btn btn-sm btn-primary", text: "Start queue",
      onclick: () => api("/api/queue/start", {}).then(pollRun).catch((e) => showProblems([e.message]))}) : null,
    el("a", {text: "See the queue", onclick: toQueue})].filter(Boolean));
}

// the web UI's own queue: reorder, remove, start or hold
function renderWebQueue(q) {
  const items = q.items || [];
  const badge = $("webQueueBadge");
  badge.textContent = !items.length ? "" : q.held ? "Held" : q.waiting ? "Waiting" : `${items.length} queued`;
  badge.dataset.state = q.held ? "paused" : q.waiting ? "pausing" : "running";
  $("webQueueState").textContent = q.held ? q.reason : q.waiting ? q.waiting :
    items.length ? "The first starts as soon as nothing else is training." :
    "Nothing queued. Add to queue on the Training tab lines a run up here.";
  $("webQueueStart").style.display = q.held ? "" : "none";
  $("webQueueHold").style.display = q.held || !items.length ? "none" : "";
  $("webQueueClear").style.display = items.length ? "" : "none";
  S.queueIds = items.map((it) => it.id);
  // rebuilt only when the queue changes: rebuilding on every poll replaced a button between press and release, and
  // the browser dropped that click (Remove sometimes did nothing)
  const sig = JSON.stringify(items.map((it) => [it.id, it.lora, it.family, it.image_folder, it.queued_at, it.continues]));
  if (sig === S.queueSig) return;
  S.queueSig = sig;
  const famName = (k) => ((S.schema.families.find((f) => f.key === k) || {}).display_name) || k;
  const host = $("webQueueList");
  host.innerHTML = "";
  items.forEach((it, i) => host.append(el("div", {class: "checkpoint-row queue-row"},
    el("span", {class: "checkpoint-name", text: `${i + 1}. ${it.lora}`}),
    el("span", {class: "field-hint", text: `${famName(it.family)} · ${it.image_folder} · queued ${it.queued_at}` +
      (it.continues ? " · continues from its last save" : "")}),
    el("span", {class: "queue-actions"},
      el("button", {class: "btn btn-sm", text: "↑", title: "Earlier", disabled: i === 0,
        onclick: () => api("/api/queue/move", {id: it.id, delta: -1}).then(pollRun)}),
      el("button", {class: "btn btn-sm", text: "↓", title: "Later", disabled: i === items.length - 1,
        onclick: () => api("/api/queue/move", {id: it.id, delta: 1}).then(pollRun)}),
      el("button", {class: "btn btn-sm", text: "Edit",
        title: "Load its settings into the form on the Training tab; Save to queue puts them back in its place",
        onclick: async () => {
          try {
            await loadIntoForm(await api("/api/queue/item?id=" + encodeURIComponent(it.id)));
            queueEditing({id: it.id, n: i + 1, lora: it.lora});
            showTab("training");
            $("runStatus").textContent = `Editing queued run ${i + 1} (${it.lora}): change what you like, then Save to queue.` +
              (it.continues ? " It still continues from its last save, so keep its LoRA name and output folder." : "");
          } catch (e) { $("runStatus").textContent = e.message; }
        }}),
      el("button", {class: "btn btn-sm", text: "Remove", onclick: () => {
        if (S.editQueue && S.editQueue.id === it.id) queueEditing(null);
        api("/api/queue/remove", {id: it.id}).then(pollRun);
      }})))));
}

function followLog() {
  const src = S.logSrc = new EventSource(`/api/log?since=${S.logSince}&t=${encodeURIComponent(TOKEN)}`);
  const out = $("console");
  // the console as finished lines (body) and the line still being written (tail: a progress bar's latest state),
  // drawn at most every 50 ms. A new tab replays the run's backlog (up to 5000 lines) at once, and rewriting the
  // whole console and laying it out again for every line froze the page on a long run
  const t0 = out.textContent, nl = t0.lastIndexOf("\n");
  let body = t0.slice(0, nl + 1), tail = t0.slice(nl + 1), done = [], queued = false;
  const flush = () => {
    queued = false;
    const atEnd = out.scrollTop + out.clientHeight >= out.scrollHeight - 30;
    if (done.length) { body = (body + done.join("")).slice(-200000); done = []; }
    out.textContent = body + tail;
    if (atEnd) out.scrollTop = out.scrollHeight;
  };
  src.onmessage = (e) => {
    S.logSince = Number(e.lastEventId) || S.logSince;
    const line = JSON.parse(e.data);
    if (line === "\u0000clear\n") { body = tail = ""; done = []; }   // a new run: its own log from here
    else if (line.includes("\r") && !line.endsWith("\n")) tail = line.split("\r").pop();   // a progress bar redraws
    else {
      const all = tail + line.split("\r").pop(), cut = all.lastIndexOf("\n") + 1;
      if (cut) done.push(all.slice(0, cut));
      tail = all.slice(cut);
    }
    if (!queued) { queued = true; setTimeout(flush, 50); }
  };
  src.onerror = () => { src.close(); if (S.logSrc === src) setTimeout(followLog, 3000); };
}

async function pollStats() {
  let s;
  try { s = await api("/api/stats"); } catch (e) { return; }     // the server restarts
  S.stats = s;
  const gb = (b) => (b / 2 ** 30).toFixed(1);
  const meter = (label, value, frac, warn) => el("div", {class: "meter" + (warn ? " warn" : "")},
    el("div", {class: "meter-head"}, el("span", {text: label}), el("span", {text: value})),
    el("div", {class: "meter-bar"}, el("span", {style: `width:${Math.max(0, Math.min(100, frac * 100)).toFixed(1)}%`})));
  const host = $("statusBar");
  host.innerHTML = "";
  host.append(el("div", {class: "gpu-name", text: s.device_name}));
  if (s.util !== undefined && s.util !== null) host.append(meter("GPU", `${Math.round(s.util)}%` +
    (s.temp ? ` · ${Math.round(s.temp)}°C` : ""), s.util / 100, s.temp >= 85));
  host.append(meter("VRAM", `${gb(s.vram[0])} / ${gb(s.vram[1])} GB`, s.vram[1] ? s.vram[0] / s.vram[1] : 0,
    s.vram[1] && s.vram[0] / s.vram[1] > 0.95));
  host.append(meter("RAM", `${gb(s.ram[0])} / ${gb(s.ram[1])} GB`, s.ram[1] ? s.ram[0] / s.ram[1] : 0,
    s.ram[1] && s.ram[0] / s.ram[1] > 0.95));
  // the drive LoRAs go to: checkpoints and saved states fill it fast (red under 5% or 20 GB free)
  const d = s.disk;
  if (d && d.total) {
    const free = d.free >= 2 ** 40 ? `${(d.free / 2 ** 40).toFixed(1)} TB free` : `${gb(d.free)} GB free`;
    const m = meter("Disk" + (d.where && d.where !== "/" ? ` ${d.where}` : ""), free,
      d.used / d.total, d.free < Math.max(20 * 2 ** 30, d.total * 0.05));
    if (d.path) m.title = `LoRAs save to ${d.path}` + (S.onPod ? ". On a pod this is what the drive reports: a " +
      "network volume reports the whole shared storage, so the free space shown may be far more than your " +
      "volume's own size" : "");
    host.append(m);
  }
  // what's loaded on the GPU now, each a click to its tab
  for (const m of s.loaded || []) host.append(el("div", {class: "side-loaded", title: `Open the ${m.tab} tab`,
    onclick: () => showTab(m.tab)}, el("span", {class: "side-loaded-dot"}),
    el("span", {class: "side-loaded-text", text: m.what + (m.name ? `: ${m.name}` : "")})));
  // jobs going in the background, wherever you are
  for (const j of s.jobs || []) {
    const frac = j.total ? j.done / j.total : 0;
    const count = !j.total ? "" : j.total > 2 ** 20 ? `${Math.round(100 * frac)}%` : `${j.done} of ${j.total}`;
    const m = meter(j.what, count, frac, false);
    m.classList.add("side-job");
    if (!j.total) m.classList.add("indeterminate");
    m.title = `Open the ${j.tab} tab`;
    m.addEventListener("click", () => showTab(j.tab));
    host.append(m);
  }
}

// ---- the Training tab's Run summary: time, learning and the run's own setup (its values, not the form's)
function runSummary() {
  const box = $("runSummary"), r = S.run || {};
  if (!r.lora || !r.values || r.stage === undefined && !r.started_at) { box.style.display = "none"; return; }
  const v = r.values, p = r.progress || {}, s = S.stats || {};
  const live = r.running || ["starting", "running", "pausing"].includes(r.state);
  const dur = (sec) => { sec = Math.max(0, Math.round(sec)); const h = Math.floor(sec / 3600), m = Math.floor(sec % 3600 / 60);
    return h ? `${h}h ${m}m` : `${m}m ${sec % 60}s`; };
  const end = live ? Date.now() / 1000 : (r.ended_at && r.ended_at > r.started_at ? r.ended_at : null);
  const cell = (k, val) => el("div", {class: "rs-cell"}, el("span", {class: "rs-key", text: k}),
    el("span", {class: "rs-val", text: val == null || val === "" ? "-" : String(val)}));
  const sec = (title, cells, extra) => {
    const open = !(S.rsShut || {})[title];
    const head = el("div", {class: "rs-head"}, el("span", {text: title}), ...(extra ? [extra] : []),
      el("button", {class: "rs-fold", text: open ? "⌃" : "⌄", title: open ? "Fold" : "Open", onclick: () => {
        S.rsShut = {...(S.rsShut || {}), [title]: open}; runSummary(); }}));
    return el("div", {class: "rs-section"}, head, open ? el("div", {class: "rs-grid"}, ...cells.filter(Boolean)) : null);
  };
  const gb = (b) => b ? `${(b / 2 ** 30).toFixed(1)} GB` : "";
  const time = sec("Time", [
    cell("Training", r.training_at && end ? dur(end - r.training_at) : ""),
    cell("Total", r.started_at && end ? dur(end - r.started_at) : ""),
    cell("Speed", p.speed), cell("Epoch", p.epoch ? `${p.epoch} of ${p.total_epochs}` : ""),
    cell("Peak VRAM", live ? gb(s.vram_peak) : ""), cell(live ? "Time left" : "Steps",
      live ? p.eta : (p.total_steps ? `${p.step} of ${p.total_steps}` : "")),
  ]);
  const lk = S.likeSummary;
  const learning = sec("Learning", [
    cell("Learning rate", p.lr), cell("Grad norm", p.grad_norm), cell("Loss", p.loss),
    lk ? cell("Likeness", `${Math.round(lk.now * 100)}% (epoch ${lk.epoch})` +
      (lk.bestEpoch !== lk.epoch ? ` · best ${Math.round(lk.best * 100)}% at ${lk.bestEpoch}` : "")) : null,
  ]);
  const fam = (S.schema.families.find((f) => f.key === r.family) || {}).display_name || r.family;
  const yes = (k) => v[k] === true || v[k] === "1" || v[k] === "true";
  const first = (x) => String(x == null ? "" : x).split(" (")[0];
  const net = [String(v.NETWORK_TYPE || "LoRA").split(" ")[0], v.NETWORK_DIM ? `rank ${v.NETWORK_DIM}` : "",
    v.NETWORK_ALPHA && v.NETWORK_ALPHA !== v.NETWORK_DIM ? `alpha ${v.NETWORK_ALPHA}` : "",
    yes("MASKED_TRAINING") ? "masked" : "", v.FAMILY_EMA && !/^off/i.test(v.FAMILY_EMA) ? `EMA ${first(v.FAMILY_EMA)}` : "",
    v.CAPTION_DROPOUT && !/^off/i.test(v.CAPTION_DROPOUT) ? `caption dropout ${first(v.CAPTION_DROPOUT)}` : ""];
  const spec = el("button", {class: "rs-spec", text: S.rsSpec ? "Hide full spec" : "{ } Full spec",
    title: "Every setting this run started with", onclick: () => { S.rsSpec = !S.rsSpec; runSummary(); }});
  const setup = sec("Setup", [
    cell("Model", fam + (v.FAMILY_PRECISION ? ` · ${first(v.FAMILY_PRECISION)}` : "")),
    cell("Dataset", String(r.image_folder || "").split(/[\\/]/).filter(Boolean).pop()),
    cell("Training", [v.MAX_TRAIN_EPOCHS ? `${v.MAX_TRAIN_EPOCHS} epochs` : "", p.total_steps ? `${p.total_steps} steps` : "",
      "batch 1", v.DATASET_MEGAPIXELS ? `${v.DATASET_MEGAPIXELS} MP` : "",
      v.STAGE_MP && v.STAGE_MP !== "Off" ? `${v.STAGE_MP} MP for the first ${v.STAGE_UNTIL || "30%"}` : ""].filter(Boolean).join(" · ")),
    cell("Optimizer", [v.OPTIMIZER_TYPE, yes("ADAPTIVE_LR") ? `Adaptive LR ${v.ADAPTIVE_LR_MIN} to ${v.ADAPTIVE_LR_MAX}` :
      v.LEARNING_RATE ? `from ${v.LEARNING_RATE}` : ""].filter(Boolean).join(" · ")),
    cell("Network", net.filter(Boolean).join(" · ")),
    cell("Memory", [v.BLOCKS_SWAP ? `block swap ${first(v.BLOCKS_SWAP)}` : "",
      v.GRADIENT_ACCUMULATION && String(v.GRADIENT_ACCUMULATION) !== "1" ? `accumulation ${v.GRADIENT_ACCUMULATION}` : ""]
      .filter(Boolean).join(" · ")),
  ], spec);
  const full = S.rsSpec ? el("pre", {class: "rs-full", text: Object.keys(v).sort()
    .filter((k) => v[k] !== "" && v[k] != null && !k.startsWith("_"))
    .map((k) => `${k} = ${typeof v[k] === "object" ? JSON.stringify(v[k]) : v[k]}`).join("\n")}) : null;
  box.replaceChildren(el("div", {class: "card-title"}, el("span", {text: `Run summary: ${r.lora}`})),
    el("div", {class: "card-content rs-body"}, time, learning, setup, ...(full ? [full] : [])));
  box.style.display = "";
}

// the sidebar run card's loss line: the run card's own loss, smoothed, about 60 points
function sideSpark(d) {
  const box = $("sideRunSpark");
  if (!box) return;
  const pts = (d && d.points || []).map((p) => p[2]).filter((v) => isFinite(v));
  if (pts.length < 3) { box.replaceChildren(); box.style.display = "none"; return; }
  const sm = [];
  pts.forEach((v, i) => sm.push(i ? sm[i - 1] * 0.95 + v * 0.05 : v));
  const step = Math.max(1, Math.floor(sm.length / 60)), ys = sm.filter((_v, i) => i % step === 0);
  const lo = Math.min(...ys), hi = Math.max(...ys), span = hi - lo || 1;
  const path = ys.map((v, i) => `${i ? "L" : "M"}${(100 * i / (ys.length - 1)).toFixed(1)},${(22 - 20 * (v - lo) / span).toFixed(1)}`).join(" ");
  const ns = "http://www.w3.org/2000/svg", svg = document.createElementNS(ns, "svg"), p = document.createElementNS(ns, "path");
  svg.setAttribute("viewBox", "0 0 100 24"); svg.setAttribute("preserveAspectRatio", "none");
  p.setAttribute("d", path);
  svg.append(p);
  box.replaceChildren(svg);
  box.style.display = "";
  box.title = `Loss, smoothed: ${ys[ys.length - 1].toFixed(4)} now`;
}

// the run's figures (the same from a web run and from a desktop run): the bar, the tiles, the sidebar
function showProgress(p, state, name, extraNote) {
  const pct = p ? p.percent : (["completed"].includes(state) ? 100 : 0);
  for (const id of ["runBar", "sideRunBar"]) $(id).style.width = pct.toFixed(1) + "%";
  for (const id of ["runBar", "sideRunBar"]) $(id).parentElement.dataset.state = state;
  $("runName").textContent = name || "";
  $("sideRunName").textContent = name || "No run yet";
  const tiles = $("runTiles");
  tiles.innerHTML = "";
  const tile = (k, v) => v ? tiles.append(el("div", {class: "run-tile"}, el("span", {text: k}), el("b", {text: v}))) : null;
  if (p) {
    tile("Epoch", p.epoch ? `${p.epoch} / ${p.total_epochs}` : "");
    tile("Step", p.total_steps ? `${p.step} / ${p.total_steps}` : "");
    tile("Speed", p.speed);
    tile("Loss", p.loss);
    tile("Time left", p.eta);
  }
  const live = ["running", "starting", "pausing"].includes(state);
  // the epoch and time left once training shows them; the stage (caching, a preview) until then
  $("sideRunText").textContent = (p && p.note) ||
    (p && live && p.total_steps ? `Epoch ${p.epoch || "?"}/${p.total_epochs || "?"} · ${Math.round(p.percent)}%` +
      (p.eta ? ` · ${p.eta} left` : "") : "") || extraNote || "";
  $("runWhen").textContent = (p && p.note) || "";
}

// the run on show: the one training, else the LoRA name in the form
const currentRunName = () => S.run.lora || S.values.LORA_NAME || "";

// what a preview is, from its name (the server reads it): "Epoch 4 · Prompt 1 · strength 0.5"
function previewLabel(it) {
  if (it.epoch == null) return it.name;
  return [`Epoch ${it.epoch}`, `Prompt ${it.idx + 1}`, it.strength != null && `strength ${it.strength}`]
    .filter(Boolean).join(" · ");
}
function previewFigure(it, prompts, eager, byEpoch) {
  // its own epoch's prompt (recorded by the trainer, so a mid-run change shows), else the run's
  const own = byEpoch && it.epoch != null && byEpoch[String(it.epoch)];
  const prompt = it.idx == null ? "" : (own && own[it.idx]) || (prompts ? prompts[it.idx] : "");
  return withMode(el("figure", {class: "gallery-item", title: it.name}, mediaFor(it, eager && eager.has(it)),
    el("figcaption", {}, el("b", {text: previewLabel(it)}), prompt ? el("span", {class: "preview-prompt", text: prompt, title: prompt}) : null,
      el("span", {class: "preview-file", text: it.name}), it.lora ? studioButton(it.lora, it.family) : null)), it);
}
// the family's line on how this preview was made (the run's record), shown over it in the lightbox
function withMode(fig, it) {
  const L = S.galleryLabels;
  if (L && it.epoch != null && (!it.run || it.run === L.run)) fig._mode = L.labels[String(it.epoch)] || "";
  return fig;
}

// prompts as rows, epochs as columns (full-strength pictures; a slider's strip is its own picture)
function previewGrid(items, prompts, eager) {
  const full = items.filter((it) => it.epoch != null && it.strength == null);
  const epochs = [...new Set(full.map((it) => it.epoch))].sort((a, b) => a - b);
  const idxs = [...new Set(full.map((it) => it.idx))].sort((a, b) => a - b);
  const at = new Map(full.map((it) => [it.epoch + ":" + it.idx, it]));
  const table = el("table", {class: "prompt-grid"});
  table.append(el("tr", {}, el("th", {}), ...epochs.map((e) => el("th", {text: `Epoch ${e}`}))));
  for (const i of idxs) {
    table.append(el("tr", {},
      el("th", {class: "prompt-head"}, el("b", {text: `Prompt ${i + 1}`}),
        prompts[i] ? el("span", {class: "preview-prompt", text: prompts[i], title: prompts[i]}) : null),
      ...epochs.map((e) => { const it = at.get(e + ":" + i);
        return el("td", {}, it ? withMode(el("figure", {class: "gallery-item", title: it.name}, mediaFor(it, eager && eager.has(it)),
          el("figcaption", {class: "visually-hidden"}, el("b", {text: previewLabel(it)}),
            prompts[i] ? el("span", {text: " · " + prompts[i]}) : null)), it) : null); })));
  }
  // a second scroll bar over the grid, in step with the one under it (a long grid puts that one far down the page)
  const wrap = el("div", {class: "gallery prompt-grid-wrap"}, table);
  const spacer = el("div", {class: "prompt-grid-spacer"});
  const top = el("div", {class: "prompt-grid-top"}, spacer);
  const follow = (from, to) => from.addEventListener("scroll", () => {    // the echo finds them level and stops
    if (to.scrollLeft !== from.scrollLeft) to.scrollLeft = from.scrollLeft; });
  follow(top, wrap); follow(wrap, top);
  new ResizeObserver(() => {
    spacer.style.width = table.scrollWidth + "px";
    top.hidden = table.scrollWidth <= wrap.clientWidth;
  }).observe(table);
  return el("div", {}, top, wrap);
}

async function loadGallery() {
  const out = S.run.output_dir || S.values.LORA_OUTPUT_DIR || S.outputDefault;   // an empty box = the default folder
  const host = $("gallery");
  if (!out) { host.innerHTML = ""; $("galleryControls").style.display = "none"; return; }
  const sep = out.includes("\\") ? "\\" : "/";
  const dir = out + sep + "sample";
  // the run picked here stays picked until another run starts
  const cur = currentRunName();
  if (S.galleryFor !== cur) { S.galleryRun = ""; S.galleryFor = cur; }
  const want = S.galleryRun || cur;
  const get = (run) => api(`/api/samples?dir=${encodeURIComponent(dir)}&run=${encodeURIComponent(run)}`)
    .catch(() => ({items: [], runs: [], prompts: []}));
  let g = await get(want);
  let run = want;
  if (!g.items.length && g.runs.length && !g.runs.includes(want)) { run = g.runs[0]; g = await get(run); }
  previewMode(g.preview_mode);
  S.galleryLabels = {run, labels: g.preview_labels || {}};        // the lightbox's line per epoch (previewFigure)
  const sel = $("galleryRun");
  sel.innerHTML = "";
  // "this run" while it trains; once it has ended it is the last run
  const R = S.run || {}, liveNow = !!(R.running || ["starting", "running", "pausing"].includes(R.state));
  for (const r of g.runs) sel.append(el("option", {value: r, text: r + (r === cur ? (liveNow ? " (this run)" : " (last run)") : "")}));
  sel.value = run;
  $("galleryControls").style.display = g.runs.length ? "" : "none";
  const view = S.galleryView || "list";
  document.querySelectorAll("#galleryView button").forEach((b) => b.classList.toggle("active", b.dataset.view === view));
  // redrawn only when something changed (a new preview, another run or view), and never under a clip or song that
  // is playing - the 15 s refresh used to rebuild the gallery and stop it
  const sig = [run, view, training(), ...g.items.map((it) => it.name + "@" + it.time)].join("|");
  const playing = [...host.querySelectorAll("video, audio")].some((m) => !m.paused);
  if (S.gal && S.gal.sig === sig || (playing && S.gal && S.gal.run === run)) {
    likenessShow(dir, g.items, run, view); styleShow(dir, g.items); return; }
  host.innerHTML = "";
  host.classList.toggle("gallery-as-grid", view === "grid");
  // the newest epoch's clips download at once (the ones you'll play); the rest when the pointer comes to them
  const epochs = g.items.map((it) => it.epoch).filter((e) => e != null);
  const top = epochs.length ? Math.max(...epochs) : null;
  const eager = new Set((top != null ? g.items.filter((it) => it.epoch === top) : g.items.slice(0, 3)).slice(0, 8));
  if (view === "grid" && g.items.length) host.append(previewGrid(g.items, g.prompts || [], eager));
  else for (const it of g.items.slice(0, 120))
    host.append(previewFigure(withLora(it, g.loras, S.fam && S.fam.key), g.prompts || [], eager, g.epoch_prompts));
  S.gal = {items: g.items, run, sig};
  likenessShow(dir, g.items, run, view);
  styleShow(dir, g.items);
  const others = g.runs.filter((r) => r !== run).length;
  $("galleryInfo").textContent = g.items.length ?
    `${g.items.length} preview${g.items.length === 1 ? "" : "s"} of ${run} in ${dir}` +
    (others ? ` · ${others} other run${others === 1 ? "" : "s"} have previews here too (pick one above)` : "") :
    "No previews yet. They appear here as the run makes them.";
}

// ---- likeness to the subject (fizgig/web/gallery_likeness.py; the desktop gallery's likeness.json) ------------
S.lik = {dir: "", picked: []};
const LIK_PICS = /\.(png|jpe?g|webp)$/i;
// ArcFace never reaches 100%: two photos of the same person score about 72% against each other. Shown wherever ArcFace
// scores are (a scorer label that isn't ArcFace's means another embedder, on its own scale; none recorded = ArcFace).
const ARC_PEAK = "Peak ArcFace is about 72%: that's what two photos of the same person score against each other, " +
  "so no picture reaches 100%.";
function arcPeak(label) { return !label || String(label).startsWith("ArcFace") ? ARC_PEAK : ""; }

// the gallery's Style score (web/gallery_style.py): each preview against the training pictures' look as a whole
async function styleShow(dir, items) {
  let d;
  try { d = await api(`/api/style?dir=${encodeURIComponent(dir)}&dataset=${encodeURIComponent(S.imageFolder || "")}`); }
  catch (e) { return; }
  S.sty = {dir, on: d.on};
  $("styBtn").classList.toggle("active", d.on);
  $("styBtn").textContent = d.on ? "🎨 Style score: on" : "🎨 Style score";
  $("styStatus").textContent = d.on && d.status ? "🎨 " + d.status : "";
  for (const fig of document.querySelectorAll("#gallery .gallery-item")) {
    fig.querySelector(".sty-badge")?.remove();
    if (!d.on || !(fig.title in d.scores)) continue;
    const s = d.scores[fig.title];
    fig.append(el("span", {class: `sty-badge ${s == null ? "na" : s >= 0.7 ? "good" : s >= 0.4 ? "mid" : "bad"}`,
      title: "Style score: how close this preview is to the training pictures' look", text: s == null ? "style ?" :
      `style ${Math.round(s * 100)}%`}));
  }
}

async function likenessShow(dir, items, run, view) {
  S.lik.dir = dir;
  const ask = S.lik.ask = (S.lik.ask || 0) + 1;     // an older refresh's answer arriving late changes nothing
  let d;
  try { d = await api(`/api/likeness?dir=${encodeURIComponent(dir)}&dataset=${encodeURIComponent(S.imageFolder || "")}`); }
  catch (e) { return; }
  if (ask !== S.lik.ask) return;
  S.lik.data = d;
  $("likBtn").style.display = d.available ? "" : "none";
  const on = d.baselines.length === 3;
  $("likStatus").textContent = on ? "🎯 " + (d.status || "") : "";
  const pics = items.filter((it) => LIK_PICS.test(it.name));
  // the trend: the shown run's epochs, each the mean of its previews' scores
  const by = {};
  if (on) for (const it of pics) { const s = d.scores[it.name]; if (typeof s === "number" && it.epoch != null) (by[it.epoch] = by[it.epoch] || []).push(s); }
  const epochs = Object.keys(by).map(Number).sort((a, b) => a - b);
  const avgs = epochs.map((e) => by[e].reduce((a, b) => a + b, 0) / by[e].length);
  let best = 0;
  avgs.forEach((v, i) => { if (v > avgs[best]) best = i; });
  const bestEpoch = epochs.length ? epochs[best] : null;
  if (run === (S.run && S.run.lora)) S.likeSummary = epochs.length ?      // the run summary's Likeness
    {now: avgs[avgs.length - 1], epoch: epochs[epochs.length - 1], best: avgs[best], bestEpoch} : null;
  // a badge on every scored picture (the best epoch's outlined)
  for (const fig of document.querySelectorAll("#gallery .gallery-item")) {
    fig.querySelector(".lik-badge")?.remove();
    const it = pics.find((p) => p.name === fig.title);
    if (!on || !it) continue;
    const s = d.scores[it.name];
    const t = d.thresholds || {match: 0.45, borderline: 0.30};     // the embedder's own bands (ArcFace / DINOv2 / CLIP)
    const cls = s === undefined || s === null ? "na" : s >= t.match ? "good" : s >= t.borderline ? "mid" : "bad";
    fig.append(el("span", {class: `lik-badge ${cls}${it.epoch === bestEpoch && bestEpoch != null ? " best" : ""}`,
      text: s === undefined ? "…" : s === null ? "no face" : `${Math.round(s * 100)}%`}));
  }
  $("likPanel").style.display = epochs.length ? "" : "none";
  if (!epochs.length) return;
  $("likPeak").textContent = arcPeak(d.embedder);
  $("likRun").textContent = `${run} - best so far: epoch ${bestEpoch} (${Math.round(avgs[best] * 100)}% average likeness)`;
  const cv = $("likChart");
  const w = cv.clientWidth || 900;
  if (cv.width !== w) cv.width = w;
  const g = cv.getContext("2d"), W = cv.width, H = cv.height, L = 42, R = 12, T = 12, B = 22;
  g.clearRect(0, 0, W, H);
  const ymax = Math.min(1, Math.max(0.7, (d.thresholds || {}).match + 0.15 || 0, Math.max(...avgs) + 0.05));
  const x = (i) => epochs.length === 1 ? L + (W - L - R) / 2 : L + (W - L - R) * i / (epochs.length - 1);
  const y = (v) => T + (H - T - B) * (1 - v / ymax);
  g.font = "11px sans-serif"; g.lineWidth = 1;
  const bands = d.thresholds || {match: 0.45, borderline: 0.30};
  for (const band of [bands.borderline, bands.match]) {         // the badge colour bands, for orientation
    g.strokeStyle = themeColor("--border"); g.fillStyle = themeColor("--text-muted");
    g.beginPath(); g.moveTo(L, y(band)); g.lineTo(W - R, y(band)); g.stroke();
    g.fillText(Math.round(band * 100) + "%", 8, y(band) + 4);
  }
  g.strokeStyle = themeColor("--accent"); g.lineWidth = 2; g.beginPath();
  avgs.forEach((v, i) => (i ? g.lineTo(x(i), y(v)) : g.moveTo(x(i), y(v))));
  g.stroke();
  const every = Math.max(1, Math.ceil(epochs.length / 40));
  avgs.forEach((v, i) => {
    g.fillStyle = themeColor(i === best ? "--success" : "--accent");
    g.beginPath(); g.arc(x(i), y(v), i === best ? 5 : 3.5, 0, Math.PI * 2); g.fill();
    if (i % every === 0 || i === best) { g.fillStyle = themeColor("--text-secondary"); g.fillText(epochs[i], x(i) - 6, H - 6); }
  });
}
async function likenessPicker() {
  S.lik.mode = "gallery";
  await likGridShow(S.imageFolder || "", (S.lik.data && S.lik.data.baselines) || []);
}
// the 3-photo picker, for the Samples gallery (exactly 3, from the training images) or LoRA Royale (1-3, any folder)
async function likGridShow(folder, picked) {
  const royale = S.lik.mode === "royale";
  S.lik.folder = folder; S.lik.paths = {};
  $("likTitle").textContent = `Likeness: pick ${royale ? "up to " : ""}3 photos of your subject`;
  $("likFrom").textContent = royale ? "this folder" : "the training images";
  $("likFolder").textContent = folder || "no training image folder set on the Start tab";
  $("likFolderPick").style.display = royale ? "" : "none";
  $("likClear").style.display = royale ? "none" : "";
  $("likStart").textContent = royale ? "Use these" : "Start scoring";
  S.lik.picked = [...picked];
  const grid = $("likGrid");
  grid.innerHTML = "";
  $("likModal").classList.add("active");
  if (!folder) return;
  const r = await api(`/api/captions?folder=${encodeURIComponent(folder)}&ext=${encodeURIComponent(capExt())}&family=${encodeURIComponent(S.family || "")}`).catch(() => ({items: []}));
  for (const it of r.items.filter((x) => LIK_PICS.test(x.name))) {
    S.lik.paths[it.name] = it.path;
    const fig = el("figure", {title: it.name, onclick: () => {
      const i = S.lik.picked.indexOf(it.name);
      if (i >= 0) S.lik.picked.splice(i, 1);
      else { if (S.lik.picked.length >= 3) S.lik.picked.shift(); S.lik.picked.push(it.name); }
      likenessMarks();
    }}, el("img", {loading: "lazy", alt: "", src: `/api/thumb?path=${encodeURIComponent(it.path)}&t=${encodeURIComponent(TOKEN)}`}));
    fig.dataset.name = it.name;
    grid.append(fig);
  }
  if (!grid.children.length) grid.append(el("div", {class: "field-hint", text: `No pictures in ${royale ? "this" : "the training image"} folder.`}));
  likenessMarks();
}
function likenessMarks() {
  for (const f of $("likGrid").querySelectorAll("figure")) {
    const i = S.lik.picked.indexOf(f.dataset.name);
    f.classList.toggle("on", i >= 0);
    f.querySelector(".lik-num")?.remove();
    if (i >= 0) f.append(el("span", {class: "lik-num", text: String(i + 1)}));
  }
  $("likStart").disabled = S.lik.mode === "royale" ? !S.lik.picked.length : S.lik.picked.length !== 3;
  $("likPickStatus").textContent = `${S.lik.picked.length}/3 selected`;
}
async function likenessSet(names) {
  $("likPickStatus").textContent = "Starting…";
  try {
    await api("/api/likeness", {dir: S.lik.dir, dataset: S.imageFolder || "", baselines: names});
    $("likModal").classList.remove("active");
    loadGallery();
  } catch (e) { $("likPickStatus").textContent = e.message; }
}

// ---- the Training Run Visualiser: one prompt's previews epoch by epoch (the desktop gallery's) -----------------
S.rv = {frames: [], i: 0, timer: null, dir: 1};
const rvSrc = (p) => `/api/media?path=${encodeURIComponent(p)}&t=${encodeURIComponent(TOKEN)}`;
function rvPictures() {
  // the shown run's full-strength pictures (a slider's strip and a clip aren't one frame of the run)
  return ((S.gal && S.gal.items) || []).filter((it) => it.epoch != null && it.strength == null && LIK_PICS.test(it.name));
}
function rvOpen() {
  const pics = rvPictures();
  if (!pics.length) { $("galleryInfo").textContent = "No preview pictures in this run yet."; return; }
  const slots = [...new Set(pics.map((it) => it.idx ?? 0))].sort((a, b) => a - b);
  const keep = $("rvSlot").value;
  $("rvSlot").replaceChildren(...slots.map((s) => el("option", {value: s, text: String(s + 1)})));
  if (slots.map(String).includes(keep)) $("rvSlot").value = keep;
  $("rvRun").textContent = S.gal.run || "";
  $("rvModal").classList.add("active");
  rvBuild();
}
function rvBuild() {
  const slot = Number($("rvSlot").value || 0), by = {};
  for (const it of rvPictures()) {
    if ((it.idx ?? 0) !== slot) continue;
    if (!by[it.epoch] || it.time > by[it.epoch].time) by[it.epoch] = it;    // an epoch rendered twice (a resume): the newest
  }
  const scores = (S.lik.data && S.lik.data.baselines.length === 3 && S.lik.data.scores) || {};
  S.rv.frames = Object.keys(by).map(Number).sort((a, b) => a - b).map((e) => ({epoch: e, path: by[e].path, sim: scores[by[e].name]}));
  $("rvSlider").max = Math.max(0, S.rv.frames.length - 1);
  rvShow(S.rv.frames.length - 1);                // the newest epoch
}
function rvShow(i) {
  const F = S.rv.frames;
  if (!F.length) return;
  S.rv.i = Math.max(0, Math.min(i, F.length - 1));
  const fr = F[S.rv.i];
  $("rvImg").src = rvSrc(fr.path);
  $("rvSlider").value = S.rv.i;
  $("rvEpoch").textContent = `Epoch ${fr.epoch}` + (typeof fr.sim === "number" ? `  ·  likeness ${Math.round(fr.sim * 100)}%` : "");
}
function rvStop() { clearTimeout(S.rv.timer); S.rv.timer = null; $("rvPlay").textContent = "▶ Play"; }
function rvPlay() {
  if (S.rv.timer) return rvStop();
  if (S.rv.frames.length < 2) return;
  S.rv.dir = 1;
  $("rvPlay").textContent = "⏸ Pause";
  const tick = () => {
    let next = S.rv.i + S.rv.dir;
    if ($("rvPing").checked) { if (next >= S.rv.frames.length || next < 0) { S.rv.dir = -S.rv.dir; next = S.rv.i + S.rv.dir; } }
    else if (next >= S.rv.frames.length) next = 0;
    rvShow(next);
    S.rv.timer = setTimeout(tick, Number($("rvSpeed").value));
  };
  S.rv.timer = setTimeout(tick, Number($("rvSpeed").value));
}
function rvDraw(g, im, fr, W, H) {
  // the video's own frames (shown here and exported): fixed colours, not the theme's
  g.fillStyle = "#000"; g.fillRect(0, 0, W, H);
  const sc = Math.min(W / im.naturalWidth, H / im.naturalHeight), dw = im.naturalWidth * sc, dh = im.naturalHeight * sc;
  g.drawImage(im, (W - dw) / 2, (H - dh) / 2, dw, dh);
  if ($("rvTicker").checked) {
    const label = "Epoch " + fr.epoch;
    g.font = "bold 26px sans-serif";
    const tw = g.measureText(label).width;
    g.fillStyle = "rgba(0,0,0,0.65)"; g.fillRect(16, H - 56, tw + 24, 40);
    g.fillStyle = "#fff"; g.fillText(label, 28, H - 28);
  }
  if ($("rvTag").checked) {                      // scaled with the frame, so it reads at full size too
    const fs = Math.max(28, Math.round(H * 0.055));
    g.font = `bold ${fs}px sans-serif`;
    const tw = g.measureText("Fizgig").width;
    g.fillStyle = "rgba(255,255,255,0.7)"; g.fillText("Fizgig", W - tw - 20, H - 20);
  }
}
const rvLoad = (fr) => new Promise((res) => { const im = new Image(); im.onload = () => res({fr, im}); im.onerror = () => res(null); im.src = rvSrc(fr.path); });
function rvDownload(blob, name) {
  const a = el("a", {href: URL.createObjectURL(blob), download: name});
  document.body.append(a); a.click(); a.remove();
  setTimeout(() => URL.revokeObjectURL(a.href), 5000);
}
async function rvClip() {
  if (S.rv.frames.length < 2) { $("rvStatus").textContent = "A clip needs at least two epochs."; return; }
  rvStop();
  $("rvStatus").textContent = "Preparing frames…";
  const loaded = (await Promise.all(S.rv.frames.map(rvLoad))).filter(Boolean);
  if (loaded.length < 2) { $("rvStatus").textContent = "Couldn't load the frames."; return; }
  const seq = $("rvPing").checked ? loaded.concat([...loaded].reverse().slice(1, -1)) : loaded;
  const first = loaded[0].im, W = Math.min(1024, first.naturalWidth), H = Math.round(W * first.naturalHeight / first.naturalWidth);
  const cv = el("canvas", {width: W, height: H}), g = cv.getContext("2d");
  let mime = "video/webm;codecs=vp9";
  if (!MediaRecorder.isTypeSupported(mime)) mime = "video/webm";
  const rec = new MediaRecorder(cv.captureStream(30), {mimeType: mime, videoBitsPerSecond: 8000000});
  const chunks = [];
  rec.ondataavailable = (e) => { if (e.data.size) chunks.push(e.data); };
  const done = new Promise((r) => { rec.onstop = r; });
  const step = Number($("rvSpeed").value);
  rec.start();
  for (let i = 0; i < seq.length; i++) {
    rvDraw(g, seq[i].im, seq[i].fr, W, H);
    $("rvStatus").textContent = `Recording… ${i + 1}/${seq.length}`;
    await new Promise((r) => setTimeout(r, step));
  }
  await new Promise((r) => setTimeout(r, 200));   // so the last frame lands
  rec.stop(); await done;
  const name = `${S.gal.run || "run"}_training_run.webm`;
  rvDownload(new Blob(chunks, {type: "video/webm"}), name);
  $("rvStatus").textContent = `Saved ${name} (${seq.length} frames). LoRA Royale saves MP4 and GIF.`;
}
async function rvFrame() {
  const fr = S.rv.frames[S.rv.i];
  if (!fr) return;
  const got = await rvLoad(fr);
  if (!got) return;
  const cv = el("canvas", {width: got.im.naturalWidth, height: got.im.naturalHeight});
  rvDraw(cv.getContext("2d"), got.im, fr, cv.width, cv.height);
  cv.toBlob((b) => rvDownload(b, `${S.gal.run || "run"}_epoch${fr.epoch}.png`), "image/png");
}

async function loadCheckpoints() {
  const out = S.run.output_dir || S.values.LORA_OUTPUT_DIR;
  const host = $("checkpoints");
  if (!out) { host.innerHTML = ""; return; }
  const r = await api("/api/checkpoints?dir=" + encodeURIComponent(out)).catch(() => ({items: []}));
  host.innerHTML = "";
  const size = (b) => b >= 2 ** 30 ? (b / 2 ** 30).toFixed(1) + " GB" : (b / 2 ** 20).toFixed(0) + " MB";
  const run = currentRunName();
  const mine = r.items.filter((it) => it.run === run);
  const shown = S.ckptAll ? r.items : mine;
  for (const it of shown) {
    const href = `/api/download?path=${encodeURIComponent(it.path)}&t=${encodeURIComponent(TOKEN)}`;
    host.append(el("div", {class: "checkpoint-row"},
      el("span", {class: "checkpoint-name", text: it.name}),
      el("span", {class: "field-hint", text: `${it.kind === "finetune" ? "Fine-tune (whole model)" : "LoRA"} · ` +
        `${size(it.size)} · ${new Date(it.time * 1000).toLocaleString()}`}),
      it.kind === "finetune" ? null : studioButton(it.path, S.fam && S.fam.key),
      el("a", {class: "btn btn-sm btn-primary", href, download: it.name, text: "Download"})));
  }
  const others = r.items.length - mine.length;
  $("checkpointsInfo").replaceChildren(
    document.createTextNode(S.ckptAll ? `All ${r.items.length} saved in ${out}, newest first. ` :
      mine.length ? `${mine.length} saved by ${run} in ${out}, newest first. ` :
      `Nothing saved by ${run || "this run"} yet: checkpoints appear here as the run saves them. `),
    others ? el("a", {href: "#", class: "inline-link", text: S.ckptAll ? `Only ${run}'s` :
      `Show the ${others} from other runs in this folder`,
      onclick: (e) => { e.preventDefault(); S.ckptAll = !S.ckptAll; loadCheckpoints(); }}) : document.createTextNode(""));
}

// the Previews so far card: what this run's previews render with (recorded by the trainer when the run started)
function previewMode(m) {
  const box = $("galleryMode");
  if (!box) return;
  box.className = "preview-mode" + (m && m.turbo ? " turbo" : "");
  if (!m) { box.style.display = "none"; return; }
  const how = `${m.steps} steps, CFG ${Number(m.cfg).toFixed(1).replace(/\.0$/, "")}`;
  box.textContent = m.turbo
    ? `⚡ Turbo LoRA in use: ${m.turbo} at strength ${m.strength}, ${how}. ${m.turbo_note ? m.turbo_note + " " : ""}` +
      "For full-quality previews, set Turbo " +
      "LoRA strength to 0 above (Steps and CFG go back with it); tick Change the previews of the run going now " +
      "for it to apply from this run's next preview."
    : `Previews render on ${m.model || "the model being trained"}: ${how}, no Turbo LoRA.`;
  box.style.display = "";
}

// ---- Start tab: the News card (fizgig/web/news.py, newest first) ----------------------------------------------
function newsCard() {
  const host = $("newsCard");
  const items = (S.schema.news || []).slice(0, 3);
  host.style.display = items.length ? "" : "none";
  const body = $("newsBody");
  body.innerHTML = "";
  for (const it of items) {
    const item = el("div", {class: "news-item"},
      el("div", {class: "news-head"}, el("b", {text: it.title}), el("span", {class: "field-hint", text: " · " + it.date})));
    for (const para of String(it.body || "").split("\n\n")) item.append(el("p", {class: "news-text", text: para}));
    if (it.links && it.links.length) item.append(el("div", {class: "news-links"}, ...it.links.map((l) =>
      el("a", {class: "inline-link", target: "_blank", rel: "noopener", href: l.href, text: l.text + " →"}))));
    body.append(item);
  }
}

// ---- Start tab: your training data (the folders in the training data folder) ---------------------------------
async function loadDatasets() {
  try { S.datasets = await api("/api/datasets"); } catch (e) { return; }
  $("datasetsDesc").textContent = `A shortcut: the folders in ${S.datasets.root}, and the ones you've picked from ` +
    "anywhere else (outlined). Click one to make it the training folder above. For a folder that isn't listed " +
    "(another drive, or a subfolder), press Choose… above; to add a new one, upload it below.";
  $("datasetsSearchRow").style.display = S.datasets.total > 8 ? "" : "none";
  renderDatasets();
}

// a dataset's pictures in the viewer, stepping through them all
async function datasetPictures(d) {
  const r = await api(`/api/captions?folder=${encodeURIComponent(d.path)}&ext=${encodeURIComponent(capExt())}&family=${encodeURIComponent(S.family || "")}`)
    .catch(() => ({items: []}));
  const q = (p) => `/api/media?path=${encodeURIComponent(p)}&t=${encodeURIComponent(TOKEN)}`;
  const pics = r.items.filter((it) => /\.(png|jpe?g|webp|bmp|avif|gif)$/i.test(it.name))
    .map((it) => ({src: q(it.path), name: it.name + (it.caption ? " · " + it.caption : "")}));
  if (!pics.length) return;
  S.lb.list = pics;
  lightboxShow(0);
}

// Image Prep and Captions work on the one folder: a training folder with sub-folders says how to reach theirs
function subfoldersHint() {
  const norm = (p) => (p || "").replace(/\\/g, "/").replace(/\/+$/, "").toLowerCase();
  const d = ((S.datasets || {}).datasets || []).find((x) => norm(x.path) === norm(S.imageFolder));
  return d && d.sub_folders ? ` · This works on this folder only, not its ${d.sub_folders} sub-folder` +
    `${d.sub_folders === 1 ? "" : "s"}: to caption or mask one, choose it as the training folder on the Start tab.` : "";
}

function renderDatasets() {
  const r = S.datasets;
  if (!r) return;
  const q = $("datasetsSearch").value.trim().toLowerCase();
  const host = $("datasetsGrid");
  host.innerHTML = "";
  const norm = (p) => (p || "").replace(/\\/g, "/").replace(/\/+$/, "").toLowerCase();
  const list = r.datasets.filter((d) => !q || d.name.toLowerCase().includes(q));
  const shown = q || S.datasetsAll ? list.slice(0, 300) : list.slice(0, 12);     // the newest twelve, unless asked
  for (const d of shown) {
    const what = [d.pictures && `${d.pictures} picture${d.pictures === 1 ? "" : "s"}`,
      d.clips && `${d.clips} clip${d.clips === 1 ? "" : "s"}`, d.songs && `${d.songs} song${d.songs === 1 ? "" : "s"}`]
      .filter(Boolean).join(" · ");
    const total = d.pictures + d.clips + d.songs;
    const caps = d.captioned >= total ? "all captioned" : d.captioned ? `${d.captioned} of ${total} captioned` : "no captions yet";
    const mosaic = el("div", {class: "dataset-mosaic n" + Math.min(4, d.thumbs.length)},
      d.thumbs.length ? d.thumbs.map((p) => el("img", {loading: "lazy", alt: "",
        src: `/api/thumb?path=${encodeURIComponent(p)}&t=${encodeURIComponent(TOKEN)}`})) :
        el("span", {class: "dataset-kind", text: d.songs ? "♪" : d.clips ? "▶" : ""}));
    const choose = () => {
      if (norm(d.path) !== norm(S.imageFolder)) {
        $("imageFolder").value = d.path;
        $("imageFolder").dispatchEvent(new Event("change"));
      }
      renderDatasets();
    };
    const act = (text, title, fn) => el("button", {class: "btn btn-sm", text, title, onclick: (e) => {
      e.stopPropagation(); fn(); }});
    // the tile chooses the folder; its buttons also open it in Captions, or show its pictures big
    const card = el("div", {class: "dataset-card" + (norm(d.path) === norm(S.imageFolder) ? " active" : "") +
      (d.elsewhere ? " elsewhere" : ""), title: d.path, role: "button", tabindex: "0", onclick: choose,
      onkeydown: (e) => { if (e.key === "Enter" || e.key === " ") { e.preventDefault(); choose(); } }}, mosaic,
      el("div", {class: "dataset-meta"}, el("b", {text: d.name}),
        d.elsewhere ? el("span", {class: "dataset-where", text: d.path, title: "A folder you picked, outside the training data folder"}) : null,
        el("span", {text: what}),
        d.sub_items ? el("span", {class: "dataset-subs", text: `+ ${d.sub_items} in ${d.sub_folders} sub-folder` +
          (d.sub_folders === 1 ? "" : "s"), title: "Trained on only when Also train on its sub-folders is ticked on " +
          "the Training tab"}) : null,
        total ? el("span", {class: d.captioned >= total ? "ok" : "warn", text: caps}) : null,
        el("span", {class: "dataset-when", text: d.time ? new Date(d.time * 1000).toLocaleDateString() : ""}),
        el("div", {class: "dataset-actions"},
          act("Captions", "Train on this folder and open it on the Captions tab", () => { choose(); showTab("captions"); }),
          d.pictures ? act("Pictures", "Look through its pictures, big", () => datasetPictures(d)) : null,
          d.elsewhere && norm(d.path) !== norm(S.imageFolder) ? act("✕", "Take it off this list (the folder and its files stay)",
            () => api("/api/datasets/forget", {path: d.path}).then(loadDatasets)) : null)));
    host.append(card);
  }
  if (shown.length < list.length) host.append(el("button", {class: "btn dataset-more", text: `Show all ${list.length}`,
    onclick: () => { S.datasetsAll = true; renderDatasets(); }}));
  if (!list.length) host.append(el("div", {class: "field-hint datasets-empty", text: r.total ?
    "No folder matches that." : "No training data yet. Upload a folder below, or pick one with Choose… above."}));
  // nothing in Fizgig's own training_data folder: your datasets may be in a folder of your own - one press points
  // Fizgig at it (Preferences' Training data folder), and its folders are listed here from then on
  if (!r.total && !S.onPod && !(S.uploadRoot && S.uploadRoot.set)) {
    const use = async (p) => {
      try { await api("/api/prefs", {changes: {input_dataset_dir: p}}); showUploadRoot(); loadDatasets(); }
      catch (e) { host.append(el("div", {class: "field-hint warn", text: e.message})); }
    };
    host.append(el("div", {class: "datasets-elsewhere-hint"},
      el("span", {text: "Your datasets are in another folder? Point Fizgig at it and its folders show here: "}),
      el("button", {class: "btn btn-sm btn-primary", text: "Use that folder…",
        title: "Pick the folder that holds your training folders (it becomes Preferences > Training data folder)",
        onclick: () => pick("", true, use)})));
  }
}

// ---- Runs & outputs: the run history (fizgig/web/history.py) -----------------------------------------------
const RESULT_STATE = {running: "running", finished: "completed", failed: "failed", stopped: "stopped",
                      paused: "paused", unknown: "idle"};
const RESULT_LABEL = {running: "Running", finished: "Finished", failed: "Failed", stopped: "Stopped", paused: "Paused",
                      unknown: "Ended"};

async function loadRuns() {
  let r;
  try { r = await api("/api/runs"); } catch (e) { return; }
  const host = $("runsList");
  host.innerHTML = "";
  const famName = (k) => ((S.schema.families.find((f) => f.key === k) || {}).display_name) || k;
  const dur = (a, b) => { const m = Math.round(((b || Date.now() / 1000) - a) / 60);
    return m < 60 ? `${m} min` : `${Math.floor(m / 60)} h ${m % 60} min`; };
  for (const run of r.runs) {
    const p = run.progress;
    const figs = p ? [p.epoch && `epoch ${p.epoch}/${p.total_epochs}`, p.loss && `loss ${p.loss}`].filter(Boolean).join(" · ") : "";
    const files = el("div", {class: "run-files", style: "display:none"});
    const showFiles = el("button", {class: "btn btn-sm", text: "Previews & checkpoints", onclick: async () => {
      if (files.style.display === "none") {
        files.style.display = "";
        files.textContent = "Loading…";
        files.replaceChildren(...await runFiles(run));
      } else files.style.display = "none";
    }});
    const again = el("button", {class: "btn btn-sm btn-primary", text: "Start again with these settings",
      title: "Loads this run's model, training folder and settings into the form (nothing starts until you press Start)",
      onclick: async () => {
        try {
          await loadIntoForm(await api("/api/runs/settings", {id: run.id}));
          $("runStatus").textContent = `Loaded the settings of ${run.lora || "that run"}: check them, then Start.`;
        } catch (e) { alertRow.textContent = e.message; }
      }});
    // Continue: on from the run's newest save, with its own settings (only when that save is still there); a run that
    // trained all it was set to asks how far to train first
    const contBox = el("div", {class: "field-row", style: "display:none"});
    const cp = run.continue;
    let contGoing = false;
    const contGo = async (more) => {
      if (contGoing) return;
      contGoing = true;
      alertRow.textContent = "";
      try {
        const x = await api("/api/runs/continue", {id: run.id, more: more || ""});
        if (x.need) { contAsk(x.error); return; }
        if (x.problems) { alertRow.textContent = x.problems.join(" "); return; }
        contBox.style.display = "none";
        if (x.form) await loadIntoForm(x.form);   // the tab shows the run going on, not the settings before it
        alertRow.textContent = x.queued ? "A run is training now: this one is on the queue, and continues when its " +
          "turn comes." : `Continuing ${run.lora || "the run"}: it's on the Training tab.`;
        pollRun();
        if (!x.queued) {                    // to the run card itself, as the sidebar's run opens it
          showTab("training");
          setTimeout(() => document.querySelector(".run-card").scrollIntoView({behavior: "smooth", block: "start"}), 50);
        } else loadRuns();
      } catch (e) { alertRow.textContent = e.message; } finally { contGoing = false; }
    };
    const contAsk = (why) => {
      const box = el("input", {type: "number", min: String(cp.done + 1), step: "1", class: "w-num",
        value: String(cp.done + (cp.unit === "epoch" ? Math.max(1, Math.round(cp.done / 2)) : 1))});
      contBox.replaceChildren(el("span", {class: "field-hint", text: why + " Train to:"}), box,
        el("span", {class: "field-hint", text: cp.unit + "s"}),
        el("button", {class: "btn btn-sm btn-primary", text: "Continue", onclick: () => {
          const n = parseInt(box.value, 10);
          if (!(n > cp.done)) { alertRow.textContent = `More than ${cp.done}, please.`; return; }
          contGo(String(n));
        }}),
        el("button", {class: "btn btn-sm", text: "Cancel", onclick: () => { contBox.style.display = "none"; }}));
      contBox.style.display = "";
    };
    const cont = cp ? el("button", {class: "btn btn-sm btn-primary",
      text: cp.done >= cp.of ? "Continue…" : "Continue",
      title: `Trains on from its newest save (${cp.unit} ${cp.done}${cp.of ? " of " + cp.of : ""}), with this run's own ` +
        "settings, into this same run" + (cp.done >= cp.of ? `. It has trained all its ${cp.unit}s, so you choose how ` +
        "far to take it first" : ""),
      onclick: () => {
        if (cp.done >= cp.of) {
          if (contBox.style.display !== "none") { contBox.style.display = "none"; return; }
          contAsk(`It has trained all ${cp.of} ${cp.unit}s.`);
        } else contGo("");
      }}) : null;
    const contHint = cp ? el("span", {class: "field-hint",
      text: `Saved at ${cp.unit} ${cp.done}${cp.of ? " of " + cp.of : ""}`}) : null;
    // Delete run: says what goes first (checkpoints, saved states, previews - never the training folder), then deletes
    const delBox = el("div", {class: "run-delete", style: "display:none"});
    const del = el("button", {class: "btn btn-sm btn-danger", text: "Delete run…", onclick: async () => {
      if (delBox.style.display !== "none") { delBox.style.display = "none"; return; }
      delBox.style.display = "";
      delBox.textContent = "Checking…";
      try {
        const w = await api("/api/runs/delete", {id: run.id});
        const gb = (b) => b >= 2 ** 30 ? (b / 2 ** 30).toFixed(1) + " GB" : (b / 2 ** 20).toFixed(0) + " MB";
        const parts = [w.checkpoints && `${w.checkpoints} checkpoint${w.checkpoints > 1 ? "s" : ""}`,
          w.states && `${w.states} saved state${w.states > 1 ? "s" : ""} (for resuming)`,
          w.previews && `${w.previews} preview${w.previews > 1 ? "s" : ""}`].filter(Boolean);
        delBox.replaceChildren(
          el("div", {class: "field-hint", text: (parts.length ?
            `Deletes ${parts.join(", ")} - ${gb(w.bytes)} from ${w.folder} - for good, and takes the run off this ` +
            "list." : "Nothing of this run is left on disk: this takes it off the list.") +
            " Your training folder and its captions are not touched."}),
          el("div", {class: "field-row"},
            el("button", {class: "btn btn-sm btn-danger", text: "Delete", onclick: async () => {
              try { await api("/api/runs/delete", {id: run.id, confirm: true}); loadRuns(); loadOutputs(); }
              catch (e) { delBox.replaceChildren(el("div", {class: "problems", text: e.message})); }
            }}),
            el("button", {class: "btn btn-sm", text: "Cancel", onclick: () => { delBox.style.display = "none"; }})));
      } catch (e) { delBox.replaceChildren(el("div", {class: "problems", text: e.message})); }
    }});
    const alertRow = el("div", {class: "field-hint"});
    // notes: what you tried and what you made of it, saved as you type
    const notesBox = el("div", {class: "run-notes", style: "display:none"});
    const notesBtn = el("button", {class: "btn btn-sm", text: run.notes ? "📝 Notes" : "Notes",
      title: "Your notes on this run", onclick: () => {
        if (notesBox.style.display !== "none") { notesBox.style.display = "none"; return; }
        if (!notesBox.firstChild) {
          const ta = el("textarea", {rows: 4, class: "w-path", placeholder: "What you tried, what you thought of it…"});
          ta.value = run.notes || "";
          const said = el("span", {class: "field-hint"});
          const save = debounce(() => api("/api/runs/notes", {id: run.id, text: ta.value})
            .then(() => { run.notes = ta.value; said.textContent = "Saved."; notesBtn.textContent = ta.value ? "📝 Notes" : "Notes"; })
            .catch((e) => { said.textContent = e.message; }), 600);
          ta.addEventListener("input", () => { said.textContent = ""; save(); });
          notesBox.append(ta, said);
        }
        notesBox.style.display = "";
      }});
    const lgBox = el("div", {style: "display:none"});
    const lgBtn = run.loss_graph ? el("button", {class: "btn btn-sm", text: "Loss graph",
      title: "The run's loss at every step: smoothing, log scale, the learning rate",
      onclick: async () => {
        if (lgBox.style.display !== "none") { lgBox.style.display = "none"; return; }
        lgBox.style.display = ""; lgBox.textContent = "Loading…";
        try { lgBox.replaceChildren(lossGraph(await api("/api/runs/loss?id=" + encodeURIComponent(run.id)))); }
        catch (e) { lgBox.replaceChildren(el("div", {class: "problems", text: e.message})); }
      }}) : null;
    // per-image loss (a family with the loss watch, when the run kept its log)
    const ilBox = el("div", {class: "il-box", style: "display:none"});
    const ilFam = S.schema.families.find((f) => f.key === run.family);
    const ilBtn = !run.image_loss && ilFam && ilFam.loss_watch !== false ? el("button", {
      class: "btn btn-sm btn-lilac is-off", text: "Per-image loss", "aria-disabled": "true",
      title: "This run didn't track it. " + IL_OFF_TIP}) :
      run.image_loss ? el("button", {class: "btn btn-sm btn-lilac", text: "Per-image loss", title: IL_ON_TIP,
      onclick: async () => {
        if (ilBox.style.display !== "none") { ilBox.style.display = "none"; return; }
        ilBox.style.display = ""; ilBox.textContent = "Loading…";
        try { imageLossPanel(ilBox, await api("/api/runs/image_loss?id=" + encodeURIComponent(run.id)), run); }
        catch (e) { ilBox.replaceChildren(el("div", {class: "problems", text: e.message})); }
      }}) : null;
    host.append(el("div", {class: "run-row"},
      el("div", {class: "run-row-head"},
        el("span", {class: "run-badge", "data-state": RESULT_STATE[run.result] || "idle", text: RESULT_LABEL[run.result] || run.result}),
        el("b", {text: run.lora || "(unnamed)"}),
        el("span", {class: "field-hint", text: famName(run.family) + (run.source === "desktop" ? " · the old desktop app" : "")})),
      el("div", {class: "field-hint", text: [run.image_folder && `Trained on ${run.image_folder}`,
        `${new Date(run.started_at * 1000).toLocaleString()} · ${dur(run.started_at, run.ended_at)}`, figs]
        .filter(Boolean).join("  ·  ")}),
      lossChart(run.curve),
      el("div", {class: "field-row"}, cont, contHint, showFiles, lgBtn, ilBtn, notesBtn, again, del), contBox, delBox,
      alertRow, notesBox, files,
      lgBox, ilBox));
  }
  if (!r.runs.length) host.append(el("div", {class: "field-hint", text: "No runs yet. Every run you start is listed here " +
    "with its results."}));
}

// ---- per-image loss: each image's residual by epoch (its loss minus the dataset's mean at the same noise level, the
// loss watch's own measure) with the watch's verdict - the desktop's Problem Images colours and words
const IL_VERDICTS = {
  excluded: ["var(--text-muted)", "Excluded from training - two AI captions couldn't fix it. Edit the caption to re-admit it, or remove it from the dataset."],
  stuck: ["var(--danger)", "Stuck - persistently hard, not improving. Review this image and its caption."],
  suspect: ["var(--warning-strong)", "Suspect - extremely hard from the start; slowed while the trend confirms. Worth a caption check now."],
  watch: ["var(--warning)", "Watching - looked stuck this epoch; needs more epochs to confirm."],
  warmup: ["var(--special)", "Look-filter outlier easing in on an LR ramp; releases early once it starts improving."],
  exhausted: ["var(--data-teal)", "Fully mined - improved a lot, then plateaued. Caption is fine; LR eased to prevent overbake."],
  learning: ["var(--data-blue)", "Learning - hard but improving. Leave it alone."],
  mid: ["var(--text-secondary)", "Normal."],
  easy: ["var(--success)", "Learned - consistently easy. Gets a gentle x1.1 boost."]};
const IL_ORDER = ["excluded", "stuck", "suspect", "watch", "warmup", "exhausted", "learning", "mid", "easy"];
const IL_PROBLEMS = new Set(["excluded", "stuck", "suspect", "watch"]);
const IL_PAGE = 48;

function imageLossPanel(box, d, run) {
  const imgs = d.images || [], rep = d.report || {};
  // one scale for every tile, so tiles compare: room for every image's recent level with a margin (a one-epoch
  // spike past it is clipped; a consistently hard image keeps its shape)
  const ymax = Math.max(1e-6, 1.25 * Math.max(...imgs.map((im) => Math.abs(im.recent)), 0));
  const counts = {};
  for (const im of imgs) counts[im.verdict || "mid"] = (counts[im.verdict || "mid"] || 0) + 1;
  const tally = IL_ORDER.filter((v) => counts[v]).map((v) => `${counts[v]} ${v}`).join(" · ");
  const head = [`${imgs.length} images · ${d.epochs.length} epochs`, rep.epoch ? `the watch's verdicts as of epoch ${rep.epoch}` : "",
    tally].filter(Boolean).join(" · ");
  const sort = el("select", {class: "w-combo"}, ...[["hard", "Hardest now first"], ["easy", "Easiest now first"],
    ["verdict", "By verdict"], ["name", "By name"]].map(([v, t]) => el("option", {value: v, text: t})));
  const show = el("select", {class: "w-combo"}, el("option", {value: "all", text: "All images"}),
    el("option", {value: "problems", text: "Worth a look (excluded, stuck, suspect, watching)"}));
  const grid = el("div", {class: "il-grid"}), pager = el("div", {class: "field-row"});
  let page = 0;
  const draw = () => {
    let list = show.value === "problems" ? imgs.filter((im) => IL_PROBLEMS.has(im.verdict)) : [...imgs];
    const o = (v) => { const i = IL_ORDER.indexOf(v || "mid"); return i < 0 ? 2 : i; };
    list.sort(sort.value === "name" ? (a, b) => a.key.localeCompare(b.key) :
      sort.value === "easy" ? (a, b) => a.recent - b.recent :
      sort.value === "verdict" ? (a, b) => o(a.verdict) - o(b.verdict) || b.recent - a.recent : (a, b) => b.recent - a.recent);
    const pages = Math.max(1, Math.ceil(list.length / IL_PAGE));
    page = Math.min(page, pages - 1);
    grid.replaceChildren(...list.slice(page * IL_PAGE, (page + 1) * IL_PAGE).map((im) => imageLossTile(im, ymax, run, d.live)));
    if (!list.length) grid.append(el("div", {class: "field-hint", text: "No image needs a look right now."}));
    pager.replaceChildren(...(pages > 1 ? [
      el("button", {class: "btn btn-sm", text: "‹ Prev", disabled: page === 0, onclick: () => { page--; draw(); }}),
      el("span", {class: "field-hint", text: `page ${page + 1} of ${pages}`}),
      el("button", {class: "btn btn-sm", text: "Next ›", disabled: page >= pages - 1, onclick: () => { page++; draw(); }})] : []));
  };
  sort.addEventListener("change", () => { page = 0; draw(); });
  show.addEventListener("change", () => { page = 0; draw(); });
  box.replaceChildren(
    el("div", {class: "field-hint", text: head}),
    rep.plateaued && rep.best_epoch_estimate ? el("div", {class: "il-plateau",
      text: `The run plateaued: the best checkpoint is likely around epoch ${rep.best_epoch_estimate} - compare a few epochs either side.`}) : "",      // (replaceChildren writes a null as the word)
    el("div", {class: "field-hint", text: "Each line is one image's loss minus the dataset's average at the same noise level, " +
      "epoch by epoch. Above the dotted line it is harder than the rest; a line falling is the image being learned. " +
      "All tiles share one scale." + (d.live ? " Edit caption fixes one while the run goes: the trainer re-encodes it at " +
      "the next epoch and that image's history starts again - it should turn blue (learning) if the fix worked." : "")}),
    el("div", {class: "field-row"}, el("span", {class: "field-label", text: "Sort:"}), sort, show), grid, pager);
  draw();
}

function imageLossTile(im, ymax, run, live) {
  const [col, say] = IL_VERDICTS[im.verdict] || IL_VERDICTS.mid;
  const W = 200, H = 56, pad = 3, pts = im.series;
  const e0 = pts[0][0], e1 = pts[pts.length - 1][0];
  const X = (e) => pad + (W - 2 * pad) * (e1 === e0 ? 0.5 : (e - e0) / (e1 - e0));
  const Y = (v) => H / 2 - (H / 2 - pad) * Math.max(-1, Math.min(1, v / ymax));
  const ns = "http://www.w3.org/2000/svg", svg = document.createElementNS(ns, "svg");
  svg.setAttribute("viewBox", `0 0 ${W} ${H}`); svg.setAttribute("class", "il-chart");
  const zero = document.createElementNS(ns, "line");
  zero.setAttribute("x1", 0); zero.setAttribute("x2", W); zero.setAttribute("y1", H / 2); zero.setAttribute("y2", H / 2);
  zero.setAttribute("class", "il-zero");
  const line = document.createElementNS(ns, "polyline");
  line.setAttribute("points", pts.map((p) => `${X(p[0]).toFixed(1)},${Y(p[1]).toFixed(1)}`).join(" "));
  line.style.stroke = col;
  const t = document.createElementNS(ns, "title");
  t.textContent = pts.map((p) => `epoch ${p[0]}: ${p[1] >= 0 ? "+" : ""}${p[1].toFixed(4)}`).join("\n");
  svg.append(zero, line, t);
  const sign = (v) => (v >= 0 ? "+" : "") + v.toFixed(4);
  return el("figure", {class: "gallery-item il-tile"},
    im.path ? el("img", {loading: "lazy", alt: im.key, src: `/api/thumb?path=${encodeURIComponent(im.path)}&t=${encodeURIComponent(TOKEN)}`}) :
      el("div", {class: "il-nopic", text: "(not in the training folder now)"}),
    el("figcaption", {},
      el("div", {class: "il-name", title: im.key, text: im.key}),
      el("span", {class: "il-verdict", style: `background:${col}`, title: say,
        text: (im.verdict || "mid") + (im.multiplier != null && im.multiplier !== 1 ? ` ×${im.multiplier}` : "")}),
      el("span", {class: "field-hint", text: ` now ${sign(im.recent)}`, title: "Mean over the last 5 epochs (the watch's window)"}),
      im.queued ? el("span", {class: "il-cap-state", text: "caption queued", title: "Waiting for the next epoch boundary, when the trainer re-encodes it"}) :
        im.applied != null ? el("span", {class: "il-cap-state done", text: `caption fixed at epoch ${im.applied}`}) : null),
    svg, run ? imageLossCaption(im, run, live) : null);
}

// one image's caption, edited from its tile (the desktop's Problem Images editor)
function imageLossCaption(im, run, live) {
  const box = el("div", {class: "il-cap", style: "display:none"});
  const btn = el("button", {class: "btn btn-sm", text: "Edit caption", onclick: async () => {
    if (box.style.display !== "none") { box.style.display = "none"; return; }
    box.style.display = "";
    box.replaceChildren(el("div", {class: "field-hint", text: "Loading…"}));
    let r;
    try { r = await api(`/api/runs/caption?id=${encodeURIComponent(run.id)}&key=${encodeURIComponent(im.key)}&ext=${encodeURIComponent(capExt())}&family=${encodeURIComponent(S.family || "")}`); }
    catch (e) { box.replaceChildren(el("div", {class: "problems", text: e.message})); return; }
    const ta = el("textarea", {rows: 5, class: "il-cap-text"});
    ta.value = r.caption;
    const say = el("div", {class: "field-hint", text: "Caption what the picture actually shows - viewpoint (from behind, side profile), pose, lighting, what's in the way."});
    const save = el("button", {class: "btn btn-sm btn-primary", text: live ? "Save and queue for re-encode" : "Save caption", onclick: async () => {
      try {
        const res = await api("/api/runs/caption", {id: run.id, key: im.key, caption: ta.value, ext: capExt()});
        say.textContent = res.queued ? "Saved and queued - the trainer re-encodes it at the next epoch." +
          (res.wrote ? "" : " (The caption file couldn't be written: the run is fixed, a later re-cache uses the old caption.)")
          : "Saved. It's used the next time this folder trains.";
      } catch (e) { say.textContent = e.message; }
    }});
    box.replaceChildren(ta, el("div", {class: "field-row"}, save), say);
    ta.focus();
  }});
  return el("div", {}, btn, box);
}

// ---- LoRA Royale: every epoch on one seed, crossfaded (fizgig/web/royale.py; renders on the GPU worker) ----
S.ry = {wired: false, info: null, items: [], gen: 0, shown: -1, scores: {}, best: null, poll: null};
const RY_KEY = "fizgig_royale";
const rySaved = () => { try { return JSON.parse(localStorage.getItem(RY_KEY) || "{}"); } catch (e) { return {}; } };
function ryRemember() {
  const v = (id) => $(id).value;
  const keep = {};                                  // the travel cards' own fields, as the desktop remembers them
  for (const id of ["ryPPrompt", "ryPCustom", "ryPSubject", "ryPPreset", "rySA", "rySB", "rySWay", "ryLA", "ryLB",
    "ryCMode", "ryCEpochs", "ryCTrigger", "ryCSeed",
    // each card's knobs, as the desktop keeps them: reference, strength, size, anchor, interpolation, frames, W x H
    "rySPreset", "rySRef", "rySRefStr", "rySMp", "rySSeq", "rySUseEpoch", "rySFrames", "rySW", "rySH",
    "ryPRef", "ryPRefStr", "ryPMp", "ryPSeq", "ryPUseEpoch", "ryPAnchor", "ryPAnchorStr", "ryPInterp", "ryPDrift",
    "ryPVary", "ryPStart", "ryPEnd", "ryPFrames", "ryPW", "ryPH", "ryLFrames", "ryLW", "ryLH", "ryCW", "ryCH"])
    if ($(id)) keep[id] = $(id).type === "checkbox" ? $(id).checked : $(id).value;
  try { localStorage.setItem(RY_KEY, JSON.stringify({tool: S.ry.tool, keep, ptDim: S.ry.ptDim, source: rySource(),
    single: v("rySingle"), cmp: v("ryCPrompts"),
    family: v("ryFamily"), folder: v("ryFolder"), prompt: v("ryPrompt"),
    seed: v("rySeed"), ref: v("ryRef"), refStr: v("ryRefStr"), w: v("ryW"), h: v("ryH"), max: v("ryMax"),
    like: S.ry.likeRefs || [], out: v("ryOut"), fmt: v("ryFmt"), speed: v("rySpeed")})); } catch (e) { /* private */ }
}
const ryLabel = (l) => (String(l).match(/^\d+$/) ? `Epoch ${l}` : String(l));

async function royaleOpen() {
  if (!S.ry.wired) {
    S.ry.wired = true;
    const sv = rySaved();
    for (const [id, k] of [["ryFolder", "folder"], ["ryPrompt", "prompt"], ["rySeed", "seed"], ["ryRef", "ref"],
      ["ryRefStr", "refStr"], ["ryOut", "out"]]) if (sv[k] != null) $(id).value = sv[k];
    S.ry.likeRefs = Array.isArray(sv.like) ? sv.like.slice(0, 3) : sv.like ? [sv.like] : [];
    ryLikeShow();
    for (const [id, k] of [["ryW", "w"], ["ryH", "h"], ["ryMax", "max"], ["ryFmt", "fmt"], ["rySpeed", "speed"]])
      if (sv[k]) $(id).value = sv[k]; else if (id === "ryW" || id === "ryH") $(id).value = "512";
    await royaleFamily(sv.family);
    wireRoyale();
    await wireRoyaleTravel();
    royaleScan();
  }
  royalePoll();
  ryTravelPoll();
}

// ---- Royale's travel modes and comparison sheet ----
const RY_MODES = {seed: "ryS", strength: "ryL", prompt: "ryP"};
S.ry.tr = {};

function ryShowTool(key) {
  // one tool at a time; the epoch tools need a run (Single LoRA starts on seed travel)
  if (key === "epochs" && rySource() === "single") key = "seed";
  S.ry.tool = key;
  document.querySelectorAll("#tab-royale [data-ry]").forEach((x) => {
    if (x.classList.contains("ry-tab")) x.classList.toggle("active", x.dataset.ry === key);
    else x.classList.toggle("ry-off", x.dataset.ry !== key);
  });
  try { const d = rySaved(); d.tool = key; localStorage.setItem(RY_KEY, JSON.stringify(d)); } catch (e) { /* private */ }
}

function rySource() { return (document.querySelector('input[name="rySource"]:checked') || {}).value || "folder"; }
function rySourceChanged() {
  $("tab-royale").classList.toggle("single", rySource() === "single"); ryRemember();
  ryShowTool(S.ry.tool || rySaved().tool || "epochs");
}

async function ryPtRefresh(applyPreset) {
  // the prompt-travel card: a preset fills it (subject, prompt, knobs); the travel's waypoints fill Start / End
  const preset = $("ryPPreset").value, subj = $("ryPSubject").value || "Woman";
  const dim = preset && preset !== "Custom words" ? null : (preset === "Custom words" ? "Custom" : null);
  const r = await api(`/api/royale/pt?preset=${encodeURIComponent(applyPreset ? preset : "")}&subject=${encodeURIComponent(subj)}` +
    `&dim=${encodeURIComponent(S.ry.ptDim || dim || "")}&custom=${encodeURIComponent($("ryPCustom").value)}`).catch(() => null);
  if (!r) return;
  if (applyPreset === "subject" && r.preset) {      // a new subject: the prompt re-composed, the knobs left alone
    $("ryPPrompt").value = r.preset.prompt; S.ry.ptDim = r.preset.dim;
    return ryPtRefresh(false);
  }
  if (applyPreset && r.preset) {
    const p = r.preset;
    if (p.subject && p.subject !== $("ryPSubject").value && applyPreset === "preset") { $("ryPSubject").value = p.subject; return ryPtRefresh("preset2"); }
    S.ry.ptDim = p.dim; $("ryPPrompt").value = p.prompt; $("ryPInterp").value = p.interp; $("ryPDrift").value = p.drift;
    $("ryPVary").checked = !!p.vary_seed; $("ryPSeq").checked = !!p.sequential; $("ryPAnchor").checked = !!p.anchor;
    $("ryPAnchorStr").value = p.anchor_str; $("ryPRefStr").value = p.ref_strength; $("ryPMp").value = p.ref_mp;
    return ryPtRefresh(false);
  }
  const words = r.words || [];
  for (const id of ["ryPStart", "ryPEnd"]) {
    const cur = $(id).value;
    $(id).replaceChildren(...words.map((w) => el("option", {value: w, text: w})));
    $(id).value = words.includes(cur) ? cur : (id === "ryPStart" ? words[0] : words[words.length - 1]) || "";
  }
  S.ry.ptWords = words;
  ryPtWords();
}

function ryPtRange() {
  const w = S.ry.ptWords || [];
  if (w.length < 2) return w;
  const si = Math.max(0, w.indexOf($("ryPStart").value)), ei = w.indexOf($("ryPEnd").value) < 0 ? w.length - 1 : w.indexOf($("ryPEnd").value);
  return si <= ei ? w.slice(si, ei + 1) : w.slice(ei, si + 1).reverse();
}
function ryPtWords() {
  const w = ryPtRange();
  $("ryPWords").textContent = w.length >= 2 ? "Waypoints:  " + w.join("  →  ") : w.length ?
    "Pick a Start/End that span at least two waypoints." : "Add at least two comma-separated custom words to travel between.";
  $("ryPDrift").disabled = $("ryPVary").checked;
}

async function ryTravel(mode) {
  const p = RY_MODES[mode], v = (id) => $(p + id).value;
  const body = {mode, family: $("ryFamily").value, source: rySource(), single: $("rySingle").value.trim(),
    i: royaleCurrent(), prompt: $("ryPrompt").value.trim(), seed: seedOf("rySeed"),
    setup_ref: $("ryRef").value.trim(), width: Number(v("W")), height: Number(v("H")), frames: Number(v("Frames"))};
  if (mode !== "strength") Object.assign(body, {ref: v("Ref").trim(), use_epoch: $(p + "UseEpoch").checked, ref_strength: v("RefStr"),
    ref_mp: v("Mp"), sequential: $(p + "Seq").checked});
  if (mode === "seed") Object.assign(body, {seed_a: v("A"), seed_b: v("B"), waypoints: $("rySWay").value});
  if (mode === "strength") Object.assign(body, {s_start: $("ryLA").value, s_end: $("ryLB").value});
  if (mode === "prompt") Object.assign(body, {base: $("ryPPrompt").value.trim(), words: ryPtRange(), interp: $("ryPInterp").value,
    drift: $("ryPDrift").value, vary_seed: $("ryPVary").checked, anchor: $("ryPAnchor").checked, anchor_str: $("ryPAnchorStr").value});
  try {
    const r = await api("/api/royale/travel", body);
    if (r.family !== $("ryFamily").value) await royaleFamily(r.family);
    $(p + "Sc").style.display = "none";
    ryRemember(); ryTravelPoll();
  } catch (e) { $(p + "Status").textContent = e.message; }
}

async function rySheet(confirm) {
  try {
    const r = await api("/api/royale/travel", {mode: "sheet", confirm: confirm === true, family: $("ryFamily").value, source: rySource(),
      single: $("rySingle").value.trim(), i: royaleCurrent(), prompts: $("ryCPrompts").value, sheet_mode: $("ryCMode").value,
      epochs: $("ryCEpochs").value, trigger: $("ryCTrigger").value.trim(), seed: seedOf("ryCSeed"),
      width: Number($("ryCW").value), height: Number($("ryCH").value), row_labels: $("ryCRows").checked, brand: $("ryCTag").checked});
    if (r.confirm) {                  // a big sheet: say so first, as the desktop does
      $("ryCStatus").replaceChildren(document.createTextNode(r.confirm + " "),
        el("button", {class: "btn btn-sm", text: "Render it", onclick: () => rySheet(true)}));
      return;
    }
    if (r.family !== $("ryFamily").value) await royaleFamily(r.family);
    $("ryCSheet").style.display = "none";
    $("ryCStatus").textContent = "Starting…";    // the Render it button goes, so the poll's progress shows (#215)
    ryRemember(); ryTravelPoll();
  } catch (e) { $("ryCStatus").textContent = e.message; }
}

async function ryTravelPoll() {
  clearTimeout(S.ry.tpoll);
  let st;
  try { st = await api("/api/royale/travel/status"); } catch (e) { return; }
  let running = false;
  for (const [mode, t] of Object.entries(st)) {
    running = running || t.state === "rendering";
    if (mode === "sheet") {
      $("ryCGo").disabled = t.state === "rendering";
      // not over the big sheet's Render it button (#215, @Nessa2026691)
      if (t.status && !$("ryCStatus").querySelector("button")) $("ryCStatus").textContent = t.status;
      if (t.state === "done" && t.sheet && S.ry.tr.sheet !== t.gen) {
        S.ry.tr.sheet = t.gen;
        $("ryCImg").src = `/api/royale/travel/image?mode=sheet&g=${t.gen}&t=${encodeURIComponent(TOKEN)}`;
        $("ryCSheet").style.display = "";
      }
      continue;
    }
    const p = RY_MODES[mode], sc = $(p + "Sc");
    $(p + "Go").disabled = t.state === "rendering";
    if (t.status) $(p + "Status").textContent = t.status;
    const key = `${t.gen}:${t.frames.length}`;
    if (t.frames.length && S.ry.tr[mode] !== key) {
      S.ry.tr[mode] = key;
      sc._frames = t.frames.slice().sort((a, b) => a.i - b.i); sc._gen = t.gen;
      const range = sc.querySelector(".ry-scrub");
      range.max = sc._frames.length - 1;
      if (t.state === "done" || Number(range.value) > range.max) range.value = t.state === "done" ? 0 : range.max;
      sc.style.display = "";
      ryScScrub(sc);
    }
  }
  if (running && $("tab-royale").classList.contains("active")) S.ry.tpoll = setTimeout(ryTravelPoll, 1000);
}

function ryScScrub(sc) {
  const f = sc._frames || [];
  if (!f.length) return;
  const i = Math.max(0, Math.min(Number(sc.querySelector(".ry-scrub").value), f.length - 1));
  sc.querySelector("img").src = `/api/royale/travel/image?mode=${sc.dataset.mode}&i=${f[i].i}&g=${sc._gen}&t=${encodeURIComponent(TOKEN)}`;
  sc.querySelector(".ry-label").textContent = `Frame ${i + 1}/${f.length}    ${f[i].label}`;
}

async function ryScSave(sc, fmt) {
  const mode = sc.dataset.mode, p = RY_MODES[mode], msg = sc.querySelector(".ry-sc-msg");
  const f = sc._frames || [], i = Math.max(0, Math.min(Number(sc.querySelector(".ry-scrub").value), f.length - 1));
  msg.textContent = "Saving…";
  try {
    const r = fmt === "frame" ? await api("/api/royale/travel/frame", {mode, i: f[i].i, out_dir: $("ryOut").value.trim()}) :
      await api("/api/royale/travel/save", {mode, format: fmt, out_dir: $("ryOut").value.trim(), speed: $(p + "Speed").value,
        loop: $(p + "Loop").checked, badge: $(p + "Badge").checked, tag: $(p + "Tag").checked, deflicker: $(p + "Deflicker").value});
    msg.replaceChildren(el("span", {text: `Saved ${r.path} `}), el("a", {class: "btn btn-sm", download: r.path.split(/[\\/]/).pop(),
      text: "Download", href: `/api/media?path=${encodeURIComponent(r.path)}&t=${encodeURIComponent(TOKEN)}`}));
  } catch (e) { msg.textContent = e.message; }
}

async function wireRoyaleTravel() {
  document.querySelectorAll('input[name="rySource"]').forEach((r) => r.addEventListener("change", rySourceChanged));
  document.querySelectorAll("#ryTabs .ry-tab").forEach((b) => b.addEventListener("click", () => ryShowTool(b.dataset.ry)));
  const setSingle = (p) => { $("rySingle").value = p; ryRemember(); };
  const up = uploadButton({kind: "path", key: "ROYALE_SINGLE"}, $("rySingle"), setSingle);
  $("rySingleBtns").replaceChildren(el("button", {class: "btn", text: "Choose…", onclick: () => pick($("rySingle").value || loraStart(), false, setSingle)}),
    up.btn, up.status);
  for (const [mode, p] of Object.entries(RY_MODES)) {
    $(p + "Go").addEventListener("click", () => ryTravel(mode));
    const sc = $(p + "Sc");
    sc.querySelector(".ry-scrub").addEventListener("input", () => ryScScrub(sc));
    sc.querySelectorAll(".ry-sc-save button").forEach((b) => b.addEventListener("click", () => ryScSave(sc, b.dataset.fmt)));
    if (mode !== "strength") {
      $(p + "RefPick").addEventListener("click", () => pick($(p + "Ref").value || refStart(), false, (x) => { $(p + "Ref").value = x; ryRemember(); }));
      $(p + "RefClear").addEventListener("click", () => { $(p + "Ref").value = ""; ryRemember(); });
      $(p + "UseEpoch").addEventListener("change", () => { for (const id of ["Ref", "RefPick", "RefClear"]) $(p + id).disabled = $(p + "UseEpoch").checked; });
    }
  }
  $("ryCGo").addEventListener("click", () => rySheet());
  $("ryCSave").addEventListener("click", async () => {
    try { const r = await api("/api/royale/travel/save", {mode: "sheet", out_dir: $("ryOut").value.trim()});
      $("ryCMsg").textContent = `Saved ${r.path}`; } catch (e) { $("ryCMsg").textContent = e.message; }
  });
  $("ryCImg").addEventListener("click", () => window.open($("ryCImg").src, "_blank"));
  $("rySDice").addEventListener("click", () => { $("rySA").value = Math.floor(Math.random() * 1e6); $("rySB").value = Math.floor(Math.random() * 1e6); });
  const info = await api("/api/royale/pt").catch(() => ({presets: [], subjects: [], seed_presets: {}}));
  S.ry.seedPresets = info.seed_presets;
  $("rySPreset").append(...Object.keys(info.seed_presets).map((n) => el("option", {value: n, text: n})));
  $("rySPreset").addEventListener("change", () => {
    const p = S.ry.seedPresets[$("rySPreset").value]; if (!p) return;
    $("rySRefStr").value = p.ref_strength; $("rySMp").value = p.ref_mp; $("rySSeq").checked = !!p.sequential; $("rySWay").value = p.waypoints;
  });
  $("ryPPreset").append(...info.presets.map((n) => el("option", {value: n, text: n})));
  $("ryPSubject").replaceChildren(...info.subjects.map((n) => el("option", {value: n, text: n})));
  $("ryPPreset").addEventListener("change", () => { if ($("ryPPreset").value) ryPtRefresh("preset"); });
  $("ryPSubject").addEventListener("change", () => ryPtRefresh($("ryPPreset").value ? "subject" : false));
  $("ryPCustom").addEventListener("change", () => ryPtRefresh(false));
  for (const id of ["ryPStart", "ryPEnd", "ryPVary"]) $(id).addEventListener("change", ryPtWords);
  $("ryPSlot").addEventListener("click", () => { const e = $("ryPPrompt"), at = e.selectionStart ?? e.value.length;
    e.value = e.value.slice(0, at) + "{x}" + e.value.slice(at); e.focus(); });
  for (const p of Object.values(RY_MODES).concat(["ryC"])) for (const d of ["W", "H"]) if (!$(p + d).dataset.set) $(p + d).value = "512";
  const sv = rySaved();
  if (sv.source === "single") document.querySelector('input[name="rySource"][value="single"]').checked = true;
  if (sv.single) $("rySingle").value = sv.single;
  if (sv.cmp) $("ryCPrompts").value = sv.cmp;
  for (const [id, val] of Object.entries(sv.keep || {})) {
    if (!$(id) || val == null) continue;
    if ($(id).type === "checkbox") $(id).checked = !!val;
    else if ($(id).tagName !== "SELECT" || [...$(id).options].some((o) => o.value === val)) { $(id).value = val; $(id).dataset.set = "1"; }
  }
  if (sv.ptDim) S.ry.ptDim = sv.ptDim;
  document.querySelectorAll("#tab-royale input, #tab-royale select, #tab-royale textarea").forEach((x) =>
    x.addEventListener("change", () => ryRemember()));
  rySourceChanged();
  if (S.ry.ptDim === undefined) S.ry.ptDim = "Age";
  ryPtRefresh(false);
  ryTravelPoll();
}

async function royaleFamily(key) {
  const r = await api("/api/royale/info?family=" + encodeURIComponent(key || "")).catch(() => null);
  if (!r) return;
  S.ry.info = r;
  $("ryFamily").replaceChildren(...r.families.map((f) => el("option", {value: f.key, text: f.name + (f.video ? " (clip previews)" : "")})));
  $("ryFamily").value = r.family;
  $("ryRefRow").style.display = r.reference_kind ? "" : "none";
  $("ryRefStrBox").style.display = r.reference_strength ? "" : "none";
  $("tab-royale").classList.toggle("no-ref", !r.reference_kind);
  $("tab-royale").classList.toggle("no-refstr", !r.reference_strength);
  if (!$("ryOut").value) $("ryOut").value = r.out_dir;
  renderChoiceRefresh();
}

async function royaleScan() {
  const f = $("ryFolder").value.trim();
  if (!f) { $("ryScan").textContent = ""; return; }
  const r = await api("/api/royale/scan?folder=" + encodeURIComponent(f)).catch(() => ({summary: ""}));
  $("ryScan").textContent = r.summary || "";
}

async function royaleRender() {
  try {
    const r = await api("/api/royale/render", {family: $("ryFamily").value, folder: $("ryFolder").value.trim(),
      prompt: $("ryPrompt").value.trim(), seed: seedOf("rySeed"), ref: $("ryRef").value.trim(),
      ref_strength: refStrength($("ryRefStr").value), width: Number($("ryW").value), height: Number($("ryH").value),
      max: $("ryMax").value});
    if (r.family !== $("ryFamily").value) {         // the checkpoints are another model's: follow them (as the desktop)
      await royaleFamily(r.family);
      $("ryStatus").textContent = `Switched to ${$("ryFamily").selectedOptions[0].text} to match the checkpoints.`;
    }
    S.ry.items = []; S.ry.shown = -1; S.ry.scores = {}; S.ry.best = null;
    $("ryLikeStatus").textContent = ""; $("ryBest").disabled = true;
    ryRemember();
    royalePoll();
  } catch (e) { $("ryStatus").textContent = e.message; }
}

async function royalePoll() {
  clearTimeout(S.ry.poll);
  let st;
  try { st = await api("/api/royale/status"); } catch (e) { return; }
  const running = st.state === "rendering";
  $("ryRender").disabled = running; $("ryStop").disabled = !running;
  $("rySetupBadge").textContent = running ? "Rendering" : "";
  $("rySetupBadge").dataset.state = running ? "running" : "idle";
  const done = st.items.filter((x) => x.png).length;
  $("ryBar").style.width = running ? `${Math.round(100 * Math.max(done, 0.3) / Math.max(st.total, 1))}%` : "0";
  if (st.status) $("ryStatus").textContent = st.status + (st.error && st.state === "failed" ? ` ${st.error}` : "");
  const items = st.items.filter((x) => x.png).sort((a, b) => a.i - b.i);
  if (st.gen !== S.ry.gen || items.length !== S.ry.items.length) {
    const fresh = st.gen !== S.ry.gen;
    S.ry.gen = st.gen; S.ry.items = items;
    if (fresh) { S.ry.scores = {}; S.ry.best = null; }
    royaleGrid();
    $("ryScrub").max = Math.max(0, items.length - 1);
    if (items.length && (fresh || S.ry.shown < 0)) { $("ryScrub").value = 0; S.ry.shown = -1; }
    royaleScrub();
  }
  if (running && $("tab-royale").classList.contains("active")) S.ry.poll = setTimeout(royalePoll, 1000);
}

const ryImg = (i) => `/api/royale/image?i=${S.ry.items[i].i}&gen=${S.ry.gen}&t=${encodeURIComponent(TOKEN)}`;

function royaleScrub() {
  // the crossfade: the two epochs either side of the slider, the upper one faded in by how far along it is
  const items = S.ry.items;
  if (!items.length) { $("ryLabel").textContent = ""; return; }
  const p = Number($("ryScrub").value), lo = Math.max(0, Math.min(Math.floor(p), items.length - 1));
  const hi = Math.min(lo + 1, items.length - 1), a = p - lo;
  if ($("ryA").dataset.i !== String(lo)) { $("ryA").src = ryImg(lo); $("ryA").dataset.i = String(lo); }
  if ($("ryB").dataset.i !== String(hi)) { $("ryB").src = ryImg(hi); $("ryB").dataset.i = String(hi); }
  $("ryB").style.opacity = lo === hi ? 0 : a;
  $("ryLabel").textContent = a <= 0.01 || lo === hi ? ryLabel(items[lo].label) :
    `${ryLabel(items[lo].label)}  →  ${ryLabel(items[hi].label)}    (${Math.round(a * 100)}%)`;
  const at = Math.round(p);
  document.querySelectorAll("#ryGrid .ry-cell").forEach((c, i) => c.classList.toggle("at", i === at));
}

function royaleGrid() {
  $("ryGrid").replaceChildren(...S.ry.items.map((it, i) => {
    const base = String(it.label).match(/^\d+$/) ? `e${it.label}` : String(it.label).slice(0, 16);
    const sc = S.ry.scores[String(it.label)];
    const cap = sc === undefined ? base : `${base}  ${sc == null ? "—" : sc.toFixed(2)}`;
    return el("div", {class: "ry-cell" + (S.ry.best != null && String(it.label) === String(S.ry.best) ? " best" : ""),
      title: it.path, onclick: () => { $("ryScrub").value = i; royaleScrub(); }},
      el("img", {src: ryImg(i), alt: base, loading: "lazy"}), el("div", {text: cap}));
  }));
}

function royaleCurrent() { return Math.max(0, Math.min(Math.round(Number($("ryScrub").value)), S.ry.items.length - 1)); }

async function royaleScore() {
  $("ryLikeStatus").textContent = "Scoring… (the first run loads the face model)";
  $("ryLikePeak").textContent = "";
  try {
    const r = await api("/api/royale/score", {refs: S.ry.likeRefs || []});
    S.ry.scores = r.scores; S.ry.best = r.best;
    royaleGrid();
    if (r.best == null) { $("ryLikeStatus").textContent = "No faces detected in the renders or subject photos. Likeness needs a face: for a style or object LoRA, judge the epochs by eye."; $("ryBest").disabled = true; return; }
    $("ryBest").disabled = false;
    $("ryLikeStatus").textContent = `Best: ${ryLabel(r.best).toLowerCase()} (${r.scores[r.best].toFixed(3)})` +
      (r.no_face ? `  (${r.no_face} no-face)` : "") +
      (r.embedder && !r.embedder.startsWith("ArcFace") ? `  · ${r.embedder}` : "");     // named when it isn't the usual one
    $("ryLikePeak").textContent = arcPeak(r.embedder).replace("72%", "0.72 (72%)");
    royaleBest();
  } catch (e) { $("ryLikeStatus").textContent = e.message; }
}

// the Likeness score card's chosen photos, as thumbnails
function ryLikeShow() {
  const refs = S.ry.likeRefs || [];
  $("ryLikeRefs").replaceChildren(...(refs.length ? refs.map((p) => el("img", {title: p, alt: "",
    src: `/api/thumb?path=${encodeURIComponent(p)}&t=${encodeURIComponent(TOKEN)}`}))
    : [el("span", {class: "field-hint", text: "None chosen"})]));
}

function royaleBest() {
  const i = S.ry.items.findIndex((it) => String(it.label) === String(S.ry.best));
  if (i >= 0) { $("ryScrub").value = i; royaleScrub(); }
}

async function royaleExport(likeness) {
  const say = likeness ? $("ryLikeStatus") : $("ryExportStatus");
  say.textContent = likeness ? "Building likeness clip…" : "Building frames…";
  try {
    const r = await api("/api/royale/export", {likeness, format: $("ryFmt").value, speed: $("rySpeed").value,
      loop: $("ryLoop").checked, ticker: $("ryTicker").checked, tag: $("ryTag").checked, out_dir: $("ryOut").value.trim()});
    say.replaceChildren(el("span", {text: `Saved ${r.frames} frames → ${r.path} `}),
      el("a", {class: "btn btn-sm", download: r.path.split(/[\\/]/).pop(), text: "Download",
        href: `/api/media?path=${encodeURIComponent(r.path)}&t=${encodeURIComponent(TOKEN)}`}));
  } catch (e) { say.textContent = e.message + (/codec|writer/i.test(e.message) ? " Try the GIF format instead." : ""); }
}

function wireRoyale() {
  const remember = () => ryRemember();
  for (const id of ["ryPrompt", "rySeed", "ryRefStr", "ryW", "ryH", "ryMax", "ryFmt", "rySpeed", "ryOut"])
    $(id).addEventListener("change", remember);
  $("ryFamily").addEventListener("change", () => { royaleFamily($("ryFamily").value); remember(); });
  $("ryFolder").addEventListener("change", () => { royaleScan(); remember(); });
  $("ryFolderPick").addEventListener("click", () => pick($("ryFolder").value || S.outputDefault, true, (p) => {
    $("ryFolder").value = p; royaleScan(); remember(); }));
  $("ryRefPick").addEventListener("click", () => pick($("ryRef").value || refStart(), false, (p) => { $("ryRef").value = p; remember(); }));
  $("ryRefClear").addEventListener("click", () => { $("ryRef").value = ""; remember(); });
  $("ryLikePick").addEventListener("click", () => {      // the Samples gallery's picker, from the photos' folder
    const refs = S.ry.likeRefs || [], dir = (p) => p.replace(/[\\/][^\\/]*$/, "");
    const folder = refs.length ? dir(refs[0]) : refStart() || "";
    S.lik.mode = "royale";
    likGridShow(folder, refs.filter((p) => dir(p) === folder).map((p) => p.split(/[\\/]/).pop()));
  });
  $("ryLikeClear").addEventListener("click", () => { S.ry.likeRefs = []; ryLikeShow(); remember(); });
  $("ryOutPick").addEventListener("click", () => pick($("ryOut").value, true, (p) => { $("ryOut").value = p; remember(); }));
  $("ryRender").addEventListener("click", royaleRender);
  $("ryStop").addEventListener("click", () => api("/api/royale/cancel", {}).then(royalePoll));
  $("tab-royale").addEventListener("click", (e) => { if (e.target.matches('[id$="Go"]')) setTimeout(ryTravelPoll, 400); });
  $("ryUnload").addEventListener("click", async () => {
    await api("/api/studio/reset", {});            // the shared GPU worker quits (Studio's Unload)
    $("ryStatus").textContent = "Model unloaded - the renders stay here until the next render.";
  });
  $("ryScrub").addEventListener("input", royaleScrub);
  $("ryScore").addEventListener("click", held($("ryScore"), royaleScore));
  $("ryBest").addEventListener("click", royaleBest);
  $("ryExport").addEventListener("click", held($("ryExport"), () => royaleExport(false)));
  $("ryLikeExport").addEventListener("click", held($("ryLikeExport"), () => royaleExport(true)));
  $("ryStills").addEventListener("click", held($("ryStills"), async () => {
    $("ryExportStatus").textContent = "Saving stills…";
    try { const r = await api("/api/royale/stills", {out_dir: $("ryOut").value.trim()});
      $("ryExportStatus").textContent = `Saved ${r.saved.length} stills → ${$("ryOut").value.trim()}`; }
    catch (e) { $("ryExportStatus").textContent = e.message; }
  }));
  $("ryPromote").addEventListener("click", held($("ryPromote"), async () => {
    try { const r = await api("/api/royale/promote", {i: royaleCurrent(), out_dir: $("ryOut").value.trim(),
      name: $("ryPromoteName").value.trim()});
      $("ryPromoteStatus").textContent = `Saved ${ryLabel(S.ry.items[royaleCurrent()].label).toLowerCase()} → ${r.path}`; }
    catch (e) { $("ryPromoteStatus").textContent = e.message; }
  }));
}

// ---- Profiler: what a LoRA's blocks do (fizgig/web/tools.py runs scripts/profile_run.py, the desktop Profiler's job) ----
S.pf = {wired: false, photos: [], poll: null, last: null};
const PF_KEY = "fizgig_profiler";
const pfSaved = () => { try { return JSON.parse(localStorage.getItem(PF_KEY) || "{}"); } catch (e) { return {}; } };
function pfRemember() {
  try { localStorage.setItem(PF_KEY, JSON.stringify({family: $("pfFamily").value, lora: $("pfLora").value, mode: pfMode(),
    size: $("pfSize").value, prompt: $("pfPrompt").value, cls: $("pfClass").value, photos: S.pf.photos})); } catch (e) { /* private */ }
}
const pfMode = () => (document.querySelector('input[name="pfMode"]:checked') || {}).value || "quick";

async function profilerOpen() {
  if (!S.pf.wired) {
    S.pf.wired = true;
    const r = await api("/api/profile/info").catch(() => ({families: []}));
    $("pfFamily").replaceChildren(...r.families.map((f) => el("option", {value: f.key, text: f.name})));
    const sv = pfSaved();
    if (sv.family && r.families.some((f) => f.key === sv.family)) $("pfFamily").value = sv.family;
    $("pfLora").value = sv.lora || ""; $("pfPrompt").value = sv.prompt || ""; $("pfClass").value = sv.cls || "";
    if (sv.size) $("pfSize").value = sv.size;
    const m = document.querySelector(`input[name="pfMode"][value="${sv.mode || "quick"}"]`); if (m) m.checked = true;
    S.pf.photos = sv.photos || [];
    wireProfiler();
    pfPhotosDraw(); pfModeChanged();
  }
  renderChoiceRefresh();
  profilerPoll();
}

function pfModeChanged() { $("pfRender").classList.toggle("off", pfMode() === "weights"); pfRemember(); }

function pfPhotosDraw() {
  $("pfPhotos").replaceChildren(...S.pf.photos.map((p, i) => el("span", {class: "pf-photo", title: p},
    el("img", {src: `/api/thumb?path=${encodeURIComponent(p)}&t=${encodeURIComponent(TOKEN)}`, alt: ""}),
    el("button", {class: "btn btn-sm", text: "×", onclick: () => { S.pf.photos.splice(i, 1); pfPhotosDraw(); pfRemember(); }}))));
  if (!S.pf.photos.length) $("pfPhotos").append(el("span", {class: "field-hint", text: "None (picture change only)"}));
  $("pfPhotoBtns").style.display = S.pf.photos.length >= 3 ? "none" : "";
}

async function pfLora(p, fresh) {
  // a LoRA chosen: its model, then (a new one) its trigger prompts and its run's subject photos, as the desktop does
  $("pfLora").value = p;
  try {
    const f = await api(`/api/studio/lorafamily?path=${encodeURIComponent(p)}`);
    if (f.key && [...$("pfFamily").options].some((o) => o.value === f.key)) { $("pfFamily").value = f.key; renderChoiceRefresh(); }
  } catch (e) { /* unreadable: the run says so */ }
  if (fresh) {
    const r = await api(`/api/profile/prefill?path=${encodeURIComponent(p)}`).catch(() => ({}));
    $("pfPrompt").value = r.prompt || ""; $("pfClass").value = r.class_prompt || "";
    S.pf.photos = r.baselines || []; pfPhotosDraw();
  }
  pfRemember();
}

async function profilerRun() {
  try {
    await api("/api/profile/run", {family: $("pfFamily").value, lora: $("pfLora").value.trim(), mode: pfMode(),
      prompt: $("pfPrompt").value.trim(), class_prompt: $("pfClass").value.trim(), baselines: S.pf.photos,
      size: Number($("pfSize").value)});
    $("pfReport").style.display = "none";
    profilerPoll();
  } catch (e) { $("pfStatus").textContent = e.message; }
}

async function profilerPoll() {
  clearTimeout(S.pf.poll);
  let st;
  try { st = await api("/api/profile/status"); } catch (e) { return; }
  const running = st.state === "running";
  $("pfRun").disabled = running; $("pfStop").disabled = !running;
  $("pfBadge").textContent = {running: "Running", done: "Done", failed: "Failed", stopped: "Stopped"}[st.state] || "";
  $("pfBadge").dataset.state = running ? "running" : st.state === "done" ? "completed" : st.state === "idle" ? "idle" : "failed";
  const p = st.progress;
  $("pfBar").style.width = running ? (p && p[1] ? `${Math.round(100 * p[0] / p[1])}%` : "100%") : st.state === "done" ? "100%" : "0";
  $("pfBar").parentElement.classList.toggle("indeterminate", running && !(p && p[1]));
  const res = st.result;
  $("pfStatus").textContent = running ? st.text : st.state === "done" ? "Done." : st.state === "stopped" ? "Stopped." :
    st.state === "failed" ? "Error - see below." : "";
  $("pfLog").textContent = [...st.log.filter((l) => !/^INFO:/.test(l)), ...(res ? ["", ...res.lines] : []), ...(st.state === "failed" ? ["", st.error] : [])].join("\n");
  if (res && res.report) {
    $("pfReport").href = `/api/profile/report?path=${encodeURIComponent(res.report)}&t=${encodeURIComponent(TOKEN)}`;
    $("pfReport").style.display = "";
  }
  if (running && $("tab-profiler").classList.contains("active")) S.pf.poll = setTimeout(profilerPoll, 800);
}

function wireProfiler() {
  document.querySelectorAll('input[name="pfMode"]').forEach((r) => r.addEventListener("change", pfModeChanged));
  for (const id of ["pfFamily", "pfSize", "pfPrompt", "pfClass"]) $(id).addEventListener("change", pfRemember);
  $("pfFamily").addEventListener("change", renderChoiceRefresh);
  $("pfLora").addEventListener("change", () => pfLora($("pfLora").value.trim(), true));
  const up = uploadButton({kind: "path", key: "PROFILER_LORA"}, $("pfLora"), (p) => pfLora(p, true));
  $("pfLoraBtns").replaceChildren(el("button", {class: "btn", text: "Choose…", onclick: () =>
    pick($("pfLora").value || loraStart(), false, (p) => pfLora(p, true))}), up.btn, up.status);
  const addPhoto = (p) => { if (S.pf.photos.length < 3 && !S.pf.photos.includes(p)) S.pf.photos.push(p); pfPhotosDraw(); pfRemember(); };
  const pup = uploadButton({kind: "path", key: "PROFILER_PHOTO"}, el("input", {type: "text"}), addPhoto);
  $("pfPhotoBtns").replaceChildren(el("button", {class: "btn btn-sm", text: "Add photo…", onclick: () => pick(S.imageFolder, false, addPhoto)}),
    pup.btn, pup.status);
  $("pfPhotosClear").addEventListener("click", () => { S.pf.photos = []; pfPhotosDraw(); pfRemember(); });
  $("pfRun").addEventListener("click", profilerRun);
  $("pfStop").addEventListener("click", () => api("/api/profile/stop", {}).then(profilerPoll));
}

// ---- Extract: a LoRA reduced to a lower rank (exact weight SVD from the file; fizgig/web/tools.py runs it) ----
S.ex = {info: null, poll: null, wired: false};
const EX_KEY = "fizgig_extract";
const exSaved = () => { try { return JSON.parse(localStorage.getItem(EX_KEY) || "{}"); } catch (e) { return {}; } };
const EX_FT_RANKS = [8, 16, 32, 64, 128, 256];
function exRemember() {
  try { localStorage.setItem(EX_KEY, JSON.stringify({family: $("exFamily").value, source: $("exSource").value,
    rank: $("exRank").value, preset: $("exPreset").value, mode: S.ex.mode, ftBase: $("exFtBase").value,
    ftTuned: $("exFtTuned").value, ftName: $("exFtName").value, ftOut: $("exFtOut").value,
    ftRanks: [...$("exFtRanks").querySelectorAll("input:checked")].map((c) => Number(c.value))})); } catch (e) { /* private */ }
}
function exMode(mode) {
  // From a LoRA (a LoRA reduced) or From a fine-tune (the old Checkpoint to LoRA app); one Run card for both
  S.ex.mode = mode;
  document.querySelectorAll("#exModes .ry-tab").forEach((b) => b.classList.toggle("active", b.dataset.exmode === mode));
  document.querySelectorAll("#tab-extract .section-card[data-exmode]").forEach((c) => c.classList.toggle("ex-off", c.dataset.exmode !== mode));
  $("exRun").textContent = mode === "ft" ? "Extract LoRAs" : "Extract LoRA";
  exRemember();
}

// Target rank: the dropdown shows the rank when it's one it lists, else Other… with the number in the box beside it
function exRankShow() {
  const v = String($("exRank").value), listed = [...$("exRankPick").options].some((o) => o.value === v);
  $("exRankPick").value = listed ? v : "__other";
  $("exRank").style.display = listed ? "none" : "";
}

async function extractOpen() {
  if (!S.ex.wired) wireExtract();
  const sv = exSaved();
  if (!S.ex.info) {
    if (sv.source) $("exSource").value = sv.source;
    if (sv.rank) $("exRank").value = sv.rank;
    exRankShow();
    const on = new Set(sv.ftRanks || [32, 64]);
    $("exFtRanks").replaceChildren(...EX_FT_RANKS.map((r) => { const c = el("input", {type: "checkbox", value: r, onchange: exRemember});
      c.checked = on.has(r); return el("label", {class: "checkbox-label"}, c, " " + r); }));
    for (const [id, k] of [["exFtBase", "ftBase"], ["exFtTuned", "ftTuned"], ["exFtName", "ftName"], ["exFtOut", "ftOut"]])
      if (sv[k]) $(id).value = sv[k];
    await extractFamily(sv.family, sv.preset);
    if (!$("exFtOut").value) $("exFtOut").value = $("exOutDir").value;
    exMode(sv.mode === "ft" ? "ft" : "lora");
  }
  extractPoll();
}

async function extractFamily(key, preset) {
  let r;
  try { r = await api("/api/extract/info?family=" + encodeURIComponent(key || "")); }
  catch (e) { $("exStatus").textContent = e.message; return; }
  S.ex.info = r;
  $("exFamily").replaceChildren(...r.families.map((f) => el("option", {value: f.key, text: f.name})));
  $("exFamily").value = r.family;
  $("exOutDir").value = r.out_dir;
  const names = r.presets.map((p) => p.name).concat(r.presets.length ? ["Custom"] : []);
  $("exPresetCard").style.display = names.length ? "" : "none";
  $("exPreset").replaceChildren(...names.map((n) => el("option", {value: n, text: n})));
  $("exPreset").value = names.includes(preset) ? preset : (names[0] || "");
  // the Custom picker: All / None, a quick pick per preset, the model's block groups
  const box = $("exCustom");
  box.replaceChildren();
  if (r.presets.length) {
    const tick = (on) => box.querySelectorAll("input[type=checkbox]").forEach((c) => { c.checked = on; });
    box.append(el("div", {class: "field-row"}, el("span", {class: "field-label", text: "Select blocks:"}),
      el("button", {class: "btn btn-sm", text: "All", onclick: () => tick(true)}),
      el("button", {class: "btn btn-sm", text: "None", onclick: () => tick(false)}),
      ...r.presets.filter((p) => p.blocks.length).map((p) => el("button", {class: "btn btn-sm", text: p.name, onclick: () =>
        box.querySelectorAll("input[type=checkbox]").forEach((c) => { c.checked = p.blocks.includes(c.value); })}))));
    for (const g of r.groups)
      box.append(el("div", {class: "ex-group"}, el("span", {class: "field-label", text: g.label.split(" ")[0] + ":"}),
        ...g.blocks.map((b) => el("label", {class: "checkbox-label"}, el("input", {type: "checkbox", value: b.id}),
          " " + b.label.split(" ").pop()))));
  }
  extractPresetChanged();
}

function extractPresetChanged() {
  $("exCustom").style.display = $("exPreset").value === "Custom" ? "" : "none";
  extractName();
  exRemember();
}

function extractName() {
  // the desktop's suggestion: <source>_<preset>_r<rank>
  const src = $("exSource").value.trim();
  if (!src) return;
  const base = src.split(/[\\/]/).pop().replace(/\.safetensors$/i, "");
  const slug = ($("exPreset").value || "all_blocks").toLowerCase().replace(/\+/g, "_").replace(/ /g, "_");
  $("exName").value = `${base}_${slug}_r${$("exRank").value}.safetensors`;
}

async function extractSource(p) {
  $("exSource").value = p;
  try {                                           // a LoRA of another model: switch to it, as the desktop does
    const f = await api(`/api/studio/lorafamily?path=${encodeURIComponent(p)}`);
    if (f.key && f.key !== $("exFamily").value && [...$("exFamily").options].some((o) => o.value === f.key))
      await extractFamily(f.key, $("exPreset").value);
  } catch (e) { /* unreadable: the run says so */ }
  extractName(); exRemember();
}

async function extractRun() {
  if (S.ex.mode === "ft") {
    const ranks = [...$("exFtRanks").querySelectorAll("input:checked")].map((c) => Number(c.value));
    try {
      await api("/api/extract/finetune", {base: $("exFtBase").value.trim(), tuned: $("exFtTuned").value.trim(),
        out_dir: $("exFtOut").value.trim(), name: $("exFtName").value.trim(), ranks});
      $("exStatus").textContent = "Extracting…";
      $("exDone").replaceChildren();
      exRemember();
      extractPoll();
    } catch (e) { $("exStatus").textContent = e.message; }
    return;
  }
  const custom = $("exPreset").value === "Custom";
  const blocks = custom ? [...$("exCustom").querySelectorAll("input:checked")].map((c) => c.value) : [];
  if (custom && !blocks.length) { $("exStatus").textContent = "Custom is selected but no blocks are ticked."; return; }
  const rank = Number($("exRank").value);
  if (!Number.isInteger(rank) || rank < 1) { $("exStatus").textContent = "Target rank is a whole number, 1 or more."; return; }
  try {
    const r = await api("/api/extract/run", {family: $("exFamily").value, source: $("exSource").value.trim(),
      name: $("exName").value.trim(), out_dir: $("exOutDir").value.trim(), rank,
      preset: custom || !(S.ex.info.presets || []).length ? "" : $("exPreset").value, blocks});
    $("exStatus").textContent = `Extracting to ${r.output}…`;
    $("exDone").replaceChildren();
    extractPoll();
  } catch (e) { $("exStatus").textContent = e.message; }
}

async function extractPoll() {
  clearTimeout(S.ex.poll);
  let st;
  try { st = await api("/api/extract/status"); } catch (e) { return; }
  const running = st.state === "running";
  $("exRun").disabled = running;
  $("exBadge").textContent = {running: "Running", done: "Done", failed: "Failed"}[st.state] || "";
  $("exBadge").dataset.state = running ? "running" : st.state === "done" ? "completed" : st.state === "failed" ? "failed" : "idle";
  const p = st.progress;
  $("exBar").style.width = running ? (p && p[1] ? `${Math.round(100 * (p[0] + 1) / p[1])}%` : "100%") : st.state === "done" ? "100%" : "0";
  $("exBar").parentElement.classList.toggle("indeterminate", running && !(p && p[1]));
  $("exLog").textContent = st.log.join("\n");
  if (running && p) $("exStatus").textContent = S.ex.mode === "ft" ? `SVD: layer ${p[0]} of ${p[1]}…` : `SVD: ${p[0] + 1} of ${p[1]} layers…`;
  if (st.state === "done") {
    const outs = st.outputs && st.outputs.length ? st.outputs : [st.output];
    $("exStatus").textContent = outs.length > 1 ? `Saved ${outs.length} LoRAs in ${outs[0].replace(/[\\/][^\\/]*$/, "")}` : `Saved ${outs[0]}`;
    if (!$("exDone").children.length) for (const o of outs) $("exDone").append(el("a", {class: "btn btn-sm btn-primary", download: "",
      text: outs.length > 1 ? "Download r" + ((o.match(/_r(\d+)\.safetensors$/i) || [])[1] || "") : "Download",
      href: `/api/download?path=${encodeURIComponent(o)}&t=${encodeURIComponent(TOKEN)}`}));
  }
  if (st.state === "failed") $("exStatus").textContent = `The extraction failed: ${st.error}`;
  if (running && $("tab-extract").classList.contains("active")) S.ex.poll = setTimeout(extractPoll, 500);
}

function wireExtract() {
  S.ex.wired = true;
  exRankShow();
  $("exFamily").addEventListener("change", () => { extractFamily($("exFamily").value, $("exPreset").value); exRemember(); });
  $("exPreset").addEventListener("change", extractPresetChanged);
  for (const ev of ["change", "input"]) $("exRank").addEventListener(ev, () => { extractName(); exRemember(); });
  $("exRankPick").addEventListener("change", () => {      // a listed rank, or Other… and a box for any number
    const other = $("exRankPick").value === "__other";
    if (!other) $("exRank").value = $("exRankPick").value;
    $("exRank").style.display = other ? "" : "none";
    if (other) { $("exRank").focus(); $("exRank").select(); }
    extractName(); exRemember();
  });
  $("exSource").addEventListener("change", () => extractSource($("exSource").value.trim()));
  const up = uploadButton({kind: "path", key: "EXTRACT_SOURCE"}, $("exSource"), extractSource);
  $("exSourceBtns").replaceChildren(el("button", {class: "btn", text: "Choose…", onclick: () =>
    pick($("exSource").value || loraStart(), false, extractSource)}), up.btn, up.status);
  $("exOutPick").addEventListener("click", () => pick($("exOutDir").value, true, (p) => { $("exOutDir").value = p; }));
  $("exRun").addEventListener("click", extractRun);
  document.querySelectorAll("#exModes .ry-tab").forEach((b) => b.addEventListener("click", () => exMode(b.dataset.exmode)));
  for (const [id, host] of [["exFtBase", "exFtBaseBtns"], ["exFtTuned", "exFtTunedBtns"]]) {
    const set = (p) => { $(id).value = p; exRemember(); };
    const up = uploadButton({kind: "path", key: "EXTRACT_" + id}, $(id), set);
    $(host).replaceChildren(el("button", {class: "btn", text: "Choose…", onclick: () => pick($(id).value || S.outputDefault, false, set)}),
      up.btn, up.status);
    $(id).addEventListener("change", exRemember);
  }
  $("exFtName").addEventListener("change", exRemember);
  $("exFtOut").addEventListener("change", exRemember);
  $("exFtOutPick").addEventListener("click", () => pick($("exFtOut").value, true, (p) => { $("exFtOut").value = p; exRemember(); }));
}

// ---- Metadata: any .safetensors file's ModelSpec metadata, edited (the header rewritten; tensors untouched) ----
const MD_FIELDS = [["modelspec.title", "Title"], ["modelspec.author", "Author"], ["modelspec.license", "License"],
  ["modelspec.tags", "Tags"], ["modelspec.trigger_phrase", "Trigger phrase"], ["modelspec.usage_hint", "Usage hint"]];
S.md = {path: "", thumb: null, wired: false};

function mdWire() {
  if (S.md.wired) return;
  S.md.wired = true;
  $("mdStandard").replaceChildren(...MD_FIELDS.map(([k, label]) => el("div", {class: "field-row"},
    el("span", {class: "field-label", text: label + ":"}), el("input", {type: "text", class: "w-path", "data-key": k}))),
  el("div", {class: "field-row"}, el("span", {class: "field-label", text: "Description:"}),
    el("textarea", {rows: 4, "data-key": "modelspec.description"})));
  const load = (p) => { $("mdPath").value = p; mdLoad(p); };
  const up = uploadButton({kind: "path", key: "METADATA_FILE"}, $("mdPath"), load);
  $("mdPathBtns").replaceChildren(el("button", {class: "btn", text: "Choose…", onclick: () => pick($("mdPath").value || loraStart(), false, load)}),
    up.btn, up.status);
  $("mdPath").addEventListener("change", () => mdLoad($("mdPath").value.trim()));
  const setThumb = async (p) => {
    try { const r = await api("/api/metadata/thumbnail", {path: p}); S.md.thumb = r.uri; mdThumb(); }
    catch (e) { $("mdSaveMsg").textContent = e.message; }
  };
  const tup = uploadButton({kind: "path", key: "METADATA_THUMB"}, el("input", {type: "text"}), setThumb);
  $("mdThumbBtns").replaceChildren(el("button", {class: "btn", text: "Replace…", onclick: () => pick(S.imageFolder, false, setThumb)}),
    tup.btn, tup.status);
  $("mdThumbClear").addEventListener("click", () => { S.md.thumb = null; mdThumb(); });
  $("mdAdd").addEventListener("click", () => mdRow("", ""));
  $("mdSave").addEventListener("click", () => mdSave(""));
  $("mdSaveAs").addEventListener("click", () => mdSave($("mdDest").value.trim()));
}

function mdThumb() {
  $("mdThumb").replaceChildren(S.md.thumb && String(S.md.thumb).startsWith("data:image") ?
    el("img", {src: S.md.thumb, alt: "thumbnail"}) : document.createTextNode("(no thumbnail)"));
}

function mdRow(k, v) {
  const row = el("div", {class: "md-row"}, el("input", {type: "text", class: "md-key", value: k, placeholder: "key"}),
    el("input", {type: "text", class: "md-val", value: v, placeholder: "value"}),
    el("button", {class: "btn btn-sm", text: "✕", title: "Remove this field", onclick: () => row.remove()}));
  $("mdCustom").append(row);
}

async function mdLoad(p) {
  mdWire();
  if (!p) return;
  let r;
  try { r = await api(`/api/metadata?path=${encodeURIComponent(p)}`); }
  catch (e) { $("mdStatus").textContent = e.message; return; }
  S.md.path = p;
  const meta = r.meta || {};
  document.querySelectorAll("#mdStandard [data-key]").forEach((i) => { i.value = meta[i.dataset.key] || ""; });
  S.md.thumb = meta["modelspec.thumbnail"] || null;
  mdThumb();
  const std = new Set([...MD_FIELDS.map((f) => f[0]), "modelspec.description", "modelspec.thumbnail"]);
  $("mdCustom").replaceChildren();
  for (const k of Object.keys(meta).filter((k) => !std.has(k)).sort()) mdRow(k, meta[k]);
  const n = Object.keys(meta).length;
  $("mdStatus").textContent = `Loaded - ${n} metadata key${n !== 1 ? "s" : ""} found.`;
  $("mdSaveMsg").textContent = "";
  $("mdDest").value = "";
}

async function mdSave(dest, overwrite) {
  if (!S.md.path) { $("mdSaveMsg").textContent = "Load a .safetensors file first."; return; }
  const meta = {};
  $("mdCustom").querySelectorAll(".md-row").forEach((r) => {
    const k = r.querySelector(".md-key").value.trim();
    if (k) meta[k] = r.querySelector(".md-val").value;
  });
  document.querySelectorAll("#mdStandard [data-key]").forEach((i) => { if (i.value.trim()) meta[i.dataset.key] = i.value.trim(); });
  if (S.md.thumb) meta["modelspec.thumbnail"] = S.md.thumb;
  if (!dest && !overwrite) {                      // the file itself: say so first, as the desktop asks
    $("mdSaveMsg").replaceChildren(el("span", {text: `Save the changes into ${S.md.path}? `}),
      el("button", {class: "btn btn-sm btn-primary", text: "Save", onclick: () => mdSave("", true)}));
    return;
  }
  $("mdSaveMsg").textContent = "Saving…";
  try {
    const r = await api("/api/metadata/save", {path: S.md.path, dest, meta, overwrite: !!overwrite});
    if (r.exists) {
      $("mdSaveMsg").replaceChildren(el("span", {text: "That file is there already. "}),
        el("button", {class: "btn btn-sm btn-danger", text: "Replace it", onclick: () => mdSave(dest, true)}));
      return;
    }
    if (dest) { $("mdPath").value = r.dest; await mdLoad(r.dest); }       // the copy is the file being edited now
    $("mdSaveMsg").textContent = `Saved ${r.dest}`;
  } catch (e) { $("mdSaveMsg").textContent = e.message; }
}

// a run's loss, epoch by epoch, as a small chart (the history keeps each epoch's last average loss)
function lossChart(curve) {
  const pts = (curve || []).filter((p) => Array.isArray(p) && isFinite(p[1]));
  if (pts.length < 2) return null;
  const W = 220, Hh = 44, pad = 3;
  const xs = pts.map((p) => p[0]), ys = pts.map((p) => p[1]);
  const x0 = Math.min(...xs), x1 = Math.max(...xs), y0 = Math.min(...ys), y1 = Math.max(...ys);
  const X = (x) => pad + (W - 2 * pad) * (x1 === x0 ? 0.5 : (x - x0) / (x1 - x0));
  const Y = (y) => pad + (Hh - 2 * pad) * (y1 === y0 ? 0.5 : 1 - (y - y0) / (y1 - y0));
  const ns = "http://www.w3.org/2000/svg";
  const svg = document.createElementNS(ns, "svg");
  svg.setAttribute("viewBox", `0 0 ${W} ${Hh}`);
  svg.setAttribute("class", "loss-chart");
  const line = document.createElementNS(ns, "polyline");
  line.setAttribute("points", pts.map((p) => `${X(p[0]).toFixed(1)},${Y(p[1]).toFixed(1)}`).join(" "));
  svg.append(line);
  const best = pts.reduce((a, b) => (b[1] < a[1] ? b : a));
  const dot = document.createElementNS(ns, "circle");
  dot.setAttribute("cx", X(best[0])); dot.setAttribute("cy", Y(best[1])); dot.setAttribute("r", 2.6);
  svg.append(dot);
  const t = document.createElementNS(ns, "title");
  t.textContent = pts.map((p) => `epoch ${p[0]}: ${p[1]}`).join("\n");
  svg.append(t);
  return el("div", {class: "loss-chart-box"}, svg, el("span", {class: "field-hint",
    text: `Loss by epoch: ${ys[0]} → ${ys[ys.length - 1]} · lowest ${best[1]} at epoch ${best[0]}`}));
}

// ---- the loss graph: a run's loss at every step (the trainer's .fizgig_loss_<name>.jsonl), with smoothing, a log
// scale and outliers clipped - its settings kept in this browser
const LG_KEY = "fizgig_lossgraph";
const lgSet = () => { try { return {smooth: 0.9, log: false, clip: true, lr: true, ...JSON.parse(localStorage.getItem(LG_KEY) || "{}")}; } catch (e) { return {smooth: 0.9, log: false, clip: true, lr: true}; } };
const lgSave = (s) => { try { localStorage.setItem(LG_KEY, JSON.stringify(s)); } catch (e) { /* private window */ } };

function lossGraph(data) {
  const box = el("div", {class: "loss-graph"});
  const pts = (data && data.points || []).filter((p) => isFinite(p[2]));
  if (pts.length < 2) {
    box.append(el("div", {class: "field-hint", text: "The loss graph fills in as the run trains (a point a step)."}));
    return box;
  }
  const set = lgSet();
  const svgHost = el("div", {class: "loss-graph-plot"});
  const read = el("div", {class: "field-hint loss-graph-read"});
  const ctl = (label, input) => el("label", {class: "checkbox-label"}, input, " " + label);
  const smooth = el("input", {type: "range", min: "0", max: "0.99", step: "0.01", value: set.smooth});
  const log = el("input", {type: "checkbox"}); log.checked = set.log;
  const clip = el("input", {type: "checkbox"}); clip.checked = set.clip;
  const lrOn = el("input", {type: "checkbox"}); lrOn.checked = set.lr;
  const hasLr = pts.some((p) => p[3] != null);
  const redraw = () => {
    Object.assign(set, {smooth: Number(smooth.value), log: log.checked, clip: clip.checked, lr: lrOn.checked});
    lgSave(set);
    draw();
  };
  for (const i of [smooth, log, clip, lrOn]) i.addEventListener("input", redraw);
  // what each line and mark is
  const key = (cls, text) => el("span", {class: "lg-key"}, el("i", {class: "lg-sw " + cls}), text);
  const legend = el("div", {class: "lg-legend"},
    key("lg-sw-raw", "loss at each step"), key("lg-sw-smooth", "smoothed loss"),
    hasLr ? key("lg-sw-lr", "learning rate (right-hand scale)") : null,
    key("lg-sw-best", "lowest smoothed loss"), key("lg-sw-epoch", "start of an epoch (e1, e2…)"));
  box.append(el("div", {class: "field-row loss-graph-ctl"},
    el("span", {class: "field-label", text: "Smoothing:"}), smooth, ctl("Log scale", log), ctl("Clip outliers", clip),
    hasLr ? ctl("Learning rate", lrOn) : null), svgHost, legend, read);

  function draw() {
    // drawn at the width it has, so the text stays its own size on a phone
    const W = Math.max(300, Math.round(svgHost.clientWidth || 800)), H = W < 600 ? 220 : 240;
    const L = 54, R = hasLr && set.lr ? 54 : 12, T = 10, B = 26;
    const xs = pts.map((p) => p[0]);
    const x0 = xs[0], x1 = xs[xs.length - 1];
    const tf = (v) => (set.log ? Math.log10(Math.max(v, 1e-8)) : v);
    // the smoothed line: an exponential moving average, run forward from the first step
    const sm = [];
    let a = null;
    for (const p of pts) { a = a === null ? p[2] : set.smooth * a + (1 - set.smooth) * p[2]; sm.push(a); }
    let ys = pts.map((p) => p[2]).concat(sm);
    if (set.clip && pts.length > 20) {              // the 1st-99th percentile: one spike doesn't flatten the rest
      const s = pts.map((p) => p[2]).sort((u, v) => u - v);
      ys = [s[Math.floor(s.length * 0.01)], s[Math.ceil(s.length * 0.99) - 1]].concat(sm);
    }
    let y0 = Math.min(...ys.map(tf)), y1 = Math.max(...ys.map(tf));
    if (y1 - y0 < 1e-9) { y0 -= 0.5; y1 += 0.5; }
    const X = (x) => L + (W - L - R) * (x1 === x0 ? 0.5 : (x - x0) / (x1 - x0));
    const Y = (v) => T + (H - T - B) * (1 - (Math.min(Math.max(tf(v), y0), y1) - y0) / (y1 - y0));
    const ns = "http://www.w3.org/2000/svg";
    const mk = (tag, attrs) => { const n = document.createElementNS(ns, tag); for (const [k, v] of Object.entries(attrs)) n.setAttribute(k, v); return n; };
    const svg = mk("svg", {viewBox: `0 0 ${W} ${H}`, class: "loss-graph-svg"});
    // epochs along the bottom: a faint line at each epoch's first step, labels thinned to fit
    const eps = (data.epochs || []).filter((e) => e[1] >= x0 && e[1] <= x1);
    const every = Math.max(1, Math.ceil(eps.length / 12));
    eps.forEach((e, i) => {
      svg.append(mk("line", {x1: X(e[1]), x2: X(e[1]), y1: T, y2: H - B, class: "lg-epoch"}));
      if (i % every === 0) { const t = mk("text", {x: X(e[1]) + 2, y: H - 8, class: "lg-axis"}); t.textContent = `e${e[0]}`; svg.append(t); }
    });
    // the loss axis: three ticks
    for (const f of [0, 0.5, 1]) {
      const v = y0 + (y1 - y0) * f, yy = T + (H - T - B) * (1 - f);
      svg.append(mk("line", {x1: L, x2: W - R, y1: yy, y2: yy, class: "lg-grid"}));
      const t = mk("text", {x: L - 6, y: yy + 4, class: "lg-axis", "text-anchor": "end"});
      t.textContent = (set.log ? 10 ** v : v).toPrecision(3);
      svg.append(t);
    }
    svg.append(mk("polyline", {class: "lg-raw", points: pts.map((p) => `${X(p[0]).toFixed(1)},${Y(p[2]).toFixed(1)}`).join(" ")}));
    svg.append(mk("polyline", {class: "lg-smooth", points: pts.map((p, i) => `${X(p[0]).toFixed(1)},${Y(sm[i]).toFixed(1)}`).join(" ")}));
    // the learning rate, on its own scale at the right (adaptive LR moves it)
    const lrPts = pts.filter((p) => p[3] != null);
    if (hasLr && set.lr && lrPts.length > 1) {
      const lv = lrPts.map((p) => p[3]), l0 = Math.min(...lv), l1 = Math.max(...lv);
      const LY = (v) => T + (H - T - B) * (l1 === l0 ? 0.5 : 1 - (v - l0) / (l1 - l0));
      svg.append(mk("polyline", {class: "lg-lr", points: lrPts.map((p) => `${X(p[0]).toFixed(1)},${LY(p[3]).toFixed(1)}`).join(" ")}));
      for (const [v, yy] of [[l1, T + 4], [l0, H - B]]) {
        const t = mk("text", {x: W - R + 4, y: yy, class: "lg-axis lg-lr-axis"}); t.textContent = v.toExponential(1); svg.append(t);
      }
    }
    // the lowest smoothed point
    let bi = 0;
    sm.forEach((v, i) => { if (v < sm[bi]) bi = i; });
    const best = mk("circle", {cx: X(pts[bi][0]), cy: Y(sm[bi]), r: 4, class: "lg-best"});
    const bt = mk("title", {}); bt.textContent = `Lowest smoothed loss: ${sm[bi].toFixed(4)} at step ${pts[bi][0]}`;
    best.append(bt);
    svg.append(best);
    const axisT = (x, text, cls, anchor) => { const t = mk("text", {x, y: T + 2, class: "lg-axis " + cls, "text-anchor": anchor,
      "dominant-baseline": "hanging"}); t.textContent = text; svg.append(t); };
    axisT(4, "loss", "lg-axis-title", "start");
    if (hasLr && set.lr && lrPts.length > 1) axisT(W - 4, "LR", "lg-axis-title lg-lr-axis", "end");
    const cursor = mk("line", {y1: T, y2: H - B, class: "lg-cursor", style: "display:none"});
    svg.append(cursor);
    const say = (i) => read.textContent = `Step ${pts[i][0]} · epoch ${pts[i][1]} · loss ${pts[i][2].toFixed(4)} ` +
      `(smoothed ${sm[i].toFixed(4)})` + (pts[i][3] != null ? ` · LR ${Number(pts[i][3]).toExponential(2)}` : "");
    const base = () => read.textContent = `${data.steps} steps · lowest smoothed loss ${sm[bi].toFixed(4)} at step ` +
      `${pts[bi][0]} (epoch ${pts[bi][1]})` + (data.stride > 1 ? ` · every ${data.stride}th step drawn` : "");
    svg.addEventListener("pointermove", (ev) => {
      const r = svg.getBoundingClientRect(), x = x0 + (x1 - x0) * Math.min(1, Math.max(0, ((ev.clientX - r.left) / r.width * W - L) / (W - L - R)));
      let i = 0;
      while (i < pts.length - 1 && pts[i + 1][0] <= x) i++;
      cursor.setAttribute("x1", X(pts[i][0])); cursor.setAttribute("x2", X(pts[i][0])); cursor.style.display = "";
      say(i);
    });
    svg.addEventListener("pointerleave", () => { cursor.style.display = "none"; base(); });
    svgHost.replaceChildren(svg);
    base();
  }
  requestAnimationFrame(draw);               // once it is on the page and has a width
  let lastW = 0;
  new ResizeObserver(() => { const w = svgHost.clientWidth; if (w && Math.abs(w - lastW) > 8) { lastW = w; draw(); } })
    .observe(svgHost);
  return box;
}

// the run card's graph: the run training now (here or in the desktop app), every 10 s while it trains
async function runLossTick(training) {
  const now = Date.now();
  if (!training && S.lossDone) return;
  if (training && now - (S.lossAt || 0) < 10000) return;
  S.lossAt = now; S.lossDone = !training;
  const d = await api("/api/runs/loss?live=1").catch(() => null);
  sideSpark(d);
  const host = $("runLoss");
  if (!d || !d.points || d.points.length < 2) { host.replaceChildren(); return; }
  host.replaceChildren(lossGraph(d));
}

// a form value as the tab shows it: the row's default while it's never been set
function valueOrDefault(key) {
  if (S.values[key] !== undefined) return S.values[key];
  for (const s of S.fam.form.sections) for (const r of s.rows) if (r && r.key === key) return r.default;
  return undefined;
}

// the Samples tab's "Change the previews of the run going now": its prompts, seed, size, steps, CFG and Turbo
// strength go to the run as the
// trainer's sample override while ticked (sent again on every change), and come off when it's unticked
const sampleLiveSoon = debounce(async () => {
  const v = S.values, note = $("note___SAMPLE_LIVE");
  try {
    const r = await api("/api/run/sample_override", {on: !!v.__SAMPLE_LIVE, prompts: v.SAMPLE_PROMPT || "",
      seed: v.SAMPLE_SEED, width: v.SAMPLE_WIDTH, height: v.SAMPLE_HEIGHT, steps: v.SAMPLE_STEPS,
      cfg: v.SAMPLE_CFG_SCALE, turbo: v.FAMILY_TURBO_STRENGTH, every: v.SAMPLE_EVERY_N_EPOCHS,
      until: v.SAMPLE_LATER ? v.SAMPLE_UNTIL_EPOCH : 0, after: v.SAMPLE_LATER ? v.SAMPLE_EVERY_AFTER : 0,
      look: !!v.FAMILY_PREVIEW_LOOK, checkpoint: !!valueOrDefault("SAMPLE_CHECKPOINT")});
    if (note) note.textContent = r.on ? "Sent: the run's next previews use these." : "The run is back on its own prompts.";
  } catch (e) {
    if (note) note.textContent = S.run && (S.run.running || S.run.state === "starting") ? e.message : liveIdleText();
  }
}, 700);

// the tick's note follows the run: what a change would do now, never an old message from before the run started
const liveIdleText = () => "On: a change here goes to a run while it trains, from its next preview.";
function liveNote(live) {
  const note = $("note___SAMPLE_LIVE");
  if (!note || S._liveNoteFor === live) return;
  S._liveNoteFor = live;
  note.textContent = !S.values.__SAMPLE_LIVE ? "" : live ?
    "On: a change here goes to the run training now, from its next preview." : liveIdleText();
}

// a run's model, training folder and settings into the form (Start again, a queued run's Edit)
async function loadIntoForm(s) {
  if (s.family !== S.family) await chooseFamily(s.family);
  const kept = Object.fromEntries(Object.entries(s.values || {}).filter(([, v]) => v != null));
  S.values = {...S.values, ...kept};
  // a run recorded before it had a Turbo value: its steps and CFG follow the Turbo on now (refreshState)
  if (!("FAMILY_TURBO_STRENGTH" in kept)) delete S.values.__turbo_live;
  S.all[S.family] = S.values;
  if (s.image_folder) {
    const was = S.imageFolder;
    $("imageFolder").value = s.image_folder; S.imageFolder = s.image_folder; await loadTrigger();
    if (S.imageFolder !== was) lookForget();
  }
  renderSections("samples", $("samplesForm"));
  renderSections("training", $("trainingForm"));
  await refreshState();
  saveState();
  showTab("training");
}

// a queued run being edited: Add to queue saves it back in its place
// editing a queued run: a bar at the bottom of the window, on every tab, with Save to queue and Cancel edit, until
// one of them (or a Start) ends it; the Training tab's Add to queue steps aside meanwhile
function queueEditing(item) {
  S.editQueue = item;
  $("btnQueue").textContent = item ? `Save to queue (run ${item.n})` : "Add to queue";
  $("btnQueue").style.display = item ? "none" : "";
  $("queueEditBar").style.display = item ? "" : "none";
  $("queueEditText").replaceChildren(...(item ? [el("b", {text: `Editing queued run ${item.n}`}),
    el("span", {text: (item.lora ? `: ${item.lora}. ` : ". ") + "Change what you like on the tabs, then Save to " +
      "queue puts it back in its place. Cancel edit leaves it as it was."})] : []));
}

// a run's previews and checkpoints, from its output folder
async function runFiles(run) {
  const out = [];
  const sep = (run.output_dir || "").includes("\\") ? "\\" : "/";
  const [g, c] = await Promise.all([
    api(`/api/samples?dir=${encodeURIComponent(run.output_dir + sep + "sample")}&run=${encodeURIComponent(run.lora)}`)
      .catch(() => ({items: [], prompts: []})),
    api("/api/checkpoints?dir=" + encodeURIComponent(run.output_dir)).catch(() => ({items: []}))]);
  const shots = g.items.slice(0, 12);
  if (shots.length) out.push(el("div", {class: "gallery run-gallery"}, shots.map((it) => previewFigure(withLora(it, g.loras, run.family), g.prompts, null, g.epoch_prompts))));
  const size = (b) => b >= 2 ** 30 ? (b / 2 ** 30).toFixed(1) + " GB" : (b / 2 ** 20).toFixed(0) + " MB";
  const ckpts = c.items.filter((it) => it.run === run.lora);
  if (ckpts.length) out.push(el("div", {class: "checkpoint-list"}, ckpts.map((it) => el("div", {class: "checkpoint-row"},
    el("span", {class: "checkpoint-name", text: it.name}),
    el("span", {class: "field-hint", text: `${it.kind === "finetune" ? "Fine-tune" : "LoRA"} · ${size(it.size)}`}),
    it.kind === "finetune" ? null : studioButton(it.path, run.family),
    el("a", {class: "btn btn-sm btn-primary", download: it.name, text: "Download",
      href: `/api/download?path=${encodeURIComponent(it.path)}&t=${encodeURIComponent(TOKEN)}`})))));
  if (!out.length) out.push(el("div", {class: "field-hint", text: "Nothing saved by this run in " + run.output_dir + "."}));
  return out;
}

// ---- Download your outputs tab ------------------------------------------------------------------------------
async function loadOutputs() {
  const all = !$("outFizgigOnly").checked;
  let r;
  try { r = await api("/api/outputs" + (all ? "?all=1" : "")); } catch (e) { $("outSummary").textContent = e.message; return; }
  S.outputs = r;
  renderOutputs();
}

function renderOutputs() {
  const r = S.outputs;
  if (!r) return;
  const q = $("outSearch").value.trim().toLowerCase();
  // saved when: the last N days, or between two dates (the end date counts in full); a run keeps its files in range
  const when$ = $("outWhen").value;
  let lo = 0, hi = Infinity;
  if (when$ === "custom") {
    if ($("outFrom").value) lo = new Date($("outFrom").value + "T00:00:00").getTime() / 1000;
    if ($("outTo").value) hi = new Date($("outTo").value + "T23:59:59").getTime() / 1000;
  } else if (when$ !== "any") lo = Date.now() / 1000 - Number(when$) * 86400;
  const inRange = (f) => f.time >= lo && f.time <= hi;
  const runs = r.runs
    .filter((x) => !q || (x.name + " " + x.folder).toLowerCase().includes(q))
    .map((x) => ({...x, files: x.files.filter(inRange)}))
    .filter((x) => x.files.length);
  const size = (b) => b >= 2 ** 30 ? (b / 2 ** 30).toFixed(1) + " GB" : (b / 2 ** 20).toFixed(0) + " MB";
  const when = (t) => new Date(t * 1000).toLocaleString();
  $("outSummary").textContent = `${r.runs.length} run${r.runs.length === 1 ? "" : "s"} in ${r.folders.length} ` +
    `folder${r.folders.length === 1 ? "" : "s"}` + (q || when$ !== "any" ? ` · ${runs.length} shown` : "") +
    (r.hidden ? ` · ${r.hidden} other LoRA${r.hidden === 1 ? "" : "s"} in these folders hidden` : "") + ".";
  const host = $("outRuns");
  host.innerHTML = "";
  const dl = (f) => el("a", {class: "btn btn-sm btn-primary", download: f.name, text: "Download",
    href: `/api/download?path=${encodeURIComponent(f.path)}&t=${encodeURIComponent(TOKEN)}`});
  const row = (f) => el("div", {class: "checkpoint-row"},
    el("span", {class: "checkpoint-name", text: f.name}),
    el("span", {class: "field-hint", text: `${f.kind === "finetune" ? "Fine-tune (whole model)" : "LoRA"} · ` +
      `${size(f.size)} · ${when(f.time)}`}), dl(f));
  for (const run of runs.slice(0, 200)) {
    const [top, ...rest] = run.files;
    const more = el("div", {class: "checkpoint-list", style: "display:none"}, rest.map(row));
    const body = el("div", {class: "card-content"}, el("div", {class: "checkpoint-list"}, row(top)));
    if (rest.length) body.append(el("button", {class: "btn btn-sm", text: `Show ${rest.length} earlier checkpoint` +
      (rest.length === 1 ? "" : "s"), onclick: (e) => {
        const open = more.style.display === "none";
        more.style.display = open ? "" : "none";
        e.target.textContent = open ? "Hide the earlier checkpoints" : `Show ${rest.length} earlier checkpoint` +
          (rest.length === 1 ? "" : "s");
      }}), more);
    if (run.samples) body.append(el("div", {class: "field-row"}, el("a", {class: "btn btn-sm",
      text: `Download the previews (${run.samples} files, ${size(run.samples_bytes || 0)}, .zip)`,
      href: `/api/outputs/previews?dir=${encodeURIComponent(run.folder)}&name=${encodeURIComponent(run.name)}` +
        `&t=${encodeURIComponent(TOKEN)}`})));
    host.append(el("div", {class: "section-card"},
      el("div", {class: "card-title"}, el("span", {text: run.name})),
      el("div", {class: "card-desc", text: `${run.folder} · last saved ${when(run.time)}`}), body));
  }
  if (!runs.length) host.append(el("div", {class: "section-card"}, el("div", {class: "card-content"},
    el("div", {class: "field-hint", text: r.runs.length ? "Nothing matches that search and date." :
      "Nothing saved yet. A run's LoRAs and checkpoints appear here as it saves them."}))));
}

// a song's waveform (the picture beside it with the same name) above its player, and its length; any other media
// as before
const fmtSecs = (s) => s == null ? "" : `${Math.floor(s / 60)}:${String(Math.round(s % 60)).padStart(2, "0")}`;
function mediaFor(it, eager) {
  // its modified time in the address: the browser keeps what it fetched (a clip played again isn't downloaded again)
  const q = (p) => `/api/media?path=${encodeURIComponent(p)}&t=${encodeURIComponent(TOKEN)}` +
    (p === it.path && it.time ? `&v=${encodeURIComponent(it.time)}` : "");
  const kind = it.name.split(".").pop().toLowerCase();
  if (["wav", "mp3", "flac", "m4a"].includes(kind)) {
    // nothing downloads until it plays (a page of songs is hundreds of MB); the length comes from the server
    const audio = el("audio", {controls: true, preload: "none", src: q(it.path)});
    const wave = el("div", {class: "song-wave-box", title: "Play / pause",
      onclick: () => (audio.paused ? audio.play() : audio.pause())},
      it.image ? el("img", {loading: "lazy", src: q(it.image), alt: "", class: "song-wave"}) :
        el("div", {class: "song-wave song-wave-none", text: "♪"}),
      it.duration != null ? el("span", {class: "song-len", text: fmtSecs(it.duration)}) : null);
    return el("div", {class: "song-media"}, wave, audio);
  }
  // a clip shows its still (a preview's middle frame) until it plays. eager: it downloads now (the newest previews);
  // else only its header, and the rest once the pointer (or keyboard) comes to it, so Play rarely waits on a pod
  if (["mp4", "webm"].includes(kind)) {
    const v = el("video", {controls: true, preload: eager ? "auto" : "metadata", src: q(it.path),
      ...(it.image ? {poster: q(it.image)} : {})});
    if (!eager) for (const ev of ["pointerenter", "focus"]) v.addEventListener(ev, () => { v.preload = "auto"; }, {once: true});
    return v;
  }
  return el("img", {loading: "lazy", src: q(it.path), alt: it.name});
}
const mediaLabel = (it) => it.name;

// ---- Image Prep tab: resize / PNG / face close-ups, and the Look Filter (fizgig/web/prep_jobs.py) -------------
S.prep = {opts: null, logSize: 0, timer: null, look: null, marked: new Set(), bases: [], page: 0, picking: false};
const thumbOf = (p) => `/api/thumb?path=${encodeURIComponent(p)}&t=${encodeURIComponent(TOKEN)}`;
const prepSettings = () => ({
  mode: (document.querySelector('input[name="prepMode"]:checked') || {}).value || "Auto Prep (Face Crops)",
  mp: $("prepMp").value || "1.0", face: $("prepFace").value || "Largest Face", padding: $("prepPadding").value || "20",
  replace: (document.querySelector('input[name="prepReplace"]:checked') || {}).value === "1",
  crops_only: $("prepCropsOnly").checked});
const savePrepSoon = debounce(() => { S.prepSaved = prepSettings(); api("/api/state", {prep: S.prepSaved}).catch(() => {}); }, 600);

async function loadPrep() {
  if (!S.prep.opts) {
    S.prep.opts = await api("/api/prep/options");
    const o = S.prep.opts, saved = S.prepSaved || {};
    $("prepMp").innerHTML = ""; for (const m of o.megapixels) $("prepMp").append(el("option", {value: m, text: m}));
    $("prepFace").innerHTML = ""; for (const f of o.faces) $("prepFace").append(el("option", {value: f, text: f}));
    $("prepMp").value = saved.mp || "1.0";
    $("prepFace").value = saved.face || "Largest Face";
    if (saved.padding) $("prepPadding").value = saved.padding;
    const mode = saved.mode || "Auto Prep (Face Crops)";
    document.querySelectorAll('input[name="prepMode"]').forEach((r) => { r.checked = r.value === mode; });
    document.querySelectorAll('input[name="prepReplace"]').forEach((r) => { r.checked = r.value === (saved.replace ? "1" : "0"); });
    $("prepCropsOnly").checked = !!saved.crops_only;
    $("prepNoFaces").style.display = o.face_tools ? "none" : "";
    faceSettingsShow(o.face_settings);
  }
  $("prepFolder").textContent = (S.imageFolder ? `Folder: ${S.imageFolder}  (set on the Start tab)` :
    "Choose the training folder on the Start tab first.") + subfoldersHint();
  await prepSummary();
  pollPrep();
  lookPoll();
  maskInfo();
}

// Image Prep's Make masks (fizgig/dataset/automask.py through prep_jobs.start_masks): the same job and log as a prep run
const MASK_HINTS = {
  person: "For a person LoRA: the room and the background count far less.",
  no_clothes: "For a person LoRA that shouldn't learn their outfits: their clothes and the background both count far less.",
  body: "For a body or skin LoRA (skin texture, tattoos) that shouldn't learn the face: face, hair, clothes and background count far less.",
  body_clothes: "For an outfit or body LoRA that shouldn't learn the face: face, hair and background count far less.",
  head: "For likeness only: the body, clothes and background all count far less.",
  clothes: "For an outfit LoRA: the person wearing it and the background count far less.",
  not_faces: "For an outfit, a body or a style LoRA that shouldn't learn anyone's face.",
  background: "For a place or scene LoRA: the people in the pictures count far less."};
async function maskInfo() {
  if (!S.imageFolder) { $("maskInfo").textContent = ""; return; }
  // the editor follows the folder, and the masks a run just made
  if ($("maskEdit").open && S.me && !S.me.dirty && (S.me.folder !== S.imageFolder || S.me.wantList)) {
    S.me.wantList = false; meOpen();
  }
  const r = await api("/api/masks/info?folder=" + encodeURIComponent(S.imageFolder)).catch(() => null);
  if (!r) return;
  if (!$("maskKinds").childElementCount) {
    const sv = S.prepSaved || {};
    $("maskSam").checked = !!sv.maskSam;
    if (sv.maskSamInclude != null) $("maskSamInclude").value = sv.maskSamInclude;
    if (sv.maskSamExclude != null) $("maskSamExclude").value = sv.maskSamExclude;
    $("maskMethod").querySelectorAll("button").forEach((b) => b.addEventListener("click", () => {
      $("maskSam").checked = b.dataset.sam === "1"; maskSamSave(); maskShow(S.prep.maskOpts || r); }));
    $("maskSamInclude").addEventListener("change", maskSamSave);
    $("maskSamExclude").addEventListener("change", maskSamSave);
    $("maskSamPreview").addEventListener("click", maskSamPreview);
    const saved = (S.prepSaved || {}).maskKind || "person";
    for (const [k, label] of Object.entries(r.kinds)) {
      $("maskKinds").append(el("label", {class: "radio-label"}, el("input", {type: "radio", name: "maskKind", value: k,
        checked: k === saved, onchange: () => { S.prepSaved = {...(S.prepSaved || {}), maskKind: k};
          api("/api/state", {prep: S.prepSaved}).catch(() => {}); maskShow(r); }}), " " + label),
        el("div", {class: "field-hint radio-hint", text: MASK_HINTS[k] || ""}));
    }
  }
  S.prep.maskOpts = r;
  maskShow(r);
}
// A model download's failure, for the hint beside its button: an expired Hugging Face login reads as itself.
function downloadFailedText(err) {
  return /^Your Hugging Face login/.test(err || "") ? ` ${err}` : ` The download failed: ${err}`;
}

// SAM 3.1 on the masks card: words instead of the person finder's choices; its file is downloaded from here
function maskSamShow(r) {
  const on = $("maskSam").checked, sam = r.sam || {}, f = sam.fetch || {};
  $("maskSamBox").style.display = on ? "" : "none";
  $("maskKinds").style.display = on ? "none" : "";
  $("maskMethod").querySelectorAll("button").forEach((b) => b.classList.toggle("active", (b.dataset.sam === "1") === on));
  $("maskMethodHint").textContent = on ? "Meta's Segment Anything: type what to mask (\"person\", \"hair\", " +
    "\"the dog\") and it finds it, with sharper edges. A 1.75 GB download, once; it runs on the GPU." :
    "Google's person finder: pick a part of the person below. Small and quick, on the CPU. For an object, an animal or anything else, use SAM 3.1.";
  $("maskRunHint").textContent = on ? "It runs on the GPU, a fraction of a second a picture once loaded (several " +
    "seconds a picture without a GPU). Make masks when nothing is training." :
    "It runs on the CPU, a fraction of a second a picture. The first time, it downloads Google's person finder (16 MB).";
  const st = $("maskSamState");
  st.style.display = on ? "" : "none";
  if (f.state === "running") {
    st.replaceChildren(`Downloading SAM 3.1: ${f.gb} of ${f.of} GB …`);
    clearTimeout(S.prep.samTimer);
    S.prep.samTimer = setTimeout(maskInfo, 2000);
  } else if (sam.path) {
    st.replaceChildren("SAM 3.1 is downloaded.");
  } else {
    st.replaceChildren(el("button", {class: "btn btn-primary",
      text: f.state === "failed" ? "Try the download again" : `Download SAM 3.1 (${sam.gb || 1.75} GB)`, onclick: async () => {
        await api("/api/masks/sam_fetch", {}).catch((e) => { st.textContent = e.message; });
        maskInfo(); }}), el("span", {class: "field-hint", text: f.state === "failed" ? downloadFailedText(f.error) :
        " Needed once, before the first masks."}));
  }
  $("maskSamPreview").disabled = !on || !sam.path || !r.pictures || S.prep.running;
}
async function maskSamPreview() {
  $("maskSamPrevState").textContent = "Finding them (the first time loads SAM 3.1, a few seconds) …";
  $("maskSamPreview").disabled = true;
  try {
    // two pictures not shown yet in this folder, until every one has been: then a fresh round
    if (S.prep.samShownFor !== S.imageFolder) { S.prep.samShown = []; S.prep.samShownFor = S.imageFolder; }
    const r = await api("/api/masks/sam_preview", {folder: S.imageFolder, include: $("maskSamInclude").value,
      exclude: $("maskSamExclude").value, shown: S.prep.samShown});
    S.prep.samShown = r.fresh ? r.fresh : [...S.prep.samShown, ...r.pictures.map((p) => p.name)];
    $("maskSamPreviews").replaceChildren(...r.pictures.map((p) => el("figure", {class: "mask-sam-shot"},
      el("img", {src: p.src, alt: p.name}),
      el("figcaption", {text: p.cover ? `${p.name} · ${Math.round(p.cover * 100)}% masked` : `${p.name} · nothing found`}))));
    $("maskSamPrevState").textContent = "Blue is what the LoRA learns; nothing is saved yet. Press again for two " +
      `more (${Math.min(S.prep.samShown.length, r.of)} of ${r.of} seen).`;
  } catch (e) { $("maskSamPrevState").textContent = e.message; }
  $("maskSamPreview").disabled = false;
}
function maskSamSave() {
  S.prepSaved = {...(S.prepSaved || {}), maskSam: $("maskSam").checked, maskSamInclude: $("maskSamInclude").value,
    maskSamExclude: $("maskSamExclude").value};
  api("/api/state", {prep: S.prepSaved}).catch(() => {});
}
function maskShow(r) {
  const kind = (document.querySelector('input[name="maskKind"]:checked') || {}).value || "person";
  maskSamShow(r);
  const sam = $("maskSam").checked;
  $("maskInfo").textContent = !r.available && !sam ? "Making masks needs an update: run update_fizgig to add its one new " +
    "package (ai-edge-litert)." : r.pictures ? `${r.with_mask} of ${r.pictures} pictures have a mask.` : "";
  const subj = $("maskSubject");
  subj.disabled = !r.baselines || !r.faces || (!sam && (kind === "not_faces" || kind === "background"));
  if (subj.disabled) subj.checked = false;
  $("maskSubjectHint").textContent = sam ? (!r.faces ? "Needs face detection, which isn't installed here (run install_fizgig.py)." :
    r.baselines ? "Finds your subject by their face, with the Look Filter's 3 baselines, and keeps only what's on them." :
    "Needs the Look Filter's 3 baselines: choose them and scan the folder (the last card on this tab), then come back.") :
    kind === "not_faces" ? "Not for Everything but the faces: every face is left out." :
    kind === "background" ? "Not for Just the background: every person is left out." :
    !r.faces ? "Needs face detection, which isn't installed here (run install_fizgig.py)." :
    r.baselines ? "Finds your subject by their face, with the Look Filter's 3 baselines; everyone else counts as background." :
    "Needs the Look Filter's 3 baselines: choose them and scan the folder (the last card on this tab), then come back.";
  $("maskRun").disabled = (sam ? !(r.sam || {}).path : !r.available) || !r.pictures || S.prep.running;
}
async function maskRun() {
  if (!S.imageFolder) { $("maskState").textContent = "Choose the training folder on the Start tab first."; return; }
  const kind = (document.querySelector('input[name="maskKind"]:checked') || {}).value || "person";
  try {
    await api("/api/masks/make", {folder: S.imageFolder, kind, subject_only: $("maskSubject").checked,
      sam: $("maskSam").checked, include: $("maskSamInclude").value, exclude: $("maskSamExclude").value,
      overwrite: $("maskReplace").checked});
    S.prep.masking = true;
    $("maskLog").textContent = ""; $("maskLog").style.display = "";
    $("prepLog").textContent = ""; S.prep.logSize = 0;
    pollPrep();
  } catch (e) { $("maskState").textContent = e.message; }
}

// ---- Image Prep's mask editor (web/mask_edit.py): a picture at a time, its mask painted on a working canvas (at most
// 4 MP: training reads masks at the latent grid's size, far below that) and saved back at the picture's own size
const ME_MAX_PX = 4e6, ME_UNDO = 20;
function meState() {
  if (!S.me) S.me = {list: [], at: -1, name: "", pic: null, mask: null, dirty: false, undo: [], redo: [],
                     tool: "brush", view: "over", zoom: 1, ox: 0, oy: 0, folder: ""};
  return S.me;
}
function meShown() {
  const m = meState(), f = $("meFilter").value;
  return m.list.filter((p) => f === "all" || (f === "mask") === p.mask);
}
async function meOpen() {
  const m = meState();
  if (!S.imageFolder) { $("meState").textContent = "Choose the training folder on the Start tab first."; return; }
  try {
    const r = await api("/api/masks/edit/list?folder=" + encodeURIComponent(S.imageFolder));
    m.list = r.pictures; m.folder = S.imageFolder;
  } catch (e) { $("meState").textContent = e.message; return; }
  const shown = meShown();
  const keep = shown.findIndex((p) => p.name === m.name);
  await meGo(keep >= 0 ? keep : 0, true);
}
async function meGo(i, force) {
  const m = meState(), shown = meShown();
  if (!shown.length) {
    m.at = -1; m.name = ""; m.pic = null;
    $("meCount").textContent = "0 of 0"; $("meName").textContent = "No pictures here.";
    meDraw(); return;
  }
  i = (i + shown.length) % shown.length;
  if (!force && shown[i].name === m.name) return;
  if (m.dirty && !(await meSave())) return;           // moving on saves the one you were on
  m.at = i; m.name = shown[i].name;
  $("meCount").textContent = `${i + 1} of ${shown.length}`;
  $("meName").textContent = m.name;
  $("meState").textContent = "Loading…";
  const q = `folder=${encodeURIComponent(S.imageFolder)}&name=${encodeURIComponent(m.name)}&t=${encodeURIComponent(TOKEN)}`;
  const pic = await meImage("/api/masks/edit/picture?" + q).catch(() => null);
  if (!pic || m.name !== shown[i].name) { $("meState").textContent = pic ? "" : "Couldn't load the picture."; return; }
  const k = Math.min(1, Math.sqrt(ME_MAX_PX / (pic.naturalWidth * pic.naturalHeight)));
  const w = Math.max(1, Math.round(pic.naturalWidth * k)), h = Math.max(1, Math.round(pic.naturalHeight * k));
  const mask = el("canvas", {width: w, height: h});
  if (shown[i].mask) {
    const mimg = await meImage("/api/masks/edit/mask?" + q + "&v=" + Date.now()).catch(() => null);
    if (mimg) meFromGray(mask, mimg);
  }
  m.pic = pic; m.mask = mask; m.dirty = false; m.undo = []; m.redo = [];
  meStateLine();
  $("meState").textContent = shown[i].mask ? "" : "No mask yet: paint one with the Brush.";
  meFit();
}
function meImage(src) {
  return new Promise((ok, bad) => { const im = new Image(); im.onload = () => ok(im); im.onerror = bad; im.src = src; });
}
function meFromGray(canvas, img) {
  // the saved mask (grey = how much is learned) -> the working canvas: white, with that much opacity
  const c = canvas.getContext("2d");
  c.drawImage(img, 0, 0, canvas.width, canvas.height);
  const d = c.getImageData(0, 0, canvas.width, canvas.height), a = d.data;
  for (let p = 0; p < a.length; p += 4) { const v = a[p]; a[p] = a[p + 1] = a[p + 2] = 255; a[p + 3] = v; }
  c.putImageData(d, 0, 0);
}
function meFit() {
  const m = meState(), cv = $("meCanvas"), st = $("meStage");
  cv.width = st.clientWidth || 800;
  cv.height = Math.max(300, Math.min(window.innerHeight * 0.7, 900));
  if (m.mask) {
    m.zoom = Math.min(cv.width / m.mask.width, cv.height / m.mask.height);
    m.ox = (cv.width - m.mask.width * m.zoom) / 2; m.oy = (cv.height - m.mask.height * m.zoom) / 2;
  }
  meDraw();
}
function meLayers() {
  // the mask as two layers over the picture: blue where it's learned, black where it isn't (each as much as the mask)
  const m = meState(), w = m.mask.width, h = m.mask.height;
  if (!m.tint || m.tint.width !== w || m.tint.height !== h) {
    m.tint = el("canvas", {width: w, height: h}); m.dim = el("canvas", {width: w, height: h});
  }
  const t = m.tint.getContext("2d");
  t.globalCompositeOperation = "copy"; t.drawImage(m.mask, 0, 0);
  t.globalCompositeOperation = "source-in"; t.fillStyle = themeColor("--pic-mask"); t.fillRect(0, 0, w, h);
  t.globalCompositeOperation = "source-over";
  const d = m.dim.getContext("2d");
  d.globalCompositeOperation = "copy"; d.fillStyle = themeColor("--pic-shade"); d.fillRect(0, 0, w, h);
  d.globalCompositeOperation = "destination-out"; d.drawImage(m.mask, 0, 0);
  d.globalCompositeOperation = "source-over";
  return [m.tint, m.dim];
}
// the mask's own pixels stay white (learned) and black (not): they are the mask, not the theme
function meDraw() {
  const m = meState(), cv = $("meCanvas"), c = cv.getContext("2d");
  c.setTransform(1, 0, 0, 1, 0, 0);
  c.fillStyle = themeColor("--bg-inset"); c.fillRect(0, 0, cv.width, cv.height);
  if (!m.pic || !m.mask) return;
  c.setTransform(m.zoom, 0, 0, m.zoom, m.ox, m.oy);
  c.imageSmoothingEnabled = m.zoom < 2;
  const w = m.mask.width, h = m.mask.height, show = Number($("meShow").value) / 100;
  if (m.view === "mask") {
    c.fillStyle = "#000"; c.fillRect(0, 0, w, h); c.drawImage(m.mask, 0, 0);
  } else {
    c.drawImage(m.pic, 0, 0, w, h);
    if (m.view === "over") {
      const [tint, dim] = meLayers();
      c.globalAlpha = show * 0.8; c.drawImage(dim, 0, 0);       // what isn't learned dimmed...
      c.globalAlpha = show * 0.6; c.drawImage(tint, 0, 0);      // ...and what is, tinted blue
      c.globalAlpha = 1;
    }
  }
  if (m.samAt) {                                       // SAM at work: a ring pulsing at the click
    const ph = (performance.now() % 1000) / 1000;
    c.setTransform(1, 0, 0, 1, 0, 0);
    c.strokeStyle = themeColor("--pic-mask", 1 - ph); c.lineWidth = 3;
    c.beginPath(); c.arc(m.samAt[0], m.samAt[1], 6 + ph * 22, 0, Math.PI * 2); c.stroke();
    c.fillStyle = themeColor("--pic-mask"); c.beginPath(); c.arc(m.samAt[0], m.samAt[1], 4, 0, Math.PI * 2); c.fill();
  }
  if (m.cursor && !m.panning && m.tool !== "sam") {    // the brush, its size on screen
    // drawn as the difference with what's under it (XOR-like), so it shows on white, black and any picture (#271)
    c.setTransform(1, 0, 0, 1, 0, 0);
    c.globalCompositeOperation = "difference"; c.strokeStyle = "#fff"; c.lineWidth = 1.5;
    c.beginPath(); c.arc(m.cursor[0], m.cursor[1], Number($("meSize").value) / 2, 0, Math.PI * 2); c.stroke();
    c.globalCompositeOperation = "source-over";
  }
}
function meAt(e) {
  const m = meState(), r = $("meCanvas").getBoundingClientRect();
  const sx = (e.clientX - r.left) * ($("meCanvas").width / r.width), sy = (e.clientY - r.top) * ($("meCanvas").height / r.height);
  return {sx, sy, x: (sx - m.ox) / m.zoom, y: (sy - m.oy) / m.zoom};
}
function meSnapshot() {
  const m = meState(), copy = el("canvas", {width: m.mask.width, height: m.mask.height});
  copy.getContext("2d").drawImage(m.mask, 0, 0);
  return copy;
}
function mePush() {
  const m = meState();
  m.undo.push(meSnapshot()); if (m.undo.length > ME_UNDO) m.undo.shift();
  m.redo = [];
}
function meRestore(from, to) {
  const m = meState();
  if (!from.length) return;
  to.push(meSnapshot());
  const c = m.mask.getContext("2d");
  c.globalCompositeOperation = "copy"; c.drawImage(from.pop(), 0, 0); c.globalCompositeOperation = "source-over";
  m.dirty = true; meStateLine(); meDraw();
}
function meDab(x, y) {
  // one stamp of the brush on the mask: white with the strength's opacity, or that much taken away
  const m = meState(), c = m.mask.getContext("2d");
  const r = Number($("meSize").value) / 2 / m.zoom, soft = Number($("meSoft").value) / 100;
  const a = Number($("meStrength").value) / 100;
  const g = c.createRadialGradient(x, y, r * (1 - soft) * 0.999, x, y, r);
  g.addColorStop(0, `rgba(255,255,255,${a})`); g.addColorStop(1, "rgba(255,255,255,0)");
  // the Eraser takes away, the Brush adds; a right-click stroke does the other (#271)
  c.globalCompositeOperation = (m.tool === "erase") !== !!m.flip ? "destination-out" : "source-over";
  c.fillStyle = soft > 0 ? g : `rgba(255,255,255,${a})`;
  c.beginPath(); c.arc(x, y, r, 0, Math.PI * 2); c.fill();
  c.globalCompositeOperation = "source-over";
}
function meStroke(from, to) {
  const m = meState(), r = Math.max(0.5, Number($("meSize").value) / 2 / m.zoom);
  const dx = to.x - from.x, dy = to.y - from.y, n = Math.max(1, Math.ceil(Math.hypot(dx, dy) / (r * 0.3)));
  for (let k = 1; k <= n; k++) meDab(from.x + dx * k / n, from.y + dy * k / n);
}
function meStateLine() {
  // the save status beside Save now: unsaved changes, saved, or no mask on this picture yet
  const m = meState(), b = $("meSaved"), item = m.list.find((p) => p.name === m.name);
  const st = !m.name ? "" : m.dirty ? "unsaved" : item && item.mask ? "saved" : "none";
  b.dataset.state = st;
  b.textContent = {unsaved: "● Unsaved changes", saved: "✓ Saved", none: "No mask yet", "": ""}[st];
  $("meSave").disabled = !m.dirty;
  $("meRevert").disabled = !m.dirty;
}
async function meSave() {
  const m = meState();
  if (!m.mask || !m.name) return true;
  // the mask on opaque black, so grey is grey however it was painted
  const out = el("canvas", {width: m.mask.width, height: m.mask.height}), c = out.getContext("2d");
  c.fillStyle = "#000"; c.fillRect(0, 0, out.width, out.height); c.drawImage(m.mask, 0, 0);
  try {
    const r = await api("/api/masks/edit/save", {folder: m.folder || S.imageFolder, name: m.name,
      png: out.toDataURL("image/png")});
    m.dirty = false;
    const item = m.list.find((p) => p.name === m.name);
    if (item) item.mask = !r.removed;
    meStateLine();
    $("meState").textContent = r.removed ? "All black, so the mask was removed: this picture trains as normal." :
      `Saved ${m.name.replace(/\.[^.]+$/, "")}-masklabel.png.`;
    maskInfo();
    return true;
  } catch (e) { $("meState").textContent = e.message; return false; }
}
function meWire() {
  const m = meState(), cv = $("meCanvas");
  let last = null, pan = null, space = false;
  $("maskEdit").addEventListener("toggle", () => {
    if ($("maskEdit").open) meOpen(); else if (meState().dirty) meSave();      // closing it saves
  });
  window.addEventListener("beforeunload", (e) => {   // closing the window with edits not saved: ask first
    if (S.me && S.me.dirty) { e.preventDefault(); e.returnValue = ""; }
  });
  $("mePrev").addEventListener("click", () => meGo(meState().at - 1));
  $("meNext").addEventListener("click", () => meGo(meState().at + 1));
  $("meFilter").addEventListener("change", () => meGo(0, true));
  $("meTool").querySelectorAll("button").forEach((b) => b.addEventListener("click", () => meTool(b.dataset.tool)));
  $("meView").querySelectorAll("button").forEach((b) => b.addEventListener("click", () => {
    m.view = b.dataset.view;
    $("meView").querySelectorAll("button").forEach((x) => x.classList.toggle("active", x === b));
    meDraw();
  }));
  $("meShow").addEventListener("input", meDraw);
  $("meFit").addEventListener("click", meFit);
  $("meUndo").addEventListener("click", () => meRestore(m.undo, m.redo));
  $("meRedo").addEventListener("click", () => meRestore(m.redo, m.undo));
  const whole = (fn) => () => { if (!m.mask) return; mePush(); fn(m.mask.getContext("2d")); m.dirty = true; meStateLine(); meDraw(); };
  $("meFill").addEventListener("click", whole((c) => { c.fillStyle = "#fff"; c.fillRect(0, 0, m.mask.width, m.mask.height); }));
  $("meClear").addEventListener("click", whole((c) => c.clearRect(0, 0, m.mask.width, m.mask.height)));
  $("meInvert").addEventListener("click", whole((c) => {
    const d = c.getImageData(0, 0, m.mask.width, m.mask.height), a = d.data;
    for (let p = 0; p < a.length; p += 4) { a[p] = a[p + 1] = a[p + 2] = 255; a[p + 3] = 255 - a[p + 3]; }
    c.putImageData(d, 0, 0);
  }));
  $("meSave").addEventListener("click", meSave);
  $("meRevert").addEventListener("click", () => { m.dirty = false; meGo(m.at, true); });
  cv.addEventListener("pointerdown", (e) => {
    if (!m.mask) return;
    cv.setPointerCapture(e.pointerId);
    const p = meAt(e);
    if (e.button === 1 || space) { pan = {sx: p.sx, sy: p.sy, ox: m.ox, oy: m.oy}; m.panning = true; e.preventDefault(); return; }
    if (m.tool === "sam") {                            // a click adds; right-click or Alt+click takes away
      if (e.button === 0 || e.button === 2) meSamClick(p, e.button === 2 || e.altKey);
      return;
    }
    if (e.button !== 0 && e.button !== 2) return;
    m.flip = e.button === 2;                           // right-click: the Brush erases, the Eraser paints
    mePush(); meDab(p.x, p.y); last = p; m.dirty = true; meStateLine(); meDraw();
  });
  cv.addEventListener("pointermove", (e) => {
    const p = meAt(e);
    m.cursor = [p.sx, p.sy];
    if (pan) { m.ox = pan.ox + p.sx - pan.sx; m.oy = pan.oy + p.sy - pan.sy; }
    else if (last) { meStroke(last, p); last = p; }
    meDraw();
  });
  const up = () => { last = null; pan = null; m.panning = false; m.flip = false; meDraw(); };
  cv.addEventListener("pointerup", up); cv.addEventListener("pointercancel", up);
  cv.addEventListener("pointerleave", () => { m.cursor = null; meDraw(); });
  cv.addEventListener("contextmenu", (e) => e.preventDefault());   // right-click: SAM's "take away", the brush's erase
  cv.addEventListener("wheel", (e) => {
    if (!m.mask) return;
    e.preventDefault();
    // zooming out stops where the whole picture fits (smaller is never useful), and centres it there
    const cv = $("meCanvas"), fit = Math.min(cv.width / m.mask.width, cv.height / m.mask.height);
    const p = meAt(e), k = Math.exp(-e.deltaY * 0.0015), z = Math.min(40, Math.max(fit, m.zoom * k));
    if (z <= fit) { meFit(); return; }
    m.ox = p.sx - (p.sx - m.ox) * (z / m.zoom); m.oy = p.sy - (p.sy - m.oy) * (z / m.zoom); m.zoom = z;
    meDraw();
  }, {passive: false});
  document.addEventListener("keydown", (e) => {
    if (!$("maskEdit").open || !$("tab-prep").classList.contains("active")) return;
    if (/^(INPUT|TEXTAREA|SELECT)$/.test(document.activeElement && document.activeElement.tagName) &&
        document.activeElement.type !== "range") return;
    const k = e.key;
    if (k === " ") { space = true; e.preventDefault(); }
    else if ((e.ctrlKey || e.metaKey) && k.toLowerCase() === "z") { e.preventDefault(); meRestore(m.undo, m.redo); }
    else if ((e.ctrlKey || e.metaKey) && k.toLowerCase() === "y") { e.preventDefault(); meRestore(m.redo, m.undo); }
    else if (k === "ArrowLeft") { e.preventDefault(); meGo(m.at - 1); }
    else if (k === "ArrowRight") { e.preventDefault(); meGo(m.at + 1); }
    else if (k === "[" || k === "]") {
      const s = $("meSize"), v = Number(s.value);
      s.value = Math.max(2, Math.min(400, Math.round(k === "[" ? v / 1.2 : v * 1.2 + 1))); meDraw();
    }
    else if (k === "b" || k === "B") meTool("brush");
    else if (k === "e" || k === "E") meTool("erase");
    else if (k === "s" || k === "S") meTool("sam");
    else if (k === "0") meFit();
  });
  document.addEventListener("keyup", (e) => { if (e.key === " ") space = false; });
  window.addEventListener("resize", () => { if ($("maskEdit").open) meFit(); });
}
async function meSamClick(p, takeAway) {
  // SAM 3.1 finds the thing under the click; it's added to the mask (or, right-click, taken out of it)
  const m = meState();
  if (m.samBusy || !m.mask || p.x < 0 || p.y < 0 || p.x >= m.mask.width || p.y >= m.mask.height) return;
  m.samBusy = true;
  const name = m.name;
  meBusy(m.samLoaded ? (takeAway ? "Finding what to take away…" : "Finding it…") :
    "Loading SAM 3.1… (the first click takes a few seconds)", [p.sx, p.sy]);
  $("meState").textContent = "";
  try {
    const r = await api("/api/masks/edit/sam_click", {folder: m.folder || S.imageFolder, name,
      x: p.x * m.pic.naturalWidth / m.mask.width, y: p.y * m.pic.naturalHeight / m.mask.height});
    if (m.name !== name) return;                       // moved on meanwhile
    const found = el("canvas", {width: m.mask.width, height: m.mask.height});
    meFromGray(found, await meImage(r.png));
    mePush();
    const c = m.mask.getContext("2d");
    c.globalCompositeOperation = takeAway ? "destination-out" : "source-over";
    c.drawImage(found, 0, 0);
    c.globalCompositeOperation = "source-over";
    m.dirty = true; m.samLoaded = true; meStateLine();
    $("meState").textContent = takeAway ? "Taken away." : "Added.";
  } catch (e) { $("meState").textContent = e.message; }
  finally { m.samBusy = false; meBusy(""); meDraw(); }
}
function meBusy(text, at) {
  // SAM at work: a line over the picture, a wait cursor, and a ring pulsing where you clicked
  const m = meState();
  $("meBusy").style.display = text ? "" : "none";
  $("meBusyText").textContent = text;
  $("meCanvas").style.cursor = text ? "wait" : (m.tool === "sam" ? "crosshair" : "none");
  m.samAt = text ? at : null;
  if (text) {
    const tick = () => { if (!meState().samAt) return; meDraw(); requestAnimationFrame(tick); };
    requestAnimationFrame(tick);
  }
}
function meTool(t) {
  meState().tool = t;
  $("meCanvas").style.cursor = t === "sam" ? "crosshair" : "none";
  $("meTool").querySelectorAll("button").forEach((b) => b.classList.toggle("active", b.dataset.tool === t));
}

async function prepSummary() {
  const p = prepSettings(), faceOn = ["Auto Prep (Face Crops)", "Face Crop Only"].includes(p.mode);
  document.querySelectorAll(".prep-face select, .prep-face input").forEach((i) => { i.disabled = !faceOn || !S.prep.opts.face_tools; });
  $("prepTest").disabled = !faceOn || !S.prep.opts.face_tools;
  // Detail crops leave the pictures alone and use no face: no face rows, no originals choice, the size is the crop's
  const crops = p.mode === "Detail Crops (Style)";
  for (const id of ["prepOriginals", "prepSizingNote", "prepChars"]) $(id).style.display = crops ? "none" : "";
  for (const id of ["prepCropsOnlyRow", "prepCropsOnlyHint"]) $(id).style.display = crops ? "" : "none";
  // the face row and the face test only for the modes that make face close-ups
  $("prepTestRow").style.display = faceOn ? "" : "none";
  document.querySelectorAll(".prep-face").forEach((r) => { r.style.display = faceOn ? "" : "none"; });
  if (crops) $("prepNoFaces").style.display = "none";
  else $("prepNoFaces").style.display = S.prep.opts.face_tools ? "none" : "";
  $("prepMpHint").textContent = crops ? "MP" : "MP (larger images shrink to fit; smaller are left alone)";
  $("prepRunTitle").textContent = crops ? "2 · Run it" : "3 · Run it";
  if (!S.imageFolder) { $("prepSummary").textContent = ""; return; }
  const r = await api(`/api/prep/summary?folder=${encodeURIComponent(S.imageFolder)}&mode=${encodeURIComponent(p.mode)}` +
    `&mp=${p.mp}&replace=${p.replace ? 1 : 0}&crops_only=${p.crops_only ? 1 : 0}` +
    `&train_mp=${encodeURIComponent(S.values.DATASET_MEGAPIXELS || "")}`).catch(() => null);
  if (!r) return;
  $("prepAudioOnly").style.display = r.audio_only ? "" : "none";
  $("prepMain").style.display = r.audio_only ? "none" : "";
  $("prepSummary").replaceChildren(...r.lines.map((t) => el("div", {text: t})));
  $("prepMpWarn").textContent = r.warn || "";
  if (!S.prep.running) $("prepRun").textContent = crops ? "✨ Make Detail Crops Now" :
    r.count ? `✨ Prepare ${r.count} Image${r.count === 1 ? "" : "s"} Now` : "✨ Prepare Images Now";
}

async function pollPrep() {
  clearTimeout(S.prep.timer);
  const r = await api("/api/prep/status?since=" + S.prep.logSize).catch(() => null);
  if (r) {
    if (r.size < S.prep.logSize) { $("prepLog").textContent = ""; S.prep.logSize = 0; }
    if (r.log) { $("prepLog").textContent += r.log; $("prepLog").scrollTop = $("prepLog").scrollHeight; }
    if (r.log && S.prep.masking) { $("maskLog").textContent += r.log; $("maskLog").scrollTop = $("maskLog").scrollHeight; }
    S.prep.logSize = r.size;
    const was = S.prep.running;
    S.prep.running = r.state === "running";
    $("prepRun").disabled = S.prep.running;
    if (S.prep.running) $("prepRun").textContent = "Preparing…";
    $("prepState").textContent = S.prep.running ? "Working on the Fizgig computer: you can leave this page." :
      r.state === "done" ? "Done." : r.state === "failed" ? "It stopped with an error: see the log." : "";
    if (S.prep.masking) {
      $("maskState").textContent = S.prep.running ? "Making masks on the Fizgig computer: you can leave this page." :
        r.state === "done" ? "Done." : r.state === "failed" ? "It stopped with an error: see below." : "";
      $("maskRun").disabled = S.prep.running;
    }
    if (was && !S.prep.running) { if (S.prep.masking && S.me) S.me.wantList = true; prepSummary(); loadDatasets(); maskInfo(); S.prep.masking = false; }
  }
  if (S.prep.running) S.prep.timer = setTimeout(pollPrep, 1000);
}

async function prepRun() {
  if (!S.imageFolder) { $("prepState").textContent = "Choose the training folder on the Start tab first."; return; }
  const p = prepSettings();
  try {
    await api("/api/prep/run", {folder: S.imageFolder, ...p});
    $("prepLog").textContent = ""; S.prep.logSize = 0;
    pollPrep();
  } catch (e) { $("prepState").textContent = e.message; }
}

// Image Prep's "Anime/furry/drawn characters" (fizgig/face_utils.py, prefs.json face_*): saved as soon as they
// change. InsightFace + ArcFace are the defaults; the section opens by itself when something else is chosen.
const FACE_HINTS = {
  insightface: "For people. It can tell men from women, for Largest Male / Female Face. It doesn't see anime, furry " +
    "or drawn faces.",
  yolo: "For anime, furry or drawn characters, and fursuits, pick this. It can't tell men from women, so Largest " +
    "Male / Female Face use the largest face.",
  auto: "For a set with both people and characters in it. Both finders look at every picture, so it's a little slower."};
const LOOK_HINTS = {
  arcface: "For people's faces. For anime, furry or drawn characters, pick Anime/furry/drawn.",
  dinov2: "For anime, furry or drawn characters, and fursuits; it works on people too. Its scores run higher than the people setting's, and " +
    "the colours follow.",
  auto: "Fizgig looks at some of the pictures each time it scores: People (the default) for a set of people, " +
    "Anime/furry/drawn for anything else."};
function faceSettingsShow(f, msg) {
  if (!f) return;
  S.prep.face = f;
  for (const [id, list, v] of [["prepDetector", f.detectors, f.detector], ["prepIdentity", f.identities, f.identity]]) {
    $(id).innerHTML = "";
    for (const [k, label] of list) $(id).append(el("option", {value: k, text: label}));
    $(id).value = v;
  }
  if (document.activeElement !== $("prepYolo")) $("prepYolo").value = f.yolo_model;
  $("prepYoloRow").style.display = f.detector === "insightface" ? "none" : "";
  $("prepDetectorHint").textContent = FACE_HINTS[f.detector] || "";
  $("prepIdentityHint").textContent = f.identity === "auto" && f.detector !== "auto" ?
    `With this face finder that means ${f.detector === "yolo" ? "Anime/furry/drawn" : "People (the default)"}.` :
    LOOK_HINTS[f.identity] || "";
  $("prepYoloHint").textContent = `Blank: Fdetailer if it's saved as ${f.local_model}, otherwise a free anime head ` +
    "finder that downloads the first time it's used. docs/FACE_BACKENDS.md explains how to get Fdetailer.";
  $("prepFaceMsg").textContent = msg || "";
  const custom = f.defaults && (f.detector !== f.defaults.detector || f.identity !== f.defaults.identity);
  if (custom) $("prepChars").open = true;              // never hide a choice that is changing what you get
}
async function faceSettingsSave(changes, msg) {
  try { faceSettingsShow(await api("/api/prep/face_settings", changes), msg); }
  catch (e) { $("prepFaceMsg").textContent = e.message; }
}
function faceSettingsWire() {
  $("prepDetector").addEventListener("change", () => {
    const det = $("prepDetector").value, f = S.prep.face || {};
    // a YOLO face finder with the people-only look match would score nothing: the look match follows, said aloud
    if (det !== "insightface" && f.identity === "arcface") {
      const ident = det === "yolo" ? "dinov2" : "auto";
      return faceSettingsSave({detector: det, identity: ident}, `Look match changed to ${ident === "dinov2" ?
        "Anime/furry/drawn" : "Auto (Decide for each set)"}, so characters' faces get scored too.`);
    }
    faceSettingsSave({detector: det});
  });
  $("prepIdentity").addEventListener("change", () => faceSettingsSave({identity: $("prepIdentity").value}));
  // the YOLO face model: the standard path row (type, Choose… on the Fizgig machine, or Upload… from this device)
  const input = $("prepYolo"), set = (p) => faceSettingsSave({yolo_model: p});
  input.addEventListener("change", () => set(input.value.trim()));
  const row = $("prepYoloPick");
  row.append(el("button", {class: "btn", text: "Choose…", title: "Pick from what is on the machine running Fizgig",
    onclick: () => pick(input.value || ((S.prep.face || {}).local_model || "").replace(/[\\/][^\\/]*$/, ""), false,
      (p) => { input.value = p; set(p); })}));
  const up = uploadButton({kind: "path", key: "FACE_YOLO_MODEL", accept: ".onnx"}, input, set);
  row.append(up.btn);
  $("prepYoloRow").append(up.status);
}

function prepTest() {
  pick(S.imageFolder, false, async (path) => {
    const host = $("prepTestResult");
    host.replaceChildren(el("div", {class: "field-hint", text: "Finding faces… (the first time loads the face model)"}));
    try {
      const r = await api("/api/prep/test", {path, face: $("prepFace").value});
      if (!r.faces.length) { host.replaceChildren(el("div", {class: "field-hint", text: "No faces found in that photo."})); return; }
      host.replaceChildren(
        el("div", {class: "field-hint", text: `${r.faces.length} face${r.faces.length === 1 ? "" : "s"} found (${r.finder}): ` +
          r.faces.map((f, i) => `face ${i + 1}, ${f.gender === "unknown" ? "?" : f.gender}, ${f.area.toLocaleString()} px` +
            (f.selected ? " (chosen for the crop)" : "")).join("; ") + ". Green = chosen, yellow = the others."}),
        el("img", {class: "prep-test-img", src: r.image, alt: "faces found"}));
    } catch (e) { host.replaceChildren(el("div", {class: "problems", text: e.message})); }
  });
}

// ---- the Look Filter ----
function lookForget() {
  // a new training folder starts with no baselines or marks (#218); its own last scan, if any, comes back with the poll
  Object.assign(S.prep, {bases: [], marked: new Set(), page: 0, picking: false});
  $("lookPicker").style.display = "none";
  lookBasesShow();
  lookPoll();
}

async function lookPickOpen() {
  const host = $("lookPicker");
  if (S.prep.picking) { S.prep.picking = false; host.style.display = "none"; return; }
  if (!S.imageFolder) { $("lookStatus").textContent = "Choose the training folder on the Start tab first."; return; }
  S.prep.picking = true;
  host.style.display = "";
  host.replaceChildren(el("div", {class: "field-hint", text: "Loading the folder's pictures…"}));
  const r = await api(`/api/captions?folder=${encodeURIComponent(S.imageFolder)}&ext=${encodeURIComponent(capExt())}&family=${encodeURIComponent(S.family || "")}`).catch(() => ({items: []}));
  const pics = r.items.filter((it) => /\.(png|jpe?g|webp|bmp)$/i.test(it.name));
  host.replaceChildren(el("div", {class: "field-hint look-pick-hint", text: "Click the 3 pictures that best show the face you're training."}),
    ...pics.map((it) => el("button", {class: "look-tile pick" + (S.prep.bases.includes(it.path) ? " chosen" : ""),
      title: it.name, onclick: (e) => {
        const i = S.prep.bases.indexOf(it.path);
        if (i >= 0) S.prep.bases.splice(i, 1);
        else if (S.prep.bases.length < 3) S.prep.bases.push(it.path);
        else return;
        e.currentTarget.classList.toggle("chosen", S.prep.bases.includes(it.path));
        lookBasesShow();
      }}, el("img", {loading: "lazy", src: thumbOf(it.path), alt: ""}), el("span", {text: it.name}))));
}

function lookBasesShow() {
  $("lookBases").replaceChildren(...S.prep.bases.map((p) => el("img", {src: thumbOf(p), title: p, alt: ""})),
    el("span", {class: "field-hint", text: `${S.prep.bases.length} of 3 chosen`}));
  $("lookScan").disabled = S.prep.bases.length !== 3 || !!S.prep.lookRunning;
}

async function lookScan() {
  $("lookScan").disabled = true;                  // until the scan's status says it's over (#249)
  try {
    await api("/api/look/scan", {folder: S.imageFolder, baselines: S.prep.bases});
    S.prep.picking = false; $("lookPicker").style.display = "none";
    S.prep.marked = new Set(); S.prep.page = 0;
    lookPoll();
  } catch (e) { $("lookStatus").textContent = e.message; lookBasesShow(); }
}

async function lookPoll() {
  clearTimeout(S.prep.lookTimer);
  if (S.prep.intro == null) S.prep.intro = $("lookStatus").textContent;    // the card's own words, for a new folder
  const r = await api("/api/look/status").catch(() => null);
  if (!r) return;
  S.prep.look = r;
  // a scan that just finished saved the 3 baselines: the masks card's Only my subject can be ticked now (#242)
  const wasRunning = S.prep.lookRunning;
  S.prep.lookRunning = r.state === "running";
  if (wasRunning && !S.prep.lookRunning) maskInfo();
  $("lookScan").disabled = S.prep.lookRunning || S.prep.bases.length !== 3;
  $("lookPick").disabled = S.prep.lookRunning;
  $("lookDecision").textContent = r.decision ? r.decision.replace("[faces] ", "") : "";
  $("lookPeak").textContent = r.state === "done" && r.rows && r.rows.length ? arcPeak(r.embedder) : "";
  const mine = r.folder === S.imageFolder;        // the last scan may be of another folder
  if (r.state === "running") {
    $("lookStatus").textContent = r.progress[0] ? `Scoring faces… ${r.progress[0]}/${r.progress[1]}` :
      "Loading the face model and choosing the look match (the first time downloads about 300 MB)…";
    S.prep.lookTimer = setTimeout(lookPoll, 1000);
  } else if (r.state === "failed") $("lookStatus").textContent = r.error;
  else if (r.state === "done" && r.rows.length && !mine) $("lookStatus").textContent = S.prep.intro;
  else if (r.state === "done" && r.rows.length) {
    if (!S.prep.bases.length) { S.prep.bases = r.baselines; lookBasesShow(); }
    const faces = r.rows.filter((x) => x.score != null).length;
    $("lookStatus").textContent = `${r.rows.length} picture(s) scored: ${faces} with faces, ${r.rows.length - faces} ` +
      "without. Worst matches first. Click a picture to mark it, or use Auto-Suggest. Scores are saved with the " +
      "dataset: low scorers you keep can ease in with “Warm up look outliers” on the Training tab.";
  }
  lookRender();
}

const LOOK_PAGE = 200;
// a Look Filter verdict's colour (the theme's)
const LOOK_VERDICT = {"no face": "--text-muted", match: "--success", borderline: "--warning", drift: "--danger"};
function lookRender() {
  const r = S.prep.look, host = $("lookRows");
  const rows = r && r.state === "done" && r.folder === S.imageFolder ? r.rows : [];
  $("lookTools").style.display = rows.length ? "" : "none";
  host.innerHTML = "";
  const pages = Math.max(1, Math.ceil(rows.length / LOOK_PAGE));
  S.prep.page = Math.min(S.prep.page, pages - 1);
  $("lookPages").style.display = pages > 1 ? "" : "none";
  $("lookPageInfo").textContent = `Page ${S.prep.page + 1} of ${pages} · ${rows.length} pictures · ${S.prep.marked.size} marked`;
  $("lookPrev").disabled = S.prep.page === 0; $("lookNext").disabled = S.prep.page >= pages - 1;
  $("lookCount").textContent = S.prep.marked.size ? `${S.prep.marked.size} marked` : "";
  $("lookMove").disabled = !S.prep.marked.size;
  for (const x of rows.slice(S.prep.page * LOOK_PAGE, (S.prep.page + 1) * LOOK_PAGE)) {
    const marked = S.prep.marked.has(x.path);
    const tile = el("button", {class: "look-tile" + (marked ? " marked" : ""), style: `--verdict:var(${LOOK_VERDICT[x.verdict] || "--text-muted"})`,
      title: x.blurb, onclick: () => { S.prep.marked.has(x.path) ? S.prep.marked.delete(x.path) : S.prep.marked.add(x.path); lookRender(); }},
      el("img", {loading: "lazy", src: thumbOf(x.path), alt: ""}),
      el("span", {class: "look-name", text: x.name}),
      el("span", {class: "look-verdict", text: (x.score != null ? `${Math.round(x.score * 100)}% · ` : "") + x.verdict.toUpperCase() +
        (x.baseline ? " · ★ baseline" : "")}),
      el("span", {class: "look-mark", text: marked ? "✕ marked to move out" : ""}));
    host.append(tile);
  }
}

function lookSuggest() {
  const r = S.prep.look;
  if (!r || r.cutoff == null) { $("lookStatus").textContent = "Not enough scored faces for statistics: mark by eye."; return; }
  const newly = r.rows.filter((x) => x.suggest && !S.prep.marked.has(x.path));
  newly.forEach((x) => S.prep.marked.add(x.path));
  $("lookStatus").textContent = `Auto-suggest marked ${newly.length} picture(s) (dataset median ${Math.round(r.median * 100)}%, ` +
    `cutoff ${Math.round(r.cutoff * 100)}%). Review before moving: it flags statistical drift, not certainty.`;
  lookRender();
}

function lookMoveAsk() {
  const n = S.prep.marked.size;
  if (!n) return;
  const sep = S.imageFolder.includes("\\") ? "\\" : "/";
  $("lookConfirm").replaceChildren(el("div", {class: "run-delete"},
    el("div", {class: "field-hint", text: `Move ${n} picture(s), with their captions, out of the dataset to ` +
      `${S.imageFolder}${sep}excluded_by_look? Nothing is deleted: move them back to put them in again.`}),
    el("div", {class: "field-row"},
      el("button", {class: "btn btn-sm btn-danger", text: "Move", onclick: async (ev) => {
        ev.currentTarget.disabled = true;           // the confirm goes when it's done; on an error it says why
        try {
          const r = await api("/api/look/move", {folder: S.imageFolder, paths: [...S.prep.marked], ext: capExt()});
          S.prep.marked = new Set();
          $("lookConfirm").innerHTML = "";
          $("lookStatus").textContent = `Moved ${r.moved} picture(s) to ${r.dest}. ${r.left} remain in the dataset.` +
            (r.errors.length ? " " + r.errors.join(" ") : "");
          await lookPoll(); loadDatasets();
        } catch (e) { $("lookConfirm").replaceChildren(el("div", {class: "problems", text: e.message})); }
      }}),
      el("button", {class: "btn btn-sm", text: "Cancel", onclick: () => { $("lookConfirm").innerHTML = ""; }}))));
}

// Save preset: a name, then saved for this model in the desktop's format (asks before replacing one)
function presetSaveOpen() {
  const host = $("presetSaveRow");
  if (host.childNodes.length) { host.innerHTML = ""; return; }
  const name = el("input", {type: "text", class: "w-path", placeholder: "A name for this preset",
    value: S.values.__preset && !S.fam.form.presets.some((p) => p.name === S.values.__preset) ? S.values.__preset : ""});
  const msg = el("span", {class: "field-hint"});
  const save = async (overwrite) => {
    const n = name.value.trim();
    if (!n) { msg.textContent = "Type a name first."; return; }
    try {
      const r = await api("/api/preset/save", {family: S.family, name: n, values: S.values, overwrite: !!overwrite});
      if (r.exists) {
        msg.replaceChildren(document.createTextNode(`There's a preset called ${n} already. `),
          el("button", {class: "btn btn-sm btn-danger", text: "Replace it", onclick: () => save(true)}));
        return;
      }
      S.values.__preset = n;
      saveState();
      host.innerHTML = "";
      await renderPresets();
      $("runStatus").textContent = `Saved the preset ${n}: it's in both apps' preset list for this model.`;
    } catch (e) { msg.textContent = e.message; }
  };
  name.addEventListener("keydown", (e) => { if (e.key === "Enter") save(false); });
  host.append(el("div", {class: "field-row"}, name, el("button", {class: "btn btn-primary", text: "Save", onclick: () => save(false)}),
    el("button", {class: "btn", text: "Cancel", onclick: () => { host.innerHTML = ""; }}), msg));
  name.focus();
}

function wirePrep() {
  document.querySelectorAll('input[name="prepMode"], input[name="prepReplace"]').forEach((r) =>
    r.addEventListener("change", () => { prepSummary(); savePrepSoon(); }));
  for (const id of ["prepMp", "prepFace", "prepPadding", "prepCropsOnly"]) $(id).addEventListener("change", () => { prepSummary(); savePrepSoon(); });
  $("prepRun").addEventListener("click", prepRun);
  $("maskRun").addEventListener("click", maskRun);
  meWire();
  $("prepTest").addEventListener("click", prepTest);
  faceSettingsWire();
  $("lookPick").addEventListener("click", lookPickOpen);
  $("lookScan").addEventListener("click", lookScan);
  $("lookSuggest").addEventListener("click", lookSuggest);
  $("lookMove").addEventListener("click", lookMoveAsk);
  $("lookPrev").addEventListener("click", () => { S.prep.page--; lookRender(); });
  $("lookNext").addEventListener("click", () => { S.prep.page++; lookRender(); });
}

// ---- Repair Studio (fizgig/web/studio.py runs the desktop's engine in a worker that keeps the model loaded) -----
// The page holds the sliders; every change renders after a short pause, a newer change replacing the render going
S.st = {layout: null, blocks: {}, locks: new Set(), primaryBlocks: new Set(), donorBlocks: new Set(), donor: false,
  shown: 0, hist: [], redo: [], loadedKey: "", timer: null, poll: null, ready: false};
const ST_KEY = "fizgig_studio";
const stSaved = () => { try { return JSON.parse(localStorage.getItem(ST_KEY) || "{}"); } catch (e) { return {}; } };
const stRemember = debounce(() => {
  const d = {family: $("stFamily").value, primary: $("stPrimary").value, donor: $("stDonor").value,
    prompt: $("stPrompt").value, seed: $("stSeed").value, res: $("stRes").value,
    w: $("stW").value, h: $("stH").value, turbo: $("stTurbo").checked,
    ref: $("stRef").value, refMp: $("stRefMp").value, refStr: $("stRefStr").value, luckyLo: $("stLuckyLo").value,
    luckyHi: $("stLuckyHi").value, luckyWho: $("stLuckyWho").value, luckyCount: $("stLuckyCount").value, neg: {...(stSaved().neg || {}), [$("stFamily").value]: $("stNeg").value},
    scales: {...(stSaved().scales || {}), [$("stFamily").value]: [$("stPScale").value, $("stDScale").value]},
    h3: stSaved().h3, ace: stSaved().ace, metricsRef: stSaved().metricsRef};                           // H3's clip settings and photos (studioH3Remember's)
  try { localStorage.setItem(ST_KEY, JSON.stringify(d)); } catch (e) { /* private window */ }
}, 500);

async function studioOpen() {
  if (!S.st.familiesLoaded) {
    let r;
    try { r = await api("/api/studio/families"); }
    catch (e) {                                 // a server started before Studio existed: the page is newer than it
      $("stStatus").textContent = "This web UI was started before Studio was added: restart it (close and reopen " +
        "the Fizgig window, or Fizgig in browser), then reload this page.";
      return;
    }
    const sel = $("stFamily");
    sel.innerHTML = "";
    for (const f of r.families) sel.append(el("option", {value: f.key, text: f.name}));
    const sv = stSaved();
    if (sv.family && r.families.some((f) => f.key === sv.family)) sel.value = sv.family;
    for (const [id, k] of [["stPrimary", "primary"], ["stDonor", "donor"], ["stPrompt", "prompt"], ["stSeed", "seed"],
      ["stRef", "ref"], ["stRefMp", "refMp"], ["stRefStr", "refStr"], ["stLuckyLo", "luckyLo"], ["stLuckyHi", "luckyHi"], ["stLuckyWho", "luckyWho"]])
      if (sv[k] != null && sv[k] !== "") $(id).value = sv[k];
    if (sv.turbo != null) $("stTurbo").checked = sv.turbo;
    S.st.familiesLoaded = true;
    await studioFamily(sv.res, sv.w, sv.h);
  }
  if (S.st.openWith) await studioTake();
  studioPoll();
}

// a LoRA row's Studio button: the LoRA as the primary, on the model its file is for (from its key names; the run's
// model when the file can't tell), on the Studio tab. Nothing loads until Start.
function studioButton(path, family) {
  if (training()) return null;                    // Studio's model would share the card with the run
  return el("button", {class: "btn btn-sm", text: "Studio", title: "Open this LoRA in Studio, on the model it was trained for",
    onclick: () => { S.st.openWith = {path, family}; showTab("studio"); }});
}

function training() { return !!(S.run && (S.run.running || ["starting", "running", "pausing"].includes(S.run.state))); }
// a preview with the LoRA its epoch saved (the server's loras map), for its Studio button
function withLora(it, loras, family) {
  return it.epoch != null && loras && loras[String(it.epoch)] ? {...it, lora: loras[String(it.epoch)], family} : it;
}

async function studioTake() {
  const {path, family} = S.st.openWith;
  S.st.openWith = null;
  let key = "";
  try { key = (await api("/api/studio/lorafamily?path=" + encodeURIComponent(path))).key; } catch (e) { /* the run's */ }
  const has = (k) => k && [...$("stFamily").options].some((o) => o.value === k);
  const fam = has(key) ? key : has(family) ? family : "";
  if (!fam) { $("stStatus").textContent = "Studio doesn't work on this LoRA's model."; return; }
  if (fam !== $("stFamily").value) { $("stFamily").value = fam; await studioFamily(); }
  $("stPrimary").value = path;
  stRemember();
  $("stStatus").textContent = `${path.split(/[\\/]/).pop()} is the primary LoRA.`;
}

async function studioFamily(res, w, h) {
  const key = $("stFamily").value;
  const L = S.st.layout = await api("/api/studio/layout?family=" + encodeURIComponent(key));
  const sv = stSaved();
  studioKind(L, sv);
  $("stRes").innerHTML = "";
  for (const s of L.sizes) $("stRes").append(el("option", {value: s, text: s}));
  $("stRes").value = L.sizes.includes(String(res)) ? String(res) : L.size;
  for (const [id, v] of [["stW", w], ["stH", h]]) {             // pictures: any width and height from the list
    const keep = String(v ?? $(id).value);
    $(id).innerHTML = "";
    for (const s of L.sizes) $(id).append(el("option", {value: s, text: s}));
    $(id).value = L.sizes.includes(keep) ? keep : L.size;
  }
  const sc = (sv.scales || {})[key];
  $("stPScale").value = sc ? sc[0] : L.default_strength;
  $("stDScale").value = sc ? sc[1] : 1;
  $("stNeg").value = (sv.neg || {})[key] ?? (L.negative_default || "");
  $("stTurboRow").style.display = L.turbo_preview ? "" : "none";
  $("stRefRow").style.display = L.reference_kind ? "" : "none";
  $("stRefMp").disabled = L.reference_kind === "edit";
  $("stRefStr").style.display = $("stRefStrLabel").style.display = L.reference_strength ? "" : "none";
  // which model previews: the follows-the-Samples-tab families say so; others pick fast or full
  $("stDitRow").style.display = L.follows_samples || !L.dit_fast ? "none" : "";
  $("stDitFast").textContent = L.dit_fast; $("stDitFull").textContent = L.dit_full;
  $("stDitNote").textContent = L.follows_samples && L.follow_ck ? L.follow_ck : L.follows_samples ? `Previews ${L.follow_turbo ? "on the " + L.follow_turbo + ", " : ""}` +
    `${L.follow_steps} steps, CFG ${L.follow_cfg}${L.follow_look ? ", " + L.follow_look : ""} - from this model's Samples tab.` :
    !L.dit_fast ? L.dit_full.replace(", slow", "") : "";
  renderChoiceDraw("stRender", L);
  studioNegRow();
  S.st.blocks = {};
  for (const g of L.groups) for (const b of g.blocks) S.st.blocks[b.id] = {pe: true, ps: 1, de: true, ds: 1};
  S.st.locks = new Set(); S.st.dlocks = new Set(); S.st.frozen = new Set(); S.st.dfrozen = new Set();
  S.st.hist = []; S.st.redo = [];
  studioLuckyCounts();
  studioMasters();
  studioPresets();
  studioRows();
}

// H3 (clips) and ACE-Step (songs): their own controls, the desktop's
function studioKind(L, sv) {
  const video = L.kind === "video", audio = L.kind === "audio";
  if (S.st.kindShown && S.st.kindShown !== L.kind) {     // pictures / clips / songs: the last kind's previews go
    for (const id of ["stBase", "stTweak", "stNol"]) $(id).removeAttribute("src");
    for (const id of ["stBaseVid", "stTweakVid", "stNolVid", "stBaseAud", "stTweakAud", "stNolAud"]) {
      const m = $(id);
      m.pause(); m.removeAttribute("src"); m.removeAttribute("poster"); m.load(); m.style.display = "none";
    }
    $("stDownloads").style.display = "none";
    if (S.song) { S.song.src = {}; S.song.peaks = {}; S.song.act = {}; }     // a song comes back fresh, not "already loaded"
  }
  S.st.kindShown = L.kind;
  $("stVideoRows").style.display = video ? "" : "none";
  $("stFrames").style.display = video ? "" : "none";
  $("stRes").style.display = $("stResLabel").style.display = audio ? "" : "none";      // a song's length
  if (audio) studioDock(false, false);                  // Keep in view: never for a music model (kept as remembered)
  else if (stDockSaved().open && !$("stDock")) studioDock(true);
  else studioDock(!!$("stDock"));
  $("stSizeRow").style.display = video || audio ? "none" : "";                          // a picture's width and height
  document.querySelectorAll(".st-pane").forEach((p) => p.classList.toggle("audio", audio));
  $("stPreviewHint").textContent = video ? "Each clip plays with its sound; the picture is its middle frame." :
    audio ? "Play any song; the picture is its waveform." : "Click either picture to see both full size (← → switches between them).";
  $("stPlayBoth").style.display = video ? "" : "none";
  // Hold structure: only for a model whose block map marks the bands (lucky_zones), and on there
  $("stLuckyHoldBox").style.display = (L.lucky_zones || []).length ? "" : "none";
  $("stLuckyHold").checked = true;
  $("stSong").style.display = audio ? "" : "none";
  if (!audio) songStop();
  studioAce(L, sv);
  studioEqButtons(L);
  studioMixShow(L);
  studioMetricsShow();
  if (!video) return;
  const V = L.video, h3 = sv.h3 || {};
  const fill = (id, opts, val) => { const s = $(id); s.innerHTML = ""; for (const o of opts) s.append(el("option", {value: o, text: o})); s.value = opts.includes(val) ? val : opts[0]; };
  fill("stH3Model", V.models, h3.model || V.models[0]);
  fill("stH3Base", V.bases, h3.base);
  fill("stH3Len", V.lengths, h3.len || V.length);
  fill("stH3W", V.dims, h3.w || V.size[0]);
  fill("stH3H", V.dims, h3.h || V.size[1]);
  if (h3.steps) $("stH3Steps").value = h3.steps;
  if (h3.turbo != null) $("stH3Turbo").value = h3.turbo;
  $("stH3Sound").checked = V.sound && h3.sound !== false;
  $("stH3Sound").disabled = !V.sound;
  $("stH3Boost").checked = h3.boost !== false;           // on unless turned off
  $("stH3Boost").parentElement.style.display = V.speed_boost ? "" : "none";   // a family ability (H3 has it)
  $("stH3Early").checked = h3.early !== false;
  S.st.kf = h3.kf || {};
  S.st.kfRect = h3.kfRect || {};
  studioSlots();
}

function studioSlots() {
  const ref = !!S.st.layout.video.references[$("stH3Model").value];
  $("stFramesTitle").textContent = ref ? "Reference Images" : "First / Last Frame";
  $("stFramesDesc").textContent = ref ? "Optional: up to two photos the clip should look like (this checkpoint's " +
    "references), each cropped to the clip's shape - you place the crop when you pick one. Say <Picture 1> and " +
    "<Picture 2> in the prompt for them, e.g. \"the woman in <Picture 1> walks along the beach in <Picture 2>\"." :
    "Optional: photos the clip starts and ends on (a still uses the first only), each cropped to the clip's shape - " +
    "you place the crop when you pick one.";
  const host = $("stFrameSlots");
  host.innerHTML = "";
  const slots = ref ? [["ref1", "Reference 1"], ["ref2", "Reference 2"]] : [["first", "First frame"], ["last", "Last frame"]];
  for (const [slot, label] of slots) {
    const input = el("input", {type: "text", class: "w-path", value: (S.st.kf[slot] || ""), placeholder: "a photo (optional)"});
    const thumb = el("canvas", {width: 1, height: 1});
    const info = el("span", {class: "field-hint"});
    const show = () => {
      const p = S.st.kf[slot], r = S.st.kfRect[slot];
      input.value = p || "";
      info.textContent = p && r ? `crop ${r[2] - r[0]}×${r[3] - r[1]}` : "";
      studioSlotThumb(thumb, p, r);
    };
    // a new photo: place its crop (Cancel keeps what was there); Re-crop moves the box of the one there
    const set = (p) => {
      if (!p) { delete S.st.kf[slot]; delete S.st.kfRect[slot]; show(); studioH3Remember(); studioSchedule(); return; }
      studioCrop(p, p === S.st.kf[slot] ? S.st.kfRect[slot] : null, (rect) => {
        S.st.kf[slot] = p; S.st.kfRect[slot] = rect; show(); studioH3Remember(); studioSchedule();
      }, show);
    };
    input.addEventListener("change", () => set(input.value.trim()));
    const up = uploadButton({kind: "path", key: "STUDIO_" + slot}, input, set);
    host.append(el("div", {class: "st-slot"}, el("span", {class: "field-label", text: label + ":"}), thumb,
      el("div", {}, el("div", {class: "field-row"}, input,
        el("button", {class: "btn", text: "Choose…", onclick: () => pick(input.value || S.imageFolder, false, set)}), up.btn,
        el("button", {class: "btn", text: "Re-crop", onclick: () => S.st.kf[slot] && set(S.st.kf[slot])}),
        el("button", {class: "btn", text: "Clear", onclick: () => set("")}), info), up.status)));
    show();
  }
}

const mediaUrl = (p) => `/api/media?path=${encodeURIComponent(p)}&t=${encodeURIComponent(TOKEN)}`;

function studioSlotThumb(cv, path, rect) {
  // the slot's thumbnail: the crop itself, as the desktop shows it
  const ctx = cv.getContext("2d");
  if (!path) { cv.width = cv.height = 1; return; }
  const img = new Image();
  img.onload = () => {
    const r = rect || [0, 0, img.naturalWidth, img.naturalHeight];
    const w = r[2] - r[0], h = r[3] - r[1], k = Math.min(160 / w, 96 / h);
    cv.width = Math.max(1, Math.round(w * k)); cv.height = Math.max(1, Math.round(h * k));
    ctx.drawImage(img, r[0], r[1], w, h, 0, 0, cv.width, cv.height);
  };
  img.src = mediaUrl(path);
}

function studioCrop(path, initial, ok, cancel) {
  // the desktop's crop window: the photo fitted, a box locked to the clip's shape (W x H) - drag to move, drag a
  // corner to resize. ok([x0, y0, x1, y1] in the photo's pixels); cancel() leaves things as they were.
  const W = Number($("stH3W").value) || 640, H = Number($("stH3H").value) || 768, aspect = W / H;
  const cv = $("cropCanvas"), ctx = cv.getContext("2d"), img = new Image();
  $("cropTitle").textContent = `Crop to the clip's shape (${W}×${H})`;
  img.onerror = () => { $("stStatus").textContent = "Couldn't open that photo."; if (cancel) cancel(); };
  img.onload = () => {
    const sw = img.naturalWidth, sh = img.naturalHeight;
    const scale = Math.min((window.innerWidth * 0.9 - 40) / sw, (window.innerHeight * 0.85 - 200) / sh, 1);
    const dw = Math.max(1, Math.round(sw * scale)), dh = Math.max(1, Math.round(sh * scale));
    cv.width = dw; cv.height = dh;
    const HANDLE = 9, MIN_W = 32 * scale * 4;
    let box;
    const centre = () => { const bw = dw / dh >= aspect ? dh * aspect : dw, bh = bw / aspect;
      box = {x0: (dw - bw) / 2, y0: (dh - bh) / 2, x1: (dw + bw) / 2, y1: (dh + bh) / 2}; };
    if (initial) box = {x0: initial[0] * scale, y0: initial[1] * scale, x1: initial[2] * scale, y1: initial[3] * scale};
    else centre();
    const src = () => [box.x0, box.y0, box.x1, box.y1].map((v, i) => Math.min(i % 2 ? sh : sw, Math.max(0, Math.round(v / scale))));
    const draw = () => {
      ctx.clearRect(0, 0, dw, dh); ctx.drawImage(img, 0, 0, dw, dh);
      ctx.fillStyle = themeColor("--shadow", 0.55);
      ctx.fillRect(0, 0, dw, box.y0); ctx.fillRect(0, box.y1, dw, dh - box.y1);
      ctx.fillRect(0, box.y0, box.x0, box.y1 - box.y0); ctx.fillRect(box.x1, box.y0, dw - box.x1, box.y1 - box.y0);
      ctx.strokeStyle = themeColor("--pic-highlight"); ctx.lineWidth = 2; ctx.strokeRect(box.x0, box.y0, box.x1 - box.x0, box.y1 - box.y0);
      ctx.fillStyle = themeColor("--pic-highlight");
      for (const [hx, hy] of [[box.x0, box.y0], [box.x1, box.y0], [box.x0, box.y1], [box.x1, box.y1]])
        ctx.fillRect(hx - HANDLE, hy - HANDLE, HANDLE * 2, HANDLE * 2);
      const r = src();
      $("cropSize").textContent = `crop ${r[2] - r[0]}×${r[3] - r[1]} of ${sw}×${sh} → ${W}×${H}`;
    };
    const at = (e) => { const b = cv.getBoundingClientRect(); return [(e.clientX - b.left) * dw / b.width, (e.clientY - b.top) * dh / b.height]; };
    let drag = null;
    cv.onpointerdown = (e) => {
      const [x, y] = at(e);
      const corner = [["nw", box.x0, box.y0], ["ne", box.x1, box.y0], ["sw", box.x0, box.y1], ["se", box.x1, box.y1]]
        .find(([, hx, hy]) => Math.abs(x - hx) <= HANDLE * 1.5 && Math.abs(y - hy) <= HANDLE * 1.5);
      if (corner) drag = {mode: "resize", c: corner[0]};
      else if (x >= box.x0 && x <= box.x1 && y >= box.y0 && y <= box.y1) drag = {mode: "move", dx: x - box.x0, dy: y - box.y0};
      else return;
      cv.setPointerCapture(e.pointerId);
    };
    cv.onpointermove = (e) => {
      if (!drag) return;
      const [x, y] = at(e);
      if (drag.mode === "move") {
        const bw = box.x1 - box.x0, bh = box.y1 - box.y0;
        const nx = Math.min(Math.max(0, x - drag.dx), dw - bw), ny = Math.min(Math.max(0, y - drag.dy), dh - bh);
        box = {x0: nx, y0: ny, x1: nx + bw, y1: ny + bh};
      } else {
        // the opposite corner anchors; the width follows the pointer, the height the shape; clamped to the photo
        const c = drag.c, east = c === "ne" || c === "se", south = c === "sw" || c === "se";
        const ax = east ? box.x0 : box.x1, ay = south ? box.y0 : box.y1;
        let w = Math.abs(Math.min(Math.max(x, 0), dw) - ax), h = Math.abs(Math.min(Math.max(y, 0), dh) - ay);
        w = Math.max(MIN_W, w, h * aspect); h = w / aspect;
        const maxW = east ? dw - ax : ax, maxH = south ? dh - ay : ay;
        if (w > maxW) { w = maxW; h = w / aspect; }
        if (h > maxH) { h = maxH; w = h * aspect; }
        box = {x0: east ? ax : ax - w, x1: east ? ax + w : ax, y0: south ? ay : ay - h, y1: south ? ay + h : ay};
      }
      draw();
    };
    cv.onpointerup = () => { drag = null; };
    const close = () => { $("cropModal").classList.remove("active"); cv.onpointerdown = cv.onpointermove = cv.onpointerup = null; };
    $("cropOk").onclick = () => { close(); ok(src()); };
    $("cropCancel").onclick = $("cropX").onclick = () => { close(); if (cancel) cancel(); };
    $("cropCentre").onclick = () => { centre(); draw(); };
    draw();
    $("cropModal").classList.add("active");
  };
  img.src = mediaUrl(path);
}

function studioH3Remember() {
  try {
    const d = stSaved();
    d.h3 = {model: $("stH3Model").value, base: $("stH3Base").value, len: $("stH3Len").value, w: $("stH3W").value,
      h: $("stH3H").value, steps: $("stH3Steps").value, turbo: $("stH3Turbo").value,
      sound: $("stH3Sound").checked, boost: $("stH3Boost").checked, early: $("stH3Early").checked, kf: S.st.kf,
      kfRect: S.st.kfRect};
    localStorage.setItem(ST_KEY, JSON.stringify(d));
  } catch (e) { /* private window */ }
}

function studioExtra() {
  // what a clip or a song needs beyond the sliders
  const L = S.st.layout;
  if (L.kind === "audio") return {audio: studioAceExtra()};
  if (L.kind !== "video") return {};
  const V = L.video;
  const w = Number($("stH3W").value) || 640, h = Number($("stH3H").value) || 768;     // the clip at its Width x Height
  const ref = !!V.references[$("stH3Model").value];
  const slots = ref ? ["ref1", "ref2"] : ["first", "last"];
  return {video: {frames: V.length_frames[$("stH3Len").value] || 22, width: w, height: h,
    steps: Math.max(1, Math.min(40, parseInt($("stH3Steps").value, 10) || 4)),
    turbo: Math.max(0, Math.min(1.5, Number($("stH3Turbo").value) || 0)), sound: $("stH3Sound").checked,
    speed_boost: !!V.speed_boost && $("stH3Boost").checked, early: $("stH3Early").checked},
    keyframes: slots.filter((s) => S.st.kf[s]).map((s) => ({slot: s, path: S.st.kf[s], rect: S.st.kfRect[s] || null})),
    ref_mode: ref};
}

// ---- ACE-Step: Prompt B, audio to audio, the reference recording, the section on the waveform (the desktop's) ----
const snap05 = (v) => Math.round(Number(v) / 0.05) * 0.05;

function studioAce(L, sv) {
  const on = L.kind === "audio";
  $("stAudioRows").style.display = on && (L.prompt_blend || L.audio_source) ? "" : "none";
  $("stSongTools").style.display = L.audio_source ? "" : "none";
  if (!on) { S.st.region = null; studioRegionDraw(); return; }
  const a = sv.ace || {};
  $("stPbOn").parentElement.style.display = L.prompt_blend ? "" : "none";
  $("stSrcOn").parentElement.style.display = $("stSndOn").parentElement.style.display = L.audio_source ? "" : "none";
  $("stPbOn").checked = !!a.pbOn && L.prompt_blend; $("stPromptB").value = a.pb || "";
  $("stPbBlend").value = a.blend ?? 0.5; $("stPbKeep").checked = a.keep !== false;
  $("stSrcOn").checked = !!a.srcOn && L.audio_source; $("stSrcPath").value = a.src || "";
  $("stSrcMode").value = a.mode || "song"; $("stSrcFollow").value = a.follow ?? 1; $("stSrcDenoise").value = a.denoise ?? 1;
  $("stSrcStrength").value = a.strength ?? 0.5;
  $("stSndOn").checked = !!a.sndOn && L.audio_source; $("stSndPath").value = a.snd || "";
  S.st.region = null;
  studioAceLayout();
}

function studioAceLayout() {
  $("stPbRows").style.display = $("stPbOn").checked ? "" : "none";
  $("stSrcRows").style.display = $("stSrcOn").checked ? "" : "none";
  $("stSndRows").style.display = $("stSndOn").checked ? "" : "none";
  const whole = $("stSrcMode").value === "song";
  $("stSrcWhole").style.display = whole ? "" : "none";
  $("stSrcRep").style.display = whole ? "none" : "";
  const w = snap05($("stPbBlend").value);
  $("stPbReadout").textContent = `${Math.round((1 - w) * 100)}% A · ${Math.round(w * 100)}% B`;
}

function studioAceRemember() {
  try {
    const d = stSaved();
    d.ace = {pbOn: $("stPbOn").checked, pb: $("stPromptB").value, blend: $("stPbBlend").value, keep: $("stPbKeep").checked,
      srcOn: $("stSrcOn").checked, src: $("stSrcPath").value.trim(), mode: $("stSrcMode").value,
      follow: $("stSrcFollow").value, denoise: $("stSrcDenoise").value, strength: $("stSrcStrength").value,
      sndOn: $("stSndOn").checked, snd: $("stSndPath").value.trim()};
    localStorage.setItem(ST_KEY, JSON.stringify(d));
  } catch (e) { /* private window */ }
}

function studioPromptB() {
  // the Prompt B the engine renders: B as typed, or (Keep Prompt A's lyrics) B's style with A's lyrics
  if (!$("stPbOn").checked) return null;
  const b = $("stPromptB").value.trim();
  if (!b) return null;
  if (!$("stPbKeep").checked) return b;
  const a = $("stPrompt").value, lyrics = a.includes("|") ? a.split("|").slice(1).join("|").trim() : "";
  const style = b.split("|")[0].trim();
  return lyrics ? `${style} | ${lyrics}` : style;
}

function studioAceExtra() {
  const pb = studioPromptB();
  return {seconds: Number($("stRes").value) || 30, prompt_b: pb, blend: pb ? snap05($("stPbBlend").value) : 0,
    source: $("stSrcOn").checked && $("stSrcPath").value.trim() ? {path: $("stSrcPath").value.trim(), mode: $("stSrcMode").value,
      follow: snap05($("stSrcFollow").value), denoise: snap05($("stSrcDenoise").value),
      strength: snap05($("stSrcStrength").value), region: S.st.region} : null,
    reference: $("stSndOn").checked && $("stSndPath").value.trim() ? $("stSndPath").value.trim() : null};
}

function studioRegionDraw() {
  const r = S.st.region;
  $("stRegionClear").style.display = r ? "" : "none";
  $("stRegionNote").textContent = r ? `Section ${r[0].toFixed(1)}–${r[1].toFixed(1)} s` +
    ($("stSongLoop").checked ? " - Loop repeats it" : " - tick Loop to repeat it") +
    ($("stSrcOn").checked && $("stSrcMode").value === "repaint" ? "; Repaint remakes it." : ".") + " Drag its tabs to refine it." :
    "Click the waveform to jump, drag the playhead to scrub, drag across it to choose a section to loop" +
    ($("stSrcOn").checked && $("stSrcMode").value === "repaint" ? " (and what Repaint remakes)" : "") + ". Right-click: the whole song.";
  songDraw();
}

// ---- the song player: one full-width waveform of the side being heard, its playhead, the section (the desktop's) ----
S.song = {side: null, pos: 0, src: {}, gen: {}, peaks: {}, act: {}, glow: {}, drag: null};
const SONG_SIDES = {baseline: "stBaseAud", tweaked: "stTweakAud", nolora: "stNolAud"};
const songPlayer = (side) => $(SONG_SIDES[side]);
const songHas = (side) => !!S.song.src[side];
function songStripSide() {
  return songHas(S.song.side) ? S.song.side : ["tweaked", "baseline", "nolora"].find(songHas) || null;
}
function songDur(side) {
  const pk = S.song.peaks[side || songStripSide()];
  return pk ? pk.dur : 0;
}
function songPlaying() { const p = S.song.side && songPlayer(S.song.side); return !!(p && !p.paused); }
// where the song is: the player's own place, or - while a new render loads into it - the moment it left
function songNow() { const p = S.song.side && songPlayer(S.song.side); return p && p.getAttribute("src") && p.readyState >= 1 ? p.currentTime : S.song.pos; }
function songSpan() {
  // where Loop repeats: the section when one is set, else the whole song
  const d = songDur(), r = S.st.region;
  return r && $("stSongLoop").checked && r[0] < d ? [r[0], Math.min(r[1], d)] : [0, d];
}
function songLoad(side, url, gen, actUrl) {
  // a new render of one side: its waveform, and - when it's the side being heard - the new version carries on at the
  // same moment (Restart on tweak: a new Tweaked plays from the top)
  if (S.song.src[side] === url) return;
  S.song.src[side] = url;
  delete S.song.act[side];                          // the block activity of the new render, when the model gives one
  if (actUrl) fetch(actUrl, {headers: {"X-Fizgig-Token": TOKEN}}).then((r) => r.json())
    .then((a) => { if (S.song.src[side] === url) { S.song.act[side] = a; songDraw(); } }).catch(() => {});
  const p = songPlayer(side), heard = S.song.side === side, was = heard && songPlaying(), at = songNow();
  if (heard) S.song.pos = at;                       // the playhead holds its place while the new version loads
  fetch(url.replace("/api/studio/image?", "/api/studio/peaks?"), {headers: {"X-Fizgig-Token": TOKEN}})
    .then((r) => r.json()).then((pk) => { if (S.song.src[side] === url) { S.song.peaks[side] = pk; songDraw(); } }).catch(() => {});
  p.src = url;
  if (heard) {
    const restart = side === "tweaked" && $("stSongRestart").checked;
    p.addEventListener("loadedmetadata", () => {
      p.currentTime = restart ? songSpan()[0] : Math.min(at, Math.max(0, p.duration - 0.05));
      // a new src paused the player, which ended the playhead's loop: start it again with the song
      if (was || restart) p.play().then(() => { songMark(); songTick(); }).catch(() => {});
      else songDraw();
    }, {once: true});
  }
  songMark();
}
function songPlay(side, at) {
  if (!songHas(side)) { $("stStatus").textContent = side === "nolora" ? "The No LoRA song renders after the first preview." :
    "That song is not rendered yet."; return; }
  let pos = at ?? songNow();
  const [a, b] = songSpan(), d = songDur(side) || Infinity;
  if ($("stSongLoop").checked && S.st.region && !(pos >= a && pos < b - 0.05)) pos = a;   // outside the section: its start
  if (pos >= d - 0.05 || pos < 0) pos = 0;
  for (const s of Object.keys(SONG_SIDES)) if (s !== side) songPlayer(s).pause();
  const p = songPlayer(side);
  S.song.side = side;
  p.loop = $("stSongLoop").checked && !S.st.region;      // the whole song loops by itself; a section loops in songTick
  const go = () => { p.currentTime = pos; p.play().catch(() => {}); };
  if (p.readyState >= 1) go(); else p.addEventListener("loadedmetadata", go, {once: true});
  songMark(); songTick();
}
function songStop() {
  for (const s of Object.keys(SONG_SIDES)) { const p = songPlayer(s); if (p && !p.paused) { S.song.pos = p.currentTime; p.pause(); } }
  songMark(); songDraw();
}
function songTick() {
  if (!songPlaying()) { songMark(); songDraw(); return; }
  const p = songPlayer(S.song.side), [a, b] = songSpan();
  if ($("stSongLoop").checked && S.st.region && p.currentTime >= b) p.currentTime = a;     // the section, again
  songDraw();
  requestAnimationFrame(songTick);
}
function songMark() {
  const playing = songPlaying();
  for (const s of Object.keys(SONG_SIDES)) {
    const b = $("stSong" + s[0].toUpperCase() + s.slice(1));
    b.classList.toggle("on", playing && S.song.side === s);
    b.disabled = !songHas(s);
  }
  const any = Object.keys(SONG_SIDES).some(songHas);
  $("stSongPause").textContent = playing || !any ? "⏸ Pause" : "▶ Resume";
  $("stSongPause").disabled = !any;
  const t = S.song.src.tweaked;
  $("stSongSave").style.pointerEvents = t ? "" : "none";
  $("stSongSave").style.opacity = t ? "" : "0.5";
  if (t) {
    const lora = ($("stPrimary").value.trim().split(/[\\/]/).pop() || "song").replace(/\.safetensors$/i, "");
    const seed = ($("stSeed") && $("stSeed").value.trim()) || "";
    const name = `${lora}_tweaked${seed ? "_seed" + seed : ""}.wav`;
    $("stSongSave").href = t + "&dl=" + encodeURIComponent(name);
    $("stSongSave").setAttribute("download", name);
  }
}
function songRegion(r) {
  // the section Loop repeats (null = the whole song): setting one ticks Loop; a playing song jumps into it
  S.st.region = r;
  if (r) $("stSongLoop").checked = true;
  studioAceRemember();
  if ($("stSrcOn").checked && $("stSrcMode").value === "repaint") studioSchedule();   // Repaint remakes the new section
  if (songPlaying()) songPlay(S.song.side, r ? r[0] : undefined);
  else if (r) S.song.pos = r[0];
  for (const s of Object.keys(SONG_SIDES)) songPlayer(s).loop = $("stSongLoop").checked && !r;
  studioRegionDraw();
}
function songDraw() {
  const cv = $("stSongBar");
  if (!cv || $("stSong").style.display === "none") return;
  const w = cv.clientWidth || 600, h = cv.height;
  if (cv.width !== w) cv.width = w;
  const g = cv.getContext("2d");
  const accent = themeColor("--accent"), blue = themeColor("--accent-hover"), muted = themeColor("--text-muted");
  g.clearRect(0, 0, w, h);
  const side = songStripSide(), pk = side && S.song.peaks[side], d = songDur();
  const fmt = (t) => `${Math.floor(t / 60)}:${String(Math.floor(t % 60)).padStart(2, "0")}`;
  const now = Math.min(songNow(), d || 0);
  $("stSongTime").textContent = `${fmt(now)} / ${fmt(d || 0)}`;
  songGlow(side, d ? now / d : 0);
  if (!pk || !d) return;
  const x = Math.round(w * now / d), mid = h / 2, n = pk.pk.length;
  for (let px = 0; px < w; px++) {                  // the peaks (dim) and the loudness (bright); the played part lit
    const i0 = Math.floor(px * n / w), i1 = Math.max(i0 + 1, Math.floor((px + 1) * n / w));
    let p = 0, r = 0;
    for (let i = i0; i < i1; i++) { p = Math.max(p, pk.pk[i]); r = Math.max(r, pk.rms[i]); }
    const played = px < x;
    g.fillStyle = themeColor(played ? "--song-peak-played" : "--song-peak");
    g.fillRect(px, mid - p * mid * 0.92, 1, 2 * p * mid * 0.92);
    g.fillStyle = themeColor(played ? "--accent-text" : "--song-rms");
    g.fillRect(px, mid - r * mid * 0.92, 1, 2 * r * mid * 0.92);
  }
  const drag = S.song.drag;
  if (drag && drag.kind === "new" && Math.abs(drag.x1 - drag.x0) >= 5) {
    g.strokeStyle = blue; g.lineWidth = 2; g.setLineDash([]);
    g.strokeRect(Math.min(drag.x0, drag.x1), 1, Math.abs(drag.x1 - drag.x0), h - 2);
  } else {
    const r = (drag && drag.kind === "edge" && drag.live) || S.st.region;
    if (r) {
      const live = $("stSongLoop").checked, col = live ? blue : muted;
      const x0 = Math.round(w * r[0] / d), x1 = Math.round(w * Math.min(r[1], d) / d);
      g.strokeStyle = col; g.lineWidth = 2; g.setLineDash(live ? [] : [3, 2]);
      g.strokeRect(x0, 1, x1 - x0, h - 2); g.setLineDash([]);
      g.fillStyle = col;
      for (const xe of [x0, x1]) g.fillRect(xe - 5, mid - 12, 10, 24);      // the tabs: drag to refine
    }
  }
  g.strokeStyle = accent; g.lineWidth = 2;
  g.beginPath(); g.moveTo(x, 0); g.lineTo(x, h); g.stroke();
  g.fillStyle = accent;                             // grab handles top and bottom: drag them to scrub
  for (const [y0, dy] of [[0, 1], [h, -1]]) { g.beginPath(); g.moveTo(x - 7, y0); g.lineTo(x + 7, y0); g.lineTo(x, y0 + 10 * dy); g.fill(); }
}
// block activity (description block_activity): each block's name glows orange with how hard its LoRA pushes at the
// playhead - playing, paused or scrubbed - on the side heard; scaled to the song's own strongest pushes, a 5-point
// window so it reads steady; none for a side without a map (No LoRA)
function songGlow(side, frac) {
  const act = side && S.song.act[side];
  if (!act) {
    if (Object.values(S.song.glow).some(Boolean)) {
      for (const [id, lvl] of Object.entries(S.song.glow)) if (lvl) songGlowSet(id, 0);
      S.song.glow = {};
    }
    return;
  }
  for (const [id, v] of Object.entries(act.blocks)) {
    const i = Math.floor(Math.min(0.999, Math.max(0, frac)) * v.length), lo = Math.max(0, i - 2), hi = Math.min(v.length, i + 3);
    let sum = 0;
    for (let j = lo; j < hi; j++) sum += v[j];
    const level = Math.round(Math.min(1, sum / Math.max(1, hi - lo) / act.top) * 8);
    if ((S.song.glow[id] || 0) !== level) songGlowSet(id, level);
  }
}
function songGlowSet(id, level) {
  S.song.glow[id] = level;
  const box = document.querySelector(`#stBlocks .st-block[data-id="${CSS.escape(id)}"]`);
  const lab = box && box._rows && box._rows[0].querySelector(".st-label");
  if (lab) lab.style.background = level ? themeColor("--data-orange", (0.85 * level / 8).toFixed(3)) : "";
}

function wireSong() {
  const cv = $("stSongBar"), GRAB = 10;
  const xOf = (e) => e.clientX - cv.getBoundingClientRect().left;
  const tOf = (x) => Math.max(0, Math.min(1, x / (cv.clientWidth || 1))) * songDur();
  const headX = () => { const d = songDur(); return d ? cv.clientWidth * Math.min(songNow(), d) / d : 0; };
  const edgeAt = (e) => {                          // 0 / 1 on the section's start / end tab (the middle of the bar)
    const r = S.st.region, d = songDur(), y = e.clientY - cv.getBoundingClientRect().top;
    if (!r || !d || y < cv.clientHeight * 0.25 || y > cv.clientHeight * 0.75) return null;
    const xs = [cv.clientWidth * r[0] / d, cv.clientWidth * Math.min(r[1], d) / d], x = xOf(e);
    const best = Math.abs(x - xs[0]) <= Math.abs(x - xs[1]) ? 0 : 1;
    return Math.abs(x - xs[best]) <= GRAB ? best : null;
  };
  const onHead = (e) => !!songDur() && Math.abs(xOf(e) - headX()) <= GRAB;
  cv.addEventListener("pointermove", (e) => {
    const D = S.song.drag;
    if (!D) { cv.style.cursor = edgeAt(e) != null || onHead(e) ? "ew-resize" : "pointer"; return; }
    if (D.kind === "edge") { const r = [...S.st.region]; r[D.edge] = tOf(xOf(e)); D.live = [Math.min(...r), Math.max(...r)]; }
    else if (D.kind === "scrub") { S.song.pos = tOf(xOf(e)); const p = S.song.side && songPlayer(S.song.side); if (p) p.currentTime = S.song.pos; }
    else D.x1 = xOf(e);
    songDraw();
  });
  cv.addEventListener("pointerdown", (e) => {
    if (e.button !== 0 || !songDur()) return;
    cv.setPointerCapture(e.pointerId);
    const edge = edgeAt(e);
    if (edge != null) S.song.drag = {kind: "edge", edge, live: null};
    else if (onHead(e)) { const was = songPlaying(); if (was) songStop(); S.song.drag = {kind: "scrub", resume: was}; }
    else S.song.drag = {kind: "new", x0: xOf(e), x1: xOf(e)};
  });
  cv.addEventListener("pointerup", (e) => {
    const D = S.song.drag;
    S.song.drag = null;
    if (!D) return;
    const d = songDur();
    if (D.kind === "edge") {
      if (!D.live) return songDraw();
      if (D.live[0] <= 0.15 && D.live[1] >= d - 0.15) return songRegion(null);   // out to both ends: the whole song
      return D.live[1] - D.live[0] >= 0.25 ? songRegion(D.live) : songDraw();
    }
    if (D.kind === "scrub") {
      S.song.pos = tOf(xOf(e));
      return D.resume ? songPlay(S.song.side || songStripSide(), S.song.pos) : songDraw();
    }
    if (Math.abs(xOf(e) - D.x0) < 5) {              // a click: jump there
      const t = tOf(xOf(e));
      if (songPlaying()) songPlay(S.song.side, t); else { S.song.pos = t; const p = S.song.side && songPlayer(S.song.side); if (p && p.getAttribute("src")) p.currentTime = t; songDraw(); }
      return;
    }
    const t0 = tOf(Math.min(D.x0, xOf(e))), t1 = tOf(Math.max(D.x0, xOf(e)));
    if (t1 - t0 < 0.25) return songDraw();
    songRegion([t0, t1]);
  });
  cv.addEventListener("contextmenu", (e) => { e.preventDefault(); songRegion(null); });   // right-click: the whole song
  for (const s of Object.keys(SONG_SIDES)) {
    $("stSong" + s[0].toUpperCase() + s.slice(1)).addEventListener("click", () => songPlay(s));
    songPlayer(s).addEventListener("ended", () => { S.song.pos = 0; songMark(); songDraw(); });
  }
  $("stSongPause").addEventListener("click", () => {
    if (songPlaying()) return songStop();
    const s = songHas(S.song.side) ? S.song.side : songStripSide();
    if (s) songPlay(s);
  });
  $("stSongLoop").addEventListener("change", () => {     // switched while playing: carry on from the same moment
    for (const s of Object.keys(SONG_SIDES)) songPlayer(s).loop = $("stSongLoop").checked && !S.st.region;
    studioRegionDraw();
    if (songPlaying()) songPlay(S.song.side);
  });
  window.addEventListener("resize", songDraw);
  songMark();
}

function studioRecordButton(input, set) {
  // a take from this device's microphone (the browser's), the Length long, sent to the Fizgig machine
  const btn = el("button", {class: "btn", text: "🎙 Record", title: "Record from the microphone of the device you're using, for the Length setting, then use the take."});
  btn.addEventListener("click", async () => {
    if (btn.disabled) return;
    let stream;
    try { stream = await navigator.mediaDevices.getUserMedia({audio: true}); }
    catch (e) { $("stStatus").textContent = `Could not start the microphone: ${e.message}`; return; }
    const secs = Math.max(1, Number($("stRes").value) || 30), rec = new MediaRecorder(stream), parts = [];
    rec.ondataavailable = (e) => e.data.size && parts.push(e.data);
    rec.onstop = async () => {
      stream.getTracks().forEach((t) => t.stop());
      clearInterval(tick); btn.disabled = false; btn.textContent = "🎙 Record";
      const ext = (rec.mimeType || "audio/webm").includes("ogg") ? "ogg" : rec.mimeType.includes("mp4") ? "m4a" : "webm";
      const file = new File(parts, `take_${Date.now()}.${ext}`, {type: rec.mimeType || "audio/webm"});
      try {
        const res = await uploadResumable({single: 1, folder: "", name: file.name}, file, () => {});
        if (!res.ok) throw new Error(res.data.error || "upload failed");
        input.value = res.data.path; set(res.data.path);
        $("stStatus").textContent = `Recorded ${secs} s.`;
      } catch (e) { $("stStatus").textContent = `Recording failed: ${e.message}`; }
    };
    const t0 = Date.now();
    const tick = setInterval(() => { $("stStatus").textContent = `Recording: ${Math.max(0, secs - (Date.now() - t0) / 1000).toFixed(0)} s left…`; }, 250);
    btn.disabled = true; btn.textContent = "⏺ Recording…";
    rec.start(); setTimeout(() => rec.state !== "inactive" && rec.stop(), secs * 1000);
  });
  return btn;
}

function wireStudioAce() {
  const change = () => { studioAceLayout(); studioAceRemember(); studioRegionDraw(); studioSchedule(); };
  for (const id of ["stPbOn", "stSrcOn", "stSndOn", "stSrcMode", "stPbKeep", "stPbBlend", "stSrcFollow", "stSrcDenoise", "stSrcStrength"])
    $(id).addEventListener("change", change);
  for (const id of ["stPbBlend"]) $(id).addEventListener("input", studioAceLayout);
  // typed text and new files render on Update, as the prompt does
  const typed = () => { studioAceRemember(); if (S.st.ready) $("stStart").textContent = "Update"; };
  for (const id of ["stPromptB", "stSrcPath", "stSndPath"]) $(id).addEventListener("input", typed);
  for (const [id, host] of [["stSrcPath", "stSrcBtns"], ["stSndPath", "stSndBtns"]]) {
    const input = $(id), set = (p) => { input.value = p; studioAceRemember(); studioSchedule(); };
    const up = uploadButton({kind: "path", key: "STUDIO_" + id}, input, set);
    $(host).replaceChildren(el("button", {class: "btn", text: "Choose…", onclick: () => pick(input.value || S.imageFolder, false, set)}),
      up.btn, studioRecordButton(input, set), up.status);
  }
  $("stTweak").addEventListener("click", (e) => { if (S.st.layout && S.st.layout.kind === "audio") e.stopPropagation(); });  // no viewer for a waveform
  $("stBase").addEventListener("click", (e) => { if (S.st.layout && S.st.layout.kind === "audio") e.stopPropagation(); });
  $("stRegionClear").addEventListener("click", () => songRegion(null));
  wireSong();
  $("stKeepBtn").addEventListener("click", async () => {
    try {
      const r = await api("/api/studio/keep", {});
      $("stSrcOn").checked = true; $("stSrcPath").value = r.path;
      if ($("stSrcMode").value === "song") $("stSrcMode").value = "repaint";
      studioAceLayout(); studioAceRemember(); studioRegionDraw();
      $("stStatus").textContent = "The Tweaked song is now the source: drag across the waveform to choose the section to repaint" +
        (S.st.region ? " (or keep this one)." : ".");
      if (S.st.region) studioSchedule();
    } catch (e) { $("stStatus").textContent = e.message; }
  });
}

function studioAceNotes(n) {
  if (!n) return;
  $("stSrcDenoiseNote").textContent = n.denoise || ""; $("stSrcFollowNote").textContent = n.follow || "";
  $("stSrcRepNote").textContent = n.strength || "";
}

// ---- Model EQ Mode (a family with a Model EQ LoRA in Preferences): the row holds the model's own weights ----
const ST_EQ_ROWS = {primary: ["stPrimary", "stPrimaryPick", "stPScale", "stEqP"], donor: ["stDonor", "stDonorPick", "stDScale", "stEqD"]};

function studioEqButtons(L) {
  const on = !!(L && L.model_eq);
  for (const role of ["primary", "donor"]) {
    if (!on && S.st.eq[role]) studioEq(role, false, false);
    $(ST_EQ_ROWS[role][3]).style.display = on ? "" : "none";
  }
}

function studioEq(role, on, restore = true) {
  // on: the row loads the Model EQ LoRA at strength 1, its path, Choose and strength locked (the button lit);
  // off: unlocked and (restore) the LoRA and strength it had before come back - as the desktop's _repair_set_eq_mode
  const [pathId, pickId, scaleId, btnId] = ST_EQ_ROWS[role];
  const L = S.st.layout;
  if (on) {
    if (!L || !L.model_eq) { $("stStatus").textContent = "Set the Model EQ LoRA in Preferences first."; return; }
    S.st.eqSaved[role] = [$(pathId).value, $(scaleId).value];
    $(pathId).value = L.model_eq; $(scaleId).value = "1";
  }
  S.st.eq[role] = on;
  for (const id of [pathId, pickId, scaleId]) $(id).disabled = on;
  $(btnId).classList.toggle("btn-primary", on);
  if (on) {
    if (role === "primary") { $("stStatus").textContent = "Model EQ as the primary: press Start / Update to load it (1.0 = the stock model).";
      if (S.st.ready) $("stStart").textContent = "Update"; }
    else if (S.st.ready) api("/api/studio/donor", {path: L.model_eq}).then(studioPoll).catch((e) => { $("stStatus").textContent = e.message; });
    stRemember();
    return;
  }
  if (!restore) { delete S.st.eqSaved[role]; return; }
  const [prev, prevScale] = S.st.eqSaved[role] || ["", String((L && L.default_strength) || 1)];
  delete S.st.eqSaved[role];
  $(scaleId).value = prevScale || "1";
  if (role === "primary") {
    $(pathId).value = prev;
    $("stStatus").textContent = "Model EQ Mode off: the primary is your own LoRA again - press Update to load it.";
    if (S.st.ready) $("stStart").textContent = "Update";
  } else {
    $(pathId).value = prev;
    if (S.st.ready) api("/api/studio/donor", {path: prev}).then(() => { studioPoll(); studioSchedule(); });
  }
  stRemember();
}

function studioEqNote(st) {
  // after a load: what the sliders mean for a Model EQ row (the engine's gain for that file)
  const eq = st.eq || {}, bits = [];
  for (const [role, g] of Object.entries(eq)) if (g != null)
    bits.push(`${role === "primary" ? "Primary" : "Donor"} is the Model EQ: 1.0 = the stock model, 0 / 2 = -${Math.round(g * 100)}% / +${Math.round(g * 100)}% of each block's main directions.`);
  return bits.join(" ");
}

// ---- the mix: Epochs of this run / LoRA mix for the primary, the donor's epoch blend (the desktop's) ----
// S.st.mix = {mode: "off"|"epochs"|"mix", epochs: [n...], upto, hist, rows: [[path, amount]...]}; S.st.dmix the donor's
function studioMixReset(note) {
  S.st.mix = {mode: "off", epochs: [], rows: []}; S.st.dmix = {on: false, epochs: []};
  document.querySelector('input[name="stMixMode"][value="off"]').checked = true;
  $("stDmixOn").checked = false;
  $("stMixInfo").textContent = note || ""; $("stDmixInfo").textContent = "";
  studioMixLayout();
}

function studioDmixOff() {
  if (!S.st.dmix) return;
  S.st.dmix = {on: false, epochs: []}; $("stDmixOn").checked = false; $("stDmixInfo").textContent = ""; studioMixLayout();
}

function studioMixShow(L) {
  const on = !!L && L.kind !== "video";                 // H3's engine has no mix (as the desktop)
  $("stMixRows").style.display = $("stDmixRows").style.display = on ? "" : "none";
  studioMixReset();
}

function studioMixLayout() {
  const m = S.st.mix || {mode: "off"};
  $("stMixEpochs").style.display = m.mode === "epochs" ? "" : "none";
  $("stMixList").style.display = m.mode === "mix" ? "" : "none";
  for (const [p, st, on] of [["stMix", m, m.mode === "epochs"], ["stDmix", S.st.dmix || {}, !!(S.st.dmix || {}).on]]) {
    const eps = st.epochs || [];
    for (const id of [p + "Upto", p + "Hist"]) $(id).disabled = !on || eps.length < 2;
    $(p + "Upto").max = Math.max(1, eps.length - 1);
    const i = Math.max(0, Math.min(eps.length - 1, parseInt($(p + "Upto").value, 10) || 0));
    $(p + "UptoTxt").textContent = eps.length ? `epoch ${eps[i]}` : "";
    const h = snap05($(p + "Hist").value);
    $(p + "HistTxt").textContent = !eps.length ? "" : h < 0.025 ? "0 (that epoch alone)" : `${h.toFixed(2)} (about ${Math.round(1 / (1 - h))} epochs)`;
  }
}

async function studioMixEpochs(role) {
  // the run beside the role's file; false when there is nothing to blend (the note says why)
  const path = (role === "primary" ? $("stPrimary") : $("stDonor")).value.trim();
  const info = role === "primary" ? "stMixInfo" : "stDmixInfo";
  if (!S.st.ready || (role === "donor" && !S.st.donor)) {
    $("stStatus").textContent = `Load the ${role} LoRA first, then ${role === "primary" ? "pick a mix" : "tick its epoch blend"}.`;
    return false;
  }
  let r;
  try { r = await api(`/api/studio/epochs?path=${encodeURIComponent(path)}`); }
  catch (e) { $("stStatus").textContent = `Could not read the ${role}'s epochs: ${e.message}`; return false; }
  if ((r.epochs || []).length < 2) {
    $(info).textContent = `Only one epoch of the ${role}'s run is beside its file, so there is nothing to blend.`;
    return false;
  }
  const st = role === "primary" ? S.st.mix : S.st.dmix, p = role === "primary" ? "stMix" : "stDmix";
  st.epochs = r.epochs;
  $(p + "Upto").max = r.epochs.length - 1;
  $(p + "Upto").value = r.epochs.indexOf(r.own) >= 0 ? r.epochs.indexOf(r.own) : r.epochs.length - 1;
  return true;
}

function studioMixRows() {
  const host = $("stMixListRows");
  host.replaceChildren();
  S.st.mix.rows.forEach(([path, amount], i) => {
    const name = path.split(/[\\/]/).pop().replace(/\.safetensors$/i, "");
    const val = el("span", {class: "field-hint", text: Number(amount).toFixed(2)});
    const range = el("input", {type: "range", min: "-1", max: "2", step: "0.05", value: String(amount)});
    range.addEventListener("input", () => { val.textContent = Number(range.value).toFixed(2); });
    range.addEventListener("change", () => { S.st.mix.rows[i][1] = snap05(range.value); studioSchedule(); });
    host.append(el("div", {class: "field-row"}, el("span", {class: "field-label"}), el("div", {class: "st-mixrow"},
      el("span", {class: "st-mixname", title: path, text: (i === 0 ? "★ " : "") + name}), range, val,
      i === 0 ? el("span", {class: "field-hint", text: "(the primary)"}) :
        el("button", {class: "btn btn-sm", text: "✕", title: "Take it out of the mix", onclick: () => {
          S.st.mix.rows.splice(i, 1); studioMixRows(); studioSchedule(); }}))));
  });
}

async function studioMixMode(mode) {
  const m = S.st.mix;
  if (mode !== "off" && !S.st.ready) {
    document.querySelector('input[name="stMixMode"][value="off"]').checked = true;
    $("stStatus").textContent = "Load the primary LoRA first, then pick a mix.";
    return;
  }
  m.mode = mode;
  if (mode === "epochs" && !(await studioMixEpochs("primary"))) {
    m.mode = "off"; document.querySelector('input[name="stMixMode"][value="off"]').checked = true;
  }
  if (mode === "mix") {
    const prim = $("stPrimary").value.trim();
    if (!m.rows.length || m.rows[0][0] !== prim) m.rows = [[prim, 1.0]];
    studioMixRows();
  }
  if (m.mode === "off") $("stMixInfo").textContent = "Off: the primary file alone.";
  studioMixLayout();
  studioSchedule();
}

function studioMixExtra() {
  // what the worker mixes before the render: {primary, donor}, each null for the file alone
  const L = S.st.layout;
  if (!L || L.kind === "video" || !S.st.mix) return {};
  const m = S.st.mix, d = S.st.dmix || {};
  const dial = (p, eps) => ({upto: eps[Math.max(0, Math.min(eps.length - 1, parseInt($(p + "Upto").value, 10) || 0))],
    hist: snap05($(p + "Hist").value)});
  return {mix: {
    primary: m.mode === "epochs" && m.epochs.length > 1 ? {mode: "epochs", ...dial("stMix", m.epochs)} :
      m.mode === "mix" && m.rows.length ? {mode: "mix", paths: m.rows.map((r) => r[0]), amounts: m.rows.map((r) => r[1])} : null,
    donor: d.on && d.epochs.length > 1 && S.st.donor ? {mode: "epochs", ...dial("stDmix", d.epochs)} : null}};
}

function wireStudioMix() {
  document.querySelectorAll('input[name="stMixMode"]').forEach((r) => r.addEventListener("change", () => studioMixMode(r.value)));
  for (const p of ["stMix", "stDmix"]) for (const id of [p + "Upto", p + "Hist"]) {
    $(id).addEventListener("input", studioMixLayout);
    $(id).addEventListener("change", () => { studioMixLayout(); studioSchedule(); });
  }
  $("stDmixOn").addEventListener("change", async () => {
    S.st.dmix.on = $("stDmixOn").checked;
    if (S.st.dmix.on && !(await studioMixEpochs("donor"))) { S.st.dmix.on = false; $("stDmixOn").checked = false; }
    if (!S.st.dmix.on) $("stDmixInfo").textContent = "Off: the donor file alone.";
    studioMixLayout(); studioSchedule();
  });
  $("stMixAdd").addEventListener("click", () => pick($("stPrimary").value.trim(), false, async (p) => {
    if (S.st.mix.rows.some((r) => r[0].toLowerCase() === p.toLowerCase())) return;
    try {
      const f = await api(`/api/studio/lorafamily?path=${encodeURIComponent(p)}`);
      if (f.key && S.st.layout && f.key !== S.st.layout.key) {
        $("stStatus").textContent = `Not added: ${p.split(/[\\/]/).pop()} is a LoRA for ${f.name}.`;
        return;
      }
    } catch (e) { /* unreadable: the render says so */ }
    S.st.mix.rows.push([p, 0.5]); studioMixRows(); studioSchedule();
  }));
}

function studioMixNotes(n) {
  if (!n) return;
  if (n.mix_primary != null && S.st.mix && S.st.mix.mode !== "off") $("stMixInfo").textContent = n.mix_primary;
  if (n.mix_donor != null && S.st.dmix && S.st.dmix.on) $("stDmixInfo").textContent = n.mix_donor;
}

// ---- the metrics strip (the desktop's pop-out strip): paired numbers for the renders on screen ----
function studioMetricsShow() {
  const L = S.st.layout, on = !!L && L.kind !== "audio";
  $("stMetrics").style.display = on ? "" : "none";
  const ref = (stSaved().metricsRef) || "";
  $("stMetRefName").textContent = ref ? ref.split(/[\\/]/).pop() : "(none)";
}

async function studioMetrics() {
  const L = S.st.layout;
  if (!L || L.kind === "audio" || !S.st.shown) return;
  const gen = (S.st.metGen = (S.st.metGen || 0) + 1), ref = stSaved().metricsRef || "";
  const chips = {like: "stMetLike", grid: "stMetGrid", tex: "stMetTex", clip: "stMetClip", sat: "stMetSat"};
  const put = (k, text, cls) => { const c = $(chips[k]); c.textContent = text; c.className = "st-chip" + (cls ? " " + cls : ""); };
  for (const k in chips) put(k, "…");
  let m;
  try { m = await api(`/api/studio/metrics?ref=${encodeURIComponent(ref)}`); } catch (e) { m = {error: e.message}; }
  if (gen !== S.st.metGen) return;                    // a newer pair is on screen: these numbers aren't its
  if (m.error) { put("like", `metrics failed: ${String(m.error).slice(0, 60)}`, "bad"); for (const k of ["grid", "tex", "clip", "sat"]) put(k, ""); return; }
  if (!m.ref_set) put("like", "Face likeness: set a reference photo →");
  else if (!m.ref_face) put("like", "Face likeness: no face in reference", "bad");
  else if (m.like_base == null || m.like_tweak == null) put("like", "Face likeness: no face in render");
  else {
    const lb = m.like_base * 100, lt = m.like_tweak * 100, d = lt - lb;
    put("like", `Face likeness ${lb.toFixed(0)}% ${d > 0.5 ? "▲" : d < -0.5 ? "▼" : "→"} ${lt.toFixed(0)}%`, d > 0.5 ? "good" : d < -0.5 ? "bad" : "");
  }
  $(chips.like).title = m.embedder ? `Look match: ${m.embedder} (scores are on its own scale)` : "";
  put("grid", `Grid ${m.grid_base.toFixed(2)} → ${m.grid_tweak.toFixed(2)}`, m.grid_delta > 0.05 ? "bad" : m.grid_delta < -0.05 ? "good" : "");
  const td = (m.texture_tweak - m.texture_base) / Math.max(m.texture_base, 1e-6);
  put("tex", `Detail ${m.texture_base.toFixed(0)} → ${m.texture_tweak.toFixed(0)}`, Math.abs(td) > 0.10 ? "warm" : "");
  put("clip", `Clipped ${m.clip_base.toFixed(1)}% → ${m.clip_tweak.toFixed(1)}%`, m.clip_delta > 0.2 ? "bad" : m.clip_delta < -0.2 ? "good" : "");
  put("sat", `Sat ${m.sat_base.toFixed(0)} → ${m.sat_tweak.toFixed(0)}`, Math.abs(m.sat_delta) > 12 ? "warm" : "");
}

function wireStudioMetrics() {
  const setRef = (p) => {
    try { const d = stSaved(); d.metricsRef = p || ""; localStorage.setItem(ST_KEY, JSON.stringify(d)); } catch (e) { /* private window */ }
    studioMetricsShow(); studioMetrics();
  };
  $("stMetRefPick").addEventListener("click", () => pick(stSaved().metricsRef || refStart(), false, setRef));
  $("stMetRefClear").addEventListener("click", () => setRef(""));
}

// ---- Save / Load Setup: the whole session in one file, in the desktop's format (its variable names), so either
// app loads the other's. What only the web has (H3's photos and crops, the mix, the section) rides along in "web".
const ST_SRC_MODES = [["song", "Whole song"], ["repaint", "Repaint the loop section"], ["extend", "Extend to the Length"]];
const ST_SETUP_MAP = [
  ["stPrimary", "repair_primary_var"], ["stDonor", "repair_donor_var"], ["stPScale", "repair_primary_scale_var"],
  ["stDScale", "repair_donor_scale_var"], ["stPrompt", "repair_prompt_var"], ["stSeed", "repair_seed_var"],
  ["stRes", "repair_res_var"], ["stW", "repair_width_var"], ["stH", "repair_height_var"], ["stNeg", "repair_negative_var"], ["stTurbo", "repair_turbo_var", "check"],
  ["stRef", "repair_ref_path_var"], ["stRefMp", "repair_ref_mp_var"], ["stRefStr", "repair_ref_strength_var"],
  ["stLuckyLo", "repair_lucky_min_var"], ["stLuckyHi", "repair_lucky_max_var"],
  ["stH3Model", "repair_h3_model_var"], ["stH3Base", "repair_h3_base_var"], ["stH3Len", "repair_h3_frames_var"],
  ["stH3W", "repair_h3_width_var"], ["stH3H", "repair_h3_height_var"],
  ["stH3Steps", "repair_h3_steps_var"], ["stH3Turbo", "repair_h3_turbo_var"], ["stH3Sound", "repair_h3_sound_var", "check"],
  ["stH3Boost", "repair_h3_speed_boost_var", "check"],
  ["stH3Early", "repair_h3_early_var", "check"],
  ["stPbOn", "repair_pb_on_var", "check"], ["stPromptB", "repair_prompt_b_var"], ["stPbBlend", "repair_pb_blend_var", "num"],
  ["stPbKeep", "repair_pb_keep_lyrics_var", "check"], ["stSrcOn", "repair_src_on_var", "check"],
  ["stSrcPath", "repair_src_path_var"], ["stSrcMode", "repair_src_mode_var", "srcmode"],
  ["stSrcFollow", "repair_src_follow_var", "num"], ["stSrcDenoise", "repair_src_denoise_var", "num"],
  ["stSrcStrength", "repair_src_strength_var", "num"], ["stSndOn", "repair_sound_on_var", "check"],
  ["stSndPath", "repair_sound_path_var"]];

function studioSetupData() {
  const values = {};
  for (const [id, name, kind] of ST_SETUP_MAP) {
    const e = $(id);
    values[name] = kind === "check" ? e.checked : kind === "num" ? Number(e.value) :
      kind === "srcmode" ? (ST_SRC_MODES.find((m) => m[0] === e.value) || ST_SRC_MODES[0])[1] : e.value;
  }
  values.repair_dit_choice_var = (document.querySelector('input[name="stDit"]:checked') || {}).value === "full" ? "base" : "distilled";
  values.repair_metrics_ref_var = stSaved().metricsRef || "";
  const blocks = {};
  for (const [id, b] of Object.entries(S.st.blocks || {}))
    blocks[id] = {primary_enabled: b.pe, primary_strength: b.ps, donor_enabled: b.de, donor_strength: b.ds};
  return {fizgig_repair_setup: 1, family: $("stFamily").value, values, blocks,
    web: {kf: S.st.kf || {}, kfRect: S.st.kfRect || {}, region: S.st.region || null,
      mix: S.st.mix ? {mode: S.st.mix.mode, rows: S.st.mix.rows, upto: $("stMixUpto").value, hist: $("stMixHist").value} : null,
      dmix: S.st.dmix ? {on: S.st.dmix.on, upto: $("stDmixUpto").value, hist: $("stDmixHist").value} : null}};
}

async function studioSetupList(select) {
  try {
    const r = await api("/api/studio/setups");
    const sel = $("stSetupList");
    sel.replaceChildren(el("option", {value: "", text: "Load a setup…"}),
      ...r.setups.map((x) => el("option", {value: x.name, text: x.family ? `${x.name}  (${x.family})` : x.name})));
    if (select) sel.value = select;
  } catch (e) { /* the list stays as it was */ }
}

async function studioSetupSave(overwrite) {
  const name = $("stSetupName").value.trim() || `${$("stFamily").value}_setup`;
  try {
    const r = await api("/api/studio/setup_save", {name, data: studioSetupData(), overwrite: !!overwrite});
    if (r.exists) {
      $("stSetupMsg").replaceChildren(el("span", {text: `"${r.name}" is there already. `}),
        el("button", {class: "btn btn-sm btn-danger", text: "Replace it", onclick: () => studioSetupSave(true)}));
      return;
    }
    $("stSetupMsg").textContent = `Setup saved: ${r.name}`;
    $("stSetupName").value = r.name;
    studioSetupList();
  } catch (e) { $("stSetupMsg").textContent = e.message; }
}

async function studioSetupLoad(name) {
  // its model first (waiting for the switch to land), then every control and slider, then Start - which loads its
  // LoRAs and renders (the desktop's _load_repair_setup)
  let d;
  try { d = await api(`/api/studio/setup?name=${encodeURIComponent(name)}`); }
  catch (e) { $("stSetupMsg").textContent = e.message; return; }
  const fam = d.family && [...$("stFamily").options].some((o) => o.value === d.family) ? d.family : $("stFamily").value;
  if (fam !== $("stFamily").value) { $("stFamily").value = fam; await studioFamily(); stRemember(); }
  const v = d.values || {};
  for (const [id, key, kind] of ST_SETUP_MAP) {
    if (!(key in v)) continue;
    const e = $(id), val = v[key];
    if (kind === "check") e.checked = !!val;
    else if (kind === "srcmode") e.value = (ST_SRC_MODES.find((m) => m[1] === val || m[0] === val) || ST_SRC_MODES[0])[0];
    else if (e.tagName === "SELECT" && ![...e.options].some((o) => o.value === String(val))) continue;
    else e.value = String(val);
  }
  if ("repair_dit_choice_var" in v) {
    const want = v.repair_dit_choice_var === "base" ? "full" : "fast";
    const r = document.querySelector(`input[name="stDit"][value="${want}"]`); if (r) r.checked = true;
  }
  try { const s = stSaved(); s.metricsRef = v.repair_metrics_ref_var || ""; localStorage.setItem(ST_KEY, JSON.stringify(s)); } catch (e) { /* private */ }
  for (const [id, b] of Object.entries(d.blocks || {})) {
    if (!S.st.blocks[id]) continue;
    S.st.blocks[id] = {pe: !!b.primary_enabled, ps: Number(b.primary_strength), de: !!b.donor_enabled, ds: Number(b.donor_strength)};
  }
  const w = d.web || {};
  if (w.kf) { S.st.kf = w.kf; S.st.kfRect = w.kfRect || {}; }
  if (S.st.layout && S.st.layout.kind === "video") studioSlots();
  S.st.region = w.region || null;
  studioEq("primary", false, false); studioEq("donor", false, false);
  studioAceLayout(); studioAceRemember(); studioH3Remember(); stRemember(); studioMetricsShow(); studioNegRow();
  studioSync(); studioCommit();
  studioMixReset();
  S.st.pendingMix = w.mix || w.dmix ? w : null;          // the mix follows once its LoRAs are loaded
  $("stSetupMsg").textContent = `Setup loaded: ${name}`;
  $("stSetupName").value = name;
  if ($("stPrimary").value.trim()) {
    await studioStart();                                // a new LoRA / model loads; the same one just renders
    if (S.st.ready && S.st.pendingMix) studioSetupMix();
  }
}

async function studioSetupMix() {
  // after a setup's load: its mix back (the run's epochs need the LoRA there)
  const w = S.st.pendingMix;
  S.st.pendingMix = null;
  if (!w) return;
  if (w.mix && w.mix.mode === "mix" && (w.mix.rows || []).length) {
    S.st.mix.rows = w.mix.rows; document.querySelector('input[name="stMixMode"][value="mix"]').checked = true;
    await studioMixMode("mix");
  } else if (w.mix && w.mix.mode === "epochs") {
    $("stMixHist").value = w.mix.hist;
    document.querySelector('input[name="stMixMode"][value="epochs"]').checked = true;
    await studioMixMode("epochs");
    $("stMixUpto").value = w.mix.upto; studioMixLayout();
  }
  if (w.dmix && w.dmix.on && S.st.donor) {
    $("stDmixOn").checked = true; S.st.dmix.on = true;
    if (await studioMixEpochs("donor")) { $("stDmixUpto").value = w.dmix.upto; $("stDmixHist").value = w.dmix.hist; }
    else { S.st.dmix.on = false; $("stDmixOn").checked = false; }
    studioMixLayout();
  }
  studioSchedule();
}

function wireStudioSetups() {
  $("stSetupSave").addEventListener("click", () => studioSetupSave(false));
  $("stSetupList").addEventListener("change", () => { const n = $("stSetupList").value; $("stSetupList").value = ""; if (n) studioSetupLoad(n); });
  studioSetupList();
}

function studioCanvas() {
  // H3: [width, height, frames, keyframes, references] of the first clip, so Auto loads the base with room for it
  if (S.st.layout.kind !== "video") return null;
  const x = studioExtra(), n = x.keyframes.length;
  return [x.video.width, x.video.height, x.video.frames, x.ref_mode ? 0 : n, x.ref_mode ? n : 0];
}

function studioNegRow() {
  const L = S.st.layout;
  if (!L) return;
  const fast = (document.querySelector('input[name="stDit"]:checked') || {}).value !== "full";
  const cfg = L.follows_samples ? (L.follow_ck ? L.cfg_fast : L.follow_cfg) : fast && L.dit_fast ? L.cfg_fast : L.cfg_full;   // Samples-tab models: theirs
  $("stNegRow").style.display = L.negative_default != null && cfg > 1 ? "" : "none";
}

// ---- the sliders ----
const stFmt = (v) => Math.abs(v - 1) < 1e-9 ? "1.00" : (v >= 0 ? "+" : "") + v.toFixed(2);
function studioRows() {
  const host = $("stBlocks");
  host.innerHTML = "";
  for (const g of S.st.layout.groups) {
    const half = Math.ceil(g.blocks.length / 2);
    const cols = el("div", {class: "st-cols"}, el("div", {}, ...g.blocks.slice(0, half).map(studioRow)),
      el("div", {}, ...g.blocks.slice(half).map(studioRow)));
    host.append(el("div", {class: "st-group"}, el("h3", {text: g.label}), cols));
  }
  studioActivity();
}

function studioRow(b) {
  const box = el("div", {class: "st-block", "data-id": b.id});
  const mk = (side) => {
    const st = () => S.st.blocks[b.id];
    const en = side === "p" ? "pe" : "de", sv = side === "p" ? "ps" : "ds";
    const tick = el("input", {type: "checkbox", title: side === "p" ? "This block on" : "The donor's block on",
      onchange: (e) => studioSet(b.id, en, e.target.checked)});
    const range = el("input", {type: "range", min: "-3", max: "3", step: "0.05",
      oninput: (e) => studioSet(b.id, sv, Number(e.target.value), true), onchange: () => studioCommit()});
    const val = el("span", {class: "st-val"});
    const btn = (t, title, f) => el("button", {class: "btn btn-sm", text: t, title, onclick: () => { f(); studioCommit(); }});
    const label = side === "p" ? el("span", {class: "st-label", style: b.color ? `color:${b.color}` : ""},
      b.label, b.cat ? el("small", {text: `[${b.cat}]`}) : null) : el("span", {class: "st-label", text: "donor"});
    const locks = () => side === "p" ? S.st.locks : S.st.dlocks, frozen = () => side === "p" ? S.st.frozen : S.st.dfrozen;
    const lock = el("button", {class: "st-lock", title: side === "p" ?
      "Lock this slider (presets, Lucky and bulk leave it alone; the donor's slider has its own lock)" :
      "Lock the donor's slider for this block (presets, Lucky and bulk leave it alone)",
      text: "🔓", onclick: () => {
        if (locks().has(b.id)) { locks().delete(b.id); frozen().delete(b.id); } else locks().add(b.id);
        studioSync([b.id]); }});
    const row = el("div", {class: "st-row" + (side === "d" ? " donor" : "")}, lock, tick, label, range, val,
      el("span", {class: "st-btns"},
        btn("0", "Set to 0", () => studioSet(b.id, sv, 0)), btn("1", "Set to 1", () => studioSet(b.id, sv, 1)),
        btn("±", "Flip the sign", () => studioSet(b.id, sv, -st()[sv])),
        btn("−", "−0.1", () => studioSet(b.id, sv, Math.max(-3, Math.round((st()[sv] - 0.1) * 100) / 100))),
        btn("+", "+0.1", () => studioSet(b.id, sv, Math.min(3, Math.round((st()[sv] + 0.1) * 100) / 100))),
        btn("⇄", "Balance: the other side gets 1 − this (with a donor)", () => {
          const o = side === "p" ? "ds" : "ps", v = st()[sv];
          if (S.st.donor && v >= 0 && v <= 1) { studioSet(b.id, o, Math.round((1 - v) * 100) / 100); studioSet(b.id, "de", true); }
        })));
    row._sync = () => { tick.checked = st()[en]; range.value = st()[sv]; val.textContent = stFmt(st()[sv]);
      const on = locks().has(b.id); lock.textContent = on ? "🔒" : "🔓"; lock.classList.toggle("on", on); };
    row._sync();
    return row;
  };
  const p = mk("p"), d = mk("d");
  d.style.display = S.st.donor ? "" : "none";
  box.append(p, d);
  box._rows = [p, d];
  return box;
}

function studioSync(ids) {
  document.querySelectorAll("#stBlocks .st-block").forEach((box) => {
    if (!ids || ids.includes(box.dataset.id)) box._rows.forEach((r) => r._sync());
  });
  studioFreezeState();
}

// ---- I'm feeling lucky's pill: how many blocks a roll changes, and Freeze / Unfreeze the ones you've moved ----------
function studioLuckyCounts() {                     // Roll: 1, 2, 4 ... below the model's block count, then All
  const n = Object.keys(S.st.blocks).length, sel = $("stLuckyCount"), keep = stSaved().luckyCount || sel.value;
  sel.innerHTML = "";
  for (let k = 1; k < n; k *= 2) sel.append(el("option", {value: String(k), text: String(k)}));
  sel.append(el("option", {value: "all", text: "All"}));
  sel.value = [...sel.options].some((o) => o.value === keep) ? keep : "all";
}

function studioLuckyPool() {                      // the blocks a roll can change: on for the LoRA it rolls, not locked
  const who = S.st.donor ? $("stLuckyWho").value : "Primary";
  const pOk = (id) => who !== "Donor" && !S.st.locks.has(id) && !(S.st.ready && !S.st.primaryBlocks.has(id)) && S.st.blocks[id].pe;
  const dOk = (id) => who !== "Primary" && !S.st.dlocks.has(id) && S.st.donorBlocks.has(id) && S.st.blocks[id].de;
  const ids = Object.keys(S.st.blocks);
  return {pOk, dOk, who, pool: ids.filter((id) => pOk(id) || dOk(id)),
          free: ids.filter(pOk).length + ids.filter(dOk).length,       // sliders a roll can change
          total: (who !== "Donor" ? ids.length : 0) + (who !== "Primary" ? ids.length : 0)};
}

function studioTweaked(id, side) {                 // a slider moved from its default (its strength or on / off)
  const b = S.st.blocks[id];
  if (!b) return false;
  if (side === "d") return !!(S.st.donor && S.st.donorBlocks.has(id) && (b.ds !== 1 || !b.de));
  return !(S.st.ready && !S.st.primaryBlocks.has(id)) && (b.ps !== 1 || !b.pe);
}

function studioFreezeState() {
  if (!$("stFreeze") || !S.st.frozen) return;
  const ids = Object.keys(S.st.blocks);
  const fresh = ids.filter((id) => studioTweaked(id, "p") && !S.st.locks.has(id)).length +
    ids.filter((id) => studioTweaked(id, "d") && !S.st.dlocks.has(id)).length;
  const held = S.st.frozen.size + S.st.dfrozen.size;
  $("stFreeze").textContent = `🔒 Freeze tweaked${fresh ? ` (${fresh})` : ""}`;
  $("stFreeze").disabled = !fresh;
  $("stUnfreeze").style.display = held ? "" : "none";
  $("stUnfreeze").textContent = `🔓 Unfreeze (${held})`;
  // what is left to roll: not frozen, not locked by hand, on for the LoRA being rolled (the dropdown's sliders)
  const pl = studioLuckyPool();
  $("stLuckyFree").textContent = pl.total ? `${pl.free} of ${pl.total} sliders free to roll` : "";
}

function studioFreeze(on) {
  if (on) {
    for (const id of Object.keys(S.st.blocks)) {
      if (studioTweaked(id, "p") && !S.st.locks.has(id)) { S.st.locks.add(id); S.st.frozen.add(id); }
      if (studioTweaked(id, "d") && !S.st.dlocks.has(id)) { S.st.dlocks.add(id); S.st.dfrozen.add(id); }
    }
  } else {
    for (const id of S.st.frozen) S.st.locks.delete(id);
    for (const id of S.st.dfrozen) S.st.dlocks.delete(id);
    S.st.frozen.clear(); S.st.dfrozen.clear();
  }
  studioSync();
}

function studioActivity() {
  // a block the LoRA doesn't touch is greyed; the donor's rows show only with a donor
  // (as the desktop: its controls off, and an absent primary block back at on / 1 so no stale edit rides along into a
  // preset, a setup or the saved file)
  let reset = false;
  document.querySelectorAll("#stBlocks .st-block").forEach((box) => {
    const id = box.dataset.id;
    const pAbsent = S.st.ready && !S.st.primaryBlocks.has(id), dAbsent = S.st.donor && !S.st.donorBlocks.has(id);
    box._rows[0].classList.toggle("absent", pAbsent);
    box._rows[1].style.display = S.st.donor ? "" : "none";
    box._rows[1].classList.toggle("absent", dAbsent);
    box._rows[0].querySelectorAll("input,button").forEach((i) => { i.disabled = pAbsent; });
    box._rows[1].querySelectorAll("input,button").forEach((i) => { i.disabled = dAbsent; });
    const b = S.st.blocks[id];
    if (pAbsent && b && (!b.pe || b.ps !== 1)) { b.pe = true; b.ps = 1; reset = true; }
  });
  if (reset) studioSync();
  document.querySelectorAll(".st-donor-only").forEach((x) => { x.style.display = S.st.donor ? "" : "none"; });
  document.querySelectorAll(".st-detail-only").forEach((x) => {
    x.style.display = S.st.layout && (S.st.layout.detail_blocks || []).length ? "" : "none"; });
  $("stTargetDonor").disabled = !S.st.donor;
  studioFreezeState();                             // a donor in or out changes what is free to roll
}

function studioSet(id, k, v, live) {
  if ((k === "ds" || k === "de" ? S.st.dlocks : S.st.locks).has(id)) { studioSync([id]); return; }
  S.st.blocks[id][k] = v;
  studioSync([id]);
  studioSchedule(live);
}

// undo / redo: a snapshot after each change settles
function studioSnap() { return JSON.stringify(S.st.blocks); }
function studioCommit() {
  const now = studioSnap();
  if (S.st.hist[S.st.hist.length - 1] !== now) { S.st.hist.push(now); if (S.st.hist.length > 200) S.st.hist.shift(); S.st.redo = []; }
}
function studioUndo(redo) {
  const from = redo ? S.st.redo : S.st.hist, to = redo ? S.st.hist : S.st.redo;
  if (!redo && from.length < 2) { $("stStatus").textContent = "Nothing to undo."; return; }
  if (redo && !from.length) { $("stStatus").textContent = "Nothing to redo."; return; }
  if (redo) { const s = from.pop(); to.push(s); S.st.blocks = JSON.parse(s); }
  else { to.push(from.pop()); S.st.blocks = JSON.parse(from[from.length - 1]); }
  studioSync(); studioSchedule();
  $("stStatus").textContent = `${redo ? "Redone" : "Undone"} (${Math.max(0, S.st.hist.length - 1)} steps to undo, ${S.st.redo.length} to redo).`;
}

// ---- masters (Klein): one slider per category, on the primary or the donor ----
function studioMasters() {
  const L = S.st.layout;
  $("stMasters").style.display = L.masters.length ? "" : "none";
  const host = $("stMasterRows");
  host.innerHTML = "";
  for (const m of L.masters) {
    const range = el("input", {type: "range", min: "-3", max: "3", step: "0.05", value: "1"});
    const val = el("span", {class: "st-val", text: "1.00"});
    const apply = (v) => {
      const donor = (document.querySelector('input[name="stTarget"]:checked') || {}).value === "donor";
      const has = donor ? S.st.donorBlocks : S.st.primaryBlocks;
      for (const [id, c] of Object.entries(L.category_of))
        if (c === m.cat && (!S.st.ready || has.has(id)) && !(donor ? S.st.dlocks : S.st.locks).has(id)) S.st.blocks[id][donor ? "ds" : "ps"] = v;
      range.value = v; val.textContent = stFmt(v);
      studioSync(); studioSchedule(true);
    };
    range.addEventListener("input", () => apply(Number(range.value)));
    range.addEventListener("change", studioCommit);
    const b = (t, f, title) => el("button", {class: "btn btn-sm", text: t, title: title || "", onclick: () => { f(); studioCommit(); }});
    const balance = () => {              // the desktop's ⚖: the other side's category at 1 minus this one (0 to 1)
      const v = Number(range.value), donor = (document.querySelector('input[name="stTarget"]:checked') || {}).value === "donor";
      if (!(S.st.donorBlocks && S.st.donorBlocks.size) || v < 0 || v > 1) return;
      for (const [id, c] of Object.entries(L.category_of)) {
        if (c !== m.cat || (donor ? S.st.locks : S.st.dlocks).has(id)) continue;     // the other side's lock
        if (donor) S.st.blocks[id].ps = Math.round((1 - v) * 100) / 100;
        else Object.assign(S.st.blocks[id], {ds: Math.round((1 - v) * 100) / 100, de: true});
      }
      studioSync(); studioSchedule(true);
    };
    host.append(el("div", {class: "st-row"}, el("span", {}), el("span", {}),
      el("span", {class: "st-label", style: `color:${m.color}`, text: m.label}), range, val,
      el("span", {class: "st-btns"}, b("0", () => apply(0)), b("1", () => apply(1)), b("±", () => apply(-Number(range.value))),
        b("⚖", balance, "Balance: the other side (primary or donor) gets 1 minus this value, for values from 0 to 1"))));
    host.lastChild._cat = m.cat; host.lastChild._range = range; host.lastChild._val = val;
  }
}

function studioMastersFollow() {
  // flipping the target shows each category's average on that side
  const donor = (document.querySelector('input[name="stTarget"]:checked') || {}).value === "donor";
  for (const row of $("stMasterRows").children) {
    const ids = Object.entries(S.st.layout.category_of).filter(([, c]) => c === row._cat).map(([id]) => id);
    const vals = ids.map((id) => S.st.blocks[id][donor ? "ds" : "ps"]);
    const avg = vals.length ? vals.reduce((a, b) => a + b, 0) / vals.length : 1;
    row._range.value = avg; row._val.textContent = stFmt(avg);
  }
}

// ---- presets and bulk ----
async function studioPresets() {
  const L = S.st.layout, sel = $("stPreset");
  const user = (await api("/api/studio/presets?family=" + encodeURIComponent(L.key)).catch(() => ({user: []}))).user;
  sel.innerHTML = "";
  sel.append(el("option", {value: "", text: "Presets…"}));
  for (const p of L.presets) sel.append(el("option", {value: "b:" + p.name, text: p.name}));
  if (user.length) sel.append(el("optgroup", {label: "Your presets"}, ...user.map((n) => el("option", {value: "u:" + n, text: n}))));
}

async function studioPresetApply(v) {
  const L = S.st.layout;
  if (v.startsWith("b:")) {
    const p = L.presets.find((x) => x.name === v.slice(2));
    for (const [id, b] of Object.entries(S.st.blocks)) {
      if (!S.st.dlocks.has(id)) Object.assign(b, {de: true, ds: 1});
      if (S.st.locks.has(id)) continue;
      Object.assign(b, {pe: true, ps: 1});
      if (p.keep) b.pe = p.keep.includes(L.category_of[id]);
      if (p.blocks && id in p.blocks) { b.pe = true; b.ps = Number(p.blocks[id]); }
    }
  } else {
    const r = await api("/api/studio/preset_load", {family: L.key, name: v.slice(2)});
    for (const [id, bs] of Object.entries(r.blocks || {})) {
      const b = S.st.blocks[id];
      if (!b) continue;
      if (!S.st.locks.has(id)) Object.assign(b, {pe: !!bs.primary_enabled, ps: Number(bs.primary_strength)});
      if (!S.st.dlocks.has(id)) Object.assign(b, {de: !!bs.donor_enabled, ds: Number(bs.donor_strength)});
    }
  }
  studioSync(); studioCommit(); studioSchedule();
  $("stPreset").value = "";
}

// Royale's reference strength: 0 to 2 (0 = off), 1 when the box holds no number - as the desktop's
function refStrength(t) { const x = Number(String(t).trim()); return String(t).trim() === "" || !isFinite(x) ? 1 : Math.max(0, Math.min(2, x)); }

// a seed box: its number (0 is a seed too), 42 when it holds none
function seedOf(id) { const n = parseInt($(id).value, 10); return Number.isFinite(n) ? n : 42; }

function studioBulk(kind) {
  const L = S.st.layout;
  // Min / Max as the desktop reads them: within the sliders' -3..3, a box that isn't a number its default, in order
  const bound = (id, d) => { const t = $(id).value.trim(), x = Number(t); return t === "" || !isFinite(x) ? d : Math.max(-3, Math.min(3, x)); };
  const [lo, hi] = [bound("stLuckyLo", 0), bound("stLuckyHi", 1.5)].sort((x, y) => x - y);
  const order = L.groups.flatMap((g) => g.blocks.map((b) => b.id));
  // Alternate / Toggle detail blocks (the desktop's, H3's 0-49 and 46-49): for a model naming its detail blocks
  // (description detail_blocks); Alternate runs over its main blocks, its biggest group
  const main = L.groups.reduce((a, g) => (g.blocks.length > a.length ? g.blocks : a), []).map((b) => b.id);
  if (kind === "alternate" || kind === "detail") {
    const tail = (L.detail_blocks || []).filter((id) => S.st.blocks[id] && !S.st.locks.has(id));
    const on = kind === "detail" && !tail.every((id) => S.st.blocks[id].pe);
    (kind === "detail" ? tail : main).forEach((id, i) => {
      if (S.st.blocks[id] && !S.st.locks.has(id)) S.st.blocks[id].pe = kind === "detail" ? on : i % 2 === 0;
    });
    studioSync(); studioCommit(); studioSchedule(); stRemember();
    return;
  }
  let rolled = 0;
  for (const [id, b] of Object.entries(S.st.blocks)) {
    const p = !S.st.locks.has(id), d = !S.st.dlocks.has(id);
    if (kind === "reset") { if (p) Object.assign(b, {pe: true, ps: 1}); if (d) Object.assign(b, {de: true, ds: 1}); }
    else if (kind === "off") { if (p) b.pe = false; }
    else if (kind === "on") { if (p) b.pe = true; }
    else if (kind === "invert") { if (p) b.pe = !b.pe; }
    else if (kind === "donor_on") { if (d) Object.assign(b, {de: true, ds: 1}); }
    else if (kind === "donor_off") { if (d) b.ds = 0; }
  }
  if (kind === "lucky") {
    // which LoRA rolls: the primary, or with a donor loaded the dropdown's choice (Both rolls each on its own)
    // the sliders it can change (a block's primary and donor are two), shuffled, the Roll count of them
    const {pOk, dOk} = studioLuckyPool();
    const pool = Object.keys(S.st.blocks).flatMap((id) => [...(pOk(id) ? [[id, "ps"]] : []), ...(dOk(id) ? [[id, "ds"]] : [])]);
    for (let j = pool.length - 1; j > 0; j--) { const k = Math.floor(Math.random() * (j + 1)); [pool[j], pool[k]] = [pool[k], pool[j]]; }
    const want = $("stLuckyCount").value === "all" ? pool.length : Number($("stLuckyCount").value);
    for (const [id, key] of pool.slice(0, want)) {
      // a held band (lucky_zones: [first, last, low, high] by the block's place in the model's order), bounded by
      // Min / Max as the desktop does
      const i = order.indexOf(id), b = S.st.blocks[id];
      const zone = $("stLuckyHold").checked ? (L.lucky_zones || []).find((zn) => zn[0] <= i && i <= zn[1]) : null;
      let a = zone ? Math.max(zone[2], lo) : lo, z = zone ? Math.min(zone[3], hi) : hi;
      if (a > z) a = z = Math.min(Math.max(1, lo), hi);
      const roll = () => Math.round((a + Math.random() * (z - a)) * 20) / 20;
      b[key] = roll();
      rolled++;
    }
  }
  studioSync(); studioCommit(); studioSchedule();
  if (kind === "lucky") $("stStatus").textContent = `Lucky mix: ${rolled} slider(s) rolled, ${lo} to ${hi}` +
    `${S.st.locks.size + S.st.dlocks.size ? `, ${S.st.locks.size + S.st.dlocks.size} locked or frozen slider(s) left alone` : ""} (Undo for the mix before; Freeze tweaked to keep ` +
    "what you like and roll the rest; Save preset… to keep it).";
  stRemember();
}

function studioPresetSaveOpen() {
  const host = $("stPresetSaveRow");
  if (host.childNodes.length) { host.innerHTML = ""; return; }
  const name = el("input", {type: "text", class: "w-path", placeholder: "A name for these block settings"});
  const msg = el("span", {class: "field-hint"});
  const save = async (overwrite) => {
    const blocks = {};
    for (const [id, b] of Object.entries(S.st.blocks)) blocks[id] = {primary_enabled: b.pe, primary_strength: b.ps, donor_enabled: b.de, donor_strength: b.ds};
    try {
      const r = await api("/api/studio/preset_save", {family: S.st.layout.key, name: name.value, blocks, overwrite: !!overwrite});
      if (r.exists) { msg.replaceChildren(document.createTextNode("That name is taken. "), el("button", {class: "btn btn-sm btn-danger", text: "Replace it", onclick: () => save(true)})); return; }
      host.innerHTML = ""; studioPresets(); $("stStatus").textContent = "Preset saved.";
    } catch (e) { msg.textContent = e.message; }
  };
  host.append(el("div", {class: "field-row"}, name, el("button", {class: "btn btn-primary", text: "Save", onclick: () => save(false)}),
    el("button", {class: "btn", text: "Cancel", onclick: () => { host.innerHTML = ""; }}), msg));
  name.focus();
}

// ---- rendering ----
function studioState() {
  const blocks = {};
  for (const [id, b] of Object.entries(S.st.blocks)) blocks[id] = {primary_enabled: b.pe, primary_strength: b.ps, donor_enabled: b.de, donor_strength: b.ds};
  const num = (id, d) => { const v = Number($(id).value); return isFinite(v) ? v : d; };
  return {blocks, seed: seedOf("stSeed"), prompt: $("stPrompt").value.trim(), preview_width: Number($("stW").value) || 768,
    preview_height: Number($("stH").value) || 768, ref_image_path: $("stRefRow").style.display === "none" ? "" : $("stRef").value.trim(),
    ref_megapixels: num("stRefMp", 1), ref_strength: num("stRefStr", 1), primary_scale: num("stPScale", 1),
    donor_scale: num("stDScale", 1)};
}

function studioSchedule(live) {
  if (!S.st.ready) return;
  clearTimeout(S.st.timer);
  S.st.timer = setTimeout(studioRender, live ? ($("stTurbo").checked ? 150 : 400) : 100);
}

async function studioRender() {
  if (!$("stPrompt").value.trim()) { $("stStatus").textContent = "Enter a prompt (include the LoRA's trigger word) to see previews."; return; }
  try { await api("/api/studio/render", {state: studioState(), ...studioExtra(), ...studioMixExtra()}); studioPoll(); }
  catch (e) { $("stStatus").textContent = e.message; }
}

async function studioStart() {
  stRemember();
  const L = S.st.layout;
  try {
    const st = await api("/api/studio/status");
    const fast = (document.querySelector('input[name="stDit"]:checked') || {}).value !== "full";
    const key = [L.key, $("stPrimary").value.trim(), fast, L.kind === "video" ? $("stH3Model").value + $("stH3Base").value : ""].join("|");
    if (st.state === "ready" && S.st.loadedKey === key) {     // the same LoRA: the donor may change, then a render
      if ($("stDonor").value.trim() !== (st.donor || "")) await api("/api/studio/donor", {path: $("stDonor").value.trim()});
      studioRender();
      return;
    }
    S.st.ready = false;
    await api("/api/studio/load", {family: L.key, primary: $("stPrimary").value.trim(), donor: $("stDonor").value.trim(),
      fast, turbo: $("stTurbo").checked, negative: $("stNeg").value.trim(),
      h3_model: L.kind === "video" ? $("stH3Model").value : "", base: L.kind === "video" ? $("stH3Base").value : "",
      canvas: studioCanvas()});
    if (S.st.mix && S.st.mix.mode !== "off" || (S.st.dmix && S.st.dmix.on)) studioMixReset();
    S.st.loadedKey = key;
    S.st.pendingFirst = true;
    studioPoll();
  } catch (e) { $("stStatus").textContent = e.message; }
}

async function studioPoll() {
  clearTimeout(S.st.poll);
  let st;
  try { st = await api("/api/studio/status"); } catch (e) { return; }
  const wasReady = S.st.ready;
  S.st.ready = st.state === "ready";
  S.st.donor = !!st.donor;
  S.st.primaryBlocks = new Set(st.primary_blocks || []); S.st.donorBlocks = new Set(st.donor_blocks || []);
  // a donor loaded (or unloaded) after the model: its rows and buttons show (or go) now, not at the next load
  const actKey = [S.st.ready, st.donor || "", (st.primary_blocks || []).length, (st.donor_blocks || []).length].join("|");
  if (wasReady && S.st.ready && actKey !== S.st.actKey) studioActivity();
  S.st.actKey = actKey;
  // loaded with nothing to render yet: say what it waits for (a prompt), not that a preview is coming
  const waiting = st.state === "ready" && !st.rendering && !st.shown && !$("stPrompt").value.trim();
  $("stStatus").textContent = (waiting ? "Loaded. Enter a prompt (include the LoRA's trigger word), then press Update." :
    st.status || $("stStatus").textContent) +
    (st.status && /^(Loaded|Ready|Donor loaded)/.test(st.status) && studioEqNote(st) ? " " + studioEqNote(st) : "");
  $("stBadge").textContent = {loading: "Loading", ready: st.rendering ? "Rendering" : "Loaded", idle: ""}[st.state] || "";
  $("stBadge").dataset.state = st.state === "loading" || st.rendering ? "running" : st.state === "ready" ? "completed" : "idle";
  const p = st.progress;
  $("stBar").style.width = p && p[1] ? `${Math.round(100 * p[0] / p[1])}%` : st.rendering || st.state === "loading" ? "100%" : "0";
  $("stBar").parentElement.classList.toggle("indeterminate", !(p && p[1]) && (st.rendering || st.state === "loading"));
  $("stStart").textContent = S.st.ready ? "Update" : "Start";
  if (S.st.ready && !wasReady) {
    studioActivity();
    if (!$("stOut").value) S.st.autoOut = "";
    if (S.st.pendingFirst) { S.st.pendingFirst = false; studioCommit(); studioRender(); }
    if (S.st.pendingMix) studioSetupMix();
  }
  if (S.st.ready && st.primary && $("stOut").value === (S.st.autoOut ?? null)) {
    // the save name the desktop suggests: <primary>_with_<donor> while donor blocks are on, else <primary>_repaired
    // (only while the box still holds its own suggestion)
    const stem = (p) => p.replace(/\.safetensors$/i, "");
    const donorOn = st.donor && Object.values(S.st.blocks).some((b) => b.de);
    S.st.autoOut = $("stOut").value = stem(st.primary) + (donorOn ? "_with_" + stem(st.donor.split(/[\\/]/).pop()) : "_repaired") + ".safetensors";
  }
  if (st.early && st.early !== S.st.earlyShown) {        // H3 Show early: the unfinished look, until the clip lands
    S.st.earlyShown = st.early;
    $("stTweakVid").style.display = "none"; $("stTweak").style.display = "";
    $("stTweak").src = `/api/studio/image?which=early&gen=${st.early}&t=${encodeURIComponent(TOKEN)}`;
  }
  if (st.shown && st.shown !== S.st.shown) {
    S.st.shown = st.shown;
    const q = (w) => `/api/studio/image?which=${w}&gen=${st.shown}&t=${encodeURIComponent(TOKEN)}`;
    $("stBase").src = q("baseline"); $("stTweak").src = q("tweaked");
    // a clip plays in its pane (the picture its poster); a song gets its player under the waveform
    const kind = S.st.layout && S.st.layout.kind;
    for (const [w, img, vid, aud] of [["baseline", "stBase", "stBaseVid", "stBaseAud"], ["tweaked", "stTweak", "stTweakVid", "stTweakAud"]]) {
      const has = st.media && st.media[w];
      $(vid).style.display = kind === "video" && has ? "" : "none";
      $(aud).style.display = "none";
      $(img).style.display = kind === "video" && has ? "none" : "";
      if (kind === "video" && has) { $(vid).poster = q(w); $(vid).src = q(w + "_media"); }
      if (kind === "audio" && has) songLoad(w, q(w + "_media"), st.shown, st.act && st.act[w] ? q(w + "_act") : "");
    }
    studioMetrics();
    studioDownloads(st, q);
    const ps = Number($("stPScale").value) || 1;
    $("stBaseCap").textContent = (ps === 1 ? "Baseline (LoRA at default 1.0)" : `Baseline (LoRA at ${ps})`) +
      (st.donor ? " - the donor is not in the baseline" : "");
    $("stTweakCap").textContent = "Tweaked (current sliders)" + (st.donor ? " - includes the donor" : "");
  }
  studioNoLoraPane(st);
  studioAceNotes(st.notes);
  studioMixNotes(st.notes);
  const pr = st.profile;
  $("stProfile").style.display = pr ? "" : "none";
  if (pr) {
    $("stProfileLine").textContent = `${pr.lora_name} · profiled ${pr.created} · most active blocks:`;
    const col = {"Style+Comp": "blue", "Style/ID": "teal", Identity: "green", "ID/Detail": "olive", Details: "orange",
      style_composition: "blue", style_ident_overlap: "teal", identity: "green", ident_details_overlap: "olive",
      details: "orange"};
    $("stPills").replaceChildren(...pr.blocks.map((b) => el("span", {class: "st-pill",
      style: `background:var(${col[b.category] ? "--data-" + col[b.category] : "--text-secondary"})`,
      text: `${b.name}  ${Number(b.pct).toFixed(1)}%`})));
    $("stReport").style.display = pr.report ? "" : "none";
    $("stReport").href = `/api/studio/report?t=${encodeURIComponent(TOKEN)}`;
  }
  if (st.saved && st.saved.out !== S.st.savedShown) { S.st.savedShown = st.saved.out; studioSavedSay(st.saved); }
  const active = st.state === "loading" || st.rendering || st.pending;
  if ($("tab-studio").classList.contains("active")) S.st.poll = setTimeout(studioPoll, active ? 400 : 2000);
}

// Download original / Download tweaked: the pair as rendered (a still's picture, a clip's mp4); songs have their
// own Save in the song player
function studioDownloads(st, q) {
  const kind = S.st.layout && S.st.layout.kind, row = $("stDownloads");
  const on = kind !== "audio" && !!st.shown;
  row.style.display = on ? "" : "none";
  if (!on) return;
  const lora = ($("stPrimary").value.trim().split(/[\\/]/).pop() || "render").replace(/\.safetensors$/i, "");
  const seed = ($("stSeed").value || "").trim();
  for (const [w, id, word] of [["baseline", "stDlBase", "original"], ["tweaked", "stDlTweak", "tweaked"]]) {
    const video = kind === "video" && st.media && st.media[w];
    const name = `${lora}_${word}${seed ? "_seed" + seed : ""}${video ? ".mp4" : ".png"}`;
    $(id).href = q(video ? w + "_media" : w) + "&dl=" + encodeURIComponent(name);
    $(id).setAttribute("download", name);
  }
}

function studioNoLoraPane(st) {
  // the base model's own clip / song, left of the pair (the desktop's player has it there): H3 with the tick, ACE always
  const kind = S.st.layout && S.st.layout.kind;
  const on = kind === "audio";                        // No LoRA: ACE-Step's song only (H3's clip was too slow)
  $("stNolPane").style.display = on && kind === "video" ? "" : "none";     // a song's No LoRA is heard in the player
  $("stPreview").classList.toggle("three", on && kind === "video");
  if (kind === "audio") $("stSongNolora").textContent = st.nolora_busy ? "▶ No LoRA (rendering…)" : "▶ No LoRA";
  if (!on) return;
  $("stNolCap").textContent = st.nolora_busy ? "No LoRA (rendering…)" : "No LoRA (base model, same seed)";
  if (st.nolora_gen && st.nolora_gen !== S.st.nolShown && st.media && st.media.nolora != null) {
    S.st.nolShown = st.nolora_gen;
    const q = (w) => `/api/studio/image?which=${w}&gen=${st.nolora_gen}&t=${encodeURIComponent(TOKEN)}`;
    const media = st.media.nolora;
    $("stNol").src = q("nolora");
    $("stNolVid").style.display = kind === "video" && media ? "" : "none";
    $("stNolAud").style.display = "none";
    $("stNol").style.display = kind === "video" && media ? "none" : "";
    if (kind === "video" && media) { $("stNolVid").poster = q("nolora"); $("stNolVid").src = q("nolora_media"); }
    if (kind === "audio" && media) songLoad("nolora", q("nolora_media"), st.nolora_gen);
  }
}

function studioSavedSay(sv) {
  const s = sv.summary || {};
  const lines = [`Saved: ${sv.out}`];
  if (s.keys_in != null) lines.push(`Keys: ${s.keys_in} → ${s.keys_out}`);
  for (const [k, t] of [["dropped_blocks", "Dropped blocks"], ["rescaled_blocks", "Rescaled blocks"], ["blended_blocks", "Donor-blended blocks"]])
    if (s[k] && s[k].length) lines.push(`${t} (${s[k].length}): ${s[k].join(", ")}`);
  if (s.use_at != null) lines.push(`Use it at strength ${s.use_at}.`);
  $("stSaveMsg").replaceChildren(el("pre", {class: "field-hint st-saved", text: lines.join("\n")}));
}

async function studioSave(overwrite) {
  const st = studioState();
  const includeDonor = S.st.donor && Object.values(S.st.blocks).some((b) => b.de);
  try {
    const r = await api("/api/studio/save", {out: $("stOut").value.trim(), state: st, include_donor: includeDonor, overwrite: !!overwrite});
    if (r.exists) {
      $("stSaveMsg").replaceChildren(el("div", {class: "field-row"}, el("span", {class: "field-hint", text: "That file is there already."}),
        el("button", {class: "btn btn-sm btn-danger", text: "Replace it", onclick: () => studioSave(true)})));
      return;
    }
    $("stSaveMsg").replaceChildren(el("div", {class: "field-hint", text: "Saving…"}));
    studioPoll();
  } catch (e) { $("stSaveMsg").replaceChildren(el("div", {class: "problems", text: e.message})); }
}

// ---- the preview kept in view: the Preview card's panes moved into a panel that floats (move it, resize it from the
// corner) or is pinned to the top of the tab (its bottom edge sets the height); renders, clips, songs and Play
// together keep working because it is the same panes, moved - and put back when it closes
const ST_DOCK_KEY = "fizgig_studio_dock";
const stDockSaved = () => { try { return JSON.parse(localStorage.getItem(ST_DOCK_KEY) || "{}"); } catch (e) { return {}; } };
const stDockSave = (d) => { try { localStorage.setItem(ST_DOCK_KEY, JSON.stringify({...stDockSaved(), ...d})); } catch (e) { /* private window */ } };

function studioDockClamp(dock) {                   // a floating panel stays on screen (a smaller window, a phone)
  const r = dock.getBoundingClientRect(), vw = window.innerWidth, vh = window.innerHeight;
  const w = Math.min(r.width, vw - 16), h = Math.min(r.height, vh - 16);
  dock.style.width = w + "px"; dock.style.height = dock.classList.contains("collapsed") ? "" : h + "px";
  dock.style.left = Math.max(8, Math.min(r.left, vw - w - 8)) + "px";
  dock.style.top = Math.max(8, Math.min(r.top, vh - 40)) + "px";
}

function studioDockMode(pinned) {
  const dock = $("stDock"), sv = stDockSaved();
  dock.classList.toggle("pinned", pinned); dock.classList.toggle("floating", !pinned);
  $("stDockPin").textContent = pinned ? "⧉ Float" : "📌 Pin to top";
  $("stDockPin").title = pinned ? "Float the panel: move it anywhere and resize it from its corner" :
    "Lock the panel to the top of the tab, full width; drag its bottom edge to set how much room it takes";
  if (pinned) {                                    // in the tab's flow, stuck to the top as the page scrolls under it
    $("tab-studio").prepend(dock);
    Object.assign(dock.style, {left: "", top: "", width: "", height: (sv.pinH || 320) + "px"});
  } else {                                         // inside the tab too: it goes whenever the tab does
    $("tab-studio").append(dock);
    const w = sv.w || Math.min(560, window.innerWidth - 32), h = sv.h || 340;
    Object.assign(dock.style, {width: w + "px", height: h + "px",
      left: (sv.x ?? window.innerWidth - w - 24) + "px", top: (sv.y ?? 90) + "px"});
    studioDockClamp(dock);
  }
  if (sv.collapsed) dock.style.height = "";
  stDockSave({pinned});
}

const studioDockAllowed = () => !(S.st.layout && S.st.layout.kind === "audio");     // never for a music model

function studioDock(open, remember = true) {
  let dock = $("stDock");
  if (!open) {
    if (dock) {
      $("stDockHome").after($("stPreview"));
      dock.remove();
    }
    $("stDockHome").style.display = "none"; $("stDockOpen").style.display = studioDockAllowed() ? "" : "none";
    if (remember) stDockSave({open: false});
    return;
  }
  if (!studioDockAllowed()) return;
  if (!dock) {
    const btn = (id, text, title) => el("button", {class: "btn btn-sm", id, text, title});
    dock = el("div", {class: "st-dock", id: "stDock"},
      el("div", {class: "st-dock-bar", id: "stDockBar"}, el("span", {class: "st-dock-title", text: "Preview"}),
        btn("stDockPlay", "▶ Play together", "Play the clips or songs together"),
        btn("stDockPin", "", ""), btn("stDockFold", "▁", "Collapse to the title bar / open it again"),
        btn("stDockClose", "✕", "Close - the preview goes back to its card")),
      el("div", {class: "st-dock-body", id: "stDockBody"}),
      el("div", {class: "st-dock-grip", id: "stDockGrip", title: "Drag to set how much room the panel takes"}));
    $("tab-studio").append(dock);                  // in the page first: its buttons are found by id below
    $("stDockClose").addEventListener("click", () => studioDock(false));
    $("stDockPin").addEventListener("click", () => studioDockMode(!dock.classList.contains("pinned")));
    $("stDockPlay").addEventListener("click", () => $("stPlayBothBtn").click());
    $("stDockFold").addEventListener("click", () => {
      const c = !dock.classList.contains("collapsed");
      dock.classList.toggle("collapsed", c); stDockSave({collapsed: c});
      const sv = stDockSaved();
      dock.style.height = c ? "" : (dock.classList.contains("pinned") ? (sv.pinH || 320) : (sv.h || 340)) + "px";
    });
    // floating: drag by the title bar
    $("stDockBar").addEventListener("pointerdown", (e) => {
      if (!dock.classList.contains("floating") || e.target.closest("button")) return;
      const r = dock.getBoundingClientRect(), dx = e.clientX - r.left, dy = e.clientY - r.top;
      $("stDockBar").setPointerCapture(e.pointerId);
      const move = (m) => { dock.style.left = (m.clientX - dx) + "px"; dock.style.top = (m.clientY - dy) + "px"; };
      const up = () => { $("stDockBar").removeEventListener("pointermove", move); studioDockClamp(dock);
        const q = dock.getBoundingClientRect(); stDockSave({x: Math.round(q.left), y: Math.round(q.top)}); };
      $("stDockBar").addEventListener("pointermove", move);
      $("stDockBar").addEventListener("pointerup", up, {once: true});
    });
    // pinned: drag the bottom edge for the height
    $("stDockGrip").addEventListener("pointerdown", (e) => {
      if (!dock.classList.contains("pinned")) return;
      const y0 = e.clientY, h0 = dock.getBoundingClientRect().height;
      $("stDockGrip").setPointerCapture(e.pointerId);
      const move = (m) => { dock.style.height = Math.max(140, Math.min(window.innerHeight * 0.85, h0 + m.clientY - y0)) + "px"; };
      const up = () => { $("stDockGrip").removeEventListener("pointermove", move);
        stDockSave({pinH: Math.round(dock.getBoundingClientRect().height)}); };
      $("stDockGrip").addEventListener("pointermove", move);
      $("stDockGrip").addEventListener("pointerup", up, {once: true});
    });
    // while the panel is being resized its clips step aside (scaling playing video every frame is the slow part),
    // back a moment after the size stops changing - the pinned edge drag and the floating corner alike
    let sizing = null;
    new ResizeObserver(() => {
      if (!dock.isConnected) return;
      dock.classList.add("st-resizing");
      clearTimeout(sizing);
      sizing = setTimeout(() => dock.classList.remove("st-resizing"), 200);
    }).observe(dock);
    // floating: the corner resize is the browser's own; its size is remembered
    new ResizeObserver(debounce(() => {
      if (dock.isConnected && dock.classList.contains("floating") && !dock.classList.contains("collapsed")) {
        const q = dock.getBoundingClientRect(); stDockSave({w: Math.round(q.width), h: Math.round(q.height)});
      }
    }, 300)).observe(dock);
  }
  $("stDockBody").append($("stPreview"));
  $("stDockHome").style.display = ""; $("stDockOpen").style.display = "none";
  const sv = stDockSaved();
  dock.classList.toggle("collapsed", !!sv.collapsed);
  studioDockMode(!!sv.pinned);
  studioDockPlay();
  stDockSave({open: true});
}

function studioDockPlay() {                        // the panel's Play together: when the card has one (clips, songs)
  if ($("stDockPlay")) $("stDockPlay").style.display = $("stPlayBoth").style.display === "none" ? "none" : "";
}

function wireStudio() {
  $("stDockOpen").addEventListener("click", () => studioDock(true));
  $("stDockBack").addEventListener("click", () => studioDock(false));
  new MutationObserver(studioDockPlay).observe($("stPlayBoth"), {attributes: true, attributeFilter: ["style"]});
  window.addEventListener("resize", debounce(() => { const d = $("stDock"); if (d && d.classList.contains("floating")) studioDockClamp(d); }, 150));
  $("stFamily").addEventListener("change", () => { studioFamily(); stRemember(); });
  // H3: a length / size / steps / sound change renders; Model and Base take effect at the next Start
  for (const id of ["stH3Len", "stH3W", "stH3H", "stH3Steps", "stH3Turbo", "stH3Sound", "stH3Boost"])
    $(id).addEventListener("change", () => { studioH3Remember(); studioSchedule(); });
  $("stH3Early").addEventListener("change", studioH3Remember);
  $("stH3Model").addEventListener("change", () => { studioH3Remember(); studioSlots(); $("stStart").textContent = "Update"; });
  $("stH3Base").addEventListener("change", () => { studioH3Remember(); $("stStart").textContent = "Update"; });
  playTogether($("stPlayBothBtn"), () => {
    const v = S.st.layout?.kind === "video", ids = v ? ["stBaseVid", "stTweakVid"] : ["stBaseAud", "stTweakAud"];
    if ($("stNolPane").style.display !== "none") ids.unshift(v ? "stNolVid" : "stNolAud");
    return ids;
  });
  $("stStart").addEventListener("click", studioStart);
  wireStudioAce();
  S.st.eq = {primary: false, donor: false}; S.st.eqSaved = {};
  $("stEqP").addEventListener("click", () => studioEq("primary", !S.st.eq.primary));
  wireStudioMix();
  wireStudioMetrics();
  wireStudioSetups();
  $("stEqD").addEventListener("click", () => studioEq("donor", !S.st.eq.donor));
  $("stUnload").addEventListener("click", async () => { await api("/api/studio/reset", {}); S.st.ready = false; S.st.loadedKey = ""; studioPoll(); });
  $("stPrimaryPick").addEventListener("click", () => pick($("stPrimary").value || loraStart(), false, (p) => { $("stPrimary").value = p; stRemember(); }));
  $("stDonorPick").addEventListener("click", () => pick($("stDonor").value || $("stPrimary").value, false, (p) => { $("stDonor").value = p; stRemember();
    studioDmixOff();
    if (S.st.ready) api("/api/studio/donor", {path: p}).then(studioPoll).catch((e) => { $("stStatus").textContent = e.message; }); }));
  $("stDonorUnload").addEventListener("click", () => { $("stDonor").value = ""; stRemember(); studioDmixOff();
    if (S.st.ready) api("/api/studio/donor", {path: ""}).then(() => { studioPoll(); studioSchedule(); }); });
  $("stRefPick").addEventListener("click", () => pick($("stRef").value || refStart(), false, (p) => { $("stRef").value = p; stRemember(); studioSchedule(); }));
  $("stRefClear").addEventListener("click", () => { $("stRef").value = ""; stRemember(); studioSchedule(); });
  $("stSeedNew").addEventListener("click", () => { $("stSeed").value = 1 + Math.floor(Math.random() * 99999); stRemember(); studioSchedule(); });
  for (const id of ["stSeed", "stRes", "stW", "stH", "stRefMp", "stRefStr", "stPScale", "stDScale"])
    $(id).addEventListener("change", () => { stRemember(); studioSchedule(); });
  for (const id of ["stPrompt", "stPrimary", "stDonor", "stLuckyLo", "stLuckyHi"]) $(id).addEventListener("input", stRemember);
  $("stLuckyWho").addEventListener("change", () => { stRemember(); studioFreezeState(); });
  $("stLuckyCount").addEventListener("change", stRemember);
  $("stFreeze").addEventListener("click", () => studioFreeze(true));
  $("stUnfreeze").addEventListener("click", () => studioFreeze(false));
  $("stNeg").addEventListener("change", () => { stRemember();
    if (S.st.ready) api("/api/studio/settings", {family: S.st.layout.key, negative: $("stNeg").value.trim()}).then(() => studioSchedule()); });
  $("stTurbo").addEventListener("change", () => { stRemember(); if (S.st.ready) api("/api/studio/settings", {family: S.st.layout.key, turbo: $("stTurbo").checked}); });
  document.querySelectorAll('input[name="stDit"]').forEach((r) => r.addEventListener("change", studioNegRow));
  document.querySelectorAll('input[name="stTarget"]').forEach((r) => r.addEventListener("change", studioMastersFollow));
  document.querySelectorAll("[data-bulk]").forEach((b) => b.addEventListener("click", () => studioBulk(b.dataset.bulk)));
  $("stPreset").addEventListener("change", () => { if ($("stPreset").value) studioPresetApply($("stPreset").value); });
  $("stPresetSave").addEventListener("click", studioPresetSaveOpen);
  $("stUndo").addEventListener("click", () => studioUndo(false));
  $("stRedo").addEventListener("click", () => studioUndo(true));
  $("stSave").addEventListener("click", () => studioSave(false));
  document.addEventListener("keydown", (e) => {
    if (!$("tab-studio").classList.contains("active") || /INPUT|TEXTAREA|SELECT/.test((document.activeElement || {}).tagName || "")) return;
    if ((e.ctrlKey || e.metaKey) && e.key.toLowerCase() === "z") { e.preventDefault(); studioUndo(e.shiftKey); }
    else if ((e.ctrlKey || e.metaKey) && e.key.toLowerCase() === "y") { e.preventDefault(); studioUndo(true); }
  });
}

// ---- Captions tab ----------------------------------------------------------------------------------------
// The desktop Captions tab's tools, run on the Fizgig machine (fizgig/web/captions.py). Settings are remembered.
let CAP_PAGE = 12;                         // Per page (the desktop's sizes), remembered with the tab's settings
S.cap = {page: 0, items: [], opts: null, lastState: ""};

// the folder the Captions tab works on (its tools take the folder's largest detail crops in too, server side)
function capFolder() {
  return S.imageFolder;
}
// the folder one listed file is in: the training folder, or its largest detail crop folder
const capItemFolder = (it) => it.path.slice(0, it.path.length - it.name.length - 1);

function capSettings() {
  return {trigger: $("capTrigger").value, model: $("capModel").value, task: $("capTask").value,
    tokens: $("capTokens").value, overwrite: $("capOverwrite").checked, songWords: $("capSongWords").value,
    skipCjk: $("capSkipCjk").checked, perPage: $("capPerPage").value, concept: $("capConcept").value.trim()};
}
const saveCapSoon = debounce(() => { S.capSaved = capSettings(); api("/api/state", {captions: S.capSaved}).catch(() => {}); }, 600);
const qwenOn = () => !!S.cap.opts && S.cap.opts.qwen_available && $("capModel").value === S.cap.opts.qwen_model;
const qwenTask = () => (S.cap.opts.qwen_tasks || []).find((t) => t.label === $("capTask").value);

async function loadCaptionOptions() {
  const o = S.cap.opts = await api("/api/captions/options?family=" + encodeURIComponent(S.family));
  const saved = S.capSaved || {};
  const model = $("capModel");
  model.innerHTML = "";
  for (const m of o.models) model.append(el("option", {value: m, text: m}));
  model.value = o.models.includes(saved.model) ? saved.model : o.default_model;
  $("capModelHint").textContent = o.secondary ?
    `Qwen3-VL captions with your secondary captioning model (${o.secondary}, set in Preferences), following an ` +
    "instruction you can edit. Empty that row in Preferences to caption with the Krea 2 text encoder again. " +
    "Florence-2 downloads itself the first time." : o.qwen_available ?
    "Qwen3-VL (the Krea 2 text encoder from Preferences) follows an instruction you can edit and writes the best " +
    "training captions. Florence-2 downloads itself the first time." :
    "Set the Krea 2 text encoder in Preferences to caption with Qwen3-VL, which writes better training captions. " +
    "Florence-2 downloads itself the first time.";
  if (S.trigger !== undefined) $("capTrigger").value = S.trigger;      // the training folder's word
  if (saved.overwrite !== undefined) $("capOverwrite").checked = saved.overwrite;
  if (saved.skipCjk !== undefined) $("capSkipCjk").checked = saved.skipCjk;
  if (saved.perPage) { $("capPerPage").value = saved.perPage; CAP_PAGE = Number(saved.perPage) || 12; }
  fillTasks(saved.task);
  if (saved.tokens) $("capTokens").value = saved.tokens;
  if (saved.concept !== undefined) $("capConcept").value = saved.concept;
  capConceptShow();
  $("capSongCard").style.display = o.song_words ? "" : "none";
  // a model that trains on no pictures or clips (ACE-Step): the AI card has nothing to caption, the song card does
  const media = (S.fam && S.fam.media) || ["photo"];
  const pictures = media.includes("photo") || media.includes("clip");
  document.querySelector("#capAll").closest(".section-card").style.display = pictures ? "" : "none";
  for (const id of ["capModel", "capTask", "capTokens"]) $(id).closest(".field-row").style.display = pictures ? "" : "none";
  $("capModelHint").style.display = pictures ? "" : "none";
  if (!pictures) $("capInstrBox").style.display = "none";
  $("capTaskHint").style.display = pictures && qwenOn() ? "" : "none";       // fillTasks ran while the row was hidden
  if (o.song_words) { $("capSongWords").placeholder = "e.g. " + o.song_words; $("capSongWords").value = saved.songWords || ""; }
}

function fillTasks(want) {
  const o = S.cap.opts, task = $("capTask");
  task.innerHTML = "";
  const labels = qwenOn() ? o.qwen_tasks.map((t) => t.label) : o.florence_tasks;
  for (const l of labels) task.append(el("option", {value: l, text: l}));
  task.value = labels.includes(want) ? want : (qwenOn() ? o.default_qwen_task : "<DETAILED_CAPTION>");
  $("capEditInstr").style.display = qwenOn() ? "" : "none";
  $("capTaskHint").style.display = qwenOn() && $("capTask").closest(".field-row").style.display !== "none" ? "" : "none";
  $("capInstrBox").style.display = "none";
}

// the Concept task's box: shown with it (or with any instruction that uses {concept})
function capConceptShow() {
  const t = qwenOn() && qwenTask();
  const on = !!t && t.instruction.includes("{concept}") && $("capTask").closest(".field-row").style.display !== "none";
  for (const id of ["capConceptRow", "capConceptHint"]) $(id).style.display = on ? "" : "none";
}

function onCapTask() {
  capConceptShow();
  const t = qwenOn() && qwenTask();
  if (t) $("capTokens").value = t.tokens;            // each Qwen task suggests its own length
  if ($("capInstrBox").style.display !== "none") openInstr();
  saveCapSoon();
}

function openInstr() {
  const t = qwenTask();
  if (!t) return;
  $("capInstrBox").style.display = "";
  $("capInstr").value = t.instruction;
  $("capInstrHint").textContent = `What Qwen3-VL is told for "${t.label}"` + (t.edited ? " (edited)" : " (as shipped)") +
    ". Saved here. Auto-recaption uses it too when the Training tab's First try / Second try picks this task " +
    "(by default Training caption first, then Exhaustive detail).";
  $("capInstrStatus").textContent = "";
}

async function saveInstr(text) {
  const t = qwenTask();
  await api("/api/captions/instruction", {key: t.key, text});
  const keep = $("capTask").value;
  await loadCaptionOptions();
  $("capTask").value = keep;
  openInstr();
  $("capInstrStatus").textContent = "Saved.";
}

const capExt = () => {                    // Training > Other Options > Caption Extension (.txt unless set)
  let e = String(S.values.DATASET_CAPTION_EXT || ".txt").trim();
  if (e && !e.startsWith(".")) e = "." + e;
  return /^\.[A-Za-z0-9_]{1,12}$/.test(e) ? e : ".txt";
};

function capJobBody(extra = {}) {
  const t = qwenOn() && qwenTask();
  return {family: S.family, folder: capFolder(), model: $("capModel").value, task: $("capTask").value, ext: capExt(),
    instruction: t ? t.instruction.split("{concept}").join($("capConcept").value.trim() || "the same concept") : "",
    max_tokens: Number($("capTokens").value) || 120,
    trigger: $("capTrigger").value, overwrite: $("capOverwrite").checked, ...extra};
}

async function capRun(path, body) {
  try {
    const r = await api(path, body);
    if (r.error) $("capStatus").textContent = r.error;
    pollCaptions();
    return r;
  } catch (e) { $("capStatus").textContent = e.message; }
}

// while captioning runs: the captions written so far fill in every few seconds - never over one you're editing
async function captionsLive() {
  if (!S.imageFolder || !document.querySelector("#tab-captions.active")) return;
  const r = await api(`/api/captions?folder=${encodeURIComponent(capFolder())}&ext=${encodeURIComponent(capExt())}&family=${encodeURIComponent(S.family || "")}`)
    .catch(() => null);
  if (!r || !r.items) return;
  const now = new Map(r.items.map((i) => [i.path, i]));
  for (const it of S.cap.items) {
    const n = now.get(it.path);
    if (!n || (n.caption ?? "") === (it.caption ?? "")) continue;
    const ta = it._ta, untouched = !ta || (document.activeElement !== ta && ta.value === (it.caption ?? ""));
    if (!untouched) continue;
    const pic = S.lb.list[S.lb.at], big = $("lightboxCapText");
    if (pic && pic.cap && pic.cap.ta === ta && $("lightbox").style.display !== "none" &&
        document.activeElement !== big && big.value === (it.caption ?? "")) {
      big.value = n.caption ?? "";
      $("lightboxCapNote").textContent = "Re-captioned";
    }
    it.caption = n.caption;
    if (ta) ta.value = n.caption ?? "";
  }
}

async function pollCaptions() {
  clearTimeout(S.cap.timer);
  let st;
  try { st = await api("/api/captions/status"); } catch (e) { return; }
  const busy = ["loading", "running", "translating"].includes(st.state);
  const pct = st.total ? (100 * st.done) / st.total : 0;
  $("capBar").style.width = (busy ? pct : 0).toFixed(1) + "%";
  $("capStatus").textContent = {
    idle: "No model loaded.", loaded: `${st.model} is loaded, ready.`,
    loading: `Loading ${st.model}… (the first time it downloads)`,
    running: `Captioning ${st.done} of ${st.total}` + (st.fail ? ` · ${st.fail} failed` : ""),
    translating: st.total ? `Translating ${st.done} of ${st.total}` : "Loading the translator…",
  }[st.state] || st.state;
  $("capAll").disabled = busy; $("capStatic").disabled = busy; $("capTranslate").disabled = busy;
  $("capStop").disabled = !busy; $("capUnload").disabled = busy || !st.loaded;
  const log = $("capLog");
  const atEnd = log.scrollTop + log.clientHeight >= log.scrollHeight - 30;
  log.textContent = st.log.join("\n");
  if (atEnd) log.scrollTop = log.scrollHeight;
  if (S.cap.lastState && ["loading", "running", "translating"].includes(S.cap.lastState) && !busy) loadCaptions();   // done: the new captions
  else if (busy && Date.now() - (S.cap.liveAt || 0) > 5000) { S.cap.liveAt = Date.now(); captionsLive(); }
  S.cap.lastState = st.state;
  if (busy || document.querySelector("#tab-captions.active")) S.cap.timer = setTimeout(pollCaptions, busy ? 1500 : 5000);
}

// the caption models on offer follow Preferences (Qwen3-VL needs the Krea 2 text encoder): after a download or a
// path change the Captions tab asks again - at once if it is open, else when it next opens
function captionModelsChanged() {
  S.cap.opts = null;
  if (document.querySelector("#tab-captions.active")) loadCaptions();
}

async function loadCaptions() {
  if (!S.cap.opts || S.cap.optsFamily !== S.family) {        // a model change can bring the song card
    await loadCaptionOptions().catch((e) => {        // said, and asked again next time the tab opens
      S.cap.opts = null; $("capModelHint").textContent = `Couldn't get the caption models: ${e.message}`; });
    S.cap.optsFamily = S.family;
  }
  pollCaptions();
  $("captionExt").value = capExt();         // the files this tab writes and reads are the ones training reads
  const zipUrl = (caps) => `/api/datasets/zip?folder=${encodeURIComponent(S.imageFolder || "")}&captions=${caps ? 1 : 0}` +
    `&ext=${encodeURIComponent(capExt())}&t=${encodeURIComponent(TOKEN)}`;
  $("dsZip").href = zipUrl(false); $("dsZipCaps").href = zipUrl(true);
  for (const id of ["dsZip", "dsZipCaps"]) $(id).style.display = S.imageFolder ? "" : "none";
  $("capFolder").textContent = (S.imageFolder ? `Folder: ${S.imageFolder}  (set on the Start tab)` :
    "Choose the training folder on the Start tab first.") + subfoldersHint();
  if (!S.imageFolder) { $("captionsSummary").textContent = "Choose the training folder on the Start tab."; return; }
  const ext = $("captionExt").value.trim() || ".txt";
  const r = await api(`/api/captions?folder=${encodeURIComponent(capFolder())}` +
    `&ext=${encodeURIComponent(ext)}&family=${encodeURIComponent(S.family || "")}`);
  S.cap.items = r.items;                      // the pictures, then the largest detail crops (it.crop: their folder)
  renderCaptions();
}

function renderCaptions() {
  const host = $("captionsGrid");
  host.innerHTML = "";
  const ext = $("captionExt").value.trim() || ".txt";
  const q = $("capSearch").value.trim().toLowerCase();
  const all = S.cap.items;
  let items = q ? all.filter((i) => i.name.toLowerCase().includes(q) || (i.caption || "").toLowerCase().includes(q)) : all;
  const sort = $("capSort").value, px = (i) => (i.size ? i.size[0] * i.size[1] : null);
  if (sort !== "name") {                       // by pixel count; songs (no size) go last either way
    items = [...items].sort((a, b) => (px(a) == null) - (px(b) == null) ||
      (sort === "small" ? px(a) - px(b) : px(b) - px(a)) || a.name.localeCompare(b.name));
  }
  const ts = S.cap.tsort;                       // the table's column sort (its header), over the Sort box
  if ($("capView").value === "table" && ts) {
    const v = CAP_COLS.find((c) => c.key === ts.key).sort;
    items = [...items].sort((a, b) => ((v(a) > v(b)) - (v(a) < v(b))) * ts.dir || a.name.localeCompare(b.name));
  }
  const pages = Math.max(1, Math.ceil(items.length / CAP_PAGE));
  S.cap.page = Math.min(S.cap.page, pages - 1);
  const missing = all.filter((i) => i.caption === null).length;
  $("captionsSummary").textContent = `${all.length} files${missing ? `, ${missing} without a caption` : ""}` +
    (q ? ` · ${items.length} match` : "") + (pages > 1 ? ` · page ${S.cap.page + 1} of ${pages}` : "");
  $("capPrev").disabled = S.cap.page === 0; $("capNext").disabled = S.cap.page >= pages - 1;
  host.classList.toggle("captions-as-table", $("capView").value === "table");
  if ($("capView").value === "table") { host.append(captionsTable(items.slice(S.cap.page * CAP_PAGE, (S.cap.page + 1) * CAP_PAGE), ext)); return; }
  for (const it of items.slice(S.cap.page * CAP_PAGE, (S.cap.page + 1) * CAP_PAGE)) {
    const src = `/api/media?path=${encodeURIComponent(it.path)}&t=${encodeURIComponent(TOKEN)}`;
    const ta = el("textarea", {rows: 4});
    ta.value = it.caption ?? "";
    it._ta = ta;                                 // for the live refresh while captioning runs
    const status = el("span", {class: "field-hint"});
    const saveCap = async () => {
      await api("/api/captions/save", {folder: capItemFolder(it), name: it.name, text: ta.value, ext});
      it.caption = ta.value;
      status.textContent = it.crop ? "Saved, at every crop size" : "Saved";
    };
    const save = el("button", {class: "btn btn-sm", text: "Save", onclick: saveCap});
    const kind = it.name.split(".").pop().toLowerCase();
    const isAudio = ["wav", "mp3", "flac", "m4a"].includes(kind);
    const trainsPictures = !S.fam || !S.fam.media || S.fam.media.includes("photo") || kind === "mp4";
    const recapRun = async (say = (t) => { status.textContent = t; }) => {
      say("Re-captioning…");
      const r = await capRun("/api/captions/ai", capJobBody({only: [it.path], overwrite: true}));
      if (r && r.error) say(r.error);
    };
    const recap = isAudio || !trainsPictures ? null : el("button", {class: "btn btn-sm", text: "Re-caption",
      title: "Write this file's caption again with the model and task above", onclick: () => recapRun()});
    // a clip with its sound (not Gizmo's _mute): Whisper its speech onto the caption, "... saying "words""
    const speechRun = kind === "mp4" && !/_mute$/i.test(it.name.replace(/\.[^.]+$/, "")) ? async (say = (t) => { status.textContent = t; }) => {
      say("Transcribing… (the first time downloads Whisper, ~300 MB)");
      try {
        const r = await api("/api/captions/transcribe", {family: S.family, folder: capItemFolder(it), name: it.name});
        if (!r.text) { say("Whisper heard no words in this clip."); return; }
        const cap = ta.value.trim().replace(/\.$/, "");
        ta.value = (cap ? cap + " " : "") + `saying "${r.text}"`;
        await saveCap();
        say("Transcript appended and saved - edit it freely.");
        return ta.value;
      } catch (e) { say(e.message); }
    } : null;
    const speech = speechRun ? el("button", {class: "btn btn-sm", text: "🎤 Append speech",
      title: "Whisper this clip's speech and add it to the end of the caption: … saying \"the words\"", onclick: () => speechRun()}) : null;
    const media = mediaFor(it);
    const card = el("div", {class: "caption-card"});
    // the big view edits, saves, re-captions and removes the same caption
    card._cap = {ta, it, save: saveCap, recap: recap ? recapRun : null, speech: speechRun,
      remove: it.crop ? null : () => capRemove(it, card, status)};
    const remove = it.crop ? null : el("button", {class: "btn btn-sm", text: "Remove",
      title: "Take it out of the training set: it and its caption move into the folder's removed subfolder (nothing is deleted).",
      onclick: () => capRemove(it, card, status)});
    const size = it.size ? `${it.size[0]}×${it.size[1]} · ${(it.size[0] * it.size[1] / 1e6).toFixed(1)} MP` : "";
    // masked training: the picture's mask (photo-masklabel.png), shown in its place on a click
    const pic = it.mask ? media.querySelector("img") || (media.tagName === "IMG" ? media : null) : null;
    const maskBtn = pic ? el("button", {class: "btn btn-sm", text: "Show mask",
      title: `${it.mask}: white trains fully, black only as much as the Training tab's Masked training card says.`,
      onclick: () => {
        const on = maskBtn.textContent === "Show mask";
        if (on && !pic.dataset.picture) pic.dataset.picture = pic.src;
        pic.src = on ? `/api/media?path=${encodeURIComponent(it.path.slice(0, it.path.length - it.name.length) + it.mask)}&t=${encodeURIComponent(TOKEN)}` : pic.dataset.picture;
        maskBtn.textContent = on ? "Show picture" : "Show mask";
      }}) : null;
    card.append(media, el("div", {class: "caption-card-body"},
      el("div", {class: "caption-card-title", text: mediaLabel(it) + (it.crop ? " · detail crop" : ""),
        title: it.crop ? "This picture's detail crop (Image Prep's Detail crops), the largest size: its caption is " +
          "copied to the crop at every other size" : ""}),
      size ? el("div", {class: "field-hint caption-card-size", text: size + (it.mask ? " · has a mask" : "")}) : null, ta,
      el("div", {class: "field-row"}, save, recap, speech, maskBtn, remove, status)));
    host.append(card);
  }
}

// the table view: one row a file, sorted by a click on a column's head (again: the other way)
const capWords = (i) => (i.caption || "").trim() ? (i.caption || "").trim().split(/\s+/).length : 0;
const CAP_COLS = [
  {key: "name", label: "File", sort: (i) => i.name.toLowerCase()},
  {key: "caption", label: "Caption", sort: (i) => (i.caption || "").toLowerCase()},
  {key: "words", label: "Words", sort: capWords},
  {key: "size", label: "Size", sort: (i) => (i.size ? i.size[0] * i.size[1] : -1)},
  {key: "mask", label: "Mask", sort: (i) => (i.mask ? 1 : 0)},
];
function captionsTable(items, ext) {
  const ts = S.cap.tsort || {key: "name", dir: 1};
  const head = el("tr", {}, el("th", {}), ...CAP_COLS.map((c) => el("th", {class: "cap-th",
    text: c.label + (ts.key === c.key ? (ts.dir > 0 ? " ▲" : " ▼") : ""), title: "Sort by this column",
    onclick: () => { S.cap.tsort = {key: c.key, dir: ts.key === c.key ? -ts.dir : 1}; S.cap.page = 0; renderCaptions(); }})));
  const rows = items.map((it) => {
    const ta = el("textarea", {rows: 2, class: "cap-cell-text"});
    ta.value = it.caption ?? "";
    it._ta = ta;                                 // the live refresh while captioning runs fills it too
    const words = el("td", {class: "cap-num", text: String(capWords(it))});
    ta.addEventListener("change", async () => {  // saved when you leave the box, as typed
      try {
        await api("/api/captions/save", {folder: capItemFolder(it), name: it.name, text: ta.value, ext});
        it.caption = ta.value; words.textContent = String(capWords(it));
      } catch (e) { ta.title = e.message; }
    });
    const thumb = it.size ? el("img", {class: "cap-thumb", loading: "lazy",
      src: `/api/thumb?path=${encodeURIComponent(it.path)}&t=${encodeURIComponent(TOKEN)}`}) : el("span", {text: ""});
    return el("tr", {}, el("td", {}, thumb), el("td", {class: "cap-name", text: mediaLabel(it) +
      (it.crop ? " · detail crop" : ""), title: it.name}),
      el("td", {}, ta), words,
      el("td", {class: "cap-num", text: it.size ? `${it.size[0]}×${it.size[1]}` : ""}),
      el("td", {class: "cap-num", text: it.mask ? "yes" : ""}));
  });
  return el("table", {class: "captions-table"}, head, ...rows);
}

async function addText(apply) {
  const host = $("frChanges");
  try {
    const r = await api("/api/captions/add", {family: S.family, folder: capFolder(), start: $("frStart").value,
      end: $("frEnd").value, trigger: $("capTrigger").value, apply, ext: capExt()});
    host.innerHTML = "";
    $("frStatus").textContent = r.changes.length ? (apply ? `Added to ${r.changes.length} captions.` :
      `${r.changes.length} captions would change:`) : "Every caption already has that.";
    if (!apply) for (const c of r.changes.slice(0, 100)) host.append(el("div", {class: "checkpoint-row"},
      el("span", {class: "checkpoint-name", text: c.name}),
      el("span", {class: "field-hint", text: "Before: " + c.old}), el("span", {class: "field-hint", text: "After: " + c.new})));
    if (apply) loadCaptions();
  } catch (e) { $("frStatus").textContent = e.message; }
}

async function capRemove(it, card, status) {
  // out of the training set (into removed/), with an Undo on the spot
  const folder = capItemFolder(it);
  try {
    const r = await api("/api/captions/remove", {folder, name: it.name, ext: $("captionExt").value.trim() || ".txt"});
    S.cap.items = S.cap.items.filter((x) => x !== it);
    const undo = el("button", {class: "btn btn-sm", text: "Undo", onclick: async () => {
      try {
        await api("/api/captions/restore", {folder, moved: r.moved});
        S.cap.items.push(it); S.cap.items.sort((a, b) => a.name.localeCompare(b.name)); renderCaptions();
      } catch (e) { note.textContent = e.message; }
    }});
    const note = el("span", {class: "field-hint", text: `${it.name} moved to removed.`});
    card.replaceChildren(el("div", {class: "caption-card-body caption-removed"}, note, el("div", {class: "field-row"}, undo)));
    $("captionsSummary").textContent = `${S.cap.items.length} files · ${it.name} removed`;
    return true;
  } catch (e) { status.textContent = e.message; return false; }
}

async function findReplace(apply) {
  const host = $("frChanges");
  try {
    const r = await api("/api/captions/replace", {family: S.family, folder: capFolder(), find: $("frFind").value,
      replace: $("frReplace").value, apply, ext: capExt()});
    host.innerHTML = "";
    $("frStatus").textContent = r.changes.length ? (apply ? `Replaced in ${r.changes.length} captions.` :
      `${r.changes.length} captions would change:`) : "No caption contains that.";
    if (!apply) for (const c of r.changes.slice(0, 100)) host.append(el("div", {class: "checkpoint-row"},
      el("span", {class: "checkpoint-name", text: c.name}),
      el("span", {class: "field-hint", text: "Before: " + c.old}), el("span", {class: "field-hint", text: "After: " + c.new})));
    if (apply) loadCaptions();
  } catch (e) { $("frStatus").textContent = e.message; }
}

function wireCaptions() {
  $("captionsReload").addEventListener("click", loadCaptions);
  // the box is the Training tab's Caption Extension (one setting, as in the desktop app): a change here is that
  $("captionExt").addEventListener("change", () => {
    S.values.DATASET_CAPTION_EXT = $("captionExt").value.trim() || ".txt";
    syncInputs(["DATASET_CAPTION_EXT"]); saveStateSoon(); loadCaptions();
  });
  $("capModel").addEventListener("change", () => { fillTasks(); onCapTask(); });
  $("capPerPage").addEventListener("change", () => {
    CAP_PAGE = Number($("capPerPage").value) || 12; S.cap.page = 0; renderCaptions(); saveCapSoon(); });
  $("capTask").addEventListener("change", onCapTask);
  $("capConcept").addEventListener("input", saveCapSoon);
  for (const id of ["capTrigger", "capTokens", "capOverwrite", "capSongWords", "capSkipCjk"])
    $(id).addEventListener("change", saveCapSoon);
  // a trigger word typed here fills auto-recaption's on the Training tab (an empty one leaves it alone)
  $("capTrigger").addEventListener("input", () => {
    S.trigger = $("capTrigger").value.trim().replace(/,+$/, "").trim();
    S.values.CAPTION_TRIGGER_WORD = S.trigger;
    syncInputs(["CAPTION_TRIGGER_WORD"]);
    saveTriggerSoon($("capTrigger").value.trim());
    saveStateSoon();
  });
  $("capEditInstr").addEventListener("click", () =>
    $("capInstrBox").style.display === "none" ? openInstr() : ($("capInstrBox").style.display = "none"));
  $("capInstrSave").addEventListener("click", () => saveInstr($("capInstr").value));
  $("capInstrReset").addEventListener("click", () => saveInstr(""));
  $("capAll").addEventListener("click", () => capRun("/api/captions/ai", capJobBody()));
  $("capStatic").addEventListener("click", async () => {
    const r = await capRun("/api/captions/static", capJobBody());
    if (r && !r.error) { $("capStatus").textContent = `${r.written} written, ${r.skipped} kept.`; loadCaptions(); }
  });
  $("capSongs").addEventListener("click", async () => {
    const r = await capRun("/api/captions/songs", capJobBody({words: $("capSongWords").value}));
    if (r && !r.error) { $("capStatus").textContent = `${r.written} songs captioned "${r.caption}", ${r.skipped} kept.`; loadCaptions(); }
  });
  $("capTranslate").addEventListener("click", () =>
    capRun("/api/captions/translate", capJobBody({skip_existing: $("capSkipCjk").checked})));
  $("capStop").addEventListener("click", () => capRun("/api/captions/stop", {}));
  $("capUnload").addEventListener("click", () => capRun("/api/captions/unload", {}));
  $("frPreview").addEventListener("click", () => findReplace(false));
  $("frAddPreview").addEventListener("click", () => addText(false));
  $("frAdd").addEventListener("click", () => addText(true));
  $("capView").addEventListener("change", () => { S.cap.page = 0; renderCaptions(); });
  $("frApply").addEventListener("click", held($("frApply"), () => findReplace(true)));
  $("capSearch").addEventListener("input", debounce(() => { S.cap.page = 0; renderCaptions(); }, 250));
  $("capSort").addEventListener("change", () => { S.cap.page = 0; renderCaptions(); });
  $("capPrev").addEventListener("click", () => { S.cap.page--; renderCaptions(); window.scrollTo(0, 0); });
  $("capNext").addEventListener("click", () => { S.cap.page++; renderCaptions(); });
  const goto = () => {                          // the desktop's Go to page: a number, kept within the pages there are
    const n = parseInt($("capGoto").value, 10);
    if (!Number.isFinite(n)) return;
    S.cap.page = Math.max(0, n - 1); $("capGoto").value = ""; renderCaptions(); window.scrollTo(0, 0);
  };
  $("capGo").addEventListener("click", goto);
  $("capGoto").addEventListener("keydown", (e) => { if (e.key === "Enter") goto(); });
}

// ---- RefMod Toolkit (the desktop tab's): mods, how they apply, a render beside the same seed without them ----------
S.rm = {wired: false, info: null, mods: {}, rows: [], calc: {}, seen: {}, left: null, lastT: 0, polling: false};
const RM_KEY = "fizgig.refmod";
const rmSaved = () => { try { return JSON.parse(localStorage.getItem(RM_KEY) || "null"); } catch (e) { return null; } };
const rmMedia = (it, mp4) => `/api/refmod/media?gen=${it.gen}&side=${it.side}&i=${it.i}${mp4 ? "&mp4=1" : ""}&t=${encodeURIComponent(TOKEN)}`;

function rmCurve(k) { return [$(`rm${k}Dir`).value, $(`rm${k}Shape`).value, Math.round(Number($(`rm${k}Val`).value) * 100) / 100]; }
function rmState() {
  // the desktop's setup keys (presets/refmod_studio/*.json load in either app)
  return {family: $("rmFamily").value, base: $("rmBase").value, model: $("rmModel").value, prompt: $("rmPrompt").value.trim(),
    seed: $("rmSeed").value.trim(), frames: $("rmLength").value, width: $("rmW").value, height: $("rmH").value,
    steps: $("rmSteps").value.trim(), turbo: $("rmTurbo").value.trim(), sound: $("rmSound").checked, early: $("rmEarly").checked,
    folder: $("rmFolder").value.trim(), rows: S.rm.rows.map((r) => ({on: r.on, mod: r.mod, value: r.value, copies: r.copies})),
    retention: Number($("rmRet").value), scramble: $("rmScramble").value.trim(), frame_curve: rmCurve("Fc"),
    step_curve: rmCurve("Sc"), step_on: $("rmScOn").checked, numbered: $("rmNumbered").checked,
    lora: $("rmLora").value.trim(), compare: $("rmCompare").value, out: $("rmOut").value.trim()};
}
function rmSetCurve(k, c) {
  if (!c) return;
  if (S.rm.info.directions.includes(c[0])) $(`rm${k}Dir`).value = c[0];
  if (S.rm.info.shapes.includes(c[1])) $(`rm${k}Shape`).value = c[1];
  $(`rm${k}Val`).value = Number(c[2] ?? 1);
}
function rmApplyState(d) {
  if (!d) return;
  const set = (id, v) => { if (v != null && v !== "") $(id).value = String(v); };
  const opt = (id, v) => { if (v != null && [...$(id).options].some((o) => o.value === String(v))) $(id).value = String(v); };
  opt("rmFamily", d.family); opt("rmBase", d.base); opt("rmModel", d.model); opt("rmLength", d.frames);
  opt("rmW", d.width); opt("rmH", d.height); opt("rmCompare", d.compare);
  set("rmPrompt", d.prompt); set("rmSeed", d.seed); set("rmSteps", d.steps); set("rmTurbo", d.turbo); set("rmFolder", d.folder);
  set("rmScramble", d.scramble); set("rmOut", d.out);
  $("rmLora").value = d.lora || "";
  if ("sound" in d) $("rmSound").checked = !!d.sound;
  if ("early" in d) $("rmEarly").checked = !!d.early;
  $("rmNumbered").checked = !!d.numbered;
  $("rmScOn").checked = !!d.step_on;
  if (d.retention != null) $("rmRet").value = Number(d.retention);
  rmSetCurve("Fc", d.frame_curve); rmSetCurve("Sc", d.step_curve);
  if (Array.isArray(d.rows)) S.rm.rows = d.rows.slice(0, S.rm.info.max_rows).map((r) => ({on: r.on !== false,
    mod: String(r.mod || S.rm.info.none), value: Math.max(0, Math.min(1, Number(r.value ?? 1))),
    copies: Math.max(1, Math.min(S.rm.info.max_copies, parseInt(r.copies, 10) || 1))}));
  if (!S.rm.rows.length) S.rm.rows = [{on: true, mod: S.rm.info.none, value: 1, copies: 1}];
}

function rmRemember() { try { localStorage.setItem(RM_KEY, JSON.stringify(rmState())); } catch (e) { /* private */ } }

function rmChanged() {
  $("rmRetNum").value = Number($("rmRet").value).toFixed(2);
  $("rmFcValNum").textContent = Number($("rmFcVal").value).toFixed(2);
  $("rmScValNum").textContent = Number($("rmScVal").value).toFixed(2);
  $("rmNumNote").style.display = $("rmNumbered").checked ? "" : "none";
  rmTitles();
  rmDrawCurves();
  rmBakeRows();
  rmRemember();
  rmCalcSoon();
}
const rmCalcSoon = debounce(rmCalc, 300);
async function rmCalc() {
  try { S.rm.calc = await api("/api/refmod/calc", {state: rmState()}); } catch (e) { return; }
  const c = S.rm.calc, cap = S.rm.info.token_cap;
  $("rmTokens").textContent = `Tokens: ${c.tokens.toLocaleString()} / ${cap.toLocaleString()}`;
  $("rmTokens").classList.toggle("over", c.tokens > cap);
  $("rmRefMap").textContent = !$("rmNumbered").checked ? "" : (c.ref_map.length ? c.ref_map.join("   ") : "No active mod - nothing to number.");
  if ($("rmReadout").style.display !== "none") $("rmReadout").textContent = c.readout;
}

// the mods ---------------------------------------------------------------------------------------------------------
async function rmRescan(quiet) {
  const st = rmState();
  let r;
  try { r = await api(`/api/refmod/scan?folder=${encodeURIComponent(st.folder)}&lora=${encodeURIComponent(st.lora)}`); }
  catch (e) { $("rmScanNote").textContent = e.message; return; }
  S.rm.mods = Object.fromEntries(r.mods.map((m) => [m.name, m]));
  const dl = $("rmModList");
  dl.innerHTML = "";
  for (const n of [S.rm.info.none, ...Object.keys(S.rm.mods)]) dl.append(el("option", {value: n}));
  if (!quiet || r.mods.length) $("rmScanNote").textContent = r.status;
  rmDrawRows();
  rmGraphs();
  rmChanged();
}
function rmResolve(typed, prev) {
  // a typed name: an exact match (any case), else the only name containing it, else the previous pick
  const names = [S.rm.info.none, ...Object.keys(S.rm.mods)];
  const t = typed.trim().toLowerCase();
  const exact = names.find((n) => n.toLowerCase() === t);
  if (exact) return exact;
  const hits = t ? names.filter((n) => n.toLowerCase().includes(t)) : [];
  return hits.length === 1 ? hits[0] : prev;
}
function rmDrawRows() {
  const box = $("rmRows");
  box.innerHTML = "";
  S.rm.rows.forEach((r, i) => {
    const m = S.rm.mods[r.mod];
    const info = el("div", {class: "rm-info", text: m ? m.info : r.mod !== S.rm.info.none ? "(mod not in this folder)" : ""});
    const num = el("input", {type: "text", class: "w-num", value: r.value.toFixed(2)});
    const rng = el("input", {type: "range", min: 0, max: 1, step: 0.01, value: r.value, class: "rm-slider-sm",
      title: "How strongly this mod applies (0 = off, 1 = as stored)",
      oninput: () => { r.value = Number(rng.value); num.value = r.value.toFixed(2); rmChanged(); }});
    num.addEventListener("change", () => {
      const v = Number(num.value);
      r.value = Number.isFinite(v) ? Math.max(0, Math.min(1, v)) : r.value;
      num.value = r.value.toFixed(2); rng.value = r.value; rmChanged();
    });
    const mod = el("input", {type: "text", class: "w-path rm-mod", list: "rmModList", value: r.mod,
      placeholder: "type to search the mods", onfocus: (e) => e.target.select()});
    mod.addEventListener("change", () => {
      r.mod = rmResolve(mod.value, r.mod); mod.value = r.mod;
      const mm = S.rm.mods[r.mod];
      info.textContent = mm ? mm.info : r.mod !== S.rm.info.none ? "(mod not in this folder)" : "";
      rmChanged();
    });
    const cp = el("input", {type: "number", min: 1, max: S.rm.info.max_copies, value: r.copies, class: "w-num",
      title: "The same reference repeated in the bundle - 2-3 is the pack's sweet spot for a stronger pull; every copy costs its tokens.",
      onchange: () => { r.copies = Math.max(1, Math.min(S.rm.info.max_copies, parseInt(cp.value, 10) || 1)); cp.value = r.copies; rmChanged(); }});
    const on = el("input", {type: "checkbox", onchange: () => { r.on = on.checked; rmChanged(); }});
    on.checked = r.on;
    box.append(el("div", {class: "rm-row"},
      el("div", {class: "field-row"}, on, mod, el("span", {class: "field-hint", text: "Strength"}), rng, num,
        el("span", {class: "field-hint", text: "Copies"}), cp,
        el("button", {class: "btn btn-sm", text: "✕", title: "Remove this row", onclick: () => {
          if (S.rm.rows.length > 1) S.rm.rows.splice(i, 1);
          else S.rm.rows[0] = {on: true, mod: S.rm.info.none, value: 1, copies: 1};   // the last row empties instead
          rmDrawRows(); rmChanged();
        }})),
      info));
  });
  $("rmAddRow").disabled = S.rm.rows.length >= S.rm.info.max_rows;
}
function rmBakeRows() {
  const sel = $("rmBakeRow"), was = sel.value;
  sel.innerHTML = "";
  const ret = Number($("rmRet").value);
  S.rm.rows.forEach((r, i) => {
    if (r.on && r.value > 0 && S.rm.mods[r.mod]) sel.append(el("option", {value: i, text: `${r.mod} @ ${(Math.min(1, r.value) * ret).toFixed(2)}`}));
  });
  if ([...sel.options].some((o) => o.value === was)) sel.value = was;
}
async function rmLoraChanged() {
  const path = $("rmLora").value.trim();
  if (path === S.rm.seen.lora) return;
  S.rm.seen.lora = path;
  await rmLoraInfo(true);
}
async function rmLoraInfo(replaceRows) {
  // the LoRA card's line, the pickers with a ReFLoRA's references; a newly opened ReFLoRA's become the rows
  const path = $("rmLora").value.trim();
  let r;
  try { r = await api("/api/refmod/lora", {path, folder: $("rmFolder").value.trim()}); }
  catch (e) { $("rmLoraInfo").textContent = e.message; return; }
  $("rmLoraInfo").textContent = r.line;
  await rmRescan(true);
  if (replaceRows && r.refs && r.refs.length) {   // a ReFLoRA: its own RefMods become the rows
    S.rm.rows = r.refs.slice(0, S.rm.info.max_rows).map((n) => ({on: true, mod: n, value: 1, copies: 1}));
    rmDrawRows();
    $("rmStatus").textContent = `Opened the ReFLoRA ${path.split(/[\\/]/).pop()} - press Render.`;
  }
  rmChanged();
}
function rmTitles() {
  const lora = !!$("rmLora").value.trim() && !/^File not found/.test($("rmLoraInfo").textContent);
  $("rmCompareBox").style.display = lora ? "" : "none";
  $("rmRflRow").style.display = lora ? "" : "none";
  const mode = lora ? $("rmCompare").selectedIndex : 0;
  $("rmBaseCap").textContent = [lora ? "LoRA alone (no mod, same seed)" : "No mod (base, same seed)",
    "Mods alone (no LoRA, same seed)", "Neither (base, same seed)"][mode];
  if (!S.rm.left) $("rmTweakCap").textContent = lora ? "LoRA + mods" : "With mods";
}

// curves -----------------------------------------------------------------------------------------------------------
function rmEase(shape, x) {
  switch (shape) {
    case "linear": return x;
    case "ease": return x * x * (3 - 2 * x);
    case "sigmoid": return 1 / (1 + Math.exp(-12 * (x - 0.5)));
    case "tanh": return 0.5 * (Math.tanh(8 * (x - 0.5)) + 1);
    case "quadratic": return x * x;
    case "cubic": return x * x * x;
    case "exponential": return 2 ** x - 1;
    case "stair": return Math.min(1, Math.floor(x * 4) / 3);
    case "elastic": return x <= 0 ? 0 : x >= 1 ? 1 : 2 ** (-10 * x) * Math.sin((x * 10 - 0.75) * (2 * Math.PI / 3)) + 1;
    case "bump": return 1 - Math.abs(2 * x - 1);
    case "dip": return Math.abs(2 * x - 1);
    default: return x;
  }
}
function rmCurveAt([dir, shape, v], x) {
  // fizgig.minimax.refmod_meta.curve_value_at
  const c = (t) => Math.max(0, Math.min(1, t));
  if (dir === "constant") return shape === "linear" ? v : c(v * rmEase(shape, x));
  const y = rmEase(shape, x);
  if (dir === "concept_at_end") return c(v * y);
  if (dir === "concept_at_start") return c(v * (1 - y));
  if (dir === "concept_at_middle") return c(v * (1 - Math.abs(2 * y - 1)));
  if (dir === "concept_at_ends") return c(v * Math.abs(2 * y - 1));
  return 1;
}
function rmDrawCurves() {
  const cv = $("rmCurves"), g = cv.getContext("2d");
    const W = cv.width, H = cv.height, x0 = 36, y0 = 12, x1 = W - 12, y1 = H - 26;
  g.clearRect(0, 0, W, H);
  g.strokeStyle = themeColor("--border"); g.fillStyle = themeColor("--text-secondary"); g.lineWidth = 1;
  g.font = "9px sans-serif"; g.textAlign = "right"; g.textBaseline = "middle";
  for (let i = 0; i < 5; i++) {
    const y = y0 + (y1 - y0) * i / 4;
    g.beginPath(); g.moveTo(x0, y); g.lineTo(x1, y); g.stroke();
    g.fillText((1 - i / 4).toFixed(2), x0 - 6, y);
    const x = x0 + (x1 - x0) * i / 4;
    g.beginPath(); g.moveTo(x, y0); g.lineTo(x, y1); g.stroke();
  }
  const line = (spec, colour, dash) => {
    g.strokeStyle = colour; g.lineWidth = 2; g.setLineDash(dash || []);
    g.beginPath();
    for (let i = 0; i <= 200; i++) {
      const x = i / 200, v = Math.max(0, Math.min(1, rmCurveAt(spec, x)));
      const px = x0 + (x1 - x0) * x, py = y1 - (y1 - y0) * v;
      i ? g.lineTo(px, py) : g.moveTo(px, py);
    }
    g.stroke(); g.setLineDash([]);
  };
  const accent = themeColor("--accent");
  line(rmCurve("Fc"), accent);
  g.textBaseline = "top"; g.fillStyle = accent;
  g.textAlign = "left"; g.fillText("reference frame 1", x0 + 4, y1 + 6);
  g.textAlign = "right"; g.fillText("last frame", x1 - 4, y1 + 6);
  if ($("rmScOn").checked) {
    line(rmCurve("Sc"), themeColor("--warning"), [4, 3]);
    g.fillStyle = themeColor("--warning"); g.textAlign = "center"; g.fillText("step curve: first step → last step", (x0 + x1) / 2, y1 + 6);
  }
  g.fillStyle = themeColor("--text-secondary"); g.textAlign = "right"; g.fillText("reference strength", x1 - 4, y0 + 2);
}
function rmFill(id, values, labels) {
  const sel = $(id);
  sel.innerHTML = "";
  values.forEach((v, i) => sel.append(el("option", {value: v, text: labels ? labels[i] : v})));
}
async function rmGraphs() {
  let r = {presets: []};
  try { r = await api(`/api/refmod/graphs?folder=${encodeURIComponent($("rmFolder").value.trim())}`); } catch (e) { /* none */ }
  S.rm.graphs = r.presets;
  rmFill("rmGraph", r.presets.map((p, i) => i), r.presets.map((p) => p.name));
  $("rmGraph").disabled = $("rmGraphUse").disabled = !r.presets.length;
  $("rmGraphNote").textContent = r.presets.length ? "" : "none in <RefMod folder>/graph_presets";
}

// rendering --------------------------------------------------------------------------------------------------------
function rmShow(img, vid, it) {
  const clip = it && it.media && it.frames > 1;
  $(vid).style.display = clip ? "" : "none";
  $(img).style.display = clip ? "none" : "";
  if (!it) { $(img).removeAttribute("src"); $(vid).removeAttribute("src"); return; }
  if (clip) { $(vid).poster = rmMedia(it); if ($(vid).dataset.src !== rmMedia(it, true)) { $(vid).dataset.src = rmMedia(it, true); $(vid).src = rmMedia(it, true); } }
  else $(img).src = rmMedia(it);
}
function rmShowLeft(it, caption) {
  S.rm.left = it;
  rmShow("rmTweak", "rmTweakVid", it);
  $("rmTweakCap").textContent = caption;
  document.querySelectorAll("#rmChips figure, #rmHist figure").forEach((f) => f.classList.toggle("on", f.dataset.key === `${it.gen}-${it.side}-${it.i}`));
}
function rmStrip(box, items, describe) {
  box.innerHTML = "";
  for (const it of items) {
    const fig = el("figure", {title: it.label || "", onclick: () => rmShowLeft(it, describe(it))},
      el("img", {src: rmMedia(it), alt: "", loading: "lazy"}), el("figcaption", {text: describe(it)}));
    fig.dataset.key = `${it.gen}-${it.side}-${it.i}`;
    box.append(fig);
  }
}
async function rmPoll() {
  let st;
  try { st = await api("/api/refmod/status"); } catch (e) { S.rm.polling = false; return; }
  const busy = st.state === "rendering";
  $("rmRender").disabled = $("rmSweep").disabled = busy;
  $("rmCancel").disabled = !busy;
  if (st.status) $("rmStatus").textContent = st.status;
  const p = st.progress;
  $("rmBar").style.width = p && p[1] ? `${Math.round(100 * p[0] / p[1])}%` : busy ? "100%" : "0";
  $("rmBar").parentElement.classList.toggle("indeterminate", busy && !(p && p[1]));
  if (busy && st.early && st.kind === "render") { $("rmTweak").style.display = ""; $("rmTweakVid").style.display = "none";
    $("rmTweak").src = `/api/refmod/media?side=early&gen=0&i=0&t=${encodeURIComponent(TOKEN)}&n=${Date.now()}`; }
  const t = st.tweaked, b = st.baseline;
  if (t && t.gen !== S.rm.lastT) {
    S.rm.lastT = t.gen;
    const lora = !!$("rmLora").value.trim();
    rmShowLeft(t, (lora ? "LoRA + mods - " : "With mods - ") + ((t.describe || []).map(([n, s]) => `${n}@${s.toFixed(2)}`).join(", ") || "none"));
  }
  const bk = b ? `${b.gen}` : "";
  if (bk !== S.rm.seen.base) { S.rm.seen.base = bk; rmShow("rmBaseImg", "rmBaseVid", b); }
  $("rmPlayBoth").style.display = t && b && t.media && b.media && t.frames > 1 && b.frames > 1 ? "" : "none";
  const ck = st.chips.map((c) => c.gen + "-" + c.i).join(",");
  if (ck !== S.rm.seen.chips) {
    S.rm.seen.chips = ck;
    $("rmSweepBox").style.display = st.chips.length ? "" : "none";
    $("rmSweepName").textContent = st.sweep || "";
    rmStrip($("rmChips"), st.chips, (c) => c.label);
  }
  const hk = st.history.map((h) => h.gen).join(",");
  if (hk !== S.rm.seen.hist) {
    S.rm.seen.hist = hk;
    $("rmHistBox").style.display = st.history.length ? "" : "none";
    rmStrip($("rmHist"), st.history, (h) => (h.describe || []).map(([n, s]) => `${n}@${s.toFixed(2)}`).join(", ") || "no mods");
    for (const [i, f] of [...$("rmHist").children].entries()) f.title = st.history[i].label || "";
  }
  if (busy) setTimeout(rmPoll, 800); else S.rm.polling = false;
}
function rmPollStart() { if (!S.rm.polling) { S.rm.polling = true; rmPoll(); } }
async function rmGo(kind) {
  rmRemember();
  try {
    const r = await api(`/api/refmod/${kind}`, {state: rmState(), sweep: $("rmSweepKind").value});
    $("rmStatus").textContent = kind === "sweep" ? `Sweep: ${$("rmSweepKind").value}…`
      : !r.active ? "No active mod - rendering With mods as the plain base." : r.need_base ? "Rendering With mods, then the comparison…" : "Rendering With mods…";
    rmPollStart();
  } catch (e) { $("rmStatus").textContent = e.message; }
}

async function rmAct(path, body, say) {
  $("rmActStatus").textContent = "Working…";
  try { const r = await api(path, body); $("rmActStatus").textContent = say(r); return r; }
  catch (e) { $("rmActStatus").textContent = e.message; return null; }
}
function rmPresets(curves, setups) {
  if (curves) { rmFill("rmCurvePreset", ["", ...curves], ["(curve presets)", ...curves]); }
  if (setups) { rmFill("rmSetup", setups); $("rmSetupLoad").disabled = !setups.length; }
}

async function refmodOpen() {
  if (!S.rm.wired) {
    S.rm.wired = true;
    try { S.rm.info = await api("/api/refmod/info"); } catch (e) { $("rmStatus").textContent = e.message; return; }
    const I = S.rm.info;
    if (!I.families.length) { $("rmStatus").textContent = "No model offers RefMod Toolkit."; $("rmRender").disabled = $("rmSweep").disabled = true; return; }
    rmFill("rmFamily", I.families.map((f) => f.key), I.families.map((f) => f.name));
    $("rmFamRow").style.display = I.families.length > 1 ? "" : "none";
    rmFill("rmModel", I.models); rmFill("rmBase", I.bases);
    rmFill("rmLength", I.lengths); rmFill("rmSweepKind", I.sweeps); rmFill("rmCompare", I.compare);
    rmFill("rmW", I.sizes); rmFill("rmH", I.sizes);
    for (const k of ["Fc", "Sc"]) { rmFill(`rm${k}Dir`, I.directions); rmFill(`rm${k}Shape`, I.shapes); }
    $("rmLength").value = I.lengths.find((l) => /^22 /.test(l)) || I.lengths[0]; $("rmW").value = I.size[0]; $("rmH").value = I.size[1];
    rmSetCurve("Fc", I.frame_curve); rmSetCurve("Sc", I.step_curve);
    $("rmPrompt").value = "a woman smiles at the camera, soft window light";
    for (const [name, v] of I.retention_presets) $("rmRetPresets").append(el("button", {class: "btn btn-sm", text: `${name} ${v}`,
      onclick: () => { $("rmRet").value = v; rmChanged(); }}));
    rmPresets(I.curve_presets, I.setups);
    S.rm.rows = [{on: true, mod: I.none, value: 1, copies: 1}];
    rmApplyState(rmSaved() || I.desktop);    // this browser's last, else the desktop tab's
    if (!$("rmOut").value) $("rmOut").value = S.outputDefault || "";
    rmWire();
    S.rm.seen.lora = $("rmLora").value.trim();
    await rmLoraInfo(false);
  }
  rmPollStart();
}

function rmWire() {
  const ch = () => rmChanged();
  for (const id of ["rmRet", "rmFcVal", "rmScVal"]) $(id).addEventListener("input", ch);
  for (const id of ["rmFcDir", "rmFcShape", "rmScDir", "rmScShape", "rmScOn", "rmNumbered", "rmCompare", "rmModel", "rmBase",
    "rmFamily", "rmLength", "rmW", "rmH", "rmSound", "rmEarly", "rmPrompt", "rmScramble", "rmSeed", "rmOut"]) $(id).addEventListener("change", ch);
  $("rmRetNum").addEventListener("change", () => {
    const v = Number($("rmRetNum").value);
    if (Number.isFinite(v)) $("rmRet").value = Math.max(0, Math.min(1, v));
    rmChanged();
  });
  $("rmSteps").addEventListener("change", () => { const v = parseInt($("rmSteps").value, 10); $("rmSteps").value = Number.isFinite(v) ? Math.max(1, Math.min(40, v)) : 6; rmChanged(); });
  $("rmTurbo").addEventListener("change", () => { const v = Number($("rmTurbo").value); $("rmTurbo").value = Number.isFinite(v) ? Math.max(0, Math.min(1.5, v)) : 0.75; rmChanged(); });
  $("rmPrompt").addEventListener("input", debounce(rmRemember, 400));
  $("rmScrRand").addEventListener("click", () => { $("rmScramble").value = Math.floor(Math.random() * 2 ** 31); rmChanged(); });
  $("rmScrOff").addEventListener("click", () => { $("rmScramble").value = "-1"; rmChanged(); });
  $("rmSeedRand").addEventListener("click", () => { $("rmSeed").value = Math.floor(Math.random() * 2 ** 31); rmChanged(); });
  $("rmFolderPick").addEventListener("click", () => pick($("rmFolder").value, true, (p) => { $("rmFolder").value = p; rmRescan(); }));
  $("rmFolder").addEventListener("change", () => rmRescan());
  $("rmRescan").addEventListener("click", () => rmRescan());
  $("rmLoraPick").addEventListener("click", () => pick($("rmLora").value || loraStart(), false, (p) => { $("rmLora").value = p; rmLoraChanged(); }));
  $("rmLora").addEventListener("change", rmLoraChanged);
  $("rmLoraClear").addEventListener("click", () => { $("rmLora").value = ""; rmLoraChanged(); });
  $("rmAddRow").addEventListener("click", () => {
    if (S.rm.rows.length < S.rm.info.max_rows) S.rm.rows.push({on: true, mod: S.rm.info.none, value: 1, copies: 1});
    rmDrawRows(); rmChanged();
  });
  $("rmHints").addEventListener("click", async () => {
    await rmCalc();
    const h = S.rm.calc.hints, cur = $("rmPrompt").value.trim();
    if (!h) { $("rmStatus").textContent = "No active mod carries a description - nothing to add."; return; }
    if (!cur.includes(h)) { $("rmPrompt").value = cur ? `${cur}, ${h}` : h; rmChanged(); }
  });
  $("rmLabels").addEventListener("click", async () => {
    $("rmNumbered").checked = true;
    await rmCalc();
    const lines = S.rm.calc.ref_map || [];
    if (!lines.length) { $("rmStatus").textContent = "No active mod - nothing to number."; return; }
    const labels = lines.map((l) => l.split(" = ")[0]).join(", "), cur = $("rmPrompt").value.trim();
    if (!cur.includes(labels)) $("rmPrompt").value = cur ? `${cur}, ${labels}` : labels;
    rmChanged();
  });
  $("rmRender").addEventListener("click", () => rmGo("render"));
  $("rmSweep").addEventListener("click", () => rmGo("sweep"));
  $("rmCancel").addEventListener("click", () => api("/api/refmod/cancel", {}).then(rmPollStart).catch(() => {}));
  $("rmUnload").addEventListener("click", async () => {
    await api("/api/studio/reset", {}).catch(() => {});   // the shared GPU worker quits (Studio's Unload)
    $("rmStatus").textContent = "Model unloaded - the next Render loads it again.";
  });
  playTogether($("rmPlayBothBtn"), () => ["rmTweakVid", "rmBaseVid"]);
  // curve presets and the pack's graph presets
  $("rmCurvePreset").addEventListener("change", async () => {
    const name = $("rmCurvePreset").value;
    if (!name) return;
    try {
      const d = await api("/api/refmod/curve_load", {name});
      rmSetCurve("Fc", d.frame_curve); rmSetCurve("Sc", d.step_curve);
      if ("step_on" in d) $("rmScOn").checked = d.step_on;
      if ("retention" in d) $("rmRet").value = d.retention;
      $("rmCurveName").value = name;
      rmChanged();
    } catch (e) { $("rmStatus").textContent = e.message; }
  });
  $("rmCurveSave").addEventListener("click", async () => {
    const name = $("rmCurveName").value.trim() || $("rmCurvePreset").value;
    let r;
    try {
      r = await api("/api/refmod/curve_save", {name, state: rmState()});
      if (r.exists && confirm(`Replace the curve preset '${r.name}'?`)) r = await api("/api/refmod/curve_save", {name, state: rmState(), overwrite: true});
    } catch (e) { $("rmStatus").textContent = e.message; return; }
    if (r.ok) { rmPresets(r.presets); $("rmCurvePreset").value = r.name; $("rmStatus").textContent = `Curve preset '${r.name}' saved.`; }
  });
  $("rmCurveDel").addEventListener("click", async () => {
    const name = $("rmCurvePreset").value;
    if (!name || !confirm(`Delete the curve preset '${name}'?`)) return;
    const r = await api("/api/refmod/curve_delete", {name}).catch((e) => ({error: e.message}));
    if (r.presets) rmPresets(r.presets);
  });
  $("rmGraphUse").addEventListener("click", () => {
    const p = (S.rm.graphs || [])[Number($("rmGraph").value)];
    if (!p) return;
    rmSetCurve("Fc", p.curve); rmChanged();
    $("rmStatus").textContent = `Fade from ${p.name}: ${p.curve[0]} / ${p.curve[1]} / ${Number(p.curve[2]).toFixed(2)}.`;
  });
  $("rmGraphImport").addEventListener("click", () => {     // a graph preset from anywhere (the desktop's Import…)
    const folder = $("rmFolder").value.trim();
    pick(folder ? folder + "/graph_presets" : "", false, async (p) => {
      try {
        const r = await api("/api/refmod/graph", {path: p});
        rmSetCurve("Fc", r.curve); rmChanged();
        $("rmStatus").textContent = `Fade from ${p.split(/[\\/]/).pop()}: ${r.curve[0]} / ${r.curve[1]} / ${Number(r.curve[2]).toFixed(2)}.`;
      } catch (e) { $("rmStatus").textContent = e.message; }
    });
  });
  // actions
  $("rmOutPick").addEventListener("click", () => pick($("rmOut").value, true, (p) => { $("rmOut").value = p; rmRemember(); }));
  $("rmSavePrev").addEventListener("click", () => {
    const it = S.rm.left || null;
    rmAct("/api/refmod/save_preview", {out: $("rmOut").value.trim(), gen: it ? it.gen : S.rm.lastT, side: it ? it.side : "tweaked", i: it ? it.i : 0},
      (r) => `Saved ${r.path}`);
  });
  $("rmStrip").addEventListener("click", () => rmAct("/api/refmod/save_strip", {out: $("rmOut").value.trim()},
    (r) => `Strip saved: ${r.path}`));
  $("rmReadoutBtn").addEventListener("click", async () => {
    const show = $("rmReadout").style.display === "none";
    if (show) { await rmCalc(); $("rmReadout").textContent = S.rm.calc.readout || ""; }
    $("rmReadout").style.display = $("rmReadoutCopy").style.display = show ? "" : "none";
  });
  $("rmReadoutCopy").addEventListener("click", async () => {
    try { await navigator.clipboard.writeText($("rmReadout").textContent); $("rmActStatus").textContent = "ComfyUI settings copied."; }
    catch (e) { $("rmActStatus").textContent = "Select the text and copy it (the browser blocked the clipboard)."; }
  });
  $("rmSetupSave").addEventListener("click", async () => {
    const name = $("rmSetupName").value.trim();
    let r = await rmAct("/api/refmod/setup_save", {name, data: rmState()}, (x) => x.ok ? `Setup saved: ${x.name}.` : "");
    if (r && r.exists && confirm(`Replace the setup '${r.name}'?`)) r = await rmAct("/api/refmod/setup_save", {name, data: rmState(), overwrite: true}, (x) => `Setup saved: ${x.name}.`);
    if (r && r.setups) { rmPresets(null, r.setups); $("rmSetup").value = r.name; }
  });
  $("rmSetupLoad").addEventListener("click", async () => {
    const name = $("rmSetup").value;
    const r = await rmAct("/api/refmod/setup_load", {name}, () => `Setup loaded: ${name}.`);
    if (!r) return;
    rmApplyState(r.data);
    $("rmSetupName").value = name;
    S.rm.seen.lora = $("rmLora").value.trim();
    await rmLoraInfo(false);
  });
  $("rmBake").addEventListener("click", async () => {
    const row = $("rmBakeRow").value;
    if (row === "") { $("rmActStatus").textContent = "No active mod with a strength above 0."; return; }
    const r = await rmAct("/api/refmod/bake", {state: rmState(), row: Number(row), name: $("rmBakeName").value.trim()}, (x) => x.message);
    if (r) rmRescan(true);
  });
  $("rmRfl").addEventListener("click", async () => {
    const r = await rmAct("/api/refmod/reflora", {state: rmState(), name: $("rmRflName").value.trim()}, (x) => x.message);
    if (r) rmRescan(true);
  });
}

// Play together / Pause together: one button for the panes' players (ids() names the ones in use). Play starts
// them from the top unless they were paused together (then they carry on, still in step); the label follows the
// players, so pausing one by its own controls turns the button back to Play.
function playTogether(btn, ids) {
  const media = () => ids().map((id) => $(id)).filter((m) => m && m.getAttribute("src"));
  const label = () => { btn.textContent = media().some((m) => !m.paused) ? "⏸ Pause together" : "▶ Play together"; };
  btn.addEventListener("click", () => {
    const ms = media();
    if (ms.some((m) => !m.paused)) { ms.forEach((m) => m.pause()); return label(); }
    const t = ms.map((m) => m.currentTime);
    const inStep = t.length && Math.max(...t) - Math.min(...t) < 0.1;
    for (const m of ms) { if (!inStep) m.currentTime = 0; m.play().catch(() => {}); }
    label();
  });
  // every player that could be in the set: the button follows them
  for (const m of document.querySelectorAll("video, audio")) for (const ev of ["play", "pause", "emptied"]) m.addEventListener(ev, label);
}

// ---- file picker -------------------------------------------------------------------------------------------
let pickDone = null, pickDir = false;
async function pick(start, dirOnly, done) {
  pickDone = done; pickDir = dirOnly;
  $("pickerTitle").textContent = dirOnly ? "Choose a folder" : "Choose a file";
  $("pickerChoose").style.display = dirOnly ? "" : "none";
  $("pickerNew").style.display = dirOnly ? "" : "none";           // a folder can be made where one is chosen
  $("pickerNewRow").style.display = "none";
  $("pickerNewMsg").textContent = "";
  // a file not on the machine running Fizgig (a pod, another computer's server): upload it from here instead
  $("pickerUpload").style.display = !dirOnly && pageIsElsewhere() ? "" : "none";
  $("pickerUpMsg").textContent = "";
  $("pickerModal").classList.add("active");
  await browse(start || "");
}
async function browse(path) {
  const r = await api("/api/browse?path=" + encodeURIComponent(path));
  S.pickAt = r.path;
  $("pickerPath").value = r.label || r.path;          // This computer has no path of its own
  $("pickerChoose").disabled = $("pickerNew").disabled = !!r.label;
  $("pickerUp").onclick = () => r.parent && browse(r.parent);
  const places = $("pickerPlaces");
  places.innerHTML = "";
  for (const p of r.places || []) places.append(el("button", {class: "btn btn-sm", text: p.name, title: p.path,
    onclick: () => browse(p.path)}));
  S.pickListing = r;
  $("pickerFilter").value = "";
  pickerRender();
}

// the picker's list: folders first, then files, sorted by the chosen column (remembered), filtered by name
const pickSort = {key: "name", desc: false};
try { Object.assign(pickSort, JSON.parse(localStorage.getItem("fizgig.pickSort") || "{}")); } catch (e) { /* none saved */ }
function pickerRender() {
  const r = S.pickListing;
  if (!r) return;
  const q = $("pickerFilter").value.trim().toLowerCase();
  const ext = (n) => (n.includes(".") ? n.slice(n.lastIndexOf(".") + 1) : "").toLowerCase();
  const cmpName = (a, b) => a.name.localeCompare(b.name, undefined, {numeric: true, sensitivity: "base"});
  const k = pickSort.key;
  const cmp = (a, b) => {
    let c = k === "name" ? 0 : k === "type" ? ext(a.name).localeCompare(ext(b.name)) : (a[k] ?? -1) - (b[k] ?? -1);
    c = c || cmpName(a, b);
    return pickSort.desc ? -c : c;
  };
  const when = (t) => t ? new Date(t * 1000).toLocaleString([], {year: "numeric", month: "short", day: "numeric",
    hour: "2-digit", minute: "2-digit"}) : "";
  const size = (n) => n == null ? "" : n < 1024 ? n + " B" : n < 2 ** 20 ? (n / 1024).toFixed(0) + " KB"
    : n < 2 ** 30 ? (n / 2 ** 20).toFixed(1) + " MB" : (n / 2 ** 30).toFixed(2) + " GB";
  const shown = (xs) => xs.filter((x) => !q || x.name.toLowerCase().includes(q)).sort(cmp);
  const row = (x, dir, go) => el("div", {class: "picker-item" + (dir ? " dir" : ""), title: x.path, onclick: go}, [
    el("span", {text: (dir ? "📁 " : "") + x.name}), el("span", {class: "m", text: when(k === "ctime" ? x.ctime : x.mtime)}),
    el("span", {class: "z", text: dir ? "" : size(x.size)})]);
  const list = $("pickerList");
  list.replaceChildren(...shown(r.dirs).map((d) => row(d, true, () => browse(d.path))),
    ...(pickDir ? [] : shown(r.files).map((f) => row(f, false, () => { $("pickerModal").classList.remove("active"); pickDone(f.path); }))));
  $("pickerSort").value = k;
  $("pickerDir").textContent = pickSort.desc ? "↓" : "↑";
  for (const h of document.querySelectorAll(".picker-head span")) {
    h.classList.toggle("on", h.dataset.sort === k || (k === "ctime" && h.dataset.sort === "mtime"));
    h.classList.toggle("desc", pickSort.desc);
  }
  document.querySelector('.picker-head span[data-sort="mtime"]').textContent = k === "ctime" ? "Date created" : "Date modified";
}
function pickerSortBy(key, flip) {
  if (flip && pickSort.key === key) pickSort.desc = !pickSort.desc;
  else { pickSort.key = key; pickSort.desc = key !== "name" && key !== "type"; }   // dates and sizes: newest / biggest first
  try { localStorage.setItem("fizgig.pickSort", JSON.stringify(pickSort)); } catch (e) { /* private window */ }
  pickerRender();
}

// ---- Open on your phone: a QR code with a one-time pass (the server's /api/phone_link). It shows for 30 s when asked
// for, then blurs again; scanned, the phone opens this Fizgig logged in for 30 days. Needs a password (without one
// Fizgig only answers on this machine), and at home on Windows, Fizgig let through the firewall.
function qrSvg(text) {
  const q = qrcode(0, "M");
  q.addData(text);
  q.make();
  return q.createSvgTag({cellSize: 4, margin: 2, scalable: true});
}

function phoneCard() {
  if (new URLSearchParams(location.search).has("phone")) {         // opened from the code: the pass is spent, drop it
    history.replaceState(null, "", location.pathname);
  }
  const card = $("phoneCard"), box = $("phoneQr"), note = $("phoneNote"), btns = $("phoneBtns");
  if (matchMedia("(pointer: coarse)").matches && innerWidth < 900) { card.style.display = "none"; return; }   // a phone
  card.style.display = "";
  const toPrefs = () => {
    showTab("prefs");
    setTimeout(() => {
      const t = [...document.querySelectorAll("#tab-prefs .card-title span")].find((e) => e.textContent === "Web UI");
      if (t) t.closest(".section-card").scrollIntoView({block: "start", behavior: "smooth"});
    }, 300);
  };
  const hidden = () => {                     // a blurred stand-in where the code goes
    box.innerHTML = qrSvg("https://github.com/shootthesound/Fizgig");
    box.classList.add("blurred");
  };
  const message = (text, label, action) => {
    box.classList.remove("blurred");
    box.replaceChildren(el("div", {class: "phone-msg", text}));
    note.textContent = "";
    btns.replaceChildren(...(label ? [el("button", {class: "btn btn-primary btn-sm", text: label, onclick: action})] : []));
  };
  let timer = 0;
  const ready = () => {
    clearInterval(timer);
    hidden();
    note.textContent = "The code shows for 30 seconds. Each one works once.";
    btns.replaceChildren(el("button", {class: "btn btn-primary btn-sm", text: "Show QR code", onclick: show}));
  };
  const show = async () => {
    let r;
    try { r = await api("/api/phone_link", {}); } catch (e) { note.textContent = e.message; return; }
    box.innerHTML = qrSvg(r.url);
    box.classList.remove("blurred");
    let left = r.seconds || 30;
    const tick = () => { note.textContent = `Scan it now: it hides in ${left} s. Each code works once.`; };
    tick();
    btns.replaceChildren(el("button", {class: "btn btn-sm", text: "Hide", onclick: ready}));
    clearInterval(timer);
    timer = setInterval(() => { left--; if (left <= 0) ready(); else tick(); }, 1000);
  };
  if (!S.passwordOn) {
    if (S.pw.saved) return message("Restart Fizgig to use the password you set, and the QR code appears here.",
      "Restart Fizgig", toPrefs);
    return message("Set a password on this Fizgig to open it on your phone: scan a QR code here and you're straight in.",
      "Set a password", toPrefs);
  }
  ready();
  if (S.onPod || !navigator.platform.startsWith("Win")) return;
  api("/api/lan/firewall").then((d) => {     // at home on Windows: a phone can't reach a Fizgig the firewall keeps out
    if (d && d.windows !== false && !d.error && !(d.allowed && !d.blocked))
      message("Windows Firewall doesn't let your phone in yet. Allow Fizgig on your network (private networks only), " +
        "then the QR code appears here.", "Allow on my network", async () => {
          note.textContent = "Waiting for Windows (it asks for an administrator's OK)...";
          try { const r = await api("/api/lan/firewall", {}); if (r.allowed && !r.blocked) ready(); else note.textContent = "Still not allowed."; }
          catch (e) { note.textContent = e.message; }
        });
  }).catch(() => {});
}

// ---- a pod's file manager (port 8080): the sidebar button, the Start tab's upload card, Runs & outputs ---------
function fileManagerLinks(fm) {
  const how = !fm ? "" : fm.login === "none" ? "Its login isn't set up: it refuses this pod's password as too common. " +
    "Give it a less common FIZGIG_PASSWORD (on RunPod: Edit pod, then restart it; on Vast.ai: rent a new instance " +
    "with it set)" :
    "Log in as admin with this pod's password" + (fm.login === "padded" ? ", with 0s added to make it 12 characters " +
    "(the file manager refused it as too common): abc123 → abc123000000. The pod's log shows it" : "");
  $("sideFiles").style.display = fm ? "" : "none";
  $("sideFilesLogin").style.display = fm ? "" : "none";
  $("uploadFm").style.display = fm ? "" : "none";
  // the login pill above the button: admin, and the password on Show (asked of the pod only then, never kept)
  const pw = $("sideFilesPw"), show = $("sideFilesShow");
  const hide = () => { pw.textContent = "••••••"; pw.classList.remove("fm-note"); show.textContent = "Show"; };
  hide();
  show.onclick = async () => {
    if (show.textContent === "Hide") return hide();
    try {
      const d = await api("/api/pod/files_login");
      pw.textContent = d.password || (d.login === "none" ? "not set up: see the pod's log on RunPod"
        : "in the pod's log on RunPod");
      pw.classList.toggle("fm-note", !d.password);
    } catch (e) { pw.textContent = "in the pod's log on RunPod"; pw.classList.add("fm-note"); }
    show.textContent = "Hide";
  };
  for (const a of [$("sideFiles"), ...document.querySelectorAll(".fm-link")]) {
    if (fm) { a.href = fm.url; a.title = `Big files in and out, by drag and drop. ${how}.`; }
    else if (a.classList.contains("fm-text")) a.removeAttribute("href");
  }
}

// ---- Pods: rented GPUs running Fizgig, opened from here (the app window goes there; a browser opens a tab) -------
S.pods = {list: [], timer: 0, waitFor: "", pending: {}};
// Start or Stop pressed here: RunPod takes a few seconds to say so, and a reload in that gap would flick the row back
// to what it was. Each press is held for 90 s (kept by pod, as reloads make fresh rows); one RunPod hasn't moved by
// then didn't start (or stop)
const POD_HOLD_MS = 90000;
function podPending(p) {
  const k = p.id || p.url, h = S.pods.pending[k];
  if (!h) return "";
  const done = h.kind === "start" ? p.running || p.state === "ready" : h.kind === "delete" ? !p.account
    : !p.running && p.state !== "ready";
  if (done) { delete S.pods.pending[k]; return ""; }
  if (Date.now() - h.t < POD_HOLD_MS) return {start: "starting", stop: "stopping", delete: "deleting"}[h.kind];
  delete S.pods.pending[k];
  $("podsNote").textContent = h.kind === "start"
    ? `${p.name || p.id} didn't start: RunPod still has it stopped. Often no GPU is free on its machine now; its page on RunPod says why.`
    : `${p.name || p.id} hasn't ${h.kind === "delete" ? "gone" : "stopped"} yet: check it on RunPod.`;
  if (h.kind === "start" && S.pods.waitFor === p.url) S.pods.waitFor = "";
  return "";
}
const inApp = () => !!window.pywebview;
const POD_STATE = {ready: ["Ready", "ok"], waiting: ["Starting - a pod can take a few minutes. If it isn't ready after that, check it on RunPod", "warn"],
  stopped: ["Stopped", "muted"], off: ["Not answering", "muted"], starting: ["Starting...", "warn"],
  stopping: ["Stopping...", "muted"], deleting: ["Deleting...", "muted"],
  gone: ["Not on your RunPod account (another account's pod, or deleted there). Forget takes it off this list", "muted"],
  away: ["Not on your network now: is it on, awake and running Fizgig?", "muted"],
  blocked: ["Found, but it doesn't let this computer in yet: on it, Preferences > Web UI > Allow on my network", "warn"]};

async function podsLoad(quiet) {
  if (!quiet) $("podsNote").textContent = "Checking your pods...";
  let r;
  try { r = await api("/api/pods"); } catch (e) { $("podsNote").textContent = e.message; return; }
  S.pods.list = r.pods || [];
  for (const k of Object.keys(S.pods.pending))     // a deleted pod off the list altogether: its hold is done
    if (!S.pods.list.some((p) => (p.id || p.url) === k)) delete S.pods.pending[k];
  S.pods.key = r.key_saved;
  $("podKeyNote").textContent = r.key_saved ? "Key saved: your Fizgig pods are listed above." : "";
  podsDraw();
  $("podsNote").textContent = r.error ? r.error : (S.pods.list.length ? "" :
    "No pods yet. Add one below, or deploy one and it shows here once you've added your RunPod API key. Fizgig on " +
    "your other computers shows here by itself once it has a password and is allowed on your network (its Preferences > Web UI).");
  // keep checking while the tab is open: every 6 s while one is on its way up (and open it when it's ready if asked
  // to), else every 30 s with an API key saved (a pod started or stopped on RunPod shows up by itself)
  clearTimeout(S.pods.timer);
  const starting = S.pods.list.some((p) => p.state === "waiting") || Object.keys(S.pods.pending).length > 0;
  S.pods.timer = setTimeout(() => {          // the network's Fizgigs come and go too: every 30 s regardless
    if ($("tab-pods").classList.contains("active")) podsLoad(true);
  }, starting ? 6000 : 30000);
  const w = S.pods.waitFor && S.pods.list.find((p) => p.url === S.pods.waitFor);
  if (w && w.state === "ready") { S.pods.waitFor = ""; podOpen(w); }
}

function podsDraw() {
  const box = $("podsList");
  box.replaceChildren(...S.pods.list.map((p) => {
    const hold = podPending(p), st = hold || p.state;
    const [label, tone] = POD_STATE[st] || [st, "muted"];
    const title = p.name || p.id || p.url;
    const build = {mine: "Newer Fizgig there: update this computer",
      theirs: "Older Fizgig there: it opens as it is; update it when it's free",
      other: "Different builds: update both when they're free"}[p.build] || "";
    const queued = p.queued ? `${p.queued} queued` + (p.queue_held ? " (held)" : "") : "";
    const sub = [p.lan ? "On your network" : "", p.gpu, p.busy ? `training ${p.busy}` : "", queued,
      p.auto_login && inApp() ? "logs you in" : "",
      p.cost ? `$${p.cost.toFixed(2)}/hr` : "", p.id && p.id !== title ? p.id : "",
      !p.id && p.url !== title ? p.url.replace(/^https?:\/\/|\/$/g, "") : "", build].filter(Boolean).join(" · ");
    const btns = [];
    if (hold === "stopping" || hold === "deleting") { /* nothing to press while it goes */ }
    else if (p.state === "ready") btns.push(el("button", {class: "btn btn-primary btn-sm", text: inApp() ? "Open" : "Open ↗",
      onclick: () => podOpen(p)}));
    else if (p.state === "waiting" || hold === "starting") btns.push(el("button", {class: "btn btn-sm",
      text: S.pods.waitFor === p.url ? "Opens when ready..." : "Open when ready", onclick: () => {
        S.pods.waitFor = p.url; podsDraw(); }}));
    if (p.account && !p.running && !hold && S.pods.key) btns.push(el("button", {class: "btn btn-primary btn-sm",
      text: "Start and open", onclick: () => podStart(p)}));
    if (p.account && p.running && !hold && S.pods.key) btns.push(el("button", {class: "btn btn-sm", text: "Stop",
      title: "Stop the pod (it stops billing for the GPU; its files stay on its volume)", onclick: () => podStop(p)}));
    if (p.account && S.pods.key && hold !== "deleting") btns.push(el("button", {class: "btn btn-sm btn-danger",
      text: "Delete pod", title: "Delete the pod on RunPod: it and its volume disk go for good",
      onclick: () => { S.pods.confirmDel = S.pods.confirmDel === p.id ? null : p.id; podsDraw(); }}));
    if (p.saved) btns.push(el("button", {class: "btn btn-sm", text: "Forget", title: "Take it off this list (the pod itself is untouched)",
      onclick: async () => { await api("/api/pods/forget", {url: p.url}).catch(() => {}); podsLoad(true); }}));
    return el("div", {class: "pod-row"},
      el("span", {class: `pod-dot ${build ? "warn" : tone}`}),
      el("div", {class: "pod-main"}, el("div", {class: "pod-name", text: title}),
        el("div", {class: "field-hint", text: `${label}${sub ? " · " + sub : ""}`}),
        // a stopped pod still pays for its volume disk; RunPod's own page has the price (its API reads 0)
        st === "stopped" && p.account ? el("div", {class: "field-hint"}, "Its cost while stopped (its disk) is on ",
          el("a", {class: "inline-link", href: "https://console.runpod.io/pods", target: "_blank", rel: "noopener",
            text: "your RunPod account"}), ".") : null,
        // only this RunPod pod's row (a computer on your network has no id)
        p.account && p.id && S.pods.confirmDel === p.id && hold !== "deleting" ? podDeleteConfirm(p) : null),
      el("div", {class: "pod-btns"}, ...btns));
  }));
}

function podOpen(p) {
  api("/api/pods/opened", {url: p.url, name: p.name || "", lan_id: p.lan_id || ""}).catch(() => {});
  if (inApp()) location.href = p.url;                 // the window goes there; Switch to this computer comes back
  else window.open(p.url, "_blank", "noopener");
}

async function podStart(p) {
  try {
    await api("/api/pods/start", {id: p.id, gpus: p.gpus || 1});
    S.pods.pending[p.id || p.url] = {kind: "start", t: Date.now()}; S.pods.waitFor = p.url; podsDraw();
    $("podsNote").textContent = `Starting ${p.name || p.id}: it opens here when it's ready. A pod can take a few minutes to start; if it isn't ` +
      "ready after that, check it on RunPod (its Logs show what it's doing).";
    setTimeout(() => podsLoad(true), 4000);
  } catch (e) { $("podsNote").textContent = e.message; }
}

// Delete pod's confirm, in the row: what goes with it, then a second press
function podDeleteConfirm(p) {
  const name = p.name || p.id;
  return el("div", {class: "pod-confirm"},
    el("div", {}, el("b", {text: `Delete ${name}? `}), "The pod and everything on its volume disk (/workspace: LoRAs, " +
      "checkpoints, datasets, outputs) go for good. A network volume is kept."),
    p.busy ? el("div", {class: "tc-warn", text: `It's training ${p.busy}. To keep its checkpoints, use Bring back on ` +
      "its run card first."}) : null,
    el("div", {class: "pod-btns"},
      el("button", {class: "btn btn-sm btn-danger", text: "Delete it", onclick: () => podDelete(p)}),
      el("button", {class: "btn btn-sm", text: "Cancel", onclick: () => { S.pods.confirmDel = null; podsDraw(); }})));
}

async function podDelete(p) {
  S.pods.confirmDel = null;
  try {
    await api("/api/pods/delete", {id: p.id});
    S.pods.pending[p.id || p.url] = {kind: "delete", t: Date.now()};
    if (S.pods.waitFor === p.url) S.pods.waitFor = "";
    podsDraw();
    $("podsNote").textContent = `${p.name || p.id} is being deleted on RunPod.`;
    setTimeout(() => podsLoad(true), 4000);
  } catch (e) { $("podsNote").textContent = e.message; podsDraw(); }
}

async function podStop(p) {
  try {
    await api("/api/pods/stop", {id: p.id});
    S.pods.pending[p.id || p.url] = {kind: "stop", t: Date.now()};
    if (S.pods.waitFor === p.url) S.pods.waitFor = "";
    podsDraw();
    $("podsNote").textContent = `${p.name || p.id} is stopping. Its files stay on its volume for next time.`;
    setTimeout(() => podsLoad(true), 4000);
  } catch (e) { $("podsNote").textContent = e.message; }
}

// ---- Rent a pod: RunPod's cards with their live price and stock, cheapest per run first; Deploy opens RunPod's page
S.pick = {data: null, dc: ""};
async function pickLoad() {
  $("pickNote").textContent = "Asking RunPod what's free...";
  const q = new URLSearchParams({count: $("pickCount").value, ...(S.pick.dc ? {dc: S.pick.dc} : {})});
  const ask = S.pick.ask = (S.pick.ask || 0) + 1;     // only the newest answer draws (1 and 2 GPUs switched fast)
  let d;
  try { d = await api("/api/pods/gpus?" + q); }
  catch (e) {
    if (ask !== S.pick.ask) return;
    S.pick.data = null; $("pickList").replaceChildren(); $("pickNote").textContent = e.message; return;
  }
  if (ask !== S.pick.ask) return;
  S.pick.data = d;
  const vols = S.pick.data.volumes || [];
  document.querySelectorAll(".pick-live").forEach((e) => { e.hidden = !S.pick.data.key; });
  $("pickVolWrap").hidden = !vols.length;
  if (vols.length && !$("pickVol").options.length) $("pickVol").replaceChildren(el("option", {value: "", text: "Any data centre"}),
    ...vols.map((v) => el("option", {value: v.dc, text: `${v.name} (${v.dc})`})));
  pickDraw();
}

function pickDraw() {
  const d = S.pick.data;
  if (!d) return;
  const min = +$("pickVram").value, cloud = $("pickCloud").value, hours = parseFloat($("pickHours").value) || 2;
  const max = parseFloat($("pickMax").value) || Infinity;
  // two cards in one pod, the second as a helper training a share: about 1.6x one card's speed (two RTX 4090s,
  // MOSS-Image, measured 22 Oct); the price RunPod gives is the whole pod's
  const n = d.count === 2 ? 2 : 1, both = n === 2 ? 1.6 : 1, card = (g) => `${n === 2 ? "2× " : ""}${g.gpu}`;
  const speedText = (g) => (g.gpu === "RTX 5090" ? "the card these speeds are measured against" : g.speed ?
    `about ${g.speed.toFixed(g.speed < 1 ? 2 : 1)}× a 5090's speed` : "") + (n === 2 && g.speed ? " (each card)" : "");
  const deploy = (g) => el("a", {class: "btn btn-primary btn-sm", href: g.url, target: "_blank", rel: "noopener",
    text: "Deploy ↗", title: `RunPod's deploy page with ${n === 2 ? "two " + g.gpu + "s" : "the " + g.gpu} and Fizgig's template chosen`});
  const fits = d.gpus.filter((g) => g.vram >= min);
  if (!d.key) {
    $("pickList").replaceChildren(...fits.map((g) => el("div", {class: "pod-row"}, el("span", {class: "pod-dot"}),
      el("div", {class: "pod-main"}, el("div", {class: "pod-name", text: `${card(g)} · ${g.vram} GB${n === 2 ? " each" : ""}`}),
        el("div", {class: "field-hint", text: speedText(g)})), el("div", {class: "pod-btns"}, deploy(g)))));
    $("pickNote").textContent = "Add your RunPod API key above to see which cards are free right now, their prices and what a run costs on each.";
    return;
  }
  const rows = [];
  let gone = 0, dear = 0;
  for (const g of fits) {
    const offers = ["secure", "community"].filter((c) => (cloud === "any" || cloud === c) && g[c]).map((c) => ({cloud: c, ...g[c]}));
    if (!offers.length) { gone++; continue; }
    const o = offers.sort((a, b) => a.price - b.price)[0];
    if (o.price > max) { dear++; continue; }
    rows.push({g, o, run: g.speed ? o.price * hours / (g.speed * both) : null});
  }
  rows.sort((a, b) => (a.run === null) - (b.run === null) || (a.run ?? a.o.price) - (b.run ?? b.o.price));
  const shown = S.pick.all ? rows : rows.slice(0, 5);
  $("pickList").replaceChildren(...shown.map(({g, o, run}, i) => {
    const sub = [`$${o.price.toFixed(2)}/hr`, o.cloud === "secure" ? "Secure Cloud" : "Community Cloud (pick it on RunPod's page)",
      o.stock && o.stock !== "Low" ? `${o.stock.toLowerCase()} stock` : "", speedText(g),
      run !== null ? `≈ $${run.toFixed(2)} a run` : ""].filter(Boolean).join(" · ");
    return el("div", {class: "pod-row"}, el("span", {class: "pod-dot ok"}),
      el("div", {class: "pod-main"}, el("div", {class: "pod-name"}, `${card(g)} · ${g.vram} GB${n === 2 ? " each" : ""}`,
        ...(i === 0 && run !== null ? [el("span", {class: "pick-best", text: "Best value"})] : [])),
        el("div", {class: "field-hint", text: sub})), el("div", {class: "pod-btns"}, deploy(g)));
  }), ...(rows.length > 5 ? [el("button", {class: "btn btn-sm pick-more", text: S.pick.all ? "Show the best 5" :
    `Show all ${rows.length}`, onclick: () => { S.pick.all = !S.pick.all; pickDraw(); }})] : []));
  const vol = S.pick.dc ? ` Pick your volume on RunPod's page too: it's in ${S.pick.dc}.` : "";
  $("pickNote").textContent = (rows.length ? "" : "Nothing fits those choices right now. ") +
    [gone ? `${gone} more card${gone === 1 ? " has" : "s have"} none free right now` : "",
     dear ? `${dear} cost more than your limit` : ""].filter(Boolean).join("; ") + (gone || dear ? ". " : "") +
    "Speeds are a rough guide from public benchmarks; prices and stock are RunPod's, from the last minute, on hosts " +
    "that run Fizgig." + vol + (n === 2 ? " Two cards: on the pod, open Helpers > Add a helper and press Use it as a " +
    "helper beside its other GPU, then switch Training share on (about 40% off each epoch on two RTX 4090s in " +
    "testing; run costs here count that)." : "");
}

// ---- Helpers: other Fizgigs that render this one's previews (training never stops for them) or train a share of
// its images. /api/helpers: {helpers, found, out, helping}; each helper's jobs are switched on its card
S.help = {data: null, timer: 0};
const HELP_WHERE = {lan: ["On your network", "lan"], pod: ["Pod", "pod"], tailscale: ["Tailscale", "ts"],
  address: ["By address", ""], local: ["This computer", "lan"]};
const HELP_ICON = {
  previews: '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><rect x="3" y="3" width="18" height="18" rx="2"/><circle cx="9" cy="9" r="2"/><path d="m21 15-3.1-3.1a2 2 0 0 0-2.8 0L6 21"/></svg>',
  share: '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M4 7h11"/><path d="m12 4 3 3-3 3"/><path d="M20 17H9"/><path d="m12 14-3 3 3 3"/></svg>'};
const svgEl = (s) => { const t = document.createElement("template"); t.innerHTML = s; return t.content.firstChild; };

async function helpLoad(quiet) {
  if (!quiet) $("helpNote").textContent = "Checking your helpers...";
  const open = $("tab-helpers").classList.contains("active");   // open: the network is searched too (off it, nothing)
  try { S.help.data = await api("/api/helpers" + (open ? "?look=1" : "")); } catch (e) { $("helpNote").textContent = e.message; return; }
  if (!open) helpStrips();                // off the tab only its strips and the run card's line show
  else if (!S.help.typing) helpDraw();    // not while speeds are being typed: a redraw would empty the boxes
  clearTimeout(S.help.timer);           // every 3 s while one works, else every 15 s on the tab and 30 s off it
  const busy = (S.help.data.helpers || []).some((h) => h.activity) || S.help.data.helping;
  S.help.timer = setTimeout(() => helpLoad(true), busy ? 3000 : open ? 15000 : 30000);
}

// one line for what a helper is doing now, and how far along it is (0-1, or null for no bar)
function helpActivity(h) {
  const a = h.activity;
  if (h.state === "login") return [/proxy\.runpod\.net/.test(h.url || "") ?
    "Its login ran out and Fizgig couldn't find its password to log in again: Remove it, then add it again with " +
    "the password from the pod's log on RunPod" :
    "Its password changed or its login ran out: Remove it, then add it again with its password", "warn", null];
  if (h.state === "away") return ["Not answering: is it on, awake and running Fizgig?", "muted", null];
  if (h.build && h.build !== "same" && h.match && h.state === "busy") return [`Training its own run (${h.training}). ` +
    h.match, "muted", null];
  if (h.build && h.build !== "same" && h.match) return [h.match.startsWith("Switching") ? h.match + "…" :
    `Another Fizgig version there. ${h.match}`, h.match.startsWith("Switching") ? "muted" : "warn", null];
  if (h.build && h.build !== "same") return [{mine: "Newer Fizgig there: update this computer to use it",
    theirs: "Older Fizgig there: update it when it's free", other: "Different Fizgig versions: update both"}[h.build] ||
    "Different Fizgig version", "warn", null];
  if (h.state === "elsewhere") return [`Busy helping ${h.elsewhere || "another Fizgig"}`, "muted", null];
  if (h.state === "busy") return [`Training its own run (${h.training}): it helps again when that finishes`, "muted", null];
  const running = (S.help.data || {}).running;
  if (!a && (h.where === "pod" || h.where === "address") && !running && (h.previews || h.share))
    return [h.where === "pod" ? "Idle, and billed while it runs: stop the pod on the Pods tab when you're done" :
      "Idle, and billed while it runs if it's rented (Vast.ai): stop it on Vast's site when you're done", "warn", null];
  if (!a) return [!(h.previews || h.share) ? "Ready" : running ? `Ready: it takes this run's next ${h.share ?
    "round" : "preview round"}` : "Ready: it starts with this computer's next run", "", null];
  if (a.kind === "models") return [`Getting ${a.what ? a.what + " " : "models "}${a.for ? `for ${a.for} ` : ""}` +
    `(${a.got.toFixed(1)} of ${a.total.toFixed(1)} GB): the job starts when it has them`, "", a.got / a.total];
  // each step of a round says what it is (sending the LoRA, loading the model there the first time - minutes for a
  // big model - waiting its turn), not "Rendering" all through
  if (a.kind === "previews") return [(a.text ? `${a.text} · ` : "Rendering ") +
    `${a.run ? a.run + "'s " : ""}previews for epoch ${a.epoch}` + (a.queued ? ` · ${a.queued} more queued` : ""),
    "", a.progress];
  if (a.kind === "share") return [a.text || `Training its share of epoch ${a.epoch}` +
    (a.images ? ` (${a.images} images)` : ""), "", a.progress];
  if (a.kind === "check") return ["Rendering a check picture", "", a.progress];
  return [a.text || "Working", "", a.progress ?? null];
}

function helpJob(h, kind) {
  const on = !!h[kind], usable = ["ready", "busy"].includes(h.state) && (!h.build || h.build === "same") &&
    (kind !== "share" || h.share_ready !== false);
  const title = kind === "previews" ? "Previews" : "Training share";
  const what = kind === "previews" ? "Renders this computer's previews, so training doesn't stop for them." :
    "Trains its share of each epoch's images alongside this GPU, then the two combine. Switched on while a run trains, it joins at the next epoch it's ready for.";
  const stat = kind === "previews" ? h.previews_stat : h.share_stat;
  const box = el("input", {type: "checkbox", class: "switch", ...(on ? {checked: true} : {}), ...(usable ? {} : {disabled: true})});
  const job = el("label", {class: `helper-job${on ? " on" : ""}${usable ? "" : " off-limits"}`},
    el("span", {class: "helper-job-icon"}, svgEl(HELP_ICON[kind])),
    el("div", {class: "helper-job-main"},
      el("div", {class: "helper-job-title"}, title,
        ...(kind === "share" ? [el("span", {class: "helper-exp", text: "EXPERIMENTAL"})] : []), box),
      el("div", {class: "field-hint", text: what}),
      ...(stat ? [el("div", {class: "helper-job-stat", text: stat})] : []),
      ...(kind === "previews" && on && (S.help.data || {}).previews_off ?
        [el("div", {class: "helper-job-stat helper-plan warn", text: S.help.data.previews_off})] : []),
      ...(kind === "share" && h.share_how ? [el("div", {class: "helper-job-stat", text: h.share_how})] : []),
      ...(kind === "share" && h.share_plan ? [(() => {
        const good = /^(Expected[^:]*: )?worth it/i.test(h.share_plan), waiting = /^Expected: an estimate/.test(h.share_plan);
        return el("div", {class: `helper-job-stat helper-plan${good ? " good" : waiting ? "" : " warn"}`,
          text: h.share_plan + (on && !good && !waiting ? ". Switch Training share off: this run is faster on this " +
            "GPU alone." : "")});
      })()] : []),
      // a helper that connected in follows its own Combine setting: nothing here to set for it
      ...(kind === "share" && on && !h.inbound ? [helpEvery()] : []),
      ...(kind === "previews" && h.check_stat ? [el("div", {class: "helper-job-stat", text: h.check_stat})] : [])));
  box.addEventListener("change", async () => {
    try { await api("/api/helpers/set", {id: h.id, [kind]: box.checked}); }
    catch (e) { box.checked = !box.checked; $("helpNote").textContent = e.message; }
    helpLoad(true);
  });
  return job;
}

// Training share: how many epochs each side trains before the two are combined (the trainer also combines before
// any epoch that previews or saves, so those always hold both halves)
function helpEvery(onlyEvery) {
  const now = (S.help.data || {}).share_every || 3;
  const sel = el("select", {class: "w-combo-sm", title: "Fewer combines send the LoRA back and forth less often (each " +
    "costs two transfers); more often keeps the two sides closer together"},
    ...[1, 2, 3, 5].map((n) => el("option", {value: n, text: n === 1 ? "every epoch" : `every ${n} epochs`,
      ...(n === now ? {selected: true} : {})})));
  sel.addEventListener("click", (e) => e.preventDefault());
  sel.addEventListener("change", () => helpCall("/api/helpers/set", {id: "", every: Number(sel.value)}));
  const send = el("input", {type: "checkbox", ...((S.help.data || {}).share_send_cache ? {checked: true} : {})});
  send.addEventListener("change", () => helpCall("/api/helpers/set", {id: "", send_cache: send.checked}));
  const every = el("div", {class: "helper-job-stat helper-every"}, "Combine ", sel,
    el("span", {text: " and before every preview and save"}));
  if (onlyEvery) return every;         // helping another's run: its cache tick is that computer's to set
  return el("div", {}, every,
    // inside the tile's own label: the words tick this box, not the tile's switch
    el("div", {class: "helper-job-stat helper-send", title: "Off: the helper gets the training folder and makes " +
      "the cache itself (with this run's own VAE and text encoder), much less to send. On: this computer's cache " +
      "goes over instead - for a fast link", onclick: (e) => { if (e.target !== send) { e.preventDefault(); send.click(); } }},
      send, " Send this computer's cache instead (a fast link)"),
    helpTick("share_wait", "wait", " Start together (the run waits for the helper before its first epoch)",
      "On: this run's first epoch waits until the helper is ready (its model loaded, its cache made or sent), so " +
      "both sides train from the start. Off: this computer starts at once and the helper joins at the first " +
      "round once it's ready"));
}

// a tick on the share tile, inside its label (the words tick this box, not the tile's switch)
function helpTick(key, field, text, title) {
  const box = el("input", {type: "checkbox", ...((S.help.data || {})[key] ? {checked: true} : {})});
  box.addEventListener("change", () => helpCall("/api/helpers/set", {id: "", [field]: box.checked}));
  return el("div", {class: "helper-job-stat helper-send", title,
    onclick: (e) => { if (e.target !== box) { e.preventDefault(); box.click(); } }}, box, text);
}

function helpCard(h) {
  const [where, cls] = HELP_WHERE[h.where] || [h.where || "", ""];
  const [line, tone, frac] = helpActivity(h);
  const dot = h.state === "away" || h.state === "login" ? "" : h.build && h.build !== "same" ? "warn" :
    h.activity ? "busy" : "ok";
  const sub = [h.gpu, h.vram ? `${h.vram} GB` : "", h.ping_ms != null ? `${h.ping_ms} ms away` : "",
    h.url ? h.url.replace(/^https?:\/\/|\/$/g, "") : ""].filter(Boolean).join(" · ");
  const m = h.models;
  return el("div", {class: `helper-card${h.previews || h.share ? " on" : ""}`},
    el("div", {class: "helper-head"},
      el("span", {class: `pod-dot ${dot}`}),
      el("div", {class: "pod-main"},
        el("div", {class: "helper-name"}, h.name,
          ...(where ? [el("span", {class: `helper-tag ${cls}`, text: where})] : []),
          ...(h.auto ? [el("span", {class: "helper-tag", text: "Added by itself",
            title: "One of your RunPod pods: added once it started, and taken off the list when the pod is deleted. Tick a job to use it"})] : []),
          ...(h.inbound ? [el("span", {class: "helper-tag in", text: "Connected from its side",
            title: "It connected to this Fizgig (a pod can't reach a home computer, so the home computer does the connecting)"})] : [])),
        el("div", {class: "field-hint", text: sub})),
      el("div", {class: "pod-btns"},
        el("button", {class: "btn btn-sm", text: "Check picture", title: "With Previews on: this run's next preview round " +
          "renders here and on the helper, and the two are compared (the same setup gives the same picture)",
          ...(h.state === "ready" ? {} : {disabled: true}), onclick: () => helpCall("/api/helpers/check", {id: h.id})}),
        el("button", {class: "btn btn-sm", text: "Remove", title: "Stop using it as a helper (the computer itself is untouched)",
          onclick: () => helpCall("/api/helpers/remove", {id: h.id})}))),
    el("div", {class: `helper-status ${tone}`}, el("span", {text: line}),
      ...(frac != null ? [el("div", {class: "meter-bar"}, el("span", {style: `width:${Math.round(frac * 100)}%`}))] : [])),
    ...(h.inbound ? (h.link ? [el("div", {class: "helper-link field-hint", text: h.link})] : [])
      : [helpLink(h.url, h.link, h.link_mbps)]),
    el("div", {class: "helper-jobs"}, helpJob(h, "previews"), helpJob(h, "share")),
    ...(m ? [el("div", {class: "helper-models"}, el("span", {text: `Models for ${m.family}:`}),
      ...m.have.map((n) => el("span", {class: "helper-model have", text: n})),
      ...m.missing.map((n) => el("span", {class: "helper-model miss", text: n})),
      ...(m.missing.length ? [el("span", {class: "helper-model-note", text: "fetched by itself when a job needs them"}),
        el("button", {class: "btn btn-sm", text: "Get them now", title: "The helper downloads them now, from Hugging " +
          "Face, so the first job doesn't wait for them", onclick: () => helpCall("/api/helpers/fetch", {id: h.id})})] : []))] : []));
}

// a helper's link: its speeds (measured, or typed in as broadband is sold, in Mb/s), Measure again, or type them
function helpLink(url, line, mbps) {
  const open = S.help.typing === url;
  const up = el("input", {type: "number", class: "w-combo-sm", min: "1", step: "1", placeholder: "up",
    value: (mbps || {}).up || ""});
  const down = el("input", {type: "number", class: "w-combo-sm", min: "1", step: "1", placeholder: "down",
    value: (mbps || {}).down || ""});
  return el("div", {class: "helper-link"},
    el("span", {class: "field-hint", text: line || "Link: not measured yet"}),
    el("button", {class: "btn btn-sm", text: "Measure", title: "Time a few pieces each way, as a training share " +
      "sends them", onclick: () => { $("helpNote").textContent = "Measuring the link..."; helpCall("/api/helpers/link", {url}); }}),
    el("button", {class: "btn btn-sm", text: open ? "Cancel" : "Type speeds", title: "Your broadband's upload and " +
      "download in Mb/s, if the measure fails or looks wrong", onclick: () => { S.help.typing = open ? "" : url; helpDraw(); }}),
    ...(open ? [el("span", {class: "helper-link-type"}, up, el("span", {class: "field-hint", text: "up"}), down,
      el("span", {class: "field-hint", text: "down Mb/s"}), el("button", {class: "btn btn-primary btn-sm", text: "Save",
        onclick: () => { S.help.typing = ""; helpCall("/api/helpers/link", {url, up: up.value, down: down.value}); }}))] : []));
}

async function helpCall(path, body) {
  try { const r = await api(path, body); if (r && r.say) $("helpNote").textContent = r.say; }
  catch (e) { $("helpNote").textContent = e.message; }
  helpLoad(true);
}

// the Help another Fizgig with this GPU card's buttons: what they say shows in that card, kept across redraws
async function helpOutCall(body) {
  try { const r = await api("/api/helpers/help", body); S.help.outSay = (r && r.say) || ""; }
  catch (e) { S.help.outSay = e.message; }
  helpLoad(true);
}

function helpDraw() {
  const d = S.help.data || {};
  const hs = d.helpers || [];
  $("helpList").replaceChildren(...hs.map(helpCard));
  $("helpNote").textContent = hs.length ? "" : "No helpers yet. Add one in the next card: a pod, or another computer with Fizgig.";
  $("helpFound").replaceChildren(...(d.found || []).map((f) => {
    const [where] = HELP_WHERE[f.where] || [""];
    const pod = /proxy\.runpod\.net/.test(f.url || "");
    const pw = el("input", {type: "password", class: "w-combo-md", autocomplete: "off",
      placeholder: pod ? "password: leave empty" : "its password",
      title: pod ? "Fizgig finds a pod's password itself. Type it only if Fizgig says it couldn't." : ""});
    return el("div", {class: "pod-row"}, el("span", {class: "pod-dot ok"}),
      el("div", {class: "pod-main"}, el("div", {class: "pod-name", text: f.name}),
        el("div", {class: "field-hint", text: [where, f.gpu, f.vram ? `${f.vram} GB` : ""].filter(Boolean).join(" · ")})),
      // a pod whose password RunPod's settings give, or that this computer is logged in to: no box to fill
      el("div", {class: "pod-btns"}, ...(f.logged_in || f.auto_login ? [] : [pw]), el("button", {class: "btn btn-primary btn-sm",
        text: "Add as a helper", onclick: () => helpAdd(f.url, pw.value)})));
  }));
  // this computer's other GPU: a second Fizgig on it, as a helper
  const lc = d.local || {gpus: []};
  $("helpLocal").replaceChildren(...(lc.gpus || []).map((g) => {
    const on = lc.on === g.index;
    return el("div", {class: "pod-row"}, el("span", {class: `pod-dot ${on ? "ok" : ""}`}),
      el("div", {class: "pod-main"}, el("div", {class: "pod-name", text: `This computer's other GPU: ${g.name}`}),
        el("div", {class: "field-hint", text: lc.text || (on ? "In use as a helper " +
          "(below, under Your helpers)" : `GPU ${g.index}${g.vram ? ` · ${g.vram} GB` : ""}: a second Fizgig ` +
          "runs on it and joins as a helper, for previews, a training share or a moved run")})),
      el("div", {class: "pod-btns"}, on ? el("button", {class: "btn btn-sm", text: "Stop using it",
        onclick: () => helpCall("/api/helpers/local", {on: false})}) : el("button", {class: "btn btn-primary btn-sm",
        text: "Use it as a helper", ...(lc.on != null ? {disabled: true} : {}),
        onclick: () => helpCall("/api/helpers/local", {on: true, index: g.index})})));
  }));
  const out = d.out || [];
  $("helpOut").replaceChildren(...out.flatMap((o) => {
    const mine = d.helping && d.helping.url === o.url;
    const isPod = /proxy\.runpod\.net/.test(o.url || "");
    const pw = el("input", {type: "password", class: "w-combo-md", autocomplete: "off",
      placeholder: isPod ? "password: leave empty" : "its password",
      title: isPod ? "Fizgig finds a pod's password itself. Type it only if Fizgig says it couldn't." : ""});
    // its checkpoints list stays as it is across redraws (the ticks too); asked again only when it's opened
    const kept = S.help.remote === o.url && S.help.panel && S.help.panel.url === o.url;
    const panel = kept ? S.help.panel.box : el("div", {class: "helper-remote", style: S.help.remote === o.url ? "" : "display:none"});
    if (S.help.remote === o.url && !kept) { S.help.panel = {url: o.url, box: panel}; helpRemote(o, panel); }
    const linkRow = o.logged_in || mine ? [helpLink(o.url, o.link, o.link_mbps)] : [];
    if (mine && d.helping.mode === "share") linkRow.push(el("div", {class: "helper-link"}, helpEvery(true)));
    linkRow.push(helpHome(o, pw));
    return [el("div", {class: "pod-row"}, el("span", {class: `pod-dot ${mine ? "busy" : ""}`}),
      el("div", {class: "pod-main"}, el("div", {class: "pod-name", text: o.name}),
        el("div", {class: "field-hint", text: mine ? d.helping.text : o.url.replace(/^https?:\/\/|\/$/g, "")})),
      el("div", {class: "pod-btns"}, ...(mine || o.logged_in ? [] : [pw]), mine ?
        el("button", {class: "btn btn-sm", text: "Stop helping", onclick: () => helpOutCall({url: o.url, on: false})}) :
        el("button", {class: "btn btn-primary btn-sm", text: "Help it with previews",
          ...(d.helping ? {disabled: true} : {}), onclick: () => helpOutCall(
            {url: o.url, on: true, password: pw.value})}),
        el("button", {class: "btn btn-sm", text: "Help it train", title: "EXPERIMENTAL: train a share of each epoch's " +
          "images of its run on this GPU (its training folder comes here, and this computer makes the run's cache)", ...(d.helping ? {disabled: true} : {}),
          onclick: () => helpOutCall({url: o.url, on: true, password: pw.value, mode: "share"})}),
        el("button", {class: "btn btn-sm", text: "Its checkpoints…", title: "Its run's checkpoints: fetch the ones you " +
          "pick and try your own prompts on them in LoRA Royale, on this GPU (type its password first if asked)",
          onclick: () => { S.help.remotePw = pw.value; S.help.remote = S.help.remote === o.url ? "" : o.url;
            S.help.panel = null; helpDraw(); }}))),
      ...linkRow, panel];
  }));
  $("helpOutNote").textContent = [S.help.outSay || "", ...(d.home_done || []),
    out.length ? "" : "Your pods show here (the Pods tab), once one is running Fizgig."].filter(Boolean).join(". ");
  helpStrips();
  // this Fizgig is a helper right now: say so at the top, whatever it's doing
  const here = d.helping;
  $("helpHere").style.display = here ? "" : "none";
  if (here) $("helpHere").replaceChildren(el("span", {class: "pod-dot warn"}),
    el("div", {class: "pod-main"}, el("div", {class: "pod-name", text: `This GPU is helping ${here.name}`}),
      el("div", {class: "field-hint", text: here.text})),
    el("button", {class: "btn btn-sm", text: "Stop helping", onclick: () => helpCall("/api/helpers/help", {url: here.url, on: false})}));
}

// a pod's run brought home as it saves (its previews too), and the pod deleted once that's all here and checked
function helpHome(o, pw) {
  const hm = o.home || {};
  // this computer's own run, moved there: the move brings it home (the run card), not these
  const mv = S.run && S.run.move;
  if (mv && ["moved", "back"].includes(mv.state) && mv.url && mv.url.replace(/\/$/, "") === o.url.replace(/\/$/, ""))
    return el("div", {class: "helper-link helper-home"}, el("span", {class: "field-hint", text:
      `${mv.run} moved here from this computer, so its move handles it: on the run card, its checkpoints and ` +
      "previews come back as they're saved, and Bring it back can delete the pod."}));
  const tick = (text, on, body, title) => {
    const box = el("input", {type: "checkbox", ...(on ? {checked: true} : {})});
    box.addEventListener("change", () => helpCall("/api/helpers/home", {url: o.url, password: pw.value,
      ...body(box.checked)}));
    return el("label", {class: "helper-job-stat helper-send", title}, box, " " + text);
  };
  return el("div", {class: "helper-link helper-home"},
    tick("Bring its checkpoints home as they're saved", hm.on, (v) => ({on: v}), "Each checkpoint its run saves " +
      "comes to this computer's LoRA output folder (From pods), for LoRA Royale and Runs & outputs here"),
    ...(hm.on ? [tick("its previews too", hm.previews, (v) => ({previews: v}), "The run's preview pictures, into " +
      "the same folder's sample folder")] : []),
    ...(o.pod ? [tick("Delete the pod once its run has finished and they're all home", hm.delete, (v) => ({delete: v}),
      "Deletes the pod (its disk too) once the run has finished with nothing queued, and only once every " +
      "checkpoint here matches the pod's copy. If that hasn't happened 15 minutes after the run finished, the pod " +
      "is stopped instead, with its files still on it. Needs your RunPod API key (Pods tab)")] : []),
    ...(hm.on && hm.text ? [el("span", {class: "field-hint", text: hm.text})] : []));
}

// where helpers show outside their tab: a strip on Samples (previews) and Training (training share), and a line on
// the run card while a run uses them
function helpStrips() {
  const hs = ((S.help.data || {}).helpers || []).filter((h) => h.state === "ready" && (!h.build || h.build === "same"));
  const names = (l) => l.map((h) => h.name).join(" and ");
  const pv = hs.filter((h) => h.previews), sh = hs.filter((h) => h.share);
  const strip = (id, list, icon, text) => {
    $(id).style.display = list.length ? "" : "none";
    if (list.length) $(id).replaceChildren(el("span", {class: "helper-job-icon"}, svgEl(HELP_ICON[icon])),
      el("div", {class: "pod-main"}, ...text), el("button", {class: "btn btn-sm", text: "Helpers", onclick: () => showTab("helpers")}));
  };
  const pvOff = (S.help.data || {}).previews_off;
  strip("samplesHelper", pv, "previews", pvOff ? [pvOff] : [el("b", {text: names(pv)}), ` render${pv.length === 1 ? "s" : ""} this ` +
    "computer's previews, so training doesn't stop for them. Everything below applies there as it does here."]);
  const off = (S.help.data || {}).share_off;
  strip("trainHelper", sh, "share", off ? [off] : [el("b", {text: names(sh)}), ` train${sh.length === 1 ? "s" : ""} ` +
    "a share of each epoch's images alongside this GPU. ", el("span", {class: "helper-exp", text: "EXPERIMENTAL"})]);
  const live = hs.filter((h) => h.activity && (h.activity.kind === "previews" || h.activity.kind === "share"));
  $("runHelper").style.display = live.length ? "" : "none";
  if (live.length) $("runHelper").replaceChildren(...live.map((h) => el("span", {},
    el("span", {class: "pod-dot busy"}), " ", el("b", {text: h.name}), " " + helpActivity(h)[0].replace(/^R/, "r"))));
}

// another computer's run: its checkpoints, the ones you tick fetched here and opened in LoRA Royale (this GPU)
async function helpRemote(o, box) {
  box.replaceChildren(el("div", {class: "field-hint", text: "Asking it for its run's checkpoints..."}));
  let r;
  try { r = await api("/api/helpers/remote", {url: o.url, password: S.help.remotePw || ""}); S.help.remotePw = ""; }
  catch (e) { box.replaceChildren(el("div", {class: "field-hint", text: e.message})); return; }
  if (!r.items.length) { box.replaceChildren(el("div", {class: "field-hint", text: r.run ?
    `${r.run} hasn't saved a checkpoint yet.` : "Nothing training there."})); return; }
  const mb = (n) => n >= 1e9 ? `${(n / 1e9).toFixed(1)} GB` : `${Math.round(n / 1e6)} MB`;
  const epoch = (n) => (n.match(/-(\d{6})\.safetensors$/) || [])[1];
  const ticks = r.items.map((it, i) => el("input", {type: "checkbox", ...(i < 4 ? {checked: true} : {})}));
  const note = el("div", {class: "field-hint"});
  const go = el("button", {class: "btn btn-primary btn-sm", text: "Get them and open LoRA Royale", onclick: async () => {
    const items = r.items.filter((_it, i) => ticks[i].checked);
    if (!items.length) { note.textContent = "Tick at least one."; return; }
    go.disabled = true;
    try {
      const g = await api("/api/helpers/remote_get", {url: o.url, items, run: r.run, family: r.family});
      const poll = async () => {
        const st = await api("/api/helpers/remote_status").catch(() => ({}));
        if (st.state === "running") {
          note.textContent = `Fetching ${st.name || ""}: ${mb(st.done)} of ${mb(st.total)}`;
          return setTimeout(poll, 1000);
        }
        go.disabled = false;
        if (st.state !== "done") { note.textContent = st.error || "The fetch stopped."; return; }
        note.textContent = `Here: ${g.folder}` + (st.te_note ? `. ${st.te_note}.` : "");
        showTab("royale");
        await royaleOpen();
        $("ryFolder").value = g.folder;
        if (r.family && $("ryFamily").value !== r.family) await royaleFamily(r.family);
        royaleScan();
      };
      poll();
    } catch (e) { note.textContent = e.message; go.disabled = false; }
  }});
  box.replaceChildren(el("div", {class: "field-hint", text: `${r.run}'s checkpoints, newest first. The ones you tick ` +
    "come to this computer, and LoRA Royale renders them here on your prompts (with this computer's own text encoder)."}),
    el("div", {class: "helper-remote-list"}, ...r.items.map((it, i) => el("label", {class: "checkbox-label"}, ticks[i],
      el("span", {text: (epoch(it.name) ? `Epoch ${Number(epoch(it.name))}` : it.name === `${r.run}.safetensors` ?
        "Final LoRA" : it.name) + ` · ${mb(it.size)}`})))),
    el("div", {class: "field-row"}, go), note);
}

async function helpAdd(url, password) {
  $("helpAddNote").textContent = "Connecting...";
  try {
    const r = await api("/api/helpers/add", {url, password});
    $("helpAddUrl").value = ""; $("helpAddPw").value = "";
    $("helpAddNote").textContent = r.say || "Added.";
  } catch (e) { $("helpAddNote").textContent = e.message; }
  helpLoad(true);
}

function wireHelpers() {
  $("helpReload").addEventListener("click", () => helpLoad());
  $("helpAdd").addEventListener("click", () => helpAdd($("helpAddUrl").value, $("helpAddPw").value));
  $("helpAddUrl").addEventListener("keydown", (e) => { if (e.key === "Enter") $("helpAddPw").focus(); });
  $("helpAddPw").addEventListener("keydown", (e) => { if (e.key === "Enter") helpAdd($("helpAddUrl").value, $("helpAddPw").value); });
}

$("tabsToggle").addEventListener("click", () =>
  tabsMenu(!$("tabsToggle").closest(".tab-strip-row").classList.contains("tabs-open")));

function wirePods() {
  $("podsReload").addEventListener("click", () => podsLoad());
  for (const id of ["pickVram", "pickCloud", "pickMax", "pickHours"]) $(id).addEventListener("input", pickDraw);
  $("pickCount").addEventListener("change", pickLoad);
  $("pickVol").addEventListener("change", () => { S.pick.dc = $("pickVol").value; pickLoad(); });
  const add = async () => {
    try {
      const r = await api("/api/pods/add", {text: $("podAddText").value, name: $("podAddName").value});
      $("podAddText").value = ""; $("podAddName").value = "";
      $("podAddNote").textContent = r.state === "ready" ? "Added. It's ready: press Open." :
        r.state === "waiting" ? "Added. Fizgig on it isn't answering yet (still starting): it shows Ready when it is." :
        "Added, but nothing answers there yet. A stopped pod needs starting on RunPod (or here, with your API key).";
      podsLoad(true);
    } catch (e) { $("podAddNote").textContent = e.message; }
  };
  $("podAdd").addEventListener("click", add);
  $("podAddText").addEventListener("keydown", (e) => { if (e.key === "Enter") add(); });
  $("podKeySave").addEventListener("click", async () => {
    $("podKeyNote").textContent = "Checking the key with RunPod...";
    try {
      const r = await api("/api/pods/key", {key: $("podKey").value});
      $("podKey").value = "";
      $("podKeyNote").textContent = `Key saved. ${r.found} Fizgig pod${r.found === 1 ? "" : "s"} on your account.`;
      podsLoad(true); pickLoad();
    } catch (e) { $("podKeyNote").textContent = e.message; }
  });
  $("podKeyClear").addEventListener("click", async () => {
    await api("/api/pods/key", {key: ""}).catch(() => {});
    $("podKeyNote").textContent = "Key removed."; podsLoad(true); S.pick.dc = ""; $("pickVol").replaceChildren(); pickLoad();
  });
}

// ---- About: the version, how this install updates, and a newer release when there is one (the desktop's) -------
function showAbout(a) {
  $("aboutVersion").textContent = "Version " + a.version + (a.commit ? ` · build ${a.commit.slice(0, 7)}` : "");
  $("aboutUpdate").textContent = a.update;
  const pv = a.pod_version;                // a pod kept on the version of the computer using it (pod_version.py)
  if (pv && pv.pinned && pv.pinned === a.commit) {
    const say = el("span", {class: "field-hint"});
    $("aboutUpdate").replaceChildren(`This pod runs the same Fizgig as ${pv.by || "the computer that last used it"}, so ` +
      "the two work together, and keeps it at every start. ", el("button", {class: "btn btn-sm", text: "Use the newest Fizgig",
      title: "Start this pod's Fizgig again on the newest version (the computer switches it back when it next opens it)",
      onclick: async () => {
        try {
          await api("/api/pod/newest", {});
          say.textContent = " Starting again on the newest Fizgig: the page reloads when it's back.";
          for (let i = 0; i < 200; i++) {
            await new Promise((res) => setTimeout(res, 3000));
            const r = await fetch("/api/about", {headers: {"X-Fizgig-Token": TOKEN}}).catch(() => null);
            if (r && r.status === 403) return location.reload();     // started again: a new page token, a new page
            const b = r && r.ok ? await r.json().catch(() => null) : null;
            if (b && b.commit !== a.commit) return location.reload();
            if (b && b.pod_version && b.pod_version.state === "failed") { say.textContent = " " + b.pod_version.error; return; }
          }
        } catch (e) { say.textContent = " " + e.message; }
      }}), say);
  }
  const r = a.release || {}, newer = r.status === "update_available";
  $("aboutRelease").replaceChildren(...(newer ? [el("div", {class: "about-banner"},
    el("strong", {text: `⬆ Fizgig ${r.tag} is available`}),
    el("div", {class: "field-hint", text: `You're on ${a.version}. ${a.update}`}),
    el("a", {href: "https://github.com/shootthesound/Fizgig/releases/latest", target: "_blank", rel: "noopener",
      text: "📖 Read the release notes"}))] : []));
  $("aboutBtn").classList.toggle("update", newer);
  $("aboutBtn").querySelector("span").textContent = newer ? "⬆ Update Available" : "About";
}

// ---- login -------------------------------------------------------------------------------------------------
function showLogin() { $("loginModal").classList.add("active"); $("loginPassword").focus(); }
async function login() {
  const r = await fetch("/api/login", {method: "POST", headers: {"X-Fizgig-Token": TOKEN, "Content-Type": "application/json"},
    body: JSON.stringify({password: $("loginPassword").value, remember: $("loginRemember").checked})});
  if (r.ok) { $("loginModal").classList.remove("active"); boot(); }
  else $("loginError").textContent = (await r.json().catch(() => ({}))).error || "Wrong password.";
}

// ---- tabs and wiring ---------------------------------------------------------------------------------------
// a phone's menu button: open or fold the tabs (a desktop never shows the button)
function tabsMenu(open) {
  const row = $("tabsToggle").closest(".tab-strip-row");
  row.classList.toggle("tabs-open", open);
  $("tabsToggle").setAttribute("aria-expanded", open ? "true" : "false");
}

function showTab(name) {
  if (name !== "prep" && S.me && S.me.dirty) meSave();      // leaving Image Prep saves the mask being edited
  const tb = document.querySelector(`.tab-item[data-tab="${name}"] span`);
  if (tb) $("tabsToggleName").textContent = tb.textContent;
  tabsMenu(false);
  document.querySelectorAll(".tab-item").forEach((b) => {
    b.classList.toggle("active", b.dataset.tab === name);
    if (b.dataset.tab === name) b.scrollIntoView({block: "nearest", inline: "nearest"});
  });
  document.querySelectorAll(".tab-panel").forEach((p) => p.classList.toggle("active", p.id === "tab-" + name));
  if (name === "captions") loadCaptions();
  if (name === "prep") loadPrep();
  if (name === "studio") studioOpen();
  if (name === "extract") extractOpen();
  if (name === "profiler") profilerOpen();
  if (name === "royale") royaleOpen();
  if (name === "refmod") refmodOpen();
  if (name === "metadata") mdWire();
  if (name === "samples") loadGallery();
  if (name === "training") loadCheckpoints();
  if (name === "outputs") { loadRuns(); if (S.outputsOpen) loadOutputs(); }
  if (S.pods.confirmDel) { S.pods.confirmDel = null; podsDraw(); }   // a Delete pod left open: gone, not waiting
  if (name === "pods") { podsLoad(); pickLoad(); }
  if (name === "helpers") helpLoad();
  // the workbench's render choice follows the Samples tab, which may have changed since the tab last drew it
  if (["studio", "royale", "profiler"].includes(name)) renderChoiceRefresh();
  api("/api/state", {tab: name}).catch(() => {});
}

// a dropped / chosen folder: every file inside it, subfolders included (each still goes up on its own)
function collect(entry, out) {
  if (entry.isFile) return new Promise((res) => entry.file((f) => { out.push(f); res(); }, () => res()));
  const reader = entry.createReader();
  return new Promise((res) => {
    const batch = () => reader.readEntries(async (list) => {          // readEntries returns at most ~100 a call
      if (!list.length) return res();
      for (const en of list) await collect(en, out);
      batch();
    }, () => res());
    batch();
  });
}
// skip what no family trains on and the system's own files, so a folder upload doesn't list them as failures
const UPLOAD_OK = /\.(png|jpe?g|webp|bmp|avif|mp4|wav|mp3|flac|m4a|txt|caption|json|zip)$/i;
const SYSTEM_FILE = (n) => n.startsWith(".") || ["thumbs.db", "desktop.ini"].includes(n.toLowerCase());
const OTHER_VIDEO = /\.(mov|webm|mkv|avi|m4v|wmv)$/i;
// what goes up: the system's own files are skipped quietly; other files are skipped and named, with a convert-to
// hint for video Fizgig doesn't read
function sortUpload(files) {
  const keep = [], video = [], other = [];
  for (const f of files) {
    if (SYSTEM_FILE(f.name)) continue;
    if (UPLOAD_OK.test(f.name)) keep.push(f);
    else (OTHER_VIDEO.test(f.name) ? video : other).push(f.name);
  }
  return {keep, video, other};
}
// what a file can't be called on every system, replaced the same way the server does for a zip (Act 1: Intro.wav ->
// Act 1_ Intro.wav), so a song and its caption files still pair up
const tidyName = (n) => n.replace(/[<>:"|?*\x00-\x1f]/g, "_").trim().replace(/[. ]+$/, "");
// a dropped folder names the new one, unless a name is typed already
function nameFromFolder(name) {
  const box = $("uploadName");
  if (name && !box.value.trim()) box.value = name.replace(/[\/\\<>:"|?*]/g, "_").replace(/^[.\s]+|[.\s]+$/g, "");
}

// a file in 8 MB pieces: asks what is already there first (an earlier, dropped upload resumes; a file already there
// is skipped), retries a piece after a drop, and reports the bytes sent. Returns {ok, data, skipped}
const PIECE = 8 * 2 ** 20;
async function uploadResumable(params, file, onProgress) {
  params = {...params, mtime: (file.lastModified || 0) / 1000};       // same name, size AND time = already there
  const q = (o) => Object.entries(o).map(([k, v]) => `${k}=${encodeURIComponent(v)}`).join("&");
  let st;
  try { st = await api(`/api/upload/status?${q({...params, size: file.size})}`); }
  catch (e) { return {ok: false, data: {error: e.message}}; }
  if (st.done) {
    onProgress(file.size);
    return {ok: true, skipped: true, data: {saved: [], skipped: [], replaced: [], path: st.path, folder: st.folder}};
  }
  let offset = st.have || 0, tries = 0, last = null;
  while (true) {
    const end = Math.min(file.size, offset + PIECE);
    const piece = file.read ? await file.read(offset, end) : file.slice(offset, end);
    try {
      const r = await sendFile(`/api/upload/chunk?${q({...params, offset, total: file.size})}`, piece,
        (sent) => onProgress(offset + sent));
      if (!r.ok) return r;
      last = r.data;
      if (last.resync !== undefined) { offset = last.resync; continue; }      // the server says where it is
      tries = 0;
      if (last.have === undefined) return {ok: true, data: last};             // the last piece: finished
      offset = last.have;
    } catch (e) {                                                             // the connection dropped: wait, carry on
      if (++tries > 6) return {ok: false, data: {error: `${file.name}: the connection kept dropping (` +
        `${Math.round(100 * offset / file.size)}% sent). Upload again to carry on from there.`}};
      onProgress(offset, `connection dropped, retrying (${tries})…`);
      await new Promise((r) => setTimeout(r, 1500 * tries));
      try { offset = (await api(`/api/upload/status?${q({...params, size: file.size})}`)).have || 0; } catch (e2) { /* still down */ }
    }
  }
}

// ---- From this computer: a pod's page in the Fizgig app window lists the training folders on the computer the
// window runs on (app_window.py's Api, read-only, training folders only) and sends one up with the upload above
const appBridge = () => (window.pywebview && window.pywebview.api && window.pywebview.api.local_datasets) ?
  window.pywebview.api : null;
const pageIsElsewhere = () => !["127.0.0.1", "localhost", "::1", "[::1]"].includes(location.hostname);

S.podLocal = {list: [], root: "", busy: false};
const PODLOCAL_SORT = {
  new: (a, b) => (b.time || 0) - (a.time || 0), old: (a, b) => (a.time || 0) - (b.time || 0),
  az: (a, b) => a.name.localeCompare(b.name, undefined, {numeric: true, sensitivity: "base"}),
  za: (a, b) => b.name.localeCompare(a.name, undefined, {numeric: true, sensitivity: "base"}),
  big: (a, b) => (b.bytes || 0) - (a.bytes || 0),
  many: (a, b) => (b.pictures + b.clips + b.songs) - (a.pictures + a.clips + a.songs)};

async function podLocalInit() {
  const api_ = appBridge();
  if (!api_ || !pageIsElsewhere()) return;
  $("podLocalCard").style.display = "";
  const sort = $("podLocalSort");
  try { const s = localStorage.getItem("fizgig.podLocalSort"); if (PODLOCAL_SORT[s]) sort.value = s; } catch (e) { /* no storage */ }
  sort.onchange = () => { try { localStorage.setItem("fizgig.podLocalSort", sort.value); } catch (e) { /* no storage */ } podLocalDraw(); };
  $("podLocalReload").onclick = podLocalLoad;
  podLocalLoad();
}

async function podLocalLoad() {
  const btn = $("podLocalReload");
  btn.disabled = true;
  if (!S.podLocal.busy) $("podLocalStatus").textContent = "Looking at this computer's training folders...";
  try {
    const r = await appBridge().local_datasets();
    S.podLocal.list = r.datasets || [];
    S.podLocal.root = r.root || "";
    if (!S.podLocal.busy) $("podLocalStatus").textContent = "";
  } catch (e) { $("podLocalStatus").textContent = String(e.message || e); }
  btn.disabled = false;
  podLocalDraw();
}

function podLocalDraw() {
  const host = $("podLocalGrid");
  const mb = (n) => n >= 2 ** 30 ? (n / 2 ** 30).toFixed(1) + " GB" : Math.max(1, Math.round(n / 2 ** 20)) + " MB";
  const list = [...S.podLocal.list].sort(PODLOCAL_SORT[$("podLocalSort").value] || PODLOCAL_SORT.new);
  host.replaceChildren(...list.map((d) => {
    const total = d.pictures + d.clips + d.songs;
    const what = [d.pictures && `${d.pictures} picture${d.pictures === 1 ? "" : "s"}`,
      d.clips && `${d.clips} clip${d.clips === 1 ? "" : "s"}`, d.songs && `${d.songs} song${d.songs === 1 ? "" : "s"}`]
      .filter(Boolean).join(" · ");
    const send = el("button", {class: "btn btn-sm btn-primary", text: S.onPod ? "Send to this pod" :
      `Send to ${S.hostName || "your other computer"}`, disabled: S.podLocal.busy,
      onclick: () => podLocalSend(d)});
    return el("div", {class: "dataset-card", title: d.path},
      el("div", {class: "dataset-mosaic n" + Math.min(4, d.thumbs.length)},
        d.thumbs.length ? d.thumbs.map((src) => el("img", {alt: "", src})) :
          el("span", {class: "dataset-kind", text: d.songs ? "♪" : d.clips ? "▶" : ""})),
      el("div", {class: "dataset-meta"}, el("b", {text: d.name}), el("span", {text: `${what} · ${mb(d.bytes)}`}),
        d.sub_items ? el("span", {class: "dataset-subs", text: `+ ${d.sub_items} in ${d.sub_folders} sub-folder` +
          (d.sub_folders === 1 ? "" : "s") + ((S.values || {}).TRAIN_SUBFOLDERS ? " (sent too)" : " (not sent)"),
          title: "Sent only when Also train on its sub-folders is ticked on the Training tab"}) : null,
        total ? el("span", {class: d.captioned >= total ? "ok" : "warn",
          text: d.captioned >= total ? "all captioned" : `${d.captioned} of ${total} captioned`}) : null,
        el("div", {class: "dataset-actions"}, send)));
  }));
  if (!list.length && !S.podLocal.busy && !$("podLocalStatus").textContent)
    $("podLocalStatus").textContent = `No training folders on this computer yet (${S.podLocal.root}).`;
}

async function podLocalSend(d) {
  const api_ = appBridge();
  const say = (t, bad) => { $("podLocalStatus").textContent = t; $("podLocalStatus").classList.toggle("bad", !!bad); };
  S.podLocal.busy = true;                     // one send at a time: every Send button waits, through a Refresh too
  podLocalDraw();
  try {
    // its sub-folders go too when Also train on its sub-folders is ticked here (uploadFiles keeps them by rel)
    const files = await api_.local_files(d.path, !!(S.values || {}).TRAIN_SUBFOLDERS);
    const CHUNK = 2 * 2 ** 20;                 // what one bridge read carries
    const local = files.map((f) => ({name: f.name.split("/").pop(), rel: d.name + "/" + f.name, size: f.size,
      lastModified: f.mtime * 1000,
      async read(a, b) {
        const parts = [];
        for (let o = a; o < b; o += CHUNK) {
          const b64 = await api_.read_local(d.path, f.name, o, Math.min(CHUNK, b - o));
          parts.push(Uint8Array.from(atob(b64), (c) => c.charCodeAt(0)));
        }
        return new Blob(parts);
      }}));
    await uploadFiles(local, {name: d.name, say});
  } catch (e) { say(String(e.message || e), true); }
  S.podLocal.busy = false;
  podLocalDraw();
}
window.addEventListener("pywebviewready", podLocalInit);    // the window's bridge arrives after the page loads
// tell the app window the page runs in it (app_window.py gives a white window up for the browser otherwise)
window.addEventListener("pywebviewready", () => {
  const a = window.pywebview && window.pywebview.api;
  if (a && a.ready) a.ready();
});

// the app window and a pod must run the same build (app_window.py blocks the page when they don't). A window from
// before that check has no build() to ask: it is older than this pod, so the page says to update this computer
function podBuildGuard() {
  const a = window.pywebview && window.pywebview.api;
  if (!a || !pageIsElsewhere() || a.build || $("fizgigBuildBlock")) return;
  const o = el("div", {id: "fizgigBuildBlock", class: "build-block"},
    el("div", {class: "build-block-card"},
      el("div", {class: "build-block-title", text: S.onPod ? "This computer and the pod run different builds of Fizgig" :
        "Your two computers run different builds of Fizgig"}),
      el("p", {text: `Fizgig on this computer is older than ${S.onPod ? "the pod's" : "the one on " + (S.hostName || "your other computer")}. Update it - close Fizgig and run update_fizgig.bat ` +
        "(update_fizgig_rocm.bat on AMD; venv/bin/python update_fizgig.py on Linux or a Mac), then open it again."}),
      el("div", {class: "field-row"}, a.home ? el("button", {class: "btn", text: "Switch to this computer", onclick: () => a.home()}) : null,
        el("button", {class: "btn", text: "Open anyway", title: "Use it as it is: most of it works, something newer than this computer's Fizgig may not",
          onclick: () => o.remove()}))));
  document.body.append(o);
}
window.addEventListener("pywebviewready", podBuildGuard);

// one file as the request body, reporting bytes sent (fetch can't report upload progress)
function sendFile(url, file, onProgress) {
  return new Promise((resolve, reject) => {
    const x = new XMLHttpRequest();
    x.open("POST", url);
    x.setRequestHeader("X-Fizgig-Token", TOKEN);
    x.upload.onprogress = (e) => onProgress(e.loaded);
    x.onload = () => { let data = {}; try { data = JSON.parse(x.responseText); } catch (e) { /* not JSON */ }
      resolve({ok: x.status >= 200 && x.status < 300, data}); };
    x.onerror = () => reject(new Error("the connection dropped"));
    x.send(file);
  });
}

// ---- upload a training folder from this computer ------------------------------------------------------------
// opts: name (the new folder's), say(text, bad), done(folder) - by default the Start tab's boxes and training folder
async function uploadFiles(picked, opts = {}) {
  const {keep: files, video, other} = sortUpload(picked);
  const notes = [];
  if (video.length) notes.push(`Left out ${video.length} video file${video.length === 1 ? "" : "s"} Fizgig can't ` +
    `train on (${video.slice(0, 3).join(", ")}${video.length > 3 ? ", …" : ""}): convert them to .mp4 first.`);
  if (other.length) notes.push(`Left out (not pictures, clips, songs, captions or zips): ` +
    `${other.slice(0, 5).join(", ")}${other.length > 5 ? ", …" : ""}.`);
  // its sub-folders are kept only when the Training tab's Also train on its sub-folders is ticked; otherwise every
  // file goes into the one folder, as before
  const keepSubs = !!(S.values || {}).TRAIN_SUBFOLDERS;
  const sub = (f) => {
    const parts = String(f.webkitRelativePath || f.rel || "").split("/").slice(1, -1).map(tidyName).filter(Boolean);
    // its detail crops keep their folders whatever the tick: each size's crops share their pictures' names
    const i = parts.indexOf("fizgig_detail_crops");
    if (i >= 0 && parts.length === i + 2) return parts.slice(i).join("/");
    if (parts.length === 1 && parts[0] === "detail_masters") return "detail_masters";     // Train on the crops only: their source
    return keepSubs ? parts.join("/") : "";
  };
  const seen = new Map();
  for (const f of files) { const k = (sub(f) ? sub(f) + "/" : "") + f.name; seen.set(k, (seen.get(k) || 0) + 1); }
  const dupes = [...seen].filter(([, n]) => n > 1).map(([k]) => k);
  if (dupes.length) notes.push(`More than one file is named ${dupes.slice(0, 3).join(", ")}` +
    `${dupes.length > 3 ? ", …" : ""} (from different subfolders); they all go into one folder, so the last one is kept. ` +
    "To keep the sub-folders, tick Also train on its sub-folders on the Training tab and upload again.");
  else if (keepSubs && files.some((f) => sub(f))) notes.push("Its sub-folders are kept, as Also train on its " +
    "sub-folders is ticked on the Training tab.");
  const name = opts.name !== undefined ? opts.name : $("uploadName").value.trim();
  const say = opts.say || ((t, bad) => { $("uploadStatus").textContent = t; $("uploadStatus").classList.toggle("bad", !!bad); });
  if (!name) return say("Type a name for the new folder first.", true);
  if (!files.length) return say(["Nothing to upload: no pictures, .mp4 clips, songs, captions or zips in there."]
    .concat(notes).join(" "), true);
  let folder = "", saved = 0, already = 0, doneBytes = 0;
  const skipped = [], failed = [], replaced = [], badname = [];
  const allBytes = files.reduce((a, f) => a + f.size, 0);
  const mb = (n) => n >= 2 ** 30 ? (n / 2 ** 30).toFixed(2) + " GB" : (n / 2 ** 20).toFixed(n < 10 * 2 ** 20 ? 1 : 0) + " MB";
  const listTick = setInterval(() => loadDatasets(), 4000);   // the folder list above fills in as the files land
  // three files at once: each file costs round trips through a pod's proxy (is it there? then its pieces), so one at a
  // time spent much of its time waiting - several together keep the line busy
  const queue = [...files], inFlight = new Map();       // file -> bytes sent so far
  let done = 0;
  const show = (f, note) => {
    const sentNow = doneBytes + [...inFlight.values()].reduce((x, y) => x + y, 0);
    say(`Uploading ${files.length} file${files.length === 1 ? "" : "s"}: ${done} done, ${inFlight.size} going` +
      (f ? ` (${f.name}${note ? ", " + note : ""})` : "") +
      ` · ${Math.round(100 * sentNow / Math.max(1, allBytes))}% of ${mb(allBytes)}`);
  };
  const worker = async () => {
    for (let f; (f = queue.shift());) {
      inFlight.set(f, 0);
      show(f);
      try {
        const r = await uploadResumable({folder: sub(f) ? name + "/" + sub(f) : name, name: tidyName(f.name)}, f, (sent, note) => {
          inFlight.set(f, sent);
          show(f, note || (sent >= f.size && f.name.toLowerCase().endsWith(".zip") ? "unpacking…" : ""));
        });
        const d = r.data;
        if (r.skipped) { already++; folder = folder || d.folder; }
        else if (!r.ok) failed.push(d.error || f.name);
        else {
          folder = d.folder; saved += d.saved.length; skipped.push(...d.skipped); replaced.push(...(d.replaced || []));
          badname.push(...(d.badname || []));
        }
      } catch (e) { failed.push(`${f.name}: ${e.message}`); }
      inFlight.delete(f);
      doneBytes += f.size; done++;
      show();
    }
  };
  await Promise.all(Array.from({length: Math.min(3, files.length)}, worker));

  clearInterval(listTick);
  let msg = saved || already ? `${saved} file${saved === 1 ? "" : "s"} uploaded` +
    (already ? `, ${already} already there (same size, skipped)` : "") + ` in ${folder}.` +
    (opts.done ? " It is filled in above." : " That is now the training folder.") : "Nothing was saved.";
  if (!saved && already) saved = already;           // all already there: still the folder to use
  if (skipped.length) msg += ` Left out (not pictures, clips, songs or captions): ${skipped.join(", ")}.`;
  if (replaced.length) msg += ` Replaced files that were already there with the same name: ` +
    `${replaced.slice(0, 5).join(", ")}${replaced.length > 5 ? ", …" : ""}.`;
  if (notes.length) msg += " " + notes.join(" ");
  if (badname.length) msg += ` Left out of the zip, as their names can't be stored here: ${badname.slice(0, 5).join(", ")}` +
    `${badname.length > 5 ? ", …" : ""}. Rename them (no leading dot, no slashes) and upload them again.`;
  if (failed.length) msg += ` Not uploaded: ${failed.join("; ")}`;
  say(msg, !saved || failed.length);
  if (saved) loadDatasets();
  if (saved && opts.done) opts.done(folder);
  else if (saved) {
    $("imageFolder").value = folder;
    $("imageFolder").dispatchEvent(new Event("change"));
  }
}

// an Upload button beside a form box whose file or folder may be on this computer, not the one training
function uploadButton(r, input, set) {
  const status = el("div", {class: "field-hint upload-status"});
  const say = (t, bad) => { status.textContent = t; status.classList.toggle("bad", !!bad); };
  const picker = r.kind === "folder" ? el("input", {type: "file", webkitdirectory: true, hidden: true}) :
    el("input", {type: "file", hidden: true, accept: r.accept || ".safetensors,image/*,.mp4,audio/*,.m4a"});
  picker.addEventListener("change", async () => {
    const files = [...picker.files];
    picker.value = "";
    if (!files.length) return;
    if (r.kind === "folder") {
      const top = (files[0].webkitRelativePath || "").split("/")[0];
      const name = tidyName(top).replace(/^[.\s]+/, "") || r.key.toLowerCase();
      return uploadFiles(files, {name, say, done: (folder) => { input.value = folder; set(folder); }});
    }
    const f = files[0];
    const mb = (n) => (n / 2 ** 20).toFixed(n < 10 * 2 ** 20 ? 1 : 0);
    try {
      const res = await uploadResumable({single: 1, folder: "", name: tidyName(f.name)}, f,
        (sent, note) => say(`Uploading ${f.name}: ${mb(sent)} of ${mb(f.size)} MB` + (note ? ` (${note})` : "")));
      if (!res.ok) return say(res.data.error || "Upload failed.", true);
      if (res.skipped) {                     // the same file is already there: use it, nothing to send
        input.value = res.data.path;
        set(res.data.path);
        return say(`Already uploaded (same size): ${res.data.path}`);
      }
      input.value = res.data.path;
      set(res.data.path);
      say(`Uploaded to ${res.data.path}` + ((res.data.replaced || []).length ? " (replacing the one there)." : "."));
    } catch (e) { say(`${f.name}: ${e.message}`, true); }
  });
  const btn = el("label", {class: "btn", title: "From the device you're using now, sent to the machine running Fizgig. On the computer running Fizgig itself there's no need: Choose… picks it directly."},
    r.kind === "folder" ? "Upload a folder…" : "Upload…", picker);
  return {btn, status};
}

let wired = false;
// ---- the picture viewer: any preview or Captions picture, big, with the others in its list a step away --------
const LIGHTBOX_PICS = ".gallery-item > img, .caption-card > img";      // (Repair Studio's panes are gallery items)
S.lb = {list: [], at: 0};
// the caption shown under a Captions picture: an edit goes back to its card and is saved before the view moves on
async function lightboxCapFlush() {
  const pic = S.lb.list[S.lb.at], cap = pic && pic.cap;
  if (!cap || $("lightboxCap").style.display === "none" || $("lightboxCapText").value === cap.ta.value) return true;
  cap.ta.value = $("lightboxCapText").value;
  try { await cap.save(); $("lightboxCapNote").textContent = "Saved"; return true; }
  catch (e) { $("lightboxCapNote").textContent = e.message; return false; }
}
async function lightboxStep(i) {
  if (!S.lb.list[i] || !(await lightboxCapFlush())) return;
  lightboxShow(i);
}
function lightboxShow(i) {
  const pic = S.lb.list[i];
  if (!pic) return;
  S.lb.at = i;
  $("lightboxCap").style.display = pic.cap ? "" : "none";
  $("lightbox").classList.toggle("with-cap", !!pic.cap);
  if (pic.cap) {
    $("lightboxCapRecap").style.display = pic.cap.recap ? "" : "none";
    $("lightboxCapSpeech").style.display = pic.cap.speech ? "" : "none";
    $("lightboxCapText").value = pic.cap.ta.value;
    $("lightboxCapNote").textContent = "Changes save when you move to the next picture or close.";
  }
  $("lightboxImg").src = pic.src;
  $("lightboxName").textContent = pic.name;
  $("lightboxMode").textContent = pic.mode || "";
  $("lightboxMode").style.display = pic.mode ? "" : "none";
  $("lightboxPrev").style.visibility = i > 0 ? "" : "hidden";
  $("lightboxNext").style.visibility = i < S.lb.list.length - 1 ? "" : "hidden";
  $("lightbox").style.display = "";
}
async function lightboxClose() {
  if (!(await lightboxCapFlush())) return;
  $("lightbox").style.display = "none"; $("lightboxImg").src = "";
}
// a tile's thumbnail (/api/thumb, small and squashed) opens big as the picture itself (/api/media), as the Start tab's
// dataset pictures do
function fullPicture(src) {
  try {
    const u = new URL(src, location.href);
    if (u.pathname === "/api/thumb" && u.searchParams.get("path"))
      return `/api/media?path=${encodeURIComponent(u.searchParams.get("path"))}&t=${encodeURIComponent(TOKEN)}`;
  } catch (e) { /* not a URL: as it is */ }
  return src;
}

function wireLightbox() {
  document.addEventListener("click", (e) => {
    const img = e.target.closest && e.target.closest(LIGHTBOX_PICS);
    if (!img) return;
    const host = img.closest(".gallery, .captions-grid, .st-preview") || document;   // step through the pictures beside it
    const imgs = [...host.querySelectorAll(LIGHTBOX_PICS)]       // only the pictures on show (a hidden pane is not one)
      .filter((i) => i === img || (i.getClientRects().length && (i.currentSrc || i.getAttribute("src"))));
    S.lb.list = imgs.map((i) => {
      const fig = i.closest(".gallery-item, .caption-card");
      const cap = fig && fig.querySelector("figcaption, .caption-card-title");
      const label = cap ? [...cap.childNodes].map((n) => n.textContent).filter(Boolean).join(" · ") : "";
      const m = /_e(\d{6})_(\d{2})_/.exec((fig && fig.title) || "");     // a preview: its epoch and prompt number
      return {src: fullPicture(i.currentSrc || i.src), name: label || i.alt, cap: fig && fig._cap,
              mode: (fig && fig._mode) || "", ep: m ? Number(m[1]) : null, idx: m ? Number(m[2]) : null};
    });
    lightboxShow(imgs.indexOf(img));
  });
  $("lightbox").addEventListener("click", (e) => { if (e.target.id === "lightbox" || e.target.id === "lightboxImg") lightboxClose(); });
  $("lightboxClose").addEventListener("click", lightboxClose);
  $("lightboxPrev").addEventListener("click", () => lightboxStep(S.lb.at - 1));
  $("lightboxNext").addEventListener("click", () => lightboxStep(S.lb.at + 1));
  $("lightboxCapSave").addEventListener("click", async () => {
    const pic = S.lb.list[S.lb.at];
    if (!pic || !pic.cap) return;
    pic.cap.ta.value = $("lightboxCapText").value;
    try { await pic.cap.save(); $("lightboxCapNote").textContent = "Saved"; }
    catch (e) { $("lightboxCapNote").textContent = e.message; }
  });
  $("lightboxCapRecap").addEventListener("click", async () => {
    const pic = S.lb.list[S.lb.at];
    if (!pic || !pic.cap || !pic.cap.recap || !(await lightboxCapFlush())) return;
    await pic.cap.recap((t) => { $("lightboxCapNote").textContent = t; });
  });
  $("lightboxCapSpeech").addEventListener("click", async () => {
    const pic = S.lb.list[S.lb.at];
    if (!pic || !pic.cap || !pic.cap.speech || !(await lightboxCapFlush())) return;
    const text = await pic.cap.speech((t) => { $("lightboxCapNote").textContent = t; });
    if (text && S.lb.list[S.lb.at] === pic) $("lightboxCapText").value = text;
  });
  $("lightboxCapRemove").addEventListener("click", async () => {
    const pic = S.lb.list[S.lb.at];
    if (!pic || !pic.cap || !(await pic.cap.remove())) return;
    S.lb.list.splice(S.lb.at, 1);              // gone from the set: the next picture takes its place
    if (!S.lb.list.length) { $("lightbox").style.display = "none"; $("lightboxImg").src = ""; return; }
    lightboxShow(Math.min(S.lb.at, S.lb.list.length - 1));
    $("lightboxCapNote").textContent = `${pic.cap.it.name} moved to removed (Undo is on its card).`;
  });
  document.addEventListener("keydown", (e) => {
    if ($("lightbox").style.display === "none") return;
    const typing = e.target === $("lightboxCapText");
    if (typing && e.key === "Enter" && (e.ctrlKey || e.metaKey)) { e.preventDefault(); $("lightboxCapSave").click(); }
    else if (e.key === "Escape") lightboxClose();
    else if (typing) return;                     // the arrows move the cursor in the caption
    else if (e.key.startsWith("Arrow")) {
      // previews: ← → the same prompt an epoch earlier / later, ↑ ↓ the prompt before / after at the same epoch;
      // any other pictures step in order
      const d = {ArrowLeft: -1, ArrowRight: 1, ArrowUp: -1, ArrowDown: 1}[e.key];
      const cur = S.lb.list[S.lb.at], across = e.key === "ArrowLeft" || e.key === "ArrowRight";
      if (!cur || cur.ep == null) { if (across) lightboxStep(S.lb.at + d); return; }
      e.preventDefault();
      const same = across ? (p) => p.idx === cur.idx : (p) => p.ep === cur.ep;
      const key = across ? (p) => p.ep : (p) => p.idx;
      let best = -1;
      S.lb.list.forEach((p, i) => {
        if (p.ep == null || !same(p) || (key(p) - key(cur)) * d <= 0) return;
        if (best < 0 || Math.abs(key(p) - key(cur)) < Math.abs(key(S.lb.list[best]) - key(cur))) best = i;
      });
      if (best >= 0) lightboxStep(best);
    }
  });
}

// back on this page after a while (another tab or a phone may have changed things): take the saved settings of the
// model on show, unless this page has a change of its own still waiting to save
async function catchUp() {
  if (document.visibilityState !== "visible" || S.saveWaiting) return;
  let st;
  try { st = await api("/api/state"); } catch (e) { return; }
  const saved = st && st.values && st.values[S.family];
  if (!saved || JSON.stringify(saved) === JSON.stringify(S.values)) return;
  S.all[S.family] = saved;
  S.values = {...defaults(), ...saved};
  renderSections("samples", $("samplesForm"));
  renderSections("training", $("trainingForm"));
  renderPresets();
  await refreshState();
}

function wire() {
  if (wired) return;
  wired = true;
  document.addEventListener("visibilitychange", catchUp);
  window.addEventListener("focus", catchUp);
  wireLightbox();
  wireStudio();
  wirePrep();
  $("presetSave").addEventListener("click", presetSaveOpen);
  $("btnImageLoss").addEventListener("click", async () => {      // the same view as Runs & outputs, for the run going now
    const box = $("runImageLoss"), id = S.run && S.run.history_id;
    if (!id || $("btnImageLoss").classList.contains("is-off")) return;
    if (box.style.display !== "none") { box.style.display = "none"; return; }
    box.style.display = ""; box.textContent = "Loading…";
    try { imageLossPanel(box, await api("/api/runs/image_loss?id=" + encodeURIComponent(id)), {...S.run, id}); }
    catch (e) { box.replaceChildren(el("div", {class: "problems", text: e.message})); }
  });
  document.querySelectorAll(".tab-item").forEach((b) => b.addEventListener("click", () => showTab(b.dataset.tab)));
  document.querySelectorAll("[data-goto]").forEach((b) => b.addEventListener("click", () => showTab(b.dataset.goto)));
  // the sidebar's run opens the run card itself (it sits under the Training form)
  $("sideRun").addEventListener("click", () => setTimeout(() =>
    document.querySelector(".run-card").scrollIntoView({behavior: "smooth", block: "start"}), 50));
  document.querySelectorAll("[data-close]").forEach((b) =>
    b.addEventListener("click", () => $(b.dataset.close).classList.remove("active")));
  for (const sel of document.querySelectorAll(".family-pick"))
    sel.addEventListener("change", (e) => chooseFamily(e.target.value));
  const folder = $("imageFolder");
  const setFolder = () => {
    const was = S.imageFolder;
    S.imageFolder = folder.value.trim(); loadTrigger(); refreshStateSoon(); saveStateSoon();
    if (S.imageFolder !== was) lookForget();
  };
  folder.addEventListener("change", setFolder);
  $("uploadFiles").addEventListener("change", (e) => { uploadFiles([...e.target.files]); e.target.value = ""; });
  $("uploadDir").addEventListener("change", (e) => {
    const files = [...e.target.files];
    nameFromFolder(files.length ? (files[0].webkitRelativePath || "").split("/")[0] : "");
    uploadFiles(files);
    e.target.value = "";
  });
  const drop = $("uploadDrop");
  drop.addEventListener("dragover", (e) => { e.preventDefault(); drop.classList.add("over"); });
  drop.addEventListener("dragleave", () => drop.classList.remove("over"));
  drop.addEventListener("drop", async (e) => {
    e.preventDefault();
    drop.classList.remove("over");
    const entries = [...e.dataTransfer.items].map((it) => it.webkitGetAsEntry && it.webkitGetAsEntry()).filter(Boolean);
    if (!entries.some((en) => en.isDirectory)) return uploadFiles([...e.dataTransfer.files]);
    const dirs = entries.filter((en) => en.isDirectory);
    if (dirs.length === 1) nameFromFolder(dirs[0].name);
    $("uploadStatus").textContent = "Reading the folder…";
    const files = [];
    for (const en of entries) await collect(en, files);
    uploadFiles(files);
  });
  showUploadRoot();
  document.querySelector('[data-pick="imageFolder"]').addEventListener("click", () =>
    pick(folder.value, true, (p) => { folder.value = p; setFolder(); }));
  $("pickerChoose").addEventListener("click", () => { $("pickerModal").classList.remove("active"); pickDone($("pickerPath").value); });
  // Upload from this device…: into Fizgig's uploads folder there (as a form box's Upload… does), then picked
  $("pickerUpFile").addEventListener("change", async () => {
    const f = $("pickerUpFile").files[0], say = (t, bad) => { $("pickerUpMsg").textContent = t; $("pickerUpMsg").classList.toggle("bad", !!bad); };
    $("pickerUpFile").value = "";
    if (!f) return;
    const mb = (n) => (n / 2 ** 20).toFixed(n < 10 * 2 ** 20 ? 1 : 0);
    $("pickerUpload").classList.add("disabled");
    try {
      const res = await uploadResumable({single: 1, folder: "", name: tidyName(f.name)}, f,
        (sent, note) => say(`Uploading ${f.name}: ${mb(sent)} of ${mb(f.size)} MB` + (note ? ` (${note})` : "")));
      if (!res.ok) return say(res.data.error || "Upload failed.", true);
      $("pickerModal").classList.remove("active");
      pickDone(res.data.path);
    } catch (e) { say(`${f.name}: ${e.message}`, true); }
    finally { $("pickerUpload").classList.remove("disabled"); }
  });
  $("pickerPath").addEventListener("change", (e) => browse(e.target.value));
  $("pickerFilter").addEventListener("input", pickerRender);
  $("pickerSort").addEventListener("change", (e) => pickerSortBy(e.target.value, false));
  $("pickerDir").addEventListener("click", () => pickerSortBy(pickSort.key, true));
  for (const h of document.querySelectorAll(".picker-head span"))
    h.addEventListener("click", () => pickerSortBy(h.dataset.sort === "mtime" && pickSort.key === "ctime" ? "ctime" : h.dataset.sort, true));
  // New folder: made in the folder on show, then opened (Use this folder picks it)
  const newRow = (on) => { $("pickerNewRow").style.display = on ? "" : "none"; if (on) $("pickerNewName").focus(); };
  $("pickerNew").addEventListener("click", () => { $("pickerNewMsg").textContent = ""; $("pickerNewName").value = ""; newRow(true); });
  $("pickerNewCancel").addEventListener("click", () => newRow(false));
  const makeFolder = async () => {
    try {
      const r = await api("/api/mkdir", {parent: $("pickerPath").value, name: $("pickerNewName").value});
      newRow(false);
      await browse(r.path);
      $("pickerNewMsg").textContent = "Made it. Press Use this folder to choose it.";
    } catch (e) { $("pickerNewMsg").textContent = e.message; }
  };
  $("pickerNewMake").addEventListener("click", makeFolder);
  $("pickerNewName").addEventListener("keydown", (e) => { if (e.key === "Enter") makeFolder(); });
  wireCaptions();
  $("outReload").addEventListener("click", loadOutputs);
  $("outToggle").addEventListener("click", () => {     // Saved files: folded until asked for, scanned only then
    S.outputsOpen = !S.outputsOpen;
    for (const id of ["outBody", "outRuns"]) $(id).style.display = S.outputsOpen ? "" : "none";
    $("outFolded").style.display = S.outputsOpen ? "none" : "";
    $("outToggle").textContent = S.outputsOpen ? "Hide" : "Show";
    if (S.outputsOpen) loadOutputs();
  });
  $("runsReload").addEventListener("click", loadRuns);
  $("datasetsReload").addEventListener("click", loadDatasets);
  $("aboutBtn").addEventListener("click", async () => {
    $("aboutModal").classList.add("active");
    try { showAbout(await api("/api/about")); } catch (e) { /* the box still reads well without them */ }
  });
  $("aboutCheck").addEventListener("click", async () => {      // the desktop's Check for updates
    $("aboutCheckSay").textContent = "Checking…";
    await api("/api/about?check=1").catch(() => null);
    for (let i = 0; i < 30; i++) {
      await new Promise((r) => setTimeout(r, 1000));
      const a = await api("/api/about").catch(() => null);
      if (a && !a.release.running) {
        showAbout(a);
        $("aboutCheckSay").textContent = {up_to_date: "You're on the latest release.", update_available: "",
          unknown: "Couldn't check (offline, or not a git checkout)."}[a.release.status] || "";
        return;
      }
    }
    $("aboutCheckSay").textContent = "";
  });
  setTimeout(() => api("/api/about").then(showAbout).catch(() => {}), 20000);   // the start-up check's answer
  $("datasetsSearch").addEventListener("input", debounce(renderDatasets, 200));
  loadDatasets();
  $("outFizgigOnly").addEventListener("change", loadOutputs);
  $("outSearch").addEventListener("input", debounce(renderOutputs, 200));
  $("outWhen").addEventListener("change", () => {
    $("outRange").style.display = $("outWhen").value === "custom" ? "" : "none";
    renderOutputs();
  });
  $("outFrom").addEventListener("change", renderOutputs);
  $("outTo").addEventListener("change", renderOutputs);
  $("btnCheck").addEventListener("click", check);
  $("btnStart").addEventListener("click", () => start("/api/start"));
  $("galleryRun").addEventListener("change", () => { S.galleryRun = $("galleryRun").value; loadGallery(); });
  $("likBtn").addEventListener("click", likenessPicker);
  $("styBtn").addEventListener("click", async () => {       // the Style score on or off for the shown run's folder
    const dir = S.lik && S.lik.dir;
    if (!dir) return;
    try { await api("/api/style", {dir, dataset: S.imageFolder || "", on: !(S.sty && S.sty.on)}); }
    catch (e) { $("styStatus").textContent = e.message; return; }
    styleShow(dir, []);
  });
  // Gizmo is a window on the Fizgig computer: offered only to a page open there (not a phone, not a pod)
  const gizmoHere = !!(S.uploadRoot && S.uploadRoot.local && !S.onPod);
  $("gizmoCard").style.display = gizmoHere ? "" : "none";
  $("gizmoAway").style.display = gizmoHere ? "none" : "";
  $("gizmoOpen").addEventListener("click", async () => {
    try { await api("/api/gizmo/open", {}); $("gizmoNote").textContent = "Gizmo is opening in its own window."; }
    catch (e) { $("gizmoNote").textContent = e.message; }
  });
  $("podStopKeep").addEventListener("click", () => api("/api/pod/keep", {}).then(pollRun).catch(() => {}));
  $("rvBtn").addEventListener("click", rvOpen);
  $("rvSlot").addEventListener("change", rvBuild);
  $("rvPlay").addEventListener("click", rvPlay);
  $("rvSlider").addEventListener("input", () => { rvStop(); rvShow(Number($("rvSlider").value)); });
  $("rvClip").addEventListener("click", rvClip);
  $("rvFrame").addEventListener("click", rvFrame);
  document.addEventListener("keydown", (e) => {
    if (!$("rvModal").classList.contains("active") || /INPUT|SELECT|TEXTAREA/.test(e.target.tagName)) return;
    if (e.key === "ArrowLeft") { rvStop(); rvShow(S.rv.i - 1); }
    else if (e.key === "ArrowRight") { rvStop(); rvShow(S.rv.i + 1); }
    else if (e.key === " ") { e.preventDefault(); rvPlay(); }
  });
  new MutationObserver(() => { if (!$("rvModal").classList.contains("active")) rvStop(); })
    .observe($("rvModal"), {attributes: true, attributeFilter: ["class"]});
  $("likStart").addEventListener("click", held($("likStart"), async () => {
    if (S.lik.mode !== "royale") return likenessSet(S.lik.picked);
    S.ry.likeRefs = S.lik.picked.map((n) => S.lik.paths[n]).filter(Boolean);
    ryLikeShow(); ryRemember();
    $("likModal").classList.remove("active");
  }));
  $("likFolderPick").addEventListener("click", () => pick(S.lik.folder || refStart(), true, (p) => likGridShow(p, [])));
  $("likClear").addEventListener("click", () => likenessSet([]));
  document.querySelectorAll("#galleryView button").forEach((b) => b.addEventListener("click", () => {
    S.galleryView = b.dataset.view; loadGallery(); }));
  $("btnViewPreviews").addEventListener("click", async () => {   // the Samples tab, at its previews
    showTab("samples");
    const go = () => $("galleryCard").scrollIntoView({block: "start"});
    requestAnimationFrame(go);
    try { await loadGallery(); } catch (e) { /* the previews show when they can */ }
    go();                                    // again with the previews in (their tiles keep a fixed shape)
  });
  $("btnQueue").addEventListener("click", async () => {
    try {
      const ed = S.editQueue;
      const r = await api(ed ? "/api/queue/update" : "/api/queue/add", ed ? {...runBody(), id: ed.id} : runBody());
      if (r.ok) {
        const q = (S.run && S.run.web_queue && S.run.web_queue.items) || [], live = S.run.running || S.run.state === "starting";
        const before = r.position > 1 ? (q[r.position - 2] || {}).lora : live ? S.run.lora : "";
        const eta = r.position === 1 && live && S.run.progress && S.run.progress.eta;
        showProblems([], ed ? `Saved: queued run ${r.position} has these settings now.` :
          `Added to the queue as number ${r.position}` + (before ? `: it starts after ${before}` + (eta ? ` (about ${eta})` : "") :
          ": it starts as soon as nothing else is training") + ".");
        queueEditing(null);
      } else {
        showProblems(r.problems || (r.error ? [r.error] : []));
        if (ed) $("queueEditText").replaceChildren(el("b", {text: `Not saved: `}),   // the bar is where you pressed it
          el("span", {text: (r.problems || [r.error]).join(" ")}));
      }
      pollRun();
    } catch (e) {
      showProblems([e.message]);
      if (S.editQueue) $("queueEditText").replaceChildren(el("b", {text: "Not saved: "}), el("span", {text: e.message}));
    }
  });
  $("queueEditSave").addEventListener("click", () => $("btnQueue").click());
  $("btnQueueCancel").addEventListener("click", () => { queueEditing(null); $("runStatus").textContent = "Edit cancelled: the queued run is as it was."; });
  $("webQueueStart").addEventListener("click", () => api("/api/queue/start", {}).then(pollRun)
    .catch((e) => { $("runStatus").textContent = e.message; }));
  $("webQueueHold").addEventListener("click", () => api("/api/queue/hold", {}).then(pollRun));
  $("webQueueClear").addEventListener("click", async () => {      // the desktop's Clear queue: asked on the button
    const b = $("webQueueClear"), ids = S.queueIds || [];
    if (!ids.length) return;
    if (!b.dataset.sure) {
      b.dataset.sure = "1"; b.textContent = `Clear ${ids.length} queued run${ids.length > 1 ? "s" : ""}?`;
      setTimeout(() => { delete b.dataset.sure; b.textContent = "Clear queue"; }, 4000);
      return;
    }
    delete b.dataset.sure; b.textContent = "Clear queue";
    if (S.editQueue) queueEditing(null);
    for (const id of ids) await api("/api/queue/remove", {id}).catch(() => {});
    pollRun();
  });
  $("sideQueue").addEventListener("click", toQueue);
  $("sideQueue").addEventListener("keydown", (e) => { if (e.key === "Enter" || e.key === " ") { e.preventDefault(); toQueue(); } });
  $("sideQueueStart").addEventListener("click", (e) => {
    e.stopPropagation();                          // starts it; the panel's own click opens the queue
    api("/api/queue/start", {}).then(pollRun).catch((er) => showProblems([er.message]));
  });
  $("btnPause").addEventListener("click", () => start("/api/pause"));
  $("btnStop").addEventListener("click", () => start("/api/stop"));
  $("btnResume").addEventListener("click", resumeClick);
  $("btnMove").addEventListener("click", () => { if (S.moveOpen) { S.moveOpen = false; pollRun(); } else moveOpen(); });
  const now = (what, say) => api("/api/run/now", {what}).then(() => { $("runNowSay").textContent = say; })
    .catch((e) => { $("runNowSay").textContent = e.message; });
  $("btnSampleNow").addEventListener("click", () => now("sample", "Previews are made after the current step - " +
    "they show with the others, under the epoch in progress."));
  $("btnSaveNow").addEventListener("click", () => now("save", "The LoRA is saved after the current step, as " +
    "<name>-step<step>.safetensors beside its checkpoints."));
  wirePods();
  wireHelpers();
  helpLoad(true);
  wireSliderLive();
}

// the login box works before anything else: a page that needs a login stops boot() at its first request, before wire()
$("loginButton").addEventListener("click", login);
$("loginPassword").addEventListener("keydown", (e) => { if (e.key === "Enter") login(); });
// ---- the theme: the Theme card at the top of Preferences. Each device keeps its own (a phone can differ from the desktop);
// index.html sets it before the page paints. A theme is a :root[data-theme] block in style.css.
const THEMES = [{key: "", swatch: "fizgig", name: "Fizgig", say: "dark, blue"},
  {key: "light-orange", swatch: "light-orange", name: "Light n’ Orange", say: "white, orange"},
  {key: "seventies", swatch: "seventies", name: "Seventies", say: "brown, orange"},
  {key: "synthwave", swatch: "synthwave", name: "Synthwave", say: "purple, pink"},
  {key: "graphite", swatch: "graphite", name: "Graphite", say: "grey, teal"},
  {key: "colour-safe", swatch: "colour-safe", name: "Colour-blind safe", say: "red-green colour blindness: no red against green"},
  {key: "high-contrast", swatch: "high-contrast", name: "High contrast", say: "low vision: black, white, yellow"},
  {key: "dim", swatch: "dim", name: "Dim", say: "light sensitivity, migraine: low glare, soft warm greys"}];
function applyTheme(key) {
  if (key) document.documentElement.dataset.theme = key;
  else delete document.documentElement.dataset.theme;
  try { key ? localStorage.setItem("fizgig.theme", key) : localStorage.removeItem("fizgig.theme"); } catch (e) {}
  // the phone's browser bar takes the page colour (the meta's own value covers the first paint)
  document.querySelector('meta[name="theme-color"]').content = themeColor("--bg-deep");
  window.dispatchEvent(new Event("resize"));      // the canvases draw again in the new colours
}
function wireTheme() {
  const render = () => {
    const now = document.documentElement.dataset.theme || "";
    $("themePick").replaceChildren(...THEMES.map((t) => el("button", {class: t.key === now ? "on" : "",
      "aria-pressed": String(t.key === now), onclick: () => { applyTheme(t.key); render(); }},
      el("span", {class: "theme-swatch", style: `--sw-bg:var(--swatch-${t.swatch}-bg);--sw-accent:var(--swatch-${t.swatch}-accent)`}),
      el("span", {class: "theme-name", text: t.name}), el("small", {text: t.say}))));
  };
  render();
  document.querySelector('meta[name="theme-color"]').content = themeColor("--bg-deep");
}
wireTheme();

boot().catch((e) => {             // a start that fails says so, rather than sitting on "Starting Fizgig…"
  if (e && e.message === "login") return;
  $("bootCoverText").textContent = "Fizgig didn't finish starting: " + (e && e.message || e) +
    ". Reload the page; if it happens again, the console window says why.";
  document.querySelector("#bootCover .boot-spin").style.display = "none";
});
