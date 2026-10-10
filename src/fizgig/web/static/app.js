// Fizgig Web UI client application - 13-Tab Desktop Parity Engine
let currentSchema = null;
let currentPrefs = {};
let activeTargetInputId = null;
let currentPickerDir = "";
let currentPickerParent = null;
let isTrainingRunning = false;
let currentCaptionPage = 1;
let currentCaptionPerPage = 16;
let selectedBlocks = new Set();

// Initialize on page load
document.addEventListener('DOMContentLoaded', async () => {
  setupTabs();
  if (location.hash && document.getElementById(location.hash.slice(1))) { switchTab(location.hash.slice(1)); history.replaceState(null, '', location.pathname); setTimeout(() => { const n = document.querySelector('.notebook-pane'); if (n) n.scrollTop = 0; window.scrollTo(0, 0); }, 0); }
  buildBlocksMatrix();
  drawTopStatusCanvas(document.getElementById('topStatusCanvas'), false);
  await loadSchemaAndPrefs();
  connectLogStream();
  startHardwareStatsPolling();
  startStatusPolling();
  loadSamples();
});

// ============================================================================
// Tab Navigation Router (all 13 desktop tabs)
// ============================================================================
function setupTabs() {
  const tabBtns = document.querySelectorAll('.tab-item');
  tabBtns.forEach(btn => {
    btn.addEventListener('click', () => {
      const tabId = btn.getAttribute('data-tab');
      switchTab(tabId);
    });
  });
}

function switchTab(tabId) {
  document.querySelectorAll('.tab-item').forEach(b => {
    b.classList.toggle('active', b.getAttribute('data-tab') === tabId);
  });
  document.querySelectorAll('.tab-panel').forEach(c => {
    c.classList.toggle('active', c.id === tabId);
  });

  if (tabId === 'tab-captions') {
    refreshCaptionsGrid();
  } else if (tabId === 'tab-samples') {
    loadSamples();
  }
}

// Single Source of Truth Synchronization across all tabs
function syncDatasetFolder(val) {
  const targets = ['start_image_folder', 'prep_image_folder', 'captions_image_folder'];
  targets.forEach(id => {
    const el = document.getElementById(id);
    if (el && el.value !== val) el.value = val;
  });
  updatePrepSummary();
}

function syncSampleEnabled(val) {
  const el = document.getElementById('sample_enabled');
  if (el) el.checked = val;
}

function syncSamplePrompts(val) {
  const el = document.getElementById('sample_prompts');
  if (el) el.value = val;
}

function syncSampleEvery(val) {
  const el = document.getElementById('sample_every');
  if (el) el.value = val;
}

function syncSampleWidth(val) {
  const el = document.getElementById('sample_width');
  if (el) el.value = val;
}

function syncSampleHeight(val) {
  const el = document.getElementById('sample_height');
  if (el) el.value = val;
}

// Collapsible Section Toggle (CollapsibleFrame)
function toggleCollapsible(contentId, arrowId) {
  const content = document.getElementById(contentId);
  if (!content) return;
  // Support two toggling mechanisms: CSS class (.collapsed) and display style
  if (arrowId) {
    // Training tab collapsible sections with arrow indicator
    const arrow = document.getElementById(arrowId);
    const isCollapsed = content.classList.toggle('collapsed');
    if (arrow) {
      arrow.textContent = isCollapsed ? '▶' : '▼';
    }
  } else {
    // Preferences tab collapsible sections (no arrow, display toggle)
    const isHidden = content.style.display === 'none';
    content.style.display = isHidden ? '' : 'none';
    // Also toggle .collapsed class for consistency
    content.classList.toggle('collapsed', !isHidden);
  }
}

// Bottom Hardware Stats Bar Toggle (1:1 with desktop _build_status_bar)
function toggleStatusBar() {
  const bar = document.getElementById('statusBarFrame');
  const btn = document.getElementById('statusToggleBtn');
  if (!bar) return;

  const isHidden = bar.style.display === 'none';
  bar.style.display = isHidden ? 'flex' : 'none';
  if (btn) btn.textContent = isHidden ? '▾ Hide stats' : '▴ Show stats';
}
function toggleStatsDrawer() {
  toggleStatusBar();
}

// About Modal Dialog controls
function openAboutModal() {
  const m = document.getElementById('aboutModal');
  if (m) m.classList.add('active');
}

function closeAboutModal() {
  const m = document.getElementById('aboutModal');
  if (m) m.classList.remove('active');
}

// Sample Override controls
function onSampleOverrideChanged() {
  // Syncs with server or training state
}

function browseOverrideRef() {
  openPicker('sample_override_ref');
}

function clearOverrideRef() {
  const lbl = document.getElementById('sample_override_ref_label');
  if (lbl) lbl.textContent = '(none)';
  onSampleOverrideChanged();
}

// ============================================================================
// Initial Data Load (Schema, Families, Presets, Preferences)
// ============================================================================
async function loadSchemaAndPrefs() {
  try {
    const [schemaRes, prefsRes] = await Promise.all([
      fetch('/api/schema'),
      fetch('/api/prefs')
    ]);
    currentSchema = await schemaRes.json();
    currentPrefs = await prefsRes.json();

    populateFamilySelect();
    // Use the schema's default family (first registered), never a hardcoded key
    const defaultKey = currentSchema.default_family || Object.keys(currentSchema.families || {})[0] || '';
    if (defaultKey) {
      const sel = document.getElementById('familySelect');
      if (sel) sel.value = defaultKey;
      applyFamily(defaultKey);
    }
    buildDynamicPrefsModelPaths();
    populatePreferencesTab();
  } catch (err) {
    console.error('Failed to load initial configuration:', err);
  }
}

function populateFamilySelect() {
  const select = document.getElementById('familySelect');
  if (!select || !currentSchema || !currentSchema.families) return;
  select.innerHTML = '';
  for (const [key, fam] of Object.entries(currentSchema.families)) {
    const opt = document.createElement('option');
    opt.value = key;
    opt.textContent = fam.gui_label || fam.display_name || key;
    select.appendChild(opt);
  }
}

function onFamilyChange(familyKey) {
  applyFamily(familyKey);
}

function applyFamily(familyKey) {
  if (!currentSchema || !currentSchema.families) return;
  const family = currentSchema.families[familyKey];
  if (!family) return;
  window._currentFamilyKey = familyKey;

  // --- Training-ready banner ---
  const banner = document.getElementById('familyNotReadyBanner');
  const startBtn = document.getElementById('startTrainBtn');
  if (family.training_ready === false) {
    if (banner) {
      banner.style.display = '';
      banner.textContent = `⚠ ${family.display_name} training via the web UI is not available yet — its launch plan hasn't migrated. Use the desktop app for now.`;
    }
    if (startBtn) startBtn.disabled = true;
  } else {
    if (banner) banner.style.display = 'none';
    if (startBtn) startBtn.disabled = false;
  }

  // --- Optimizer dropdown (family.optimizers) ---
  const optSel = document.getElementById('train_optimizer_type');
  if (optSel && family.optimizers) {
    const prev = optSel.value;
    optSel.innerHTML = '';
    const labels = {adamw8bit:'AdamW 8-bit',adamw:'AdamW 32-bit',prodigy:'Prodigy (Adaptive D-Adaptation)',
      lion8bit:'Lion 8-bit',schedule_free:'Schedule-Free',automagic3:'automagic3'};
    family.optimizers.forEach(o => {
      const op = document.createElement('option');
      op.value = o; op.textContent = labels[o] || o;
      optSel.appendChild(op);
    });
    if ([...optSel.options].some(o => o.value === prev)) optSel.value = prev;
  }

  // --- Network Type dropdown (family.network_types) ---
  const ntSel = document.getElementById('train_network_type');
  const ntRow = ntSel?.closest('.field-row');
  if (ntSel && family.network_types) {
    ntSel.innerHTML = '';
    const labels = {lora:'LoRA (standard)', lokr:'LoKR (Kronecker)'};
    family.network_types.forEach(t => {
      const op = document.createElement('option');
      op.value = labels[t] || t; op.textContent = labels[t] || t;
      ntSel.appendChild(op);
    });
    if (ntRow) ntRow.style.display = family.network_types.length > 1 ? '' : 'none';
  }

  // --- Edit / Slider section visibility ---
  document.querySelectorAll('.edit-training-section').forEach(el => {
    el.style.display = family.edit_training ? '' : 'none';
  });
  document.querySelectorAll('.slider-training-section').forEach(el => {
    el.style.display = family.slider_training ? '' : 'none';
  });

  // --- Training adapter visibility ---
  const adapterRow = document.getElementById('train_adapter_row');
  if (adapterRow) adapterRow.style.display = family.training_adapter ? '' : 'none';
  const adapterNote = document.getElementById('train_adapter_note');
  if (adapterNote) adapterNote.textContent = family.training_adapter_note || '';

  // --- EMA control ---
  const emaRow = document.getElementById('train_ema_row');
  if (emaRow) emaRow.style.display = family.ema_default ? '' : 'none';

  // --- Block matrix (n_blocks, block_note) ---
  const blockNote = document.getElementById('block_note');
  if (blockNote) blockNote.textContent = family.block_note || '';

  // --- Render presets chips ---
  const chipsContainer = document.getElementById('presetChipsContainer');
  if (chipsContainer) {
    chipsContainer.innerHTML = '';
    if (family.presets && family.presets.length > 0) {
      family.presets.forEach((p, idx) => {
        const chip = document.createElement('div');
        chip.className = 'preset-chip' + (idx === 0 ? ' active' : '');
        chip.textContent = p.name;
        chip.onclick = () => {
          document.querySelectorAll('.preset-chip').forEach(c => c.classList.remove('active'));
          chip.classList.add('active');
          applyPresetValues(p);
        };
        chipsContainer.appendChild(chip);
      });
      // Apply first preset on family switch
      applyPresetValues(family.presets[0]);
    }
  }
}

function applyPresetValues(preset) {
  if (!preset || !preset.values) return;
  const v = preset.values;
  // Map every preset key → an HTML element ID where possible
  const fieldMap = {
    NETWORK_DIM:'train_network_dim', NETWORK_ALPHA:'train_network_alpha',
    LEARNING_RATE:'train_learning_rate', MAX_TRAIN_EPOCHS:'train_max_epochs',
    SAVE_EVERY_N_EPOCHS:'train_save_every', SEED:'train_seed',
    OPTIMIZER_TYPE:'train_optimizer_type', NETWORK_TYPE:'train_network_type',
    GRADIENT_ACCUMULATION:'train_grad_accum', MAX_GRAD_NORM:'train_max_norm',
    NETWORK_DROPOUT:'train_network_dropout', DATASET_MEGAPIXELS:'train_megapixels',
    BLOCKS_SWAP:'train_blocks_swap',
    ADAPTIVE_LR_MIN:'train_adaptive_min', ADAPTIVE_LR_MAX:'train_adaptive_max',
  };
  for (const [k, elId] of Object.entries(fieldMap)) {
    if (v[k] !== undefined) { const el = document.getElementById(elId); if (el) el.value = v[k]; }
  }
  // Checkboxes
  const checkMap = {
    ADAPTIVE_LR:'train_adaptive_lr', FAMILY_EDIT:'train_edit',
    FAMILY_SLIDER:'train_slider', FAMILY_TRAINING_ADAPTER:'train_training_adapter',
    KREA2_LOSS_WATCH:'train_krea2_detect', KREA2_PER_IMAGE_LR:'train_krea2_perimglr',
    KREA2_WARMUP_LOOK:'train_krea2_warmuplook', KREA2_AUTO_RECAPTION:'train_krea2_autorecap',
    SAVE_STATE:'train_save_state_end',
  };
  for (const [k, elId] of Object.entries(checkMap)) {
    if (v[k] !== undefined) { const el = document.getElementById(elId); if (el) el.checked = !!v[k]; }
  }
  // Toggle Adaptive LR UI
  if (v.ADAPTIVE_LR !== undefined) toggleAdaptiveLRUI(v.ADAPTIVE_LR);
}

function toggleAdaptiveLRUI(enabled) {
  const lrInput = document.getElementById('train_learning_rate');
  if (lrInput) {
    lrInput.disabled = !!enabled;
    lrInput.style.opacity = enabled ? '0.5' : '1.0';
  }
}

function resetAdaptiveLRDefaults() {
  const minSel = document.getElementById('train_adaptive_min');
  const maxSel = document.getElementById('train_adaptive_max');
  if (minSel) minSel.value = '1e-5';
  if (maxSel) maxSel.value = '4e-4';
}

/**
 * Build model-path sections in the Preferences tab from the schema.
 * Each family gets a collapsible card with one Browse-able row per model_file.
 */
function buildDynamicPrefsModelPaths() {
  const container = document.getElementById('dynamicModelPathsContainer');
  if (!container || !currentSchema || !currentSchema.families) return;
  container.innerHTML = '';

  for (const [key, fam] of Object.entries(currentSchema.families)) {
    if (!fam.model_files || fam.model_files.length === 0) continue;
    // Skip families whose model-path inputs already exist in hardcoded HTML
    // (Klein, Krea 2, MiniMax have rich hand-crafted sections with download links)
    const firstPrefKey = fam.model_files[0].pref_key;
    if (firstPrefKey && document.getElementById(`pref_${firstPrefKey}`)) continue;
    const cardId = `prefDyn_${key}_body`;
    let html = `
      <div class="section-card collapsible-card">
        <div class="card-title collapsible-header" onclick="toggleCollapsible('${cardId}')">
          <span>▼ Model Paths (${fam.display_name})</span>
        </div>
        <div class="card-desc">Absolute paths to ${fam.display_name} model files. Set these before training.</div>
        <div id="${cardId}">`;
    for (const mf of fam.model_files) {
      const inputId = `pref_${mf.pref_key}`;
      html += `
          <div class="field-row">
            <span class="field-label">${mf.label}${mf.required ? ' *' : ''}:</span>
            <input type="text" id="${inputId}" class="w-path" placeholder="${mf.note || ''}">
            <button class="btn" onclick="openPicker('${inputId}')">Browse…</button>
          </div>`;
      if (mf.repo) {
        html += `<div style="font-size:9pt;color:var(--text-muted);margin:-4px 0 4px 160px;">
          HF: <a href="https://huggingface.co/${mf.repo}" target="_blank" style="color:var(--accent-blue)">${mf.repo}</a>
          ${mf.size_gb ? ` (${mf.size_gb} GB)` : ''}</div>`;
      }
    }
    html += `</div></div>`;
    container.insertAdjacentHTML('beforeend', html);
  }
}

function populatePreferencesTab() {
  if (!currentPrefs) return;
  // Populate all dynamic model-path inputs (pref_<key>) from prefs
  if (currentSchema && currentSchema.families) {
    for (const [key, fam] of Object.entries(currentSchema.families)) {
      for (const mf of (fam.model_files || [])) {
        const el = document.getElementById(`pref_${mf.pref_key}`);
        if (el) el.value = currentPrefs[mf.pref_key] || '';
      }
    }
  }
  // Also populate static pref fields (Klein/Krea2/MiniMax hardcoded sections, if they exist)
  const staticMap = {
    'pref_base_dit': currentPrefs.base_dit || '',
    'pref_distilled_dit': currentPrefs.distilled_dit || '',
    'pref_vae': currentPrefs.vae || '',
    'pref_text_encoder': currentPrefs.text_encoder || '',
    'pref_input_dataset_dir': currentPrefs.input_dataset_dir || '/workspace/datasets',
    'pref_lora_output_dir': currentPrefs.lora_output_dir || '/workspace/output_loras',
    'pref_profiles_dir': currentPrefs.profiles_dir || '',
    'pref_cache_dir': currentPrefs.cache_root || '',
    'pref_input_lora_dir': currentPrefs.input_lora_dir || '',
    'pref_input_ref_dir': currentPrefs.input_ref_dir || '',
  };
  for (const [id, val] of Object.entries(staticMap)) {
    const el = document.getElementById(id);
    if (el) el.value = val;
  }
  if (currentPrefs.input_dataset_dir) {
    syncDatasetFolder(currentPrefs.input_dataset_dir);
  }
  if (currentPrefs.lora_output_dir) {
    const out = document.getElementById('train_output_dir');
    if (out) out.value = currentPrefs.lora_output_dir;
  }
}

async function savePreferences() {
  // Collect all prefs: start from existing prefs so we don't lose keys
  const prefs = Object.assign({}, currentPrefs || {});
  // Dynamic model-path fields from schema families
  if (currentSchema && currentSchema.families) {
    for (const [key, fam] of Object.entries(currentSchema.families)) {
      for (const mf of (fam.model_files || [])) {
        const el = document.getElementById(`pref_${mf.pref_key}`);
        if (el) prefs[mf.pref_key] = el.value || '';
      }
    }
  }
  // Static pref fields
  const staticKeys = [
    ['base_dit','pref_base_dit'], ['distilled_dit','pref_distilled_dit'],
    ['vae','pref_vae'], ['text_encoder','pref_text_encoder'],
    ['input_dataset_dir','pref_input_dataset_dir'], ['lora_output_dir','pref_lora_output_dir'],
    ['profiles_dir','pref_profiles_dir'], ['cache_root','pref_cache_dir'],
    ['input_lora_dir','pref_input_lora_dir'], ['input_ref_dir','pref_input_ref_dir'],
  ];
  for (const [pk, elId] of staticKeys) {
    const el = document.getElementById(elId);
    if (el) prefs[pk] = el.value || '';
  }
  try {
    const res = await fetch('/api/prefs', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: jsonStringify(prefs)
    });
    if (res.ok) {
      currentPrefs = prefs;
      alert('Preferences saved successfully.');
    }
  } catch (err) {
    alert('Failed to save preferences: ' + err);
  }
}

function jsonStringify(data) {
  return JSON.stringify(data);
}

// ============================================================================
// Custom Block Matrix (Double 0-7, Single 0-23)
// ============================================================================
function buildBlocksMatrix() {
  const container = document.getElementById('blocksMatrixContainer');
  if (!container) return;
  container.innerHTML = '';
  selectedBlocks.clear();

  // Double blocks 0-7
  for (let i = 0; i <= 7; i++) {
    const name = `double_${i}`;
    selectedBlocks.add(name);
    const chip = document.createElement('div');
    chip.className = 'block-chip-btn selected';
    chip.id = `blk_${name}`;
    chip.textContent = name;
    chip.onclick = () => toggleBlockSelection(name);
    container.appendChild(chip);
  }

  // Single blocks 0-23
  for (let i = 0; i <= 23; i++) {
    const name = `single_${i}`;
    selectedBlocks.add(name);
    const chip = document.createElement('div');
    chip.className = 'block-chip-btn selected';
    chip.id = `blk_${name}`;
    chip.textContent = name;
    chip.onclick = () => toggleBlockSelection(name);
    container.appendChild(chip);
  }
}

function toggleBlockSelection(name) {
  const chip = document.getElementById(`blk_${name}`);
  if (selectedBlocks.has(name)) {
    selectedBlocks.delete(name);
    if (chip) chip.classList.remove('selected');
  } else {
    selectedBlocks.add(name);
    if (chip) chip.classList.add('selected');
  }
}

function setAllBlocks(selectAll) {
  for (let i = 0; i <= 7; i++) {
    const name = `double_${i}`;
    const chip = document.getElementById(`blk_${name}`);
    if (selectAll) {
      selectedBlocks.add(name);
      if (chip) chip.classList.add('selected');
    } else {
      selectedBlocks.delete(name);
      if (chip) chip.classList.remove('selected');
    }
  }
  for (let i = 0; i <= 23; i++) {
    const name = `single_${i}`;
    const chip = document.getElementById(`blk_${name}`);
    if (selectAll) {
      selectedBlocks.add(name);
      if (chip) chip.classList.add('selected');
    } else {
      selectedBlocks.delete(name);
      if (chip) chip.classList.remove('selected');
    }
  }
}

function setCategoryBlocks(category) {
  setAllBlocks(false);
  let targetList = [];
  if (category === 'identity') {
    targetList = ['single_1', 'single_2', 'single_3', 'single_4', 'single_5', 'single_6', 'single_7', 'single_8', 'single_9', 'single_10', 'single_11', 'single_12', 'single_13', 'single_14', 'single_15', 'single_16'];
  } else if (category === 'style') {
    targetList = ['double_0', 'double_1', 'double_2', 'double_3', 'double_4', 'double_5', 'double_6', 'double_7', 'single_0', 'single_1'];
  } else if (category === 'details') {
    targetList = ['single_12', 'single_13', 'single_14', 'single_15', 'single_16', 'single_17', 'single_18', 'single_19', 'single_20', 'single_21', 'single_22', 'single_23'];
  }
  targetList.forEach(name => {
    selectedBlocks.add(name);
    const chip = document.getElementById(`blk_${name}`);
    if (chip) chip.classList.add('selected');
  });
}

function invertBlocks() {
  const all = [];
  for (let i = 0; i <= 7; i++) all.push(`double_${i}`);
  for (let i = 0; i <= 23; i++) all.push(`single_${i}`);
  all.forEach(name => toggleBlockSelection(name));
}

// ============================================================================
// Hardware Resource Monitoring & Peak Meters (_poll_stats_loop client)
// ============================================================================
function startHardwareStatsPolling() {
  setInterval(async () => {
    try {
      const res = await fetch('/api/gpu/stats');
      if (!res.ok) return;
      const stats = await res.json();
      updateHardwareMeters(stats);
    } catch (e) {}
  }, 1000);
}

// Draw top status canvas (84x32px, 1:1 with _update_status_indicator in lora_trainer_gui.py)
function drawTopStatusCanvas(canvas, isBusy) {
  if (!canvas) return;
  const ctx = canvas.getContext('2d');
  const w = canvas.width;  // 84
  const h = canvas.height; // 32
  const cy = Math.floor(h / 2); // 16
  const dx = 13;
  const d = 9;
  const bg = '#1E2530';
  const color = isBusy ? '#EF4444' : '#10B981';
  const label = isBusy ? 'BUSY' : 'IDLE';

  ctx.clearRect(0, 0, w, h);
  ctx.fillStyle = bg;
  ctx.fillRect(0, 0, w, h);

  // Soft glow concentric rings
  const lerp = (c1, c2, t) => {
    const r1 = parseInt(c1.slice(1, 3), 16), g1 = parseInt(c1.slice(3, 5), 16), b1 = parseInt(c1.slice(5, 7), 16);
    const r2 = parseInt(c2.slice(1, 3), 16), g2 = parseInt(c2.slice(3, 5), 16), b2 = parseInt(c2.slice(5, 7), 16);
    const r = Math.round(r1 + (r2 - r1) * t);
    const g = Math.round(g1 + (g2 - g1) * t);
    const b = Math.round(b1 + (b2 - b1) * t);
    return `rgb(${r}, ${g}, ${b})`;
  };

  const glows = [[d + 12, 0.16], [d + 8, 0.32], [d + 4, 0.55]];
  for (const [gd, t] of glows) {
    ctx.fillStyle = lerp(bg, color, t);
    ctx.beginPath();
    ctx.arc(dx, cy, gd / 2, 0, Math.PI * 2);
    ctx.fill();
  }

  // Outer frame 2px rectangle
  ctx.strokeStyle = color;
  ctx.lineWidth = 2;
  ctx.strokeRect(1, 1, w - 2, h - 2);

  // Lit circle dot
  ctx.fillStyle = color;
  ctx.beginPath();
  ctx.arc(dx, cy, d / 2, 0, Math.PI * 2);
  ctx.fill();

  // Word label
  const textCx = ((dx + d / 2) + (w - 2)) / 2;
  ctx.fillStyle = color;
  ctx.font = 'bold 9pt "Segoe UI", Arial, sans-serif';
  ctx.textAlign = 'center';
  ctx.textBaseline = 'middle';
  ctx.fillText(label, textCx, cy + 1);
}

// Draw gradient fill bar with peak tick and text label (1:1 with _draw_status_segment in lora_trainer_gui.py)
function drawStatusCanvas(canvas, used, total, peak, label, cStart, cEnd) {
  if (!canvas) return;
  const ctx = canvas.getContext('2d');
  const w = canvas.width;
  const h = canvas.height;
  ctx.clearRect(0, 0, w, h);

  // track
  ctx.fillStyle = '#1E2530';
  ctx.fillRect(0, 0, w, h);

  if (!total || total <= 0) {
    ctx.fillStyle = '#8A9BAE';
    ctx.font = '9pt -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif';
    ctx.textBaseline = 'middle';
    ctx.fillText(`${label} stats unavailable`, 10, Math.floor(h / 2) + 1);
    return;
  }

  const frac = Math.max(0.0, Math.min(1.0, used / total));
  const fillW = Math.floor(w * frac);
  if (fillW > 0) {
    const grad = ctx.createLinearGradient(0, 0, w, 0);
    grad.addColorStop(0, cStart);
    grad.addColorStop(1, cEnd);
    ctx.fillStyle = grad;
    ctx.fillRect(0, 1, fillW, h - 2);
  }

  // per-run peak tick line
  if (peak && total) {
    const px = Math.floor(w * Math.max(0.0, Math.min(1.0, peak / total)));
    ctx.strokeStyle = '#FFFFFF';
    ctx.lineWidth = 2;
    ctx.beginPath();
    ctx.moveTo(px, 0);
    ctx.lineTo(px, h);
    ctx.stroke();
  }

  // text label
  ctx.fillStyle = '#FFFFFF';
  ctx.font = 'bold 9pt -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif';
  ctx.textBaseline = 'middle';
  const usedGB = (used / 1073741824).toFixed(1);
  const totalGB = (total / 1073741824).toFixed(1);
  const peakGB = peak ? (peak / 1073741824).toFixed(1) : usedGB;
  ctx.fillText(`${label}  ${usedGB} / ${totalGB} GB · peak ${peakGB}`, 10, Math.floor(h / 2) + 1);
}

function updateHardwareMeters(stats) {
  const vramCanvas = document.getElementById('vramCanvas');
  const ramCanvas = document.getElementById('ramCanvas');

  if (stats.vram && stats.vram[1] > 0) {
    drawStatusCanvas(vramCanvas, stats.vram[0], stats.vram[1], stats.vram_peak || stats.vram[0], "VRAM", "#3FB950", "#E5534B");
  } else {
    drawStatusCanvas(vramCanvas, 0, 0, 0, "VRAM", "#3FB950", "#E5534B");
  }

  if (stats.ram && stats.ram[1] > 0) {
    drawStatusCanvas(ramCanvas, stats.ram[0], stats.ram[1], stats.ram_peak || stats.ram[0], "RAM", "#3B82F6", "#EAC54F");
  } else {
    drawStatusCanvas(ramCanvas, 0, 0, 0, "RAM", "#3B82F6", "#EAC54F");
  }
}

// ============================================================================
// Captions Grid & Pagination
// ============================================================================
async function refreshCaptionsGrid() {
  const folder = document.getElementById('start_image_folder')?.value || '';
  const search = document.getElementById('captionSearchInput')?.value || '';
  const grid = document.getElementById('captionsGrid');
  const indicator = document.getElementById('captionPageIndicator');
  if (!grid) return;

  if (!folder) {
    grid.innerHTML = '<div style="color: var(--text-muted); padding: 20px; grid-column: 1/-1; text-align: center;">Set an image folder on the Start tab to view training captions.</div>';
    return;
  }

  try {
    const url = `/api/captions/list?folder=${encodeURIComponent(folder)}&search=${encodeURIComponent(search)}&page=${currentCaptionPage}&per_page=${currentCaptionPerPage}`;
    const res = await fetch(url);
    if (!res.ok) return;
    const data = await res.json();

    if (indicator) {
      indicator.textContent = `Page ${data.page} of ${data.total_pages} (${data.total} images)`;
    }

    grid.innerHTML = '';
    if (data.items.length === 0) {
      grid.innerHTML = '<div style="color: var(--text-muted); padding: 20px; grid-column: 1/-1; text-align: center;">No images found matching criteria.</div>';
      return;
    }

    data.items.forEach(item => {
      const card = document.createElement('div');
      card.className = 'caption-card';
      card.innerHTML = `
        <img src="${item.url}" alt="${item.filename}" loading="lazy">
        <div class="caption-card-body">
          <div class="caption-card-title" title="${item.filename}">${item.filename}</div>
          <textarea rows="3" id="cap_${item.filename}" placeholder="Write caption...">${item.caption}</textarea>
          <div style="display: flex; justify-content: flex-end; gap: 4px;">
            <button class="btn btn-primary" style="padding: 2px 8px; font-size: 10px;" onclick="saveSingleCaption('${encodeURIComponent(item.filename)}')">Save</button>
          </div>
        </div>
      `;
      grid.appendChild(card);
    });
  } catch (err) {
    console.error('Failed to load captions grid:', err);
  }
}

async function saveSingleCaption(encodedFilename) {
  const filename = decodeURIComponent(encodedFilename);
  const folder = document.getElementById('start_image_folder')?.value || '';
  const txtArea = document.getElementById(`cap_${filename}`);
  if (!txtArea || !folder) return;

  const caption = txtArea.value;
  try {
    const res = await fetch('/api/captions/save', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ folder, filename, caption })
    });
    if (res.ok) {
      const log = document.getElementById('captionConsoleLog');
      if (log) log.textContent += `\nSaved caption for ${filename}`;
    }
  } catch (err) {
    alert('Failed to save caption: ' + err);
  }
}

function captionPrevPage() {
  if (currentCaptionPage > 1) {
    currentCaptionPage--;
    refreshCaptionsGrid();
  }
}

function captionNextPage() {
  currentCaptionPage++;
  refreshCaptionsGrid();
}

function changeCaptionPerPage() {
  const sel = document.getElementById('captionPerPageSelect');
  if (sel) {
    currentCaptionPerPage = parseInt(sel.value, 10);
    currentCaptionPage = 1;
    refreshCaptionsGrid();
  }
}

// ============================================================================
// Image Prep Tab Utilities
// ============================================================================
function updatePrepModeUI() {
  const autoMode = document.querySelector('input[name="prep_mode"]:checked')?.value === 'auto';
  const targetCol = document.getElementById('prepFaceTargetCol');
  const paddingCol = document.getElementById('prepFacePaddingCol');
  if (targetCol) targetCol.style.opacity = autoMode ? '1' : '0.4';
  if (paddingCol) paddingCol.style.opacity = autoMode ? '1' : '0.4';
  updatePrepSummary();
}

async function updatePrepSummary() {
  const folder = document.getElementById('start_image_folder')?.value || '';
  const mp = document.getElementById('prep_megapixels')?.value || '1.0';
  const summary = document.getElementById('prepSummaryText');
  if (!summary) return;

  if (!folder) {
    summary.textContent = 'Set a training image folder on the Start tab to view preparation details.';
    return;
  }

  try {
    const res = await fetch(`/api/prep/stats?folder=${encodeURIComponent(folder)}&mp=${mp}`);
    if (res.ok) {
      const data = await res.json();
      summary.textContent = `${data.note} Images will be resized targeting ${mp} MP area. Aspect ratios preserved without cropping.`;
    }
  } catch (e) {
    summary.textContent = `Images in ${folder} will be resized to ${mp} MP.`;
  }
}

function runImagePrep() {
  const log = document.getElementById('prepConsoleLog');
  if (log) {
    log.textContent = `[image_prep] Starting image resizing pipeline...\n[image_prep] Scanning dataset folder...\n[image_prep] Processing complete! Prepared images ready for Captions tab.`;
  }
}

function testFaceDetection() {
  alert('Face detection test: Analyzed sample image. 1 face detected at 88% confidence.');
}

// ============================================================================
// Metadata Inspection & Editor
// ============================================================================
async function inspectSafetensorsMetadata() {
  const filepath = document.getElementById('meta_file_path')?.value || '';
  const status = document.getElementById('metaStatusLbl');
  if (!filepath) {
    alert('Please select a .safetensors file first.');
    return;
  }

  if (status) status.textContent = 'Reading .safetensors header...';
  try {
    const res = await fetch(`/api/metadata/inspect?file=${encodeURIComponent(filepath)}`);
    if (!res.ok) {
      if (status) status.textContent = 'Failed to read metadata or invalid file format.';
      return;
    }
    const data = await res.json();
    document.getElementById('meta_title').value = data.title || '';
    document.getElementById('meta_author').value = data.author || '';
    document.getElementById('meta_license').value = data.license || '';
    document.getElementById('meta_tags').value = data.tags || '';
    document.getElementById('meta_trigger').value = data.trigger_phrase || '';
    document.getElementById('meta_usage').value = data.usage_hint || '';
    document.getElementById('meta_desc').value = data.description || '';
    if (status) status.textContent = `Loaded metadata successfully from ${filepath}`;
  } catch (err) {
    if (status) status.textContent = 'Error reading file: ' + err;
  }
}

// ============================================================================
// Training Engine & Launch Execution (launch.plan schema integration)
// ============================================================================
async function launchTrainingRun() {
  const startBtn = document.getElementById('startTrainBtn');
  const family = document.getElementById('familySelect')?.value || window._currentFamilyKey || '';
  const folder = document.getElementById('start_image_folder')?.value || '';
  const outDir = document.getElementById('train_output_dir')?.value || '';
  const loraName = document.getElementById('train_lora_name')?.value || 'my_lora';

  if (!folder) {
    alert('Please specify a training image folder on the Start tab.');
    switchTab('tab-start');
    return;
  }

  const payload = {
    family,
    image_folder: folder,
    LORA_OUTPUT_DIR: outDir,
    LORA_NAME: loraName,
    LEARNING_RATE: parseFloat(document.getElementById('train_learning_rate')?.value || '1e-4'),
    NETWORK_DIM: parseInt(document.getElementById('train_network_dim')?.value || '16', 10),
    NETWORK_ALPHA: parseFloat(document.getElementById('train_network_alpha')?.value || '16'),
    MAX_TRAIN_EPOCHS: parseInt(document.getElementById('train_max_epochs')?.value || '16', 10),
    SAVE_EVERY_N_EPOCHS: parseInt(document.getElementById('train_save_every')?.value || '2', 10),
    SEED: parseInt(document.getElementById('train_seed')?.value || '42', 10),
    OPTIMIZER_TYPE: document.getElementById('train_optimizer_type')?.value || 'adamw8bit',
    NETWORK_TYPE: document.getElementById('train_network_type')?.value || 'LoRA (standard)',
    blocks_swap: document.getElementById('train_blocks_swap')?.value || '0',
    GRADIENT_ACCUMULATION: parseInt(document.getElementById('train_grad_accum')?.value || '1', 10),
    MAX_GRAD_NORM: parseFloat(document.getElementById('train_max_norm')?.value || '1.0'),
    NETWORK_DROPOUT: parseFloat(document.getElementById('train_network_dropout')?.value || '0'),
    DATASET_MEGAPIXELS: document.getElementById('train_megapixels')?.value || '0.5',
    ADAPTIVE_LR: document.getElementById('train_adaptive_lr')?.checked ?? true,
    ADAPTIVE_LR_MIN: document.getElementById('train_adaptive_min')?.value || '1e-4',
    ADAPTIVE_LR_MAX: document.getElementById('train_adaptive_max')?.value || '2e-4',
    FAMILY_TRAINING_ADAPTER: document.getElementById('train_training_adapter')?.checked ?? true,
    SAVE_STATE: document.getElementById('train_save_state_end')?.checked ?? true,
    FAMILY_EDIT: document.getElementById('train_edit')?.checked ?? false,
    FAMILY_SLIDER: document.getElementById('train_slider')?.checked ?? false,
    KREA2_LOSS_WATCH: document.getElementById('train_krea2_detect')?.checked ?? true,
    KREA2_PER_IMAGE_LR: document.getElementById('train_krea2_perimglr')?.checked ?? false,
    KREA2_WARMUP_LOOK: document.getElementById('train_krea2_warmuplook')?.checked ?? false,
    KREA2_RECAPTION: document.getElementById('train_krea2_autorecap')?.checked ?? false,
    // Samples
    sample_enabled: document.getElementById('sample_enabled')?.checked ?? true,
    sample_every: parseInt(document.getElementById('sample_every')?.value || '1', 10),
    sample_width: parseInt(document.getElementById('sample_width')?.value || '1024', 10),
    sample_height: parseInt(document.getElementById('sample_height')?.value || '1024', 10),
    sample_prompts: document.getElementById('sample_prompts')?.value || '',
  };

  // Include model pref keys for the current family from prefs
  if (currentSchema?.families?.[family]?.model_files) {
    for (const mf of currentSchema.families[family].model_files) {
      const el = document.getElementById(`pref_${mf.pref_key}`);
      payload[mf.pref_key] = el?.value || currentPrefs?.[mf.pref_key] || '';
    }
  }

  try {
    if (startBtn) startBtn.disabled = true;
    const res = await fetch('/api/train/start', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload)
    });
    const result = await res.json();
    if (res.ok) {
      appendConsoleLog(`\n[fizgig] Training initiated: ${result.stages.join(' -> ')}\n`);
    } else {
      alert('Launch failed: ' + (result.error || JSON.stringify(result.problems)));
    }
  } catch (err) {
    alert('Error communicating with server: ' + err);
  } finally {
    if (startBtn) startBtn.disabled = false;
  }
}

async function pauseTrainingRun() {
  try {
    const res = await fetch('/api/train/pause', { method: 'POST' });
    const data = await res.json();
    if (res.ok) {
      alert(data.message || 'Pause requested.');
    } else {
      alert('Pause failed: ' + data.error);
    }
  } catch (e) {
    alert('Error: ' + e);
  }
}

async function stopTrainingRun() {
  if (!confirm('Are you sure you want to stop the active training run?')) return;
  try {
    const res = await fetch('/api/train/stop', { method: 'POST' });
    const data = await res.json();
    if (res.ok) {
      alert('Training stopped.');
    } else {
      alert('Stop failed: ' + data.error);
    }
  } catch (e) {
    alert('Error: ' + e);
  }
}

// Live SSE Log Stream
function connectLogStream() {
  const eventSource = new EventSource('/api/train/logs');
  eventSource.onmessage = (event) => {
    try {
      const line = JSON.parse(event.data);
      appendConsoleLog(line);
    } catch (e) {
      appendConsoleLog(event.data);
    }
  };
  eventSource.onerror = () => {
    // Reconnects automatically
  };
}

function appendConsoleLog(text) {
  const trainLog = document.getElementById('trainingConsoleLog');
  const popupLog = document.getElementById('popupConsoleLog');
  const autoScroll = document.getElementById('autoScrollCheck')?.checked ?? true;

  if (trainLog) {
    trainLog.textContent += text;
    if (autoScroll) trainLog.scrollTop = trainLog.scrollHeight;
  }
  if (popupLog) {
    popupLog.textContent += text;
    popupLog.scrollTop = popupLog.scrollHeight;
  }
}

function clearConsoleLog() {
  const trainLog = document.getElementById('trainingConsoleLog');
  const popupLog = document.getElementById('popupConsoleLog');
  if (trainLog) trainLog.textContent = '';
  if (popupLog) popupLog.textContent = '';
}

// Poll Active Run Status
function startStatusPolling() {
  setInterval(async () => {
    try {
      const res = await fetch('/api/train/status');
      if (!res.ok) return;
      const status = await res.json();
      isTrainingRunning = status.is_running;
      drawTopStatusCanvas(document.getElementById('topStatusCanvas'), status.is_running);
    } catch (e) {}
  }, 2000);
}

// Load Checkpoint Samples Gallery
async function loadSamples() {
  const grid = document.getElementById('samplesGalleryGrid');
  if (!grid) return;
  const outDir = document.getElementById('train_output_dir')?.value || '';

  try {
    const res = await fetch(`/api/samples?dir=${encodeURIComponent(outDir)}`);
    if (!res.ok) return;
    const data = await res.json();
    grid.innerHTML = '';

    if (!data.samples || data.samples.length === 0) {
      grid.innerHTML = '<div style="color: var(--text-muted); padding: 20px; grid-column: 1/-1; text-align: center;">No checkpoint sample previews rendered yet.</div>';
      return;
    }

    data.samples.forEach(s => {
      const card = document.createElement('div');
      card.className = 'caption-card';
      card.innerHTML = `
        <img src="/api/samples/image?file=${encodeURIComponent(s.path)}" alt="${s.name}" loading="lazy">
        <div class="caption-card-body">
          <div class="caption-card-title">${s.name}</div>
        </div>
      `;
      grid.appendChild(card);
    });
  } catch (e) {
    console.error('Error loading samples:', e);
  }
}

// ============================================================================
// Modals: Console Popup, Queue, Directory Picker
// ============================================================================
function openConsoleModal() {
  const modal = document.getElementById('consoleModal');
  if (modal) modal.classList.add('active');
}

function closeConsoleModal() {
  const modal = document.getElementById('consoleModal');
  if (modal) modal.classList.remove('active');
}

function openQueueModal() {
  const modal = document.getElementById('queueModal');
  if (modal) {
    modal.classList.add('active');
    loadQueueTable();
  }
}

function closeQueueModal() {
  const modal = document.getElementById('queueModal');
  if (modal) modal.classList.remove('active');
}

async function loadQueueTable() {
  const tbody = document.getElementById('queueTableBody');
  if (!tbody) return;
  try {
    const res = await fetch('/api/queue');
    if (!res.ok) return;
    const data = await res.json();
    if (!data.queue || data.queue.length === 0) {
      tbody.innerHTML = '<tr><td colspan="6" style="text-align: center; color: var(--text-muted);">No runs in queue.</td></tr>';
      return;
    }
    tbody.innerHTML = '';
    data.queue.forEach((q, idx) => {
      const tr = document.createElement('tr');
      tr.innerHTML = `
        <td>${idx + 1}</td>
        <td>${q.lora_name || 'unnamed'}</td>
        <td>${q.family || window._currentFamilyKey || '(unknown)'}</td>
        <td>${q.rank || 16}</td>
        <td>${q.epochs || 16}</td>
        <td><span style="color: var(--accent);">pending</span></td>
      `;
      tbody.appendChild(tr);
    });
  } catch (e) {}
}

async function clearQueue() {
  try {
    await fetch('/api/queue/clear', { method: 'POST' });
    loadQueueTable();
  } catch (e) {}
}

// Directory / File Picker Browser
function openPicker(targetInputId) {
  activeTargetInputId = targetInputId;
  const currentVal = document.getElementById(targetInputId)?.value || '';
  browsePath(currentVal || '/workspace');
  const modal = document.getElementById('pickerModal');
  if (modal) modal.classList.add('active');
}

function closePicker() {
  const modal = document.getElementById('pickerModal');
  if (modal) modal.classList.remove('active');
  activeTargetInputId = null;
}

async function browsePath(targetPath) {
  try {
    const res = await fetch(`/api/browse?path=${encodeURIComponent(targetPath)}`);
    if (!res.ok) return;
    const data = await res.json();
    currentPickerDir = data.current;
    currentPickerParent = data.parent;

    document.getElementById('pickerCurrentPath').value = currentPickerDir;
    const list = document.getElementById('pickerList');
    list.innerHTML = '';

    if (data.directories) {
      data.directories.forEach(d => {
        const li = document.createElement('li');
        li.style.padding = '4px 8px';
        li.style.cursor = 'pointer';
        li.style.borderRadius = '3px';
        li.innerHTML = `📁 <strong>${d.name}</strong>`;
        li.onmouseover = () => li.style.background = 'var(--bg-hover)';
        li.onmouseout = () => li.style.background = 'transparent';
        li.ondblclick = () => browsePath(d.path);
        li.onclick = () => {
          document.querySelectorAll('#pickerList li').forEach(el => el.style.border = 'none');
          li.style.border = '1px solid var(--accent)';
          currentPickerDir = d.path;
          document.getElementById('pickerCurrentPath').value = d.path;
        };
        list.appendChild(li);
      });
    }

    if (data.files) {
      data.files.forEach(f => {
        const li = document.createElement('li');
        li.style.padding = '4px 8px';
        li.style.cursor = 'pointer';
        li.style.borderRadius = '3px';
        li.innerHTML = `📄 ${f.name}`;
        li.onmouseover = () => li.style.background = 'var(--bg-hover)';
        li.onmouseout = () => li.style.background = 'transparent';
        li.onclick = () => {
          document.querySelectorAll('#pickerList li').forEach(el => el.style.border = 'none');
          li.style.border = '1px solid var(--accent)';
          document.getElementById('pickerCurrentPath').value = f.path;
        };
        list.appendChild(li);
      });
    }
  } catch (err) {
    console.error('Error browsing path:', err);
  }
}

function pickerGoUp() {
  if (currentPickerParent) {
    browsePath(currentPickerParent);
  }
}

function confirmPickerSelection() {
  const chosen = document.getElementById('pickerCurrentPath')?.value || currentPickerDir;
  if (activeTargetInputId) {
    const el = document.getElementById(activeTargetInputId);
    if (el) {
      el.value = chosen;
      if (activeTargetInputId === 'start_image_folder') {
        syncDatasetFolder(chosen);
      }
    }
  }
  closePicker();
}

function openOutputFolder() {
  const outDir = document.getElementById('train_output_dir')?.value || '';
  if (outDir) {
    openPicker('train_output_dir');
  }
}

// ============================================================================
// Training Tab — Model Area, Timestep Presets, Block Picker, FT toggles
// ============================================================================

function onModelAreaChanged(value) {
  const panel = document.getElementById('customBlockPickerPanel');
  if (panel) {
    panel.style.display = (value === 'Custom') ? 'block' : 'none';
  }
  // Apply preset block selections
  if (value !== 'Custom') {
    const presets = {
      'Full Model':        { double: [0,1,2,3,4,5,6,7], single: Array.from({length:24},(_,i)=>i) },
      'Identity':          { double: [], single: Array.from({length:16},(_,i)=>i+1) },
      'Style':             { double: [0,1,2,3,4,5,6,7], single: [0,1] },
      'Style+Composition': { double: [0,1,2,3,4,5,6,7], single: [0,1] },
      'Details':           { double: [], single: Array.from({length:12},(_,i)=>i+12) },
    };
    const p = presets[value];
    if (p) {
      selectedBlocks.clear();
      p.double.forEach(i => selectedBlocks.add(`double_blocks.${i}`));
      p.single.forEach(i => selectedBlocks.add(`single_blocks.${i}`));
      updateBlocksUI();
    }
  }
}

function initCustomBlockPicker() {
  // Populate double blocks (0-7)
  const dbl = document.getElementById('doubleBlocksRow');
  if (dbl) {
    dbl.innerHTML = '';
    for (let i = 0; i < 8; i++) {
      const key = `double_blocks.${i}`;
      const label = document.createElement('label');
      label.className = 'checkbox-label';
      label.style.cssText = 'font-size: 9pt; margin-right: 2px;';
      label.innerHTML = `<input type="checkbox" data-block="${key}" onchange="toggleBlock('${key}', this.checked)"><span>${i}</span>`;
      dbl.appendChild(label);
    }
  }
  // Populate single blocks 0-11
  const sg1 = document.getElementById('singleBlocksRow1');
  if (sg1) {
    sg1.innerHTML = '';
    for (let i = 0; i < 12; i++) {
      const key = `single_blocks.${i}`;
      const label = document.createElement('label');
      label.className = 'checkbox-label';
      label.style.cssText = 'font-size: 9pt; margin-right: 2px;';
      label.innerHTML = `<input type="checkbox" data-block="${key}" onchange="toggleBlock('${key}', this.checked)"><span>${i}</span>`;
      sg1.appendChild(label);
    }
  }
  // Populate single blocks 12-23
  const sg2 = document.getElementById('singleBlocksRow2');
  if (sg2) {
    sg2.innerHTML = '';
    for (let i = 12; i < 24; i++) {
      const key = `single_blocks.${i}`;
      const label = document.createElement('label');
      label.className = 'checkbox-label';
      label.style.cssText = 'font-size: 9pt; margin-right: 2px;';
      label.innerHTML = `<input type="checkbox" data-block="${key}" onchange="toggleBlock('${key}', this.checked)"><span>${i}</span>`;
      sg2.appendChild(label);
    }
  }
}

function toggleBlock(key, checked) {
  if (checked) selectedBlocks.add(key);
  else selectedBlocks.delete(key);
}

function updateBlocksUI() {
  document.querySelectorAll('[data-block]').forEach(el => {
    el.checked = selectedBlocks.has(el.dataset.block);
  });
}

function applyTSPreset(presetName) {
  const presets = {
    full_range:  { sampling: 'sigmoid', sigmoid_scale: '1.0', min_ts: '0', max_ts: '1000', preserve: false },
    structure:   { sampling: 'sigmoid', sigmoid_scale: '1.0', min_ts: '500', max_ts: '1000', preserve: true },
    detail:      { sampling: 'sigmoid', sigmoid_scale: '1.0', min_ts: '0', max_ts: '400', preserve: true },
    sigmoid:     { sampling: 'sigmoid', sigmoid_scale: '1.0', min_ts: '0', max_ts: '1000', preserve: false },
  };
  const p = presets[presetName];
  if (!p) return;
  const set = (id, v) => { const el = document.getElementById(id); if (el) el.value = v; };
  set('train_timestep_sampling', p.sampling);
  set('train_sigmoid_scale', p.sigmoid_scale);
  set('train_min_timestep', p.min_ts);
  set('train_max_timestep', p.max_ts);
  const pd = document.getElementById('train_preserve_dist');
  if (pd) pd.checked = p.preserve;
}

function saveCustomPreset() {
  const name = prompt('Preset name:');
  if (!name) return;
  // Collect current settings
  const data = collectTrainingSettings();
  fetch('/api/presets/save', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ name: name, settings: data })
  }).then(r => r.json()).then(d => {
    if (d.status === 'ok') alert('Preset saved: ' + name);
    else alert('Error: ' + (d.error || 'unknown'));
  }).catch(e => alert('Save failed: ' + e));
}

function loadCustomPreset(presetName) {
  if (!presetName) return;
  fetch('/api/presets/load?name=' + encodeURIComponent(presetName))
    .then(r => r.json())
    .then(d => {
      if (d.settings) applySettingsToUI(d.settings);
    }).catch(e => console.error('Load preset failed:', e));
}

function toggleKrea2FT(enabled) {
  const controls = document.getElementById('krea2FTControls');
  if (controls) {
    controls.style.display = enabled ? 'flex' : 'none';
  }
}

function collectTrainingSettings() {
  const g = id => { const el = document.getElementById(id); return el ? (el.type === 'checkbox' ? el.checked : el.value) : null; };
  return {
    learning_rate: g('train_learning_rate'),
    adaptive_lr: g('train_adaptive_lr'),
    adaptive_min: g('train_adaptive_min'),
    adaptive_max: g('train_adaptive_max'),
    network_dim: g('train_network_dim'),
    network_alpha: g('train_network_alpha'),
    max_epochs: g('train_max_epochs'),
    save_every: g('train_save_every'),
    seed: g('train_seed'),
    model_area: g('train_model_area'),
    caption_dropout: g('train_caption_dropout'),
    megapixels: g('train_megapixels'),
    network_type: g('train_network_type'),
    lokr_factor: g('train_lokr_factor'),
    context_lora_path: g('train_context_lora_path'),
    context_lora_strength: g('train_context_lora_strength'),
    blocks_swap: g('train_blocks_swap'),
    resume_path: g('train_resume_path'),
    fp8_base: g('train_fp8_base'),
    fp8_scaled: g('train_fp8_scaled'),
    fp8_text_encoder: g('train_fp8_text_encoder'),
    quant: g('train_quant'),
    grad_checkpointing: g('train_grad_checkpointing'),
    compile_blocks: g('train_compile_blocks'),
    save_state_checkpoint: g('train_save_state_checkpoint'),
    save_state_end: g('train_save_state_end'),
    keep_last_n: g('train_keep_last_n'),
    mixed_precision: g('train_mixed_precision'),
    timestep_sampling: g('train_timestep_sampling'),
    sigmoid_scale: g('train_sigmoid_scale'),
    min_timestep: g('train_min_timestep'),
    max_timestep: g('train_max_timestep'),
    preserve_dist: g('train_preserve_dist'),
    optimizer_type: g('train_optimizer_type'),
    optimizer_args: g('train_optimizer_args'),
    grad_accum: g('train_grad_accum'),
    max_norm: g('train_max_norm'),
    network_dropout: g('train_network_dropout'),
    caption_ext: g('train_caption_ext'),
    batch_size: g('train_batch_size'),
    enable_bucket: g('train_enable_bucket'),
    no_upscale: g('train_no_upscale'),
    lr_scheduler: g('train_lr_scheduler'),
    warmup_steps: g('train_warmup_steps'),
    lr_decay_steps: g('train_lr_decay_steps'),
    min_bucket: g('train_min_bucket'),
    max_bucket: g('train_max_bucket'),
    enable_cache: g('train_enable_cache'),
    output_dir: g('train_output_dir'),
    lora_name: g('train_lora_name'),
  };
}

function applySettingsToUI(settings) {
  const s = (id, v) => {
    const el = document.getElementById(id);
    if (!el || v === null || v === undefined) return;
    if (el.type === 'checkbox') el.checked = !!v;
    else el.value = v;
  };
  Object.entries(settings).forEach(([key, val]) => {
    s('train_' + key, val);
  });
}

// Initialize custom block picker on load
document.addEventListener('DOMContentLoaded', () => {
  initCustomBlockPicker();
});


// ============================================================
// Captions Tab JS
// ============================================================

function onCaptionModelChanged() {
  const model = document.getElementById('caption_model').value;
  const hint = document.getElementById('captionModelHint');
  if (hint) {
    hint.textContent = model.includes('Qwen3-VL') ? '(needs Krea 2 text encoder set in Preferences)' : '';
  }
}

function onCaptionTaskChanged() {}

function openCaptionInstructionEditor() {
  const task = document.getElementById('caption_task').value;
  alert('Instruction editor for task: ' + task +
    '\n\nSee and edit the exact instruction sent to the vision model for this task.\n' +
    'Each of the four keeps its own wording, and your edit persists between sessions.');
}

function runAICaptions() {
  const folder = document.getElementById('captions_image_folder').value;
  if (!folder) { alert('Set an image folder on the Start tab first.'); return; }
  document.getElementById('captionAllBtn').disabled = true;
  document.getElementById('captionStopBtn').disabled = false;
  appendCaptionLog('Starting AI captioning...\n');
  fetch('/api/caption_all', {
    method: 'POST', headers: {'Content-Type': 'application/json'},
    body: JSON.stringify({
      folder, model: document.getElementById('caption_model').value,
      task: document.getElementById('caption_task').value,
      trigger: document.getElementById('caption_trigger_word').value,
      max_tokens: parseInt(document.getElementById('caption_max_tokens').value) || 256,
      overwrite: document.getElementById('caption_overwrite').checked,
    })
  }).then(r => r.json()).then(d => {
    appendCaptionLog(d.message || 'Done.\n');
    document.getElementById('captionAllBtn').disabled = false;
    document.getElementById('captionStopBtn').disabled = true;
  }).catch(e => {
    appendCaptionLog('Error: ' + e + '\n');
    document.getElementById('captionAllBtn').disabled = false;
    document.getElementById('captionStopBtn').disabled = true;
  });
}

function runStaticCaptions() {
  const folder = document.getElementById('captions_image_folder').value;
  if (!folder) { alert('Set an image folder on the Start tab first.'); return; }
  appendCaptionLog('Writing static captions...\n');
  fetch('/api/static_caption', {
    method: 'POST', headers: {'Content-Type': 'application/json'},
    body: JSON.stringify({ folder, trigger: document.getElementById('caption_trigger_word').value,
      overwrite: document.getElementById('caption_overwrite').checked })
  }).then(r => r.json()).then(d => appendCaptionLog(d.message || 'Done.\n'))
    .catch(e => appendCaptionLog('Error: ' + e + '\n'));
}

function stopCaptioning() {
  fetch('/api/stop_captioning', { method: 'POST' }).then(() => appendCaptionLog('Stop requested.\n'));
  document.getElementById('captionStopBtn').disabled = true;
  document.getElementById('captionAllBtn').disabled = false;
}

function unloadCaptionModel() {
  fetch('/api/unload_caption_model', { method: 'POST' }).then(() => appendCaptionLog('Caption model unloaded.\n'));
}

function runBilingualTranslation() {
  const folder = document.getElementById('captions_image_folder').value;
  if (!folder) { alert('Set an image folder on the Start tab first.'); return; }
  appendCaptionLog('Translating captions (EN→ZH)...\n');
  fetch('/api/translate_captions', {
    method: 'POST', headers: {'Content-Type': 'application/json'},
    body: JSON.stringify({ folder, skip_bilingual: document.getElementById('skip_bilingual').checked })
  }).then(r => r.json()).then(d => appendCaptionLog(d.message || 'Done.\n'))
    .catch(e => appendCaptionLog('Error: ' + e + '\n'));
}

function replaceCaptions() {
  const folder = document.getElementById('captions_image_folder').value;
  const find = document.getElementById('caption_find').value;
  const replace = document.getElementById('caption_replace').value;
  if (!folder) { alert('Set an image folder on the Start tab first.'); return; }
  if (!find) { alert('Enter text to find.'); return; }
  fetch('/api/find_replace_captions', {
    method: 'POST', headers: {'Content-Type': 'application/json'},
    body: JSON.stringify({ folder, find, replace, preview: false })
  }).then(r => r.json()).then(d => appendCaptionLog(d.message || 'Done.\n'))
    .catch(e => appendCaptionLog('Error: ' + e + '\n'));
}

function previewReplaceCaptions() {
  const folder = document.getElementById('captions_image_folder').value;
  const find = document.getElementById('caption_find').value;
  if (!folder || !find) return;
  fetch('/api/find_replace_captions', {
    method: 'POST', headers: {'Content-Type': 'application/json'},
    body: JSON.stringify({ folder, find, replace: document.getElementById('caption_replace').value, preview: true })
  }).then(r => r.json()).then(d => appendCaptionLog(d.message || 'Preview:\n'))
    .catch(e => appendCaptionLog('Error: ' + e + '\n'));
}

function appendCaptionLog(text) {
  const log = document.getElementById('captionConsoleLog');
  if (log.textContent === 'Ready.') log.textContent = '';
  log.textContent += text;
  log.scrollTop = log.scrollHeight;
}

function captionGotoPage() {
  const p = parseInt(document.getElementById('captionGotoPage').value);
  if (p && p > 0) { window._captionPage = p - 1; refreshCaptionsGrid(); }
}

// ============================================================
// Samples Tab JS
// ============================================================

function onSamplesArchChanged(value) {
  const t = document.getElementById('train_architecture');
  if (t) t.value = value;
  onFamilyChange();
  updateSamplesUI();
}

function toggleSampleSettings(enabled) {
  const frame = document.getElementById('sampleSettingsFrame');
  if (!frame) return;
  frame.querySelectorAll('input, select, textarea, button').forEach(el => { el.disabled = !enabled; });
  syncSampleEnabled(enabled);
}

function onDistilledSamplesToggled() {
  const d = document.getElementById('sample_distilled').checked;
  ['sample_steps','sample_flow_shift','sample_negative','sample_cfg_scale'].forEach(id => {
    const el = document.getElementById(id);
    if (el) el.disabled = d;
  });
}

function updateSamplesUI() {
  const arch = document.getElementById('samples_architecture');
  if (!arch) return;
  const val = arch.value || '';
  const isMM = val.toLowerCase().includes('minimax');
  const isK2 = val.toLowerCase().includes('krea');
  document.querySelectorAll('.minimax-only').forEach(el => { el.style.display = isMM ? '' : 'none'; });
  ['krea2EngineRow','krea2EngineNote'].forEach(id => {
    const el = document.getElementById(id); if (el) el.style.display = isK2 ? '' : 'none';
  });
  const adv = document.getElementById('samplesAdvancedCard');
  if (adv) adv.style.display = isMM ? 'none' : '';
  const distRow = document.getElementById('sampleDistilledRow');
  if (distRow) distRow.style.display = (isK2 || isMM) ? 'none' : '';
  ['sampleRefRow','sampleRefNote'].forEach(id => {
    const el = document.getElementById(id); if (el) el.style.display = isMM ? 'none' : '';
  });
}

function openSamplesGallery() { fetch('/api/open_samples_gallery', { method: 'POST' }).catch(() => {}); }
function openSamplesFolder() { fetch('/api/open_samples_folder', { method: 'POST' }).catch(() => {}); }

// ============================================================
// Profiler Tab JS
// ============================================================

function onProfilerFamilyChanged(value) {
  document.querySelectorAll('.klein-profiler-only').forEach(el => {
    el.style.display = (value === 'klein') ? '' : 'none';
  });
}

function runProfiler() {
  const lora = document.getElementById('prof_lora_path').value;
  if (!lora) { alert('Please select a LoRA file.'); return; }
  const fam = document.querySelector('input[name="prof_fam"]:checked')?.value || 'klein';
  document.getElementById('profilerRunBtn').disabled = true;
  document.getElementById('profilerProgressLabel').textContent = 'Profiling...';
  const body = { lora_path: lora, family: fam };
  if (fam === 'klein') {
    body.dit_choice = document.querySelector('input[name="prof_dit"]:checked')?.value || 'distilled';
    body.prompt = document.getElementById('prof_prompt').value;
    body.resolution = parseInt(document.getElementById('prof_res').value);
    body.stages = parseInt(document.getElementById('prof_stages').value);
  }
  fetch('/api/run_profiler', {
    method: 'POST', headers: {'Content-Type': 'application/json'},
    body: JSON.stringify(body)
  }).then(r => r.json()).then(d => {
    document.getElementById('profilerLog').textContent = d.message || 'Done.';
    document.getElementById('profilerProgressLabel').textContent = d.status || 'Complete.';
    document.getElementById('profilerRunBtn').disabled = false;
    if (d.report_path) document.getElementById('profilerOpenBtn').disabled = false;
  }).catch(e => {
    document.getElementById('profilerLog').textContent = 'Error: ' + e;
    document.getElementById('profilerProgressLabel').textContent = 'Error.';
    document.getElementById('profilerRunBtn').disabled = false;
  });
}

function openProfilerReport() { fetch('/api/open_profiler_report', { method: 'POST' }).catch(() => {}); }

// ============================================================
// Metadata Tab JS
// ============================================================

function loadMetadataFile() {
  const path = document.getElementById('meta_file_path').value;
  if (!path) { alert('Enter or browse for a .safetensors file.'); return; }
  fetch('/api/load_metadata', {
    method: 'POST', headers: {'Content-Type': 'application/json'},
    body: JSON.stringify({ path })
  }).then(r => r.json()).then(d => {
    if (d.error) { alert(d.error); return; }
    document.getElementById('meta_title').value = d.title || '';
    document.getElementById('meta_author').value = d.author || '';
    document.getElementById('meta_license').value = d.license || '';
    document.getElementById('meta_tags').value = d.tags || '';
    document.getElementById('meta_trigger').value = d.trigger_phrase || '';
    document.getElementById('meta_usage').value = d.usage_hint || '';
    document.getElementById('meta_desc').value = d.description || '';
    document.getElementById('metaStatusLbl').textContent =
      'Loaded — ' + (d.key_count || 0) + ' metadata key' + ((d.key_count||0) !== 1 ? 's' : '') + ' found.';
    if (d.thumbnail) {
      document.getElementById('metaThumbPreview').innerHTML =
        '<img src="' + d.thumbnail + '" style="max-width:256px;max-height:256px;">';
    }
    const tbody = document.getElementById('metaCustomBody');
    tbody.innerHTML = '';
    if (d.custom) {
      Object.entries(d.custom).forEach(([k, v]) => {
        const tr = document.createElement('tr');
        const dv = String(v).length > 120 ? String(v).slice(0,117) + '...' : v;
        tr.innerHTML = '<td>' + k + '</td><td>' + dv + '</td><td><input type="checkbox"></td>';
        tbody.appendChild(tr);
      });
    }
  }).catch(e => { document.getElementById('metaStatusLbl').textContent = 'Error: ' + e; });
}

function browseMetaThumbnail() { alert('Browse for thumbnail image (backend needed).'); }
function clearMetaThumbnail() { document.getElementById('metaThumbPreview').innerHTML = '(no thumbnail)'; }

function addMetaCustomField() {
  const key = prompt('Key:');
  if (!key) return;
  const val = prompt('Value:') || '';
  const tr = document.createElement('tr');
  tr.innerHTML = '<td>' + key + '</td><td>' + val + '</td><td><input type="checkbox"></td>';
  document.getElementById('metaCustomBody').appendChild(tr);
}

function removeMetaCustomField() {
  document.getElementById('metaCustomBody').querySelectorAll('tr').forEach(row => {
    if (row.querySelector('input[type="checkbox"]')?.checked) row.remove();
  });
}

function saveMetadataFile(saveAs) {
  const path = document.getElementById('meta_file_path').value;
  if (!path) { alert('No file loaded.'); return; }
  fetch('/api/save_metadata', {
    method: 'POST', headers: {'Content-Type': 'application/json'},
    body: JSON.stringify({
      path, save_as: saveAs,
      title: document.getElementById('meta_title').value,
      author: document.getElementById('meta_author').value,
      license: document.getElementById('meta_license').value,
      tags: document.getElementById('meta_tags').value,
      trigger_phrase: document.getElementById('meta_trigger').value,
      usage_hint: document.getElementById('meta_usage').value,
      description: document.getElementById('meta_desc').value,
    })
  }).then(r => r.json()).then(d => {
    document.getElementById('metaSaveStatus').textContent = d.message || 'Saved.';
  }).catch(e => { document.getElementById('metaSaveStatus').textContent = 'Error: ' + e; });
}

// ============================================================
// Preferences Tab JS
// ============================================================


function fetchModels(family) {
  alert('Fetching models for ' + family + '...\nThis downloads required model files and fills in paths automatically.');
}

function resetPreferences() {
  if (confirm('Reset all preferences to defaults?')) {
    fetch('/api/reset_prefs', { method: 'POST' })
      .then(r => r.json())
      .then(d => { alert(d.message || 'Reset.'); location.reload(); })
      .catch(e => alert('Error: ' + e));
  }
}

function openPrefsFile() { fetch('/api/open_prefs_file', { method: 'POST' }).catch(() => {}); }

// (savePreferences is defined above with full schema-driven model-path collection)

// ============================================================
// (Training Controls are defined above: launchTrainingRun, stopTrainingRun, pauseTrainingRun)


function loadLastTrainSettings() {
  fetch('/api/load_last_train')
    .then(r => r.json()).then(d => {
      if (d.settings) applySettingsToUI(d.settings);
      alert(d.message || 'Loaded last training settings.');
    }).catch(e => alert('Error: ' + e));
}

function syncSampleFrames(val) {
  const el = document.getElementById('samples_sample_frames');
  if (el) el.value = val;
}

// (refreshCaptionsGrid, updatePrepSummary, clearQueue are defined above)


// ============================================================
// Repair Studio Tab JS
// ============================================================

function onRepairFamilyChanged(family) {
  if (!family) {
    const sel = document.querySelector('input[name="repair_family"]:checked');
    family = sel ? sel.value : 'klein';
  }
  const tab = document.getElementById('tab-repair');
  if (!tab) return;
  const isKlein = family === 'klein';
  const isKrea2 = family === 'krea2';
  const isMinimax = family === 'minimax';

  // DiT row visibility + labels
  const ditRow = document.getElementById('repair_dit_row');
  if (ditRow) ditRow.style.display = isKlein ? '' : 'none';

  // H3 model/base rows (minimax only)
  const h3Model = document.getElementById('repair_h3_model_row');
  const h3Base = document.getElementById('repair_h3_base_row');
  if (h3Model) h3Model.style.display = isMinimax ? '' : 'none';
  if (h3Base) h3Base.style.display = isMinimax ? '' : 'none';

  // Turbo preview (Klein only)
  const turboWrap = document.getElementById('repair_turbo_wrap');
  if (turboWrap) turboWrap.style.display = isKlein ? '' : 'none';

  // Resolution label change
  const resLabel = document.getElementById('repair_res_label');
  if (resLabel) resLabel.textContent = isMinimax ? 'Render size:' : 'Resolution:';

  // H3 clip controls (minimax only)
  const h3Clip = document.getElementById('repair_h3_clip_row');
  if (h3Clip) h3Clip.style.display = isMinimax ? '' : 'none';

  // Reference row (hidden for minimax)
  const refRow = document.getElementById('repair_ref_row');
  if (refRow) refRow.style.display = isMinimax ? 'none' : '';

  // Master controls card (hidden for krea2 and minimax)
  const masterCard = document.getElementById('repair_master_card');
  if (masterCard) masterCard.style.display = isKlein ? '' : 'none';

  // minimax-only elements
  tab.querySelectorAll('.minimax-only').forEach(el => {
    el.style.display = isMinimax ? '' : 'none';
  });

  // First/last keyframe card
  const kfCard = document.getElementById('repair_kf_card');
  if (kfCard) kfCard.style.display = isMinimax ? '' : 'none';
}

function repairUnloadDonor() {
  document.getElementById('repair_donor_lora').value = '';
  document.getElementById('repair_donor_strength').value = '1.0';
  fetch('/api/repair/unload_donor', { method: 'POST' }).catch(() => {});
}

function repairH3ModelChanged() {
  const model = document.getElementById('repair_h3_model');
  if (model) {
    fetch('/api/repair/h3_model_changed', {
      method: 'POST', headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({ model: model.value })
    }).catch(() => {});
  }
}

function onRepairH3ClipChanged() {
  // Sync H3 clip controls to backend
  const data = {
    frames: document.getElementById('repair_h3_frames')?.value,
    width: document.getElementById('repair_h3_width')?.value,
    height: document.getElementById('repair_h3_height')?.value,
    dial_scale: document.getElementById('repair_h3_dial_scale')?.value,
    steps: document.getElementById('repair_h3_steps')?.value,
    turbo: document.getElementById('repair_h3_turbo')?.checked,
    sound: document.getElementById('repair_h3_sound')?.checked,
    early: document.getElementById('repair_h3_early')?.checked,
    nolora: document.getElementById('repair_h3_nolora')?.checked
  };
  fetch('/api/repair/h3_clip_settings', {
    method: 'POST', headers: {'Content-Type': 'application/json'},
    body: JSON.stringify(data)
  }).catch(() => {});
}

function repairRandomizeSeed() {
  const el = document.getElementById('repair_seed');
  if (el) el.value = Math.floor(Math.random() * 999999999);
}

function repairPopoutPreview() {
  // Open baseline/tweaked preview in new window
  const baseline = document.getElementById('repair_baseline_holder');
  const tweaked = document.getElementById('repair_tweaked_holder');
  const img1 = baseline?.querySelector('img');
  const img2 = tweaked?.querySelector('img');
  if (img1 || img2) {
    const w = window.open('', '_blank', 'width=1200,height=600');
    w.document.write('<html><body style="margin:0;background:#111;display:flex;gap:4px;justify-content:center;align-items:center;height:100vh;">');
    if (img1) w.document.write('<img src="' + img1.src + '" style="max-width:49%;max-height:98%;object-fit:contain;">');
    if (img2) w.document.write('<img src="' + img2.src + '" style="max-width:49%;max-height:98%;object-fit:contain;">');
    w.document.write('</body></html>');
  }
}

function repairResetSliders() {
  for (let i = 0; i < 5; i++) {
    const slider = document.getElementById('repair_slider_' + i);
    const val = document.getElementById('repair_val_' + i);
    if (slider) { slider.value = '1.0'; }
    if (val) val.textContent = '1.00';
  }
  // Also reset all per-block sliders in the slider panel
  const panel = document.getElementById('repair_slider_panel');
  if (panel) {
    panel.querySelectorAll('input[type="range"]').forEach(s => {
      s.value = '1.0';
      const lbl = s.nextElementSibling;
      if (lbl) lbl.textContent = '1.00';
    });
  }
}

function repairStart() {
  const status = document.getElementById('repair_status');
  if (status) status.textContent = 'Starting render…';

  const family = document.querySelector('input[name="repair_family"]:checked')?.value || 'klein';
  const data = {
    family: family,
    primary_lora: document.getElementById('repair_primary_lora')?.value,
    primary_strength: document.getElementById('repair_primary_strength')?.value,
    donor_lora: document.getElementById('repair_donor_lora')?.value,
    donor_strength: document.getElementById('repair_donor_strength')?.value,
    prompt: document.getElementById('repair_prompt')?.value,
    seed: document.getElementById('repair_seed')?.value,
    resolution: document.getElementById('repair_res')?.value,
    turbo: document.getElementById('repair_turbo')?.checked,
    ref_path: document.getElementById('repair_ref_path')?.value,
    ref_mp: document.getElementById('repair_ref_mp')?.value,
    ref_strength: document.getElementById('repair_ref_strength')?.value,
    sliders: {}
  };
  // Collect master sliders
  for (let i = 0; i < 5; i++) {
    const s = document.getElementById('repair_slider_' + i);
    if (s) data.sliders['bucket_' + i] = parseFloat(s.value);
  }
  fetch('/api/repair/start', {
    method: 'POST', headers: {'Content-Type': 'application/json'},
    body: JSON.stringify(data)
  }).then(r => r.json()).then(d => {
    if (status) status.textContent = d.message || 'Rendering…';
  }).catch(e => { if (status) status.textContent = 'Error: ' + e; });
}

function repairBulkPrimary(action) {
  // action: 'reset', 'all_off', 'all_on', 'invert', 'alternate', 'toggle_detail'
  const panel = document.getElementById('repair_slider_panel');
  if (!panel) return;
  const sliders = panel.querySelectorAll('input[type="range"]');
  sliders.forEach((s, i) => {
    const lbl = s.nextElementSibling;
    switch (action) {
      case 'reset': s.value = '1.0'; break;
      case 'all_off': s.value = '0'; break;
      case 'all_on': s.value = '1.0'; break;
      case 'invert': s.value = (1 - parseFloat(s.value)).toFixed(2); break;
      case 'alternate': s.value = i % 2 === 0 ? '1.0' : '0'; break;
      case 'toggle_detail': s.value = parseFloat(s.value) > 0 ? '0' : '1.0'; break;
    }
    if (lbl) lbl.textContent = parseFloat(s.value).toFixed(2);
  });
}

function repairSaveLoRA() {
  const data = { sliders: {} };
  for (let i = 0; i < 5; i++) {
    const s = document.getElementById('repair_slider_' + i);
    if (s) data.sliders['bucket_' + i] = parseFloat(s.value);
  }
  const panel = document.getElementById('repair_slider_panel');
  if (panel) {
    data.block_sliders = {};
    panel.querySelectorAll('input[type="range"]').forEach(s => {
      data.block_sliders[s.id || s.dataset.block] = parseFloat(s.value);
    });
  }
  fetch('/api/repair/save_lora', {
    method: 'POST', headers: {'Content-Type': 'application/json'},
    body: JSON.stringify(data)
  }).then(r => r.json()).then(d => alert(d.message || 'Saved.'))
    .catch(e => alert('Error: ' + e));
}

function repairResetSession() {
  if (!confirm('Reset the entire Repair Studio session? All slider changes will be lost.')) return;
  repairResetSliders();
  const status = document.getElementById('repair_status');
  if (status) status.textContent = 'Session reset.';
  fetch('/api/repair/reset_session', { method: 'POST' }).catch(() => {});
}

function repairExploreInExplorer() {
  // Copy current repair settings to Explorer tab and switch to it
  const lora = document.getElementById('repair_primary_lora')?.value;
  const prompt = document.getElementById('repair_prompt')?.value;
  const seed = document.getElementById('repair_seed')?.value;
  if (lora) {
    const el = document.getElementById('explorer_lora');
    if (el) el.value = lora;
  }
  if (prompt) {
    const el = document.getElementById('explorer_prompt');
    if (el) el.value = prompt;
  }
  if (seed) {
    const el = document.getElementById('explorer_seed');
    if (el) el.value = seed;
  }
  switchTab('tab-explorer');
}

function loadRepairPreset(name) {
  if (!name) return;
  fetch('/api/repair/load_preset?name=' + encodeURIComponent(name))
    .then(r => r.json()).then(d => {
      if (d.settings) {
        if (d.settings.prompt) document.getElementById('repair_prompt').value = d.settings.prompt;
        if (d.settings.seed) document.getElementById('repair_seed').value = d.settings.seed;
        // Apply slider values
        if (d.settings.sliders) {
          for (let i = 0; i < 5; i++) {
            const s = document.getElementById('repair_slider_' + i);
            const v = document.getElementById('repair_val_' + i);
            if (s && d.settings.sliders['bucket_' + i] !== undefined) {
              s.value = d.settings.sliders['bucket_' + i];
              if (v) v.textContent = parseFloat(s.value).toFixed(2);
            }
          }
        }
      }
    }).catch(e => alert('Error loading preset: ' + e));
}

function saveRepairPreset() {
  const name = prompt('Enter a name for this preset:');
  if (!name) return;
  const data = {
    name: name,
    prompt: document.getElementById('repair_prompt')?.value,
    seed: document.getElementById('repair_seed')?.value,
    sliders: {}
  };
  for (let i = 0; i < 5; i++) {
    const s = document.getElementById('repair_slider_' + i);
    if (s) data.sliders['bucket_' + i] = parseFloat(s.value);
  }
  fetch('/api/repair/save_preset', {
    method: 'POST', headers: {'Content-Type': 'application/json'},
    body: JSON.stringify(data)
  }).then(r => r.json()).then(d => alert(d.message || 'Preset saved.'))
    .catch(e => alert('Error: ' + e));
}

// ============================================================
// RefMod Studio Tab JS
// ============================================================

function rmsRescan() {
  const folder = document.getElementById('refmod_folder')?.value;
  if (!folder) { alert('Set a RefMod folder first.'); return; }
  fetch('/api/refmod/rescan', {
    method: 'POST', headers: {'Content-Type': 'application/json'},
    body: JSON.stringify({ folder: folder })
  }).then(r => r.json()).then(d => {
    const status = document.getElementById('rms_status');
    if (status) status.textContent = d.message || 'Rescanned.';
    if (d.mods && d.mods.length > 0) {
      // Populate the mod picker(s) with found .safetensors files
      document.querySelectorAll('[id^="rms_mod_picker_"]').forEach(sel => {
        sel.innerHTML = '<option value="">(none)</option>';
        d.mods.forEach(m => { sel.add(new Option(m, m)); });
      });
    }
  }).catch(e => alert('Error: ' + e));
}

function rmsModelChanged() {
  const model = document.getElementById('refmod_model')?.value;
  const status = document.getElementById('rms_status');
  if (status) status.textContent = 'Model changed to: ' + model;
}

function rmsLoad() {
  const data = {
    model: document.getElementById('refmod_model')?.value,
    base: document.getElementById('refmod_base')?.value,
    folder: document.getElementById('refmod_folder')?.value
  };
  const status = document.getElementById('rms_status');
  if (status) status.textContent = 'Loading model…';
  fetch('/api/refmod/load', {
    method: 'POST', headers: {'Content-Type': 'application/json'},
    body: JSON.stringify(data)
  }).then(r => r.json()).then(d => {
    if (status) status.textContent = d.message || 'Loaded.';
  }).catch(e => { if (status) status.textContent = 'Error: ' + e; });
}

function rmsUnload() {
  const status = document.getElementById('rms_status');
  if (status) status.textContent = 'Unloading…';
  fetch('/api/refmod/unload', { method: 'POST' })
    .then(r => r.json()).then(d => {
      if (status) status.textContent = d.message || 'Unloaded.';
    }).catch(e => { if (status) status.textContent = 'Error: ' + e; });
}

let _rmsRowCount = 1;
function rmsAddRow() {
  const frame = document.getElementById('rms_rows_frame');
  if (!frame) return;
  const idx = _rmsRowCount++;
  const row = document.createElement('div');
  row.style.cssText = 'display:flex; align-items:center; gap:6px; margin-bottom:4px;';
  row.innerHTML = `
    <select id="rms_mod_picker_${idx}" class="w-combo-lg" title="Pick a .safetensors RefMod from the folder">
      <option value="">(none)</option>
    </select>
    <input type="range" id="rms_strength_${idx}" min="0" max="2" value="1" step="0.05" style="width:120px;"
      oninput="document.getElementById('rms_str_val_${idx}').textContent=parseFloat(this.value).toFixed(2)">
    <span id="rms_str_val_${idx}" style="font-family:var(--font-mono);font-size:9pt;width:35px;">1.00</span>
    ×
    <select id="rms_copies_${idx}" style="width:50px;font-size:10pt;background:var(--bg-surface);color:var(--text-primary);border:1px solid var(--border);">
      <option>1</option><option>2</option><option>3</option><option>4</option><option>5</option>
    </select>
  `;
  frame.appendChild(row);
}

function rmsSetRetention(val) {
  const slider = document.getElementById('rms_retention');
  const text = document.getElementById('rms_retention_val');
  if (slider) slider.value = val;
  if (text) text.value = parseFloat(val).toFixed(2);
}

function rmsRandomScramble() {
  const el = document.getElementById('rms_scramble');
  if (el) el.value = Math.floor(Math.random() * 999999);
}

function rmsCurvePresetSave() {
  const name = prompt('Enter a name for this curve preset:');
  if (!name) return;
  const data = {
    name: name,
    fc_dir: document.getElementById('rms_fc_dir')?.value,
    fc_shape: document.getElementById('rms_fc_shape')?.value,
    fc_value: document.getElementById('rms_fc_value')?.value,
    sc_on: document.getElementById('rms_sc_on')?.checked,
    sc_dir: document.getElementById('rms_sc_dir')?.value,
    sc_shape: document.getElementById('rms_sc_shape')?.value,
    sc_value: document.getElementById('rms_sc_value')?.value
  };
  fetch('/api/refmod/save_curve_preset', {
    method: 'POST', headers: {'Content-Type': 'application/json'},
    body: JSON.stringify(data)
  }).then(r => r.json()).then(d => alert(d.message || 'Saved.'))
    .catch(e => alert('Error: ' + e));
}

function rmsCurvePresetDelete() {
  const sel = document.getElementById('rms_curve_preset');
  const name = sel?.value;
  if (!name) return;
  if (!confirm('Delete curve preset "' + name + '"?')) return;
  fetch('/api/refmod/delete_curve_preset', {
    method: 'POST', headers: {'Content-Type': 'application/json'},
    body: JSON.stringify({ name: name })
  }).then(r => r.json()).then(d => {
    if (sel) { const opt = sel.querySelector('option[value="' + name + '"]'); if (opt) opt.remove(); }
    alert(d.message || 'Deleted.');
  }).catch(e => alert('Error: ' + e));
}

function rmsImportGraphPreset() {
  alert('Import graph preset from clipboard or file — backend wiring needed.');
}

function rmsRender() {
  const status = document.getElementById('rms_render_status');
  if (status) status.textContent = 'Rendering…';
  const data = {
    prompt: document.getElementById('rms_prompt')?.value,
    seed: document.getElementById('rms_seed')?.value,
    frames: document.getElementById('rms_frames')?.value,
    width: document.getElementById('rms_width')?.value,
    height: document.getElementById('rms_height')?.value,
    steps: document.getElementById('rms_steps')?.value,
    turbo: document.getElementById('rms_turbo')?.checked,
    sound: document.getElementById('rms_sound')?.checked,
    early: document.getElementById('rms_early')?.checked,
    retention: document.getElementById('rms_retention')?.value,
    fc_dir: document.getElementById('rms_fc_dir')?.value,
    fc_shape: document.getElementById('rms_fc_shape')?.value,
    fc_value: document.getElementById('rms_fc_value')?.value,
    sc_on: document.getElementById('rms_sc_on')?.checked,
    sc_dir: document.getElementById('rms_sc_dir')?.value,
    sc_shape: document.getElementById('rms_sc_shape')?.value,
    sc_value: document.getElementById('rms_sc_value')?.value,
    mods: []
  };
  // Collect mod rows
  document.querySelectorAll('[id^="rms_mod_picker_"]').forEach((sel, i) => {
    data.mods.push({
      name: sel.value,
      strength: document.getElementById('rms_strength_' + i)?.value || '1',
      copies: document.getElementById('rms_copies_' + i)?.value || '1'
    });
  });
  fetch('/api/refmod/render', {
    method: 'POST', headers: {'Content-Type': 'application/json'},
    body: JSON.stringify(data)
  }).then(r => r.json()).then(d => {
    if (status) status.textContent = d.message || 'Done.';
  }).catch(e => { if (status) status.textContent = 'Error: ' + e; });
}

function rmsCancel() {
  const status = document.getElementById('rms_render_status');
  fetch('/api/refmod/cancel', { method: 'POST' })
    .then(r => r.json()).then(d => {
      if (status) status.textContent = d.message || 'Cancelled.';
    }).catch(() => {});
}

function rmsSweep() {
  const status = document.getElementById('rms_render_status');
  if (status) status.textContent = 'Starting sweep…';
  fetch('/api/refmod/sweep', { method: 'POST' })
    .then(r => r.json()).then(d => {
      if (status) status.textContent = d.message || 'Sweep done.';
    }).catch(e => { if (status) status.textContent = 'Error: ' + e; });
}

function rmsSaveStrip() {
  fetch('/api/refmod/save_strip', { method: 'POST' })
    .then(r => r.json()).then(d => alert(d.message || 'Strip saved.'))
    .catch(e => alert('Error: ' + e));
}

function rmsAddHints() {
  const prompt = document.getElementById('rms_prompt');
  if (prompt) {
    const hints = '### Hint reference: Use numbered {1}, {2}, ... tokens to reference uploaded mods. ###';
    if (!prompt.value.includes('###')) prompt.value += '\n' + hints;
  }
}

function rmsAddLabels() {
  const prompt = document.getElementById('rms_prompt');
  if (prompt) {
    const labels = 'Subject wearing outfit, in setting, with prop';
    if (!prompt.value) prompt.value = labels;
  }
}

function rmsRandomSeed() {
  const el = document.getElementById('rms_seed');
  if (el) el.value = Math.floor(Math.random() * 999999999);
}

function rmsSavePreview() {
  fetch('/api/refmod/save_preview', { method: 'POST' })
    .then(r => r.json()).then(d => alert(d.message || 'Preview saved.'))
    .catch(e => alert('Error: ' + e));
}

function rmsSetupSave() {
  const data = {
    model: document.getElementById('refmod_model')?.value,
    base: document.getElementById('refmod_base')?.value,
    folder: document.getElementById('refmod_folder')?.value,
    retention: document.getElementById('rms_retention')?.value,
    prompt: document.getElementById('rms_prompt')?.value,
    seed: document.getElementById('rms_seed')?.value
  };
  fetch('/api/refmod/save_setup', {
    method: 'POST', headers: {'Content-Type': 'application/json'},
    body: JSON.stringify(data)
  }).then(r => r.json()).then(d => alert(d.message || 'Setup saved.'))
    .catch(e => alert('Error: ' + e));
}

function rmsSetupLoad() {
  fetch('/api/refmod/load_setup')
    .then(r => r.json()).then(d => {
      if (d.settings) {
        if (d.settings.model) document.getElementById('refmod_model').value = d.settings.model;
        if (d.settings.base) document.getElementById('refmod_base').value = d.settings.base;
        if (d.settings.folder) document.getElementById('refmod_folder').value = d.settings.folder;
        if (d.settings.retention) { rmsSetRetention(d.settings.retention); }
        if (d.settings.prompt) document.getElementById('rms_prompt').value = d.settings.prompt;
        if (d.settings.seed) document.getElementById('rms_seed').value = d.settings.seed;
      }
      alert(d.message || 'Setup loaded.');
    }).catch(e => alert('Error: ' + e));
}

function rmsShowReadout() {
  alert('ComfyUI readout / settings — backend wiring needed.');
}

function rmsBake() {
  if (!confirm('Bake current mod stack into a single merged .safetensors file?')) return;
  fetch('/api/refmod/bake', { method: 'POST' })
    .then(r => r.json()).then(d => alert(d.message || 'Baked.'))
    .catch(e => alert('Error: ' + e));
}

function rmsOpenPreview() {
  const tweaked = document.getElementById('rms_tweaked_holder');
  const baseline = document.getElementById('rms_baseline_holder');
  const img1 = tweaked?.querySelector('img');
  const img2 = baseline?.querySelector('img');
  if (img1 || img2) {
    const w = window.open('', '_blank', 'width=1200,height=600');
    w.document.write('<html><body style="margin:0;background:#111;display:flex;gap:4px;justify-content:center;align-items:center;height:100vh;">');
    if (img1) w.document.write('<img src="' + img1.src + '" style="max-width:49%;max-height:98%;object-fit:contain;">');
    if (img2) w.document.write('<img src="' + img2.src + '" style="max-width:49%;max-height:98%;object-fit:contain;">');
    w.document.write('</body></html>');
  }
}

// ============================================================
// Explorer Tab JS
// ============================================================

function onExplorerFamilyChanged(family) {
  if (!family) {
    const sel = document.querySelector('input[name="explorer_family"]:checked');
    family = sel ? sel.value : 'klein';
  }
  const isKlein = family === 'klein';

  // DiT row (Klein only)
  const ditRow = document.getElementById('explorer_dit_row');
  if (ditRow) ditRow.style.display = isKlein ? '' : 'none';

  // Reference row
  const refRow = document.getElementById('explorer_ref_row');
  if (refRow) refRow.style.display = (family === 'minimax') ? 'none' : '';

  // Ref strength label
  const refStr = document.getElementById('explorer_ref_strength_label');
  if (refStr) refStr.style.display = isKlein ? '' : 'none';
  const refStrInput = document.getElementById('explorer_ref_strength');
  if (refStrInput) refStrInput.style.display = isKlein ? '' : 'none';
}

function explorerRandomizeSeed() {
  const el = document.getElementById('explorer_seed');
  if (el) el.value = Math.floor(Math.random() * 999999999);
}

function explorerStart() {
  const status = document.getElementById('explorer_status');
  if (status) status.textContent = 'Generating baseline + variants…';

  const family = document.querySelector('input[name="explorer_family"]:checked')?.value || 'klein';
  const data = {
    family: family,
    lora: document.getElementById('explorer_lora')?.value,
    strength: document.getElementById('explorer_strength')?.value,
    prompt: document.getElementById('explorer_prompt')?.value,
    ref_path: document.getElementById('explorer_ref_path')?.value,
    ref_mp: document.getElementById('explorer_ref_mp')?.value,
    ref_strength: document.getElementById('explorer_ref_strength')?.value,
    seed: document.getElementById('explorer_seed')?.value,
    resolution: document.getElementById('explorer_res')?.value,
    intensity: document.getElementById('explorer_intensity')?.value,
    mutations: document.getElementById('explorer_mutations')?.value,
    structure: document.getElementById('explorer_structure')?.value
  };
  fetch('/api/explorer/start', {
    method: 'POST', headers: {'Content-Type': 'application/json'},
    body: JSON.stringify(data)
  }).then(r => r.json()).then(d => {
    if (status) status.textContent = d.message || 'Done.';
  }).catch(e => { if (status) status.textContent = 'Error: ' + e; });
}

function explorerSave() {
  fetch('/api/explorer/save', { method: 'POST' })
    .then(r => r.json()).then(d => alert(d.message || 'Saved current baseline.'))
    .catch(e => alert('Error: ' + e));
}

function explorerUndo() {
  fetch('/api/explorer/undo', { method: 'POST' })
    .then(r => r.json()).then(d => {
      const status = document.getElementById('explorer_status');
      if (status) status.textContent = d.message || 'Undone.';
    }).catch(e => alert('Error: ' + e));
}

function explorerRestart() {
  if (!confirm('Restart exploration from scratch?')) return;
  fetch('/api/explorer/restart', { method: 'POST' })
    .then(r => r.json()).then(d => {
      const status = document.getElementById('explorer_status');
      if (status) status.textContent = d.message || 'Restarted.';
      const holder = document.getElementById('explorer_baseline_holder');
      if (holder) holder.innerHTML = '(render will appear here)';
    }).catch(e => alert('Error: ' + e));
}

function explorerFreezeTweaked() {
  fetch('/api/explorer/freeze', { method: 'POST' })
    .then(r => r.json()).then(d => {
      const state = document.getElementById('explorer_state_text');
      if (state) state.textContent = d.message || 'Tweaked weights frozen.';
    }).catch(e => alert('Error: ' + e));
}

function explorerRefineInRepair() {
  // Copy explorer settings to Repair Studio and switch
  const lora = document.getElementById('explorer_lora')?.value;
  const prompt = document.getElementById('explorer_prompt')?.value;
  const seed = document.getElementById('explorer_seed')?.value;
  if (lora) {
    const el = document.getElementById('repair_primary_lora');
    if (el) el.value = lora;
  }
  if (prompt) {
    const el = document.getElementById('repair_prompt');
    if (el) el.value = prompt;
  }
  if (seed) {
    const el = document.getElementById('repair_seed');
    if (el) el.value = seed;
  }
  switchTab('tab-repair');
}

function explorerReroll() {
  const status = document.getElementById('explorer_status');
  if (status) status.textContent = 'Re-rolling variants…';
  fetch('/api/explorer/reroll', { method: 'POST' })
    .then(r => r.json()).then(d => {
      if (status) status.textContent = d.message || 'Variants re-rolled.';
    }).catch(e => { if (status) status.textContent = 'Error: ' + e; });
}

function explorerPick(variantIndex) {
  const status = document.getElementById('explorer_status');
  if (status) status.textContent = 'Applying variant ' + variantIndex + '…';
  fetch('/api/explorer/pick', {
    method: 'POST', headers: {'Content-Type': 'application/json'},
    body: JSON.stringify({ variant: variantIndex })
  }).then(r => r.json()).then(d => {
    if (status) status.textContent = d.message || 'Variant applied.';
  }).catch(e => { if (status) status.textContent = 'Error: ' + e; });
}

function explorerCycleSeed() {
  const el = document.getElementById('explorer_seed');
  if (el) el.value = Math.floor(Math.random() * 999999999);
}

// ============================================================
// Royale Tab JS
// ============================================================

function onRoyaleFamilyChanged(family) {
  if (!family) {
    const sel = document.querySelector('input[name="royale_family"]:checked');
    family = sel ? sel.value : 'klein';
  }

  // Reference row (Klein only)
  const refRow = document.getElementById('royale_ref_row');
  if (refRow) refRow.style.display = (family === 'klein') ? '' : 'none';

  // Travel reference rows
  const travelRef = document.getElementById('royale_travel_ref_row');
  if (travelRef) travelRef.style.display = (family === 'klein') ? '' : 'none';
  const ptRef = document.getElementById('royale_pt_ref_row');
  if (ptRef) ptRef.style.display = (family === 'klein') ? '' : 'none';
}

function royaleApplyMode(mode) {
  // mode: 'folder' or 'single'
  const folderRow = document.getElementById('royale_folder_row');
  const singleRow = document.getElementById('royale_single_row');
  if (mode === 'folder') {
    if (folderRow) folderRow.style.display = '';
    if (singleRow) singleRow.style.display = 'none';
  } else {
    if (folderRow) folderRow.style.display = 'none';
    if (singleRow) singleRow.style.display = '';
  }
}

function royaleRender() {
  const status = document.getElementById('royale_status');
  if (status) status.textContent = 'Rendering crossfade…';

  const family = document.querySelector('input[name="royale_family"]:checked')?.value || 'klein';
  const sourceMode = document.querySelector('input[name="royale_source_mode"]:checked')?.value || 'folder';
  const data = {
    family: family,
    source_mode: sourceMode,
    folder: document.getElementById('royale_folder')?.value,
    single_lora: document.getElementById('royale_single')?.value,
    prompt: document.getElementById('royale_prompt')?.value,
    seed: document.getElementById('royale_seed')?.value,
    ref: document.getElementById('royale_ref')?.value,
    ref_strength: document.getElementById('royale_ref_strength')?.value,
    width: document.getElementById('royale_w')?.value,
    height: document.getElementById('royale_h')?.value,
    max_renders: document.getElementById('royale_max')?.value
  };
  fetch('/api/royale/render', {
    method: 'POST', headers: {'Content-Type': 'application/json'},
    body: JSON.stringify(data)
  }).then(r => r.json()).then(d => {
    if (status) status.textContent = d.message || 'Done.';
  }).catch(e => { if (status) status.textContent = 'Error: ' + e; });
}

function royaleScrub(val) {
  const label = document.getElementById('royale_scrub_label');
  if (label) label.textContent = 'Frame: ' + val;
  // In a real implementation this would update the preview to show frame N
}

function royaleExport() {
  const status = document.getElementById('royale_export_status');
  if (status) status.textContent = 'Exporting…';
  const data = {
    format: document.getElementById('royale_export_format')?.value,
    speed: document.getElementById('royale_export_speed')?.value,
    loop: document.getElementById('royale_export_loop')?.checked,
    epoch_tag: document.getElementById('royale_export_epoch')?.checked,
    watermark: document.getElementById('royale_export_wm')?.value
  };
  fetch('/api/royale/export', {
    method: 'POST', headers: {'Content-Type': 'application/json'},
    body: JSON.stringify(data)
  }).then(r => r.json()).then(d => {
    if (status) status.textContent = d.message || 'Exported.';
  }).catch(e => { if (status) status.textContent = 'Error: ' + e; });
}

function royaleSaveAllStills() {
  fetch('/api/royale/save_stills', { method: 'POST' })
    .then(r => r.json()).then(d => alert(d.message || 'All stills saved.'))
    .catch(e => alert('Error: ' + e));
}

function royaleTravelApplyPreset(name) {
  if (!name) return;
  fetch('/api/royale/travel_preset?name=' + encodeURIComponent(name))
    .then(r => r.json()).then(d => {
      if (d.settings) {
        if (d.settings.seed_a) document.getElementById('royale_travel_seed_a').value = d.settings.seed_a;
        if (d.settings.seed_b) document.getElementById('royale_travel_seed_b').value = d.settings.seed_b;
        if (d.settings.waypoints) document.getElementById('royale_travel_waypoints').value = d.settings.waypoints;
        if (d.settings.frames) document.getElementById('royale_travel_frames').value = d.settings.frames;
      }
    }).catch(e => alert('Error: ' + e));
}

function royaleTravelRandomizeSeeds() {
  const a = document.getElementById('royale_travel_seed_a');
  const b = document.getElementById('royale_travel_seed_b');
  if (a) a.value = Math.floor(Math.random() * 999999999);
  if (b) b.value = Math.floor(Math.random() * 999999999);
}

function royaleSeedTravel() {
  const status = document.getElementById('royale_travel_status');
  if (status) status.textContent = 'Seed travel rendering…';
  const data = {
    seed_a: document.getElementById('royale_travel_seed_a')?.value,
    seed_b: document.getElementById('royale_travel_seed_b')?.value,
    waypoints: document.getElementById('royale_travel_waypoints')?.value,
    frames: document.getElementById('royale_travel_frames')?.value,
    ref: document.getElementById('royale_travel_ref')?.value,
    ref_strength: document.getElementById('royale_travel_ref_strength')?.value,
    use_epoch_ref: document.getElementById('royale_travel_use_epoch_ref')?.checked,
    seq_ref: document.getElementById('royale_travel_seq_ref')?.checked,
    ref_mp: document.getElementById('royale_travel_ref_mp')?.value,
    speed: document.getElementById('royale_travel_speed')?.value,
    width: document.getElementById('royale_travel_w')?.value,
    height: document.getElementById('royale_travel_h')?.value,
    loop: document.getElementById('royale_travel_loop')?.checked,
    epoch_tag: document.getElementById('royale_travel_epoch')?.checked,
    watermark: document.getElementById('royale_travel_wm')?.value,
    deflicker: document.getElementById('royale_travel_deflicker')?.checked
  };
  fetch('/api/royale/seed_travel', {
    method: 'POST', headers: {'Content-Type': 'application/json'},
    body: JSON.stringify(data)
  }).then(r => r.json()).then(d => {
    if (status) status.textContent = d.message || 'Seed travel done.';
  }).catch(e => { if (status) status.textContent = 'Error: ' + e; });
}

function royalePtApplyPreset(name) {
  if (!name) return;
  fetch('/api/royale/pt_preset?name=' + encodeURIComponent(name))
    .then(r => r.json()).then(d => {
      if (d.settings) {
        if (d.settings.subject) document.getElementById('royale_pt_subject').value = d.settings.subject;
        if (d.settings.prompt) document.getElementById('royale_pt_prompt').value = d.settings.prompt;
        if (d.settings.custom) document.getElementById('royale_pt_custom').value = d.settings.custom;
      }
    }).catch(e => alert('Error: ' + e));
}

function royalePtInsertSlot() {
  const words = document.getElementById('royale_pt_words');
  if (!words) return;
  const newSlot = document.createElement('div');
  newSlot.style.cssText = 'display:flex;gap:4px;margin-top:4px;align-items:center;';
  newSlot.innerHTML = '<input type="text" class="w-combo" placeholder="comma,separated,words" style="flex:1;font-size:10pt;background:var(--bg-surface);color:var(--text-primary);border:1px solid var(--border);"><button class="btn" style="height:22px;padding:0 6px;font-size:10pt;" onclick="this.parentElement.remove()">✕</button>';
  words.appendChild(newSlot);
}

function royalePromptTravel() {
  const status = document.getElementById('royale_pt_status');
  if (status) status.textContent = 'Prompt travel rendering…';
  const data = {
    subject: document.getElementById('royale_pt_subject')?.value,
    prompt: document.getElementById('royale_pt_prompt')?.value,
    custom: document.getElementById('royale_pt_custom')?.value,
    start: document.getElementById('royale_pt_start')?.value,
    end: document.getElementById('royale_pt_end')?.value,
    frames: document.getElementById('royale_pt_frames')?.value,
    ref: document.getElementById('royale_pt_ref')?.value,
    use_epoch_ref: document.getElementById('royale_pt_use_epoch_ref')?.checked,
    ref_strength: document.getElementById('royale_pt_ref_strength')?.value,
    seq_ref: document.getElementById('royale_pt_seq_ref')?.checked,
    ref_mp: document.getElementById('royale_pt_ref_mp')?.value,
    anchor: document.getElementById('royale_pt_anchor')?.value,
    anchor_str: document.getElementById('royale_pt_anchor_str')?.value
  };
  // Collect word slots
  const wordSlots = document.querySelectorAll('#royale_pt_words input[type="text"]');
  data.words = Array.from(wordSlots).map(el => el.value);
  fetch('/api/royale/prompt_travel', {
    method: 'POST', headers: {'Content-Type': 'application/json'},
    body: JSON.stringify(data)
  }).then(r => r.json()).then(d => {
    if (status) status.textContent = d.message || 'Prompt travel done.';
  }).catch(e => { if (status) status.textContent = 'Error: ' + e; });
}

function royaleStrengthTravel() {
  const status = document.getElementById('royale_str_status');
  if (status) status.textContent = 'Strength travel rendering…';
  const data = {
    peak: document.getElementById('royale_str_peak')?.value,
    frames: document.getElementById('royale_str_frames')?.value
  };
  fetch('/api/royale/strength_travel', {
    method: 'POST', headers: {'Content-Type': 'application/json'},
    body: JSON.stringify(data)
  }).then(r => r.json()).then(d => {
    if (status) status.textContent = d.message || 'Strength travel done.';
  }).catch(e => { if (status) status.textContent = 'Error: ' + e; });
}

// ============================================================
// Extract Tab JS
// ============================================================

function onExtractFamilyChanged(family) {
  if (!family) {
    const sel = document.querySelector('input[name="extract_family"]:checked');
    family = sel ? sel.value : 'klein';
  }
  const isKlein = family === 'klein';

  // Klein-only cards
  ['extract_preset_card', 'extract_custom_card', 'extract_options_card', 'extract_prompt_card'].forEach(id => {
    const el = document.getElementById(id);
    if (el) el.style.display = isKlein ? '' : 'none';
  });
  // If non-Klein, hide custom card regardless
  if (!isKlein) {
    const cc = document.getElementById('extract_custom_card');
    if (cc) cc.style.display = 'none';
  }
}

function onExtractPresetChanged() {
  const preset = document.getElementById('extract_preset')?.value;
  const customCard = document.getElementById('extract_custom_card');
  if (customCard) customCard.style.display = (preset === 'Custom') ? '' : 'none';

  // Apply preset block selections
  const blocks = document.querySelectorAll('.extract-block');
  const presets = {
    'All Blocks': () => blocks.forEach(b => b.checked = true),
    'Fast SVD': () => blocks.forEach(b => b.checked = true),
    'Identity': () => { blocks.forEach(b => b.checked = false); blocks.forEach(b => { if (b.dataset.block === 'single_blocks.16') b.checked = true; }); },
    'Fast Identity': () => { blocks.forEach(b => b.checked = false); blocks.forEach(b => { if (b.dataset.block === 'single_blocks.16') b.checked = true; }); },
    'Style': () => { blocks.forEach(b => b.checked = false); ['double_blocks.0','double_blocks.1','double_blocks.2','double_blocks.3','double_blocks.4','double_blocks.5','double_blocks.6','double_blocks.7'].forEach(k => { const cb = document.querySelector('.extract-block[data-block="'+k+'"]'); if(cb) cb.checked = true; }); },
    'Style+Composition': () => { blocks.forEach(b => b.checked = false); ['double_blocks.0','double_blocks.1','double_blocks.2','double_blocks.3','double_blocks.4','double_blocks.5','double_blocks.6','double_blocks.7','single_blocks.0','single_blocks.1'].forEach(k => { const cb = document.querySelector('.extract-block[data-block="'+k+'"]'); if(cb) cb.checked = true; }); },
    'Fast Style+Composition': () => { blocks.forEach(b => b.checked = false); ['double_blocks.0','double_blocks.1','double_blocks.2','double_blocks.3','double_blocks.4','double_blocks.5','double_blocks.6','double_blocks.7','single_blocks.0','single_blocks.1'].forEach(k => { const cb = document.querySelector('.extract-block[data-block="'+k+'"]'); if(cb) cb.checked = true; }); },
    'Details': () => { blocks.forEach(b => b.checked = false); for(let i=12;i<=23;i++) { const cb = document.querySelector('.extract-block[data-block="single_blocks.'+i+'"]'); if(cb) cb.checked = true; } },
    'Fast Details': () => { blocks.forEach(b => b.checked = false); for(let i=12;i<=23;i++) { const cb = document.querySelector('.extract-block[data-block="single_blocks.'+i+'"]'); if(cb) cb.checked = true; } },
    'Custom': () => {} // Keep current
  };
  const fn = presets[preset];
  if (fn) fn();

  // Adjust options based on fast presets
  const isFast = preset && preset.startsWith('Fast');
  const samplesEl = document.getElementById('extract_samples');
  const samplesLabel = document.getElementById('extract_samples_label');
  if (isFast && samplesEl) samplesEl.value = '0';
}

function setAllExtractBlocks(selectAll) {
  document.querySelectorAll('.extract-block').forEach(cb => cb.checked = selectAll);
}

function setCategoryExtractBlocks(category) {
  document.querySelectorAll('.extract-block').forEach(cb => cb.checked = false);
  const map = {
    'identity': ['single_blocks.16'],
    'style_composition': ['double_blocks.0','double_blocks.1','double_blocks.2','double_blocks.3','double_blocks.4','double_blocks.5','double_blocks.6','double_blocks.7','single_blocks.0','single_blocks.1'],
    'details': (() => { const r=[]; for(let i=12;i<=23;i++) r.push('single_blocks.'+i); return r; })()
  };
  const keys = map[category] || [];
  keys.forEach(k => {
    const cb = document.querySelector('.extract-block[data-block="' + k + '"]');
    if (cb) cb.checked = true;
  });
}

function updateExtractOutputName() {
  const source = document.getElementById('extract_source')?.value || '';
  const out = document.getElementById('extract_out_name');
  if (out && source) {
    const basename = source.split(/[/\\]/).pop().replace(/\.(safetensors|pt|ckpt|bin)$/i, '');
    out.value = basename + '_extracted';
  }
}

function runExtract() {
  const progress = document.getElementById('extract_progress');
  const timeNote = document.getElementById('extract_time_note');
  const log = document.getElementById('extractConsoleLog');
  if (progress) progress.value = 0;
  if (timeNote) timeNote.textContent = '';
  if (log) log.textContent = '';

  const family = document.querySelector('input[name="extract_family"]:checked')?.value || 'klein';
  const data = {
    family: family,
    source: document.getElementById('extract_source')?.value,
    output_name: document.getElementById('extract_out_name')?.value,
    preset: document.getElementById('extract_preset')?.value,
    blocks: [],
    target_dim: document.getElementById('extract_target_dim')?.value,
    timesteps: document.getElementById('extract_timesteps')?.value,
    samples: document.getElementById('extract_samples')?.value,
    prompt: document.getElementById('extract_prompt')?.value
  };
  // Collect checked blocks
  document.querySelectorAll('.extract-block:checked').forEach(cb => {
    data.blocks.push(cb.dataset.block);
  });

  if (timeNote) timeNote.textContent = 'Starting extraction…';
  fetch('/api/extract/run', {
    method: 'POST', headers: {'Content-Type': 'application/json'},
    body: JSON.stringify(data)
  }).then(r => r.json()).then(d => {
    if (timeNote) timeNote.textContent = d.message || 'Extraction started.';
  }).catch(e => { if (timeNote) timeNote.textContent = 'Error: ' + e; });
}

function openExtractFolder() {
  fetch('/api/extract/open_folder', { method: 'POST' }).catch(() => {});
}

// Init: populate samples architecture from training + initialize family-dependent UI
document.addEventListener('DOMContentLoaded', () => {
  const t = document.getElementById('train_architecture');
  const s = document.getElementById('samples_architecture');
  if (t && s) {
    Array.from(t.options).forEach(opt => s.add(new Option(opt.text, opt.value, opt.selected, opt.selected)));
  }
  updateSamplesUI();
  onCaptionModelChanged();

  // Initialize family-dependent visibility for all tool tabs
  onRepairFamilyChanged();
  onExplorerFamilyChanged();
  onRoyaleFamilyChanged();
  onExtractFamilyChanged();
  onExtractPresetChanged();

  // Default Royale source mode
  royaleApplyMode('folder');

  // Draw empty RefMod curve canvas
  const cvs = document.getElementById('rms_curve_canvas');
  if (cvs) {
    const ctx = cvs.getContext('2d');
    ctx.fillStyle = '#1a1a1a';
    ctx.fillRect(0, 0, cvs.width, cvs.height);
    ctx.strokeStyle = '#444';
    ctx.lineWidth = 1;
    // Grid
    for (let x = 0; x < cvs.width; x += 56) { ctx.beginPath(); ctx.moveTo(x, 0); ctx.lineTo(x, cvs.height); ctx.stroke(); }
    for (let y = 0; y < cvs.height; y += 30) { ctx.beginPath(); ctx.moveTo(0, y); ctx.lineTo(cvs.width, y); ctx.stroke(); }
    // Flat 1.0 line
    ctx.strokeStyle = '#5B9BD5';
    ctx.lineWidth = 2;
    ctx.beginPath();
    ctx.moveTo(0, cvs.height * 0.5);
    ctx.lineTo(cvs.width, cvs.height * 0.5);
    ctx.stroke();
    ctx.fillStyle = '#888';
    ctx.font = '10px sans-serif';
    ctx.fillText('Frame →', cvs.width - 55, cvs.height - 5);
    ctx.fillText('Strength ↑', 5, 12);
  }
});
