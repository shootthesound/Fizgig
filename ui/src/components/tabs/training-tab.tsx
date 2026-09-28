'use client';

import React, { useState, useEffect } from 'react';
import { useSettingsStore } from '@/store/settings-store';
import {
  ARCHITECTURE_LIST,
  MINIMAX_TRAIN_BASE_OPTIONS,
  MINIMAX_STRUCTURE_OPTIONS,
  MINIMAX_STRUCTURE_DESC,
  MINIMAX_STRUCTURE_DEFAULT,
  MINIMAX_BLOCK_OPTIONS,
  MINIMAX_BASE_QUANT_OPTIONS,
  OPTIMIZER_TYPES,
} from '@/lib/constants';

/* ──────────────────────────────────────────────────────────────────────────
 * Collapsible Section — maps Python CollapsibleFrame 1:1
 * ────────────────────────────────────────────────────────────────────────── */
function CollapsibleSection({
  title, defaultOpen = false, children,
}: { title: string; defaultOpen?: boolean; children: React.ReactNode }) {
  return (
    <details className="collapsible-section" open={defaultOpen || undefined}>
      <summary>{title}</summary>
      <div className="collapsible-content">{children}</div>
    </details>
  );
}

/* Field row helper */
function FieldRow({ label, children }: { label: string; children: React.ReactNode }) {
  return (
    <div className="field-row">
      <label>{label}</label>
      {children}
    </div>
  );
}

export default function TrainingTab() {
  const { state, dispatch } = useSettingsStore();
  const settings = state.settings;

  const isKlein = settings.ARCHITECTURE === 'Flux 2 Klein Base 9B';
  const isKrea2 = settings.ARCHITECTURE === 'Krea 2';
  const isMinimax = settings.ARCHITECTURE === 'MiniMax H3';

  // --- Presets State ---
  const [presetsList, setPresetsList] = useState<string[]>([]);
  const [selectedPreset, setSelectedPreset] = useState('');

  // --- Training Lifecycle State ---
  const [enableCache, setEnableCache] = useState(true);
  const [trainingState, setTrainingState] = useState<'idle' | 'running' | 'paused' | 'stopped'>('idle');
  const [consoleOutput, setConsoleOutput] = useState('');

  // Helper to update store settings
  const updateSetting = (key: string, value: any) => {
    dispatch({
      type: 'UPDATE_SETTINGS',
      payload: { [key]: value },
    });
  };

  // Fetch presets and poll training status
  useEffect(() => {
    // Load presets
    fetch('/api/training', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ action: 'list_presets' }),
    })
      .then(res => res.json())
      .then(data => {
        if (data && data.presets) {
          setPresetsList(data.presets);
        }
      })
      .catch(() => {});

    // Status poll interval
    const interval = setInterval(() => {
      fetch('/api/training')
        .then(res => res.json())
        .then(data => {
          if (data) {
            setTrainingState(data.state || 'idle');
            if (data.logs && Array.isArray(data.logs)) {
              setConsoleOutput(data.logs.join(''));
            }
          }
        })
        .catch(() => {});
    }, 2000);

    return () => clearInterval(interval);
  }, []);

  // Handlers for Presets
  const handleSavePreset = async () => {
    const name = prompt('Enter preset name:');
    if (!name) return;
    try {
      const res = await fetch('/api/training', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          action: 'save_preset',
          presetName: name,
          presetData: settings,
        }),
      });
      const data = await res.json();
      if (data.success) {
        alert(`Preset '${data.name}' saved.`);
        if (!presetsList.includes(data.name)) {
          setPresetsList(prev => [...prev, data.name]);
        }
        setSelectedPreset(data.name);
      }
    } catch (e: any) {
      alert('Failed to save preset: ' + e?.message);
    }
  };

  const handleLoadPreset = async (name: string) => {
    if (!name) return;
    try {
      const res = await fetch('/api/training', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ action: 'load_preset', presetName: name }),
      });
      const data = await res.json();
      if (data.success && data.settings) {
        dispatch({ type: 'UPDATE_SETTINGS', payload: data.settings });
        setSelectedPreset(name);
      }
    } catch (e: any) {
      alert('Failed to load preset: ' + e?.message);
    }
  };

  const handleLoadLastTrain = async () => {
    try {
      const res = await fetch('/api/training', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ action: 'load_last_train' }),
      });
      const data = await res.json();
      if (data.success && data.settings) {
        dispatch({ type: 'UPDATE_SETTINGS', payload: data.settings });
        alert('Settings restored from last training session.');
      }
    } catch (e: any) {
      alert('No previous training session found.');
    }
  };

  // Handlers for Training Execution
  const handleStartTraining = async () => {
    try {
      // 1. Snapshot dataset config
      await fetch('/api/datasets', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          image_folder: state.image_folder,
          caption_trigger: state.caption_text,
          batch_size: state.dataset_batch_size,
          megapixels: state.dataset_megapixels,
          enable_bucket: state.dataset_enable_bucket,
        }),
      });

      // 2. Start training run
      const res = await fetch('/api/training', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          action: 'start',
          settings,
          prefs: state.prefs,
        }),
      });
      const data = await res.json();
      if (data.success) {
        setTrainingState('running');
      } else {
        alert(data.error || 'Failed to start training');
      }
    } catch (e: any) {
      alert('Failed to launch training: ' + e?.message);
    }
  };

  const handleStopTraining = async () => {
    if (!confirm('Stop the current training run?')) return;
    try {
      const res = await fetch('/api/training', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ action: 'stop' }),
      });
      const data = await res.json();
      if (data.success) {
        setTrainingState('stopped');
      }
    } catch (e: any) {
      alert('Failed to stop: ' + e?.message);
    }
  };

  const handlePauseTraining = async () => {
    try {
      const res = await fetch('/api/training', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ action: 'pause' }),
      });
      const data = await res.json();
      if (data.success) {
        setTrainingState('paused');
      }
    } catch (e: any) {
      alert('Failed to pause: ' + e?.message);
    }
  };

  const handleResumeTraining = async () => {
    try {
      const res = await fetch('/api/training', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ action: 'resume' }),
      });
      const data = await res.json();
      if (data.success) {
        setTrainingState('running');
      }
    } catch (e: any) {
      alert('Failed to resume: ' + e?.message);
    }
  };

  const blocksSwapOptions = [
    'Auto (detect from GPU)',
    '0  (No swap — 16GB fp8 / 24GB bf16)',
    '4  (Light — 14GB fp8 / 20GB bf16)',
    '8  (Moderate — 12GB fp8 / 16GB bf16)',
    '12 (Aggressive — 10GB fp8 / 12GB bf16)',
    '16 (Max — 8GB fp8 / 10GB bf16)',
    '1', '2', '3', '5', '6', '7', '9', '10', '11', '13', '14', '15',
  ];

  return (
    <div className="fizgig-tab-page">
      {/* ── Banner ── */}
      <div className="tab-banner">
        <h1>Training</h1>
        <p>Pick a preset or dial in a custom run. Sections below collapse to reduce clutter — click any header to toggle. Dataset-config knobs live in Other Options.</p>
      </div>

      {/* ── Base Model card ── */}
      {ARCHITECTURE_LIST.length > 1 && (
        <div className="panel source-card">
          <h2>Base Model</h2>
          <p className="muted">Pick the model family to train. Krea 2 needs its model paths set on the Preferences tab before training.</p>
          <div className="field-row">
            <label>Model:</label>
            <select
              value={settings.ARCHITECTURE}
              onChange={(e) => updateSetting('ARCHITECTURE', e.target.value)}
              style={{ maxWidth: 280 }}
            >
              {ARCHITECTURE_LIST.map(a => <option key={a} value={a}>{a}</option>)}
            </select>
          </div>

          {/* Training Base (MiniMax only) */}
          {isMinimax && (
            <>
              <div className="field-row" style={{ marginTop: 10 }}>
                <label>Training Base:</label>
                <select
                  value={settings.MINIMAX_TRAIN_BASE}
                  onChange={(e) => updateSetting('MINIMAX_TRAIN_BASE', e.target.value)}
                  style={{ maxWidth: 360 }}
                >
                  {MINIMAX_TRAIN_BASE_OPTIONS.map(o => <option key={o} value={o}>{o}</option>)}
                </select>
              </div>
              <p className="hint">Pick the H3 model you deploy on. First/last frame (fl2va) is the standard model most workflows run. Reference (ref2va) is the Reference-to-Video fine-tune — choose it if your LoRA&apos;s home is the r2v workflow (needs &apos;DiT (reference)&apos; set in Preferences). Presets never change this; reference distillation always trains on ref2va regardless.</p>
              <p style={{ color: '#F59E0B', fontSize: 11 }}>⏱ Previews track LIKENESS, not quality. Judge quality in ComfyUI — Pause frees the GPU, so you can check an epoch there and Resume.{'\n'}Defaults are 768×768 56-frame clips with sound; Sample length has stills and other lengths. 📖 Full write-ups in the README.</p>
            </>
          )}
        </div>
      )}

      {/* ── Presets card ── */}
      <div className="panel source-card">
        <h2>Presets</h2>
        <p className="muted">Save the current settings under a name, load a saved preset, or restore the exact configuration from your last training run.</p>
        <div style={{ display: 'flex', gap: 12, alignItems: 'center', flexWrap: 'wrap' }}>
          <button type="button" className="secondary" onClick={handleSavePreset}>Save Preset</button>
          <label style={{ color: '#8a9bae', fontSize: 13 }}>Load Preset:</label>
          <select
            value={selectedPreset}
            onChange={(e) => handleLoadPreset(e.target.value)}
            style={{ minHeight: 32, minWidth: 220, border: '1px solid #647284', background: '#18212b', color: '#f3f5f7', padding: '6px 8px' }}
          >
            <option value="">—</option>
            {presetsList.map(p => <option key={p} value={p}>{p}</option>)}
          </select>
        </div>
        <button type="button" className="secondary" style={{ marginTop: 8 }} onClick={handleLoadLastTrain}>
          Load Settings From Last Train
        </button>
      </div>

      {/* ══════════════════════════════════════════════════════════════════
       *  Training Parameters (expanded by default)
       * ══════════════════════════════════════════════════════════════════ */}
      <CollapsibleSection title="Training Parameters" defaultOpen>
        <FieldRow label="Model Type:">
          <select
            value={settings.MODEL_TYPE}
            onChange={(e) => updateSetting('MODEL_TYPE', e.target.value)}
          >
            <option value="">—</option>
            <option value="i2v">i2v (Image-to-Video)</option>
            <option value="t2v">t2v (Text-to-Video)</option>
          </select>
        </FieldRow>

        <FieldRow label="Learning Rate:">
          <input
            type="text"
            value={settings.LEARNING_RATE}
            disabled={settings.ADAPTIVE_LR}
            onChange={(e) => updateSetting('LEARNING_RATE', parseFloat(e.target.value) || 0)}
          />
        </FieldRow>

        {/* Adaptive LR */}
        <label className="inline-check" style={{ marginLeft: 180 }}>
          <input
            type="checkbox"
            checked={settings.ADAPTIVE_LR}
            onChange={(e) => updateSetting('ADAPTIVE_LR', e.target.checked)}
          />
          Adaptive LR (auto-adjust based on loss, gradient clipping &amp; weight-norm growth)
        </label>
        {settings.ADAPTIVE_LR && (
          <div style={{ display: 'flex', gap: 12, alignItems: 'center', marginLeft: 200, paddingBottom: 4 }}>
            <label style={{ color: '#8a9bae', fontSize: 12 }}>Min LR:</label>
            <select
              value={settings.ADAPTIVE_LR_MIN}
              onChange={(e) => updateSetting('ADAPTIVE_LR_MIN', e.target.value)}
              style={{ minHeight: 28, border: '1px solid #647284', background: '#18212b', color: '#f3f5f7', padding: '4px 6px' }}
            >
              {['1e-5', '5e-5', '1e-4', '2e-4 - rank 4/8 only', '3e-4 - low-rank only'].map(v => <option key={v}>{v}</option>)}
            </select>
            <label style={{ color: '#8a9bae', fontSize: 12 }}>Max LR:</label>
            <select
              value={settings.ADAPTIVE_LR_MAX}
              onChange={(e) => updateSetting('ADAPTIVE_LR_MAX', e.target.value)}
              style={{ minHeight: 28, border: '1px solid #647284', background: '#18212b', color: '#f3f5f7', padding: '4px 6px' }}
            >
              {['1e-4', '2e-4', '3e-4', '4e-4'].map(v => <option key={v}>{v}</option>)}
            </select>
            <button
              type="button"
              className="secondary"
              style={{ fontSize: 11, minHeight: 28, padding: '4px 10px' }}
              onClick={() => {
                updateSetting('ADAPTIVE_LR_MIN', '1e-5');
                updateSetting('ADAPTIVE_LR_MAX', '4e-4');
              }}
            >
              Reset Defaults
            </button>
          </div>
        )}
        <p className="hint" style={{ marginLeft: 200 }}>When on, the Learning Rate box is IGNORED (greyed out) — the run starts at the geometric midpoint of Min/Max (e.g. 1e-4 &amp; 4e-4 → 2e-4) and the watcher owns the LR from there: probes UP on steady loss descent; reduces DOWN on loss plateau, heavy gradient clipping, or runaway weight-norm growth (with a rollback to the previous epoch&apos;s weights on stability events).</p>

        <FieldRow label="Network Dim (Rank):">
          <input
            type="text"
            value={settings.NETWORK_DIM}
            onChange={(e) => updateSetting('NETWORK_DIM', parseInt(e.target.value, 10) || 0)}
          />
        </FieldRow>

        <FieldRow label="Network Alpha:">
          <input
            type="text"
            value={settings.NETWORK_ALPHA}
            onChange={(e) => updateSetting('NETWORK_ALPHA', parseFloat(e.target.value) || 0)}
          />
        </FieldRow>

        <FieldRow label="Max Epochs:">
          <input
            type="text"
            value={settings.MAX_TRAIN_EPOCHS}
            onChange={(e) => updateSetting('MAX_TRAIN_EPOCHS', parseInt(e.target.value, 10) || 0)}
          />
        </FieldRow>

        <FieldRow label="Save Every N Epochs:">
          <input
            type="text"
            value={settings.SAVE_EVERY_N_EPOCHS}
            onChange={(e) => updateSetting('SAVE_EVERY_N_EPOCHS', parseInt(e.target.value, 10) || 1)}
          />
        </FieldRow>

        <FieldRow label="Seed:">
          <input
            type="text"
            value={settings.SEED}
            onChange={(e) => updateSetting('SEED', parseInt(e.target.value, 10) || 42)}
          />
        </FieldRow>

        {/* Network Type (Krea 2 only) */}
        {isKrea2 && (
          <>
            <FieldRow label="Network Type:">
              <select
                value={settings.NETWORK_TYPE}
                onChange={(e) => updateSetting('NETWORK_TYPE', e.target.value)}
              >
                <option value="LoRA (standard)">LoRA (standard)</option>
                <option value="LoKR (Kronecker product)">LoKR (Kronecker product)</option>
              </select>
            </FieldRow>
            {settings.NETWORK_TYPE.includes('LoKR') && (
              <FieldRow label="LoKR Factor:">
                <select
                  value={settings.LOKR_FACTOR}
                  onChange={(e) => updateSetting('LOKR_FACTOR', parseInt(e.target.value, 10) || 8)}
                >
                  {['-1 (auto)', '4', '8', '16'].map(v => <option key={v} value={v.split(' ')[0]}>{v}</option>)}
                </select>
              </FieldRow>
            )}
          </>
        )}

        {/* MiniMax Training Structure & Shift (Clean-end share) */}
        {isMinimax && (
          <>
            <FieldRow label="Structure:">
              <select
                value={MINIMAX_STRUCTURE_DEFAULT}
                onChange={(e) => {
                  const opt = MINIMAX_STRUCTURE_OPTIONS[e.target.value];
                  if (opt) {
                    updateSetting('MINIMAX_LOWNOISE_PCT', opt[0]);
                    updateSetting('MINIMAX_HIGHNOISE_LR_PCT', opt[1]);
                  }
                }}
                style={{ maxWidth: 320 }}
              >
                {Object.keys(MINIMAX_STRUCTURE_OPTIONS).map(k => <option key={k}>{k}</option>)}
              </select>
            </FieldRow>

            {/* Clean-end share (MINIMAX_LOWNOISE_PCT) */}
            <FieldRow label="Clean-end share:">
              <div style={{ display: 'flex', gap: 8, alignItems: 'center' }}>
                <input
                  type="text"
                  value={settings.MINIMAX_LOWNOISE_PCT}
                  onChange={(e) => updateSetting('MINIMAX_LOWNOISE_PCT', e.target.value)}
                  style={{ width: 80 }}
                />
                <span style={{ color: '#8a9bae', fontSize: 13 }}>% of steps</span>
              </div>
            </FieldRow>

            {/* Medium to High Noise LR (MINIMAX_HIGHNOISE_LR_PCT) */}
            <FieldRow label="Medium to High Noise LR:">
              <div style={{ display: 'flex', gap: 8, alignItems: 'center' }}>
                <input
                  type="text"
                  value={settings.MINIMAX_HIGHNOISE_LR_PCT}
                  onChange={(e) => updateSetting('MINIMAX_HIGHNOISE_LR_PCT', e.target.value)}
                  style={{ width: 80 }}
                />
                <span style={{ color: '#8a9bae', fontSize: 13 }}>% — best left at 100 unless you are experimenting.</span>
              </div>
            </FieldRow>
            <p className="hint">Scales the learning rate of the noisy-half steps (pose, framing, face shape). See the MiniMax section of the README.</p>
          </>
        )}

        {/* Per-category retirement — MIXED datasets only */}
        <div style={{ marginTop: 14, padding: '10px 14px', background: '#18212b', border: '1px solid #334155', borderRadius: 4 }}>
          <div style={{ display: 'flex', gap: 10, alignItems: 'center', flexWrap: 'wrap' }}>
            <span style={{ fontWeight: 500, fontSize: 13, color: '#f3f5f7' }}>Finish one category early:</span>
            <select
              value={settings.MIXED_STOP_CATEGORY}
              onChange={(e) => updateSetting('MIXED_STOP_CATEGORY', e.target.value)}
              style={{ minHeight: 28, border: '1px solid #647284', background: '#121820', color: '#f3f5f7', padding: '2px 8px' }}
            >
              <option value="voice">voice</option>
              <option value="photos & clips">photos &amp; clips</option>
            </select>
            <span style={{ color: '#8a9bae', fontSize: 13 }}>after epoch</span>
            <input
              type="text"
              value={settings.MIXED_STOP_EPOCH}
              onChange={(e) => updateSetting('MIXED_STOP_EPOCH', e.target.value)}
              placeholder=""
              style={{ width: 50, minHeight: 28, padding: '2px 6px', background: '#121820', border: '1px solid #647284', color: '#f3f5f7' }}
            />
            <select
              value={settings.MIXED_STOP_MODE}
              onChange={(e) => updateSetting('MIXED_STOP_MODE', e.target.value)}
              style={{ minHeight: 28, border: '1px solid #647284', background: '#121820', color: '#f3f5f7', padding: '2px 8px' }}
            >
              <option value="anchor at 10% LR (recommended)">anchor at 10% LR (recommended)</option>
              <option value="stop completely (faster)">stop completely (faster)</option>
            </select>
          </div>
          <p className="hint" style={{ marginTop: 6, marginBottom: 0 }}>
            If one category is a substantially different size from the other, it may be done (or start to overbake) well before the rest — finish it early instead of overtraining it. Blank = both train to the end. Anchor keeps the finished category at a true 10% learning rate. Stop skips its steps entirely.
          </p>
        </div>

        {/* Model Area to Train (Klein only) */}
        {isKlein && (
          <FieldRow label="Model Area to Train:">
            <select style={{ maxWidth: 220 }}>
              <option>Full Model</option>
              <option>Identity Only</option>
              <option>Style &amp; Details</option>
              <option>Custom Block Selection</option>
            </select>
          </FieldRow>
        )}

        {/* Context LoRA */}
        <FieldRow label="Context LoRA:">
          <div style={{ display: 'flex', gap: 6, flex: 1, alignItems: 'center' }}>
            <input
              type="text"
              value={settings.CONTEXT_LORA_PATH}
              onChange={(e) => updateSetting('CONTEXT_LORA_PATH', e.target.value)}
              placeholder="Optional pre-trained LoRA to compose during training"
              style={{ flex: 1 }}
            />
            <span style={{ color: '#8a9bae', fontSize: 13 }}>Strength:</span>
            <input
              type="text"
              value={settings.CONTEXT_LORA_STRENGTH}
              onChange={(e) => updateSetting('CONTEXT_LORA_STRENGTH', e.target.value)}
              style={{ width: 60 }}
            />
          </div>
        </FieldRow>

        {/* Target Megapixels */}
        <FieldRow label="Target Megapixels:">
          <div style={{ display: 'flex', gap: 8, alignItems: 'center' }}>
            <select
              value={state.dataset_megapixels}
              onChange={(e) => dispatch({ type: 'UPDATE_STATE', payload: { dataset_megapixels: e.target.value } })}
              style={{ width: 120 }}
            >
              <option value="0.25">0.25 MP (512×512)</option>
              <option value="0.5">0.5 MP (768×768)</option>
              <option value="1.0">1.0 MP (1024×1024)</option>
            </select>
          </div>
        </FieldRow>

        {/* Krea 2 loss watch */}
        {isKrea2 && (
          <div style={{ marginTop: 12 }}>
            <label className="inline-check">
              <input
                type="checkbox"
                checked={settings.KREA2_LOSS_WATCH}
                onChange={(e) => updateSetting('KREA2_LOSS_WATCH', e.target.checked)}
              />
              Per-image loss tracking + stuck image detection (loss watch)
            </label>
            <label className="inline-check" style={{ marginLeft: 20 }}>
              <input
                type="checkbox"
                checked={settings.KREA2_PER_IMAGE_LR}
                onChange={(e) => updateSetting('KREA2_PER_IMAGE_LR', e.target.checked)}
              />
              Per-image adaptive LR (experimental)
            </label>
          </div>
        )}
      </CollapsibleSection>

      {/* ══════════════════════════════════════════════════════════════════
       *  Output (expanded by default)
       * ══════════════════════════════════════════════════════════════════ */}
      <CollapsibleSection title="Output" defaultOpen>
        <FieldRow label="Output Directory:">
          <div style={{ display: 'flex', gap: 6 }}>
            <input
              type="text"
              value={settings.LORA_OUTPUT_DIR}
              onChange={(e) => updateSetting('LORA_OUTPUT_DIR', e.target.value)}
              style={{ flex: 1 }}
            />
            <button type="button" className="secondary" style={{ minHeight: 32 }}>Browse</button>
          </div>
        </FieldRow>

        <FieldRow label="LoRA Name:">
          <input
            type="text"
            value={settings.LORA_NAME}
            onChange={(e) => updateSetting('LORA_NAME', e.target.value)}
          />
        </FieldRow>
      </CollapsibleSection>

      {/* ══════════════════════════════════════════════════════════════════
       *  Memory & Precision
       * ══════════════════════════════════════════════════════════════════ */}
      <CollapsibleSection title="Memory &amp; Precision">
        <FieldRow label="Blocks to Swap:">
          <select
            value={settings.BLOCKS_SWAP}
            onChange={(e) => updateSetting('BLOCKS_SWAP', e.target.value)}
          >
            {blocksSwapOptions.map(v => <option key={v}>{v}</option>)}
          </select>
        </FieldRow>

        <FieldRow label="Resume Training:">
          <div style={{ display: 'flex', gap: 6 }}>
            <input
              type="text"
              value={settings.RESUME_TRAINING}
              onChange={(e) => updateSetting('RESUME_TRAINING', e.target.value)}
              placeholder="Path to saved state directory"
              style={{ flex: 1 }}
            />
            <button type="button" className="secondary" style={{ minHeight: 32 }}>Browse</button>
          </div>
        </FieldRow>

        <div style={{ marginTop: 8, display: 'flex', gap: 16, flexWrap: 'wrap' }}>
          <label className="inline-check">
            <input
              type="checkbox"
              checked={settings.QUANT_4BIT}
              onChange={(e) => updateSetting('QUANT_4BIT', e.target.checked)}
            />
            4-bit Base (NF4 quantisation)
          </label>
          <label className="inline-check">
            <input
              type="checkbox"
              checked={settings.FP8}
              onChange={(e) => updateSetting('FP8', e.target.checked)}
            />
            FP8 Base
          </label>
          <label className="inline-check">
            <input
              type="checkbox"
              checked={settings.SCALED}
              onChange={(e) => updateSetting('SCALED', e.target.checked)}
            />
            Scaled FP8
          </label>
          <label className="inline-check">
            <input
              type="checkbox"
              checked={settings.GRADIENT_CHECKPOINTING}
              onChange={(e) => updateSetting('GRADIENT_CHECKPOINTING', e.target.checked)}
            />
            Gradient Checkpointing
          </label>
        </div>

        <div style={{ marginTop: 8, display: 'flex', gap: 16, alignItems: 'center' }}>
          <label className="inline-check">
            <input
              type="checkbox"
              checked={settings.SAVE_STATE}
              onChange={(e) => updateSetting('SAVE_STATE', e.target.checked)}
            />
            Save Training State
          </label>
          <span style={{ color: '#8a9bae', fontSize: 13 }}>Keep last N states:</span>
          <input
            type="text"
            value={settings.KEEP_LAST_N_STATES}
            onChange={(e) => updateSetting('KEEP_LAST_N_STATES', parseInt(e.target.value, 10) || 1)}
            style={{ width: 60 }}
          />
        </div>
      </CollapsibleSection>

      {/* ══════════════════════════════════════════════════════════════════
       *  Timestep & Noise Schedule
       * ══════════════════════════════════════════════════════════════════ */}
      <CollapsibleSection title="Timestep &amp; Noise Schedule">
        <FieldRow label="Timestep Sampling:">
          <select
            value={settings.TIMESTEP_SAMPLING}
            onChange={(e) => updateSetting('TIMESTEP_SAMPLING', e.target.value)}
          >
            <option value="flux2_shift">flux2_shift</option>
            <option value="shift">shift</option>
            <option value="sigmoid">sigmoid</option>
            <option value="uniform">uniform</option>
          </select>
        </FieldRow>

        <FieldRow label="Discrete Flow Shift:">
          <input
            type="text"
            value={settings.DISCRETE_FLOW_SHIFT}
            onChange={(e) => updateSetting('DISCRETE_FLOW_SHIFT', e.target.value)}
          />
        </FieldRow>

        <FieldRow label="Sigmoid Scale:">
          <input
            type="text"
            value={settings.SIGMOID_SCALE}
            onChange={(e) => updateSetting('SIGMOID_SCALE', e.target.value)}
          />
        </FieldRow>

        <FieldRow label="Min Timestep:">
          <input
            type="text"
            value={settings.MIN_TIMESTEP}
            onChange={(e) => updateSetting('MIN_TIMESTEP', e.target.value)}
          />
        </FieldRow>

        <FieldRow label="Max Timestep:">
          <input
            type="text"
            value={settings.MAX_TIMESTEP}
            onChange={(e) => updateSetting('MAX_TIMESTEP', e.target.value)}
          />
        </FieldRow>

        <label className="inline-check" style={{ marginLeft: 180 }}>
          <input
            type="checkbox"
            checked={settings.PRESERVE_DISTRIBUTION}
            onChange={(e) => updateSetting('PRESERVE_DISTRIBUTION', e.target.checked)}
          />
          Preserve Distribution
        </label>
      </CollapsibleSection>

      {/* ══════════════════════════════════════════════════════════════════
       *  Other Options
       * ══════════════════════════════════════════════════════════════════ */}
      <CollapsibleSection title="Other Options">
        <FieldRow label="Optimizer Type:">
          <select
            value={settings.OPTIMIZER_TYPE}
            onChange={(e) => updateSetting('OPTIMIZER_TYPE', e.target.value)}
          >
            {OPTIMIZER_TYPES.map(o => <option key={o} value={o}>{o}</option>)}
          </select>
        </FieldRow>

        <FieldRow label="Optimizer Args:">
          <input
            type="text"
            value={settings.OPTIMIZER_ARGS}
            onChange={(e) => updateSetting('OPTIMIZER_ARGS', e.target.value)}
          />
        </FieldRow>

        <FieldRow label="Gradient Accumulation:">
          <input
            type="text"
            value={settings.GRADIENT_ACCUMULATION}
            onChange={(e) => updateSetting('GRADIENT_ACCUMULATION', parseInt(e.target.value, 10) || 1)}
          />
        </FieldRow>

        <FieldRow label="Max Grad Norm:">
          <input
            type="text"
            value={settings.MAX_GRAD_NORM}
            onChange={(e) => updateSetting('MAX_GRAD_NORM', parseFloat(e.target.value) || 0)}
          />
        </FieldRow>

        <FieldRow label="Network Dropout:">
          <input
            type="text"
            value={settings.NETWORK_DROPOUT}
            onChange={(e) => updateSetting('NETWORK_DROPOUT', parseFloat(e.target.value) || 0)}
          />
        </FieldRow>

        <FieldRow label="Attention Mechanism:">
          <select
            value={settings.ATTENTION_MECHANISM}
            onChange={(e) => updateSetting('ATTENTION_MECHANISM', e.target.value)}
          >
            <option value="sdpa">sdpa</option>
            <option value="flash_attn">flash_attn</option>
            <option value="sage_attn">sage_attn</option>
            <option value="xformers">xformers</option>
          </select>
        </FieldRow>

        <FieldRow label="LR Scheduler:">
          <select
            value={settings.LR_SCHEDULER}
            onChange={(e) => updateSetting('LR_SCHEDULER', e.target.value)}
          >
            {['constant', 'linear', 'cosine', 'cosine_with_restarts', 'polynomial', 'constant_with_warmup'].map(s => <option key={s} value={s}>{s}</option>)}
          </select>
        </FieldRow>

        {isMinimax && (
          <>
            <FieldRow label="EMA:">
              <select
                value={settings.MINIMAX_EMA}
                onChange={(e) => updateSetting('MINIMAX_EMA', e.target.value)}
              >
                {['Off', '0.99', '0.999', '0.9999'].map(v => <option key={v}>{v}</option>)}
              </select>
            </FieldRow>

            <FieldRow label="Caption Dropout:">
              <select
                value={settings.MINIMAX_CAPTION_DROPOUT}
                onChange={(e) => updateSetting('MINIMAX_CAPTION_DROPOUT', e.target.value)}
              >
                {['0 (disabled)', '0.05 (default)', '0.1', '0.2'].map(v => <option key={v}>{v}</option>)}
              </select>
            </FieldRow>

            <FieldRow label="Blocks to Train:">
              <select
                value={settings.MINIMAX_BLOCKS}
                onChange={(e) => updateSetting('MINIMAX_BLOCKS', e.target.value)}
                style={{ maxWidth: 300 }}
              >
                {MINIMAX_BLOCK_OPTIONS.map(v => <option key={v}>{v}</option>)}
              </select>
            </FieldRow>

            <div style={{ marginTop: 12 }}>
              <label className="inline-check">
                <input
                  type="checkbox"
                  checked={settings.MINIMAX_DISTILL}
                  onChange={(e) => updateSetting('MINIMAX_DISTILL', e.target.checked)}
                />
                Learn identity from my dataset (reference distillation)
              </label>
            </div>
          </>
        )}
      </CollapsibleSection>

      {/* ── Run card ── */}
      <div className="panel source-card">
        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 12, flexWrap: 'wrap', gap: 10 }}>
          <h2 style={{ margin: 0, fontSize: 14 }}>Run Configuration</h2>
          <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
            <span style={{ fontSize: 12, color: '#94a3b8' }}>Target Cloud GPU:</span>
            <select
              value={state.prefs.modal_gpu || 'A100-40GB'}
              onChange={(e) => {
                const val = e.target.value;
                dispatch({ type: 'UPDATE_PREFS', payload: { modal_gpu: val } });
                fetch('/api/prefs', {
                  method: 'POST',
                  headers: { 'Content-Type': 'application/json' },
                  body: JSON.stringify({ prefs: { ...state.prefs, modal_gpu: val } }),
                }).catch(() => {});
              }}
              style={{
                fontSize: 12,
                padding: '4px 10px',
                borderRadius: 4,
                backgroundColor: '#0f172a',
                color: '#38bdf8',
                border: '1px solid #0284c7',
                fontWeight: 600,
                cursor: 'pointer',
              }}
            >
              <option value="A100-40GB">⚡ Modal: NVIDIA A100 (40GB VRAM) — Recommended</option>
              <option value="A100-80GB">⚡ Modal: NVIDIA A100 (80GB VRAM) — Large Batches</option>
              <option value="H100">⚡ Modal: NVIDIA H100 (80GB SXM5) — Maximum Speed</option>
              <option value="A10G">⚡ Modal: NVIDIA A10G (24GB VRAM) — Balanced</option>
              <option value="L4">⚡ Modal: NVIDIA L4 (24GB VRAM) — Ada Lovelace</option>
              <option value="T4">⚡ Modal: NVIDIA T4 (16GB VRAM) — Budget</option>
            </select>
          </div>
        </div>
        <label className="inline-check" style={{ marginBottom: 12 }}>
          <input
            type="checkbox"
            checked={enableCache}
            onChange={(e) => setEnableCache(e.target.checked)}
          />
          Enable Cache Preparation
        </label>
        <div style={{ display: 'flex', gap: 12, flexWrap: 'wrap' }}>
          <button
            type="button"
            className="primary sparkle-button"
            onClick={handleStartTraining}
            disabled={trainingState === 'running'}
          >
            Start Training
          </button>
          <button
            type="button"
            className="secondary"
            onClick={handlePauseTraining}
            disabled={trainingState !== 'running'}
          >
            Pause Training
          </button>
          <button
            type="button"
            className="primary"
            onClick={handleResumeTraining}
            style={{ display: trainingState === 'paused' ? 'inline-block' : 'none' }}
          >
            Resume Training
          </button>
          <button
            type="button"
            className="danger-btn"
            onClick={handleStopTraining}
            disabled={trainingState === 'idle' || trainingState === 'stopped'}
          >
            Stop Training
          </button>
          <button
            type="button"
            className="secondary"
            style={{ marginLeft: 'auto' }}
            onClick={() => window.open('/samples', '_blank')}
          >
            View Samples Gallery
          </button>
          <button
            type="button"
            className="secondary"
            onClick={() => alert(`Output folder: ${settings.LORA_OUTPUT_DIR}`)}
          >
            Open Samples Folder
          </button>
        </div>
      </div>

      {/* ── Console Output card ── */}
      <div className="panel source-card">
        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 10 }}>
          <h2 style={{ margin: 0, fontSize: 14 }}>
            Console Output {trainingState !== 'idle' && <span style={{ color: '#38bdf8', fontSize: 12 }}>({trainingState.toUpperCase()})</span>}
          </h2>
          <button
            type="button"
            className="secondary small"
            onClick={() => setConsoleOutput('')}
          >
            Clear Log
          </button>
        </div>
        <pre className="output-log" style={{ minHeight: 200, maxHeight: 400 }}>
          {consoleOutput || 'Ready — configure settings above and click Start Training.'}
        </pre>
      </div>
    </div>
  );
}
