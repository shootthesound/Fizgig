'use client';

import React, { useState, useEffect } from 'react';
import { useSettingsStore } from '@/store/settings-store';
import { ARCHITECTURE_LIST, ARCHITECTURES, SAMPLE_RESOLUTIONS } from '@/lib/constants';

interface SampleItem {
  name: string;
  path: string;
  mtime: number;
}

export default function SamplesTab() {
  const { state, dispatch } = useSettingsStore();
  const settings = state.settings;

  const architecture = settings.ARCHITECTURE || 'Flux 2 Klein Base 9B';
  const isKlein = architecture === 'Flux 2 Klein Base 9B';
  const isKrea2 = architecture === 'Krea 2';
  const isMinimax = architecture === 'MiniMax H3';

  const currentArchConfig = ARCHITECTURES[architecture] || ARCHITECTURES['Flux 2 Klein Base 9B'];
  const supportsSamples = currentArchConfig?.supports_samples ?? true;

  // Local sample gallery state
  const [sampleGallery, setSampleGallery] = useState<SampleItem[]>([]);
  const [loadingGallery, setLoadingGallery] = useState(false);
  const [showGallery, setShowGallery] = useState(false);

  const sampleFramesOptions = [
    'Still (1 frame)',
    '22 frames (~1s)',
    '56 frames (~2.3s)',
    '124 frames (~5s — trained minimum)',
    '141 frames (~6s)',
    '22 frames with sound (~1s)',
    '56 frames with sound (~2.3s)',
    '124 frames with sound (~5s)',
  ];

  const updateSetting = (key: string, val: any) => {
    dispatch({
      type: 'UPDATE_SETTINGS',
      payload: { [key]: val },
    });
  };

  const loadGallery = async () => {
    setLoadingGallery(true);
    try {
      const res = await fetch(`/api/samples?dir=${encodeURIComponent(state.prefs.lora_output_dir || 'output_loras')}`);
      const data = await res.json();
      if (data && data.samples) {
        setSampleGallery(data.samples);
      }
    } catch (e) {
      console.error('Failed to load samples:', e);
    } finally {
      setLoadingGallery(false);
    }
  };

  useEffect(() => {
    loadGallery();
  }, [state.prefs.lora_output_dir]);

  const handleArchChange = (newArch: string) => {
    updateSetting('ARCHITECTURE', newArch);
    const cfg = ARCHITECTURES[newArch] || ARCHITECTURES['Flux 2 Klein Base 9B'];
    if (cfg?.sample_width_default) updateSetting('SAMPLE_WIDTH', cfg.sample_width_default);
    if (cfg?.sample_height_default) updateSetting('SAMPLE_HEIGHT', cfg.sample_height_default);
    if (cfg?.sample_steps_default) updateSetting('SAMPLE_STEPS', cfg.sample_steps_default);
    if (cfg?.sample_cfg_default !== undefined) updateSetting('SAMPLE_CFG_SCALE', cfg.sample_cfg_default);
    if (cfg?.sample_flow_shift_default !== null && cfg?.sample_flow_shift_default !== undefined) {
      updateSetting('SAMPLE_FLOW_SHIFT', String(cfg.sample_flow_shift_default));
    }
  };

  const getBannerSubtitle = () => {
    if (isMinimax) {
      return 'Preview prompts rendered periodically during training, as short clips on the model being trained. Samples land in <output_dir>/sample/ and the Gallery button below opens the viewer.';
    }
    if (isKlein) {
      return 'Preview prompts rendered periodically during training (Distilled 4-step). Samples land in <output_dir>/sample/ and the Gallery button below opens the viewer.';
    }
    return 'Preview prompts rendered periodically during training. Samples land in <output_dir>/sample/ and the Gallery button below opens the viewer.';
  };

  return (
    <div className="fizgig-tab-page">
      {/* ── Banner ── */}
      <div className="tab-banner">
        <h1>Sample Previews</h1>
        <p>{getBannerSubtitle()}</p>
      </div>

      {/* ── Master Enable ── */}
      <div className="panel source-card">
        <label className="check-line" style={{ margin: 0 }}>
          <input
            type="checkbox"
            checked={settings.SAMPLE_ENABLED}
            onChange={(e) => updateSetting('SAMPLE_ENABLED', e.target.checked)}
          />
          <span style={{ fontSize: 14, fontWeight: 'bold' }}>Enable Sample Generation</span>
        </label>
        <p className="explanation" style={{ margin: '4px 0 0 24px' }}>
          When ticked, generates preview images periodically during training so you can monitor quality and likeness without stopping.
        </p>
      </div>

      {/* Architecture selector if multi-arch */}
      {ARCHITECTURE_LIST.length > 1 && (
        <div className="panel source-card">
          <h2>Architecture</h2>
          <div className="field-row">
            <label>Model Family:</label>
            <select
              value={architecture}
              onChange={(e) => handleArchChange(e.target.value)}
              style={{ maxWidth: 280 }}
            >
              {ARCHITECTURE_LIST.map(a => <option key={a} value={a}>{a}</option>)}
            </select>
          </div>
        </div>
      )}

      {/* ── Card 1: Prompt & Dimensions ── */}
      <div className="panel source-card">
        <h2>Prompt &amp; Dimensions</h2>
        <p className="muted">Configure what gets rendered. Separate multiple prompts with newlines — each will produce its own sample image.</p>

        {/* Prompt */}
        <div style={{ marginTop: 12 }}>
          <label style={{ display: 'block', fontSize: 13, fontWeight: 500, color: '#f3f5f7', marginBottom: 4 }}>
            Sample Prompt:
          </label>
          <textarea
            rows={3}
            value={settings.SAMPLE_PROMPT}
            onChange={(e) => updateSetting('SAMPLE_PROMPT', e.target.value)}
            style={{ width: '100%', fontFamily: 'monospace', fontSize: 12 }}
          />
          <p className="hint">Tip: Use [trigger] as a placeholder — it will be automatically replaced with the trigger word from the Captions tab.</p>
        </div>

        {/* Dimensions grid */}
        <div className="field-grid" style={{ marginTop: 12 }}>
          <div className="field">
            <span>Width:</span>
            <select
              value={String(settings.SAMPLE_WIDTH)}
              onChange={(e) => updateSetting('SAMPLE_WIDTH', parseInt(e.target.value, 10) || 768)}
            >
              {SAMPLE_RESOLUTIONS.map(r => <option key={r} value={String(r)}>{r}</option>)}
            </select>
          </div>

          <div className="field">
            <span>Height:</span>
            <select
              value={String(settings.SAMPLE_HEIGHT)}
              onChange={(e) => updateSetting('SAMPLE_HEIGHT', parseInt(e.target.value, 10) || 768)}
            >
              {SAMPLE_RESOLUTIONS.map(r => <option key={r} value={String(r)}>{r}</option>)}
            </select>
          </div>

          <div className="field">
            <span>Steps:</span>
            <input
              type="text"
              value={settings.SAMPLE_STEPS}
              onChange={(e) => updateSetting('SAMPLE_STEPS', parseInt(e.target.value, 10) || 4)}
              style={{ width: 80 }}
            />
          </div>

          <div className="field">
            <span>Seed:</span>
            <input
              type="text"
              value={settings.SAMPLE_SEED}
              onChange={(e) => updateSetting('SAMPLE_SEED', parseInt(e.target.value, 10) || 42)}
              style={{ width: 80 }}
            />
          </div>
        </div>

        {/* MiniMax H3 sample length */}
        {isMinimax && (
          <div className="field-row" style={{ marginTop: 12 }}>
            <label>Sample length:</label>
            <select
              value={settings.SAMPLE_FRAMES}
              onChange={(e) => updateSetting('SAMPLE_FRAMES', e.target.value)}
              style={{ maxWidth: 320 }}
            >
              {sampleFramesOptions.map(o => <option key={o} value={o}>{o}</option>)}
            </select>
          </div>
        )}
      </div>

      {/* ── Card 2: Generation Frequency ── */}
      <div className="panel source-card">
        <h2>Generation Frequency</h2>
        <p className="muted">Control when and how often sample images are produced.</p>

        <div className="field-grid" style={{ marginTop: 12 }}>
          <div className="field">
            <span>Every N Epochs:</span>
            <input
              type="text"
              value={settings.SAMPLE_EVERY_N_EPOCHS}
              onChange={(e) => updateSetting('SAMPLE_EVERY_N_EPOCHS', parseInt(e.target.value, 10) || 1)}
              style={{ width: 80 }}
            />
          </div>

          <div className="field">
            <span>Every N Steps:</span>
            <input
              type="text"
              value={settings.SAMPLE_EVERY_N_STEPS}
              onChange={(e) => updateSetting('SAMPLE_EVERY_N_STEPS', parseInt(e.target.value, 10) || 0)}
              style={{ width: 80 }}
            />
          </div>
        </div>

        <label className="check-line" style={{ marginTop: 12 }}>
          <input
            type="checkbox"
            checked={settings.SAMPLE_AT_FIRST}
            onChange={(e) => updateSetting('SAMPLE_AT_FIRST', e.target.checked)}
          />
          <span>Sample at first step (step 1 preview before training begins)</span>
        </label>
      </div>

      {/* ── Card 3: Advanced ── */}
      <div className="panel source-card">
        <h2>Advanced</h2>
        <p className="muted">Architecture-specific inference tuning.</p>

        <div className="field-grid" style={{ marginTop: 12 }}>
          <div className="field">
            <span>Flow Shift:</span>
            <input
              type="text"
              value={settings.SAMPLE_FLOW_SHIFT}
              onChange={(e) => updateSetting('SAMPLE_FLOW_SHIFT', e.target.value)}
              style={{ width: 80 }}
            />
          </div>

          <div className="field">
            <span>CFG Scale:</span>
            <input
              type="text"
              value={settings.SAMPLE_CFG_SCALE}
              onChange={(e) => updateSetting('SAMPLE_CFG_SCALE', parseFloat(e.target.value) || 1.0)}
              style={{ width: 80 }}
            />
          </div>

          <div className="field" style={{ gridColumn: '1 / -1' }}>
            <span>Negative Prompt:</span>
            <input
              type="text"
              value={settings.SAMPLE_NEGATIVE}
              onChange={(e) => updateSetting('SAMPLE_NEGATIVE', e.target.value)}
              placeholder="Optional negative prompt for guidance"
            />
          </div>
        </div>
      </div>

      {/* ── Card 4: Viewer / Output ── */}
      <div className="panel source-card">
        <h2>Viewer</h2>
        <p className="muted">
          Output directory: <code style={{ color: '#38bdf8' }}>{state.prefs.lora_output_dir || 'output_loras'}</code>
        </p>
        <div style={{ display: 'flex', gap: 12, marginTop: 8 }}>
          <button
            type="button"
            className="primary"
            onClick={() => {
              loadGallery();
              setShowGallery(!showGallery);
            }}
          >
            {showGallery ? 'Hide Gallery' : `View Samples Gallery (${sampleGallery.length})`}
          </button>
          <button
            type="button"
            className="secondary"
            onClick={() => alert(`Samples save to: ${state.prefs.lora_output_dir}/samples`)}
          >
            Open Samples Folder
          </button>
        </div>

        {/* Gallery View */}
        {showGallery && (
          <div style={{ marginTop: 16, borderTop: '1px solid #334155', paddingTop: 14 }}>
            {loadingGallery ? (
              <p style={{ color: '#8a9bae' }}>Loading samples...</p>
            ) : sampleGallery.length === 0 ? (
              <p style={{ color: '#5a6b7e' }}>No samples found yet in {state.prefs.lora_output_dir}. Start training to generate samples.</p>
            ) : (
              <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fill, minmax(200px, 1fr))', gap: 12 }}>
                {sampleGallery.map((s) => (
                  <div key={s.path} style={{ background: '#18212b', padding: 8, borderRadius: 4, border: '1px solid #334155' }}>
                    <div style={{ height: 140, background: '#111827', display: 'flex', alignItems: 'center', justifyContent: 'center', color: '#64748b', fontSize: 11 }}>
                      🖼️ {s.name}
                    </div>
                    <div style={{ fontSize: 11, color: '#94a3b8', marginTop: 4, overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>
                      {s.name}
                    </div>
                  </div>
                ))}
              </div>
            )}
          </div>
        )}
      </div>
    </div>
  );
}
