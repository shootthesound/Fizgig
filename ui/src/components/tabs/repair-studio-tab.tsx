'use client';

import React, { useState } from 'react';

const REPAIR_CAT_COLOR: Record<string, string> = {
  style_composition: '#5B9BD5',
  style_ident_overlap: '#5BB3A6',
  identity: '#70AD47',
  ident_details_overlap: '#B8A547',
  details: '#ED7D31',
};

const REPAIR_CAT_SHORT: Record<string, string> = {
  style_composition: 'Style+Comp',
  style_ident_overlap: 'Style/ID',
  identity: 'Identity',
  ident_details_overlap: 'ID/Detail',
  details: 'Details',
};

const REPAIR_MASTER_CATS = [
  { id: 'style_composition', label: 'Style+Comp' },
  { id: 'style_ident_overlap', label: 'Style/ID' },
  { id: 'identity', label: 'Identity' },
  { id: 'ident_details_overlap', label: 'ID/Detail' },
  { id: 'details', label: 'Details' },
];

function getKleinCategory(blockId: string): string {
  const [kind, numStr] = blockId.split('_');
  const idx = parseInt(numStr, 10);
  if (kind === 'double') return 'style_composition';
  if (idx === 0) return 'style_composition';
  if (idx === 1) return 'style_ident_overlap';
  if (idx >= 2 && idx <= 11) return 'identity';
  if (idx >= 12 && idx <= 16) return 'ident_details_overlap';
  return 'details';
}

function getBlockDisplay(blockId: string) {
  if (blockId.startsWith('h3blk_')) {
    return { name: `Block ${blockId.split('_')[1]}`, color: '#8a9bae', cat: null };
  }
  if (blockId.startsWith('h3_rf_')) {
    return { name: `Refiner ${blockId.split('_')[2]}`, color: '#8a9bae', cat: null };
  }
  if (blockId.startsWith('block_')) {
    return { name: `Block ${blockId.split('_')[1]}`, color: '#8a9bae', cat: null };
  }
  if (blockId.startsWith('txt_')) {
    const parts = blockId.split('_');
    return { name: `Txt ${parts[1].toUpperCase()} ${parts[2]}`, color: '#8a9bae', cat: null };
  }
  const cat = getKleinCategory(blockId);
  const [kind, idx] = blockId.split('_');
  return {
    name: `${kind} ${idx}`,
    color: REPAIR_CAT_COLOR[cat],
    cat: REPAIR_CAT_SHORT[cat],
  };
}

export default function RepairStudioTab() {
  const [repairFamily, setRepairFamily] = useState<'klein' | 'krea2' | 'minimax'>('klein');
  const [ditChoice, setDitChoice] = useState('distilled');
  const [primaryLora, setPrimaryLora] = useState('');
  const [primaryScale, setPrimaryScale] = useState('1.0');
  const [donorLora, setDonorLora] = useState('');
  const [donorScale, setDonorScale] = useState('1.0');
  const [prompt, setPrompt] = useState('');
  const [seed, setSeed] = useState('42');
  const [res, setRes] = useState('512');
  const [turboPreview, setTurboPreview] = useState(true);

  // MiniMax H3 specifics
  const [h3Model, setH3Model] = useState('First/last frame (fl2va) — standard');
  const [h3Base, setH3Base] = useState('Auto (by free VRAM)');
  const [h3Length, setH3Length] = useState('22 frames (~1s)');
  const [h3Width, setH3Width] = useState('768');
  const [h3Height, setH3Height] = useState('640');
  const [h3DialScale, setH3DialScale] = useState('Full');
  const [h3Steps, setH3Steps] = useState('4');
  const [h3Turbo, setH3Turbo] = useState('1.0');
  const [h3Sound, setH3Sound] = useState(true);
  const [h3Early, setH3Early] = useState(true);
  const [h3NoLora, setH3NoLora] = useState(false);

  // Reference Image (Klein & Krea 2)
  const [refPath, setRefPath] = useState('');
  const [refMp, setRefMp] = useState('1.0');
  const [refStrength, setRefStrength] = useState('1.0');

  // Preset
  const [preset, setPreset] = useState('');

  // Status message
  const [statusMsg] = useState('Set a LoRA path and prompt, then click Start.');

  // Profile-match info sidecar panel state
  const [profileMatch] = useState<{
    styleCompPct: number;
    identityPct: number;
    detailsPct: number;
  } | null>(null);

  // Master Sliders (Klein only)
  const [masterTarget, setMasterTarget] = useState<'primary' | 'donor'>('primary');
  const [masterStrengths, setMasterStrengths] = useState<Record<string, number>>({
    style_composition: 1.0,
    style_ident_overlap: 1.0,
    identity: 1.0,
    ident_details_overlap: 1.0,
    details: 1.0,
  });

  const isKlein = repairFamily === 'klein';
  const isKrea2 = repairFamily === 'krea2';
  const isMinimax = repairFamily === 'minimax';

  const hasDonor = Boolean(donorLora.trim());

  // Block definitions for each family
  const blockIds = isKlein
    ? [
        ...Array.from({ length: 8 }, (_, i) => `double_${i}`),
        ...Array.from({ length: 24 }, (_, i) => `single_${i}`),
      ]
    : isKrea2
    ? [
        ...Array.from({ length: 28 }, (_, i) => `block_${i}`),
        'txt_lw_0',
        'txt_lw_1',
        'txt_rf_0',
        'txt_rf_1',
      ]
    : [
        ...Array.from({ length: 50 }, (_, i) => `h3blk_${i}`),
        'h3_rf_0',
        'h3_rf_1',
      ];

  // Per-block Primary and Donor state
  const [primaryEnabled, setPrimaryEnabled] = useState<Record<string, boolean>>(() => {
    const initial: Record<string, boolean> = {};
    blockIds.forEach((id) => (initial[id] = true));
    return initial;
  });

  const [primaryStrengths, setPrimaryStrengths] = useState<Record<string, number>>(() => {
    const initial: Record<string, number> = {};
    blockIds.forEach((id) => (initial[id] = 1.0));
    return initial;
  });

  const [donorEnabled, setDonorEnabled] = useState<Record<string, boolean>>(() => {
    const initial: Record<string, boolean> = {};
    blockIds.forEach((id) => (initial[id] = true));
    return initial;
  });

  const [donorStrengths, setDonorStrengths] = useState<Record<string, number>>(() => {
    const initial: Record<string, number> = {};
    blockIds.forEach((id) => (initial[id] = 0.0));
    return initial;
  });

  const handleQuicksetMaster = (cat: string, val: number) => {
    setMasterStrengths((prev) => ({ ...prev, [cat]: val }));
    if (isKlein) {
      if (masterTarget === 'primary') {
        setPrimaryStrengths((prev) => {
          const next = { ...prev };
          blockIds.forEach((bid) => {
            if (getKleinCategory(bid) === cat) next[bid] = val;
          });
          return next;
        });
      } else {
        setDonorStrengths((prev) => {
          const next = { ...prev };
          blockIds.forEach((bid) => {
            if (getKleinCategory(bid) === cat) next[bid] = val;
          });
          return next;
        });
      }
    }
  };

  const handleBalanceMaster = (cat: string) => {
    const cur = masterStrengths[cat] ?? 1.0;
    const comp = 1.0 - cur;
    if (masterTarget === 'primary') {
      setDonorStrengths((prev) => {
        const next = { ...prev };
        blockIds.forEach((bid) => {
          if (getKleinCategory(bid) === cat) next[bid] = comp;
        });
        return next;
      });
    } else {
      setPrimaryStrengths((prev) => {
        const next = { ...prev };
        blockIds.forEach((bid) => {
          if (getKleinCategory(bid) === cat) next[bid] = comp;
        });
        return next;
      });
    }
  };

  const handleBalanceBlock = (bid: string, source: 'primary' | 'donor') => {
    if (source === 'primary') {
      const cur = primaryStrengths[bid] ?? 1.0;
      setDonorStrengths((prev) => ({ ...prev, [bid]: 1.0 - cur }));
      setDonorEnabled((prev) => ({ ...prev, [bid]: true }));
    } else {
      const cur = donorStrengths[bid] ?? 0.0;
      setPrimaryStrengths((prev) => ({ ...prev, [bid]: 1.0 - cur }));
    }
  };

  return (
    <div className="fizgig-tab-page">
      {/* ── Banner (lines 19280-19286) ── */}
      <div className="tab-banner">
        <h1>Repair Studio</h1>
        <p>
          Tweak each block&apos;s contribution live with side-by-side preview. Optional donor LoRA blends in via rank concatenation. Save the repaired result as a new .safetensors. Turbo Preview is on by default for faster updates — turn it off if VRAM is tight.
        </p>
      </div>

      {/* ── Setup Card (lines 19288-20213) ── */}
      <div className="panel source-card">
        <h2>Setup</h2>
        <p className="muted">
          Paths come from Preferences. Load the primary LoRA first; donor is optional. Changing prompt / seed / resolution triggers a fresh baseline render.
        </p>

        {/* Model Family selector */}
        <div style={{ display: 'flex', gap: 20, marginBottom: 10, alignItems: 'center' }}>
          <span style={{ fontSize: 13, color: '#8a9bae', minWidth: 60 }}>Model:</span>
          <label className="radio-option">
            <input
              type="radio"
              name="repair_family"
              value="klein"
              checked={repairFamily === 'klein'}
              onChange={() => setRepairFamily('klein')}
            />
            <span>Klein 9B</span>
          </label>
          <label className="radio-option">
            <input
              type="radio"
              name="repair_family"
              value="krea2"
              checked={repairFamily === 'krea2'}
              onChange={() => setRepairFamily('krea2')}
            />
            <span>Krea 2</span>
          </label>
          <label className="radio-option">
            <input
              type="radio"
              name="repair_family"
              value="minimax"
              checked={repairFamily === 'minimax'}
              onChange={() => setRepairFamily('minimax')}
            />
            <span>MiniMax H3</span>
          </label>
        </div>

        {/* DiT toggle */}
        <div style={{ display: 'flex', gap: 20, marginBottom: 12, alignItems: 'center' }}>
          <span style={{ fontSize: 13, color: '#8a9bae', minWidth: 60 }}>DiT:</span>
          {isMinimax ? (
            <span style={{ fontSize: 13, color: '#8a9bae' }}>Auto (int8/NF4 by VRAM + Turbo LoRA)</span>
          ) : isKrea2 ? (
            <>
              <label className="radio-option">
                <input
                  type="radio"
                  name="repair_dit"
                  value="distilled"
                  checked={ditChoice === 'distilled'}
                  onChange={() => setDitChoice('distilled')}
                />
                <span>Turbo (8-step, default)</span>
              </label>
              <label className="radio-option">
                <input
                  type="radio"
                  name="repair_dit"
                  value="base"
                  checked={ditChoice === 'base'}
                  onChange={() => setDitChoice('base')}
                />
                <span>RAW (slow, precise)</span>
              </label>
            </>
          ) : (
            <>
              <label className="radio-option">
                <input
                  type="radio"
                  name="repair_dit"
                  value="distilled"
                  checked={ditChoice === 'distilled'}
                  onChange={() => setDitChoice('distilled')}
                />
                <span>Distilled (4-step, fast)</span>
              </label>
              <label className="radio-option">
                <input
                  type="radio"
                  name="repair_dit"
                  value="base"
                  checked={ditChoice === 'base'}
                  onChange={() => setDitChoice('base')}
                />
                <span>Base (20-step, precise but slow)</span>
              </label>
            </>
          )}
        </div>

        {/* Primary LoRA */}
        <div className="field-grid" style={{ marginBottom: 8 }}>
          <div className="field" style={{ gridColumn: '1 / -1' }}>
            <span>Primary LoRA:</span>
            <div style={{ display: 'flex', gap: 8, alignItems: 'center' }}>
              <input
                type="text"
                value={primaryLora}
                onChange={(e) => setPrimaryLora(e.target.value)}
                style={{ flex: 1 }}
              />
              <button className="secondary" style={{ minHeight: 32 }}>
                Browse
              </button>
              {isMinimax && (
                <>
                  <span style={{ color: '#8a9bae', fontSize: 13, marginLeft: 8 }}>at strength</span>
                  <input
                    type="text"
                    value={primaryScale}
                    onChange={(e) => setPrimaryScale(e.target.value)}
                    style={{ width: 50 }}
                  />
                </>
              )}
            </div>
          </div>

          {/* Donor LoRA */}
          <div className="field" style={{ gridColumn: '1 / -1' }}>
            <span>Donor LoRA (optional):</span>
            <div style={{ display: 'flex', gap: 8, alignItems: 'center' }}>
              <input
                type="text"
                value={donorLora}
                onChange={(e) => setDonorLora(e.target.value)}
                style={{ flex: 1 }}
              />
              <button className="secondary" style={{ minHeight: 32 }}>
                Browse
              </button>
              <button className="secondary" style={{ minHeight: 32 }} onClick={() => setDonorLora('')}>
                Unload Donor
              </button>
              {isMinimax && (
                <>
                  <span style={{ color: '#8a9bae', fontSize: 13, marginLeft: 8 }}>at strength</span>
                  <input
                    type="text"
                    value={donorScale}
                    onChange={(e) => setDonorScale(e.target.value)}
                    style={{ width: 50 }}
                  />
                </>
              )}
            </div>
          </div>

          {/* MiniMax H3 Model & Base */}
          {isMinimax && (
            <>
              <div className="field">
                <span>Model:</span>
                <select value={h3Model} onChange={(e) => setH3Model(e.target.value)}>
                  <option value="First/last frame (fl2va) — standard">
                    First/last frame (fl2va) — standard
                  </option>
                  <option value="Reference (ref2va)">Reference (ref2va)</option>
                </select>
              </div>
              <div className="field">
                <span>Base:</span>
                <select value={h3Base} onChange={(e) => setH3Base(e.target.value)}>
                  <option value="Auto (by free VRAM)">Auto (by free VRAM)</option>
                  <option value="Stream blocks (exact int8, room for big clips)">
                    Stream blocks (exact int8, room for big clips)
                  </option>
                  <option value="NF4 (smallest, 9.5% base error)">NF4 (smallest, 9.5% base error)</option>
                </select>
              </div>
            </>
          )}

          {/* Prompt */}
          <div className="field" style={{ gridColumn: '1 / -1' }}>
            <span>Prompt:</span>
            <textarea
              className="prompt-editor"
              style={{ minHeight: 64, margin: '4px 0' }}
              value={prompt}
              onChange={(e) => setPrompt(e.target.value)}
            />
          </div>
        </div>

        {/* Seed + Res/Clip */}
        <div style={{ display: 'flex', gap: 16, alignItems: 'center', flexWrap: 'wrap', marginBottom: 8 }}>
          <div style={{ display: 'flex', gap: 6, alignItems: 'center' }}>
            <span style={{ color: '#8a9bae', fontSize: 13 }}>Seed:</span>
            <input
              type="text"
              value={seed}
              onChange={(e) => setSeed(e.target.value)}
              style={{ width: 80 }}
            />
            <button
              className="secondary"
              style={{ minHeight: 28, padding: '2px 8px' }}
              onClick={() => setSeed(String(Math.floor(Math.random() * 1000000)))}
            >
              ↻
            </button>
          </div>

          {!isMinimax && (
            <>
              <div style={{ display: 'flex', gap: 6, alignItems: 'center' }}>
                <span style={{ color: '#8a9bae', fontSize: 13 }}>Res:</span>
                <select value={res} onChange={(e) => setRes(e.target.value)} style={{ width: 70 }}>
                  {['256', '384', '512', '768'].map((v) => (
                    <option key={v} value={v}>
                      {v}
                    </option>
                  ))}
                </select>
              </div>
              {isKlein && (
                <label className="check-line">
                  <input
                    type="checkbox"
                    checked={turboPreview}
                    onChange={(e) => setTurboPreview(e.target.checked)}
                  />
                  <span>Turbo Preview</span>
                </label>
              )}
            </>
          )}
        </div>

        {/* MiniMax H3 Clip Controls */}
        {isMinimax && (
          <div style={{ background: '#1c2430', padding: 10, borderRadius: 4, marginBottom: 8 }}>
            <div style={{ display: 'flex', gap: 12, alignItems: 'center', flexWrap: 'wrap', marginBottom: 6 }}>
              <span style={{ color: '#8a9bae', fontSize: 12 }}>Length:</span>
              <select value={h3Length} onChange={(e) => setH3Length(e.target.value)}>
                {[
                  'Still (1 frame)',
                  '5 frames (~0.2s)',
                  '9 frames (~0.4s, off-grid)',
                  '13 frames (~0.5s, off-grid)',
                  '22 frames (~1s)',
                  '39 frames (~1.6s)',
                  '56 frames (~2.3s)',
                  '73 frames (~3s)',
                  '90 frames (~3.8s)',
                  '107 frames (~4.5s)',
                  '124 frames (~5.2s)',
                ].map((v) => (
                  <option key={v} value={v}>
                    {v}
                  </option>
                ))}
              </select>
              <span style={{ color: '#8a9bae', fontSize: 12 }}>W:</span>
              <select value={h3Width} onChange={(e) => setH3Width(e.target.value)} style={{ width: 65 }}>
                {['512', '640', '768', '960', '1024', '1152', '1280', '1536'].map((v) => (
                  <option key={v} value={v}>
                    {v}
                  </option>
                ))}
              </select>
              <span style={{ color: '#8a9bae', fontSize: 12 }}>H:</span>
              <select value={h3Height} onChange={(e) => setH3Height(e.target.value)} style={{ width: 65 }}>
                {['512', '640', '768', '960', '1024', '1152', '1280', '1536'].map((v) => (
                  <option key={v} value={v}>
                    {v}
                  </option>
                ))}
              </select>
              <span style={{ color: '#5a6b7e', fontSize: 11, fontStyle: 'italic' }}>
                H3 likes one side at 768+
              </span>
              <span style={{ color: '#8a9bae', fontSize: 12, marginLeft: 'auto' }}>Render size:</span>
              <select
                value={h3DialScale}
                onChange={(e) => setH3DialScale(e.target.value)}
                style={{ width: 70 }}
              >
                <option value="Full">Full</option>
                <option value="⅔">⅔</option>
                <option value="½">½</option>
              </select>
            </div>
            <div style={{ display: 'flex', gap: 12, alignItems: 'center', flexWrap: 'wrap' }}>
              <span style={{ color: '#8a9bae', fontSize: 12 }}>Steps:</span>
              <input
                type="text"
                value={h3Steps}
                onChange={(e) => setH3Steps(e.target.value)}
                style={{ width: 35 }}
              />
              <span style={{ color: '#8a9bae', fontSize: 12 }}>Turbo:</span>
              <input
                type="text"
                value={h3Turbo}
                onChange={(e) => setH3Turbo(e.target.value)}
                style={{ width: 45 }}
              />
              <label className="check-line">
                <input
                  type="checkbox"
                  checked={h3Sound}
                  onChange={(e) => setH3Sound(e.target.checked)}
                />
                <span>Sound</span>
              </label>
              <label className="check-line">
                <input
                  type="checkbox"
                  checked={h3Early}
                  onChange={(e) => setH3Early(e.target.checked)}
                />
                <span>Show early</span>
              </label>
              <label className="check-line">
                <input
                  type="checkbox"
                  checked={h3NoLora}
                  onChange={(e) => setH3NoLora(e.target.checked)}
                />
                <span>No-LoRA clip</span>
              </label>
            </div>
          </div>
        )}

        {/* Reference Image (Klein & Krea 2 only — lines 20137-20165) */}
        {!isMinimax && (
          <div style={{ display: 'flex', gap: 8, alignItems: 'center', marginBottom: 8, flexWrap: 'wrap' }}>
            <span style={{ color: '#8a9bae', fontSize: 13, minWidth: 60 }}>Reference:</span>
            <input
              type="text"
              value={refPath}
              readOnly
              style={{ flex: 1, minWidth: 150 }}
              placeholder="(optional image conditioning)"
            />
            <button className="secondary" style={{ minHeight: 32 }}>
              Browse
            </button>
            <button className="secondary" style={{ minHeight: 32 }} onClick={() => setRefPath('')}>
              Clear
            </button>
            <span style={{ color: '#8a9bae', fontSize: 13 }}>MP:</span>
            <select value={refMp} onChange={(e) => setRefMp(e.target.value)} style={{ width: 65 }}>
              {['0.25', '0.5', '1.0', '2.0'].map((v) => (
                <option key={v} value={v}>
                  {v}
                </option>
              ))}
            </select>
            {isKlein && (
              <>
                <span style={{ color: '#8a9bae', fontSize: 13 }}>Strength:</span>
                <input
                  type="text"
                  value={refStrength}
                  onChange={(e) => setRefStrength(e.target.value)}
                  style={{ width: 45 }}
                />
              </>
            )}
          </div>
        )}

        {/* Preset Row */}
        <div style={{ display: 'flex', gap: 8, alignItems: 'center', marginBottom: 12 }}>
          <span style={{ color: '#8a9bae', fontSize: 13, minWidth: 60 }}>Preset:</span>
          <select
            value={preset}
            onChange={(e) => setPreset(e.target.value)}
            style={{ flex: 1, maxWidth: 300 }}
          >
            <option value="">(Select Preset)</option>
            {isKlein && (
              <>
                <option value="Reset All">Reset All</option>
                <option value="Identity Only">Identity Only</option>
                <option value="Style Only">Style Only</option>
                <option value="Details Only">Details Only</option>
              </>
            )}
          </select>
          <button className="secondary" style={{ minHeight: 32 }}>
            Save Preset…
          </button>
        </div>

        {/* Status + Action buttons */}
        <div style={{ display: 'flex', gap: 10, alignItems: 'center', flexWrap: 'wrap', paddingTop: 6, borderTop: '1px solid #3a4555' }}>
          <span style={{ color: '#3b82f6', fontStyle: 'italic', fontSize: 13, flex: 1 }}>
            {statusMsg}
          </span>
          <button
            className="secondary"
            style={{ fontWeight: 'bold', background: '#3B6FA0', borderColor: '#3B6FA0', color: '#fff' }}
          >
            {isMinimax ? '▶ Play clips + Metrics' : '⧉ Compare + Metrics'}
          </button>
          <button className="secondary">Reset All Sliders</button>
          <button
            className="primary"
            style={{ background: '#2E8B57', borderColor: '#2E8B57', fontWeight: 'bold' }}
          >
            Start
          </button>
        </div>
      </div>

      {/* ── Profile Match Panel (lines 19297-19303) ── */}
      {profileMatch && (
        <div
          className="panel source-card"
          style={{ borderColor: '#3b82f6', background: '#1c2532', padding: 12 }}
        >
          <h2 style={{ fontSize: 13, color: '#60A5FA', margin: '0 0 6px 0' }}>
            Matching Profiler Sidecar Detected
          </h2>
          <p style={{ fontSize: 12, color: '#c3cdd9', margin: 0 }}>
            Style+Comp: {profileMatch.styleCompPct}% · Identity: {profileMatch.identityPct}% · Details: {profileMatch.detailsPct}%
          </p>
        </div>
      )}

      {/* ── First / Last Frame Card (MiniMax only — lines 19307-19321) ── */}
      {isMinimax && (
        <div className="panel source-card">
          <h2>First / Last Frame</h2>
          <p className="muted">
            Pin the clip&apos;s first and/or last frame to a photo (H3&apos;s fl2va conditioning) so the shot you judge is predictable. Pick a photo, crop it to the clip&apos;s shape, done. Applies to every render — sliders, library and Confirm alike.
          </p>
          <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 16, marginTop: 8 }}>
            <div>
              <small className="muted" style={{ display: 'block', marginBottom: 4 }}>
                First frame:
              </small>
              <div
                style={{
                  height: 128,
                  background: '#1c1c1c',
                  border: '1px dashed #3a4555',
                  display: 'flex',
                  alignItems: 'center',
                  justifyContent: 'center',
                }}
              >
                <span style={{ color: '#5a6b7e', fontSize: 11 }}>(drop photo here or Browse)</span>
              </div>
            </div>
            <div>
              <small className="muted" style={{ display: 'block', marginBottom: 4 }}>
                Last frame:
              </small>
              <div
                style={{
                  height: 128,
                  background: '#1c1c1c',
                  border: '1px dashed #3a4555',
                  display: 'flex',
                  alignItems: 'center',
                  justifyContent: 'center',
                }}
              >
                <span style={{ color: '#5a6b7e', fontSize: 11 }}>(drop photo here or Browse)</span>
              </div>
            </div>
          </div>
        </div>
      )}

      {/* ── Preview Card (lines 20215-20265) ── */}
      <div className="panel source-card">
        <h2>Preview</h2>
        <p className="muted">
          Left side is the baseline (LoRA at its original strengths); right side is the tweaked render using your slider state.
        </p>
        <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 16, marginTop: 8 }}>
          <div>
            <span
              style={{
                textAlign: 'center',
                display: 'block',
                fontSize: 12,
                fontWeight: 'bold',
                color: '#f0f4f8',
                marginBottom: 4,
              }}
            >
              Baseline (LoRA at default 1.0)
            </span>
            <div
              style={{
                aspectRatio: '1',
                maxHeight: 480,
                background: '#1c1c1c',
                border: '1px solid #3a4555',
                display: 'flex',
                alignItems: 'center',
                justifyContent: 'center',
                cursor: 'pointer',
              }}
            >
              <span style={{ color: '#5a6b7e', fontSize: 13 }}>(no baseline yet)</span>
            </div>
          </div>
          <div>
            <span
              style={{
                textAlign: 'center',
                display: 'block',
                fontSize: 12,
                fontWeight: 'bold',
                color: '#f0f4f8',
                marginBottom: 4,
              }}
            >
              Tweaked (current sliders)
            </span>
            <div
              style={{
                aspectRatio: '1',
                maxHeight: 480,
                background: '#1c1c1c',
                border: '1px solid #3a4555',
                display: 'flex',
                alignItems: 'center',
                justifyContent: 'center',
                cursor: 'pointer',
              }}
            >
              <span style={{ color: '#5a6b7e', fontSize: 13 }}>(no preview yet)</span>
            </div>
          </div>
        </div>
        <small
          style={{
            display: 'block',
            textAlign: 'center',
            marginTop: 8,
            color: '#8a9bae',
            fontStyle: 'italic',
          }}
        >
          {isMinimax
            ? '▶ Click either image to play both clips side by side — sound, swap, scrub — with likeness + quality metrics on the middle frame'
            : '🔍 Click either image for the full-size side-by-side compare with likeness + quality metrics'}
        </small>

        {/* MiniMax H3 Library Status Row (lines 20286-20308) */}
        {isMinimax && (
          <div
            style={{
              display: 'flex',
              gap: 10,
              alignItems: 'center',
              marginTop: 12,
              padding: '6px 8px',
              background: '#1c2430',
              borderRadius: 3,
            }}
          >
            <span style={{ color: '#8a9bae', fontSize: 12 }}>Library: no LoRA loaded.</span>
            <div style={{ flex: '0 0 140px', height: 12, background: '#18212b', borderRadius: 2, overflow: 'hidden' }}>
              <div style={{ width: '0%', height: '100%', background: '#10B981' }} />
            </div>
            <button className="secondary" style={{ padding: '2px 8px', fontSize: 11, minHeight: 24 }}>
              Pause build
            </button>
            <button className="secondary" style={{ padding: '2px 8px', fontSize: 11, minHeight: 24 }}>
              Clear cache…
            </button>
          </div>
        )}

        {/* MiniMax H3 History Row (lines 20310-20330) */}
        {isMinimax && (
          <div style={{ marginTop: 10 }}>
            <span style={{ color: '#8a9bae', fontSize: 12, display: 'block', marginBottom: 4 }}>
              History — click to view, right-click to save or pin as baseline
            </span>
            <div
              style={{
                height: 100,
                background: '#18212b',
                border: '1px solid #3a4555',
                display: 'flex',
                alignItems: 'center',
                padding: '0 8px',
                gap: 8,
                overflowX: 'auto',
              }}
            >
              <span style={{ color: '#5a6b7e', fontSize: 11 }}>(renders will appear here)</span>
            </div>
          </div>
        )}
      </div>

      {/* ── Master Controls (Klein only — lines 20384-20445) ── */}
      {isKlein && (
        <div className="panel source-card">
          <h2>Master Controls</h2>
          <p className="muted">
            Bulk-tune by category. Flip the target radio to switch between primary and donor. Category toggles next to the donor ones bulk on/off the donor&apos;s contribution per bucket.
          </p>
          <div style={{ display: 'flex', gap: 16, alignItems: 'center', marginBottom: 12 }}>
            <span style={{ fontSize: 13, fontWeight: 'bold', color: '#f0f4f8' }}>
              Master sliders affect:
            </span>
            <label className="radio-option">
              <input
                type="radio"
                name="master_target"
                value="primary"
                checked={masterTarget === 'primary'}
                onChange={() => setMasterTarget('primary')}
              />
              <span>Primary</span>
            </label>
            <label className="radio-option">
              <input
                type="radio"
                name="master_target"
                value="donor"
                checked={masterTarget === 'donor'}
                onChange={() => setMasterTarget('donor')}
              />
              <span>Donor</span>
            </label>
          </div>

          <div style={{ display: 'flex', flexDirection: 'column', gap: 6 }}>
            {REPAIR_MASTER_CATS.map((c) => (
              <div key={c.id} style={{ display: 'grid', gridTemplateColumns: '100px 1fr 50px 120px', gap: 8, alignItems: 'center' }}>
                <span style={{ color: REPAIR_CAT_COLOR[c.id], fontWeight: 'bold', fontSize: 12 }}>
                  {c.label}
                </span>
                <input
                  type="range"
                  min={-3}
                  max={3}
                  step={0.01}
                  value={masterStrengths[c.id]}
                  onChange={(e) => handleQuicksetMaster(c.id, parseFloat(e.target.value))}
                  style={{ accentColor: REPAIR_CAT_COLOR[c.id] }}
                />
                <span style={{ color: '#f0f4f8', fontSize: 12, textAlign: 'right', fontFamily: 'monospace' }}>
                  {masterStrengths[c.id].toFixed(2)}
                </span>
                <div style={{ display: 'flex', gap: 2 }}>
                  <button className="secondary" style={{ padding: '2px 6px', fontSize: 10, minHeight: 22 }} onClick={() => handleQuicksetMaster(c.id, 0)}>0</button>
                  <button className="secondary" style={{ padding: '2px 6px', fontSize: 10, minHeight: 22 }} onClick={() => handleQuicksetMaster(c.id, 1)}>1</button>
                  <button className="secondary" style={{ padding: '2px 6px', fontSize: 10, minHeight: 22 }} onClick={() => handleQuicksetMaster(c.id, -masterStrengths[c.id])}>±</button>
                  <button className="secondary" style={{ padding: '2px 6px', fontSize: 10, minHeight: 22 }} onClick={() => handleBalanceMaster(c.id)}>⚖</button>
                </div>
              </div>
            ))}
          </div>
        </div>
      )}

      {/* ── Per-Block Sliders Card (lines 20565-20849) ── */}
      <div className="panel source-card">
        <h2>Per-Block Sliders</h2>
        <p className="muted">
          Range ±3.0. Greyed-out rows are blocks the LoRA doesn&apos;t touch. Colour bands match the Profiler&apos;s 5-bucket scheme: blue Style+Comp, teal Style/ID, green Identity, olive ID/Detail, orange Details.
        </p>

        {/* Bulk row (lines 19348-19364) */}
        <div style={{ display: 'flex', gap: 6, alignItems: 'center', flexWrap: 'wrap', marginBottom: 8 }}>
          <button
            className="secondary"
            style={{ fontSize: 11, padding: '3px 8px', minHeight: 26 }}
            onClick={() => {
              setPrimaryStrengths((prev) => {
                const next = { ...prev };
                blockIds.forEach((id) => (next[id] = 1.0));
                return next;
              });
            }}
          >
            Reset all
          </button>
          <button
            className="secondary"
            style={{ fontSize: 11, padding: '3px 8px', minHeight: 26 }}
            onClick={() => {
              setPrimaryEnabled((prev) => {
                const next = { ...prev };
                blockIds.forEach((id) => (next[id] = false));
                return next;
              });
            }}
          >
            All off
          </button>
          <button
            className="secondary"
            style={{ fontSize: 11, padding: '3px 8px', minHeight: 26 }}
            onClick={() => {
              setPrimaryEnabled((prev) => {
                const next = { ...prev };
                blockIds.forEach((id) => (next[id] = true));
                return next;
              });
            }}
          >
            All on
          </button>
          <button
            className="secondary"
            style={{ fontSize: 11, padding: '3px 8px', minHeight: 26 }}
            onClick={() => {
              setPrimaryEnabled((prev) => {
                const next = { ...prev };
                blockIds.forEach((id) => (next[id] = !next[id]));
                return next;
              });
            }}
          >
            Invert
          </button>
          <button
            className="secondary"
            style={{ fontSize: 11, padding: '3px 8px', minHeight: 26 }}
            onClick={() => {
              setPrimaryEnabled((prev) => {
                const next = { ...prev };
                blockIds.forEach((id, idx) => (next[id] = idx % 2 === 0));
                return next;
              });
            }}
          >
            Alternate
          </button>
          <button
            className="secondary"
            style={{ fontSize: 11, padding: '3px 8px', minHeight: 26 }}
            onClick={() => {
              setPrimaryEnabled((prev) => {
                const next = { ...prev };
                blockIds.slice(-4).forEach((id) => (next[id] = !next[id]));
                return next;
              });
            }}
          >
            Toggle detail blocks
          </button>
        </div>

        {/* MiniMax H3 Bank Chips Strip (lines 19368-19371) */}
        {isMinimax && (
          <div style={{ display: 'flex', gap: 6, alignItems: 'center', marginBottom: 10, padding: '4px 6px', background: '#1c2430', borderRadius: 3 }}>
            <span style={{ color: '#8a9bae', fontSize: 11 }}>Library:</span>
            {Array.from({ length: 10 }, (_, i) => (
              <span
                key={i}
                style={{
                  fontSize: 11,
                  color: '#5a6b7e',
                  cursor: 'pointer',
                  padding: '1px 4px',
                  background: '#18212b',
                  borderRadius: 2,
                }}
                title={`Bank ${i + 1} (blocks ${i * 5}-${i * 5 + 4})`}
              >
                ○ {i * 5}–{i * 5 + 4}
              </span>
            ))}
          </div>
        )}

        {/* Scrollable Sliders list */}
        <div
          style={{
            maxHeight: 520,
            overflowY: 'auto',
            border: '1px solid #3a4555',
            padding: 8,
            background: '#18212b',
          }}
        >
          {blockIds.map((bid) => {
            const { name, color, cat } = getBlockDisplay(bid);
            const isPEnabled = primaryEnabled[bid] ?? true;
            const pStrVal = primaryStrengths[bid] ?? 1.0;
            const isDEnabled = donorEnabled[bid] ?? true;
            const dStrVal = donorStrengths[bid] ?? 0.0;

            return (
              <div
                key={bid}
                style={{
                  padding: '3px 0',
                  borderBottom: '1px solid #252e3b',
                }}
              >
                {/* Primary Row */}
                <div
                  style={{
                    display: 'grid',
                    gridTemplateColumns: '24px 85px 80px 1fr 45px 110px',
                    gap: 6,
                    alignItems: 'center',
                    opacity: isPEnabled ? 1 : 0.4,
                  }}
                >
                  <input
                    type="checkbox"
                    checked={isPEnabled}
                    onChange={(e) => setPrimaryEnabled((prev) => ({ ...prev, [bid]: e.target.checked }))}
                    style={{ accentColor: '#3b82f6' }}
                  />
                  <span style={{ color, fontFamily: 'Consolas, monospace', fontSize: 12, fontWeight: 'bold' }}>
                    {name}
                  </span>
                  <span style={{ color, fontSize: 11, fontStyle: 'italic' }}>
                    {cat ? `[${cat}]` : ''}
                  </span>
                  <input
                    type="range"
                    min={-3}
                    max={3}
                    step={0.01}
                    disabled={!isPEnabled}
                    value={pStrVal}
                    onChange={(e) => {
                      const val = parseFloat(e.target.value);
                      setPrimaryStrengths((prev) => ({ ...prev, [bid]: val }));
                    }}
                    style={{ accentColor: color }}
                  />
                  <span style={{ color: '#f0f4f8', fontFamily: 'monospace', fontSize: 11, textAlign: 'right' }}>
                    {pStrVal.toFixed(2)}
                  </span>
                  <div style={{ display: 'flex', gap: 2 }}>
                    <button className="secondary" style={{ padding: '1px 5px', fontSize: 10, minHeight: 20 }} onClick={() => setPrimaryStrengths((prev) => ({ ...prev, [bid]: 0 }))}>0</button>
                    <button className="secondary" style={{ padding: '1px 5px', fontSize: 10, minHeight: 20 }} onClick={() => setPrimaryStrengths((prev) => ({ ...prev, [bid]: 1 }))}>1</button>
                    <button className="secondary" style={{ padding: '1px 5px', fontSize: 10, minHeight: 20 }} onClick={() => setPrimaryStrengths((prev) => ({ ...prev, [bid]: -pStrVal }))}>±</button>
                    <button className="secondary" style={{ padding: '1px 5px', fontSize: 10, minHeight: 20 }} onClick={() => handleBalanceBlock(bid, 'primary')}>⚖</button>
                  </div>
                </div>

                {/* Donor Row (visible when donor LoRA is loaded — lines 20775-20793) */}
                {hasDonor && (
                  <div
                    style={{
                      display: 'grid',
                      gridTemplateColumns: '24px 85px 80px 1fr 45px 110px',
                      gap: 6,
                      alignItems: 'center',
                      paddingLeft: 20,
                      marginTop: 2,
                      opacity: isDEnabled ? 1 : 0.4,
                    }}
                  >
                    <input
                      type="checkbox"
                      checked={isDEnabled}
                      onChange={(e) => setDonorEnabled((prev) => ({ ...prev, [bid]: e.target.checked }))}
                      style={{ accentColor: '#3b82f6' }}
                    />
                    <span style={{ color: '#888', fontStyle: 'italic', fontSize: 11 }}>
                      donor
                    </span>
                    <span />
                    <input
                      type="range"
                      min={-3}
                      max={3}
                      step={0.01}
                      disabled={!isDEnabled}
                      value={dStrVal}
                      onChange={(e) => {
                        const val = parseFloat(e.target.value);
                        setDonorStrengths((prev) => ({ ...prev, [bid]: val }));
                      }}
                      style={{ accentColor: '#888' }}
                    />
                    <span style={{ color: '#f0f4f8', fontFamily: 'monospace', fontSize: 11, textAlign: 'right' }}>
                      {dStrVal.toFixed(2)}
                    </span>
                    <div style={{ display: 'flex', gap: 2 }}>
                      <button className="secondary" style={{ padding: '1px 5px', fontSize: 10, minHeight: 20 }} onClick={() => setDonorStrengths((prev) => ({ ...prev, [bid]: 0 }))}>0</button>
                      <button className="secondary" style={{ padding: '1px 5px', fontSize: 10, minHeight: 20 }} onClick={() => setDonorStrengths((prev) => ({ ...prev, [bid]: 1 }))}>1</button>
                      <button className="secondary" style={{ padding: '1px 5px', fontSize: 10, minHeight: 20 }} onClick={() => setDonorStrengths((prev) => ({ ...prev, [bid]: -dStrVal }))}>±</button>
                      <button className="secondary" style={{ padding: '1px 5px', fontSize: 10, minHeight: 20 }} onClick={() => handleBalanceBlock(bid, 'donor')}>⚖</button>
                    </div>
                  </div>
                )}
              </div>
            );
          })}
        </div>
      </div>

      {/* ── Actions Card (lines 19380-19395) ── */}
      <div className="actions" style={{ marginTop: 12 }}>
        <button
          type="button"
          className="primary"
          onClick={async () => {
            if (!primaryLora) {
              alert('Please specify a primary LoRA first.');
              return;
            }
            const outPath = window.prompt(
              'Save repaired LoRA as:',
              primaryLora.replace('.safetensors', '_repaired.safetensors')
            );
            if (!outPath) return;
            try {
              const res = await fetch('/api/repair', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                  action: 'save_repaired',
                  loraPath: primaryLora,
                  donorPath: donorLora,
                  outputPath: outPath,
                }),
              });
              const data = await res.json();
              if (data.success) {
                alert(`Repaired LoRA successfully written to: ${data.savedTo}`);
              } else {
                alert('Repair save failed: ' + (data.error || 'Unknown error'));
              }
            } catch (e: any) {
              alert('Error saving repaired LoRA: ' + e?.message);
            }
          }}
        >
          Save Repaired LoRA…
        </button>
        <button
          type="button"
          className="secondary"
          onClick={() => {
            setPrimaryStrengths((prev) => {
              const next = { ...prev };
              blockIds.forEach((id) => (next[id] = 1.0));
              return next;
            });
            setPrimaryEnabled((prev) => {
              const next = { ...prev };
              blockIds.forEach((id) => (next[id] = true));
              return next;
            });
          }}
        >
          Reset All Sliders
        </button>
        <button
          type="button"
          className="secondary"
          onClick={() => alert('Repair studio session reset.')}
        >
          Reset Session (unload models)
        </button>
        <button
          type="button"
          className="primary"
          style={{
            marginLeft: 'auto',
            background: '#2E8B57',
            borderColor: '#2E8B57',
            fontWeight: 'bold',
          }}
          onClick={() => alert('Opening LoRA the Explorer with current slider parameters.')}
        >
          Explore this in LoRA the Explorer →
        </button>
      </div>
    </div>
  );
}
