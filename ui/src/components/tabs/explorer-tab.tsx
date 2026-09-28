'use client';

import React, { useState } from 'react';

export default function ExplorerTab() {
  const [family, setFamily] = useState<'klein' | 'krea2' | 'minimax'>('klein');
  const [ditChoice, setDitChoice] = useState('distilled');
  const [loraPath, setLoraPath] = useState('');
  const [strength, setStrength] = useState('1.0');
  const [prompt, setPrompt] = useState('');
  const [refPath, setRefPath] = useState('');
  const [refMp, setRefMp] = useState('1.0');
  const [refStrength, setRefStrength] = useState('1.0');
  const [seed, setSeed] = useState('42');
  const [res, setRes] = useState('512');
  const [intensity, setIntensity] = useState(0.964);
  const [mutations, setMutations] = useState('8');
  const [structure, setStructure] = useState(1.0);
  const [statusMsg, setStatusMsg] = useState('Set a LoRA path and prompt, then click Start.');
  const [activeVariant, setActiveVariant] = useState<number | null>(null);

  const isKlein = family === 'klein';
  const isKrea2 = family === 'krea2';
  const isMinimax = family === 'minimax';

  const intensityMag = (0.2 + intensity * 2.8).toFixed(1);

  return (
    <div className="fizgig-tab-page">
      {/* ── Banner (lines 15270-15275) ── */}
      <div className="tab-banner">
        <h1>LoRA the Explorer</h1>
        <p>
          The computer randomly adjusts blocks and shows you 4 variants — pick your favourite and it evolves. Find a direction you like? Reduce Structure to stabilise composition, then Freeze to lock tweaked blocks in place.
        </p>
      </div>

      {/* ── Model Family Card (lines 15294-15308) ── */}
      <div className="panel source-card">
        <h2>Model Family</h2>
        <p className="muted">
          Klein 9B (Distilled/Base), Krea 2 (fp8 Turbo, 8-step) or MiniMax H3 (22-frame clip previews, middle frame shown). Block roles are mapped for Klein; the other two explore their blocks generically.
        </p>
        <div style={{ display: 'flex', gap: 24 }}>
          <label className="radio-option">
            <input
              type="radio"
              name="explorer_family"
              value="klein"
              checked={family === 'klein'}
              onChange={() => setFamily('klein')}
            />
            <span>Klein 9B</span>
          </label>
          <label className="radio-option">
            <input
              type="radio"
              name="explorer_family"
              value="krea2"
              checked={family === 'krea2'}
              onChange={() => setFamily('krea2')}
            />
            <span>Krea 2</span>
          </label>
          <label className="radio-option">
            <input
              type="radio"
              name="explorer_family"
              value="minimax"
              checked={family === 'minimax'}
              onChange={() => setFamily('minimax')}
            />
            <span>MiniMax H3</span>
          </label>
        </div>
      </div>

      {/* ── Setup Card (lines 15309-15452) ── */}
      <div className="panel source-card">
        <h2>Setup</h2>
        <p className="muted">Load a LoRA and configure the exploration parameters.</p>

        {/* DiT toggle (Klein only — hidden in Krea 2 & MiniMax lines 15608-15631) */}
        {isKlein && (
          <div style={{ display: 'flex', gap: 16, alignItems: 'center', marginBottom: 8 }}>
            <span style={{ color: '#8a9bae', fontSize: 13, minWidth: 60 }}>DiT:</span>
            <label className="radio-option">
              <input
                type="radio"
                name="explorer_dit"
                value="distilled"
                checked={ditChoice === 'distilled'}
                onChange={() => setDitChoice('distilled')}
              />
              <span>Distilled (4-step, fast)</span>
            </label>
            <label className="radio-option">
              <input
                type="radio"
                name="explorer_dit"
                value="base"
                checked={ditChoice === 'base'}
                onChange={() => setDitChoice('base')}
              />
              <span>Base (20-step, precise)</span>
            </label>
          </div>
        )}

        {/* LoRA + Strength */}
        <div className="field-grid" style={{ marginBottom: 8 }}>
          <div className="field" style={{ gridColumn: '1 / -1' }}>
            <span>LoRA:</span>
            <div style={{ display: 'flex', gap: 8, alignItems: 'center' }}>
              <input
                type="text"
                value={loraPath}
                onChange={(e) => setLoraPath(e.target.value)}
                style={{ flex: 1 }}
              />
              <button className="secondary" style={{ minHeight: 32 }}>
                Browse
              </button>
              <span style={{ color: '#8a9bae', fontSize: 13, marginLeft: 8 }}>Strength:</span>
              <input
                type="text"
                value={strength}
                onChange={(e) => setStrength(e.target.value)}
                style={{ width: 50 }}
              />
            </div>
          </div>

          {/* Prompt */}
          <div className="field" style={{ gridColumn: '1 / -1' }}>
            <span>Prompt:</span>
            <input
              type="text"
              value={prompt}
              onChange={(e) => setPrompt(e.target.value)}
              placeholder=""
            />
          </div>
        </div>

        {/* Reference Image (Klein & Krea 2 — hidden for MiniMax line 15608) */}
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
                  style={{ width: 50 }}
                />
              </>
            )}
          </div>
        )}

        {/* Params row: Seed, Res, Intensity, Mutations */}
        <div style={{ display: 'flex', gap: 14, alignItems: 'center', flexWrap: 'wrap', marginBottom: 8 }}>
          <div style={{ display: 'flex', gap: 6, alignItems: 'center' }}>
            <span style={{ color: '#8a9bae', fontSize: 13 }}>Seed:</span>
            <input
              type="text"
              value={seed}
              onChange={(e) => setSeed(e.target.value)}
              style={{ width: 70 }}
            />
            <button
              className="secondary"
              style={{ minHeight: 28, padding: '2px 8px' }}
              onClick={() => setSeed(String(Math.floor(Math.random() * 1000000)))}
            >
              ↻
            </button>
          </div>

          <div style={{ display: 'flex', gap: 6, alignItems: 'center' }}>
            <span style={{ color: '#8a9bae', fontSize: 13 }}>Res:</span>
            <select value={res} onChange={(e) => setRes(e.target.value)} style={{ width: 65 }}>
              {['256', '384', '512', '768'].map((v) => (
                <option key={v} value={v}>
                  {v}
                </option>
              ))}
            </select>
          </div>

          <div style={{ display: 'flex', gap: 6, alignItems: 'center' }}>
            <span style={{ color: '#8a9bae', fontSize: 13 }}>Intensity:</span>
            <input
              type="range"
              min={0.0}
              max={1.0}
              step={0.01}
              value={intensity}
              onChange={(e) => setIntensity(parseFloat(e.target.value))}
              style={{ width: 100, accentColor: '#3b82f6' }}
            />
            <span style={{ color: '#f0f4f8', fontSize: 12, minWidth: 40 }}>±{intensityMag}</span>
          </div>

          <div style={{ display: 'flex', gap: 6, alignItems: 'center' }}>
            <span style={{ color: '#8a9bae', fontSize: 13 }}>Mutations:</span>
            <select
              value={mutations}
              onChange={(e) => setMutations(e.target.value)}
              style={{ width: 50 }}
            >
              {['1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '12', '14', '16'].map((v) => (
                <option key={v} value={v}>
                  {v}
                </option>
              ))}
            </select>
          </div>
        </div>

        {/* Structure change row */}
        <div style={{ display: 'flex', gap: 8, alignItems: 'center', marginBottom: 8 }}>
          <span style={{ color: '#8a9bae', fontSize: 13 }}>Structure change:</span>
          <input
            type="range"
            min={0.0}
            max={1.0}
            step={0.01}
            value={structure}
            onChange={(e) => setStructure(parseFloat(e.target.value))}
            style={{ width: 120, accentColor: '#3b82f6' }}
          />
          <span style={{ color: '#f0f4f8', fontSize: 12, minWidth: 45 }}>{Math.round(structure * 100)}%</span>
        </div>

        {/* Status + Start row */}
        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', paddingTop: 6, borderTop: '1px solid #3a4555' }}>
          <span style={{ color: '#5a6b7e', fontSize: 11, fontStyle: 'italic' }}>
            Increase Structure if variants look too similar to baseline.
          </span>
          <div style={{ display: 'flex', gap: 12, alignItems: 'center' }}>
            <span style={{ color: '#3b82f6', fontStyle: 'italic', fontSize: 12 }}>
              {statusMsg}
            </span>
            <button
              type="button"
              className="primary"
              style={{ background: '#2E8B57', borderColor: '#2E8B57', fontWeight: 'bold' }}
              onClick={async () => {
                setStatusMsg('Generating 4 evolutionary variants...');
                try {
                  const res = await fetch('/api/explorer', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({
                      action: 'generate_variants',
                      loraPath,
                      prompt,
                      seed,
                      strength,
                    }),
                  });
                  const data = await res.json();
                  if (data.success) {
                    setStatusMsg('Variants generated. Click a tile to select or evolve.');
                  }
                } catch (e: any) {
                  setStatusMsg('Generation error: ' + e?.message);
                }
              }}
            >
              Start
            </button>
          </div>
        </div>
      </div>

      {/* ── Baseline Card (lines 15454-15523) ── */}
      <div className="panel source-card">
        <h2>Current Baseline</h2>
        <p className="muted">
          Your current best. Pick a favourite below to evolve it, or save it as a LoRA.
        </p>
        <div style={{ display: 'flex', gap: 16, marginTop: 8, flexWrap: 'wrap' }}>
          {/* Baseline image holder */}
          <div
            style={{
              width: 380,
              height: 380,
              background: '#000000',
              border: '1px solid #3a4555',
              display: 'flex',
              alignItems: 'center',
              justifyContent: 'center',
            }}
          >
            <span style={{ color: '#5a6b7e', fontSize: 13 }}>(baseline rendered)</span>
          </div>

          {/* Right column: buttons + state dump */}
          <div style={{ flex: 1, minWidth: 280, display: 'flex', flexDirection: 'column', gap: 8 }}>
            <div style={{ display: 'flex', gap: 6 }}>
              <button
                type="button"
                className="secondary"
                style={{ flex: 2 }}
                onClick={async () => {
                  const outName = window.prompt('Save LoRA as:', loraPath ? `${loraPath.replace('.safetensors', '_explored.safetensors')}` : 'output_loras/explored.safetensors');
                  if (!outName) return;
                  const res = await fetch('/api/explorer', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ action: 'export_lora', loraPath, outputPath: outName }),
                  });
                  const data = await res.json();
                  if (data.success) alert(`Saved to ${data.savedTo}`);
                }}
              >
                Save Baseline as LoRA...
              </button>
              <button type="button" className="secondary" style={{ flex: 1 }}>
                Undo
              </button>
              <button type="button" className="secondary" style={{ flex: 1 }}>
                Restart
              </button>
            </div>
            <div style={{ display: 'flex', gap: 6 }}>
              <button
                className="secondary"
                style={{ flex: 1, fontWeight: 'bold' }}
                disabled
              >
                Freeze tweaked blocks
              </button>
              <button
                className="primary"
                style={{
                  flex: 1,
                  background: '#2E8B57',
                  borderColor: '#2E8B57',
                  fontWeight: 'bold',
                }}
                disabled
              >
                Refine this baseline in Repair Studio →
              </button>
            </div>
            <pre
              className="output-log"
              style={{
                flex: 1,
                minHeight: 220,
                fontSize: 11,
                fontFamily: 'Consolas, monospace',
                margin: 0,
              }}
            >
              {'// Block state snapshot will appear here during exploration'}
            </pre>
          </div>
        </div>
      </div>

      {/* ── Variants Card (lines 15525-15568) ── */}
      <div className="panel source-card">
        <h2>Variants</h2>
        <p className="muted">
          4 random mutations of the current baseline. Click your favourite to evolve.
        </p>
        <div style={{ display: 'flex', gap: 10, alignItems: 'center', marginBottom: 12 }}>
          <button className="secondary" disabled>
            Re-roll
          </button>
          <span style={{ color: '#60A5FA', fontWeight: 'bold', fontSize: 13 }}></span>
        </div>

        {/* 2x2 grid of variants */}
        <div
          style={{
            display: 'grid',
            gridTemplateColumns: '1fr 1fr',
            gap: 12,
          }}
        >
          {[0, 1, 2, 3].map((idx) => (
            <div
              key={idx}
              onClick={() => setActiveVariant(idx)}
              style={{
                aspectRatio: '1',
                maxHeight: 340,
                background: '#000000',
                border: activeVariant === idx ? '2px solid #3b82f6' : '1px solid #3a4555',
                display: 'flex',
                alignItems: 'center',
                justifyContent: 'center',
                position: 'relative',
                cursor: 'pointer',
              }}
            >
              <span style={{ color: '#5a6b7e', fontSize: 12 }}>(variant {idx + 1})</span>
              <button
                className="secondary"
                style={{
                  position: 'absolute',
                  top: 4,
                  right: 4,
                  padding: '2px 6px',
                  minHeight: 22,
                  fontSize: 11,
                  background: '#1a2028',
                }}
                onClick={(e) => {
                  e.stopPropagation();
                  setSeed(String(Math.floor(Math.random() * 1000000)));
                }}
              >
                ↻
              </button>
            </div>
          ))}
        </div>
      </div>
    </div>
  );
}
