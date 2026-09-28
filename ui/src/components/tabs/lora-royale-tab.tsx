'use client';

import React, { useState } from 'react';

const SEED_TRAVEL_PRESETS = [
  'Full journey (4 seeds, long)',
  'A to B morph (2 seeds)',
  'Anchor to reference',
  'Identity lock (high anchor)',
  'Creative wander (low anchor)',
];

const PROMPT_TRAVEL_PRESETS = [
  'Age',
  'Era',
  'Lighting',
  'Time of day',
  'Emotion',
  'Weather',
  'Custom words',
];

const SUBJECT_LABELS = ['Woman', 'Man', 'Girl', 'Boy', 'Person', 'Dog', 'Cat', 'Subject'];

const SIZES = ['384', '512', '768', '1024', '1280', '1536', '2048'];

export default function LoraRoyaleTab() {
  const [royaleFamily, setRoyaleFamily] = useState<'klein' | 'krea2' | 'minimax'>('klein');
  const [mode, setMode] = useState<'folder' | 'single'>('folder');

  // Setup fields
  const [folderPath, setFolderPath] = useState('');
  const [singlePath, setSinglePath] = useState('');
  const [prompt, setPrompt] = useState('');
  const [seed, setSeed] = useState('42');
  const [refPath, setRefPath] = useState('');
  const [refStrength, setRefStrength] = useState('1.0');
  const [statusMsg, setStatusMsg] = useState('Pick a folder, set a prompt, then render.');

  // Crossfade card
  const [cfW, setCfW] = useState('512');
  const [cfH, setCfH] = useState('512');
  const [cfMax, setCfMax] = useState('12');
  const [scrubValue, setScrubValue] = useState(0);
  const [scrubLabel, setScrubLabel] = useState('');

  // Export morph card
  const [exportFormat, setExportFormat] = useState('MP4');
  const [exportSpeed, setExportSpeed] = useState('Normal');
  const [exportLoop, setExportLoop] = useState(true);
  const [exportEpoch, setExportEpoch] = useState(true);
  const [exportWm, setExportWm] = useState(true);

  // Seed travel card
  const [travelPreset, setTravelPreset] = useState('');
  const [travelSeedA, setTravelSeedA] = useState('42');
  const [travelSeedB, setTravelSeedB] = useState('4242');
  const [travelWaypoints, setTravelWaypoints] = useState('2');
  const [travelFrames, setTravelFrames] = useState('24');
  const [travelRef, setTravelRef] = useState('');
  const [travelUseEpochRef, setTravelUseEpochRef] = useState(false);
  const [travelRefStrength, setTravelRefStrength] = useState('0.25');
  const [travelSeqRef, setTravelSeqRef] = useState(false);
  const [travelRefMp, setTravelRefMp] = useState('0.2');
  const [travelSpeed, setTravelSpeed] = useState('Normal');
  const [travelW, setTravelW] = useState('512');
  const [travelH, setTravelH] = useState('512');
  const [travelLoop, setTravelLoop] = useState(true);
  const [travelEpochBadge, setTravelEpochBadge] = useState(true);
  const [travelWmTag, setTravelWmTag] = useState(true);
  const [travelDeflicker, setTravelDeflicker] = useState('None');

  // Prompt travel card
  const [ptPreset, setPtPreset] = useState('');
  const [ptSubject, setPtSubject] = useState('Woman');
  const [ptPrompt, setPtPrompt] = useState('');
  const [ptCustom, setPtCustom] = useState('');
  const [ptStart, setPtStart] = useState('');
  const [ptEnd, setPtEnd] = useState('');
  const [ptFrames, setPtFrames] = useState('32');
  const [ptRef, setPtRef] = useState('');
  const [ptUseEpochRef, setPtUseEpochRef] = useState(false);
  const [ptRefStrength, setPtRefStrength] = useState('1.0');
  const [ptSeqRef, setPtSeqRef] = useState(false);
  const [ptRefMp, setPtRefMp] = useState('0.2');
  const [ptAnchor, setPtAnchor] = useState(true);
  const [ptAnchorStr, setPtAnchorStr] = useState('1.0');
  const [ptSpeed, setPtSpeed] = useState('Normal');
  const [ptW, setPtW] = useState('512');
  const [ptH, setPtH] = useState('512');
  const [ptInterp, setPtInterp] = useState('Linear');
  const [ptDrift, setPtDrift] = useState('0.0');
  const [ptLoop, setPtLoop] = useState(true);
  const [ptWordBadge, setPtWordBadge] = useState(true);
  const [ptWmTag, setPtWmTag] = useState(true);
  const [ptDeflicker, setPtDeflicker] = useState('None');
  const [ptVarySeed, setPtVarySeed] = useState(false);

  // LoRA strength travel card
  const [loraStart, setLoraStart] = useState('0.0');
  const [loraEnd, setLoraEnd] = useState('1.0');
  const [loraFrames, setLoraFrames] = useState('24');
  const [loraSpeed, setLoraSpeed] = useState('Normal');
  const [loraW, setLoraW] = useState('512');
  const [loraH, setLoraH] = useState('512');
  const [loraLoop, setLoraLoop] = useState(true);
  const [loraBadge, setLoraBadge] = useState(true);
  const [loraWm, setLoraWm] = useState(true);
  const [loraDeflicker, setLoraDeflicker] = useState('None');

  // Comparison sheet card
  const [cmpPrompts, setCmpPrompts] = useState('');
  const [cmpMode, setCmpMode] = useState('Without / with LoRA');
  const [cmpEpochs, setCmpEpochs] = useState('');
  const [cmpTrigger, setCmpTrigger] = useState('');
  const [cmpSeed, setCmpSeed] = useState('42');
  const [cmpW, setCmpW] = useState('512');
  const [cmpH, setCmpH] = useState('512');
  const [cmpRowLabels, setCmpRowLabels] = useState(true);
  const [cmpBrand, setCmpBrand] = useState(true);

  // Likeness card
  const [likeRef, setLikeRef] = useState('');

  const isKlein = royaleFamily === 'klein';
  const isSingle = mode === 'single';

  return (
    <div className="fizgig-tab-page">
      {/* ── Banner (lines 21104-21105) ── */}
      <div className="tab-banner">
        <h1>LoRA Royale</h1>
        <p>
          Render every epoch of a training run on one seed, then crossfade between them to find the sweet spot.
        </p>
      </div>

      {/* ── Model Family Card (lines 21115-21128) ── */}
      <div className="panel source-card">
        <h2>Model Family</h2>
        <p className="muted">
          Klein 9B (Distilled previews), Krea 2 (fp8 Turbo previews) or MiniMax H3 (22-frame clip previews, middle frame shown — slower per epoch). Epoch comparison works for all three; the travel modes are Klein and Krea 2.
        </p>
        <div style={{ display: 'flex', gap: 24 }}>
          <label className="radio-option">
            <input
              type="radio"
              name="royale_family"
              value="klein"
              checked={royaleFamily === 'klein'}
              onChange={() => setRoyaleFamily('klein')}
            />
            <span>Klein 9B</span>
          </label>
          <label className="radio-option">
            <input
              type="radio"
              name="royale_family"
              value="krea2"
              checked={royaleFamily === 'krea2'}
              onChange={() => setRoyaleFamily('krea2')}
            />
            <span>Krea 2</span>
          </label>
          <label className="radio-option">
            <input
              type="radio"
              name="royale_family"
              value="minimax"
              checked={royaleFamily === 'minimax'}
              onChange={() => setRoyaleFamily('minimax')}
            />
            <span>MiniMax H3</span>
          </label>
        </div>
      </div>

      {/* ── Setup Card (lines 21129-21221) ── */}
      <div className="panel source-card">
        <h2>Setup</h2>
        <p className="muted">
          Point at a training output folder. Renders use the Distilled 4-step model.
        </p>

        {/* Source Radio Buttons */}
        <div style={{ display: 'flex', gap: 20, marginBottom: 8, alignItems: 'center' }}>
          <span style={{ color: '#8a9bae', fontSize: 13 }}>Source:</span>
          <label className="radio-option">
            <input
              type="radio"
              name="royale_mode"
              value="folder"
              checked={mode === 'folder'}
              onChange={() => setMode('folder')}
            />
            <span>Training folder (compare epochs)</span>
          </label>
          <label className="radio-option">
            <input
              type="radio"
              name="royale_mode"
              value="single"
              checked={mode === 'single'}
              onChange={() => setMode('single')}
            />
            <span>Single LoRA</span>
          </label>
        </div>

        {/* Folder or Single LoRA picker */}
        {mode === 'folder' ? (
          <div className="field" style={{ marginBottom: 8 }}>
            <span>Checkpoint folder:</span>
            <div style={{ display: 'flex', gap: 8, alignItems: 'center' }}>
              <input
                type="text"
                value={folderPath}
                onChange={(e) => setFolderPath(e.target.value)}
                style={{ flex: 1 }}
              />
              <button className="secondary" style={{ minHeight: 32 }}>
                Browse…
              </button>
            </div>
          </div>
        ) : (
          <div className="field" style={{ marginBottom: 8 }}>
            <span>LoRA file:</span>
            <div style={{ display: 'flex', gap: 8, alignItems: 'center' }}>
              <input
                type="text"
                value={singlePath}
                readOnly
                style={{ flex: 1 }}
              />
              <button className="secondary" style={{ minHeight: 32 }}>
                Browse…
              </button>
            </div>
          </div>
        )}

        {/* Prompt */}
        <div className="field" style={{ marginBottom: 8 }}>
          <span>Prompt:</span>
          <input
            type="text"
            value={prompt}
            onChange={(e) => setPrompt(e.target.value)}
          />
        </div>

        {/* Seed */}
        <div style={{ display: 'flex', gap: 10, alignItems: 'center', marginBottom: 8 }}>
          <span style={{ color: '#8a9bae', fontSize: 13, minWidth: 60 }}>Seed:</span>
          <input
            type="text"
            value={seed}
            onChange={(e) => setSeed(e.target.value)}
            style={{ width: 90 }}
          />
          <small style={{ color: '#5a6b7e', fontStyle: 'italic' }}>
            shared by the crossfade, prompt travel and strength travel
          </small>
        </div>

        {/* Reference Image */}
        <div style={{ display: 'flex', gap: 8, alignItems: 'center', marginBottom: 12, flexWrap: 'wrap' }}>
          <span style={{ color: '#8a9bae', fontSize: 13, minWidth: 60 }}>Reference:</span>
          <input
            type="text"
            value={refPath}
            readOnly
            style={{ flex: 1, minWidth: 150 }}
            placeholder="(optional image conditioning)"
          />
          <button className="secondary" style={{ minHeight: 32 }}>
            Browse…
          </button>
          <button className="secondary" style={{ minHeight: 32 }} onClick={() => setRefPath('')}>
            Clear
          </button>
          {isKlein && (
            <>
              <span style={{ color: '#8a9bae', fontSize: 13 }}>Strength</span>
              <input
                type="text"
                value={refStrength}
                onChange={(e) => setRefStrength(e.target.value)}
                style={{ width: 45 }}
              />
            </>
          )}
        </div>

        {/* Render Button + Status */}
        <div style={{ display: 'flex', gap: 12, alignItems: 'center', paddingTop: 6, borderTop: '1px solid #3a4555' }}>
          <button
            type="button"
            className="primary"
            style={{ background: '#2E8B57', borderColor: '#2E8B57', fontWeight: 'bold', fontSize: 14, padding: '8px 24px' }}
            onClick={async () => {
              const target = mode === 'folder' ? folderPath : singlePath;
              if (!target) {
                alert('Please provide a checkpoint folder or LoRA file.');
                return;
              }
              setStatusMsg('Scanning checkpoints and rendering epochs...');
              try {
                const res = await fetch('/api/royale', {
                  method: 'POST',
                  headers: { 'Content-Type': 'application/json' },
                  body: JSON.stringify({
                    action: 'scan_checkpoints',
                    folder: mode === 'folder' ? folderPath : target,
                  }),
                });
                const data = await res.json();
                if (data.success) {
                  setStatusMsg(`Found ${data.count} epoch checkpoints. Rendering comparisons...`);
                  setScrubLabel(`Epoch 1 to ${data.count}`);
                } else {
                  setStatusMsg('Error: ' + (data.error || 'Scan failed'));
                }
              } catch (e: any) {
                setStatusMsg('Error: ' + e?.message);
              }
            }}
          >
            Render epochs
          </button>
          <span style={{ color: '#3b82f6', fontStyle: 'italic', fontSize: 13 }}>
            {statusMsg}
          </span>
        </div>
      </div>

      {/* ── Crossfade Card (Folder mode only — lines 21222-21256) ── */}
      {!isSingle && (
        <div className="panel source-card">
          <h2>Crossfade</h2>
          <p className="muted">
            Drag to blend between consecutive epochs — stop where it looks best.
          </p>

          <div style={{ display: 'flex', gap: 12, alignItems: 'center', flexWrap: 'wrap', marginBottom: 8 }}>
            <span style={{ color: '#8a9bae', fontSize: 12 }}>Size</span>
            <select value={cfW} onChange={(e) => setCfW(e.target.value)} style={{ width: 65 }}>
              {SIZES.map((v) => (
                <option key={v} value={v}>{v}</option>
              ))}
            </select>
            <span style={{ color: '#8a9bae', fontSize: 12 }}>x</span>
            <select value={cfH} onChange={(e) => setCfH(e.target.value)} style={{ width: 65 }}>
              {SIZES.map((v) => (
                <option key={v} value={v}>{v}</option>
              ))}
            </select>
            <span style={{ color: '#8a9bae', fontSize: 12, marginLeft: 8 }}>Max renders</span>
            <select value={cfMax} onChange={(e) => setCfMax(e.target.value)} style={{ width: 70 }}>
              {['All', '6', '8', '10', '12', '16', '20'].map((v) => (
                <option key={v} value={v}>{v}</option>
              ))}
            </select>
            <small style={{ color: '#5a6b7e', fontStyle: 'italic' }}>
              how many epochs to render, newest first
            </small>
          </div>

          <div
            style={{
              width: 512,
              height: 512,
              background: '#1c1c1c',
              border: '1px solid #3a4555',
              display: 'flex',
              alignItems: 'center',
              justifyContent: 'center',
              margin: '8px auto',
            }}
          >
            <span style={{ color: '#5a6b7e', fontSize: 13 }}>(render to begin)</span>
          </div>

          {scrubLabel && (
            <p style={{ textAlign: 'center', fontWeight: 'bold', fontSize: 14, color: '#f0f4f8', margin: '4px 0' }}>
              {scrubLabel}
            </p>
          )}

          <input
            type="range"
            min={0}
            max={100}
            value={scrubValue}
            onChange={(e) => setScrubValue(parseFloat(e.target.value))}
            style={{ width: '100%', accentColor: '#3b82f6', marginTop: 6 }}
          />
        </div>
      )}

      {/* ── Export the morph Card (Folder mode only — lines 21258-21299) ── */}
      {!isSingle && (
        <div className="panel source-card">
          <h2>Export the morph</h2>
          <p className="muted">
            Save the crossfade as a looping clip — face resolving epoch by epoch, with a Fizgig · LoRA Royale tag. Made to share.
          </p>

          <div style={{ display: 'flex', gap: 12, alignItems: 'center', marginBottom: 8 }}>
            <span style={{ color: '#8a9bae', fontSize: 12 }}>Format</span>
            <select value={exportFormat} onChange={(e) => setExportFormat(e.target.value)} style={{ width: 70 }}>
              <option value="MP4">MP4</option>
              <option value="GIF">GIF</option>
            </select>
            <span style={{ color: '#8a9bae', fontSize: 12, marginLeft: 8 }}>Speed</span>
            <select value={exportSpeed} onChange={(e) => setExportSpeed(e.target.value)} style={{ width: 85 }}>
              <option value="Slow">Slow</option>
              <option value="Normal">Normal</option>
              <option value="Fast">Fast</option>
            </select>
          </div>

          <div style={{ display: 'flex', gap: 16, alignItems: 'center', marginBottom: 12 }}>
            <label className="check-line">
              <input
                type="checkbox"
                checked={exportLoop}
                onChange={(e) => setExportLoop(e.target.checked)}
              />
              <span>Loop (ping-pong)</span>
            </label>
            <label className="check-line">
              <input
                type="checkbox"
                checked={exportEpoch}
                onChange={(e) => setExportEpoch(e.target.checked)}
              />
              <span>Epoch ticker</span>
            </label>
            <label className="check-line">
              <input
                type="checkbox"
                checked={exportWm}
                onChange={(e) => setExportWm(e.target.checked)}
              />
              <span>Fizgig tag</span>
            </label>
          </div>

          <div className="actions">
            <button
              className="primary"
              style={{ background: '#8E44AD', borderColor: '#8E44AD', fontWeight: 'bold' }}
            >
              Export clip…
            </button>
            <button
              className="secondary"
              style={{ background: '#34495E', borderColor: '#34495E', color: '#fff' }}
            >
              Save all stills…
            </button>
          </div>
        </div>
      )}

      {/* ── All Epochs Grid Card (Folder mode only — lines 21923-21927) ── */}
      {!isSingle && (
        <div className="panel source-card">
          <h2>All epochs</h2>
          <p className="muted">Click a thumbnail to jump the crossfade there.</p>
          <div
            style={{
              minHeight: 80,
              background: '#18212b',
              border: '1px dashed #3a4555',
              display: 'flex',
              alignItems: 'center',
              justifyContent: 'center',
              padding: 10,
            }}
          >
            <span style={{ color: '#5a6b7e', fontSize: 12 }}>(thumbnails will appear after render)</span>
          </div>
        </div>
      )}

      {/* ── Likeness Score Card (Folder mode only — lines 21928-21962) ── */}
      {!isSingle && (
        <div className="panel source-card">
          <h2>Likeness score</h2>
          <p className="muted">
            Pick a training image of your subject — Fizgig scores each epoch&apos;s face against it (ArcFace, CPU) and highlights the closest match in gold.
          </p>
          <div style={{ display: 'flex', gap: 8, alignItems: 'center', marginBottom: 12 }}>
            <span style={{ color: '#8a9bae', fontSize: 13, minWidth: 100 }}>Subject image:</span>
            <input
              type="text"
              value={likeRef}
              readOnly
              style={{ flex: 1 }}
              placeholder="(select training photo)"
            />
            <button className="secondary" style={{ minHeight: 32 }}>
              Browse…
            </button>
            <button className="secondary" style={{ minHeight: 32 }} onClick={() => setLikeRef('')}>
              Clear
            </button>
          </div>
          <div className="actions">
            <button
              className="primary"
              style={{ background: '#3A6EA5', borderColor: '#3A6EA5', fontWeight: 'bold' }}
            >
              Score likeness
            </button>
            <button className="secondary" disabled>
              Jump to best
            </button>
            <button
              className="secondary"
              style={{ background: '#8E44AD', borderColor: '#8E44AD', color: '#fff' }}
            >
              Export likeness clip…
            </button>
          </div>
        </div>
      )}

      {/* ── Seed Travel Card (lines 21300-21467) ── */}
      <div className="panel source-card">
        <h2>Seed travel</h2>
        <p className="muted">
          Take the epoch on the crossfade and morph it smoothly between two seeds (slerp through noise space) — shows the LoRA&apos;s range, saved as a clip.
        </p>

        <div style={{ display: 'flex', gap: 8, alignItems: 'center', marginBottom: 8 }}>
          <span style={{ color: '#8a9bae', fontSize: 13, minWidth: 80 }}>Preset</span>
          <select
            value={travelPreset}
            onChange={(e) => setTravelPreset(e.target.value)}
            style={{ width: 220 }}
          >
            <option value="">(Select Preset)</option>
            {SEED_TRAVEL_PRESETS.map((p) => (
              <option key={p} value={p}>{p}</option>
            ))}
          </select>
        </div>

        <div style={{ display: 'flex', gap: 12, alignItems: 'center', flexWrap: 'wrap', marginBottom: 4 }}>
          <span style={{ color: '#8a9bae', fontSize: 13, minWidth: 80 }}>Seeds</span>
          <input
            type="text"
            value={travelSeedA}
            onChange={(e) => setTravelSeedA(e.target.value)}
            style={{ width: 80 }}
          />
          <span style={{ color: '#5a6b7e' }}>→</span>
          <input
            type="text"
            value={travelSeedB}
            onChange={(e) => setTravelSeedB(e.target.value)}
            style={{ width: 80 }}
          />
          <button
            className="secondary"
            style={{ minHeight: 28, padding: '2px 8px' }}
            onClick={() => {
              setTravelSeedA(String(Math.floor(Math.random() * 1000000)));
              setTravelSeedB(String(Math.floor(Math.random() * 1000000)));
            }}
          >
            🎲
          </button>
          <span style={{ color: '#8a9bae', fontSize: 13, marginLeft: 8 }}>Waypoints</span>
          <select
            value={travelWaypoints}
            onChange={(e) => setTravelWaypoints(e.target.value)}
            style={{ width: 50 }}
          >
            {['2', '3', '4', '5', '6', '8'].map((v) => (
              <option key={v} value={v}>{v}</option>
            ))}
          </select>
          <span style={{ color: '#8a9bae', fontSize: 13, marginLeft: 8 }}>Frames</span>
          <select
            value={travelFrames}
            onChange={(e) => setTravelFrames(e.target.value)}
            style={{ width: 60 }}
          >
            {['16', '24', '36', '48', '64', '96', '128', '192', '256'].map((v) => (
              <option key={v} value={v}>{v}</option>
            ))}
          </select>
        </div>

        <p className="explanation">
          Waypoints = seeds in the journey; 🎲 rerolls Start/End. With a reference holding the subject, more waypoints = a longer tour through compositions.
        </p>

        {/* Reference Image row for Seed Travel */}
        <div style={{ display: 'flex', gap: 8, alignItems: 'center', marginBottom: 6, flexWrap: 'wrap' }}>
          <span style={{ color: '#8a9bae', fontSize: 13, minWidth: 80 }}>Reference</span>
          <input
            type="text"
            value={travelRef}
            readOnly
            style={{ flex: 1, minWidth: 150 }}
            placeholder="(optional anchor photo)"
          />
          <button className="secondary" style={{ minHeight: 32 }}>
            Browse…
          </button>
          <button className="secondary" style={{ minHeight: 32 }} onClick={() => setTravelRef('')}>
            Clear
          </button>
        </div>

        <div style={{ display: 'flex', gap: 14, alignItems: 'center', flexWrap: 'wrap', marginBottom: 4 }}>
          <label className="check-line">
            <input
              type="checkbox"
              checked={travelUseEpochRef}
              onChange={(e) => setTravelUseEpochRef(e.target.checked)}
            />
            <span>Use the rendered epoch as the reference</span>
          </label>
          <span style={{ color: '#8a9bae', fontSize: 12 }}>Strength</span>
          <input
            type="text"
            value={travelRefStrength}
            onChange={(e) => setTravelRefStrength(e.target.value)}
            style={{ width: 45 }}
          />
          <label className="check-line">
            <input
              type="checkbox"
              checked={travelSeqRef}
              onChange={(e) => setTravelSeqRef(e.target.checked)}
            />
            <span>Sequential reference</span>
          </label>
          <span style={{ color: '#8a9bae', fontSize: 12 }}>Max MP</span>
          <input
            type="text"
            value={travelRefMp}
            onChange={(e) => setTravelRefMp(e.target.value)}
            style={{ width: 45 }}
          />
        </div>

        <div style={{ display: 'flex', gap: 12, alignItems: 'center', flexWrap: 'wrap', marginBottom: 6 }}>
          <span style={{ color: '#8a9bae', fontSize: 12 }}>Speed</span>
          <select value={travelSpeed} onChange={(e) => setTravelSpeed(e.target.value)} style={{ width: 85 }}>
            <option value="Slow">Slow</option>
            <option value="Normal">Normal</option>
            <option value="Fast">Fast</option>
          </select>
          <span style={{ color: '#8a9bae', fontSize: 12 }}>W</span>
          <select value={travelW} onChange={(e) => setTravelW(e.target.value)} style={{ width: 65 }}>
            {SIZES.map((v) => (
              <option key={v} value={v}>{v}</option>
            ))}
          </select>
          <span style={{ color: '#8a9bae', fontSize: 12 }}>H</span>
          <select value={travelH} onChange={(e) => setTravelH(e.target.value)} style={{ width: 65 }}>
            {SIZES.map((v) => (
              <option key={v} value={v}>{v}</option>
            ))}
          </select>
          <span style={{ color: '#8a9bae', fontSize: 12, marginLeft: 8 }}>Deflicker</span>
          <select value={travelDeflicker} onChange={(e) => setTravelDeflicker(e.target.value)} style={{ width: 90 }}>
            <option value="None">None</option>
            <option value="Normal">Normal</option>
            <option value="Strong">Strong</option>
          </select>
        </div>

        <div style={{ display: 'flex', gap: 16, alignItems: 'center', marginBottom: 12 }}>
          <label className="check-line">
            <input
              type="checkbox"
              checked={travelLoop}
              onChange={(e) => setTravelLoop(e.target.checked)}
            />
            <span>Loop (ping-pong)</span>
          </label>
          <label className="check-line">
            <input
              type="checkbox"
              checked={travelEpochBadge}
              onChange={(e) => setTravelEpochBadge(e.target.checked)}
            />
            <span>Epoch badge</span>
          </label>
          <label className="check-line">
            <input
              type="checkbox"
              checked={travelWmTag}
              onChange={(e) => setTravelWmTag(e.target.checked)}
            />
            <span>Fizgig tag</span>
          </label>
        </div>

        <div className="actions">
          <button
            className="primary"
            style={{ background: '#C0392B', borderColor: '#C0392B', fontWeight: 'bold' }}
          >
            Render seed-travel…
          </button>
        </div>
      </div>

      {/* ── Prompt Travel Card (lines 21471-21741) ── */}
      <div className="panel source-card">
        <h2>Prompt travel</h2>
        <p className="muted">
          Morph the parked epoch through a series of prompt variations on a fixed seed — it interpolates the text embedding, so the same subject flows (e.g. dawn → night). Saved as a clip.
        </p>

        <div style={{ display: 'flex', gap: 12, alignItems: 'center', marginBottom: 8, flexWrap: 'wrap' }}>
          <span style={{ color: '#8a9bae', fontSize: 13, minWidth: 90 }}>Preset</span>
          <select value={ptPreset} onChange={(e) => setPtPreset(e.target.value)} style={{ width: 180 }}>
            <option value="">(Select Preset)</option>
            {PROMPT_TRAVEL_PRESETS.map((p) => (
              <option key={p} value={p}>{p}</option>
            ))}
          </select>
          <span style={{ color: '#8a9bae', fontSize: 13, marginLeft: 8 }}>Subject</span>
          <select value={ptSubject} onChange={(e) => setPtSubject(e.target.value)} style={{ width: 100 }}>
            {SUBJECT_LABELS.map((s) => (
              <option key={s} value={s}>{s}</option>
            ))}
          </select>
        </div>

        <div className="field" style={{ marginBottom: 4 }}>
          <span>Prompt</span>
          <div style={{ display: 'flex', gap: 8 }}>
            <input
              type="text"
              value={ptPrompt}
              onChange={(e) => setPtPrompt(e.target.value)}
              style={{ flex: 1 }}
            />
            <button
              className="secondary"
              style={{ minHeight: 32 }}
              onClick={() => setPtPrompt((prev) => `${prev} {x}`)}
            >
              Insert &#123;x&#125;
            </button>
          </div>
        </div>
        <p className="explanation">
          Type &#123;x&#125; where the travel word goes — e.g. a portrait of sks man, &#123;x&#125; light. No &#123;x&#125;? the word is appended to the end.
        </p>

        <div className="field" style={{ marginBottom: 4 }}>
          <span>Custom words</span>
          <input
            type="text"
            value={ptCustom}
            onChange={(e) => setPtCustom(e.target.value)}
            placeholder="dawn, noon, sunset, night"
          />
        </div>
        <p className="explanation">Comma-separated — only used when Preset = Custom words.</p>

        <div style={{ display: 'flex', gap: 12, alignItems: 'center', flexWrap: 'wrap', marginBottom: 6 }}>
          <span style={{ color: '#8a9bae', fontSize: 13, minWidth: 90 }}>Travel</span>
          <input
            type="text"
            value={ptStart}
            onChange={(e) => setPtStart(e.target.value)}
            placeholder="Start word"
            style={{ width: 120 }}
          />
          <span style={{ color: '#5a6b7e' }}>→</span>
          <input
            type="text"
            value={ptEnd}
            onChange={(e) => setPtEnd(e.target.value)}
            placeholder="End word"
            style={{ width: 120 }}
          />
          <span style={{ color: '#8a9bae', fontSize: 13, marginLeft: 8 }}>Frames</span>
          <select value={ptFrames} onChange={(e) => setPtFrames(e.target.value)} style={{ width: 60 }}>
            {['24', '32', '48', '64', '96', '128', '192', '256'].map((v) => (
              <option key={v} value={v}>{v}</option>
            ))}
          </select>
        </div>

        {/* Reference row for Prompt Travel */}
        <div style={{ display: 'flex', gap: 8, alignItems: 'center', marginBottom: 6, flexWrap: 'wrap' }}>
          <span style={{ color: '#8a9bae', fontSize: 13, minWidth: 90 }}>Reference</span>
          <input
            type="text"
            value={ptRef}
            readOnly
            style={{ flex: 1, minWidth: 150 }}
            placeholder="(optional anchor photo)"
          />
          <button className="secondary" style={{ minHeight: 32 }}>
            Browse…
          </button>
          <button className="secondary" style={{ minHeight: 32 }} onClick={() => setPtRef('')}>
            Clear
          </button>
        </div>

        <div style={{ display: 'flex', gap: 14, alignItems: 'center', flexWrap: 'wrap', marginBottom: 4 }}>
          <label className="check-line">
            <input
              type="checkbox"
              checked={ptUseEpochRef}
              onChange={(e) => setPtUseEpochRef(e.target.checked)}
            />
            <span>Use the rendered epoch as the reference</span>
          </label>
          <span style={{ color: '#8a9bae', fontSize: 12 }}>Strength</span>
          <input
            type="text"
            value={ptRefStrength}
            onChange={(e) => setPtRefStrength(e.target.value)}
            style={{ width: 45 }}
          />
          <label className="check-line">
            <input
              type="checkbox"
              checked={ptSeqRef}
              onChange={(e) => setPtSeqRef(e.target.checked)}
            />
            <span>Sequential reference</span>
          </label>
          <span style={{ color: '#8a9bae', fontSize: 12 }}>Max MP</span>
          <input
            type="text"
            value={ptRefMp}
            onChange={(e) => setPtRefMp(e.target.value)}
            style={{ width: 45 }}
          />
        </div>

        <div style={{ display: 'flex', gap: 12, alignItems: 'center', marginBottom: 6 }}>
          <label className="check-line">
            <input
              type="checkbox"
              checked={ptAnchor}
              onChange={(e) => setPtAnchor(e.target.checked)}
            />
            <span>Anchor to original</span>
          </label>
          <span style={{ color: '#8a9bae', fontSize: 12 }}>Anchor str</span>
          <input
            type="text"
            value={ptAnchorStr}
            onChange={(e) => setPtAnchorStr(e.target.value)}
            style={{ width: 45 }}
          />
        </div>

        <div style={{ display: 'flex', gap: 12, alignItems: 'center', flexWrap: 'wrap', marginBottom: 6 }}>
          <span style={{ color: '#8a9bae', fontSize: 12 }}>Speed</span>
          <select value={ptSpeed} onChange={(e) => setPtSpeed(e.target.value)} style={{ width: 85 }}>
            <option value="Slow">Slow</option>
            <option value="Normal">Normal</option>
            <option value="Fast">Fast</option>
          </select>
          <span style={{ color: '#8a9bae', fontSize: 12 }}>W</span>
          <select value={ptW} onChange={(e) => setPtW(e.target.value)} style={{ width: 65 }}>
            {SIZES.map((v) => (
              <option key={v} value={v}>{v}</option>
            ))}
          </select>
          <span style={{ color: '#8a9bae', fontSize: 12 }}>H</span>
          <select value={ptH} onChange={(e) => setPtH(e.target.value)} style={{ width: 65 }}>
            {SIZES.map((v) => (
              <option key={v} value={v}>{v}</option>
            ))}
          </select>
          <span style={{ color: '#8a9bae', fontSize: 12 }}>Interpolation</span>
          <select value={ptInterp} onChange={(e) => setPtInterp(e.target.value)} style={{ width: 120 }}>
            <option value="Linear">Linear</option>
            <option value="Norm-preserved">Norm-preserved</option>
            <option value="Slerp">Slerp</option>
          </select>
          <span style={{ color: '#8a9bae', fontSize: 12 }}>Seed drift</span>
          <input
            type="text"
            value={ptDrift}
            onChange={(e) => setPtDrift(e.target.value)}
            style={{ width: 45 }}
          />
        </div>

        <div style={{ display: 'flex', gap: 14, alignItems: 'center', flexWrap: 'wrap', marginBottom: 12 }}>
          <label className="check-line">
            <input
              type="checkbox"
              checked={ptLoop}
              onChange={(e) => setPtLoop(e.target.checked)}
            />
            <span>Loop (ping-pong)</span>
          </label>
          <label className="check-line">
            <input
              type="checkbox"
              checked={ptWordBadge}
              onChange={(e) => setPtWordBadge(e.target.checked)}
            />
            <span>Word badge</span>
          </label>
          <label className="check-line">
            <input
              type="checkbox"
              checked={ptWmTag}
              onChange={(e) => setPtWmTag(e.target.checked)}
            />
            <span>Fizgig tag</span>
          </label>
          <span style={{ color: '#8a9bae', fontSize: 12 }}>Deflicker</span>
          <select value={ptDeflicker} onChange={(e) => setPtDeflicker(e.target.value)} style={{ width: 90 }}>
            <option value="None">None</option>
            <option value="Normal">Normal</option>
            <option value="Strong">Strong</option>
          </select>
          <label className="check-line">
            <input
              type="checkbox"
              checked={ptVarySeed}
              onChange={(e) => setPtVarySeed(e.target.checked)}
            />
            <span>Vary seed</span>
          </label>
        </div>

        <div className="actions">
          <button
            className="primary"
            style={{ background: '#B7791F', borderColor: '#B7791F', fontWeight: 'bold' }}
          >
            Render prompt-travel…
          </button>
        </div>
      </div>

      {/* ── LoRA Strength Travel Card (lines 21743-21823) ── */}
      <div className="panel source-card">
        <h2>LoRA strength travel</h2>
        <p className="muted">
          Hold the prompt and seed fixed and ramp the LoRA&apos;s strength from one value to another — watch the effect fade in (0 = base model) through to full strength and beyond. Saved as a clip.
        </p>

        <div style={{ display: 'flex', gap: 12, alignItems: 'center', flexWrap: 'wrap', marginBottom: 4 }}>
          <span style={{ color: '#8a9bae', fontSize: 13, minWidth: 80 }}>Strength</span>
          <input
            type="text"
            value={loraStart}
            onChange={(e) => setLoraStart(e.target.value)}
            style={{ width: 50 }}
          />
          <span style={{ color: '#5a6b7e' }}>→</span>
          <input
            type="text"
            value={loraEnd}
            onChange={(e) => setLoraEnd(e.target.value)}
            style={{ width: 50 }}
          />
          <span style={{ color: '#8a9bae', fontSize: 13, marginLeft: 12 }}>Frames</span>
          <select value={loraFrames} onChange={(e) => setLoraFrames(e.target.value)} style={{ width: 60 }}>
            {['16', '24', '36', '48', '64', '96', '128', '192', '256'].map((v) => (
              <option key={v} value={v}>{v}</option>
            ))}
          </select>
        </div>

        <p className="explanation">
          0 = base model (no LoRA); 1.0 = trained strength; &gt;1 over-drives it. Uses the Setup prompt and seed, fixed — only the LoRA strength changes.
        </p>

        <div style={{ display: 'flex', gap: 12, alignItems: 'center', flexWrap: 'wrap', marginBottom: 6 }}>
          <span style={{ color: '#8a9bae', fontSize: 12 }}>Speed</span>
          <select value={loraSpeed} onChange={(e) => setLoraSpeed(e.target.value)} style={{ width: 85 }}>
            <option value="Slow">Slow</option>
            <option value="Normal">Normal</option>
            <option value="Fast">Fast</option>
          </select>
          <span style={{ color: '#8a9bae', fontSize: 12 }}>W</span>
          <select value={loraW} onChange={(e) => setLoraW(e.target.value)} style={{ width: 65 }}>
            {SIZES.map((v) => (
              <option key={v} value={v}>{v}</option>
            ))}
          </select>
          <span style={{ color: '#8a9bae', fontSize: 12 }}>H</span>
          <select value={loraH} onChange={(e) => setLoraH(e.target.value)} style={{ width: 65 }}>
            {SIZES.map((v) => (
              <option key={v} value={v}>{v}</option>
            ))}
          </select>
        </div>

        <div style={{ display: 'flex', gap: 16, alignItems: 'center', marginBottom: 12 }}>
          <label className="check-line">
            <input
              type="checkbox"
              checked={loraLoop}
              onChange={(e) => setLoraLoop(e.target.checked)}
            />
            <span>Loop (ping-pong)</span>
          </label>
          <label className="check-line">
            <input
              type="checkbox"
              checked={loraBadge}
              onChange={(e) => setLoraBadge(e.target.checked)}
            />
            <span>Strength badge</span>
          </label>
          <label className="check-line">
            <input
              type="checkbox"
              checked={loraWm}
              onChange={(e) => setLoraWm(e.target.checked)}
            />
            <span>Fizgig tag</span>
          </label>
          <span style={{ color: '#8a9bae', fontSize: 12 }}>Deflicker</span>
          <select value={loraDeflicker} onChange={(e) => setLoraDeflicker(e.target.value)} style={{ width: 90 }}>
            <option value="None">None</option>
            <option value="Normal">Normal</option>
            <option value="Strong">Strong</option>
          </select>
        </div>

        <div className="actions">
          <button
            className="primary"
            style={{ background: '#B7791F', borderColor: '#B7791F', fontWeight: 'bold' }}
          >
            Render strength-travel…
          </button>
        </div>
      </div>

      {/* ── Comparison Sheet Card (lines 21825-21922) ── */}
      <div className="panel source-card">
        <h2>Comparison sheet</h2>
        <p className="muted">
          The share image people actually post for a new LoRA: one row per prompt, one column per condition, same seed across a row so only the LoRA changes. Saved as a single labelled PNG.
        </p>

        <div style={{ background: '#18212b', padding: 10, borderRadius: 4, marginBottom: 8 }}>
          <span style={{ color: '#3b82f6', fontWeight: 'bold', fontSize: 12, display: 'block', marginBottom: 4 }}>
            How to use it
          </span>
          <pre
            style={{
              fontSize: 11,
              color: '#8a9bae',
              lineHeight: 1.5,
              whiteSpace: 'pre-wrap',
              margin: 0,
            }}
          >
{`Without / with LoRA  —  two columns, showing what your LoRA adds.
    1. Load a LoRA:  Single-LoRA mode (pick the file), or Folder mode → Render,
        then slide the crossfade to the epoch you want to show off.
    2. Type your prompts below, one per line.
    3. Fill in Trigger so the no-LoRA column can drop it (the base model has
        never seen that word — leaving it in makes the comparison unfair).
    4. Render.

Every epoch  —  one column per epoch, showing the LoRA learning.
    1. Folder mode → pick your training output folder → Scan → Render.
        (This is the main render at the top of the tab, not this card — the
        sheet reuses those epochs, so it must happen first.)
    2. Set Epochs to keep it readable: blank = all, "every 4", or "4,8,12".
    3. Type your prompts, then Render.`}
          </pre>
        </div>

        <div className="field" style={{ marginBottom: 6 }}>
          <span>Prompts (one per line — each becomes a row):</span>
          <textarea
            className="prompt-editor"
            style={{ minHeight: 70 }}
            value={cmpPrompts}
            onChange={(e) => setCmpPrompts(e.target.value)}
          />
        </div>

        <div style={{ display: 'flex', gap: 12, alignItems: 'center', flexWrap: 'wrap', marginBottom: 6 }}>
          <span style={{ color: '#8a9bae', fontSize: 13, minWidth: 70 }}>Columns</span>
          <select value={cmpMode} onChange={(e) => setCmpMode(e.target.value)} style={{ width: 170 }}>
            <option value="Without / with LoRA">Without / with LoRA</option>
            <option value="Every epoch">Every epoch</option>
          </select>
          <span style={{ color: '#8a9bae', fontSize: 13, marginLeft: 8 }}>Epochs</span>
          <input
            type="text"
            value={cmpEpochs}
            onChange={(e) => setCmpEpochs(e.target.value)}
            placeholder="every 4"
            style={{ width: 100 }}
          />
          <span style={{ color: '#8a9bae', fontSize: 13, marginLeft: 8 }}>Trigger</span>
          <input
            type="text"
            value={cmpTrigger}
            onChange={(e) => setCmpTrigger(e.target.value)}
            placeholder="trigger word"
            style={{ width: 110 }}
          />
        </div>

        <div style={{ display: 'flex', gap: 12, alignItems: 'center', flexWrap: 'wrap', marginBottom: 10 }}>
          <span style={{ color: '#8a9bae', fontSize: 13, minWidth: 70 }}>Seed</span>
          <input
            type="text"
            value={cmpSeed}
            onChange={(e) => setCmpSeed(e.target.value)}
            style={{ width: 80 }}
          />
          <span style={{ color: '#8a9bae', fontSize: 12 }}>W</span>
          <select value={cmpW} onChange={(e) => setCmpW(e.target.value)} style={{ width: 65 }}>
            {SIZES.map((v) => (
              <option key={v} value={v}>{v}</option>
            ))}
          </select>
          <span style={{ color: '#8a9bae', fontSize: 12 }}>H</span>
          <select value={cmpH} onChange={(e) => setCmpH(e.target.value)} style={{ width: 65 }}>
            {SIZES.map((v) => (
              <option key={v} value={v}>{v}</option>
            ))}
          </select>
          <label className="check-line" style={{ marginLeft: 8 }}>
            <input
              type="checkbox"
              checked={cmpRowLabels}
              onChange={(e) => setCmpRowLabels(e.target.checked)}
            />
            <span>Row captions</span>
          </label>
          <label className="check-line">
            <input
              type="checkbox"
              checked={cmpBrand}
              onChange={(e) => setCmpBrand(e.target.checked)}
            />
            <span>Fizgig tag</span>
          </label>
        </div>

        <div className="actions">
          <button
            className="primary"
            style={{ background: '#B7791F', borderColor: '#B7791F', fontWeight: 'bold' }}
          >
            Render comparison sheet…
          </button>
        </div>
      </div>

      {/* ── Promote Winner Card (Folder mode only — lines 21963-21975) ── */}
      {!isSingle && (
        <div className="panel source-card">
          <h2>Promote winner</h2>
          <p className="muted">
            Copy the epoch currently shown on the crossfade to a new .safetensors you can drop straight into ComfyUI.
          </p>
          <div className="actions">
            <button
              className="primary"
              style={{ background: '#2E8B57', borderColor: '#2E8B57', fontWeight: 'bold' }}
            >
              Promote current epoch…
            </button>
          </div>
        </div>
      )}
    </div>
  );
}
