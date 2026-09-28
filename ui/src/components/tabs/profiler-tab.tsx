'use client';

import React, { useState } from 'react';

export default function ProfilerTab() {
  const [profilerFamily, setProfilerFamily] = useState<'klein' | 'krea2' | 'minimax'>('klein');
  const [ditChoice, setDitChoice] = useState<'distilled' | 'base'>('distilled');
  const [loraPath, setLoraPath] = useState('');
  const [prompt, setPrompt] = useState('');
  const [resolution, setResolution] = useState('1024');
  const [stages, setStages] = useState('5');
  const [progress, setProgress] = useState('');
  const [results, setResults] = useState('');

  const isKlein = profilerFamily === 'klein';

  return (
    <div className="fizgig-tab-page">
      {/* ── Banner (lines 18740-18746) ── */}
      <div className="tab-banner">
        <h1>Profiler</h1>
        <p>
          Analyze a LoRA&apos;s per-block signature. Klein: full activation profile (5-bucket report). Krea 2 and MiniMax H3: weight-only profile (flat per-block — no block-role map yet). All write a sidecar the Repair Studio reads inline.
        </p>
      </div>

      {/* ── Model Family Card (lines 18754-18767) ── */}
      <div className="panel source-card">
        <h2>Model Family</h2>
        <p className="muted">
          Klein 9B (activation profile), Krea 2 or MiniMax H3 (weight-only — the instrument to discover each family&apos;s block roles). Browsing a LoRA auto-switches to its family.
        </p>
        <div style={{ display: 'flex', gap: 24 }}>
          <label className="radio-option">
            <input
              type="radio"
              name="profiler_family"
              value="klein"
              checked={profilerFamily === 'klein'}
              onChange={() => setProfilerFamily('klein')}
            />
            <span>Klein 9B</span>
          </label>
          <label className="radio-option">
            <input
              type="radio"
              name="profiler_family"
              value="krea2"
              checked={profilerFamily === 'krea2'}
              onChange={() => setProfilerFamily('krea2')}
            />
            <span>Krea 2</span>
          </label>
          <label className="radio-option">
            <input
              type="radio"
              name="profiler_family"
              value="minimax"
              checked={profilerFamily === 'minimax'}
              onChange={() => setProfilerFamily('minimax')}
            />
            <span>MiniMax H3</span>
          </label>
        </div>
      </div>

      {/* ── Model Card (Klein only — lines 18769-18782) ── */}
      {isKlein && (
        <div className="panel source-card">
          <h2>Model</h2>
          <p className="muted">
            Paths are set on the Preferences tab. Distilled is a few seconds per probe and fine for most scans; Base produces the authoritative report but is slower.
          </p>
          <div style={{ display: 'flex', gap: 24 }}>
            <label className="radio-option">
              <input
                type="radio"
                name="profiler_dit"
                value="distilled"
                checked={ditChoice === 'distilled'}
                onChange={() => setDitChoice('distilled')}
              />
              <span>Distilled (fast, ~4-step probes)</span>
            </label>
            <label className="radio-option">
              <input
                type="radio"
                name="profiler_dit"
                value="base"
                checked={ditChoice === 'base'}
                onChange={() => setDitChoice('base')}
              />
              <span>Base (precise)</span>
            </label>
          </div>
        </div>
      )}

      {/* ── LoRA File Card (lines 18784-18794) ── */}
      <div className="panel source-card">
        <h2>LoRA File</h2>
        <p className="muted">
          Select the LoRA you want to profile. PEFT and LyCORIS (LoKR / LoHa) are auto-converted on load.
        </p>
        <div className="start-controls">
          <div className="field" style={{ flex: 1 }}>
            <span>LoRA File:</span>
            <input
              type="text"
              value={loraPath}
              onChange={(e) => setLoraPath(e.target.value)}
              placeholder=""
            />
          </div>
          <button className="secondary" style={{ alignSelf: 'flex-end', marginBottom: 2 }}>
            Browse
          </button>
        </div>
      </div>

      {/* ── Prompt Card (Klein only — lines 18796-18806) ── */}
      {isKlein && (
        <div className="panel source-card">
          <h2>Prompt</h2>
          <p className="muted">
            Include the LoRA&apos;s trigger word so the profile captures its active pathways, e.g.: <code style={{ color: '#8ed5ff' }}>zwxem, a portrait photo of a woman</code>.
          </p>
          <div className="field">
            <span>Prompt:</span>
            <input
              type="text"
              value={prompt}
              onChange={(e) => setPrompt(e.target.value)}
              placeholder=""
            />
          </div>
        </div>
      )}

      {/* ── Options Card (Klein only — lines 18808-18825) ── */}
      {isKlein && (
        <div className="panel source-card">
          <h2>Options</h2>
          <p className="muted">
            Resolution controls the probe render size; Stages is the number of denoising buckets the profiler measures.
          </p>
          <div className="inline-fields">
            <div className="field compact-field">
              <span>Resolution:</span>
              <select value={resolution} onChange={(e) => setResolution(e.target.value)}>
                {['512', '768', '1024'].map((v) => (
                  <option key={v} value={v}>
                    {v}
                  </option>
                ))}
              </select>
            </div>
            <div className="field compact-field">
              <span>Stages:</span>
              <select value={stages} onChange={(e) => setStages(e.target.value)}>
                {['3', '5', '8', '10'].map((v) => (
                  <option key={v} value={v}>
                    {v}
                  </option>
                ))}
              </select>
            </div>
          </div>
        </div>
      )}

      {/* ── Run Card (lines 18827-18840) ── */}
      <div className="panel source-card">
        <h2>Run</h2>
        <div className="actions" style={{ marginTop: 0 }}>
          <button
            type="button"
            className="primary"
            onClick={async () => {
              if (!loraPath) {
                alert('Please enter or browse a LoRA file first.');
                return;
              }
              setProgress('Profiling LoRA in progress...');
              try {
                const res = await fetch('/api/profiler', {
                  method: 'POST',
                  headers: { 'Content-Type': 'application/json' },
                  body: JSON.stringify({
                    action: 'run_profile',
                    loraPath,
                    prompt,
                    resolution,
                    stages,
                  }),
                });
                const data = await res.json();
                if (data.success && data.report) {
                  setProgress(`Profile completed! Sidecar written: ${data.sidecarWritten}`);
                  setResults(JSON.stringify(data.report, null, 2));
                } else {
                  setProgress('Profile error: ' + (data.error || 'Failed'));
                }
              } catch (e: any) {
                setProgress('Profile error: ' + e?.message);
              }
            }}
          >
            Profile LoRA
          </button>
          <button
            type="button"
            className="secondary"
            disabled={!results}
            onClick={() => alert(`Report data:\n\n${results}`)}
          >
            Open Report
          </button>
        </div>
        {progress && (
          <p style={{ marginTop: 10, fontWeight: 'bold', color: '#60A5FA', fontSize: 13 }}>
            {progress}
          </p>
        )}
      </div>

      {/* ── Results Card (lines 18842-18854) ── */}
      <div className="panel source-card">
        <h2>Results</h2>
        <p className="muted">
          Summary text lands here during profiling; the full heat-mapped report opens in your browser via Open Report.
        </p>
        <pre className="output-log" style={{ minHeight: 250, maxHeight: 450 }}>
          {results || ''}
        </pre>
      </div>
    </div>
  );
}
