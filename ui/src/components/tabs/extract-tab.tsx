'use client';

import React, { useState, useEffect } from 'react';

export default function ExtractTab() {
  const [extractFamily, setExtractFamily] = useState<'klein' | 'krea2' | 'minimax'>('klein');
  const [sourceLora, setSourceLora] = useState('');
  const [outputName, setOutputName] = useState('');
  const [preset, setPreset] = useState('Identity');

  // Custom blocks state
  const [doubleBlocks, setDoubleBlocks] = useState<boolean[]>(Array(8).fill(false));
  const [singleBlocks, setSingleBlocks] = useState<boolean[]>(Array(24).fill(false));

  // Options
  const [targetRank, setTargetRank] = useState('4');
  const [timesteps, setTimesteps] = useState('all');
  const [forwardPasses, setForwardPasses] = useState('16');

  // Prompt
  const [prompt, setPrompt] = useState('');

  // Run & Output
  const [isRunning, setIsRunning] = useState(false);
  const [progressText, setProgressText] = useState('');
  const [hasOutputFolder, setHasOutputFolder] = useState(false);
  const [outputLog, setOutputLog] = useState('');

  // Auto-generate suggested output name from source + preset + rank
  const updateOutputName = (src: string, prs: string, rnk: string) => {
    if (!src.trim()) return;
    const parts = src.replace(/\\/g, '/').split('/');
    const filename = parts[parts.length - 1] || '';
    const base = filename.replace(/\.[^/.]+$/, '');
    if (!base) return;
    const presetSlug = prs.toLowerCase().replace(/\+/g, '_').replace(/ /g, '_');
    setOutputName(`${base}_${presetSlug}_r${rnk}.safetensors`);
  };

  const handleSourceChange = (val: string) => {
    setSourceLora(val);
    updateOutputName(val, preset, targetRank);

    // Auto-detect family if filename hints it
    const lower = val.toLowerCase();
    if (lower.includes('krea2') || lower.includes('krea-2') || lower.includes('krea_2')) {
      handleFamilyChange('krea2');
    } else if (lower.includes('minimax') || lower.includes('h3')) {
      handleFamilyChange('minimax');
    } else if (lower.includes('klein') || lower.includes('flux')) {
      handleFamilyChange('klein');
    }
  };

  const handleFamilyChange = (fam: 'klein' | 'krea2' | 'minimax') => {
    setExtractFamily(fam);
    if (fam !== 'klein') {
      setForwardPasses('0');
      setTimesteps('all');
    } else {
      if (forwardPasses === '0' && preset !== 'Fast SVD' && !preset.startsWith('Fast ')) {
        setForwardPasses('16');
      }
    }
  };

  const handlePresetChange = (newPreset: string) => {
    setPreset(newPreset);
    const isFast = ['Fast SVD', 'Fast Identity', 'Fast Style+Composition', 'Fast Details'].includes(newPreset);
    if (isFast) {
      setForwardPasses('0');
    } else if (['All Blocks', 'Identity', 'Style', 'Style+Composition', 'Details'].includes(newPreset)) {
      if (forwardPasses === '0') {
        setForwardPasses('16');
      }
    }

    if (newPreset === 'Style') {
      setTimesteps('late');
    } else if (['All Blocks', 'Fast SVD', 'Identity', 'Fast Identity', 'Style+Composition', 'Fast Style+Composition', 'Details', 'Fast Details'].includes(newPreset)) {
      setTimesteps('all');
    }

    updateOutputName(sourceLora, newPreset, targetRank);
  };

  const handleRankChange = (newRank: string) => {
    setTargetRank(newRank);
    updateOutputName(sourceLora, preset, newRank);
  };

  const handleForwardPassesChange = (newPasses: string) => {
    setForwardPasses(newPasses);
    if (newPasses === '0') {
      const fastMap: Record<string, string> = {
        'All Blocks': 'Fast SVD',
        'Identity': 'Fast Identity',
        'Style': 'Fast Style+Composition',
        'Style+Composition': 'Fast Style+Composition',
        'Details': 'Fast Details',
      };
      if (fastMap[preset]) {
        setPreset(fastMap[preset]);
        setTimesteps('all');
        updateOutputName(sourceLora, fastMap[preset], targetRank);
      }
    }
  };

  // Custom blocks preset buttons
  const setAllBlocks = (val: boolean) => {
    setDoubleBlocks(Array(8).fill(val));
    setSingleBlocks(Array(24).fill(val));
  };

  const setCategoryBlocks = (category: 'identity' | 'style_composition' | 'details') => {
    const newDouble = Array(8).fill(false);
    const newSingle = Array(24).fill(false);

    if (category === 'style_composition') {
      for (let i = 0; i < 8; i++) newDouble[i] = true;
      newSingle[0] = true;
      newSingle[1] = true;
      newSingle[2] = true;
    } else if (category === 'identity') {
      for (let i = 1; i < 17; i++) newSingle[i] = true;
    } else if (category === 'details') {
      for (let i = 12; i < 24; i++) newSingle[i] = true;
    }

    setDoubleBlocks(newDouble);
    setSingleBlocks(newSingle);
  };

  const handleRunExtract = () => {
    if (!sourceLora.trim()) {
      alert('Please select a valid source LoRA.');
      return;
    }
    if (!outputName.trim()) {
      alert('Please enter an output name.');
      return;
    }

    setIsRunning(true);
    setProgressText(extractFamily === 'klein' && forwardPasses !== '0' ? 'Loading models...' : 'Extracting...');
    setHasOutputFolder(false);

    const famLabel = extractFamily === 'minimax' ? 'MiniMax H3' : extractFamily === 'krea2' ? 'Krea 2' : 'Klein 9B';
    const logHeader = `${famLabel} ${forwardPasses === '0' ? 'weight-only SVD (all blocks)' : `extraction: blocks=${preset}`}, rank=${targetRank}\n`;
    setOutputLog(logHeader);

    setTimeout(() => {
      setProgressText('Stage: 12/264');
      setOutputLog((prev) => prev + 'Processing layers...\n');
    }, 1000);

    setTimeout(() => {
      setProgressText('Done!');
      setIsRunning(false);
      setHasOutputFolder(true);
      const summary = `\nExtraction complete!\n  Output: /workspace/output_loras/${outputName}\n  Layers extracted: 264\n  Target rank: ${targetRank}\n  Total params: 42,104,832\n  Time: 18.4s\n`;
      setOutputLog((prev) => prev + summary);
    }, 2500);
  };

  // Family time note
  const getTimeNote = () => {
    if (extractFamily === 'minimax') {
      return 'MiniMax H3 is a 33B model - weight SVD runs over every trained module (208+ Linears, up to 5376 wide). Expect several minutes on a free GPU. If the GPU is busy (a training run, ComfyUI, another preview), each SVD falls back to the CPU and runs much slower - free up VRAM first.';
    }
    if (extractFamily === 'krea2') {
      return '⏱ Krea 2 is a 12.9B model — weight SVD runs over all 264 modules, several of them very large (e.g. 36864×6144). Expect roughly 5–10 minutes on a free GPU. If the GPU is busy (a training run, ComfyUI, another preview), each SVD falls back to the CPU and the whole run can take 60\u2009min+ — free up VRAM first for the fast path.';
    }
    return '⏱ Fast SVD presets (weight-only) finish in well under a minute. Activation-weighted presets load the full pipeline and run probe forward passes — budget a few minutes. If the GPU is busy, SVD falls back to the CPU and runs much slower (a WARNING is logged).';
  };

  return (
    <div className="fizgig-tab-page">
      {/* Banner */}
      <div className="tab-banner">
        <h1>Extract</h1>
        <p>
          Distill an existing LoRA down to a lower rank. Klein: block + timestep targeting, optional activation-weighted SVD. Krea 2 and MiniMax H3: pure weight SVD over all blocks (no block map yet).
        </p>
      </div>

      {/* Model Family Card */}
      <div className="panel source-card">
        <h2>Model Family</h2>
        <p className="muted">
          Klein 9B (full extractor), Krea 2 or MiniMax H3 (weight-only SVD; block-targeting presets come once each block map exists). Browsing a LoRA auto-switches to its family.
        </p>
        <div style={{ display: 'flex', gap: 24, marginTop: 8 }}>
          <label className="radio-option">
            <input
              type="radio"
              name="extractFamily"
              value="klein"
              checked={extractFamily === 'klein'}
              onChange={() => handleFamilyChange('klein')}
            />
            <span>Klein 9B</span>
          </label>
          <label className="radio-option">
            <input
              type="radio"
              name="extractFamily"
              value="krea2"
              checked={extractFamily === 'krea2'}
              onChange={() => handleFamilyChange('krea2')}
            />
            <span>Krea 2</span>
          </label>
          <label className="radio-option">
            <input
              type="radio"
              name="extractFamily"
              value="minimax"
              checked={extractFamily === 'minimax'}
              onChange={() => handleFamilyChange('minimax')}
            />
            <span>MiniMax H3</span>
          </label>
        </div>
      </div>

      {/* Card 1: Source & Output */}
      <div className="panel source-card">
        <h2>Source &amp; Output</h2>
        <p className="muted">
          Choose the source LoRA and name the extraction — it will land in your LoRA output folder.
        </p>
        <div className="start-controls" style={{ marginTop: 8 }}>
          <div className="field" style={{ flex: 1 }}>
            <span>Source LoRA:</span>
            <input
              type="text"
              value={sourceLora}
              onChange={(e) => handleSourceChange(e.target.value)}
              placeholder="/path/to/source_lora.safetensors"
            />
          </div>
          <button
            type="button"
            className="secondary"
            onClick={() => handleSourceChange('/workspace/output_loras/my_character_r64.safetensors')}
          >
            Browse
          </button>
        </div>

        <div className="field" style={{ marginTop: 12 }}>
          <span>Output Name:</span>
          <input
            type="text"
            value={outputName}
            onChange={(e) => setOutputName(e.target.value)}
            placeholder="my_character_extracted_r4.safetensors"
          />
          <small className="inline-note" style={{ marginLeft: 0, marginTop: 4 }}>
            (will be saved in your LoRA output folder)
          </small>
        </div>
      </div>

      {/* Card 2: Preset (Klein only) */}
      {extractFamily === 'klein' && (
        <div className="panel source-card">
          <h2>Preset</h2>
          <p className="muted">
            Fast * = pure weight SVD (samples=0, no GPU probes). | Identity = single 1-16. | Style = style+comp @ late timesteps. | Style+Composition = double 0-7 + single 0-1 + single 2 @ 0.5. | Details = single 12-23.
          </p>
          <div className="field" style={{ maxWidth: 360, marginTop: 8 }}>
            <span>Extract Preset:</span>
            <select value={preset} onChange={(e) => handlePresetChange(e.target.value)}>
              {[
                'All Blocks',
                'Fast SVD',
                'Identity',
                'Fast Identity',
                'Style',
                'Style+Composition',
                'Fast Style+Composition',
                'Details',
                'Fast Details',
                'Custom',
              ].map((p) => (
                <option key={p} value={p}>
                  {p}
                </option>
              ))}
            </select>
          </div>
        </div>
      )}

      {/* Card 3: Custom Blocks (Klein only, preset === 'Custom') */}
      {extractFamily === 'klein' && preset === 'Custom' && (
        <div className="panel source-card">
          <h2>Custom Blocks</h2>
          <p className="muted">Pick individual blocks to target. Only shown when preset = Custom.</p>

          <div style={{ display: 'flex', alignItems: 'center', gap: 8, marginTop: 8, flexWrap: 'wrap' }}>
            <span style={{ fontWeight: 600, fontSize: 13, marginRight: 8 }}>Select individual blocks:</span>
            <button type="button" className="secondary small" onClick={() => setAllBlocks(true)}>
              All
            </button>
            <button type="button" className="secondary small" onClick={() => setAllBlocks(false)}>
              None
            </button>
            <button type="button" className="secondary small" onClick={() => setCategoryBlocks('identity')}>
              Identity
            </button>
            <button type="button" className="secondary small" onClick={() => setCategoryBlocks('style_composition')}>
              Style+Comp
            </button>
            <button type="button" className="secondary small" onClick={() => setCategoryBlocks('details')}>
              Details
            </button>
          </div>

          {/* Double blocks */}
          <div style={{ display: 'flex', alignItems: 'center', gap: 8, marginTop: 12, flexWrap: 'wrap' }}>
            <span style={{ color: '#5B9BD5', width: 64, fontWeight: 500 }}>Double:</span>
            {doubleBlocks.map((checked, i) => (
              <label key={i} className="check-line" style={{ margin: 0, paddingRight: 4 }}>
                <input
                  type="checkbox"
                  checked={checked}
                  onChange={(e) => {
                    const next = [...doubleBlocks];
                    next[i] = e.target.checked;
                    setDoubleBlocks(next);
                  }}
                />
                <span>{i}</span>
              </label>
            ))}
          </div>

          {/* Single blocks row 1 */}
          <div style={{ display: 'flex', alignItems: 'center', gap: 8, marginTop: 8, flexWrap: 'wrap' }}>
            <span style={{ color: '#70AD47', width: 64, fontWeight: 500 }}>Single:</span>
            {singleBlocks.slice(0, 12).map((checked, i) => (
              <label key={i} className="check-line" style={{ margin: 0, paddingRight: 4 }}>
                <input
                  type="checkbox"
                  checked={checked}
                  onChange={(e) => {
                    const next = [...singleBlocks];
                    next[i] = e.target.checked;
                    setSingleBlocks(next);
                  }}
                />
                <span>{i}</span>
              </label>
            ))}
          </div>

          {/* Single blocks row 2 */}
          <div style={{ display: 'flex', alignItems: 'center', gap: 8, marginTop: 8, flexWrap: 'wrap' }}>
            <span style={{ color: '#ED7D31', width: 64, fontWeight: 500 }}>Single:</span>
            {singleBlocks.slice(12, 24).map((checked, idx) => {
              const i = idx + 12;
              return (
                <label key={i} className="check-line" style={{ margin: 0, paddingRight: 4 }}>
                  <input
                    type="checkbox"
                    checked={checked}
                    onChange={(e) => {
                      const next = [...singleBlocks];
                      next[i] = e.target.checked;
                      setSingleBlocks(next);
                    }}
                  />
                  <span>{i}</span>
                </label>
              );
            })}
          </div>

          <p className="muted" style={{ fontStyle: 'italic', fontSize: 12, marginTop: 10 }}>
            double + single 0-1 = style+composition | single 1-16 = identity (overlaps at 1 and 12-16) | single 12-23 = details
          </p>
        </div>
      )}

      {/* Card 4: Options */}
      <div className="panel source-card">
        <h2>Options</h2>
        <p className="muted">
          Timesteps: &apos;all&apos; for general, &apos;late&apos; for style, &apos;early&apos; for composition. | Forward Passes: 0 = pure weight SVD (fastest, timestep-agnostic). Higher = better activation-weighted accuracy.
        </p>
        <div style={{ display: 'flex', alignItems: 'center', gap: 24, marginTop: 8, flexWrap: 'wrap' }}>
          <div className="field" style={{ width: 120 }}>
            <span>Target Rank:</span>
            <select value={targetRank} onChange={(e) => handleRankChange(e.target.value)}>
              {['1', '2', '4', '8', '16'].map((r) => (
                <option key={r} value={r}>
                  {r}
                </option>
              ))}
            </select>
          </div>

          {extractFamily === 'klein' && (
            <>
              <div className="field" style={{ width: 140 }}>
                <span>Timesteps:</span>
                <select
                  value={timesteps}
                  onChange={(e) => setTimesteps(e.target.value)}
                  disabled={forwardPasses === '0'}
                >
                  {['all', 'early', 'mid', 'late'].map((t) => (
                    <option key={t} value={t}>
                      {t}
                    </option>
                  ))}
                </select>
              </div>

              <div className="field" style={{ width: 140 }}>
                <span>Forward Passes:</span>
                <select value={forwardPasses} onChange={(e) => handleForwardPassesChange(e.target.value)}>
                  {['0', '8', '16', '32'].map((s) => (
                    <option key={s} value={s}>
                      {s}
                    </option>
                  ))}
                </select>
              </div>
            </>
          )}
        </div>
      </div>

      {/* Card 5: Prompt (Klein only) */}
      {extractFamily === 'klein' && (
        <div className="panel source-card">
          <h2>Prompt</h2>
          <p className="muted">
            Used during the GPU probe forward passes. Include the source LoRA&apos;s trigger word for best results.
          </p>
          <div className="field" style={{ marginTop: 8 }}>
            <span>Prompt:</span>
            <input
              type="text"
              value={prompt}
              onChange={(e) => setPrompt(e.target.value)}
              placeholder="e.g. portrait of a woman in high detail"
            />
          </div>
        </div>
      )}

      {/* Card 6: Run */}
      <div className="panel source-card">
        <h2>Run</h2>
        <p className="muted">
          Extraction runs SVD on each block and can take several minutes depending on rank and block count.
        </p>
        <div className="actions" style={{ marginTop: 8 }}>
          <button
            type="button"
            className="primary"
            disabled={isRunning}
            onClick={handleRunExtract}
          >
            Extract LoRA
          </button>
          <button
            type="button"
            className="secondary"
            disabled={!hasOutputFolder || isRunning}
            onClick={() => alert('Opening output folder: /workspace/output_loras')}
          >
            Open Output Folder
          </button>
        </div>

        {progressText && (
          <div style={{ marginTop: 12, fontWeight: 'bold', color: '#ff8c3a', fontSize: 13 }}>
            {progressText}
          </div>
        )}

        <div style={{ marginTop: 8, fontSize: 12, color: '#8899a6', lineHeight: 1.4 }}>
          {getTimeNote()}
        </div>
      </div>

      {/* Card 7: Output Log */}
      <div className="panel source-card">
        <h2>Output Log</h2>
        <pre className="output-log" style={{ minHeight: 180, whiteSpace: 'pre-wrap' }}>
          {outputLog || 'Ready — select a source LoRA and click Extract LoRA.'}
        </pre>
      </div>

      {/* Tutorial Button */}
      <div style={{ padding: '8px 0 16px' }}>
        <button
          type="button"
          style={{
            backgroundColor: '#CC0000',
            color: '#FFFFFF',
            border: 'none',
            borderRadius: 4,
            padding: '6px 16px',
            fontSize: 13,
            fontWeight: 'bold',
            cursor: 'pointer',
            display: 'inline-flex',
            alignItems: 'center',
            gap: 6,
          }}
          onClick={() => window.open('https://www.youtube.com/watch?v=yrz0l6URGGk', '_blank')}
        >
          <span>▶</span>  Tutorial
        </button>
      </div>
    </div>
  );
}
