'use client';

import React, { useState } from 'react';
import { useSettingsStore } from '@/store/settings-store';

interface PrefRowProps {
  label: string;
  value: string;
  onChange: (val: string) => void;
  hint: string;
  downloadUrl?: string;
  downloadLabel?: string;
  downloadUrl2?: string;
  downloadLabel2?: string;
}

function PrefRow({
  label,
  value,
  onChange,
  hint,
  downloadUrl,
  downloadLabel = 'Download',
  downloadUrl2,
  downloadLabel2 = 'Download',
}: PrefRowProps) {
  return (
    <div style={{ marginBottom: 14 }}>
      <div style={{ display: 'flex', alignItems: 'center', gap: 10, flexWrap: 'wrap' }}>
        <span style={{ width: 180, fontWeight: 500, fontSize: 13, color: '#cbd5e1' }}>{label}</span>
        <div style={{ flex: 1, minWidth: 260 }}>
          <input
            type="text"
            value={value || ''}
            onChange={(e) => onChange(e.target.value)}
            style={{ width: '100%' }}
          />
        </div>
        <button
          type="button"
          className="secondary small"
          onClick={() => {
            const p = prompt(`Enter path for ${label}`, value || '');
            if (p !== null) onChange(p);
          }}
        >
          Browse
        </button>
        {downloadUrl && (
          <a
            href={downloadUrl}
            target="_blank"
            rel="noopener noreferrer"
            style={{
              color: '#38bdf8',
              fontSize: 12,
              textDecoration: 'underline',
              cursor: 'pointer',
              whiteSpace: 'nowrap',
            }}
          >
            {downloadLabel}
          </a>
        )}
        {downloadUrl2 && (
          <a
            href={downloadUrl2}
            target="_blank"
            rel="noopener noreferrer"
            style={{
              color: '#38bdf8',
              fontSize: 12,
              textDecoration: 'underline',
              cursor: 'pointer',
              whiteSpace: 'nowrap',
            }}
          >
            {downloadLabel2}
          </a>
        )}
      </div>
      <div style={{ marginLeft: 190, marginTop: 4, fontSize: 11, fontStyle: 'italic', color: '#7e8e9f', lineHeight: 1.4 }}>
        {hint}
      </div>
    </div>
  );
}

export default function PreferencesTab() {
  const { state, dispatch } = useSettingsStore();
  const prefs = state.prefs;

  const updatePref = (key: string, value: any) => {
    dispatch({
      type: 'UPDATE_PREFS',
      payload: { [key]: value },
    });
    // Auto-save to backend
    fetch('/api/prefs', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        prefs: { ...prefs, [key]: value },
      }),
    }).catch(() => {});
  };

  // Collapsible states
  const [isKleinOpen, setIsKleinOpen] = useState(true);
  const [isKreaOpen, setIsKreaOpen] = useState(false);
  const [isMinimaxOpen, setIsMinimaxOpen] = useState(false);

  // Status for model fetching
  const [fetchStatusKlein, setFetchStatusKlein] = useState('');
  const [fetchStatusKrea, setFetchStatusKrea] = useState('');
  const [fetchStatusMinimax, setFetchStatusMinimax] = useState('');

  // Calculate missing paths for badges
  const missingKlein = [prefs.base_dit, prefs.distilled_dit, prefs.vae, prefs.text_encoder].filter((p) => !p?.trim()).length;
  const missingKrea = [prefs.krea2_raw_dit, prefs.krea2_turbo_dit, prefs.krea2_vae, prefs.krea2_text_encoder].filter((p) => !p?.trim()).length;
  const missingMinimax = [prefs.minimax_dit, prefs.minimax_text_encoder, prefs.minimax_vae].filter((p) => !p?.trim()).length;

  const handleResetPrefs = async () => {
    if (confirm('Restore all paths to defaults?')) {
      const reset = {
        base_dit: '',
        distilled_dit: '',
        vae: '',
        text_encoder: '',
        krea2_raw_dit: '',
        krea2_turbo_dit: '',
        krea2_vae: '',
        krea2_text_encoder: '',
        krea2_turbo_lora: '',
        minimax_dit: '',
        minimax_ref_dit: '',
        minimax_text_encoder: '',
        minimax_vae: '',
        minimax_audio_vae: '',
        minimax_turbo_lora: '',
        minimax_training_adapter: '',
        minimax_ref_training_adapter: '',
        inference_blocks_to_swap: 'Auto (detect from GPU)',
        inference_int8: '1',
        lora_output_dir: 'output_loras',
        profiles_dir: 'profiles',
        cache_dir: 'cache',
        input_lora_dir: '',
        input_ref_dir: '',
        input_dataset_dir: '',
        runpod_stop_when_done: '0',
        runpod_api_key: '',
        cuda_device: '',
      };
      dispatch({ type: 'UPDATE_PREFS', payload: reset });
      await fetch('/api/prefs', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ prefs: reset }),
      });
      alert('Preferences reset to default values.');
    }
  };

  return (
    <div className="fizgig-tab-page">
      {/* Banner */}
      <div className="tab-banner">
        <h1>Preferences</h1>
        <p>
          Centralised paths + inference performance knobs. Changes here propagate to every tab automatically and persist to prefs.json.
        </p>
      </div>

      {/* Model Paths (Klein 9B) */}
      <div className="panel source-card">
        <div
          style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', cursor: 'pointer' }}
          onClick={() => setIsKleinOpen(!isKleinOpen)}
        >
          <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
            <span style={{ fontSize: 12, color: '#94a3b8' }}>{isKleinOpen ? '▼' : '▶'}</span>
            <h2 style={{ margin: 0 }}>Model Paths (Klein 9B)</h2>
          </div>
          <span
            style={{
              fontSize: 12,
              fontWeight: 500,
              color: missingKlein === 0 ? '#4ade80' : '#f59e0b',
            }}
          >
            {missingKlein === 0 ? '✓ configured' : `${missingKlein} path${missingKlein !== 1 ? 's' : ''} needed if training Klein 9B`}
          </span>
        </div>

        {isKleinOpen && (
          <div style={{ marginTop: 12 }}>
            <p className="muted" style={{ marginBottom: 16 }}>
              Absolute paths to the four model files. Each row has a Download link that opens the HuggingFace page in your browser.
            </p>

            <PrefRow
              label="Base DiT:"
              value={prefs.base_dit}
              onChange={(v) => updatePref('base_dit', v)}
              hint="Klein 9B Base model (for training &amp; precise profiling). Recommended: the fp8 version — same training quality, ~half the VRAM (stays resident at ~9.6GB, fits 16GB cards). The bf16 version is the larger alternative.  ·  ~9.5GB fp8 — Black Forest Labs (flux-2-klein-base-9b-fp8.safetensors)  ·  ~17GB bf16 (flux-2-klein-base-9b.safetensors)"
              downloadUrl="https://huggingface.co/black-forest-labs/FLUX.2-klein-base-9b-fp8/tree/main"
              downloadLabel="Download fp8 (recommended)"
              downloadUrl2="https://huggingface.co/black-forest-labs/FLUX.2-klein-base-9B/tree/main"
              downloadLabel2="Download bf16"
            />

            <PrefRow
              label="Distilled DiT:"
              value={prefs.distilled_dit}
              onChange={(v) => updatePref('distilled_dit', v)}
              hint="Klein 9B Distilled model (for Repair Studio previews, fast profiling &amp; diagnostics)  ·  ~9GB fp8 quantised — Black Forest Labs (flux-2-klein-9b-fp8.safetensors)"
              downloadUrl="https://huggingface.co/black-forest-labs/FLUX.2-klein-9b-fp8/tree/main"
            />

            <PrefRow
              label="VAE / AE:"
              value={prefs.vae}
              onChange={(v) => updatePref('vae', v)}
              hint="Flux 2 AutoEncoder — use ae.safetensors from FLUX.2-dev root (NOT the vae/ subfolder Diffusers file)  ·  ~320MB  ·  get ae.safetensors from FLUX.2-dev root  ·  NOT vae/diffusion_pytorch_model.safetensors (Diffusers format, incompatible)"
              downloadUrl="https://huggingface.co/black-forest-labs/FLUX.2-dev/blob/main/ae.safetensors"
            />

            <PrefRow
              label="Text Encoder:"
              value={prefs.text_encoder}
              onChange={(v) => updatePref('text_encoder', v)}
              hint="Qwen3-8B text encoder (used by Klein 9B)  ·  ~15GB single-file safetensors — Qwen3-8B packaged for Klein 9B (Comfy-Org)"
              downloadUrl="https://huggingface.co/Comfy-Org/vae-text-encorder-for-flux-klein-9b/blob/main/split_files/text_encoders/qwen_3_8b.safetensors"
            />

            <div style={{ padding: '10px 14px', backgroundColor: '#162230', borderRadius: 4, margin: '14px 0 10px', fontSize: 12, color: '#93c5fd', lineHeight: 1.4 }}>
              💡 Training Klein only? The Krea 2 Qwen3-VL text encoder is still worth having — the Captions tab can caption ANY dataset with it, following an instruction you can edit, and it writes better training captions than Florence-2. The download button below fetches it for you; nothing else about Krea 2 is needed.
            </div>

            <div style={{ display: 'flex', alignItems: 'center', gap: 12, marginTop: 12 }}>
              <button
                type="button"
                className="secondary"
                onClick={() => setFetchStatusKlein('downloading…')}
              >
                Download Missing Models (huggingface-cli)
              </button>
              {fetchStatusKlein && <span style={{ fontSize: 12, color: '#94a3b8' }}>{fetchStatusKlein}</span>}
            </div>
          </div>
        )}
      </div>

      {/* Model Paths (Krea 2) */}
      <div className="panel source-card">
        <div
          style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', cursor: 'pointer' }}
          onClick={() => setIsKreaOpen(!isKreaOpen)}
        >
          <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
            <span style={{ fontSize: 12, color: '#94a3b8' }}>{isKreaOpen ? '▼' : '▶'}</span>
            <h2 style={{ margin: 0 }}>Model Paths (Krea 2)</h2>
          </div>
          <span
            style={{
              fontSize: 12,
              fontWeight: 500,
              color: missingKrea === 0 ? '#4ade80' : '#94a3b8',
            }}
          >
            {missingKrea === 0 ? '✓ configured' : `${missingKrea} path${missingKrea !== 1 ? 's' : ''} unconfigured (optional)`}
          </span>
        </div>

        {isKreaOpen && (
          <div style={{ marginTop: 12 }}>
            <PrefRow
              label="DiT (RAW, for training):"
              value={prefs.krea2_raw_dit}
              onChange={(v) => updatePref('krea2_raw_dit', v)}
              hint="Full bf16 base model (~15.2GB). Trainer dynamic-quantises to fp8 on launch."
            />
            <PrefRow
              label="DiT (Turbo, for preview):"
              value={prefs.krea2_turbo_dit}
              onChange={(v) => updatePref('krea2_turbo_dit', v)}
              hint="Pre-quantised fp8 Turbo model (~7.6GB) for 8-step previews."
            />
            <PrefRow
              label="VAE:"
              value={prefs.krea2_vae}
              onChange={(v) => updatePref('krea2_vae', v)}
              hint="Qwen-Image VAE (~250MB)."
            />
            <PrefRow
              label="Text Encoder (Qwen3-VL 4B):"
              value={prefs.krea2_text_encoder}
              onChange={(v) => updatePref('krea2_text_encoder', v)}
              hint="Qwen3-VL-4B vision-language encoder (~8.5GB)."
            />
          </div>
        )}
      </div>

      {/* Model Paths (MiniMax H3) */}
      <div className="panel source-card">
        <div
          style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', cursor: 'pointer' }}
          onClick={() => setIsMinimaxOpen(!isMinimaxOpen)}
        >
          <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
            <span style={{ fontSize: 12, color: '#94a3b8' }}>{isMinimaxOpen ? '▼' : '▶'}</span>
            <h2 style={{ margin: 0 }}>Model Paths (MiniMax H3)</h2>
          </div>
          <span
            style={{
              fontSize: 12,
              fontWeight: 500,
              color: missingMinimax === 0 ? '#4ade80' : '#94a3b8',
            }}
          >
            {missingMinimax === 0 ? '✓ configured' : `${missingMinimax} path${missingMinimax !== 1 ? 's' : ''} unconfigured (optional)`}
          </span>
        </div>

        {isMinimaxOpen && (
          <div style={{ marginTop: 12 }}>
            <PrefRow
              label="DiT (fl2va, first/last frame):"
              value={prefs.minimax_dit}
              onChange={(v) => updatePref('minimax_dit', v)}
              hint="Main MiniMax H3 DiT (~15GB)."
            />
            <PrefRow
              label="Text Encoder (Qwen3-VL 32B):"
              value={prefs.minimax_text_encoder}
              onChange={(v) => updatePref('minimax_text_encoder', v)}
              hint="Qwen3-VL-32B text/vision encoder."
            />
            <PrefRow
              label="Video VAE:"
              value={prefs.minimax_vae}
              onChange={(v) => updatePref('minimax_vae', v)}
              hint="H3 Video Autoencoder."
            />
          </div>
        )}
      </div>

      {/* Cloud GPU Execution Engine (Modal.com) */}
      <div className="panel source-card" style={{ border: '1px solid #0284c7' }}>
        <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: 12 }}>
          <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
            <span style={{ fontSize: 16 }}>⚡</span>
            <h2 style={{ margin: 0, color: '#38bdf8' }}>Cloud GPU Execution Engine (Modal.com)</h2>
          </div>
          <span style={{ fontSize: 12, padding: '2px 8px', borderRadius: 4, backgroundColor: '#0369a1', color: '#e0f2fe' }}>
            Zero Local VRAM / Offloaded Training
          </span>
        </div>
        <p className="muted" style={{ marginBottom: 14 }}>
          Train models, cache latents, and generate previews on serverless cloud GPUs (Modal) without consuming local GPU VRAM or taxing local CPU resources.
        </p>

        <div style={{ marginBottom: 14 }}>
          <div style={{ display: 'flex', alignItems: 'center', gap: 10, flexWrap: 'wrap' }}>
            <span style={{ width: 180, fontWeight: 500, fontSize: 13, color: '#cbd5e1' }}>Execution Target:</span>
            <div style={{ flex: 1, minWidth: 260 }}>
              <select
                value={prefs.cloud_provider || 'modal'}
                onChange={(e) => updatePref('cloud_provider', e.target.value)}
                style={{ width: '100%', padding: '6px 10px', backgroundColor: '#1e293b', border: '1px solid #334155', color: '#f8fafc', borderRadius: 4 }}
              >
                <option value="modal">Modal Cloud GPU Cluster (Serverless A100/H100) — Recommended</option>
                <option value="local">Local Hardware (Local PyTorch / CUDA)</option>
              </select>
            </div>
          </div>
          <div style={{ marginLeft: 190, marginTop: 4, fontSize: 11, fontStyle: 'italic', color: '#7e8e9f' }}>
            When set to Modal, heavy DiT caching and training runs serverless in the cloud. Checkpoints stream back to local output_loras.
          </div>
        </div>

        <div style={{ marginBottom: 14 }}>
          <div style={{ display: 'flex', alignItems: 'center', gap: 10, flexWrap: 'wrap' }}>
            <span style={{ width: 180, fontWeight: 500, fontSize: 13, color: '#cbd5e1' }}>Cloud GPU Tier:</span>
            <div style={{ flex: 1, minWidth: 260 }}>
              <select
                value={prefs.modal_gpu || 'A100-40GB'}
                onChange={(e) => updatePref('modal_gpu', e.target.value)}
                style={{ width: '100%', padding: '6px 10px', backgroundColor: '#1e293b', border: '1px solid #334155', color: '#f8fafc', borderRadius: 4 }}
              >
                <option value="A100-40GB">NVIDIA A100 (40GB VRAM) — Optimal speed/cost</option>
                <option value="A100-80GB">NVIDIA A100 (80GB VRAM) — High batch sizes / 32B models</option>
                <option value="H100-80GB">NVIDIA H100 (80GB SXM5) — Maximum throughput</option>
                <option value="A10G">NVIDIA A10G (24GB VRAM) — Budget previews</option>
              </select>
            </div>
          </div>
          <div style={{ marginLeft: 190, marginTop: 4, fontSize: 11, fontStyle: 'italic', color: '#7e8e9f' }}>
            Modal provisions on-demand within seconds and automatically shuts down when training completes.
          </div>
        </div>
      </div>

      {/* Directories */}
      <div className="panel source-card">
        <h2>Directories</h2>
        <PrefRow
          label="LoRA Output Directory:"
          value={prefs.lora_output_dir}
          onChange={(v) => updatePref('lora_output_dir', v)}
          hint="Where finished LoRAs, intermediate epochs, and samples are saved."
        />
        <PrefRow
          label="Profiles Directory:"
          value={prefs.profiles_dir}
          onChange={(v) => updatePref('profiles_dir', v)}
          hint="Where Profiler reports (.profile.json) are stored."
        />
        <PrefRow
          label="Cache Directory:"
          value={prefs.cache_dir}
          onChange={(v) => updatePref('cache_dir', v)}
          hint="Temporary latent & text encoder cache."
        />
      </div>

      {/* Actions */}
      <div className="panel source-card">
        <h2>Actions</h2>
        <div className="actions" style={{ marginTop: 8 }}>
          <button type="button" className="secondary" onClick={handleResetPrefs}>
            Reset to Defaults
          </button>
          <button
            type="button"
            className="secondary"
            onClick={() => alert(`Saved preferences sync automatically to: prefs.json`)}
          >
            Check prefs.json status
          </button>
        </div>
      </div>
    </div>
  );
}
