'use client';

import React, { useState } from 'react';
import { useSettingsStore } from '@/store/settings-store';
import FolderPickerModal from '@/components/ui/folder-picker-modal';

interface StartTabProps {
  onNavigateTab?: (tabId: string) => void;
}

export default function StartTab({ onNavigateTab }: StartTabProps) {
  const { state, dispatch } = useSettingsStore();
  const [setupPromptDismissed, setSetupPromptDismissed] = useState(false);
  const [folderModalOpen, setFolderModalOpen] = useState(false);

  // Check if at least one model family is configured (Klein 4 paths or Krea 3 paths)
  const kleinOk = Boolean(
    state.prefs.base_dit?.trim() &&
    state.prefs.distilled_dit?.trim() &&
    state.prefs.vae?.trim() &&
    state.prefs.text_encoder?.trim()
  );
  const kreaOk = Boolean(
    state.prefs.krea2_raw_dit?.trim() &&
    state.prefs.krea2_vae?.trim() &&
    state.prefs.krea2_text_encoder?.trim()
  );
  const modelPathsOk = kleinOk || kreaOk;

  const handleFolderChange = (val: string) => {
    dispatch({ type: 'SET_IMAGE_FOLDER', payload: val });

    // Auto-save dataset TOML and .last_used on write (mirrors jj.py lines 1700-1703)
    fetch('/api/datasets', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        image_dir: val,
        caption_ext: state.dataset_caption_ext,
        resolution: parseInt(state.settings.SAMPLE_WIDTH.toString(), 10) || 1024,
      }),
    }).catch(() => null);

    fetch('/api/prefs', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        lastUsed: { ...state.lastUsed, image_folder: val },
      }),
    }).catch(() => null);
  };

  const steps = [
    { num: '1', id: 'start', tab_name: 'Start', desc: 'Choose your training image folder below.', is_optional: false },
    { num: '2', id: 'image_prep', tab_name: 'Image Prep', desc: 'Resize, convert to PNG, or face-crop. (Video Prep for MiniMax)', is_optional: true },
    { num: '3', id: 'captions', tab_name: 'Captions', desc: 'Write trigger-word captions or generate them with AI.', is_optional: false },
    { num: '4', id: 'samples', tab_name: 'Samples', desc: 'Configure in-training preview prompts.', is_optional: false },
    { num: '5', id: 'training', tab_name: 'Training', desc: 'Pick a preset, tune settings, click Start Training.', is_optional: false },
  ];

  const tools = [
    { id: 'profiler', name: 'Profiler', desc: "Analyze a LoRA's per-block activation profile and produce an HTML report." },
    { id: 'repair_studio', name: 'Repair Studio', desc: 'Live per-block sliders with side-by-side preview. Blend in a donor LoRA and bake the result to a new .safetensors.' },
    { id: 'explorer', name: 'LoRA the Explorer', desc: 'Evolutionary discovery — the computer proposes random mutations, you pick favourites, and the LoRA evolves. Seamlessly connected to Repair Studio.' },
    { id: 'lora_royale', name: 'LoRA Royale', desc: 'Compare epochs (or any LoRAs) on one seed, then export share-ready clips — seed, prompt, and LoRA-strength travels, deflickered and ready for social.' },
    { id: 'extract', name: 'Extract', desc: 'Distill a LoRA to a lower rank with optional block- and timestep-targeted presets. Supports LyCORIS (LoKR / LoHa) sources.' },
  ];

  return (
    <div className="start-screen">
      {/* Banner */}
      <div className="start-intro">
        <h1>Welcome to Fizgig</h1>
        <p>A focused, local trainer and workbench for Flux 2 Klein 9B, Krea 2 and MiniMax H3 LoRAs — train, profile, repair, explore, and extract, all in one place.</p>
      </div>

      {/* Workflow card */}
      <div className="start-workflow">
        <div className="workflow-copy">
          <h2>Training Workflow</h2>
          {steps.map((step) => (
            <div
              key={step.num}
              className="workflow-row group cursor-pointer hover:bg-[#202b36] p-1.5 rounded transition-colors"
              onClick={() => onNavigateTab && onNavigateTab(step.id)}
            >
              <b>{step.num}</b>
              <strong className="group-hover:text-[#60a5fa] transition-colors">{step.tab_name}</strong>
              {step.is_optional && <em>OPTIONAL</em>}
              <span>{step.desc}</span>
            </div>
          ))}
        </div>
        <div className="start-logo flex items-center justify-center overflow-hidden bg-[#202833]">
          <img
            src="/logo.jpg"
            alt="Fizgig LoRA Studio"
            className="w-full h-full object-cover block"
            onError={(e) => {
              (e.currentTarget as HTMLImageElement).style.display = 'none';
            }}
          />
        </div>
      </div>

      {/* Folder picker card */}
      <div className="panel start-folder-card accent-card">
        <h2>Training image folder</h2>
        <p className="muted">This is the single place you set your dataset folder. Image Prep, Captions, and Training all read from it automatically.</p>
        <div className="start-controls">
          <div className="field">
            <input
              type="text"
              value={state.image_folder}
              onChange={(e) => handleFolderChange(e.target.value)}
              placeholder="e.g. /home/user/my_dataset or click Browse..."
            />
          </div>
          <button
            type="button"
            className="secondary flex items-center gap-1.5"
            onClick={() => setFolderModalOpen(true)}
          >
            <span>📁</span>
            <span>Browse…</span>
          </button>
        </div>
        {state.image_folder ? (
          <div className="mt-2.5 flex items-center gap-2 text-xs text-[#34d399]">
            <span>✓</span>
            <span className="font-mono text-[11px] truncate">Current dataset: {state.image_folder}</span>
          </div>
        ) : (
          <div className="mt-2.5 text-xs text-[#9caaba] italic">
            Tip: Click Browse to select any directory on this machine or type an absolute path above.
          </div>
        )}
      </div>

      {/* Setup prompt */}
      {!setupPromptDismissed && !modelPathsOk && (
        <div className="start-warning">
          <h2>⚠️ Model files not configured</h2>
          <p>Head to the Preferences tab to set your model paths before training or using the tools. Each model row has a Download link that opens the correct HuggingFace page.</p>
          <div className="warning-actions">
            <button
              type="button"
              className="secondary"
              onClick={() => onNavigateTab && onNavigateTab('preferences')}
            >
              Open Preferences
            </button>
            <button
              type="button"
              className="warning-dismiss"
              onClick={() => setSetupPromptDismissed(true)}
            >
              Don&apos;t show this again
            </button>
          </div>
        </div>
      )}

      {/* Tools card */}
      <div className="panel" style={{ marginTop: 20 }}>
        <h2 style={{ margin: '0 0 10px', fontSize: 14 }}>Post-Training Tools</h2>
        <p className="muted" style={{ margin: '0 0 14px' }}>Fizgig is more than a trainer — these tabs let you understand and tune any Klein LoRA you&apos;ve made (or downloaded).</p>
        {tools.map((tool) => (
          <button
            key={tool.id}
            type="button"
            className="tool-row"
            onClick={() => onNavigateTab && onNavigateTab(tool.id)}
          >
            <strong>{tool.name}</strong>
            <span>{tool.desc}</span>
          </button>
        ))}
      </div>

      {/* Action Buttons */}
      <div className="start-help">
        <button
          type="button"
          className="help tutorial"
          onClick={() => window.open('https://www.youtube.com/watch?v=yrz0l6URGGk', '_blank')}
        >
          ▶  Tutorial
        </button>
        <button
          type="button"
          className="help coffee"
          onClick={() => window.open('https://buymeacoffee.com/lorasandlenses', '_blank')}
        >
          ☕  Buy me a coffee
        </button>
        <button
          type="button"
          className="help runpod"
          onClick={() =>
            window.open(
              'https://console.runpod.io/deploy?type=GPU&gpu=RTX+5090&count=1&template=faoq8ed6um&ref=vkb387ep',
              '_blank'
            )
          }
        >
          ⚡  Deploy on RunPod
        </button>
        <button
          type="button"
          className="help about"
          onClick={() => alert('Fizgig — Klein 9B & Krea 2 LoRA Studio\nVersion 2.8.5')}
        >
          About
        </button>
      </div>

      {/* Directory Browser Modal */}
      <FolderPickerModal
        isOpen={folderModalOpen}
        initialPath={state.image_folder || ''}
        onSelect={(chosenPath) => handleFolderChange(chosenPath)}
        onClose={() => setFolderModalOpen(false)}
        title="Select Training Image Folder"
      />
    </div>
  );
}
