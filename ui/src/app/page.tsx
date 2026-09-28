'use client';

import React, { useState, useCallback, useEffect } from 'react';
import { SettingsProvider, useSettingsStore } from '@/store/settings-store';
import StartTab from '@/components/tabs/start-tab';
import ImagePrepTab from '@/components/tabs/image-prep-tab';
import CaptionsTab from '@/components/tabs/captions-tab';
import SamplesTab from '@/components/tabs/samples-tab';
import TrainingTab from '@/components/tabs/training-tab';
import ProfilerTab from '@/components/tabs/profiler-tab';
import RepairStudioTab from '@/components/tabs/repair-studio-tab';
import ExplorerTab from '@/components/tabs/explorer-tab';
import LoraRoyaleTab from '@/components/tabs/lora-royale-tab';
import ExtractTab from '@/components/tabs/extract-tab';
import MetadataTab from '@/components/tabs/metadata-tab';
import PreferencesTab from '@/components/tabs/preferences-tab';

// Tabs ordered by natural workflow — lines 1572-1630 of jj.py
const TABS = [
  { id: 'start', label: '1. Start', component: StartTab },
  { id: 'image_prep', label: '2. Image Prep', component: ImagePrepTab },
  { id: 'captions', label: '3. Captions', component: CaptionsTab },
  { id: 'samples', label: '4. Samples', component: SamplesTab },
  { id: 'training', label: '5. Training', component: TrainingTab },
  { id: 'profiler', label: 'Profiler', component: ProfilerTab },
  { id: 'repair_studio', label: 'Repair Studio', component: RepairStudioTab },
  { id: 'explorer', label: 'LoRA the Explorer', component: ExplorerTab },
  { id: 'lora_royale', label: 'LoRA Royale', component: LoraRoyaleTab },
  { id: 'extract', label: 'Extract', component: ExtractTab },
  { id: 'metadata', label: 'Metadata', component: MetadataTab },
  { id: 'preferences', label: 'Preferences', component: PreferencesTab },
] as const;

function FizgigAppContent() {
  const [activeTab, setActiveTab] = useState('start');
  const [consoleOpen, setConsoleOpen] = useState(false);
  const { state, dispatch } = useSettingsStore();

  const handleTabChange = useCallback((tabId: string) => {
    setActiveTab(tabId);
  }, []);

  // Poll system GPU VRAM and RAM status every 3 seconds (mirrors self._status_reader_loop in jj.py)
  useEffect(() => {
    const fetchStatus = () => {
      fetch('/api/gpu')
        .then((res) => res.json())
        .then((data) => {
          if (data && data.vram && data.ram) {
            dispatch({
              type: 'SET_SYSTEM_STATS',
              payload: {
                vram_used: data.vram.used,
                vram_total: data.vram.total,
                ram_used: data.ram.used,
                ram_total: data.ram.total,
              },
            });
          }
        })
        .catch(() => null);
    };

    fetchStatus();
    const interval = setInterval(fetchStatus, 3000);
    return () => clearInterval(interval);
  }, [dispatch]);

  const ActiveComponent = TABS.find((t) => t.id === activeTab)?.component ?? StartTab;

  const vramUsedGb = (state.vram_used / (1024 * 1024 * 1024)).toFixed(1);
  const vramTotalGb = (state.vram_total / (1024 * 1024 * 1024)).toFixed(1);
  const vramPct = state.vram_total > 0 ? Math.min(100, (state.vram_used / state.vram_total) * 100) : 0;

  const ramUsedGb = (state.ram_used / (1024 * 1024 * 1024)).toFixed(1);
  const ramTotalGb = (state.ram_total / (1024 * 1024 * 1024)).toFixed(1);
  const ramPct = state.ram_total > 0 ? Math.min(100, (state.ram_used / state.ram_total) * 100) : 0;

  return (
    <div className="fizgig-window">
      {/* Mobile Tab Selector (shown on small screens) */}
      <div className="sm:hidden px-3 pt-2 pb-1 bg-[#18212a] border-b border-[#3a4555]">
        <select
          value={activeTab}
          onChange={(e) => handleTabChange(e.target.value)}
          aria-label="Select Active Tab"
          className="w-full py-1.5 px-2 text-xs font-semibold rounded bg-[#202b36] text-white border border-[#647080] focus:outline-none focus:border-[#3b82f6]"
        >
          {TABS.map((tab) => (
            <option key={tab.id} value={tab.id}>
              {tab.label}
            </option>
          ))}
        </select>
      </div>

      {/* Tab strip */}
      <div className="tab-strip">
        {TABS.map((tab) => (
          <button
            key={tab.id}
            onClick={() => handleTabChange(tab.id)}
            className={`tab ${activeTab === tab.id ? 'active' : ''}`}
          >
            {tab.label}
          </button>
        ))}
      </div>

      {/* Tab content */}
      <div className="workbench-viewport">
        {/* @ts-ignore */}
        <ActiveComponent onNavigateTab={handleTabChange} />
      </div>

      {/* Status bar */}
      <div className="status-bar">
        <div className="status-meter">
          <b>VRAM</b>
          <div className="meter-track">
            <span className="meter-fill vram" style={{ width: `${vramPct}%` }} />
          </div>
          <small>{state.vram_total > 0 ? `${vramUsedGb} / ${vramTotalGb} GB` : '— / — GB'}</small>
        </div>
        <div className="status-meter">
          <b>RAM</b>
          <div className="meter-track">
            <span className="meter-fill ram" style={{ width: `${ramPct}%` }} />
          </div>
          <small>{state.ram_total > 0 ? `${ramUsedGb} / ${ramTotalGb} GB` : '— / — GB'}</small>
        </div>
        <span className="status-hint">{state.training_state === 'running' ? 'Training in progress...' : state.status_hint}</span>
        <button
          type="button"
          className="secondary small"
          style={{ marginLeft: 'auto', padding: '2px 8px', fontSize: 11 }}
          onClick={() => setConsoleOpen(true)}
        >
          Console
        </button>
      </div>

      {/* Console Log Popup */}
      {consoleOpen && (
        <div className="modal-backdrop">
          <div className="start-dialog" style={{ width: 900 }}>
            <h2>Fizgig — Console Log</h2>
            <button className="secondary" onClick={() => setConsoleOpen(false)}>
              Close
            </button>
            <pre className="output-log" style={{ marginTop: 12, maxHeight: 400, overflowY: 'auto' }}>
              {state.training_logs.length > 0
                ? state.training_logs.join('')
                : 'Console log output will appear here...'}
            </pre>
          </div>
        </div>
      )}
    </div>
  );
}

export default function FizgigApp() {
  return (
    <SettingsProvider>
      <FizgigAppContent />
    </SettingsProvider>
  );
}
