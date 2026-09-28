"use client";

import { useState, useCallback } from "react";
import { COLORS, SAMPLE_RESOLUTIONS } from "@/lib/constants";

/**
 * Bottom status panel: stacked VRAM + system-RAM gradient fill bars (with
 * per-run peak ticks) on the left, the live sample override on the right,
 * and a remembered hide/show toggle. A daemon thread does the reads so the
 * Tk redraw never stalls on an nvidia-smi call.
 * 
 * _build_status_bar — Lines 2541-2668 of jj.py
 */
export default function StatusBar() {
  const [visible, setVisible] = useState(true);
  const [overrideEnabled, setOverrideEnabled] = useState(false);
  const [overrideSeed, setOverrideSeed] = useState("1234");
  const [overrideW, setOverrideW] = useState("768");
  const [overrideH, setOverrideH] = useState("768");
  const [overridePrompt, setOverridePrompt] = useState("");
  const [overrideRef, setOverrideRef] = useState("");
  const [vramUsed, setVramUsed] = useState(0);
  const [vramTotal, setVramTotal] = useState(0);
  const [vramPeak, setVramPeak] = useState(0);
  const [ramUsed, setRamUsed] = useState(0);
  const [ramTotal, setRamTotal] = useState(0);
  const [ramPeak, setRamPeak] = useState(0);

  // _toggle_status_bar — Lines 2670-2680 of jj.py
  const toggleBar = useCallback(() => {
    setVisible(prev => !prev);
  }, []);

  /** Gradient fill for the usage bars.
   * _draw_status_segment — Lines 2756-2780 of jj.py */
  const renderUsageBar = (used: number, total: number, peak: number, label: string) => {
    const pct = total > 0 ? (used / total) * 100 : 0;
    const peakPct = total > 0 ? (peak / total) * 100 : 0;
    const usedGB = (used / (1024 ** 3)).toFixed(1);
    const totalGB = (total / (1024 ** 3)).toFixed(1);

    return (
      <div className="flex items-center gap-3">
        <span
          className="text-xs font-medium w-12 text-right"
          style={{ color: COLORS.text_muted }}
        >
          {label}
        </span>
        <div
          className="relative w-[360px] h-[27px] rounded-sm overflow-hidden"
          style={{ backgroundColor: COLORS.bg_surface }}
        >
          {/* Fill bar */}
          <div
            className="absolute inset-y-0 left-0 rounded-sm"
            style={{
              width: `${pct}%`,
              background: `linear-gradient(to right, ${COLORS.accent}, ${COLORS.accent_hover})`,
            }}
          />
          {/* Peak tick */}
          {peakPct > 0 && (
            <div
              className="absolute inset-y-0 w-0.5"
              style={{
                left: `${peakPct}%`,
                backgroundColor: COLORS.warning,
              }}
            />
          )}
          {/* Label overlay */}
          <div className="absolute inset-0 flex items-center justify-center">
            <span className="text-xs font-mono" style={{ color: COLORS.text_primary }}>
              {usedGB} / {totalGB} GB
            </span>
          </div>
        </div>
      </div>
    );
  };

  return (
    // container = tk.Frame(master, bg=COLORS["bg_deep"])
    // container.pack(side=tk.BOTTOM, fill=tk.X)
    <div style={{ backgroundColor: COLORS.bg_deep }}>
      {/* The expandable bar (sits above the handle) — Lines 2563-2566 */}
      {visible && (
        <div
          className="flex items-stretch"
          style={{ height: 82, backgroundColor: COLORS.bg_deep }}
        >
          {/* --- left: stacked VRAM (top) + RAM (bottom) gradient bars ---
              Lines 2568-2576 of jj.py */}
          <div className="flex flex-col justify-center gap-1.5 px-3.5 py-2.5">
            {renderUsageBar(vramUsed, vramTotal, vramPeak, "VRAM")}
            {renderUsageBar(ramUsed, ramTotal, ramPeak, "RAM")}
          </div>

          {/* --- right: live sample override (surface-coloured mini panel) ---
              Lines 2596-2654 of jj.py */}
          <div
            className="flex-1 flex flex-col justify-center rounded mx-0 my-2.5 px-2"
            style={{ backgroundColor: COLORS.bg_surface }}
          >
            {/* Row 1: Override checkbox + seed/W/H/Ref */}
            <div className="flex items-center gap-2 px-2 pt-1">
              <label className="flex items-center gap-1.5 cursor-pointer">
                <input
                  type="checkbox"
                  checked={overrideEnabled}
                  onChange={(e) => setOverrideEnabled(e.target.checked)}
                  className="accent-blue-500"
                />
                <span className="text-xs" style={{ color: COLORS.text_primary }}>
                  Override next sample
                </span>
              </label>

              <span className="text-[10px]" style={{ color: COLORS.text_muted }}>seed</span>
              <input
                type="text"
                value={overrideSeed}
                onChange={(e) => setOverrideSeed(e.target.value)}
                className="w-14 text-xs px-1 py-0.5 rounded border"
                style={{
                  backgroundColor: COLORS.bg_deep,
                  color: COLORS.text_primary,
                  borderColor: COLORS.border,
                }}
                disabled={!overrideEnabled}
              />

              {/* Same list as the Samples tab (SAMPLE_RESOLUTIONS) — these two had drifted, and
                  the override's lower ceiling silently downgraded a 1280/1536 preview. */}
              <span className="text-[10px]" style={{ color: COLORS.text_muted }}>W</span>
              <select
                value={overrideW}
                onChange={(e) => setOverrideW(e.target.value)}
                className="text-xs px-1 py-0.5 rounded border"
                style={{
                  backgroundColor: COLORS.bg_deep,
                  color: COLORS.text_primary,
                  borderColor: COLORS.border,
                }}
                disabled={!overrideEnabled}
              >
                {SAMPLE_RESOLUTIONS.map(r => <option key={r} value={r}>{r}</option>)}
              </select>

              <span className="text-[10px]" style={{ color: COLORS.text_muted }}>H</span>
              <select
                value={overrideH}
                onChange={(e) => setOverrideH(e.target.value)}
                className="text-xs px-1 py-0.5 rounded border"
                style={{
                  backgroundColor: COLORS.bg_deep,
                  color: COLORS.text_primary,
                  borderColor: COLORS.border,
                }}
                disabled={!overrideEnabled}
              >
                {SAMPLE_RESOLUTIONS.map(r => <option key={r} value={r}>{r}</option>)}
              </select>

              {/* Reference image — auto-capped to ~0.20 MP by the trainer so a big image can't OOM the
                  sample. Shown for BOTH families: Klein uses it as edit conditioning, Krea 2 routes it
                  through the Qwen3-VL vision path. */}
              <span className="text-[10px]" style={{ color: COLORS.text_muted }}>Ref</span>
              <button
                className="text-[10px] px-2 py-0.5 rounded border cursor-pointer"
                style={{
                  backgroundColor: COLORS.bg_deep,
                  color: COLORS.text_primary,
                  borderColor: COLORS.border,
                }}
                disabled={!overrideEnabled}
              >
                Browse…
              </button>
              <span className="text-[10px]" style={{ color: COLORS.text_muted }}>
                {overrideRef || "(none)"}
              </span>
              {overrideRef && (
                <button
                  className="text-[10px] cursor-pointer"
                  style={{ color: COLORS.text_muted }}
                  onClick={() => setOverrideRef("")}
                >
                  ✕
                </button>
              )}
            </div>

            {/* Row 2: Prompt — Lines 2645-2650 of jj.py */}
            <div className="flex items-center gap-2 px-2 py-1">
              <span className="text-[10px]" style={{ color: COLORS.text_muted }}>Prompt</span>
              <input
                type="text"
                value={overridePrompt}
                onChange={(e) => setOverridePrompt(e.target.value)}
                className="flex-1 text-xs px-1.5 py-0.5 rounded border"
                style={{
                  backgroundColor: COLORS.bg_deep,
                  color: COLORS.text_primary,
                  borderColor: COLORS.border,
                }}
                disabled={!overrideEnabled}
              />
            </div>
          </div>

          {/* --- far right: training-queue button (lower-right corner of the app) ---
              Lines 2578-2593 of jj.py */}
          <div className="flex items-stretch py-2.5 pr-3.5 pl-0">
            <button
              className="flex items-center px-3 rounded font-bold text-sm cursor-pointer"
              style={{
                // bg=COLORS["queue_blue"], fg=COLORS["bg_deep"]
                backgroundColor: COLORS.queue_blue,
                color: COLORS.bg_deep,
              }}
            >
              📋 Queue
            </button>
          </div>
        </div>
      )}

      {/* Thin always-visible handle carrying the show/hide toggle.
          Lines 2550-2560 of jj.py */}
      <div className="flex justify-end" style={{ backgroundColor: COLORS.bg_deep }}>
        <button
          onClick={toggleBar}
          className="text-[10px] px-2.5 py-0.5 cursor-pointer"
          style={{
            color: COLORS.text_muted,
            backgroundColor: "transparent",
            border: "none",
          }}
        >
          {/* text="▾ Hide stats" / "▴ Show stats" — Lines 2554-2559 of jj.py */}
          {visible ? "▾ Hide stats" : "▴ Show stats"}
        </button>
      </div>
    </div>
  );
}
