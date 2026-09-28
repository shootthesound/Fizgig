'use client';

import React, { useState } from 'react';
import { useSettingsStore } from '@/store/settings-store';

export default function ImagePrepTab() {
  const { state, dispatch } = useSettingsStore();

  const [prepMode, setPrepMode] = useState(state.prep_mode || 'Auto Prep (Face Crops)');
  const [prepMegapixels, setPrepMegapixels] = useState(state.prep_megapixels || '1.0');
  const [faceSelection, setFaceSelection] = useState(state.face_selection || 'Largest Face');
  const [facePadding, setFacePadding] = useState(state.face_padding || '20');
  const [deleteOriginals, setDeleteOriginals] = useState(state.delete_originals || false);
  const [outputLog, setOutputLog] = useState('');
  const [isRunning, setIsRunning] = useState(false);

  const isFaceMode = prepMode !== 'Resize Only';

  const handleRunPrep = async () => {
    if (!state.image_folder.trim()) {
      alert('Please set your training folder on the Start tab first.');
      return;
    }

    setIsRunning(true);
    setOutputLog(`Starting Image Prep in: ${state.image_folder}\nMode: ${prepMode}\nMegapixels: ${prepMegapixels}\n`);

    try {
      const res = await fetch('/api/convert', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          folder: state.image_folder,
          prepMode,
          megapixels: prepMegapixels,
          faceSelection,
          deleteOriginals,
        }),
      });
      const data = await res.json();
      if (data.success) {
        setOutputLog((prev) => prev + `Found ${data.count} images.\nProcessing completed successfully!\n`);
      } else {
        setOutputLog((prev) => prev + `Error: ${data.error}\n`);
      }
    } catch (e: any) {
      setOutputLog((prev) => prev + `Error: ${e.message}\n`);
    } finally {
      setIsRunning(false);
    }
  };

  return (
    <div className="fizgig-tab-page">
      {/* ── Banner ── */}
      <div className="tab-banner">
        <h1>Image Prep</h1>
        <p>Resize, convert to PNG, and optionally face-crop your training images. Optional — skip straight to Captions if your images are already prepared.</p>
      </div>

      {/* ── Training Folder card ── */}
      <div className="panel source-card">
        <h2>Training Folder</h2>
        <p className="muted">Everything below happens inside the training folder from the Start tab — prepared images land there, ready for the Captions tab and training.</p>
        <div className="folder-readout">
          <b>Folder:</b>
          <span>{state.image_folder || '(not set)'}</span>
        </div>
        <small className="inline-note" style={{ marginLeft: 0 }}>(set on the Start tab)</small>
      </div>

      {/* ── Working from video or audio? ── */}
      <div className="panel source-card">
        <h2>Working from video or audio?</h2>
        <p className="muted">Gizmo cuts training clips AND voice segments. Video: scrub to a moment, pick a length, save — frame rate, frame count, sizing and sound all come out on spec. Audio: open a recording (or a video, for just its sound), scrub the waveform, caption the voice — with optional Whisper transcription — and export ready training segments. Video and voice training are MiniMax H3 only; still images need none of this.</p>
        <div style={{ display: 'flex', gap: 12, alignItems: 'center' }}>
          <button
            type="button"
            className="primary"
            style={{ fontSize: 13, padding: '8px 16px' }}
            onClick={() => alert('Gizmo is designed to cut video and audio clips into spec for MiniMax H3.')}
          >
            🎬🎙  Open Gizmo
          </button>
          <span style={{ color: '#5a6b7e', fontSize: 11 }}>opens in its own window — Fizgig keeps running</span>
        </div>
      </div>

      {/* ── 1 · What to do ── */}
      <div className="panel source-card">
        <h2>1 · What to do</h2>

        {/* Prep mode radios */}
        <label className="radio-option">
          <input
            type="radio"
            name="prep_mode"
            value="Auto Prep (Face Crops)"
            checked={prepMode === 'Auto Prep (Face Crops)'}
            onChange={() => setPrepMode('Auto Prep (Face Crops)')}
          />
          <span>
            Resize + face close-ups — recommended for people
            <small>Every photo is resized and saved as PNG, PLUS a zoomed-in copy of the face saved beside it — more detail shots for better likeness.{'\n'}💡 Works best on high-res originals: if your photos are already shrunk to training size, the face close-ups come out soft. Start from the biggest versions you have.</small>
          </span>
        </label>

        <label className="radio-option">
          <input
            type="radio"
            name="prep_mode"
            value="Resize Only"
            checked={prepMode === 'Resize Only'}
            onChange={() => setPrepMode('Resize Only')}
          />
          <span>
            Resize only
            <small>Just resize + convert to PNG. Use for styles, objects, or already-cropped sets.</small>
          </span>
        </label>

        <label className="radio-option">
          <input
            type="radio"
            name="prep_mode"
            value="Face Crop Only"
            checked={prepMode === 'Face Crop Only'}
            onChange={() => setPrepMode('Face Crop Only')}
          />
          <span>
            Face close-ups only
            <small>Keep only the cropped face from each photo — the full shot is not kept.</small>
          </span>
        </label>

        {/* Options row */}
        <div className="inline-fields" style={{ marginTop: 12 }}>
          <div className="field compact-field">
            <span>Target megapixels:</span>
            <select value={prepMegapixels} onChange={(e) => setPrepMegapixels(e.target.value)}>
              {['0.25', '0.5', '0.75', '1.0', '1.5', '2.0', '2.4', '3.0', '4.2'].map((v) => (
                <option key={v} value={v}>{v}</option>
              ))}
            </select>
            <small>MP  (larger images shrink to fit; smaller are left alone)</small>
          </div>

          <div className="field compact-field">
            <span style={{ opacity: isFaceMode ? 1 : 0.4 }}>Face:</span>
            <select value={faceSelection} onChange={(e) => setFaceSelection(e.target.value)} disabled={!isFaceMode}>
              {['Largest Face', 'Largest Male Face', 'Largest Female Face'].map((v) => (
                <option key={v} value={v}>{v}</option>
              ))}
            </select>
          </div>

          <div className="field compact-field">
            <span style={{ opacity: isFaceMode ? 1 : 0.4 }}>Padding:</span>
            <div style={{ display: 'flex', gap: 4, alignItems: 'center' }}>
              <input
                type="text"
                value={facePadding}
                onChange={(e) => setFacePadding(e.target.value)}
                disabled={!isFaceMode}
                style={{ width: 45 }}
              />
              <small style={{ opacity: isFaceMode ? 1 : 0.4, minHeight: 0 }}>% around the face</small>
            </div>
          </div>
        </div>

        <p className="explanation">Sizing is by target area (megapixels), not longest side — this matches how training buckets your images, so prepping no longer costs you resolution. Aspect ratio is preserved; nothing is cropped.</p>
      </div>

      {/* ── 2 · Your originals ── */}
      <div className="panel source-card">
        <h2>2 · Your originals</h2>
        <label className="radio-option">
          <input
            type="radio"
            name="originals"
            value="keep"
            checked={!deleteOriginals}
            onChange={() => setDeleteOriginals(false)}
          />
          <span>Keep them safe — moved to an &apos;originals&apos; subfolder</span>
        </label>
        <label className="radio-option">
          <input
            type="radio"
            name="originals"
            value="replace"
            checked={deleteOriginals}
            onChange={() => setDeleteOriginals(true)}
          />
          <span>Replace them  ⚠ originals are gone after this</span>
        </label>
      </div>

      {/* ── 📋 What will happen ── */}
      <div className="panel source-card accent-card">
        <h2>📋 What will happen</h2>
        <p style={{ color: '#f0f4f8', lineHeight: 1.5, margin: 0 }}>
          {prepMode === 'Auto Prep (Face Crops)'
            ? `Your images → resized to about ${prepMegapixels} MP and saved as PNG, plus one face close-up each.`
            : prepMode === 'Face Crop Only'
            ? 'Your images → replaced by just the cropped face from each photo, saved as PNG.'
            : `Your images → resized to about ${prepMegapixels} MP and saved as PNG.`}
          {' '}Everything lands in your training folder.{' '}
          {deleteOriginals
            ? 'Your original files are replaced ⚠ there is no undo.'
            : "Your originals are moved to the 'originals' subfolder — nothing is deleted."}
        </p>
        {isFaceMode && (
          <p className="explanation">Next: eyeball the face close-ups on the Captions tab and Remove any blurry ones before captioning.</p>
        )}
      </div>

      {/* ── 3 · Run it ── */}
      <div className="panel source-card">
        <h2>3 · Run it</h2>
        <button
          type="button"
          className="primary sparkle-button"
          disabled={isRunning}
          onClick={handleRunPrep}
          style={{ marginBottom: 10 }}
        >
          ✨ Prepare Images Now
        </button>
        <div className="safe-test">
          <span style={{ color: '#8a9bae', fontSize: 11 }}>Want to check first?</span>
          <button
            type="button"
            className="secondary"
            style={{ fontSize: 11, minHeight: 28, padding: '4px 12px' }}
            onClick={() => alert('Test face detection: selects the first image in dataset and checks detected face bounding boxes.')}
          >
            Test face detection on one photo…
          </button>
          <small>optional and safe — shows the crop, writes nothing</small>
        </div>
      </div>

      {/* ── Output Log ── */}
      <div className="panel source-card">
        <h2>Output Log</h2>
        <pre className="output-log">{outputLog || 'Ready — select a prep mode and click Prepare Images Now.'}</pre>
      </div>

      {/* ── Final Step: Look Consistency Filter ── */}
      <div className="panel source-card">
        <h2>Final Step: Look Consistency Filter (faces)</h2>
        <p className="muted">Run this LAST, after every other prep stage — it scores the finished training folder. Pick THREE baseline images that nail the look you want; every image&apos;s face is scored against all three and averaged (ArcFace embedding similarity) — one baseline photo would bake its own angle/expression/lighting bias into every score, three cancel it out. Great for weeding out synthetic images that drifted off-look — the subtle near-misses a loss curve can never see. Click images to mark them, or let Auto-Suggest flag the statistical outliers, then move the marked ones out of the dataset in one go (they go to an &apos;excluded_by_look&apos; subfolder — nothing is deleted). Real-but-unusual low scorers (tight angles, profiles) that you KEEP can ease into training gently — the scan saves its scores with the dataset, and the Training tab&apos;s &apos;Warm up look outliers&apos; toggle (Krea 2) ramps their LR up over the first few epochs instead of letting them fight the forming identity.</p>
        <button
          type="button"
          className="secondary"
          style={{ marginTop: 8 }}
          onClick={() => alert('Look Consistency Filter: scores images against 3 baseline ArcFace embeddings.')}
        >
          🔍 Open Look Filter…
        </button>
      </div>
    </div>
  );
}
