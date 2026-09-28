'use client';

import React, { useState, useEffect } from 'react';
import { useSettingsStore } from '@/store/settings-store';

/* Caption model options — exact from jj.py lines 1074-1098 */
const FLORENCE_MODELS = [
  'MiaoshouAI/Florence-2-base-PromptGen',
  'microsoft/Florence-2-base',
  'microsoft/Florence-2-large',
];
const QWEN_CAPTION_MODEL = 'Qwen3-VL 4B (Krea 2 text encoder)';
const FLORENCE_TASKS = ['<CAPTION>', '<DETAILED_CAPTION>', '<MORE_DETAILED_CAPTION>'];
const QWEN_TASKS = ['Training — likeness + garment + scene', 'Detailed — long-form scene', 'Concise — trigger + key traits', 'Custom…'];

interface CaptionItem {
  fileName: string;
  filePath: string;
  caption: string;
  hasCaption: boolean;
}

export default function CaptionsTab() {
  const { state, dispatch } = useSettingsStore();

  const [captionModel, setCaptionModel] = useState(FLORENCE_MODELS[0]);
  const [captionTask, setCaptionTask] = useState('<DETAILED_CAPTION>');
  const [maxTokens, setMaxTokens] = useState('256');
  const [overwriteCaptions, setOverwriteCaptions] = useState(state.overwrite_captions);
  const [skipBilingual, setSkipBilingual] = useState(state.skip_bilingual);
  const [findText, setFindText] = useState('');
  const [replaceText, setReplaceText] = useState('');
  const [captionLog, setCaptionLog] = useState('');

  // Image Preview & Pagination
  const [images, setImages] = useState<CaptionItem[]>([]);
  const [currentPage, setCurrentPage] = useState(1);
  const [totalPages, setTotalPages] = useState(1);
  const [totalImages, setTotalImages] = useState(0);
  const [loadingImages, setLoadingImages] = useState(false);
  const [editingCaption, setEditingCaption] = useState<{ [key: string]: string }>({});

  const isQwen = captionModel === QWEN_CAPTION_MODEL;
  const taskOptions = isQwen ? QWEN_TASKS : FLORENCE_TASKS;
  const allModels = [...FLORENCE_MODELS, QWEN_CAPTION_MODEL];

  // Fetch images from current training folder
  const loadImages = async (page = 1) => {
    if (!state.image_folder) return;
    setLoadingImages(true);
    try {
      const res = await fetch(`/api/captions?folder=${encodeURIComponent(state.image_folder)}&page=${page}&pageSize=8`);
      const data = await res.json();
      if (data && data.images) {
        setImages(data.images);
        setCurrentPage(data.page);
        setTotalPages(data.totalPages);
        setTotalImages(data.total);

        const initialCaptions: { [key: string]: string } = {};
        data.images.forEach((img: CaptionItem) => {
          initialCaptions[img.fileName] = img.caption;
        });
        setEditingCaption(initialCaptions);
      }
    } catch (e: any) {
      setCaptionLog(prev => `${prev}\n[error] Failed to load images: ${e?.message}`);
    } finally {
      setLoadingImages(false);
    }
  };

  useEffect(() => {
    if (state.image_folder) {
      loadImages(1);
    }
  }, [state.image_folder]);

  // Actions
  const handleFindReplace = async () => {
    if (!state.image_folder) {
      alert('Please select an image folder on the Start tab first.');
      return;
    }
    if (!findText) {
      alert('Enter text to find.');
      return;
    }
    try {
      const res = await fetch('/api/captions', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          action: 'find_replace',
          folder: state.image_folder,
          find: findText,
          replace: replaceText,
        }),
      });
      const data = await res.json();
      if (data.success) {
        setCaptionLog(prev => `${prev}\n[find_replace] Modified ${data.modified} of ${data.total} .txt files.`);
        loadImages(currentPage);
      }
    } catch (e: any) {
      alert('Error during find & replace: ' + e?.message);
    }
  };

  const handlePreviewReplace = async () => {
    if (!state.image_folder || !findText) return;
    try {
      const res = await fetch('/api/captions', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          action: 'preview_replace',
          folder: state.image_folder,
          find: findText,
          replace: replaceText,
        }),
      });
      const data = await res.json();
      if (data.success) {
        setCaptionLog(prev => `${prev}\n[preview] Found ${data.count} matching files. Showing up to 10 changes.`);
      }
    } catch (e: any) {
      alert('Preview error: ' + e?.message);
    }
  };

  const handleStaticCaptionAll = async () => {
    if (!state.image_folder) {
      alert('Please select an image folder on the Start tab first.');
      return;
    }
    if (!confirm(`Apply static trigger word "${state.caption_text}" to all images in ${state.image_folder}?`)) return;
    try {
      const res = await fetch('/api/captions', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          action: 'static_caption',
          folder: state.image_folder,
          triggerWord: state.caption_text,
          overwrite: overwriteCaptions,
        }),
      });
      const data = await res.json();
      if (data.success) {
        setCaptionLog(prev => `${prev}\n[static_caption] Written static caption to ${data.written} files.`);
        loadImages(currentPage);
      }
    } catch (e: any) {
      alert('Static captioning failed: ' + e?.message);
    }
  };

  const handleSaveSingleCaption = async (fileName: string) => {
    if (!state.image_folder) return;
    const caption = editingCaption[fileName] || '';
    try {
      const res = await fetch('/api/captions', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          action: 'update_single',
          folder: state.image_folder,
          fileName,
          caption,
        }),
      });
      const data = await res.json();
      if (data.success) {
        setCaptionLog(prev => `${prev}\n[saved] Saved caption for ${fileName}`);
        loadImages(currentPage);
      }
    } catch (e: any) {
      alert('Failed to save caption: ' + e?.message);
    }
  };

  const handleBilingualTranslate = async () => {
    if (!state.image_folder) return;
    try {
      const res = await fetch('/api/captions', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          action: 'bilingual_translate',
          folder: state.image_folder,
          skipIfChinese: skipBilingual,
        }),
      });
      const data = await res.json();
      if (data.success) {
        setCaptionLog(prev => `${prev}\n[translate] Processed ${data.translated} files for bilingual alignment.`);
      }
    } catch (e: any) {
      alert('Translation error: ' + e?.message);
    }
  };

  return (
    <div className="fizgig-tab-page">
      {/* ── Banner ── */}
      <div className="tab-banner">
        <h1>Captions</h1>
        <p>Write trigger-word captions or generate them with AI. You can optionally skip this tab if your images already have .txt caption files.</p>
      </div>

      {/* ── Captioning Settings ── */}
      <div className="panel source-card">
        <h2>Captioning Settings</h2>
        <p className="muted">Trigger word is prepended to every caption. Qwen3-VL follows a captioning instruction you can read and edit, and is the better fit for training data — it needs the Krea 2 text encoder set in Preferences, and captions any dataset, Klein included. Florence-2 is smaller and downloads itself on first use. Static Caption writes the trigger word only.</p>

        <div className="field-grid" style={{ marginTop: 12 }}>
          {/* Image Folder */}
          <div className="field">
            <span>Image Folder:</span>
            <div style={{ display: 'flex', gap: 6, alignItems: 'center' }}>
              <span style={{ color: '#8a9bae', fontSize: 13, fontWeight: 500 }}>{state.image_folder || '(not set)'}</span>
            </div>
            <small>(set on the Start tab)</small>
          </div>

          {/* Trigger Word */}
          <div className="field">
            <span>Trigger Word:</span>
            <div style={{ display: 'flex', gap: 10, alignItems: 'center' }}>
              <input
                type="text"
                value={state.caption_text}
                onChange={(e) => dispatch({ type: 'SET_CAPTION_TRIGGER', payload: e.target.value })}
                placeholder=""
              />
              <small style={{ minHeight: 0 }}>(prepended to all captions)</small>
            </div>
          </div>

          {/* Model */}
          <div className="field">
            <span>Model:</span>
            <div style={{ display: 'flex', gap: 8, alignItems: 'center' }}>
              <select
                value={captionModel}
                onChange={(e) => {
                  setCaptionModel(e.target.value);
                  setCaptionTask(e.target.value === QWEN_CAPTION_MODEL ? QWEN_TASKS[0] : '<DETAILED_CAPTION>');
                }}
              >
                {allModels.map(m => <option key={m} value={m}>{m}</option>)}
              </select>
              {isQwen && <small style={{ minHeight: 0, color: '#5a6b7e' }}>uses your Krea 2 text encoder</small>}
            </div>
          </div>

          {/* Task */}
          <div className="field">
            <span>Task:</span>
            <div style={{ display: 'flex', gap: 8, alignItems: 'center' }}>
              <select value={captionTask} onChange={(e) => setCaptionTask(e.target.value)}>
                {taskOptions.map(t => <option key={t} value={t}>{t}</option>)}
              </select>
              {isQwen && (
                <button type="button" className="secondary" style={{ fontSize: 11, minHeight: 26, padding: '4px 10px' }}>
                  Edit instructions…
                </button>
              )}
            </div>
          </div>

          {/* Max Tokens */}
          <div className="field">
            <span>Max Tokens:</span>
            <input
              type="text"
              value={maxTokens}
              onChange={(e) => setMaxTokens(e.target.value)}
              style={{ width: 80 }}
            />
          </div>
        </div>

        {/* Overwrite existing */}
        <label className="check-line" style={{ marginTop: 12 }}>
          <input
            type="checkbox"
            checked={overwriteCaptions}
            onChange={(e) => {
              setOverwriteCaptions(e.target.checked);
              dispatch({ type: 'UPDATE_STATE', payload: { overwrite_captions: e.target.checked } });
            }}
          />
          <span>Overwrite existing caption files</span>
        </label>
        <p className="explanation">Untick to caption ONLY images that don&apos;t have a .txt yet — e.g. after adding new images or face-cropping, existing captions stay untouched.</p>
      </div>

      {/* ── Generate Captions ── */}
      <div className="panel source-card">
        <h2>Generate Captions</h2>
        <div className="actions">
          <button
            type="button"
            className="primary"
            onClick={async () => {
              if (!state.image_folder) return alert('Select image folder first');
              const res = await fetch('/api/captions', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ action: 'ai_caption', folder: state.image_folder }),
              });
              const data = await res.json();
              setCaptionLog(prev => `${prev}\n[ai_caption] ${data.message} (${data.count} items)`);
            }}
          >
            Caption All Images (AI)
          </button>
          <button type="button" className="secondary" onClick={handleStaticCaptionAll}>
            Static Caption All
          </button>
          <button type="button" className="secondary" disabled>
            Stop
          </button>
          <button type="button" className="secondary" onClick={() => setCaptionLog(prev => `${prev}\n[info] Model memory unloaded.`)}>
            Unload Model
          </button>
        </div>
      </div>

      {/* ── Bilingual Translation ── */}
      <div className="panel source-card">
        <h2>Bilingual Translation (English + Chinese)</h2>
        <p className="muted">Translates each English caption to Chinese via Helsinki-NLP/opus-mt-en-zh (~300MB, auto-downloaded on first use) and appends as `english - chinese`. Trigger word preserved if it&apos;s the first comma-separated token. Hypothesis: dual-language signal may improve LoRA convergence — empirical test needed.</p>
        <div style={{ display: 'flex', gap: 16, alignItems: 'center', marginTop: 8 }}>
          <label className="check-line">
            <input
              type="checkbox"
              checked={skipBilingual}
              onChange={(e) => {
                setSkipBilingual(e.target.checked);
                dispatch({ type: 'UPDATE_STATE', payload: { skip_bilingual: e.target.checked } });
              }}
            />
            <span>Skip files that already contain Chinese</span>
          </label>
          <button type="button" className="secondary" onClick={handleBilingualTranslate}>
            Translate Captions in Folder
          </button>
        </div>
      </div>

      {/* ── Find & Replace ── */}
      <div className="panel source-card">
        <h2>Find &amp; Replace</h2>
        <p className="muted">Bulk-edit every `.txt` caption file in the image folder. Preview first to see which files change.</p>
        <div className="field-grid" style={{ marginTop: 8 }}>
          <div className="field">
            <span>Find:</span>
            <input type="text" value={findText} onChange={(e) => setFindText(e.target.value)} />
          </div>
          <div className="field">
            <span>Replace:</span>
            <input type="text" value={replaceText} onChange={(e) => setReplaceText(e.target.value)} />
          </div>
        </div>
        <div className="actions" style={{ marginTop: 8 }}>
          <button type="button" className="secondary" onClick={handleFindReplace}>
            Replace in All .txt Files
          </button>
          <button type="button" className="secondary" onClick={handlePreviewReplace}>
            Preview Changes
          </button>
        </div>
      </div>

      {/* ── Image Preview ── */}
      <div className="panel source-card">
        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
          <h2>Image Preview ({totalImages} images in folder)</h2>
          <button
            type="button"
            className="secondary"
            style={{ fontSize: 11, minHeight: 28, padding: '4px 10px' }}
            onClick={() => loadImages(currentPage)}
          >
            Refresh
          </button>
        </div>
        <p className="muted">Browse the training folder and inspect or edit individual captions.</p>

        {/* Image grid */}
        <div style={{ marginTop: 12, minHeight: 180 }}>
          {loadingImages ? (
            <p style={{ color: '#8a9bae', textAlign: 'center', paddingTop: 40 }}>Scanning folder...</p>
          ) : images.length === 0 ? (
            <p style={{ color: '#5a6b7e', textAlign: 'center', paddingTop: 40 }}>
              {state.image_folder ? 'No supported images found in this folder.' : 'Set a training folder on the Start tab, then click Refresh.'}
            </p>
          ) : (
            <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fill, minmax(280px, 1fr))', gap: 12 }}>
              {images.map((img) => (
                <div
                  key={img.fileName}
                  style={{
                    background: '#18212b',
                    border: '1px solid #334155',
                    borderRadius: 4,
                    padding: 10,
                    display: 'flex',
                    flexDirection: 'column',
                    gap: 8,
                  }}
                >
                  <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                    <span style={{ fontSize: 12, fontWeight: 600, color: '#cbd5e1', overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>
                      {img.fileName}
                    </span>
                    <span style={{ fontSize: 11, color: img.hasCaption ? '#10b981' : '#f59e0b' }}>
                      {img.hasCaption ? '✓ captioned' : 'missing .txt'}
                    </span>
                  </div>
                  <textarea
                    rows={3}
                    value={editingCaption[img.fileName] ?? img.caption}
                    onChange={(e) => setEditingCaption({ ...editingCaption, [img.fileName]: e.target.value })}
                    style={{
                      width: '100%',
                      fontSize: 12,
                      background: '#10161f',
                      border: '1px solid #475569',
                      color: '#f8fafc',
                      padding: 6,
                      borderRadius: 2,
                    }}
                  />
                  <div style={{ display: 'flex', justifyContent: 'flex-end', gap: 6 }}>
                    <button
                      type="button"
                      className="secondary small"
                      onClick={() => handleSaveSingleCaption(img.fileName)}
                    >
                      Save Caption
                    </button>
                  </div>
                </div>
              ))}
            </div>
          )}
        </div>

        {/* Pagination */}
        {totalPages > 1 && (
          <div style={{ display: 'flex', gap: 8, alignItems: 'center', marginTop: 14 }}>
            <button
              type="button"
              className="secondary"
              disabled={currentPage <= 1}
              onClick={() => loadImages(currentPage - 1)}
              style={{ fontSize: 11, minHeight: 28, padding: '4px 10px' }}
            >
              &lt;&lt; Prev
            </button>
            <span style={{ color: '#8a9bae', fontSize: 13 }}>
              Page {currentPage} of {totalPages}
            </span>
            <button
              type="button"
              className="secondary"
              disabled={currentPage >= totalPages}
              onClick={() => loadImages(currentPage + 1)}
              style={{ fontSize: 11, minHeight: 28, padding: '4px 10px' }}
            >
              Next &gt;&gt;
            </button>
          </div>
        )}
      </div>

      {/* ── Progress ── */}
      <div className="panel source-card">
        <h2>Progress</h2>
        <div style={{ display: 'flex', gap: 12, alignItems: 'center' }}>
          <div style={{ flex: '0 0 300px', height: 20, background: '#18212b', border: '1px solid #3a4555', borderRadius: 2, overflow: 'hidden' }}>
            <div style={{ width: '0%', height: '100%', background: '#3b82f6', transition: 'width 0.3s' }} />
          </div>
          <span style={{ color: '#8a9bae', fontSize: 13 }}>Ready</span>
        </div>
      </div>

      {/* ── Output Log ── */}
      <div className="panel source-card">
        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 6 }}>
          <h2 style={{ margin: 0 }}>Output Log</h2>
          <button type="button" className="secondary small" onClick={() => setCaptionLog('')}>Clear</button>
        </div>
        <pre className="output-log">{captionLog || 'Ready — configure settings above and click Caption All Images.'}</pre>
      </div>
    </div>
  );
}
