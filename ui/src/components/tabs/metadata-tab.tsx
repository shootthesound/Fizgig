'use client';

import React, { useState } from 'react';
import FolderPickerModal from '@/components/ui/folder-picker-modal';

interface CustomField {
  key: string;
  value: string;
}

export default function MetadataTab() {
  const [filePath, setFilePath] = useState('');
  const [statusText, setStatusText] = useState('No file loaded.');
  const [isLoaded, setIsLoaded] = useState(false);
  const [isPickerOpen, setIsPickerOpen] = useState(false);

  // Standard fields
  const [title, setTitle] = useState('');
  const [author, setAuthor] = useState('');
  const [license, setLicense] = useState('');
  const [tags, setTags] = useState('');
  const [triggerPhrase, setTriggerPhrase] = useState('');
  const [usageHint, setUsageHint] = useState('');
  const [description, setDescription] = useState('');

  // Thumbnail
  const [thumbnailUri, setThumbnailUri] = useState<string | null>(null);

  // Custom fields
  const [customFields, setCustomFields] = useState<CustomField[]>([]);
  const [selectedKey, setSelectedKey] = useState<string | null>(null);

  // Modal for adding custom field
  const [isAddModalOpen, setIsAddModalOpen] = useState(false);
  const [newKey, setNewKey] = useState('');
  const [newValue, setNewValue] = useState('');

  // Save status
  const [saveStatus, setSaveStatus] = useState('');

  // Real file load
  const handleLoadFile = async (path: string) => {
    if (!path.trim()) return;
    setFilePath(path);
    setStatusText('Loading metadata...');
    try {
      const res = await fetch(`/api/metadata?file=${encodeURIComponent(path)}`);
      const data = await res.json();
      if (res.ok && data.success) {
        setIsLoaded(true);
        const meta = data.metadata || {};
        setTitle(meta['modelspec.title'] || meta.title || '');
        setAuthor(meta['modelspec.author'] || meta.author || '');
        setLicense(meta['modelspec.license'] || meta.license || '');
        setTags(meta['modelspec.tags'] || meta.tags || '');
        setTriggerPhrase(meta['modelspec.trigger_phrase'] || meta.trigger_phrase || '');
        setUsageHint(meta['modelspec.usage_hint'] || meta.usage_hint || '');
        setDescription(meta['modelspec.description'] || meta.description || '');

        if (meta['modelspec.thumbnail']) {
          setThumbnailUri(meta['modelspec.thumbnail']);
        }

        const standardKeys = new Set([
          'modelspec.title', 'title',
          'modelspec.author', 'author',
          'modelspec.license', 'license',
          'modelspec.tags', 'tags',
          'modelspec.trigger_phrase', 'trigger_phrase',
          'modelspec.usage_hint', 'usage_hint',
          'modelspec.description', 'description',
          'modelspec.thumbnail', 'thumbnail',
          '__metadata__'
        ]);

        const custom: CustomField[] = [];
        for (const [k, v] of Object.entries(meta)) {
          if (!standardKeys.has(k)) {
            custom.push({ key: k, value: typeof v === 'string' ? v : JSON.stringify(v) });
          }
        }
        setCustomFields(custom);
        setStatusText(`Loaded — ${Object.keys(meta).length} metadata keys found.`);
        setSaveStatus('');
      } else {
        setStatusText(`Error: ${data.error || 'Failed to load metadata'}`);
      }
    } catch (err: any) {
      setStatusText(`Error: ${err.message || 'Failed to connect to metadata service'}`);
    }
  };

  const handleClearThumbnail = () => {
    setThumbnailUri(null);
  };

  const handleReplaceThumbnail = () => {
    const input = document.createElement('input');
    input.type = 'file';
    input.accept = 'image/*';
    input.onchange = (e: any) => {
      const file = e.target.files?.[0];
      if (file) {
        const reader = new FileReader();
        reader.onload = () => {
          setThumbnailUri(reader.result as string);
        };
        reader.readAsDataURL(file);
      }
    };
    input.click();
  };

  const handleAddCustomField = () => {
    if (!newKey.trim()) return;
    setCustomFields((prev) => {
      const filtered = prev.filter((f) => f.key !== newKey.trim());
      return [...filtered, { key: newKey.trim(), value: newValue.trim() }];
    });
    setNewKey('');
    setNewValue('');
    setIsAddModalOpen(false);
  };

  const handleRemoveCustomField = () => {
    if (!selectedKey) return;
    setCustomFields((prev) => prev.filter((f) => f.key !== selectedKey));
    setSelectedKey(null);
  };

  const handleSave = async (saveAs: boolean) => {
    if (!filePath.trim()) {
      alert('Load a .safetensors file first.');
      return;
    }
    setSaveStatus('Saving metadata...');
    try {
      const metadataPayload: Record<string, string> = {
        'modelspec.title': title,
        'modelspec.author': author,
        'modelspec.license': license,
        'modelspec.tags': tags,
        'modelspec.trigger_phrase': triggerPhrase,
        'modelspec.usage_hint': usageHint,
        'modelspec.description': description,
      };
      if (thumbnailUri) {
        metadataPayload['modelspec.thumbnail'] = thumbnailUri;
      }
      for (const field of customFields) {
        if (field.key.trim()) {
          metadataPayload[field.key.trim()] = field.value;
        }
      }

      let targetPath = filePath;
      if (saveAs) {
        const parts = filePath.replace(/\\/g, '/').split('/');
        const dir = parts.slice(0, -1).join('/');
        const name = parts[parts.length - 1];
        targetPath = dir ? `${dir}/copy_${name}` : `copy_${name}`;
      }

      const res = await fetch('/api/metadata', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          filePath,
          saveAsPath: saveAs ? targetPath : undefined,
          metadata: metadataPayload,
        }),
      });

      const data = await res.json();
      if (res.ok && data.success) {
        setSaveStatus(`Saved successfully (${targetPath})`);
        if (saveAs) {
          setFilePath(targetPath);
        }
      } else {
        setSaveStatus(`Save failed: ${data.error || 'Unknown error'}`);
      }
    } catch (err: any) {
      setSaveStatus(`Save error: ${err.message || 'Network error'}`);
    }
  };

  return (
    <div className="fizgig-tab-page">
      {/* Banner */}
      <div className="tab-banner">
        <h1>Metadata</h1>
        <p>
          View and edit the SAI ModelSpec metadata embedded in a .safetensors file — title, author, description, trigger phrase, thumbnail, and anything else ComfyUI&apos;s model browser reads. Works on any .safetensors file — LoRA, DiT, text encoder, embedding — not just ones Fizgig trained.
        </p>
      </div>

      {/* Card 1: File */}
      <div className="panel source-card">
        <h2>File</h2>
        <p className="muted">
          Pick any .safetensors file — LoRA, DiT, text encoder, embedding — its current metadata loads below.
        </p>
        <div className="start-controls" style={{ marginTop: 8 }}>
          <div className="field" style={{ flex: 1 }}>
            <span>File:</span>
            <input
              type="text"
              value={filePath}
              onChange={(e) => setFilePath(e.target.value)}
              onKeyDown={(e) => {
                if (e.key === 'Enter') handleLoadFile(filePath);
              }}
              placeholder="/workspace/output_loras/my_model.safetensors"
            />
          </div>
          <button
            type="button"
            className="secondary"
            onClick={() => handleLoadFile(filePath)}
            disabled={!filePath.trim()}
          >
            Load
          </button>
          <button
            type="button"
            className="secondary"
            onClick={() => setIsPickerOpen(true)}
          >
            Browse…
          </button>
        </div>
        <small className="inline-note" style={{ marginLeft: 0, marginTop: 4, display: 'block' }}>
          {statusText}
        </small>
      </div>

      {/* Card 2: Standard Fields */}
      <div className="panel source-card">
        <h2>Standard Fields</h2>
        <p className="muted">
          The fields ComfyUI&apos;s model browser (and other spec-aware tools) render specially.
        </p>
        <div className="field-grid" style={{ marginTop: 8 }}>
          <div className="field">
            <span>Title:</span>
            <input type="text" value={title} onChange={(e) => setTitle(e.target.value)} />
          </div>
          <div className="field">
            <span>Author:</span>
            <input type="text" value={author} onChange={(e) => setAuthor(e.target.value)} />
          </div>
          <div className="field">
            <span>License:</span>
            <input type="text" value={license} onChange={(e) => setLicense(e.target.value)} />
          </div>
          <div className="field">
            <span>Tags:</span>
            <input type="text" value={tags} onChange={(e) => setTags(e.target.value)} />
          </div>
          <div className="field">
            <span>Trigger Phrase:</span>
            <input type="text" value={triggerPhrase} onChange={(e) => setTriggerPhrase(e.target.value)} />
          </div>
          <div className="field">
            <span>Usage Hint:</span>
            <input type="text" value={usageHint} onChange={(e) => setUsageHint(e.target.value)} />
          </div>
        </div>

        <div className="field" style={{ marginTop: 12 }}>
          <span>Description:</span>
          <textarea
            className="prompt-editor"
            style={{ minHeight: 90 }}
            value={description}
            onChange={(e) => setDescription(e.target.value)}
          />
        </div>
      </div>

      {/* Card 3: Thumbnail */}
      <div className="panel source-card">
        <h2>Thumbnail</h2>
        <p className="muted">
          The image ComfyUI shows as card art. Fizgig auto-embeds the latest training sample when it trains a LoRA — replace or clear it here for any file.
        </p>

        <div style={{ marginTop: 8, marginBottom: 12 }}>
          {thumbnailUri ? (
            <div
              style={{
                width: 256,
                height: 256,
                border: '1px solid #2a3b50',
                borderRadius: 4,
                overflow: 'hidden',
                backgroundColor: '#121820',
                display: 'flex',
                alignItems: 'center',
                justifyContent: 'center',
              }}
            >
              <img
                src={thumbnailUri}
                alt="Thumbnail"
                style={{ maxWidth: '100%', maxHeight: '100%', objectFit: 'contain' }}
              />
            </div>
          ) : (
            <span className="muted" style={{ fontStyle: 'italic', fontSize: 13 }}>
              (no thumbnail)
            </span>
          )}
        </div>

        <div className="actions" style={{ marginTop: 0 }}>
          <button type="button" className="secondary" onClick={handleReplaceThumbnail}>
            Replace...
          </button>
          <button type="button" className="secondary" onClick={handleClearThumbnail}>
            Clear
          </button>
        </div>
      </div>

      {/* Card 4: Custom Fields */}
      <div className="panel source-card">
        <h2>Custom Fields</h2>
        <p className="muted">
          Like ID3 tags on an MP3 — the format isn&apos;t limited to a fixed list, and a reader just ignores whatever it doesn&apos;t recognize. Add anything you want: author_email, a colorspace profile note, whatever&apos;s useful to you. Not part of the SAI ModelSpec standard, so tools other than Fizgig won&apos;t render these specially, but they&apos;re stored in the file like any other metadata. Also shows any non-standard keys already in the file — nothing gets silently dropped on save.
        </p>

        <div
          style={{
            border: '1px solid #2a3b50',
            borderRadius: 4,
            marginTop: 10,
            overflow: 'hidden',
            backgroundColor: '#141c26',
          }}
        >
          <table style={{ width: '100%', borderCollapse: 'collapse', textAlign: 'left', fontSize: 13 }}>
            <thead>
              <tr style={{ backgroundColor: '#1a2332', borderBottom: '1px solid #2a3b50' }}>
                <th style={{ padding: '8px 12px', width: '35%', color: '#9bb0c4', fontWeight: 600 }}>Key</th>
                <th style={{ padding: '8px 12px', width: '65%', color: '#9bb0c4', fontWeight: 600 }}>Value</th>
              </tr>
            </thead>
            <tbody>
              {customFields.length === 0 ? (
                <tr>
                  <td colSpan={2} style={{ padding: 16, textAlign: 'center', color: '#5a6b7e', fontStyle: 'italic' }}>
                    No custom fields
                  </td>
                </tr>
              ) : (
                customFields.map((field) => (
                  <tr
                    key={field.key}
                    onClick={() => setSelectedKey(field.key)}
                    style={{
                      backgroundColor: selectedKey === field.key ? '#22354a' : 'transparent',
                      cursor: 'pointer',
                      borderBottom: '1px solid #1e2a38',
                    }}
                  >
                    <td style={{ padding: '6px 12px', color: '#cbd5e1', fontFamily: 'monospace' }}>{field.key}</td>
                    <td style={{ padding: '6px 12px', color: '#94a3b8', fontFamily: 'monospace' }}>
                      {field.value.length > 120 ? `${field.value.slice(0, 117)}...` : field.value}
                    </td>
                  </tr>
                ))
              )}
            </tbody>
          </table>
        </div>

        <div className="actions" style={{ marginTop: 12 }}>
          <button type="button" className="secondary" onClick={() => setIsAddModalOpen(true)}>
            Add field...
          </button>
          <button
            type="button"
            className="secondary"
            disabled={!selectedKey}
            onClick={handleRemoveCustomField}
          >
            Remove selected
          </button>
        </div>
      </div>

      {/* Save Row */}
      <div style={{ display: 'flex', alignItems: 'center', gap: 12, paddingBottom: 24 }}>
        <button type="button" className="primary" onClick={() => handleSave(false)}>
          Save
        </button>
        <button type="button" className="secondary" onClick={() => handleSave(true)}>
          Save As...
        </button>
        {saveStatus && (
          <span style={{ fontSize: 13, fontStyle: 'italic', color: '#4ade80', marginLeft: 8 }}>
            {saveStatus}
          </span>
        )}
      </div>

      {/* Add Custom Field Modal */}
      {isAddModalOpen && (
        <div
          style={{
            position: 'fixed',
            inset: 0,
            backgroundColor: 'rgba(0, 0, 0, 0.7)',
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'center',
            zIndex: 9999,
          }}
        >
          <div
            className="panel source-card"
            style={{
              width: 480,
              maxWidth: '90%',
              backgroundColor: '#161f2c',
              border: '1px solid #334458',
              boxShadow: '0 8px 32px rgba(0,0,0,0.5)',
            }}
          >
            <h2 style={{ marginTop: 0 }}>Add custom field</h2>
            <div className="field" style={{ marginTop: 12 }}>
              <span>Key:</span>
              <input
                type="text"
                value={newKey}
                onChange={(e) => setNewKey(e.target.value)}
                placeholder="e.g. author_email or training_notes"
                autoFocus
              />
            </div>
            <div className="field" style={{ marginTop: 12 }}>
              <span>Value:</span>
              <input
                type="text"
                value={newValue}
                onChange={(e) => setNewValue(e.target.value)}
                placeholder="Value text"
              />
            </div>
            <div className="actions" style={{ marginTop: 18, justifyContent: 'flex-end' }}>
              <button
                type="button"
                className="secondary"
                onClick={() => {
                  setIsAddModalOpen(false);
                  setNewKey('');
                  setNewValue('');
                }}
              >
                Cancel
              </button>
              <button type="button" className="primary" onClick={handleAddCustomField}>
                Add
              </button>
            </div>
          </div>
        </div>
      )}

      {/* SafeTensors File Picker Modal */}
      <FolderPickerModal
        isOpen={isPickerOpen}
        mode="file"
        fileFilter=".safetensors"
        title="Select SafeTensors Model File"
        initialPath={filePath || ''}
        onSelect={(p) => handleLoadFile(p)}
        onClose={() => setIsPickerOpen(false)}
      />
    </div>
  );
}
