'use client';

import React, { useState, useEffect, useRef } from 'react';

interface FolderPickerModalProps {
  isOpen: boolean;
  initialPath?: string;
  onSelect: (selectedPath: string) => void;
  onClose: () => void;
  title?: string;
}

interface DirectoryEntry {
  name: string;
  path: string;
}

interface BrowseData {
  currentPath: string;
  parentPath: string | null;
  directories: DirectoryEntry[];
  quickPaths: { label: string; path: string }[];
}

export default function FolderPickerModal({
  isOpen,
  initialPath = '',
  onSelect,
  onClose,
  title = 'Select Training Image Folder',
}: FolderPickerModalProps) {
  const [currentPath, setCurrentPath] = useState(initialPath);
  const [inputPath, setInputPath] = useState(initialPath);
  const [browseData, setBrowseData] = useState<BrowseData | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);

  const fetchDirectory = async (targetPath: string) => {
    setLoading(true);
    setError(null);
    try {
      const res = await fetch(`/api/browse?path=${encodeURIComponent(targetPath || '')}`);
      const data = await res.json();
      if (data.success) {
        setBrowseData(data);
        setCurrentPath(data.currentPath);
        setInputPath(data.currentPath);
      } else {
        setError(data.error || 'Failed to browse directory');
      }
    } catch (err: any) {
      setError(err.message || 'Network error fetching directory');
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    if (isOpen) {
      fetchDirectory(initialPath || '');
    }
  }, [isOpen, initialPath]);

  if (!isOpen) return null;

  const handleSelect = (pathToUse: string) => {
    onSelect(pathToUse);
    onClose();
  };

  const handleHtmlFolderSelect = (e: React.ChangeEvent<HTMLInputElement>) => {
    const files = e.target.files;
    if (files && files.length > 0) {
      // In Chromium/WebKit, webkitRelativePath gives "foldername/subfile.jpg"
      const relative = files[0].webkitRelativePath;
      const folderName = relative.split('/')[0];
      if (folderName) {
        // If we are currently browsing a directory, join it, otherwise use folderName
        const resolved = browseData ? `${browseData.currentPath}/${folderName}` : folderName;
        handleSelect(resolved);
      }
    }
  };

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/70 p-4 backdrop-blur-xs">
      <div className="flex flex-col w-full max-w-2xl max-h-[85vh] rounded-lg border border-[#4b5868] bg-[#1d252f] text-[#f1f3f5] shadow-2xl overflow-hidden">
        {/* Header */}
        <div className="flex items-center justify-between border-b border-[#3a4555] bg-[#18212a] px-4 py-3">
          <div className="flex items-center gap-2">
            <span className="text-lg">📁</span>
            <h3 className="text-sm font-bold text-white">{title}</h3>
          </div>
          <button
            type="button"
            onClick={onClose}
            className="text-[#9caaba] hover:text-white text-base px-2 py-0.5 rounded cursor-pointer"
          >
            ✕
          </button>
        </div>

        {/* Path Bar */}
        <div className="p-3 border-b border-[#3a4555] bg-[#202b36] space-y-2">
          <div className="flex gap-2">
            <input
              type="text"
              value={inputPath}
              onChange={(e) => setInputPath(e.target.value)}
              onKeyDown={(e) => {
                if (e.key === 'Enter') {
                  fetchDirectory(inputPath);
                }
              }}
              placeholder="/path/to/folder..."
              className="flex-1 min-h-[34px] px-3 py-1.5 text-xs font-mono rounded border border-[#647284] bg-[#18212b] text-[#f3f5f7] focus:outline-none focus:border-[#3b82f6]"
            />
            <button
              type="button"
              onClick={() => fetchDirectory(inputPath)}
              className="px-3 py-1.5 text-xs font-semibold rounded bg-[#303d4c] hover:bg-[#3d4d60] border border-[#647284] text-white cursor-pointer"
            >
              Go
            </button>
          </div>

          {/* Quick Paths */}
          {browseData?.quickPaths && browseData.quickPaths.length > 0 && (
            <div className="flex flex-wrap items-center gap-1.5 text-xs text-[#8a9bae]">
              <span className="text-[11px] font-medium text-[#718092]">Quick:</span>
              {browseData.quickPaths.map((qp, idx) => (
                <button
                  key={idx}
                  type="button"
                  onClick={() => fetchDirectory(qp.path)}
                  className="px-2 py-0.5 rounded bg-[#18212b] hover:bg-[#2c3947] border border-[#3a4555] text-[11px] text-[#93c5fd] cursor-pointer"
                >
                  {qp.label}
                </button>
              ))}
            </div>
          )}
        </div>

        {/* Directory Listing Body */}
        <div className="flex-1 overflow-y-auto p-2 min-h-[220px] max-h-[380px] bg-[#1d252f] divide-y divide-[#2a3644]">
          {loading && (
            <div className="flex items-center justify-center p-8 text-xs text-[#8a9bae]">
              <span className="animate-spin mr-2">⏳</span> Loading directories...
            </div>
          )}

          {error && (
            <div className="p-3 text-xs text-red-400 bg-red-950/30 rounded border border-red-800 m-2">
              {error}
            </div>
          )}

          {!loading && browseData && (
            <>
              {/* Go Up button */}
              {browseData.parentPath && (
                <button
                  type="button"
                  onClick={() => fetchDirectory(browseData.parentPath!)}
                  className="w-full flex items-center gap-2.5 px-3 py-2 text-xs text-left hover:bg-[#253241] rounded text-[#93c5fd] font-medium cursor-pointer"
                >
                  <span className="text-sm">⬆️</span>
                  <span>.. (Up one level)</span>
                </button>
              )}

              {/* Subdirectories */}
              {browseData.directories.length === 0 ? (
                <div className="p-6 text-center text-xs text-[#718092] italic">
                  No subdirectories found in this folder
                </div>
              ) : (
                browseData.directories.map((dir) => (
                  <div
                    key={dir.path}
                    className="flex items-center justify-between px-3 py-1.5 hover:bg-[#253241] rounded group"
                  >
                    <button
                      type="button"
                      onClick={() => fetchDirectory(dir.path)}
                      className="flex-1 flex items-center gap-2 text-xs text-left text-[#e0e5eb] hover:text-[#93c5fd] font-mono cursor-pointer truncate"
                    >
                      <span className="text-sm">📁</span>
                      <span className="truncate">{dir.name}</span>
                    </button>
                    <button
                      type="button"
                      onClick={() => handleSelect(dir.path)}
                      className="ml-2 px-2 py-0.5 text-[11px] font-semibold rounded bg-[#202b36] hover:bg-[#3b82f6] text-[#8a9bae] hover:text-white border border-[#3a4555] cursor-pointer"
                    >
                      Choose
                    </button>
                  </div>
                ))
              )}
            </>
          )}
        </div>

        {/* Hidden browser file input fallback */}
        <input
          type="file"
          ref={fileInputRef}
          // @ts-ignore
          webkitdirectory="true"
          directory="true"
          className="hidden"
          onChange={handleHtmlFolderSelect}
        />

        {/* Footer */}
        <div className="flex flex-col sm:flex-row items-stretch sm:items-center justify-between gap-2 p-3 border-t border-[#3a4555] bg-[#18212a]">
          <button
            type="button"
            onClick={() => fileInputRef.current?.click()}
            className="px-3 py-1.5 text-xs text-[#9caaba] hover:text-white bg-transparent border border-[#3a4555] rounded cursor-pointer hover:bg-[#253241]"
          >
            📂 Open Native File Dialog
          </button>
          <div className="flex items-center justify-end gap-2">
            <button
              type="button"
              onClick={onClose}
              className="px-3 py-1.5 text-xs rounded bg-[#252f3b] hover:bg-[#303d4c] text-[#c3cdd9] border border-[#4b5868] cursor-pointer"
            >
              Cancel
            </button>
            <button
              type="button"
              disabled={!currentPath}
              onClick={() => handleSelect(currentPath)}
              className="px-4 py-1.5 text-xs font-bold rounded bg-[#3b82f6] hover:bg-[#2563eb] text-white shadow-xs cursor-pointer disabled:opacity-50"
            >
              Select Current Folder
            </button>
          </div>
        </div>
      </div>
    </div>
  );
}
