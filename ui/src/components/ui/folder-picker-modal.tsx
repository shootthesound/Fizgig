'use client';

import React, { useState, useEffect, useRef } from 'react';

interface FolderPickerModalProps {
  isOpen: boolean;
  initialPath?: string;
  onSelect: (selectedPath: string) => void;
  onClose: () => void;
  title?: string;
  mode?: 'folder' | 'file';
  fileFilter?: string;
}

interface DirectoryEntry {
  name: string;
  path: string;
}

interface FileEntry {
  name: string;
  path: string;
}

interface BrowseData {
  currentPath: string;
  parentPath: string | null;
  directories: DirectoryEntry[];
  files?: FileEntry[];
  quickPaths: { label: string; path: string }[];
}

export default function FolderPickerModal({
  isOpen,
  initialPath = '',
  onSelect,
  onClose,
  title = 'Select Training Image Folder',
  mode = 'folder',
  fileFilter,
}: FolderPickerModalProps) {
  const [currentPath, setCurrentPath] = useState(initialPath);
  const [inputPath, setInputPath] = useState(initialPath);
  const [selectedFile, setSelectedFile] = useState<string | null>(null);
  const [browseData, setBrowseData] = useState<BrowseData | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);

  const fetchDirectory = async (targetPath: string) => {
    setLoading(true);
    setError(null);
    try {
      const extParam = fileFilter ? `&ext=${encodeURIComponent(fileFilter)}` : '';
      const filesParam = mode === 'file' ? '&files=true' : '';
      const res = await fetch(`/api/browse?path=${encodeURIComponent(targetPath || '')}${filesParam}${extParam}`);
      const data = await res.json();
      if (res.ok && data.success) {
        setBrowseData(data);
        setCurrentPath(data.currentPath);
        setInputPath(data.currentPath);
      } else {
        setError(data.error || `HTTP ${res.status}: Failed to browse directory`);
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

  const handleHtmlFileOrFolderSelect = (e: React.ChangeEvent<HTMLInputElement>) => {
    const files = e.target.files;
    if (files && files.length > 0) {
      if (mode === 'file') {
        const file = files[0];
        const resolved = browseData ? `${browseData.currentPath}/${file.name}` : file.name;
        handleSelect(resolved);
      } else {
        const relative = files[0].webkitRelativePath;
        const folderName = relative ? relative.split('/')[0] : files[0].name;
        if (folderName) {
          const resolved = browseData ? `${browseData.currentPath}/${folderName}` : folderName;
          handleSelect(resolved);
        }
      }
    }
  };

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/70 p-4 backdrop-blur-xs">
      <div className="flex flex-col w-full max-w-2xl max-h-[85vh] rounded-lg border border-[#4b5868] bg-[#1d252f] text-[#f1f3f5] shadow-2xl overflow-hidden">
        {/* Header */}
        <div className="flex items-center justify-between border-b border-[#3a4555] bg-[#18212a] px-4 py-3">
          <div className="flex items-center gap-2">
            <span className="text-lg">{mode === 'file' ? '📄' : '📁'}</span>
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

        {/* Quick paths banner */}
        {browseData?.quickPaths && browseData.quickPaths.length > 0 && (
          <div className="flex flex-wrap items-center gap-1.5 px-4 py-2 border-b border-[#2d3748] bg-[#161c24] text-xs">
            <span className="text-[#8a9bae] font-medium mr-1">Quick:</span>
            {browseData.quickPaths.map((qp) => (
              <button
                key={qp.path}
                type="button"
                onClick={() => fetchDirectory(qp.path)}
                className="px-2 py-0.5 rounded bg-[#202936] hover:bg-[#2b3749] text-[#93c5fd] border border-[#3b4758] text-[11px] cursor-pointer transition-colors"
              >
                {qp.label}
              </button>
            ))}
          </div>
        )}

        {/* Current Path Bar */}
        <div className="p-3 border-b border-[#2d3748] bg-[#1b232c]">
          <form
            onSubmit={(e) => {
              e.preventDefault();
              fetchDirectory(inputPath);
            }}
            className="flex items-center gap-2"
          >
            <input
              type="text"
              value={inputPath}
              onChange={(e) => setInputPath(e.target.value)}
              placeholder="Enter path (/path/to/...)"
              className="flex-1 px-3 py-1.5 text-xs bg-[#10151c] text-[#f1f3f5] border border-[#3d4b5c] rounded focus:outline-none focus:border-[#3b82f6] font-mono"
            />
            <button
              type="submit"
              className="px-3 py-1.5 text-xs font-semibold rounded bg-[#2b3749] hover:bg-[#37465d] text-white border border-[#48576b] cursor-pointer"
            >
              Go
            </button>
          </form>
        </div>

        {/* Directory & File Listing Container */}
        <div className="flex-1 overflow-y-auto p-2 space-y-0.5 min-h-[220px]">
          {loading && (
            <div className="p-8 text-center text-xs text-[#8a9bae]">
              <span className="animate-spin mr-2">⏳</span> Loading...
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
              {browseData.directories.map((dir) => (
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
                  {mode === 'folder' && (
                    <button
                      type="button"
                      onClick={() => handleSelect(dir.path)}
                      className="ml-2 px-2 py-0.5 text-[11px] font-semibold rounded bg-[#202b36] hover:bg-[#3b82f6] text-[#8a9bae] hover:text-white border border-[#3a4555] cursor-pointer"
                    >
                      Choose
                    </button>
                  )}
                </div>
              ))}

              {/* Files if mode === 'file' */}
              {mode === 'file' && browseData.files && browseData.files.length > 0 && (
                <div className="pt-2 border-t border-[#2d3748] mt-2">
                  <div className="px-3 py-1 text-[11px] font-semibold text-[#8a9bae] uppercase tracking-wider">
                    Files ({browseData.files.length})
                  </div>
                  {browseData.files.map((file) => (
                    <div
                      key={file.path}
                      onClick={() => setSelectedFile(file.path)}
                      className={`flex items-center justify-between px-3 py-1.5 rounded cursor-pointer ${
                        selectedFile === file.path ? 'bg-[#1e3a5f] border border-[#3b82f6]' : 'hover:bg-[#253241]'
                      }`}
                    >
                      <div className="flex-1 flex items-center gap-2 text-xs text-left text-[#f0f4f8] font-mono truncate">
                        <span className="text-sm">📄</span>
                        <span className="truncate">{file.name}</span>
                      </div>
                      <button
                        type="button"
                        onClick={(e) => {
                          e.stopPropagation();
                          handleSelect(file.path);
                        }}
                        className="ml-2 px-2 py-0.5 text-[11px] font-semibold rounded bg-[#202b36] hover:bg-[#3b82f6] text-[#8a9bae] hover:text-white border border-[#3a4555] cursor-pointer"
                      >
                        Select
                      </button>
                    </div>
                  ))}
                </div>
              )}

              {browseData.directories.length === 0 && (!browseData.files || browseData.files.length === 0) && (
                <div className="p-6 text-center text-xs text-[#718092] italic">
                  No items found in this folder
                </div>
              )}
            </>
          )}
        </div>

        {/* Hidden browser file/folder input fallback */}
        <input
          type="file"
          ref={fileInputRef}
          {...(mode === 'folder'
            ? {
                // @ts-ignore
                webkitdirectory: 'true',
                directory: 'true',
              }
            : {
                accept: fileFilter || undefined,
              })}
          className="hidden"
          onChange={handleHtmlFileOrFolderSelect}
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
            {mode === 'file' ? (
              <button
                type="button"
                disabled={!selectedFile}
                onClick={() => selectedFile && handleSelect(selectedFile)}
                className="px-4 py-1.5 text-xs font-bold rounded bg-[#3b82f6] hover:bg-[#2563eb] text-white shadow-xs cursor-pointer disabled:opacity-50"
              >
                Select File
              </button>
            ) : (
              <button
                type="button"
                disabled={!currentPath}
                onClick={() => handleSelect(currentPath)}
                className="px-4 py-1.5 text-xs font-bold rounded bg-[#3b82f6] hover:bg-[#2563eb] text-white shadow-xs cursor-pointer disabled:opacity-50"
              >
                Select Current Folder
              </button>
            )}
          </div>
        </div>
      </div>
    </div>
  );
}
