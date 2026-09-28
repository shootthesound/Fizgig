import { NextRequest, NextResponse } from 'next/server';
import fs from 'fs';
import path from 'path';
import os from 'os';
import { validateApiAuth } from '@/lib/auth';

/**
 * Returns canonical approved roots where directory browsing is permitted.
 */
function getApprovedRoots(): string[] {
  const projectRoot = path.resolve(process.cwd(), '..');
  const userHome = os.homedir();
  const roots: string[] = [projectRoot, userHome];

  // Common cloud container volumes (RunPod, Vast.ai, etc.)
  const cloudRoots = ['/workspace', '/data', '/content', '/notebooks'];
  for (const cr of cloudRoots) {
    if (fs.existsSync(cr)) {
      roots.push(cr);
    }
  }

  // Optional user-specified allowed roots via environment
  if (process.env.FIZGIG_ALLOWED_ROOTS) {
    const custom = process.env.FIZGIG_ALLOWED_ROOTS.split(',')
      .map((r) => r.trim())
      .filter(Boolean);
    for (const r of custom) {
      if (fs.existsSync(r)) {
        roots.push(path.resolve(r));
      }
    }
  }

  // Deduplicate and canonicalize with realpath
  const canonical = roots.map((r) => {
    try {
      return fs.realpathSync(r);
    } catch {
      return path.resolve(r);
    }
  });

  return Array.from(new Set(canonical));
}

/**
 * Ensures target path is strictly contained within at least one approved root.
 */
function isPathWithinApprovedRoots(targetPath: string, approvedRoots: string[]): boolean {
  try {
    const resolved = fs.existsSync(targetPath)
      ? fs.realpathSync(targetPath)
      : path.resolve(targetPath);

    return approvedRoots.some((root) => {
      const rel = path.relative(root, resolved);
      return !rel.startsWith('..') && !path.isAbsolute(rel);
    });
  } catch {
    return false;
  }
}

const SENSITIVE_DIRS = new Set([
  '.git',
  '.next',
  'node_modules',
  '__pycache__',
  '.venv',
  'venv',
  '.ssh',
  '.gnupg',
  '.aws',
  '.config',
  '.cache',
  '.local',
]);

export async function GET(req: NextRequest) {
  // 1. Authenticate request
  const auth = validateApiAuth(req);
  if (!auth.authorized) {
    return NextResponse.json(
      { success: false, error: auth.reason || 'Unauthorized' },
      { status: 401 }
    );
  }

  try {
    const { searchParams } = new URL(req.url);
    const requestedPath = searchParams.get('path');

    const approvedRoots = getApprovedRoots();
    const projectRoot = approvedRoots[0] || path.resolve(process.cwd(), '..');
    const defaultDataset = path.join(projectRoot, 'dataset');

    // Determine target directory
    let targetDir = requestedPath ? path.resolve(requestedPath) : projectRoot;

    // Fallback if target does not exist or is not a directory
    if (!fs.existsSync(targetDir) || !fs.statSync(targetDir).isDirectory()) {
      targetDir = fs.existsSync(defaultDataset) ? defaultDataset : projectRoot;
    }

    // 2. Security constraint: Verify target directory is within approved roots
    if (!isPathWithinApprovedRoots(targetDir, approvedRoots)) {
      return NextResponse.json(
        {
          success: false,
          error: 'Access denied: Directory browsing is restricted to project, user home, and approved roots.',
        },
        { status: 403 }
      );
    }

    // 3. Parent directory navigation (only allowed if parent remains within approved roots)
    const parentCandidate = path.dirname(targetDir);
    const parentPath =
      parentCandidate !== targetDir && isPathWithinApprovedRoots(parentCandidate, approvedRoots)
        ? parentCandidate
        : null;

    // 4. Read directory entries
    const entries = fs.readdirSync(targetDir, { withFileTypes: true });

    const directories = entries
      .filter((entry) => {
        if (!entry.isDirectory()) return false;
        if (entry.name.startsWith('.') || SENSITIVE_DIRS.has(entry.name)) {
          return false;
        }
        return true;
      })
      .map((entry) => ({
        name: entry.name,
        path: path.join(targetDir, entry.name),
      }))
      .sort((a, b) => a.name.localeCompare(b.name));

    const includeFiles = searchParams.get('files') === 'true' || Boolean(searchParams.get('ext'));
    const extFilter = searchParams.get('ext')?.toLowerCase();
    const files = includeFiles
      ? entries
          .filter((entry) => {
            if (!entry.isFile()) return false;
            if (entry.name.startsWith('.')) return false;
            if (extFilter && !entry.name.toLowerCase().endsWith(extFilter)) return false;
            return true;
          })
          .map((entry) => ({
            name: entry.name,
            path: path.join(targetDir, entry.name),
          }))
          .sort((a, b) => a.name.localeCompare(b.name))
      : [];

    // 5. Approved quick paths
    const quickPaths = [
      { label: 'Project Root', path: projectRoot },
      { label: 'Dataset Folder', path: defaultDataset },
      { label: 'Home Folder', path: os.homedir() },
    ].filter((p) => fs.existsSync(p.path) && isPathWithinApprovedRoots(p.path, approvedRoots));

    return NextResponse.json({
      success: true,
      currentPath: targetDir,
      parentPath,
      directories,
      files,
      quickPaths,
    });
  } catch (error: any) {
    return NextResponse.json(
      { success: false, error: error.message || 'Failed to read directory' },
      { status: 500 }
    );
  }
}
