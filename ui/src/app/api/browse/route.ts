import { NextRequest, NextResponse } from 'next/server';
import fs from 'fs';
import path from 'path';
import os from 'os';

export async function GET(req: NextRequest) {
  try {
    const { searchParams } = new URL(req.url);
    const requestedPath = searchParams.get('path');

    const projectRoot = path.resolve(process.cwd(), '..');
    const userHome = os.homedir();
    const defaultDataset = path.join(projectRoot, 'dataset');

    // Determine initial target directory
    let targetDir = requestedPath ? path.resolve(requestedPath) : projectRoot;

    // Fallback if target doesn't exist or isn't a directory
    if (!fs.existsSync(targetDir) || !fs.statSync(targetDir).isDirectory()) {
      targetDir = fs.existsSync(defaultDataset) ? defaultDataset : projectRoot;
    }

    const parentDir = path.dirname(targetDir);

    // Read entries
    const entries = fs.readdirSync(targetDir, { withFileTypes: true });

    const directories = entries
      .filter((entry) => {
        if (!entry.isDirectory()) return false;
        // Ignore internal build/vcs folders
        if (['.git', '.next', 'node_modules', '__pycache__', '.venv', 'venv'].includes(entry.name)) {
          return false;
        }
        return true;
      })
      .map((entry) => ({
        name: entry.name,
        path: path.join(targetDir, entry.name),
      }))
      .sort((a, b) => a.name.localeCompare(b.name));

    // Common quick navigation paths
    const quickPaths = [
      { label: 'Project Root', path: projectRoot },
      { label: 'Dataset Folder', path: defaultDataset },
      { label: 'Home Folder', path: userHome },
    ].filter((p) => fs.existsSync(p.path));

    return NextResponse.json({
      success: true,
      currentPath: targetDir,
      parentPath: parentDir !== targetDir ? parentDir : null,
      directories,
      quickPaths,
    });
  } catch (error: any) {
    return NextResponse.json(
      { success: false, error: error.message || 'Failed to read directory' },
      { status: 500 }
    );
  }
}
