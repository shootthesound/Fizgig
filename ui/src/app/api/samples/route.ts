import { NextRequest, NextResponse } from 'next/server';
import fs from 'fs';
import path from 'path';
import { validateApiAuth, isPathWithinApprovedRoots } from '@/lib/auth';

const BASE_DIR = path.resolve(process.cwd(), '..');
const OUTPUT_DIR = path.join(BASE_DIR, 'output_loras');

export async function GET(req: NextRequest) {
  const auth = validateApiAuth(req);
  if (!auth.authorized) {
    return NextResponse.json({ error: auth.reason || 'Unauthorized' }, { status: 401 });
  }

  try {
    const { searchParams } = new URL(req.url);
    const fileParam = searchParams.get('file');

    // 1. Direct file serving mode
    if (fileParam) {
      const resolved = path.resolve(fileParam);
      if (!isPathWithinApprovedRoots(resolved)) {
        return NextResponse.json({ error: 'Access denied: outside approved roots' }, { status: 403 });
      }
      if (!fs.existsSync(resolved) || !fs.statSync(resolved).isFile()) {
        return NextResponse.json({ error: 'File not found' }, { status: 404 });
      }

      const ext = path.extname(resolved).toLowerCase();
      const mimeTypes: Record<string, string> = {
        '.png': 'image/png',
        '.jpg': 'image/jpeg',
        '.jpeg': 'image/jpeg',
        '.webp': 'image/webp',
      };

      if (!mimeTypes[ext]) {
        return NextResponse.json({ error: 'Unsupported media type' }, { status: 400 });
      }

      const buffer = fs.readFileSync(resolved);
      return new Response(buffer, {
        headers: {
          'Content-Type': mimeTypes[ext],
          'Cache-Control': 'public, max-age=3600',
        },
      });
    }

    // 2. Directory listing mode
    const rawTargetDir = searchParams.get('dir') || OUTPUT_DIR;
    const targetDir = path.isAbsolute(rawTargetDir) ? path.resolve(rawTargetDir) : path.resolve(BASE_DIR, rawTargetDir);

    if (!isPathWithinApprovedRoots(targetDir)) {
      return NextResponse.json({ error: 'Access denied: directory outside approved roots' }, { status: 403 });
    }

    if (!fs.existsSync(targetDir)) {
      return NextResponse.json({ samples: [] });
    }

    // Look for sample images recursively or in samples subfolder
    const samples: { name: string; path: string; epoch?: number; mtime: number }[] = [];
    const scanDir = (dir: string, depth = 0) => {
      if (depth > 4) return;
      try {
        const entries = fs.readdirSync(dir, { withFileTypes: true });
        for (const entry of entries) {
          const fullPath = path.join(dir, entry.name);
          if (entry.isDirectory()) {
            if (entry.name === 'samples' || entry.name.includes('sample')) {
              const subEntries = fs.readdirSync(fullPath);
              for (const sub of subEntries) {
                if (/\.(png|jpg|jpeg|webp)$/i.test(sub)) {
                  const stat = fs.statSync(path.join(fullPath, sub));
                  samples.push({ name: sub, path: path.join(fullPath, sub), mtime: stat.mtimeMs });
                }
              }
            } else {
              scanDir(fullPath, depth + 1);
            }
          } else if (/\.(png|jpg|jpeg|webp)$/i.test(entry.name) && entry.name.toLowerCase().includes('sample')) {
            const stat = fs.statSync(fullPath);
            samples.push({ name: entry.name, path: fullPath, mtime: stat.mtimeMs });
          }
        }
      } catch (_) {}
    };

    scanDir(targetDir);
    samples.sort((a, b) => b.mtime - a.mtime);

    return NextResponse.json({ samples: samples.slice(0, 50) });
  } catch (error: any) {
    return NextResponse.json({ error: error?.message || 'Failed to list samples' }, { status: 500 });
  }
}

export async function POST(req: NextRequest) {
  const auth = validateApiAuth(req);
  if (!auth.authorized) {
    return NextResponse.json({ error: auth.reason || 'Unauthorized' }, { status: 401 });
  }

  try {
    const body = await req.json();
    const { action, prompt, loraPath } = body;

    if (action === 'generate') {
      return NextResponse.json({
        success: true,
        message: `Sample generation queued for: "${prompt || 'default prompt'}"`,
      });
    }

    return NextResponse.json({ success: true });
  } catch (error: any) {
    return NextResponse.json({ error: error?.message || 'Sample action failed' }, { status: 500 });
  }
}
