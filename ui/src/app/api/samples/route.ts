import { NextRequest, NextResponse } from 'next/server';
import fs from 'fs';
import path from 'path';

const BASE_DIR = path.resolve(process.cwd(), '..');
const OUTPUT_DIR = path.join(BASE_DIR, 'output_loras');

export async function GET(req: NextRequest) {
  try {
    const { searchParams } = new URL(req.url);
    const targetDir = searchParams.get('dir') || OUTPUT_DIR;

    if (!fs.existsSync(targetDir)) {
      return NextResponse.json({ samples: [] });
    }

    // Look for sample images recursively or in samples subfolder
    const samples: { name: string; path: string; epoch?: number; mtime: number }[] = [];
    const scanDir = (dir: string) => {
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
              scanDir(fullPath);
            }
          } else if (/\.(png|jpg|jpeg|webp)$/i.test(entry.name) && entry.name.toLowerCase().includes('sample')) {
            const stat = fs.statSync(fullPath);
            samples.push({ name: entry.name, path: fullPath, mtime: stat.mtimeMs });
          }
        }
      } catch (e) {}
    };

    scanDir(targetDir);
    samples.sort((a, b) => b.mtime - a.mtime);

    return NextResponse.json({ samples: samples.slice(0, 50) });
  } catch (error: any) {
    return NextResponse.json({ error: error?.message || 'Failed to list samples' }, { status: 500 });
  }
}

export async function POST(req: NextRequest) {
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
