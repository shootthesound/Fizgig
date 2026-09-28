import { NextRequest, NextResponse } from 'next/server';
import fs from 'fs';
import path from 'path';

const BASE_DIR = path.resolve(process.cwd(), '..');
const DATASET_DIR = path.join(BASE_DIR, 'dataset');
const SNAPSHOT_DIR = path.join(DATASET_DIR, 'run_snapshots');

export async function GET(req: NextRequest) {
  try {
    const { searchParams } = new URL(req.url);
    const folder = searchParams.get('folder');

    if (!folder || !fs.existsSync(folder)) {
      return NextResponse.json({ exists: false, count: 0, images: [] });
    }

    const files = fs.readdirSync(folder);
    const imageExtensions = new Set(['.png', '.jpg', '.jpeg', '.webp']);
    const images: string[] = [];
    let captionsCount = 0;

    files.forEach((f) => {
      const ext = path.extname(f).toLowerCase();
      if (imageExtensions.has(ext)) {
        images.push(f);
        const txtFile = path.join(folder, `${path.basename(f, ext)}.txt`);
        if (fs.existsSync(txtFile)) {
          captionsCount++;
        }
      }
    });

    return NextResponse.json({
      exists: true,
      count: images.length,
      captionsCount,
      images: images.slice(0, 100),
    });
  } catch (error: any) {
    return NextResponse.json({ error: error?.message || 'Failed to scan dataset' }, { status: 500 });
  }
}

export async function POST(req: NextRequest) {
  try {
    const body = await req.json();
    const {
      dataset_name = 'Fizgig_train',
      image_folder,
      image_folder2,
      caption_ext = '.txt',
      megapixels = '1.0',
      batch_size = '1',
      enable_bucket = true,
      no_upscale = true,
    } = body;

    if (!fs.existsSync(DATASET_DIR)) {
      fs.mkdirSync(DATASET_DIR, { recursive: true });
    }
    if (!fs.existsSync(SNAPSHOT_DIR)) {
      fs.mkdirSync(SNAPSHOT_DIR, { recursive: true });
    }

    const mp = parseFloat(megapixels) || 1.0;
    const side = Math.floor(Math.floor(Math.sqrt(mp * 1_000_000)) / 16) * 16;
    const batch = parseInt(String(batch_size), 10) || 1;

    const tomlLines: string[] = [
      '[general]',
      `resolution = [${side}, ${side}]`,
      `caption_extension = "${caption_ext}"`,
      `batch_size = ${batch}`,
      `num_repeats = 1`,
      `enable_bucket = ${enable_bucket ? 'true' : 'false'}`,
      `bucket_no_upscale = ${no_upscale ? 'true' : 'false'}`,
      '',
      '[[datasets]]',
      `image_directory = "${(image_folder || '').replace(/\\/g, '/')}"`,
    ];

    // Multi Concept / secondary folder support
    if (image_folder2 && image_folder2.trim()) {
      tomlLines.push('');
      tomlLines.push('[[datasets]]');
      tomlLines.push(`image_directory = "${image_folder2.trim().replace(/\\/g, '/')}"`);
    }

    const tomlContent = tomlLines.join('\n') + '\n';
    const tomlPath = path.join(DATASET_DIR, `${dataset_name}.toml`);
    fs.writeFileSync(tomlPath, tomlContent, 'utf-8');

    // Also snapshot for run immutability (#98)
    const snapPath = path.join(SNAPSHOT_DIR, `${dataset_name}-${Date.now()}.toml`);
    fs.writeFileSync(snapPath, tomlContent, 'utf-8');

    return NextResponse.json({ success: true, path: tomlPath, snapshot: snapPath });
  } catch (error: any) {
    return NextResponse.json({ error: error?.message || 'Failed to save dataset toml' }, { status: 500 });
  }
}
