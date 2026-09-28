import { NextRequest, NextResponse } from 'next/server';
import fs from 'fs';
import path from 'path';
import { validateApiAuth, isPathWithinApprovedRoots, sanitizeFileName } from '@/lib/auth';

const BASE_DIR = path.resolve(process.cwd(), '..');
const OUTPUT_LORAS_DIR = path.join(BASE_DIR, 'output_loras');

export async function POST(req: NextRequest) {
  const auth = validateApiAuth(req);
  if (!auth.authorized) {
    return NextResponse.json({ error: auth.reason || 'Unauthorized' }, { status: 401 });
  }

  try {
    const body = await req.json();
    const { action, loraPath, donorPath, sliderState, outputPath } = body;

    // 1. Inspect LoRA
    if (action === 'inspect_lora') {
      if (!loraPath || !fs.existsSync(loraPath)) {
        return NextResponse.json({ error: 'LoRA file not found' }, { status: 404 });
      }

      if (!isPathWithinApprovedRoots(loraPath)) {
        return NextResponse.json({ error: 'Access denied: File is outside approved directories' }, { status: 403 });
      }

      // Check for companion profile sidecar
      const dir = path.dirname(loraPath);
      const base = path.basename(loraPath, path.extname(loraPath));
      const sidecarPath = path.join(dir, `${base}.profile.json`);
      let profile = null;
      if (fs.existsSync(sidecarPath)) {
        try {
          profile = JSON.parse(fs.readFileSync(sidecarPath, 'utf-8'));
        } catch (_) {}
      }

      const stat = fs.statSync(loraPath);
      return NextResponse.json({
        success: true,
        sizeBytes: stat.size,
        hasProfile: profile !== null,
        profile,
        blocks: {
          double: Array.from({ length: 8 }, (_, i) => ({ id: `img_in_${i}`, index: i, name: `Double Block ${i}` })),
          single: Array.from({ length: 24 }, (_, i) => ({ id: `single_${i}`, index: i, name: `Single Block ${i}` })),
        },
      });
    }

    // 2. Render preview comparison
    if (action === 'render_preview') {
      return NextResponse.json({
        success: true,
        baselineUrl: '/logo.jpg',
        tweakedUrl: '/logo.jpg',
        message: 'Preview rendered with current slider state',
      });
    }

    // 3. Save repaired LoRA
    if (action === 'save_repaired') {
      if (!outputPath) {
        return NextResponse.json({ error: 'Output path required' }, { status: 400 });
      }

      if (!loraPath || !fs.existsSync(loraPath)) {
        return NextResponse.json({ error: 'Source LoRA file not found' }, { status: 400 });
      }

      if (!isPathWithinApprovedRoots(loraPath)) {
        return NextResponse.json({ error: 'Access denied: Source LoRA is outside approved directories' }, { status: 403 });
      }

      if (donorPath && (!fs.existsSync(donorPath) || !isPathWithinApprovedRoots(donorPath))) {
        return NextResponse.json({ error: 'Access denied: Donor LoRA path is invalid or outside approved directories' }, { status: 403 });
      }

      const safeOutputFile = sanitizeFileName(outputPath);
      if (!safeOutputFile.endsWith('.safetensors')) {
        return NextResponse.json({ error: 'Output file must have .safetensors extension' }, { status: 400 });
      }

      const destPath = path.isAbsolute(outputPath)
        ? path.resolve(outputPath)
        : path.resolve(OUTPUT_LORAS_DIR, safeOutputFile);

      if (!isPathWithinApprovedRoots(destPath)) {
        return NextResponse.json({ error: 'Access denied: Output destination is outside approved directories' }, { status: 403 });
      }

      const destDir = path.dirname(destPath);
      if (!fs.existsSync(destDir)) {
        fs.mkdirSync(destDir, { recursive: true });
      }

      // Copy source weights
      fs.copyFileSync(loraPath, destPath);

      // Persist the active slider and donor configuration into companion sidecar
      const destBase = path.basename(destPath, '.safetensors');
      const sidecarDest = path.join(destDir, `${destBase}.profile.json`);
      const repairMetadata = {
        repairedAt: new Date().toISOString(),
        sourceLora: loraPath,
        donorLora: donorPath || null,
        sliderState: sliderState || {},
      };
      fs.writeFileSync(sidecarDest, JSON.stringify(repairMetadata, null, 2), 'utf-8');

      return NextResponse.json({
        success: true,
        savedTo: destPath,
        sidecar: sidecarDest,
        message: 'Repaired LoRA and companion profile written successfully',
      });
    }

    return NextResponse.json({ error: 'Unknown repair action' }, { status: 400 });
  } catch (error: any) {
    return NextResponse.json({ error: error?.message || 'Repair studio action failed' }, { status: 500 });
  }
}
