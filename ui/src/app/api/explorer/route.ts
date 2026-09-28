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
    const { action, loraPath, prompt, seed = 42, drift = 0.25, outputPath } = body;

    // 1. Generate 4 variants
    if (action === 'generate_variants') {
      const baseSeed = parseInt(String(seed), 10) || 42;
      const driftVal = parseFloat(String(drift)) || 0.25;

      const variants = [1, 2, 3, 4].map((id) => {
        // Deterministic pseudo-mutation per variant id
        const delta = Math.sin(baseSeed + id * 100) * driftVal;
        return {
          id,
          imageUrl: '/logo.jpg',
          seed: baseSeed + id,
          drift: driftVal,
          weights: {
            style_scale: Math.max(0.1, Math.min(2.0, parseFloat((1.0 + delta).toFixed(3)))),
            identity_scale: Math.max(0.1, Math.min(2.0, parseFloat((1.0 - delta * 0.7).toFixed(3)))),
            detail_scale: Math.max(0.1, Math.min(2.0, parseFloat((1.0 + delta * 0.5).toFixed(3)))),
          },
        };
      });

      return NextResponse.json({
        success: true,
        variants,
        seed: baseSeed,
        drift: driftVal,
      });
    }

    // 2. Export selected variant
    if (action === 'export_lora') {
      if (!outputPath) {
        return NextResponse.json({ error: 'Output path required' }, { status: 400 });
      }

      if (!loraPath || !fs.existsSync(loraPath)) {
        return NextResponse.json({ error: 'Source LoRA path does not exist' }, { status: 400 });
      }

      // Security check: both source and destination must be strictly inside approved roots
      if (!isPathWithinApprovedRoots(loraPath)) {
        return NextResponse.json({ error: 'Access denied: Source LoRA path is outside approved directories' }, { status: 403 });
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

      // Ensure target directory exists
      const targetDir = path.dirname(destPath);
      if (!fs.existsSync(targetDir)) {
        fs.mkdirSync(targetDir, { recursive: true });
      }

      fs.copyFileSync(loraPath, destPath);

      return NextResponse.json({
        success: true,
        savedTo: destPath,
      });
    }

    return NextResponse.json({ error: 'Unknown explorer action' }, { status: 400 });
  } catch (error: any) {
    return NextResponse.json({ error: error?.message || 'Explorer operation failed' }, { status: 500 });
  }
}
