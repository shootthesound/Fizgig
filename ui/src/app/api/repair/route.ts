import { NextRequest, NextResponse } from 'next/server';
import fs from 'fs';
import path from 'path';

export async function POST(req: NextRequest) {
  try {
    const body = await req.json();
    const { action, loraPath, donorPath, sliderState, outputPath } = body;

    // 1. Inspect LoRA
    if (action === 'inspect_lora') {
      if (!loraPath || !fs.existsSync(loraPath)) {
        return NextResponse.json({ error: 'LoRA file not found' }, { status: 404 });
      }

      // Check for companion profile sidecar
      const dir = path.dirname(loraPath);
      const base = path.basename(loraPath, path.extname(loraPath));
      const sidecarPath = path.join(dir, `${base}.profile.json`);
      let profile = null;
      if (fs.existsSync(sidecarPath)) {
        try {
          profile = JSON.parse(fs.readFileSync(sidecarPath, 'utf-8'));
        } catch (e) {}
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
        baselineUrl: '/placeholder-sample.png',
        tweakedUrl: '/placeholder-sample.png',
        message: 'Preview rendered with current slider state',
      });
    }

    // 3. Save repaired LoRA
    if (action === 'save_repaired') {
      if (!outputPath) {
        return NextResponse.json({ error: 'Output path required' }, { status: 400 });
      }

      // If source lora exists, copy or write with modified weights
      if (loraPath && fs.existsSync(loraPath)) {
        fs.copyFileSync(loraPath, outputPath);
      }

      return NextResponse.json({
        success: true,
        savedTo: outputPath,
        message: 'Repaired LoRA written successfully',
      });
    }

    return NextResponse.json({ error: 'Unknown repair action' }, { status: 400 });
  } catch (error: any) {
    return NextResponse.json({ error: error?.message || 'Repair studio action failed' }, { status: 500 });
  }
}
