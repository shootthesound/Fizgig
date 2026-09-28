import { NextRequest, NextResponse } from 'next/server';
import fs from 'fs';
import path from 'path';

export async function POST(req: NextRequest) {
  try {
    const body = await req.json();
    const { action, loraPath, prompt, seed, strength, outputPath } = body;

    // 1. Generate 4 variants
    if (action === 'generate_variants') {
      const variants = [1, 2, 3, 4].map((id) => ({
        id,
        imageUrl: '/placeholder-sample.png',
        weights: {
          style_scale: 0.8 + id * 0.1,
          identity_scale: 1.0 - id * 0.05,
        },
      }));

      return NextResponse.json({
        success: true,
        variants,
        seed: seed || 42,
      });
    }

    // 2. Export selected variant
    if (action === 'export_lora') {
      if (!outputPath) {
        return NextResponse.json({ error: 'Output path required' }, { status: 400 });
      }

      if (loraPath && fs.existsSync(loraPath)) {
        fs.copyFileSync(loraPath, outputPath);
      }

      return NextResponse.json({
        success: true,
        savedTo: outputPath,
      });
    }

    return NextResponse.json({ error: 'Unknown explorer action' }, { status: 400 });
  } catch (error: any) {
    return NextResponse.json({ error: error?.message || 'Explorer operation failed' }, { status: 500 });
  }
}
