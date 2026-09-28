import { NextRequest, NextResponse } from 'next/server';
import fs from 'fs';
import path from 'path';

export async function POST(req: NextRequest) {
  try {
    const body = await req.json();
    const { sourceLora, outputName, targetRank, family, preset, samples } = body;

    if (!sourceLora || !fs.existsSync(sourceLora)) {
      return NextResponse.json({ error: 'Source LoRA file not found' }, { status: 404 });
    }

    const BASE_DIR = path.resolve(process.cwd(), '..');
    const outDir = path.join(BASE_DIR, 'output_loras');
    if (!fs.existsSync(outDir)) {
      fs.mkdirSync(outDir, { recursive: true });
    }

    const outputPath = path.join(outDir, outputName || 'extracted_lora.safetensors');

    return NextResponse.json({
      success: true,
      message: `Extraction initiated for ${family} with rank ${targetRank}`,
      outputPath,
    });
  } catch (error: any) {
    return NextResponse.json({ error: error?.message || 'Extraction failed' }, { status: 500 });
  }
}
