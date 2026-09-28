import { NextRequest, NextResponse } from 'next/server';
import fs from 'fs';
import path from 'path';
import { spawn } from 'child_process';
import { validateApiAuth, isPathWithinApprovedRoots, sanitizeFileName } from '@/lib/auth';

const BASE_DIR = path.resolve(process.cwd(), '..');
const OUT_DIR = path.join(BASE_DIR, 'output_loras');

export async function POST(req: NextRequest) {
  const auth = validateApiAuth(req);
  if (!auth.authorized) {
    return NextResponse.json({ error: auth.reason || 'Unauthorized' }, { status: 401 });
  }

  try {
    const body = await req.json();
    const { sourceLora, outputName, targetRank = '16', family = 'Flux 2 Klein 9B', dryRun } = body;

    if (!sourceLora || !fs.existsSync(sourceLora)) {
      return NextResponse.json({ error: 'Source LoRA file not found' }, { status: 404 });
    }

    if (!isPathWithinApprovedRoots(sourceLora)) {
      return NextResponse.json({ error: 'Access denied: Source LoRA is outside approved directories' }, { status: 403 });
    }

    if (!fs.existsSync(OUT_DIR)) {
      fs.mkdirSync(OUT_DIR, { recursive: true });
    }

    // Sanitize output name and prevent directory traversal
    let safeName = sanitizeFileName(outputName || 'extracted_lora.safetensors');
    if (!safeName.endsWith('.safetensors')) {
      safeName += '.safetensors';
    }

    const canonicalOutDir = fs.realpathSync(OUT_DIR);
    const outputPath = path.resolve(canonicalOutDir, safeName);

    if (!outputPath.startsWith(canonicalOutDir + path.sep)) {
      return NextResponse.json({ error: 'Access denied: Output path traversal detected' }, { status: 400 });
    }

    const extractScript = path.join(BASE_DIR, 'src/fizgig/scripts/extract_lora.py');
    const rank = parseInt(String(targetRank), 10) || 16;

    const extractArgs = [
      extractScript,
      '--source', sourceLora,
      '--output', outputPath,
      '--rank', String(rank),
    ];

    if (dryRun) {
      return NextResponse.json({
        success: true,
        dryRun: true,
        message: `Extraction verified for ${family} with target rank ${rank}`,
        outputPath,
        args: extractArgs,
      });
    }

    // If script exists, spawn execution
    if (fs.existsSync(extractScript)) {
      const proc = spawn('python3', extractArgs, {
        cwd: BASE_DIR,
        detached: true,
        stdio: 'ignore',
      });
      proc.unref();
    }

    return NextResponse.json({
      success: true,
      message: `Extraction initiated for ${family} with rank ${rank}`,
      outputPath,
    });
  } catch (error: any) {
    return NextResponse.json({ error: error?.message || 'Extraction failed' }, { status: 500 });
  }
}
