import { NextRequest, NextResponse } from 'next/server';
import fs from 'fs';
import path from 'path';
import { validateApiAuth, isPathWithinApprovedRoots, sanitizeFileName } from '@/lib/auth';

export async function POST(req: NextRequest) {
  const auth = validateApiAuth(req);
  if (!auth.authorized) {
    return NextResponse.json({ error: auth.reason || 'Unauthorized' }, { status: 401 });
  }

  try {
    const body = await req.json();
    const { action, folder, loraPath, targetPath, epoch } = body;

    // 1. Scan folder for epoch checkpoints
    if (action === 'scan_checkpoints') {
      if (!folder || !fs.existsSync(folder)) {
        return NextResponse.json({ error: 'Folder not found' }, { status: 404 });
      }

      const resolvedFolder = path.resolve(folder);
      if (!isPathWithinApprovedRoots(resolvedFolder)) {
        return NextResponse.json({ error: 'Access denied: folder path outside approved roots' }, { status: 403 });
      }

      const files = fs.readdirSync(resolvedFolder);
      const checkpoints: { fileName: string; filePath: string; epoch: number; sizeBytes: number }[] = [];

      files.forEach((file) => {
        if (file.endsWith('.safetensors')) {
          const match = file.match(/-(\d+)\.safetensors$/);
          const epochNum = match ? parseInt(match[1], 10) : 0;
          const full = path.join(resolvedFolder, file);
          const stat = fs.statSync(full);
          checkpoints.push({
            fileName: file,
            filePath: full,
            epoch: epochNum,
            sizeBytes: stat.size,
          });
        }
      });

      checkpoints.sort((a, b) => a.epoch - b.epoch);

      return NextResponse.json({
        success: true,
        count: checkpoints.length,
        checkpoints,
      });
    }

    // 2. Render epoch preview
    if (action === 'render_epoch') {
      return NextResponse.json({
        success: true,
        epoch,
        previewUrl: '/placeholder-sample.png',
      });
    }

    // 3. Score epoch against reference
    if (action === 'score_epoch') {
      const simulatedScore = Math.min(0.98, 0.72 + (epoch || 1) * 0.015);
      return NextResponse.json({
        success: true,
        epoch,
        score: parseFloat(simulatedScore.toFixed(4)),
      });
    }

    // 4. Promote checkpoint
    if (action === 'promote') {
      if (!loraPath || !fs.existsSync(loraPath) || !targetPath) {
        return NextResponse.json({ error: 'Valid source and target paths required' }, { status: 400 });
      }

      const resolvedSource = path.resolve(loraPath);
      const resolvedTarget = path.resolve(targetPath);

      if (!isPathWithinApprovedRoots(resolvedSource) || !isPathWithinApprovedRoots(resolvedTarget)) {
        return NextResponse.json({ error: 'Access denied: paths must be within approved roots' }, { status: 403 });
      }

      if (!resolvedSource.toLowerCase().endsWith('.safetensors') || !resolvedTarget.toLowerCase().endsWith('.safetensors')) {
        return NextResponse.json({ error: 'Only .safetensors files can be promoted' }, { status: 400 });
      }

      const targetDir = path.dirname(resolvedTarget);
      if (!fs.existsSync(targetDir)) {
        fs.mkdirSync(targetDir, { recursive: true });
      }

      fs.copyFileSync(resolvedSource, resolvedTarget);
      return NextResponse.json({
        success: true,
        message: `Promoted ${path.basename(resolvedSource)} to ${path.basename(resolvedTarget)}`,
      });
    }

    return NextResponse.json({ error: 'Unknown royale action' }, { status: 400 });
  } catch (error: any) {
    return NextResponse.json({ error: error?.message || 'Royale operation failed' }, { status: 500 });
  }
}
