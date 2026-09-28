import { NextRequest, NextResponse } from 'next/server';
import fs from 'fs';
import path from 'path';
import { validateApiAuth, isPathWithinApprovedRoots } from '@/lib/auth';

export async function GET(req: NextRequest) {
  const auth = validateApiAuth(req);
  if (!auth.authorized) {
    return NextResponse.json({ error: auth.reason || 'Unauthorized' }, { status: 401 });
  }

  try {
    const { searchParams } = new URL(req.url);
    const filePath = searchParams.get('file');

    if (!filePath || !fs.existsSync(filePath)) {
      return NextResponse.json({ error: 'File not found' }, { status: 404 });
    }

    if (!isPathWithinApprovedRoots(filePath)) {
      return NextResponse.json({ error: 'Access denied: File is outside approved directories' }, { status: 403 });
    }

    if (!filePath.toLowerCase().endsWith('.safetensors')) {
      return NextResponse.json({ error: 'Target file must be a .safetensors file' }, { status: 400 });
    }

    // Check for companion metadata sidecar first
    const sidecarPath = `${filePath}.metadata.json`;
    let metadata: Record<string, string> = {};
    if (fs.existsSync(sidecarPath)) {
      try {
        metadata = JSON.parse(fs.readFileSync(sidecarPath, 'utf-8'));
      } catch (_) {}
    }

    // Parse SafeTensors binary header
    const fd = fs.openSync(filePath, 'r');
    try {
      const lenBuf = Buffer.alloc(8);
      fs.readSync(fd, lenBuf, 0, 8, 0);
      const headerLen = lenBuf.readBigUInt64LE(0);

      if (headerLen > BigInt(0) && headerLen <= BigInt(50_000_000)) {
        const headerBuf = Buffer.alloc(Number(headerLen));
        fs.readSync(fd, headerBuf, 0, Number(headerLen), 8);
        const headerJson = JSON.parse(headerBuf.toString('utf-8'));
        const embeddedMeta = headerJson.__metadata__ || {};
        metadata = { ...embeddedMeta, ...metadata };
      }
    } finally {
      fs.closeSync(fd);
    }

    return NextResponse.json({
      success: true,
      metadata,
    });
  } catch (error: any) {
    return NextResponse.json({ error: error?.message || 'Failed to read safetensors metadata' }, { status: 500 });
  }
}

export async function POST(req: NextRequest) {
  const auth = validateApiAuth(req);
  if (!auth.authorized) {
    return NextResponse.json({ error: auth.reason || 'Unauthorized' }, { status: 401 });
  }

  try {
    const body = await req.json();
    const { filePath, metadata, saveAsPath } = body;

    const targetPath = saveAsPath || filePath;
    if (!targetPath) {
      return NextResponse.json({ error: 'Target path required' }, { status: 400 });
    }

    if (!isPathWithinApprovedRoots(targetPath)) {
      return NextResponse.json({ error: 'Access denied: Target path is outside approved directories' }, { status: 403 });
    }

    if (!targetPath.toLowerCase().endsWith('.safetensors')) {
      return NextResponse.json({ error: 'Target file must have .safetensors extension' }, { status: 400 });
    }

    // Persist companion sidecar JSON
    const sidecarPath = `${targetPath}.metadata.json`;
    const targetDir = path.dirname(sidecarPath);
    if (!fs.existsSync(targetDir)) {
      fs.mkdirSync(targetDir, { recursive: true });
    }

    fs.writeFileSync(sidecarPath, JSON.stringify(metadata || {}, null, 2), 'utf-8');

    return NextResponse.json({ success: true, path: targetPath, sidecar: sidecarPath });
  } catch (error: any) {
    return NextResponse.json({ error: error?.message || 'Failed to write metadata' }, { status: 500 });
  }
}
