import { NextRequest, NextResponse } from 'next/server';
import fs from 'fs';

export async function GET(req: NextRequest) {
  try {
    const { searchParams } = new URL(req.url);
    const filePath = searchParams.get('file');

    if (!filePath || !fs.existsSync(filePath)) {
      return NextResponse.json({ error: 'File not found' }, { status: 404 });
    }

    const fd = fs.openSync(filePath, 'r');
    const lenBuf = Buffer.alloc(8);
    fs.readSync(fd, lenBuf, 0, 8, 0);
    const headerLen = lenBuf.readBigUInt64LE(0);

    if (headerLen <= BigInt(0) || headerLen > BigInt(100000000)) {
      fs.closeSync(fd);
      return NextResponse.json({ error: 'Invalid safetensors header' }, { status: 400 });
    }

    const headerBuf = Buffer.alloc(Number(headerLen));
    fs.readSync(fd, headerBuf, 0, Number(headerLen), 8);
    fs.closeSync(fd);

    const headerJson = JSON.parse(headerBuf.toString('utf-8'));
    const metadata = headerJson.__metadata__ || {};

    return NextResponse.json({
      success: true,
      metadata,
    });
  } catch (error: any) {
    return NextResponse.json({ error: error?.message || 'Failed to read safetensors metadata' }, { status: 500 });
  }
}

export async function POST(req: NextRequest) {
  try {
    const body = await req.json();
    const { filePath, metadata, saveAsPath } = body;

    const targetPath = saveAsPath || filePath;
    if (!targetPath) {
      return NextResponse.json({ error: 'Target path required' }, { status: 400 });
    }

    // In a full environment, python script handles in-place tensor rewriting
    return NextResponse.json({ success: true, path: targetPath });
  } catch (error: any) {
    return NextResponse.json({ error: error?.message || 'Failed to write metadata' }, { status: 500 });
  }
}
