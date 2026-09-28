import { NextRequest, NextResponse } from 'next/server';
import fs from 'fs';
import path from 'path';

export async function POST(req: NextRequest) {
  try {
    const body = await req.json();
    const { folder, prepMode, megapixels, faceSelection, deleteOriginals } = body;

    if (!folder || !fs.existsSync(folder)) {
      return NextResponse.json({ error: 'Training folder not found' }, { status: 404 });
    }

    const files = fs.readdirSync(folder);
    const imageExtensions = new Set(['.png', '.jpg', '.jpeg', '.webp']);
    const images = files.filter((f) => imageExtensions.has(path.extname(f).toLowerCase()));

    return NextResponse.json({
      success: true,
      count: images.length,
      message: `Image prep configured for ${images.length} images: ${prepMode} at ${megapixels} MP`,
    });
  } catch (error: any) {
    return NextResponse.json({ error: error?.message || 'Convert operation failed' }, { status: 500 });
  }
}
