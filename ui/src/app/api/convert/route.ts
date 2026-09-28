import { NextRequest, NextResponse } from 'next/server';
import fs from 'fs';
import path from 'path';
import { validateApiAuth, isPathWithinApprovedRoots } from '@/lib/auth';

export async function POST(req: NextRequest) {
  const auth = validateApiAuth(req);
  if (!auth.authorized) {
    return NextResponse.json({ error: auth.reason || 'Unauthorized' }, { status: 401 });
  }

  try {
    const body = await req.json();
    const { folder, prepMode, megapixels, faceSelection, deleteOriginals } = body;

    if (!folder || !fs.existsSync(folder)) {
      return NextResponse.json({ error: 'Training folder not found' }, { status: 404 });
    }

    const resolvedFolder = path.resolve(folder);
    if (!isPathWithinApprovedRoots(resolvedFolder)) {
      return NextResponse.json({ error: 'Access denied: folder path outside approved roots' }, { status: 403 });
    }

    const files = fs.readdirSync(resolvedFolder);
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
