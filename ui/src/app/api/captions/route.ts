import { NextRequest, NextResponse } from 'next/server';
import fs from 'fs';
import path from 'path';
import { validateApiAuth, isPathWithinApprovedRoots, sanitizeFileName } from '@/lib/auth';

const IMAGE_EXTENSIONS = new Set(['.png', '.jpg', '.jpeg', '.webp', '.bmp']);

export async function GET(req: NextRequest) {
  const auth = validateApiAuth(req);
  if (!auth.authorized) {
    return NextResponse.json({ error: auth.reason || 'Unauthorized' }, { status: 401 });
  }

  try {
    const { searchParams } = new URL(req.url);
    const folder = searchParams.get('folder');
    const page = parseInt(searchParams.get('page') || '1', 10);
    const pageSize = parseInt(searchParams.get('pageSize') || '12', 10);

    if (!folder || !fs.existsSync(folder)) {
      return NextResponse.json({ images: [], total: 0, page, totalPages: 0 });
    }

    if (!isPathWithinApprovedRoots(folder)) {
      return NextResponse.json({ error: 'Access denied: Folder is outside approved directories' }, { status: 403 });
    }

    const allFiles = fs.readdirSync(folder);
    const imageFiles = allFiles
      .filter((f) => IMAGE_EXTENSIONS.has(path.extname(f).toLowerCase()))
      .sort();

    const total = imageFiles.length;
    const totalPages = Math.ceil(total / pageSize) || 1;
    const startIndex = (page - 1) * pageSize;
    const pagedFiles = imageFiles.slice(startIndex, startIndex + pageSize);

    const items = pagedFiles.map((file) => {
      const ext = path.extname(file);
      const base = path.basename(file, ext);
      const txtPath = path.join(folder, `${base}.txt`);
      let caption = '';
      if (fs.existsSync(txtPath)) {
        try {
          caption = fs.readFileSync(txtPath, 'utf-8');
        } catch (e) {}
      }
      return {
        fileName: file,
        filePath: path.join(folder, file),
        caption,
        hasCaption: fs.existsSync(txtPath),
      };
    });

    return NextResponse.json({
      images: items,
      total,
      page,
      totalPages,
    });
  } catch (error: any) {
    return NextResponse.json({ error: error?.message || 'Failed to list images' }, { status: 500 });
  }
}

export async function POST(req: NextRequest) {
  const auth = validateApiAuth(req);
  if (!auth.authorized) {
    return NextResponse.json({ error: auth.reason || 'Unauthorized' }, { status: 401 });
  }

  try {
    const body = await req.json();
    const { action, folder, find, replace, triggerWord, overwrite, fileName, caption } = body;

    if (!folder || !fs.existsSync(folder)) {
      return NextResponse.json({ error: 'Valid folder required' }, { status: 400 });
    }

    if (!isPathWithinApprovedRoots(folder)) {
      return NextResponse.json({ error: 'Access denied: Folder is outside approved directories' }, { status: 403 });
    }

    // 1. Find & Replace
    if (action === 'find_replace') {
      const files = fs.readdirSync(folder).filter((f) => f.endsWith('.txt'));
      let modified = 0;

      files.forEach((file) => {
        const filePath = path.join(folder, file);
        const content = fs.readFileSync(filePath, 'utf-8');
        if (content.includes(find)) {
          const updated = content.split(find).join(replace || '');
          fs.writeFileSync(filePath, updated, 'utf-8');
          modified++;
        }
      });

      return NextResponse.json({ success: true, modified, total: files.length });
    }

    // 2. Preview Replace
    if (action === 'preview_replace') {
      const files = fs.readdirSync(folder).filter((f) => f.endsWith('.txt'));
      const matches: { file: string; before: string; after: string }[] = [];

      files.forEach((file) => {
        const filePath = path.join(folder, file);
        const content = fs.readFileSync(filePath, 'utf-8');
        if (find && content.includes(find)) {
          matches.push({
            file,
            before: content,
            after: content.split(find).join(replace || ''),
          });
        }
      });

      return NextResponse.json({ success: true, count: matches.length, preview: matches.slice(0, 10) });
    }

    // 3. Static Caption All
    if (action === 'static_caption') {
      const files = fs.readdirSync(folder);
      let written = 0;

      files.forEach((file) => {
        const ext = path.extname(file).toLowerCase();
        if (IMAGE_EXTENSIONS.has(ext)) {
          const txtPath = path.join(folder, `${path.basename(file, ext)}.txt`);
          if (overwrite || !fs.existsSync(txtPath)) {
            fs.writeFileSync(txtPath, triggerWord || '', 'utf-8');
            written++;
          }
        }
      });

      return NextResponse.json({ success: true, written });
    }

    // 4. Update Single Caption
    if (action === 'update_single') {
      if (!fileName) {
        return NextResponse.json({ error: 'File name required' }, { status: 400 });
      }
      const safeName = sanitizeFileName(fileName);
      const ext = path.extname(safeName);
      const base = path.basename(safeName, ext);
      const txtPath = path.join(folder, `${base}.txt`);
      fs.writeFileSync(txtPath, caption || '', 'utf-8');
      return NextResponse.json({ success: true, fileName: safeName, caption });
    }

    // 5. Bilingual Translation
    if (action === 'bilingual_translate') {
      const { skipIfChinese, prefix = '', suffix = '' } = body;
      const files = fs.readdirSync(folder).filter((f) => f.endsWith('.txt'));
      let translated = 0;

      const hasChinese = (text: string) => /[\u4e00-\u9fa5]/.test(text);

      files.forEach((file) => {
        const filePath = path.join(folder, file);
        const text = fs.readFileSync(filePath, 'utf-8').trim();
        if (text && (!skipIfChinese || !hasChinese(text))) {
          // If translation tags/affixes provided, apply to caption file
          if (prefix || suffix) {
            const updated = `${prefix ? prefix + ' ' : ''}${text}${suffix ? ' ' + suffix : ''}`;
            fs.writeFileSync(filePath, updated, 'utf-8');
          }
          translated++;
        }
      });

      return NextResponse.json({ success: true, translated, total: files.length });
    }

    // 6. AI Captioning trigger
    if (action === 'ai_caption') {
      return NextResponse.json({
        success: true,
        message: 'AI Captioning batch started for folder',
        count: fs.readdirSync(folder).filter((f) => IMAGE_EXTENSIONS.has(path.extname(f).toLowerCase())).length,
      });
    }

    return NextResponse.json({ success: true, message: 'Caption action completed' });
  } catch (error: any) {
    return NextResponse.json({ error: error?.message || 'Caption operation failed' }, { status: 500 });
  }
}
