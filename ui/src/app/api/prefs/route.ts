import { NextRequest, NextResponse } from 'next/server';
import fs from 'fs';
import path from 'path';
import { validateApiAuth } from '@/lib/auth';

const BASE_DIR = path.resolve(process.cwd(), '..');
const PREFS_PATH = path.join(BASE_DIR, 'prefs.json');
const LAST_USED_PATH = path.join(BASE_DIR, '.last_used.json');

export async function GET(req: NextRequest) {
  const auth = validateApiAuth(req);
  if (!auth.authorized) {
    return NextResponse.json({ error: auth.reason || 'Unauthorized' }, { status: 401 });
  }

  try {
    let prefs: Record<string, any> = {};
    let lastUsed: Record<string, any> = {};

    if (fs.existsSync(PREFS_PATH)) {
      try {
        prefs = JSON.parse(fs.readFileSync(PREFS_PATH, 'utf-8'));
      } catch (_) {}
    }

    if (fs.existsSync(LAST_USED_PATH)) {
      try {
        lastUsed = JSON.parse(fs.readFileSync(LAST_USED_PATH, 'utf-8'));
      } catch (_) {}
    }

    // Mask sensitive API keys on general read
    const { searchParams } = new URL(req.url);
    const revealSecrets = searchParams.get('secrets') === '1';

    const safePrefs = { ...prefs };
    if (!revealSecrets && typeof safePrefs.hf_token === 'string' && safePrefs.hf_token.length > 6) {
      safePrefs.hf_token = `${safePrefs.hf_token.slice(0, 4)}...${safePrefs.hf_token.slice(-3)}`;
    }

    return NextResponse.json({ prefs: safePrefs, lastUsed });
  } catch (error: any) {
    return NextResponse.json({ error: error?.message || 'Failed to read preferences' }, { status: 500 });
  }
}

export async function POST(req: NextRequest) {
  const auth = validateApiAuth(req);
  if (!auth.authorized) {
    return NextResponse.json({ error: auth.reason || 'Unauthorized' }, { status: 401 });
  }

  try {
    const body = await req.json();
    const { prefs, lastUsed } = body;

    if (prefs && typeof prefs === 'object') {
      let existingPrefs: Record<string, any> = {};
      if (fs.existsSync(PREFS_PATH)) {
        try {
          existingPrefs = JSON.parse(fs.readFileSync(PREFS_PATH, 'utf-8'));
        } catch (_) {}
      }

      // If token wasn't modified (still masked), preserve original token
      const updatedPrefs = { ...existingPrefs, ...prefs };
      if (typeof prefs.hf_token === 'string' && prefs.hf_token.includes('...')) {
        updatedPrefs.hf_token = existingPrefs.hf_token || '';
      }

      fs.writeFileSync(PREFS_PATH, JSON.stringify(updatedPrefs, null, 2), 'utf-8');
    }

    if (lastUsed && typeof lastUsed === 'object') {
      let existingLastUsed: Record<string, any> = {};
      if (fs.existsSync(LAST_USED_PATH)) {
        try {
          existingLastUsed = JSON.parse(fs.readFileSync(LAST_USED_PATH, 'utf-8'));
        } catch (_) {}
      }
      fs.writeFileSync(LAST_USED_PATH, JSON.stringify({ ...existingLastUsed, ...lastUsed }, null, 2), 'utf-8');
    }

    return NextResponse.json({ success: true });
  } catch (error: any) {
    return NextResponse.json({ error: error?.message || 'Failed to save preferences' }, { status: 500 });
  }
}
