import { NextRequest, NextResponse } from 'next/server';
import fs from 'fs';
import path from 'path';

// Location of prefs and last_used in the textstesting directory
const BASE_DIR = path.resolve(process.cwd(), '..');
const PREFS_PATH = path.join(BASE_DIR, 'prefs.json');
const LAST_USED_PATH = path.join(BASE_DIR, '.last_used.json');

export async function GET() {
  try {
    let prefs = {};
    let lastUsed = {};

    if (fs.existsSync(PREFS_PATH)) {
      prefs = JSON.parse(fs.readFileSync(PREFS_PATH, 'utf-8'));
    }

    if (fs.existsSync(LAST_USED_PATH)) {
      lastUsed = JSON.parse(fs.readFileSync(LAST_USED_PATH, 'utf-8'));
    }

    return NextResponse.json({ prefs, lastUsed });
  } catch (error: any) {
    return NextResponse.json({ error: error?.message || 'Failed to read preferences' }, { status: 500 });
  }
}

export async function POST(req: NextRequest) {
  try {
    const body = await req.json();
    const { prefs, lastUsed } = body;

    if (prefs) {
      fs.writeFileSync(PREFS_PATH, JSON.stringify(prefs, null, 2), 'utf-8');
    }

    if (lastUsed) {
      fs.writeFileSync(LAST_USED_PATH, JSON.stringify(lastUsed, null, 2), 'utf-8');
    }

    return NextResponse.json({ success: true });
  } catch (error: any) {
    return NextResponse.json({ error: error?.message || 'Failed to save preferences' }, { status: 500 });
  }
}
