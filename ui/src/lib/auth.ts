import { NextRequest } from 'next/server';
import crypto from 'crypto';
import fs from 'fs';
import path from 'path';
import os from 'os';

const BASE_DIR = path.resolve(process.cwd(), '..');
const TOKEN_FILE = path.join(BASE_DIR, '.fizgig_token');

// In-memory token cache
let cachedToken: string | null = null;

/**
 * Retrieves the configured auth token or generates a secure local session secret.
 */
export function getOrCreateAuthToken(): string {
  if (process.env.FIZGIG_AUTH_TOKEN?.trim()) {
    return process.env.FIZGIG_AUTH_TOKEN.trim();
  }
  if (process.env.FIZGIG_API_KEY?.trim()) {
    return process.env.FIZGIG_API_KEY.trim();
  }

  if (cachedToken) {
    return cachedToken;
  }

  if (fs.existsSync(TOKEN_FILE)) {
    try {
      const saved = fs.readFileSync(TOKEN_FILE, 'utf-8').trim();
      if (saved) {
        cachedToken = saved;
        return saved;
      }
    } catch (_) {}
  }

  // Generate a cryptographically secure 256-bit token
  const newToken = crypto.randomBytes(32).toString('hex');
  try {
    fs.writeFileSync(TOKEN_FILE, newToken, { mode: 0o600 });
  } catch (_) {}
  cachedToken = newToken;
  return newToken;
}

/**
 * Validates request authentication via Bearer token, x-fizgig-token header,
 * session cookie, or trusted local loopback/same-origin.
 */
export function validateApiAuth(req: NextRequest): { authorized: boolean; reason?: string } {
  const serverToken = getOrCreateAuthToken();

  // 1. Authorization: Bearer <token>
  const authHeader = req.headers.get('authorization');
  if (authHeader && authHeader.startsWith('Bearer ')) {
    const token = authHeader.slice(7).trim();
    if (token === serverToken) {
      return { authorized: true };
    }
  }

  // 2. Custom header: x-fizgig-token
  const customHeader = req.headers.get('x-fizgig-token');
  if (customHeader && customHeader.trim() === serverToken) {
    return { authorized: true };
  }

  // 3. Session cookie
  const cookieToken = req.cookies.get('fizgig_token')?.value;
  if (cookieToken && cookieToken === serverToken) {
    return { authorized: true };
  }

  // 4. Same-origin or local loopback validation for browser requests
  const host = req.headers.get('host') || '';
  const origin = req.headers.get('origin');
  const referer = req.headers.get('referer');
  const secFetchSite = req.headers.get('sec-fetch-site');

  const isLoopback =
    host.startsWith('localhost') ||
    host.startsWith('127.0.0.1') ||
    host.startsWith('[::1]');

  const isSameOrigin =
    secFetchSite === 'same-origin' ||
    (origin && origin.includes(host)) ||
    (referer && referer.includes(host));

  // If remote authentication is strictly enforced via environment, require the token
  const strictRemoteAuth = Boolean(
    process.env.FIZGIG_AUTH_TOKEN ||
    process.env.FIZGIG_REQUIRE_AUTH
  );

  if (!strictRemoteAuth && (isLoopback || isSameOrigin)) {
    return { authorized: true };
  }

  return { authorized: false, reason: 'Unauthorized: Missing or invalid API authentication token' };
}

/**
 * Returns canonical approved roots where file reading and writing is permitted.
 */
export function getApprovedRoots(): string[] {
  const projectRoot = path.resolve(process.cwd(), '..');
  const userHome = os.homedir();
  const roots: string[] = [projectRoot, userHome];

  // Common cloud container volumes (RunPod, Vast.ai, Modal, etc.)
  const cloudRoots = ['/workspace', '/data', '/content', '/notebooks', '/tmp'];
  for (const cr of cloudRoots) {
    if (fs.existsSync(cr)) {
      roots.push(cr);
    }
  }

  // Optional user-specified allowed roots via environment
  if (process.env.FIZGIG_ALLOWED_ROOTS) {
    const custom = process.env.FIZGIG_ALLOWED_ROOTS.split(',')
      .map((r) => r.trim())
      .filter(Boolean);
    for (const r of custom) {
      if (fs.existsSync(r)) {
        roots.push(path.resolve(r));
      }
    }
  }

  const canonical = roots.map((r) => {
    try {
      return fs.realpathSync(r);
    } catch {
      return path.resolve(r);
    }
  });

  return Array.from(new Set(canonical));
}

const SENSITIVE_SEGMENTS = new Set([
  '.ssh',
  '.aws',
  '.gnupg',
  '.fizgig_token',
  '.env',
  '.bash_history',
  '.bashrc',
  '.profile',
  '.git',
  '.next',
]);

/**
 * Ensures target path is strictly contained within at least one approved root
 * and does not access sensitive credential directories or dotfiles.
 */
export function isPathWithinApprovedRoots(targetPath: string, approvedRoots = getApprovedRoots()): boolean {
  try {
    const resolved = fs.existsSync(targetPath)
      ? fs.realpathSync(targetPath)
      : path.resolve(targetPath);

    // Block access to sensitive system/credential directories
    const segments = resolved.split(path.sep);
    for (const seg of segments) {
      if (SENSITIVE_SEGMENTS.has(seg) || seg.startsWith('.env')) {
        return false;
      }
    }

    return approvedRoots.some((root) => {
      const rel = path.relative(root, resolved);
      return !rel.startsWith('..') && !path.isAbsolute(rel);
    });
  } catch {
    return false;
  }
}

/**
 * Sanitizes user-provided filenames to prevent path traversal.
 */
export function sanitizeFileName(name: string, allowExt = true): string {
  const base = path.basename(name).trim();
  const sanitized = base.replace(/[\/\0\\:*?"<>|]/g, '_');
  if (!allowExt) {
    return sanitized.replace(/\.[^/.]+$/, '');
  }
  return sanitized;
}
