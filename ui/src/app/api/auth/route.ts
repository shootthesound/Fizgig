import { NextRequest, NextResponse } from 'next/server';
import { getOrCreateAuthToken, validateApiAuth } from '@/lib/auth';

export async function GET(req: NextRequest) {
  const token = getOrCreateAuthToken();

  const response = NextResponse.json({
    success: true,
    token,
  });

  // Set session cookie
  response.cookies.set({
    name: 'fizgig_token',
    value: token,
    httpOnly: true,
    sameSite: 'lax',
    path: '/',
  });

  return response;
}

export async function POST(req: NextRequest) {
  const auth = validateApiAuth(req);
  return NextResponse.json({
    success: auth.authorized,
    authorized: auth.authorized,
    reason: auth.reason,
  });
}
