import { NextResponse } from 'next/server';
import { execSync } from 'child_process';
import os from 'os';

export async function GET() {
  let vram = { used: 0, total: 0 };
  let gpus: Array<{ id: number; name: string; total_gb: number }> = [];

  try {
    const smiOutput = execSync(
      'nvidia-smi --query-gpu=index,name,memory.used,memory.total --format=csv,noheader,nounits',
      { encoding: 'utf-8', timeout: 3000 }
    );

    const lines = smiOutput.trim().split('\n');
    lines.forEach((line) => {
      const parts = line.split(',').map((p) => p.trim());
      if (parts.length >= 4) {
        const idx = parseInt(parts[0], 10);
        const name = parts[1];
        const usedMb = parseFloat(parts[2]);
        const totalMb = parseFloat(parts[3]);

        gpus.push({
          id: idx,
          name,
          total_gb: Math.round(totalMb / 1024),
        });

        if (idx === 0) {
          vram = {
            used: Math.round(usedMb * 1024 * 1024),
            total: Math.round(totalMb * 1024 * 1024),
          };
        }
      }
    });
  } catch {
    // If nvidia-smi is not available, default to placeholder
    vram = { used: 0, total: 24 * 1024 * 1024 * 1024 };
  }

  const totalRam = os.totalmem();
  const freeRam = os.freemem();
  const ram = {
    used: totalRam - freeRam,
    total: totalRam,
  };

  return NextResponse.json({
    vram,
    ram,
    gpus,
  });
}
