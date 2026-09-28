'use client';

import React, { useState } from 'react';
import { COLORS } from '@/lib/constants';

interface TooltipProps {
  text: string;
  children: React.ReactNode;
}

/**
 * Simple tooltip class for tkinter widgets
 * Converted from python ToolTip class
 */
export function ToolTip({ text, children }: TooltipProps) {
  const [show, setShow] = useState(false);

  return (
    <div 
      className="relative flex items-center group"
      onMouseEnter={() => setShow(true)}
      onMouseLeave={() => setShow(false)}
    >
      {children}
      {show && (
        <div 
          className="absolute z-50 px-2 py-1.5 text-[9pt] border border-solid shadow-md whitespace-nowrap pointer-events-none"
          style={{
            backgroundColor: COLORS.bg_surface,
            color: COLORS.text_primary,
            borderColor: COLORS.border,
            fontFamily: 'Segoe UI',
            top: '100%',
            left: '50%',
            transform: 'translate(-50%, 8px)'
          }}
        >
          {text}
        </div>
      )}
    </div>
  );
}
