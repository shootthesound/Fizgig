'use client';

import React, { useEffect, useRef } from 'react';
import { COLORS } from '@/lib/constants';

interface ScrolledTextProps {
  content?: string;
  value?: string;
  className?: string;
  autoScroll?: boolean;
  readOnly?: boolean;
  style?: React.CSSProperties;
}

/**
 * Read-only scrollable text area for logs.
 * Converted from tk.scrolledtext.ScrolledText
 */
export function ScrolledText({ content, value, className = '', autoScroll = true }: ScrolledTextProps) {
  const displayContent = content || value || '';
  const textRef = useRef<HTMLTextAreaElement>(null);

  useEffect(() => {
    if (autoScroll && textRef.current) {
      textRef.current.scrollTop = textRef.current.scrollHeight;
    }
  }, [displayContent, autoScroll]);

  return (
    <div className={`relative ${className}`}>
      <textarea
        ref={textRef}
        readOnly
        value={displayContent}
        className="w-full h-full p-2 text-[10pt] border-none outline-none resize-none overflow-y-auto"
        style={{
          backgroundColor: COLORS.bg_surface,
          color: COLORS.text_primary,
          fontFamily: 'Consolas', // FONT_MONO
          scrollbarColor: `${COLORS.scrollbar_thumb} ${COLORS.bg_deep}`,
          scrollbarWidth: 'thin',
        }}
      />
    </div>
  );
}

export default ScrolledText;
