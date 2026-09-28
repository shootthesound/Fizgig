'use client';

import React, { useState, ReactNode } from 'react';
import { COLORS } from '@/lib/constants';

interface CollapsibleFrameProps {
  title: string;
  defaultExpanded?: boolean;
  badgeText?: string;
  children: ReactNode;
}

/**
 * A frame that can be collapsed/expanded with a header.
 * 
 * Features:
 * - Click header to toggle
 * - Arrow indicator (▶/▼)
 * - Optional badge showing field status
 * - Maintains child widget state when collapsed
 * 
 * Styled as a Start-tab-style card (bg_surface body, bordered outer frame).
 */
export function CollapsibleFrame({ 
  title, 
  defaultExpanded = true, 
  badgeText, 
  children 
}: CollapsibleFrameProps) {
  const [expanded, setExpanded] = useState(defaultExpanded);
  const [isHovered, setIsHovered] = useState(false);

  return (
    <div 
      className="border border-solid"
      style={{
        backgroundColor: COLORS.bg_surface,
        borderColor: COLORS.border,
      }}
    >
      {/* Create header frame */}
      <div 
        className="flex items-center w-full cursor-pointer select-none"
        style={{
          backgroundColor: isHovered ? COLORS.bg_hover : COLORS.bg_header,
        }}
        onClick={() => setExpanded(!expanded)}
        onMouseEnter={() => setIsHovered(true)}
        onMouseLeave={() => setIsHovered(false)}
      >
        {/* Arrow indicator */}
        <span 
          className="pl-4 pr-2.5 py-3 text-[10pt]"
          style={{
            color: COLORS.text_secondary,
            fontFamily: 'Segoe UI',
          }}
        >
          {expanded ? "▼" : "▶"}
        </span>

        {/* Title label — matches Start-tab card headers at 12pt bold */}
        <span 
          className="py-3 text-[12pt] font-bold"
          style={{
            color: COLORS.text_primary,
            fontFamily: 'Segoe UI',
          }}
        >
          {title}
        </span>

        {/* Badge label (shows filled/total fields) */}
        {badgeText && (
          <span 
            className="ml-auto pr-4 pl-2 py-3 text-[9pt]"
            style={{
              color: COLORS.text_secondary,
              fontFamily: 'Segoe UI',
            }}
          >
            {badgeText}
          </span>
        )}
      </div>

      {/* Content frame — bg_surface, padded from the card edge so children don't touch the border */}
      {expanded && (
        <div 
          className="px-3 pt-2 pb-3"
          style={{ backgroundColor: COLORS.bg_surface }}
        >
          {children}
        </div>
      )}
    </div>
  );
}

export default CollapsibleFrame;
