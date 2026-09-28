'use client';

import React, { ReactNode } from 'react';
import { COLORS } from '@/lib/constants';

interface SectionCardProps {
  title?: string;
  description?: string;
  accentBorder?: boolean;
  children: ReactNode;
}

/**
 * Start-tab-style surface card with an optional description line.
 * The card is packed into `parent` with horizontal padding matching the banner.
 */
export function SectionCard({ title, description, accentBorder = false, children }: SectionCardProps) {
  return (
    <div 
      className="w-full px-9 pt-0 pb-4"
      style={{ backgroundColor: COLORS.bg_deep }}
    >
      <div 
        className="w-full border border-solid"
        style={{
          backgroundColor: COLORS.bg_surface,
          borderColor: accentBorder ? COLORS.accent : COLORS.border,
        }}
      >
        {title && (
          <div 
            className="px-5 pt-4"
            style={{ 
              paddingBottom: description ? '8px' : '40px', // approximations for padding y=(16, 2/10)
              fontFamily: 'Segoe UI' 
            }}
          >
            <h3 
              className="text-[12pt] font-bold m-0"
              style={{ color: COLORS.text_primary, backgroundColor: COLORS.bg_surface }}
            >
              {title}
            </h3>
          </div>
        )}
        
        {description && (
          <div 
            className="px-5 pt-0 pb-2.5 max-w-[760px] text-left"
            style={{ fontFamily: 'Segoe UI' }}
          >
            <p 
              className="text-[10pt] m-0"
              style={{ color: COLORS.text_explain, backgroundColor: COLORS.bg_surface }}
            >
              {description}
            </p>
          </div>
        )}

        <div 
          className="w-full px-5 pt-0 pb-4"
          style={{ backgroundColor: COLORS.bg_surface }}
        >
          {children}
        </div>
      </div>
    </div>
  );
}

export default SectionCard;
