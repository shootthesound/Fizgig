'use client';

import React from 'react';
import { COLORS } from '@/lib/constants';

interface TabBannerProps {
  title: string;
  subtitle?: string;
  description?: string;
}

/**
 * Start-tab-style banner (22pt title + 11pt subtitle on bg_deep).
 */
export function TabBanner({ title, subtitle, description }: TabBannerProps) {
  const sub = subtitle || description;
  return (
    <div 
      className="w-full px-9 pt-7 pb-5"
      style={{ backgroundColor: COLORS.bg_deep }}
    >
      <div 
        className="text-[22pt] font-bold text-left m-0 leading-tight"
        style={{ color: COLORS.text_primary, fontFamily: 'Segoe UI' }}
      >
        {title}
      </div>
      
      {sub && (
        <div 
          className="text-[11pt] text-left max-w-[1050px] mt-1"
          style={{ color: COLORS.text_explain, fontFamily: 'Segoe UI' }}
        >
          {sub}
        </div>
      )}
    </div>
  );
}

export default TabBanner;
