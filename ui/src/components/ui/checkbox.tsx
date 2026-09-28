"use client";
import React from "react";
import { COLORS } from "@/lib/constants";

export function Checkbox({ children, checked, defaultChecked, onCheckedChange, id, className = "" }: {
  children?: React.ReactNode;
  checked?: boolean;
  defaultChecked?: boolean;
  onCheckedChange?: (checked: boolean) => void;
  id?: string;
  className?: string;
}) {
  return (
    <label className={`flex items-center gap-2 cursor-pointer ${className}`}>
      <input
        type="checkbox"
        id={id}
        checked={checked}
        defaultChecked={checked !== undefined ? undefined : defaultChecked}
        onChange={(e) => onCheckedChange?.(e.target.checked)}
        className="accent-blue-500"
      />
      {children && (
        <span className="text-xs" style={{ color: COLORS.text_primary }}>{children}</span>
      )}
    </label>
  );
}
