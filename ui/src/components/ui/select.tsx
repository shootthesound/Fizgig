"use client";
import React from "react";
import { COLORS } from "@/lib/constants";

export function Select({ children, value, defaultValue, onValueChange, className = "", disabled }: {
  children: React.ReactNode;
  value?: string;
  defaultValue?: string;
  onValueChange?: (value: string) => void;
  className?: string;
  disabled?: boolean;
}) {
  return (
    <select
      value={value}
      defaultValue={value ? undefined : defaultValue}
      onChange={(e) => onValueChange?.(e.target.value)}
      disabled={disabled}
      className={`text-xs px-2 py-1.5 rounded border ${className}`}
      style={{
        backgroundColor: COLORS.bg_surface,
        color: COLORS.text_primary,
        borderColor: COLORS.border,
      }}
    >
      {children}
    </select>
  );
}

export function SelectTrigger({ children, className = "" }: { children: React.ReactNode; className?: string }) {
  return <>{children}</>;
}

export function SelectValue({ placeholder }: { placeholder?: string }) {
  return <option value="" disabled>{placeholder}</option>;
}

export function SelectContent({ children, className }: { children?: React.ReactNode; className?: string }) {
  return <>{children}</>;
}

export function SelectItem({ value, children }: { value: string; children: React.ReactNode }) {
  return <option value={value}>{children}</option>;
}
