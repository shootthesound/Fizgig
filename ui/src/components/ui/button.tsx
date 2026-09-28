"use client";
import React from "react";
import { COLORS } from "@/lib/constants";

export function Button({ children, variant = "default", className = "", size, ...props }: React.ButtonHTMLAttributes<HTMLButtonElement> & { variant?: "default" | "primary" | "ghost" | "destructive" | "outline" | "secondary"; className?: string; size?: string }) {
  const baseStyle = "text-xs px-3 py-1.5 rounded font-medium cursor-pointer transition-colors";
  const variants: Record<string, React.CSSProperties> = {
    default: { backgroundColor: COLORS.bg_surface, color: COLORS.text_primary, border: `1px solid ${COLORS.border}` },
    primary: { backgroundColor: COLORS.accent, color: "#ffffff", border: "none" },
    ghost: { backgroundColor: "transparent", color: COLORS.text_secondary, border: "none" },
    destructive: { backgroundColor: COLORS.error, color: "#ffffff", border: "none" },
    outline: { backgroundColor: "transparent", color: COLORS.text_primary, border: `1px solid ${COLORS.border}` },
    secondary: { backgroundColor: COLORS.bg_hover, color: COLORS.text_primary, border: `1px solid ${COLORS.border}` },
  };
  return (
    <button className={`${baseStyle} ${className}`} style={variants[variant] || variants.default} {...props}>
      {children}
    </button>
  );
}
