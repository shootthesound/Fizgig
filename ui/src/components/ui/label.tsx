"use client";
import React from "react";
import { COLORS } from "@/lib/constants";

export function Label({ children, htmlFor, className = "" }: { children: React.ReactNode; htmlFor?: string; className?: string }) {
  return (
    <label htmlFor={htmlFor} className={`text-xs font-medium ${className}`} style={{ color: COLORS.text_secondary }}>
      {children}
    </label>
  );
}
