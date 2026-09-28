"use client";
import React from "react";
import { COLORS } from "@/lib/constants";

export function Input(props: React.InputHTMLAttributes<HTMLInputElement> & { className?: string }) {
  const { className = "", ...rest } = props;
  return (
    <input
      className={`text-xs px-2 py-1.5 rounded border w-full ${className}`}
      style={{
        backgroundColor: COLORS.bg_surface,
        color: COLORS.text_primary,
        borderColor: COLORS.border,
      }}
      {...rest}
    />
  );
}
