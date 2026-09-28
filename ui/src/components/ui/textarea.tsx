"use client";
import React from "react";
import { COLORS } from "@/lib/constants";

export function Textarea(props: React.TextareaHTMLAttributes<HTMLTextAreaElement> & { className?: string }) {
  const { className = "", ...rest } = props;
  return (
    <textarea
      className={`text-xs px-2 py-1.5 rounded border w-full resize-y ${className}`}
      style={{
        backgroundColor: COLORS.bg_surface,
        color: COLORS.text_primary,
        borderColor: COLORS.border,
      }}
      {...rest}
    />
  );
}
