"use client";
import React from "react";
import { COLORS } from "@/lib/constants";

export function Card({ children, className = "" }: { children: React.ReactNode; className?: string }) {
  return (
    <div
      className={`rounded-lg border p-4 ${className}`}
      style={{ backgroundColor: COLORS.bg_surface, borderColor: COLORS.border }}
    >
      {children}
    </div>
  );
}

export function CardHeader({ children, className = "" }: { children: React.ReactNode; className?: string }) {
  return <div className={`mb-3 ${className}`}>{children}</div>;
}

export function CardTitle({ children, className = "" }: { children: React.ReactNode; className?: string }) {
  return (
    <h3 className={`text-sm font-bold ${className}`} style={{ color: COLORS.text_primary }}>
      {children}
    </h3>
  );
}

export function CardDescription({ children, className = "" }: { children: React.ReactNode; className?: string }) {
  return (
    <p className={`text-xs mt-1 ${className}`} style={{ color: COLORS.text_explain }}>
      {children}
    </p>
  );
}

export function CardContent({ children, className = "" }: { children?: React.ReactNode; className?: string }) {
  return <div className={className}>{children}</div>;
}
