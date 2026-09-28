"use client";
import React, { useState } from "react";
import { COLORS } from "@/lib/constants";

export function Collapsible({ children, open, onOpenChange }: {
  children: React.ReactNode;
  open?: boolean;
  onOpenChange?: (open: boolean) => void;
}) {
  return <div>{children}</div>;
}

export function CollapsibleTrigger({ children, asChild, className = "" }: {
  children: React.ReactNode;
  asChild?: boolean;
  className?: string;
}) {
  return <div className={className}>{children}</div>;
}

export function CollapsibleContent({ children, className = "" }: {
  children: React.ReactNode;
  className?: string;
}) {
  return <div className={className}>{children}</div>;
}
