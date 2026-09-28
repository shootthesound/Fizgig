'use client';

import React, { useRef } from 'react';
import { COLORS } from '@/lib/constants';

interface BaseFieldProps {
  label: string;
  id?: string;
}

export function FormRow({ label, id, children, className = "" }: BaseFieldProps & { children: React.ReactNode; className?: string }) {
  return (
    <div className="flex flex-row items-center w-full py-1">
      <label 
        htmlFor={id}
        className="w-[180px] text-[10pt] pl-3 pr-2"
        style={{
          color: COLORS.text_secondary,
          backgroundColor: 'transparent',
          fontFamily: 'Segoe UI'
        }}
      >
        {label}:
      </label>
      <div className="flex-1 px-1">
        {children}
      </div>
    </div>
  );
}

interface TextInputProps extends React.InputHTMLAttributes<HTMLInputElement> {
  label: string;
}

export function TextInput({ label, id, ...props }: TextInputProps) {
  const [isFocused, setIsFocused] = React.useState(false);

  return (
    <FormRow label={label} id={id}>
      <input
        id={id}
        type="text"
        className="w-full text-[10pt] px-2 py-1 border border-solid outline-none transition-colors"
        style={{
          backgroundColor: isFocused ? '#FFFFFF' : COLORS.bg_surface,
          color: isFocused ? '#000000' : COLORS.text_primary,
          borderColor: isFocused ? COLORS.border_focus : COLORS.border,
          fontFamily: 'Segoe UI'
        }}
        onFocus={(e) => {
          setIsFocused(true);
          props.onFocus?.(e);
        }}
        onBlur={(e) => {
          setIsFocused(false);
          props.onBlur?.(e);
        }}
        {...props}
      />
    </FormRow>
  );
}

interface ComboboxProps extends React.SelectHTMLAttributes<HTMLSelectElement> {
  label: string;
  options: string[];
}

export function Combobox({ label, id, options, ...props }: ComboboxProps) {
  const [isFocused, setIsFocused] = React.useState(false);

  return (
    <FormRow label={label} id={id}>
      <select
        id={id}
        className="w-full text-[10pt] px-2 py-1 border border-solid outline-none cursor-pointer"
        style={{
          backgroundColor: isFocused ? '#FFFFFF' : COLORS.bg_surface,
          color: isFocused ? '#000000' : COLORS.text_primary,
          borderColor: isFocused ? COLORS.border_focus : COLORS.border,
          fontFamily: 'Segoe UI'
        }}
        onFocus={(e) => {
          setIsFocused(true);
          props.onFocus?.(e);
        }}
        onBlur={(e) => {
          setIsFocused(false);
          props.onBlur?.(e);
        }}
        {...props}
      >
        {options.map((opt) => (
          <option key={opt} value={opt}>{opt}</option>
        ))}
      </select>
    </FormRow>
  );
}

interface CheckboxProps extends React.InputHTMLAttributes<HTMLInputElement> {
  label: string;
}

export function Checkbox({ label, id, ...props }: CheckboxProps) {
  return (
    <FormRow label={label} id={id}>
      <div className="flex items-center">
        <input
          id={id}
          type="checkbox"
          className="w-4 h-4 cursor-pointer outline-none"
          style={{ accentColor: COLORS.accent }}
          {...props}
        />
      </div>
    </FormRow>
  );
}

interface FilePickerProps extends TextInputProps {
  onBrowse: () => void;
  type?: 'file' | 'directory';
}

export function FilePicker({ label, id, onBrowse, type = 'file', ...props }: FilePickerProps) {
  const [isFocused, setIsFocused] = React.useState(false);
  const [isHovered, setIsHovered] = React.useState(false);
  
  return (
    <FormRow label={label} id={id}>
      <div className="flex flex-row items-center gap-2">
        <input
          id={id}
          type="text"
          className="flex-1 text-[10pt] px-2 py-1 border border-solid outline-none transition-colors"
          style={{
            backgroundColor: isFocused ? '#FFFFFF' : COLORS.bg_surface,
            color: isFocused ? '#000000' : COLORS.text_primary,
            borderColor: isFocused ? COLORS.border_focus : COLORS.border,
            fontFamily: 'Segoe UI'
          }}
          onFocus={(e) => {
            setIsFocused(true);
            props.onFocus?.(e);
          }}
          onBlur={(e) => {
            setIsFocused(false);
            props.onBlur?.(e);
          }}
          {...props}
        />
        <button
          type="button"
          onClick={onBrowse}
          onMouseEnter={() => setIsHovered(true)}
          onMouseLeave={() => setIsHovered(false)}
          className="text-[10pt] font-bold px-4 py-2 border border-solid outline-none cursor-pointer transition-colors"
          style={{
            backgroundColor: isHovered ? COLORS.bg_hover : COLORS.bg_surface,
            color: COLORS.text_primary,
            borderColor: COLORS.border,
            fontFamily: 'Segoe UI'
          }}
        >
          Browse
        </button>
      </div>
    </FormRow>
  );
}

export default FormRow;
export { FormRow as FormField };

// Also export FormRow as default FormField alias for compatibility
