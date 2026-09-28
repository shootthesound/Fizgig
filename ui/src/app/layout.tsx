import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  // Line 1566 of jj.py
  title: "Fizgig — Klein 9B & Krea 2 LoRA Studio",
  description: "A focused, local trainer and workbench for Flux 2 Klein 9B, Krea 2 and MiniMax H3 LoRAs",
};

export default function RootLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <html lang="en" suppressHydrationWarning>
      <body
        // master.configure(bg=BG_COLOR)  — Line 1569 of jj.py
        style={{ backgroundColor: "#1E2530", color: "#F0F4F8" }}
        className="min-h-screen antialiased"
        suppressHydrationWarning
      >
        {children}
      </body>
    </html>
  );
}
