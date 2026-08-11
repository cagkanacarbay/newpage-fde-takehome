import type { Metadata } from "next";
import { Red_Hat_Display } from "next/font/google";
import type { ReactNode } from "react";

import "./globals.css";

const redHatDisplay = Red_Hat_Display({
  subsets: ["latin"],
  weight: ["400", "500", "600", "700"],
  variable: "--font-red-hat-display",
  fallback: [
    "-apple-system",
    "BlinkMacSystemFont",
    "Segoe UI",
    "Roboto",
    "Helvetica Neue",
    "Arial",
    "sans-serif",
  ],
});

export const metadata: Metadata = {
  title: "Live Long R&D",
  description: "A research assistant for longevity science.",
};

export default function RootLayout({ children }: Readonly<{ children: ReactNode }>) {
  return (
    <html lang="en" className={redHatDisplay.variable}>
      <body>{children}</body>
    </html>
  );
}
