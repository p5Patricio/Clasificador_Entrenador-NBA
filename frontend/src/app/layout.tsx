import type { Metadata } from "next";
import Link from "next/link";
import { Geist, Geist_Mono } from "next/font/google";
import "./globals.css";

const geistSans = Geist({
  variable: "--font-geist-sans",
  subsets: ["latin"],
});

const geistMono = Geist_Mono({
  variable: "--font-geist-mono",
  subsets: ["latin"],
});

export const metadata: Metadata = {
  title: "NBA Player Analyzer",
  description: "Clustering, análisis y reportes de jugadores NBA",
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  const apiDocsHref = `${process.env.NEXT_PUBLIC_API_BASE || "http://localhost:8000"}/docs`;
  return (
    <html lang="es">
      <body className={`${geistSans.variable} ${geistMono.variable} antialiased bg-gradient-to-b from-black via-slate-950 to-[#0b1020] text-slate-100`}>        
        <header className="fixed inset-x-0 top-0 z-30 border-b border-blue-900/40 bg-black/80 backdrop-blur">
          <nav className="mx-auto flex h-14 max-w-7xl items-center justify-between px-6">
            <Link href="/" className="text-lg font-semibold tracking-tight text-sky-400">
              NBA Analyzer
            </Link>
            <div className="flex items-center gap-6 text-sm">
              <Link href="/" className="text-slate-300 hover:text-white">Home</Link>
              <Link href="/teams" className="text-slate-300 hover:text-white">Equipos</Link>
              <a href={apiDocsHref} className="hidden text-slate-400 hover:text-white sm:inline" target="_blank" rel="noreferrer">Docs API</a>
            </div>
          </nav>
        </header>
        <main className="mx-auto max-w-7xl px-6 pb-16 pt-20">{children}</main>
        <footer className="mx-auto max-w-7xl px-6 pb-8 pt-6 text-xs text-slate-400/80">
          Hecho con Next.js y FastAPI · <span className="text-sky-400">negro + azul</span>
        </footer>
      </body>
    </html>
  );
}
