import Link from "next/link"
import { cn } from "@/lib/utils"
import Image from 'next/image';

export function MainNav({
  className,
  ...props
}: React.HTMLAttributes<HTMLElement>) {
  return (
    <nav
      className={cn("flex justify-between items-center w-full", className)}
      {...props}
    >
<div className="flex items-center space-x-2">
      <Image
        src="/logo_transparent_small.png"
        alt="Pecunia Logo"
        className="h-4 w-auto"
      />
      <Link
        href="/dashboard"
        className="text-2xl font-bold transition-colors hover:text-primary"
        style={{ fontFamily: "Century Gothic" }}
      >
        Pecunia
      </Link>
</div>
      <div className="flex items-center justify-center space-x-4">
        <Link
          href="/dashboard"
          className="text-sm font-medium text-muted-foreground transition-coly"
        >
          Dashboard
        </Link>
        <Link
          href="/predictions"
          className="text-sm font-medium transition-colors hover:text-primary"
        >
          AI Predictions
        </Link>
      </div>
    </nav>
  )
}