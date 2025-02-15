import { env } from "@/env";


export type SiteConfig = typeof siteConfig

export const siteConfig = {
  url:
    env.NEXT_PUBLIC_NODE_ENV === "development"
      ? "https://localhost:3000"
      : "https://pecunia-ai.vercel.app/",
}