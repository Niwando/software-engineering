import { AppSidebar } from "@/src/components/sidebar/app-sidebar"
import {
  SidebarInset,
  SidebarProvider
} from "@/src/components/ui/sidebar"
import { AreaChartInteractive } from "@/src/components/dashboard/area-chart-interactive"

export default function Page() {
  return (
    <SidebarProvider
      style={
        {
          "--sidebar-width": "350px",
        } as React.CSSProperties
      }
    >
      <AppSidebar />
      <SidebarInset>
        <AreaChartInteractive/>
        <div className="flex flex-1 flex-col gap-4 p-4">
          {Array.from({ length: 24 }).map((_, index) => (
            <div
              key={index}
              className="aspect-video h-12 w-full rounded-lg bg-muted/50"
            />
          ))}
        </div>
      </SidebarInset>
    </SidebarProvider>
  )
}