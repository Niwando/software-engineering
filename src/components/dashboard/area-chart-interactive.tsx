"use client"
import * as React from "react"
import { Area, AreaChart, CartesianGrid, XAxis, YAxis } from "recharts"
import {
  Card,
  CardContent,
  CardDescription,
  CardHeader,
  CardTitle,
} from "@/components/ui/card"
import {
  ChartConfig,
  ChartContainer,
  ChartTooltip,
  ChartTooltipContent,
} from "@/components/ui/chart"
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select"
import { Check, ChevronsUpDown } from "lucide-react"
import { cn } from "@/lib/utils"
import { Button } from "@/components/ui/button"
import {
  Command,
  CommandEmpty,
  CommandGroup,
  CommandInput,
  CommandItem,
  CommandList,
} from "@/components/ui/command"
import {
  Popover,
  PopoverContent,
  PopoverTrigger,
} from "@/components/ui/popover"

// ---------- Typen ----------
interface ChartDataItem {
  date: string | number
  value: number
  stock: string
}

interface AreaChartInteractiveProps {
  chartData: ChartDataItem[]
}

/** Constants für die Komprimierung */
const OFF_HOURS_BETWEEN_CLOSE_AND_NEXT_OPEN = 60
const MINUTES_PER_TRADING_DAY = 390
const DAY_BLOCK = MINUTES_PER_TRADING_DAY + OFF_HOURS_BETWEEN_CLOSE_AND_NEXT_OPEN

/** Aktien-Auswahl */
const stocks = [
  { value: "GOOGL", label: "GOOGL | Alphabet" },
  { value: "AMZN",  label: "AMZN | Amazon" },
  { value: "AAPL",  label: "AAPL | Apple" },
  { value: "AVGO",  label: "AVGO | Broadcom" },
  { value: "META",  label: "META | Meta Platforms" },
  { value: "MSFT",  label: "MSFT | Microsoft" },
  { value: "NFLX",  label: "NFLX | Netflix" },
  { value: "NVDA",  label: "NVDA | Nvidia" },
  { value: "PYPL",  label: "PYPL | PayPal" },
  { value: "TSLA",  label: "TSLA | Tesla" },
]

/** Prüft, ob Montag–Freitag (UTC) */
function isTradingDay(date: Date) {
  const day = date.getUTCDay()
  return day !== 0 && day !== 6
}

/** Prüft, ob Uhrzeit (UTC) zwischen 9:30 und 16:00 liegt */
function isTradingHour(date: Date) {
  const h = date.getUTCHours()
  const m = date.getUTCMinutes()
  if (h < 9 || (h === 9 && m < 30)) return false
  if (h > 16 || (h === 16 && m > 0)) return false
  return true
}

/** Minuten seit Mitternacht UTC */
function minutesFromMidnightUTC(d: Date) {
  return d.getUTCHours() * 60 + d.getUTCMinutes()
}

/** Korrigiert Zeit auf 9:30 bzw. 16:00 UTC */
function clampToTradingSession(d: Date, mode: "start" | "end") {
  const h = d.getUTCHours()
  const m = d.getUTCMinutes()
  if (mode === "start") {
    if (h < 9 || (h === 9 && m < 30)) {
      d.setUTCHours(9, 30, 0, 0)
    }
  } else {
    if (h > 16 || (h === 16 && m > 0)) {
      d.setUTCHours(16, 0, 0, 0)
    }
  }
}

/** Anzahl Handelstage zwischen refMid und dayMid (exklusive) */
function getDayIndex(refMid: Date, dayMid: Date): number {
  if (dayMid <= refMid) return 0
  let dayIdx = 0
  const c = new Date(refMid)
  while (c < dayMid) {
    if (isTradingDay(c)) {
      dayIdx++
    }
    c.setUTCDate(c.getUTCDate() + 1)
  }
  return dayIdx
}

/** Minuten zwischen 9:30 UTC und d (max. 390) */
function partialInDay(d: Date): number {
  if (!isTradingDay(d)) return 0
  const mm = minutesFromMidnightUTC(d)
  if (mm < 570) return 0
  if (mm > 960) return 390
  return mm - 570
}

/** Gleicher Kalendertag in UTC? */
function sameCalendarDayUTC(a: Date, b: Date) {
  return (
    a.getUTCFullYear() === b.getUTCFullYear() &&
    a.getUTCMonth() === b.getUTCMonth() &&
    a.getUTCDate() === b.getUTCDate()
  )
}

// ------------------ COMPRESS/EXPAND LOGIC ------------------

function compressDateToTradingOffset(targetDate: Date, referenceDate: Date): number {
  const ref = new Date(referenceDate)
  clampToTradingSession(ref, "start")
  const tgt = new Date(targetDate)
  clampToTradingSession(tgt, "end")
  if (ref >= tgt) return 0

  // Mitternacht UTC
  const refMid = new Date(ref)
  refMid.setUTCHours(0, 0, 0, 0)
  const tgtMid = new Date(tgt)
  tgtMid.setUTCHours(0, 0, 0, 0)

  const dIdx = getDayIndex(refMid, tgtMid)
  let offset = dIdx * DAY_BLOCK
  let partial = partialInDay(tgt)
  if (sameCalendarDayUTC(refMid, tgtMid)) {
    partial -= partialInDay(ref)
    if (partial < 0) partial = 0
  }
  offset += partial
  return offset > 0 ? offset : 0
}

function expandTradingOffsetToDate(tradingOffset: number, referenceDate: Date): Date {
  const ref = new Date(referenceDate)
  clampToTradingSession(ref, "start")
  if (tradingOffset <= 0) return ref

  const fullDays = Math.floor(tradingOffset / DAY_BLOCK)
  let leftover = tradingOffset % DAY_BLOCK
  const refMid = new Date(ref)
  refMid.setUTCHours(0, 0, 0, 0)
  const cursor = new Date(refMid)
  let usedDays = 0
  while (usedDays < fullDays) {
    if (isTradingDay(cursor)) {
      usedDays++
    }
    cursor.setUTCDate(cursor.getUTCDate() + 1)
  }
  if (leftover < 0) leftover = 0
  const dayMid = new Date(cursor)
  dayMid.setUTCHours(0, 0, 0, 0)
  if (sameCalendarDayUTC(dayMid, refMid)) {
    return addLeftoverInSameDay(ref, leftover)
  } else {
    while (!isTradingDay(cursor)) {
      cursor.setUTCDate(cursor.getUTCDate() + 1)
    }
    const startDay = new Date(cursor)
    startDay.setUTCHours(9, 30, 0, 0)
    return addLeftoverInSameDay(startDay, leftover)
  }
}

function addLeftoverInSameDay(baseDate: Date, leftover: number): Date {
  const r = new Date(baseDate)
  while (leftover > 0) {
    r.setUTCMinutes(r.getUTCMinutes() + 1)
    if (isTradingDay(r) && isTradingHour(r)) {
      leftover--
    }
  }
  return r
}

/** ChartConfig für Recharts */
const chartConfig = {
  visitors: { label: "Visitors" },
  value: { label: "Value", color: "hsl(var(--chart-1))" },
} satisfies ChartConfig

// ----------------- AGGREGATORS -----------------
function aggregateDataByDayLast(data: { date: number; value: number }[]) {
  const ag: { date: number; value: number }[] = []
  let currentDay: number | null = null
  let lastValue: { date: number; value: number } | null = null
  data.forEach((item) => {
    const d = new Date(item.date)
    const day = d.getUTCDate()
    if (currentDay === null || day !== currentDay) {
      if (lastValue) ag.push(lastValue)
      currentDay = day
    }
    lastValue = item
  })
  if (lastValue) ag.push(lastValue)
  return ag
}

function aggregateDataByDayFirst(data: { date: number; value: number }[]) {
  const ag: { date: number; value: number }[] = []
  let currentDay: number | null = null
  let firstValue: { date: number; value: number } | null = null
  data.forEach((item) => {
    const d = new Date(item.date)
    const day = d.getUTCDate()
    if (currentDay === null || day !== currentDay) {
      if (firstValue) ag.push(firstValue)
      currentDay = day
      firstValue = item
    }
  })
  if (firstValue) ag.push(firstValue)
  return ag
}

function aggregateDataHourCustom(data: { date: number; value: number }[]) {
  const bucketMap: Record<number, { date: number; value: number }> = {}
  for (const item of data) {
    const d = new Date(item.date)
    const bucketDate = getHourBucket(d)
    const bucketTs = bucketDate.getTime()
    bucketMap[bucketTs] = item
  }
  return Object.keys(bucketMap)
    .map((k) => +k)
    .sort((a, b) => a - b)
    .map((k) => bucketMap[k])
}

function getHourBucket(date: Date) {
  const h = date.getUTCHours()
  const m = date.getUTCMinutes()
  if (h === 9 && m === 30) return new Date(date)
  let aggregatorHour = h
  if (m >= 1) {
    aggregatorHour = h + 1
  }
  if (aggregatorHour > 16) {
    aggregatorHour = 16
  }
  const b = new Date(date)
  b.setUTCHours(aggregatorHour, 0, 0, 0)
  return b
}

// ----------------- TICK GENERATORS -----------------

function generateHourlyTicks(startTs: number, endTs: number) {
  const ticks: number[] = []
  const c = new Date(startTs)
  if (c.getUTCMinutes() !== 0) {
    c.setUTCHours(c.getUTCHours() + 1, 0, 0, 0)
  }
  while (c.getTime() <= endTs) {
    ticks.push(c.getTime())
    c.setUTCHours(c.getUTCHours() + 1)
  }
  return ticks
}

function generateDailyTicks(minTs: number, maxTs: number) {
  const ticks: number[] = []
  const c = new Date(minTs)
  c.setUTCHours(9, 30, 0, 0)
  while (c.getTime() <= maxTs) {
    ticks.push(c.getTime())
    c.setUTCDate(c.getUTCDate() + 1)
    c.setUTCHours(9, 30, 0, 0)
  }
  return ticks
}

/** Ein Tick pro Woche (z. B. immer Montag 9:30 UTC) */
function generateWeeklyTicks(minTs: number, maxTs: number): number[] {
  const ticks: number[] = []
  const c = new Date(minTs)
  while (c.getUTCDay() !== 1) {
    c.setUTCDate(c.getUTCDate() + 1)
  }
  c.setUTCHours(9, 30, 0, 0)
  while (c.getTime() <= maxTs) {
    ticks.push(c.getTime())
    c.setUTCDate(c.getUTCDate() + 7)
  }
  return ticks
}

function generateMonthlyTicks(startTS: number, endTS: number): number[] {
  const ticks: number[] = []
  const c = new Date(startTS)
  c.setUTCDate(1)
  c.setUTCHours(9, 30, 0, 0)
  if (c.getTime() < startTS) {
    c.setUTCMonth(c.getUTCMonth() + 1, 1)
    c.setUTCHours(9, 30, 0, 0)
  }
  while (!isTradingDay(c)) {
    c.setUTCDate(c.getUTCDate() + 1)
    c.setUTCHours(9, 30, 0, 0)
  }
  while (c.getTime() <= endTS) {
    ticks.push(c.getTime())
    c.setUTCMonth(c.getUTCMonth() + 1, 1)
    c.setUTCHours(9, 30, 0, 0)
    while (!isTradingDay(c)) {
      c.setUTCDate(c.getUTCDate() + 1)
      c.setUTCHours(9, 30, 0, 0)
    }
  }
  return ticks
}

function generateYearlyTicks(startTS: number, endTS: number): number[] {
  const ticks: number[] = []
  let c = new Date(startTS)
  c = new Date(Date.UTC(c.getUTCFullYear(), 0, 1, 9, 30, 0, 0))
  if (c.getTime() < startTS) {
    c.setUTCFullYear(c.getUTCFullYear() + 1)
  }
  while (!isTradingDay(c)) {
    c.setUTCDate(c.getUTCDate() + 1)
    c.setUTCHours(9, 30, 0, 0)
  }
  while (c.getTime() <= endTS) {
    ticks.push(c.getTime())
    c.setUTCFullYear(c.getUTCFullYear() + 1, 0, 1)
    c.setUTCHours(9, 30, 0, 0)
    while (!isTradingDay(c)) {
      c.setUTCDate(c.getUTCDate() + 1)
      c.setUTCHours(9, 30, 0, 0)
    }
  }
  return ticks
}

// ---------------------------------------------
// FINAL TICKS-Generierung: Duplikate anhand formatierten Labels entfernen
function getFinalTicks(rawTicks: number[], reference: Date): number[] {
  const tickMap = new Map<string, number>()
  rawTicks.forEach((tick) => {
    const label = finalTickFormatter(tick)
    if (!tickMap.has(label)) {
      tickMap.set(label, tick)
    }
  })
  return Array.from(tickMap.values()).sort((a, b) => a - b)
}

// ---------------------------------------------
// MAIN CHART COMPONENT
// ---------------------------------------------
export function AreaChartInteractive({ chartData }: AreaChartInteractiveProps) {
  const [timeRange, setTimeRange] = React.useState("1d")
  const [open, setOpen] = React.useState(false)
  const [selectedStock, setSelectedStock] = React.useState("GOOGL")

  // 1) Eindeutige Datumswerte (UTC) extrahieren
  const uniqueDateStrings = [...new Set(chartData.map((item) => item.date.split("T")[0]))]
  const sortedUniqueDates = uniqueDateStrings
    .map((d) => new Date(d))
    .sort((a, b) => a.getTime() - b.getTime())

  const lastUniqueDate = sortedUniqueDates[sortedUniqueDates.length - 1]

  // Für "5d": letzten 5 Tage
  let allowedDateStrings: Set<string> | null = null
  if (timeRange === "5d") {
    allowedDateStrings = new Set(
      sortedUniqueDates.slice(-5).map((d) => d.toISOString().split("T")[0])
    )
  }

  // 2) Daten filtern
  const filteredData = chartData.filter((item) => {
    if (item.stock !== selectedStock) return false

    const itemDateString = item.date.split("T")[0]
    const itemDate = new Date(itemDateString)

    if (timeRange === "1d") {
      const lastDateString = lastUniqueDate.toISOString().split("T")[0]
      if (itemDateString !== lastDateString) return false
      const dateObj = new Date(item.date)
      const hours = dateObj.getUTCHours()
      const minutes = dateObj.getUTCMinutes()
      const timeInMinutes = hours * 60 + minutes
      return timeInMinutes >= 570 && timeInMinutes <= 960
    } else if (timeRange === "5d") {
      return allowedDateStrings?.has(itemDateString)
    } else if (timeRange === "1m") {
      const oneMonthAgo = new Date(lastUniqueDate)
      oneMonthAgo.setUTCMonth(oneMonthAgo.getUTCMonth() - 1)
      return itemDate >= oneMonthAgo && itemDate <= lastUniqueDate
    } else if (timeRange === "3m") {
      const threeMonthsAgo = new Date(lastUniqueDate)
      threeMonthsAgo.setUTCMonth(threeMonthsAgo.getUTCMonth() - 3)
      return itemDate >= threeMonthsAgo && itemDate <= lastUniqueDate
    } else if (timeRange === "1y") {
      const oneYearAgo = new Date(lastUniqueDate)
      oneYearAgo.setUTCFullYear(oneYearAgo.getUTCFullYear() - 1)
      return itemDate >= oneYearAgo && itemDate <= lastUniqueDate
    } else if (timeRange === "allTime") {
      return true
    }
    return false
  })

  // 3) Aggregation
  let aggregatedData: { date: number; value: number }[]
  if (timeRange === "5d") {
    aggregatedData = aggregateDataHourCustom(filteredData)
  } else if (timeRange === "1m") {
    aggregatedData = aggregateDataByDayFirst(filteredData)
  } else if (timeRange === "3m" || timeRange === "1y" || timeRange === "allTime") {
    aggregatedData = aggregateDataByDayLast(filteredData)
  } else {
    aggregatedData = filteredData
  }

  // 4) Farbwahl
  const firstValue = aggregatedData.length ? aggregatedData[0].value : 0
  const lastValue = aggregatedData.length ? aggregatedData[aggregatedData.length - 1].value : 0
  const gradientColor = lastValue >= firstValue ? "green" : "red"

  // 5) Build compressedData
  let compressedData: { x: number; value: number; realDate: number }[] = []
  if (aggregatedData.length) {
    const refDate = new Date(aggregatedData[0].date)
    compressedData = aggregatedData.map((item) => {
      const realDate = new Date(item.date)
      const offset = compressDateToTradingOffset(realDate, refDate)
      return {
        x: offset,
        value: item.value,
        realDate: item.date,
      }
    })
  }
  const dataMinOffset = compressedData.length ? compressedData[0].x : 0
  const dataMaxOffset = compressedData.length ? compressedData[compressedData.length - 1].x : 0

  // 6) Ticks
  const realMin = aggregatedData[0]?.date ?? 0
  const realMax = aggregatedData[aggregatedData.length - 1]?.date ?? 0

  function compressTicks(realTickArray: number[], reference: Date) {
    return realTickArray.map((ts) => compressDateToTradingOffset(new Date(ts), reference))
  }

  let realTicks: number[] | undefined
  if (timeRange === "1d") {
    const todayStart = new Date(realMin)
    todayStart.setUTCHours(9, 30, 0, 0)
    const todayEnd = new Date(realMax)
    todayEnd.setUTCHours(16, 0, 0, 0)
    realTicks = generateHourlyTicks(todayStart.getTime(), todayEnd.getTime())
  } else if (timeRange === "5d") {
    realTicks = generateDailyTicks(realMin, realMax)
  } else if (timeRange === "1m") {
    realTicks = generateWeeklyTicks(realMin, realMax)
  } else if (timeRange === "3m" || timeRange === "1y") {
    realTicks = generateMonthlyTicks(realMin, realMax)
  } else if (timeRange === "allTime") {
    realTicks = generateYearlyTicks(realMin, realMax)
  }

  let finalTicks: number[] | undefined
  if (realTicks && aggregatedData.length) {
    const refDate = new Date(aggregatedData[0].date)
    const rawTicks = compressTicks(realTicks, refDate)
    // Filtere Duplikate anhand der von finalTickFormatter erzeugten Labels
    const tickMap = new Map<string, number>()
    rawTicks.forEach((tick) => {
      const label = finalTickFormatter(tick)
      if (!tickMap.has(label)) {
        tickMap.set(label, tick)
      }
    })
    finalTicks = Array.from(tickMap.values()).sort((a, b) => a - b)
  }

  // 7) Y-Axis Domain
  const yValues = compressedData.map((item) => item.value)
  const yMin = Math.min(...yValues)
  const yMax = Math.max(...yValues)
  const yPadding = (yMax - yMin) * 0.1
  const domain = [yMin - yPadding, yMax + yPadding]

  // 8) Formatter-Funktionen
  function finalTickFormatter(offsetValue: number) {
    if (!aggregatedData.length) return ""
    const date = expandTradingOffsetToDate(offsetValue, new Date(aggregatedData[0].date))
    switch (timeRange) {
      case "1d":
        return date.toLocaleTimeString("en-US", { hour: "numeric", timeZone: "UTC" })
      case "5d":
        return date.toLocaleDateString("en-US", { month: "short", day: "numeric", timeZone: "UTC" })
      case "1m":
        return date.toLocaleDateString("en-US", { weekday: "short", timeZone: "UTC" })
      case "3m":
      case "1y":
        return date.toLocaleDateString("en-US", { month: "short", timeZone: "UTC" })
      case "allTime":
        return date.toLocaleDateString("en-US", { year: "numeric", timeZone: "UTC" })
      default:
        return date.toLocaleDateString("en-US", { timeZone: "UTC" })
    }
  }

  function finalLabelFormatter(offsetValue: number) {
    if (!aggregatedData.length) return ""
    const date = expandTradingOffsetToDate(offsetValue, new Date(aggregatedData[0].date))
    if (timeRange === "1d" || timeRange === "5d") {
      return date.toLocaleString("en-US", {
        month: "short",
        day: "numeric",
        hour: "2-digit",
        minute: "2-digit",
        timeZone: "UTC",
      })
    }
    return date.toLocaleDateString("en-US", {
      month: "short",
      day: "numeric",
      year: timeRange === "allTime" ? "numeric" : undefined,
      timeZone: "UTC",
    })
  }

  function finalFormatter(value: number, name: string, props: { payload?: { x?: number } }) {
    const offsetValue = props?.payload?.x
    if (!aggregatedData.length || offsetValue == null) return `${value}`
    const date = expandTradingOffsetToDate(offsetValue, new Date(aggregatedData[0].date))
    if (timeRange === "1d" || timeRange === "5d") {
      return (
        <div className="flex">
          <div className="h-full w-2 mr-1.5" style={{ backgroundColor: gradientColor, borderRadius: "10px" }}></div>
          <div>
            <div className="text-base">{`${value} USD`}</div>
            <div>
              {date.toLocaleString("en-US", {
                month: "short",
                day: "numeric",
                hour: "2-digit",
                minute: "2-digit",
                timeZone: "UTC",
              })}
            </div>
          </div>
        </div>
      )
    }
    return (
      <div className="flex">
        <div className="h-full w-2 mr-1.5" style={{ backgroundColor: gradientColor, borderRadius: "10px" }}></div>
        <div>
          <div className="text-base">{`${value} USD`}</div>
          <div>
            {date.toLocaleDateString("en-US", {
              month: "short",
              day: "numeric",
              year: "numeric",
              timeZone: "UTC",
            })}
          </div>
        </div>
      </div>
    )
  }

  return (
    <Card className="m-8 mt-0">
      <CardHeader className="flex items-center gap-2 space-y-0 border-b py-5 sm:flex-row">
        <div className="grid flex-1 gap-1 text-center sm:text-left">
          <CardTitle>Stock Analysis</CardTitle>
          <CardDescription>
            <div>Overview of the stock’s performance. Select a stock and a time range</div>
            <div>and hover over the chart for more detailed information.</div>
          </CardDescription>
        </div>
        <Popover open={open} onOpenChange={setOpen}>
          <PopoverTrigger asChild>
            <Button variant="outline" role="combobox" aria-expanded={open} className="w-[200px] justify-between">
              {selectedStock
                ? stocks.find((stock) => stock.value === selectedStock)?.label
                : "Select stock..."}
              <ChevronsUpDown className="opacity-50" />
            </Button>
          </PopoverTrigger>
          <PopoverContent className="w-[200px] p-0">
            <Command>
              <CommandInput placeholder="Search stock..." />
              <CommandList>
                <CommandEmpty>No stock found.</CommandEmpty>
                <CommandGroup>
                  {stocks.map((stock) => (
                    <CommandItem
                      key={stock.value}
                      value={stock.value}
                      onSelect={(currentValue) => {
                        setSelectedStock(currentValue)
                        setOpen(false)
                      }}
                    >
                      {stock.label}
                      <Check
                        className={cn("ml-auto", selectedStock === stock.value ? "opacity-100" : "opacity-0")}
                      />
                    </CommandItem>
                  ))}
                </CommandGroup>
              </CommandList>
            </Command>
          </PopoverContent>
        </Popover>
        <Select value={timeRange} onValueChange={setTimeRange}>
          <SelectTrigger className="w-[160px] rounded-lg sm:ml-auto">
            <SelectValue placeholder="Last day" />
          </SelectTrigger>
          <SelectContent className="rounded-xl">
            <SelectItem value="1d">Last day</SelectItem>
            <SelectItem value="5d">Last 5 days</SelectItem>
            <SelectItem value="1m">Last 1 month</SelectItem>
            <SelectItem value="3m">Last 3 months</SelectItem>
            <SelectItem value="1y">Last 1 year</SelectItem>
            <SelectItem value="allTime">All Time</SelectItem>
          </SelectContent>
        </Select>
      </CardHeader>

      <CardContent className="px-2 pt-4 sm:px-6 sm:pt-6">
        <ChartContainer config={chartConfig} className="aspect-auto h-[500px] w-full">
          <AreaChart data={compressedData} margin={{ top: 20, bottom: 20 }}>
            <defs>
              <linearGradient id="fillValue" x1="0" y1="0" x2="0" y2="1">
                <stop offset="5%" stopColor={gradientColor} stopOpacity={0.8} />
                <stop offset="95%" stopColor={gradientColor} stopOpacity={0.1} />
              </linearGradient>
            </defs>
            <CartesianGrid vertical={false} />
            <XAxis
              dataKey="x"
              type="number"
              scale="linear"
              domain={[dataMinOffset, dataMaxOffset]}
              ticks={finalTicks}
              tickFormatter={finalTickFormatter}
            />
            <YAxis
              domain={domain}
              tickFormatter={(value) => Number(value).toFixed(2)}
              tick={{ style: { whiteSpace: "nowrap" } }}
            />
            <ChartTooltip
              cursor={false}
              content={
                <ChartTooltipContent
                  labelFormatter={finalLabelFormatter}
                  formatter={finalFormatter}
                  indicator="line"
                />
              }
            />
            <Area dataKey="value" type="linear" stroke={gradientColor} fill="url(#fillValue)" />
          </AreaChart>
        </ChartContainer>
      </CardContent>
    </Card>
  )
}
