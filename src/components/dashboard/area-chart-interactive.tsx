"use client"
import * as React from "react"
import { Area, AreaChart, CartesianGrid, XAxis, YAxis } from "recharts"
import {
  Card,
  CardContent,
  CardDescription,
  CardHeader,
  CardTitle,
} from "@/src/components/ui/card"
import {
  ChartConfig,
  ChartContainer,
  ChartTooltip,
  ChartTooltipContent,
} from "@/src/components/ui/chart"
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/src/components/ui/select"
import { Check, ChevronsUpDown } from "lucide-react"
 
import { cn } from "@/lib/utils"
import { Button } from "@/src/components/ui/button"
import {
  Command,
  CommandEmpty,
  CommandGroup,
  CommandInput,
  CommandItem,
  CommandList,
} from "@/src/components/ui/command"
import {
  Popover,
  PopoverContent,
  PopoverTrigger,
} from "@/src/components/ui/popover"

/** If you want 16:00 => next day 9:30 to be offset=60, set this to 60. */
const OFF_HOURS_BETWEEN_CLOSE_AND_NEXT_OPEN = 60
/** Each trading day => 9:30–16:00 => 6.5h =>390 min. */
const MINUTES_PER_TRADING_DAY = 390
/** One “full” day block => 390 + 60 => 450. */
const DAY_BLOCK = MINUTES_PER_TRADING_DAY + OFF_HOURS_BETWEEN_CLOSE_AND_NEXT_OPEN

const stocks = [
  {
    value: "GOOGL",
    label: "GOOGL | Alphabet",
  },
  {
    value: "AMZN",
    label: "AMZN | Amazon",
  },
  {
    value: "AAPL",
    label: "AAPL | Apple",
  },
  {
    value: "AVGO",
    label: "AVGO | Broadcom",
  },
  {
    value: "META",
    label: "META | Meta Platforms",
  },
  {
    value: "MSFT",
    label: "MSFT | Microsoft",
  },
  {
    value: "NFLX",
    label: "NFLX | Netflix",
  },
  {
    value: "NVDA",
    label: "NVDA | Nvidia",
  },
  {
    value: "PYPL",
    label: "PYPL | PayPal",
  },
  {
    value: "TSLA",
    label: "TSLA | Tesla",
  },
]

/** True if Monday–Friday (skipping Sat=6, Sun=0). */
function isTradingDay(date: Date) {
  const day = date.getDay()
  return day !== 0 && day !== 6
}

/** Check if time is between 9:30 and 16:00. */
function isTradingHour(date: Date) {
  const h = date.getHours()
  const m = date.getMinutes()
  if (h < 9 || (h === 9 && m < 30)) return false
  if (h > 16 || (h === 16 && m > 0)) return false
  return true
}

/** 9:30 => 570, 16:00 => 960. */
function minutesFromMidnight(d: Date) {
  return d.getHours() * 60 + d.getMinutes()
}

/** If 'start', clamp to 9:30 if earlier; if 'end', clamp to 16:00 if after. */
function clampToTradingSession(d: Date, mode: "start" | "end") {
  const h = d.getHours(),
    m = d.getMinutes()
  if (mode === "start") {
    if (h < 9 || (h === 9 && m < 30)) {
      d.setHours(9, 30, 0, 0)
    }
  } else {
    if (h > 16 || (h === 16 && m > 0)) {
      d.setHours(16, 0, 0, 0)
    }
  }
}

/**
 * Count how many trading days from refMid up to but not including dayMid’s day
 */
function getDayIndex(refMid: Date, dayMid: Date): number {
  if (dayMid <= refMid) return 0
  let dayIdx = 0
  const c = new Date(refMid)
  while (c < dayMid) {
    if (isTradingDay(c)) {
      dayIdx++
    }
    c.setDate(c.getDate() + 1)
  }
  return dayIdx
}

/**
 * partialInDay(d):
 *   how many minutes from 9:30 to d’s HH:MM (max 390) if d is a trading day
 */
function partialInDay(d: Date): number {
  if (!isTradingDay(d)) return 0
  const mm = minutesFromMidnight(d)
  if (mm < 570) return 0 // <9:30 =>0
  if (mm > 960) return 390 // >16:00 =>390
  return mm - 570 // minutes after 9:30
}

/** True if same calendar date (year,month,day). */
function sameCalendarDay(a: Date, b: Date) {
  return (
    a.getFullYear() === b.getFullYear() &&
    a.getMonth() === b.getMonth() &&
    a.getDate() === b.getDate()
  )
}

// ------------------ COMPRESS/EXPAND LOGIC ------------------

function compressDateToTradingOffset(targetDate: Date, referenceDate: Date): number {
  const ref = new Date(referenceDate)
  clampToTradingSession(ref, "start")

  const tgt = new Date(targetDate)
  clampToTradingSession(tgt, "end")

  if (ref >= tgt) return 0

  const refMid = new Date(ref)
  refMid.setHours(0, 0, 0, 0)

  const tgtMid = new Date(tgt)
  tgtMid.setHours(0, 0, 0, 0)

  // dayIndex => how many trading days from refMid..tgtMid (exclusive)
  const dIdx = getDayIndex(refMid, tgtMid)
  let offset = dIdx * DAY_BLOCK

  // partial day
  let partial = partialInDay(tgt)

  // if same day => partial from ref => subtract partialInDay(ref)
  if (sameCalendarDay(refMid, tgtMid)) {
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
  refMid.setHours(0, 0, 0, 0)

  const cursor = new Date(refMid)
  let usedDays = 0
  while (usedDays < fullDays) {
    if (isTradingDay(cursor)) {
      usedDays++
    }
    cursor.setDate(cursor.getDate() + 1)
  }

  if (leftover < 0) leftover = 0

  const dayMid = new Date(cursor)
  dayMid.setHours(0, 0, 0, 0)

  if (sameCalendarDay(dayMid, refMid)) {
    // same day => leftover from ref
    return addLeftoverInSameDay(ref, leftover)
  } else {
    // skip weekend
    while (!isTradingDay(cursor)) {
      cursor.setDate(cursor.getDate() + 1)
    }
    const startDay = new Date(cursor)
    startDay.setHours(9, 30, 0, 0)
    return addLeftoverInSameDay(startDay, leftover)
  }
}

/** move leftover minutes within a single day from baseDate => forward. */
function addLeftoverInSameDay(baseDate: Date, leftover: number): Date {
  const r = new Date(baseDate)
  while (leftover > 0) {
    r.setMinutes(r.getMinutes() + 1)
    if (isTradingDay(r) && isTradingHour(r)) {
      leftover--
    }
  }
  return r
}

// ----------------- RAW DATA GENERATION -----------------

function generateData(startDate: string, numDays: number, intervalMinutes: number) {
  const data: { date: number; value: number; stock: string }[] = []
  let c = new Date(startDate)
  for (let i = 0; i < numDays; i++) {
    if (isTradingDay(c)) {
      c.setHours(9, 30, 0, 0)
      const e = new Date(c)
      e.setHours(16, 0, 0, 0)
      if (i === numDays - 2) e.setHours(14, 0, 0, 0)
      while (c <= e) {
        data.push({
          date: c.getTime(),
          value: Math.floor(Math.random() * 250 + 50),
          stock: "GOOGL",
        })
        data.push({
          date: c.getTime(),
          value: Math.floor(Math.random() * 750 + 50),
          stock: "MSFT",
        })
        c.setMinutes(c.getMinutes() + intervalMinutes)
      }
      c.setDate(c.getDate() + 1)
    } else {
      // skip weekend
      c.setDate(c.getDate() + 1)
    }
  }
  return data
}

const dataStartDate = new Date()
dataStartDate.setDate(dataStartDate.getDate() - 365)
const chartData = generateData(dataStartDate.toISOString(), 365, 1)

const chartConfig = {
  visitors: { label: "Visitors" },
  value: {
    label: "Value",
    color: "hsl(var(--chart-1))",
  },
} satisfies ChartConfig

// ----------------- AGGREGATORS -----------------
function aggregateDataByDayLast(data: { date: number; value: number }[]) {
  const ag: { date: number; value: number }[] = []
  let currentDay: number | null = null
  let lastValue: { date: number; value: number } | null = null
  data.forEach((item) => {
    const d = new Date(item.date)
    const day = d.getDate()
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
    const day = d.getDate()
    if (currentDay === null || day !== currentDay) {
      if (firstValue) ag.push(firstValue)
      currentDay = day
      firstValue = item
    }
  })
  if (firstValue) ag.push(firstValue)
  return ag
}
function aggregateDataByHourLast(data: { date: number; value: number }[]) {
  const ag: { date: number; value: number }[] = []
  let currentHour: number | null = null
  let lastValue: { date: number; value: number } | null = null
  data.forEach((item) => {
    const hour = new Date(item.date).getHours()
    if (currentHour === null || hour !== currentHour) {
      if (lastValue) ag.push(lastValue)
      currentHour = hour
    }
    lastValue = item
  })
  if (lastValue) ag.push(lastValue)
  return ag
}
/** lumps 9:31..10:00 => aggregator=10:00, etc. */
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
  const h = date.getHours(),
    m = date.getMinutes()
  if (h === 9 && m === 30) return new Date(date)
  let aggregatorHour = h
  if (m >= 1) {
    aggregatorHour = h + 1
  }
  if (aggregatorHour > 16) {
    aggregatorHour = 16
  }
  const b = new Date(date)
  b.setHours(aggregatorHour, 0, 0, 0)
  return b
}

// ----------------- TICK GENERATORS (FIX) -----------------

function generateHourlyTicks(startTs: number, endTs: number) {
  const ticks: number[] = []
  const c = new Date(startTs)
  if (c.getMinutes() !== 0) {
    c.setHours(c.getHours() + 1, 0, 0, 0)
  }
  while (c.getTime() <= endTs) {
    ticks.push(c.getTime())
    c.setHours(c.getHours() + 1)
  }
  return ticks
}

function generateDailyTicks(minTs: number, maxTs: number) {
  const ticks: number[] = []
  const c = new Date(minTs)
  c.setHours(9, 30, 0, 0)
  while (c.getTime() <= maxTs) {
    ticks.push(c.getTime())
    c.setDate(c.getDate() + 1)
    c.setHours(9, 30, 0, 0)
  }
  return ticks
}

/**
 * Generate monthly ticks on the first trading day of each month
 * at 9:30. If the 1st is on a weekend, we skip forward to the
 * next Monday (or next valid trading day).
 */
function generateMonthlyTicks(startTS: number, endTS: number) {
  const ticks: number[] = []

  // Move c to the "first-of-month at 9:30" on or after startTS
  let c = new Date(startTS)
  c.setDate(1)
  c.setHours(23, 59, 59, 99)
  if (c.getTime() < startTS) {
    c.setMonth(c.getMonth() + 1, 1)
    c.setHours(23, 59, 59, 99)
  }
  // If that's a weekend, push forward to a trading day
  while (!isTradingDay(c)) {
    c.setDate(c.getDate() + 1)
  }
  c.setHours(23, 59, 59, 99)

  // Collect ticks
  while (c.getTime() <= endTS) {
    ticks.push(c.getTime())
    // Next month => go to day=1 => then skip weekend
    c.setMonth(c.getMonth() + 1, 1)
    c.setHours(23, 59, 59, 99)
    while (!isTradingDay(c)) {
      c.setDate(c.getDate() + 1)
    }
    c.setHours(23, 59, 59, 99)
  }
  return ticks
}

/**
 * Generate yearly ticks on the first trading day of January
 * at 9:30. If Jan 1 is on a weekend, skip forward to Monday.
 */
function generateYearlyTicks(startTS: number, endTS: number) {
  const ticks: number[] = []
  let c = new Date(startTS)

  // Move c to Jan 1, 9:30 of the same year or the next if < startTS
  c = new Date(c.getFullYear(), 0, 1, 9, 30, 0, 0)
  if (c.getTime() < startTS) {
    // Next year
    c.setFullYear(c.getFullYear() + 1)
  }
  while (!isTradingDay(c)) {
    c.setDate(c.getDate() + 1)
  }
  c.setHours(9, 30, 0, 0)

  while (c.getTime() <= endTS) {
    ticks.push(c.getTime())
    c.setFullYear(c.getFullYear() + 1, 0, 1)
    c.setHours(9, 30, 0, 0)
    while (!isTradingDay(c)) {
      c.setDate(c.getDate() + 1)
    }
    c.setHours(9, 30, 0, 0)
  }

  return ticks
}

// ---------------------------------------------
// MAIN CHART COMPONENT
// ---------------------------------------------
export function AreaChartInteractive() {
  const [timeRange, setTimeRange] = React.useState("1d")
  const [open, setOpen] = React.useState(false)
  const [selectedStock, setSelectedStock] = React.useState("GOOGL")

  // 1) Filter
  const filteredData = chartData.filter((item) => {
    const date = new Date(item.date)
    const referenceDate = new Date()
    let rangeStartDate = new Date(referenceDate)

    if (item.stock !== selectedStock) return false;

    if (timeRange === "1d") {
      const day = referenceDate.getDay()
      if (day === 0) {
        rangeStartDate.setDate(referenceDate.getDate() - 2)
      } else if (day === 6) {
        rangeStartDate.setDate(referenceDate.getDate() - 1)
      }
      rangeStartDate.setHours(9, 30, 0, 0)
      const endDate = new Date(rangeStartDate)
      endDate.setHours(16, 0, 0, 0)
      return date >= rangeStartDate && date <= endDate
    } else if (timeRange === "5d") {
      rangeStartDate.setDate(referenceDate.getDate() - 5)
    } else if (timeRange === "1m") {
      rangeStartDate.setMonth(referenceDate.getMonth() - 1)
    } else if (timeRange === "3m") {
      rangeStartDate.setMonth(referenceDate.getMonth() - 3)
    } else if (timeRange === "1y") {
      rangeStartDate.setFullYear(referenceDate.getFullYear() - 1)
    } else if (timeRange === "allTime") {
      rangeStartDate = new Date(chartData[0].date)
    }
    rangeStartDate.setHours(9, 30, 0, 0)
    return date >= rangeStartDate
  })

  // 2) Aggregation
  let aggregatedData
  if (timeRange === "5d") {
    aggregatedData = aggregateDataHourCustom(filteredData)
  } else if (timeRange === "1m") {
    aggregatedData = aggregateDataByDayFirst(filteredData)
  } else if (timeRange === "3m" || timeRange === "1y" || timeRange === "allTime") {
    aggregatedData = aggregateDataByDayLast(filteredData)
  } else {
    aggregatedData = filteredData
  }

  // 3) Decide color
  const firstValue = aggregatedData.length ? aggregatedData[0].value : 0
  const lastValue = aggregatedData.length ? aggregatedData[aggregatedData.length - 1].value : 0
  const gradientColor = lastValue >= firstValue ? "green" : "red"

  // 4) Build compressed data
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

  // 5) Ticks
  const realMin = aggregatedData[0]?.date ?? 0
  const realMax = aggregatedData[aggregatedData.length - 1]?.date ?? 0

  function compressTicks(realTickArray: number[], reference: Date) {
    return realTickArray.map((ts) => compressDateToTradingOffset(new Date(ts), reference))
  }

  let realTicks: number[] | undefined
  if (timeRange === "1d") {
    if (timeRange === "1d") {
      const todayStart = new Date(realMin)
      todayStart.setHours(9, 30, 0, 0) // Start at 9:30 AM
      const todayEnd = new Date(realMax)
      todayEnd.setHours(16, 0, 0, 0) // End at 4:00 PM
      realTicks = generateHourlyTicks(todayStart.getTime(), todayEnd.getTime())
    }
  } else if (timeRange === "5d" || timeRange === "1m") {
    realTicks = generateDailyTicks(realMin, realMax)
  } else if (timeRange === "3m" || timeRange === "1y") {
    realTicks = generateMonthlyTicks(realMin, realMax)
  } else if (timeRange === "allTime") {
    realTicks = generateYearlyTicks(realMin, realMax)
  }

  let finalTicks: number[] | undefined
  if (realTicks && aggregatedData.length) {
    const refDate = new Date(aggregatedData[0].date)
    finalTicks = compressTicks(realTicks, refDate)
  }

  // 6) Formatters
  function finalTickFormatter(offsetValue: number) {
    if (!aggregatedData.length) return ""
    const date = expandTradingOffsetToDate(offsetValue, new Date(aggregatedData[0].date))
    if (timeRange === "1d") {
      return date.toLocaleTimeString("en-US", { hour: "numeric" })
    } else if (timeRange === "3m" || timeRange === "1y") {
      return date.toLocaleDateString("en-US", { month: "short" })
    } else if (timeRange === "allTime") {
      return date.toLocaleDateString("en-US", { year: "numeric" })
    }
    // "5d"/"1m" => "Jan 10"
    return date.toLocaleDateString("en-US", { month: "short", day: "numeric" })
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
      })
    }
    // For "3m", "1y", "allTime", just do short date
    return date.toLocaleDateString("en-US", {
      month: "short",
      day: "numeric",
      year: timeRange === "allTime" ? "numeric" : undefined,
    })
  }

  function finalFormatter(value: number, name: string, props: any) {
    const offsetValue = props?.payload?.x
    if (!aggregatedData.length || offsetValue == null) return `${value}`
    const date = expandTradingOffsetToDate(offsetValue, new Date(aggregatedData[0].date))

    if (timeRange === "1d") {
      return (
        <div className="flex">
          <div
            className="h-full w-2 mr-1.5"
            style={{ backgroundColor: gradientColor, borderRadius: "10px" }}
          ></div>
          <div>
            <div className="text-base">{`${value} USD`}</div>
            <div>
              {date.toLocaleDateString("en-US", {
                month: "short",
                day: "numeric",
                hour: "2-digit",
                minute: "2-digit",
              })}
            </div>
          </div>
        </div>
      )
    } else if (timeRange === "5d") {
      return (
        <div className="flex">
          <div
            className="h-full w-2 mr-1.5"
            style={{ backgroundColor: gradientColor, borderRadius: "10px" }}
          ></div>
          <div>
            <div className="text-base">{`${value} USD`}</div>
            <div>
              {date.toLocaleDateString("en-US", {
                month: "short",
                day: "numeric",
                hour: "2-digit",
                minute: "2-digit",
              })}
            </div>
          </div>
        </div>
      )
    }
    // 3m, 1y, allTime => show short date with year if "allTime"
    return (
      <div className="flex">
        <div
          className="h-full w-2 mr-1.5"
          style={{ backgroundColor: gradientColor, borderRadius: "10px" }}
        ></div>
        <div>
          <div className="text-base">{`${value} USD`}</div>
          <div>
            {date.toLocaleDateString("en-US", {
              month: "short",
              day: "numeric",
              year: "numeric",
            })}
          </div>
        </div>
      </div>
    )
  }

  return (
    <Card>
      <CardHeader className="flex items-center gap-2 space-y-0 border-b py-5 sm:flex-row">
        <div className="grid flex-1 gap-1 text-center sm:text-left">
          <CardTitle>Portfolio</CardTitle>
          <CardDescription>
            Fixed monthly/yearly ticks so they land on the actual first trading day
          </CardDescription>
        </div>
        <Popover open={open} onOpenChange={setOpen}>
          <PopoverTrigger asChild>
            <Button
              variant="outline"
              role="combobox"
              aria-expanded={open}
              className="w-[200px] justify-between"
            >
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
                        className={cn(
                          "ml-auto",
                          selectedStock === stock.value ? "opacity-100" : "opacity-0"
                        )}
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
            <SelectValue placeholder="Today" />
          </SelectTrigger>
          <SelectContent className="rounded-xl">
            <SelectItem value="1d">Today</SelectItem>
            <SelectItem value="5d">Last 5 days</SelectItem>
            <SelectItem value="1m">Last 1 month</SelectItem>
            <SelectItem value="3m">Last 3 months</SelectItem>
            <SelectItem value="1y">Last 1 year</SelectItem>
            <SelectItem value="allTime">All Time</SelectItem>
          </SelectContent>
        </Select>
      </CardHeader>

      <CardContent className="px-2 pt-4 sm:px-6 sm:pt-6">
        <ChartContainer config={chartConfig} className="aspect-auto h-[250px] w-full">
          <AreaChart data={compressedData}>
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
              tickFormatter={(value) => `${value} USD`}
            /><ChartTooltip
              cursor={false}
              content={
                <ChartTooltipContent
                  labelFormatter={finalLabelFormatter}
                  formatter={finalFormatter}
                  indicator="line"
                />
              }
            />
            <Area
              dataKey="value"
              type="linear"
              stroke={gradientColor}
              fill="url(#fillValue)"
            />
          </AreaChart>
        </ChartContainer>
      </CardContent>
    </Card>
  )
}
