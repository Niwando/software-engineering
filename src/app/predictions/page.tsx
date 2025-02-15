import { AppSidebar } from "@/src/components/sidebar/app-sidebar"
import {
  SidebarInset,
  SidebarProvider
} from "@/src/components/ui/sidebar"
import { StocksOverview } from "@/src/components/predictions/stocks-overview"
import { AreaChartInteractive } from "@/src/components/predictions/area-chart-interactive"
import { MainNav } from "@/src/components/predictions/main-nav"

// ----------------- RAW DATA GENERATION -----------------

function isTradingDay(date: Date) {
  const day = date.getDay()
  return day !== 0 && day !== 6
}

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
          value: Number((Math.random() * 250 + 50).toFixed(2)), // GOOGL value with 2 decimal places
          stock: "GOOGL",
        })
        data.push({
          date: c.getTime(),
          value: Number((Math.random() * 750 + 50).toFixed(2)), // MSFT value with 2 decimal places
          stock: "MSFT",
        })
        c.setMinutes(c.getMinutes() + intervalMinutes)
      }
      c.setDate(c.getDate() + 1)
    } else {
      // skip weekends
      c.setDate(c.getDate() + 1)
    }
  }
  return data
}

// Helper function to compute overview data from chartData for the last day only
function getOverviewData(
  chartData: { date: number; value: number; stock: string }[]
) {
  if (chartData.length === 0) {
    return {
      overallPerformance: 0,
      overallValue: 0,
      bestStock: { stock: "", performance: 0, value: 0 },
      worstStock: { stock: "", performance: 0, value: 0 },
      perStockPerformance: {},
    }
  }

  // Convert dates to numbers
  const timestamps = chartData.map((item) => Number(item.date))
  if (timestamps.length === 0) {
    return {
      overallPerformance: 0,
      overallValue: 0,
      bestStock: { stock: "", performance: 0, value: 0 },
      worstStock: { stock: "", performance: 0, value: 0 },
      perStockPerformance: {},
    }
  }

  // Find the maximum timestamp (last data point)
  let maxTimestamp = -Infinity
  for (let i = 0; i < timestamps.length; i++) {
    if (timestamps[i] > maxTimestamp) {
      maxTimestamp = timestamps[i]
    }
  }
  const maxDate = new Date(maxTimestamp)

  // Filter data for the last day
  const lastDayData = chartData.filter((item) => {
    const d = new Date(item.date)
    return (
      d.getFullYear() === maxDate.getFullYear() &&
      d.getMonth() === maxDate.getMonth() &&
      d.getDate() === maxDate.getDate()
    )
  })

  // Build an object that stores each stock's performance and closing value
  const stockStats: Record<string, { performance: number; closingValue: number }> = {}
  const stocks = Array.from(new Set(lastDayData.map((item) => item.stock)))
  stocks.forEach((stock) => {
    const stockData = lastDayData
      .filter((item) => item.stock === stock)
      .sort((a, b) => a.date - b.date)
    if (stockData.length > 0) {
      const opening = stockData[0].value
      const closing = stockData[stockData.length - 1].value
      // Calculate performance in percentage ((closing - opening) / opening) * 100
      const performance = ((closing - opening) / opening) * 100
      stockStats[stock] = {
        performance: Number(performance.toFixed(2)),
        closingValue: closing,
      }
    }
  })

  // Compute overall performance by averaging individual performances
  const totalPerformance = Object.values(stockStats).reduce(
    (acc, cur) => acc + cur.performance,
    0
  )
  let overallPerformance =
    Object.keys(stockStats).length > 0
      ? totalPerformance / Object.keys(stockStats).length
      : 0
  overallPerformance = Number(overallPerformance.toFixed(2))

  // Calculate overall value (sum of closing values of all stocks)
  const overallValue = Object.values(stockStats).reduce(
    (acc, cur) => acc + cur.closingValue,
    0
  ).toFixed(2)

  // Determine best and worst performers, including their closing values
  let bestStock = { stock: "", performance: -Infinity, value: 0 }
  let worstStock = { stock: "", performance: Infinity, value: 0 }
  for (const stock in stockStats) {
    const { performance, closingValue } = stockStats[stock]
    if (performance > bestStock.performance) {
      bestStock = { stock, performance, value: closingValue }
    }
    if (performance < worstStock.performance) {
      worstStock = { stock, performance, value: closingValue }
    }
  }

  // Build perStockPerformance as a simple mapping from stock to its performance
  const perStockPerformance: Record<string, number> = {}
  for (const stock in stockStats) {
    perStockPerformance[stock] = stockStats[stock].performance
  }

  return {
    overallPerformance,
    overallValue,
    bestStock,
    worstStock,
    perStockPerformance,
  }
}

const dataStartDate = new Date()
dataStartDate.setDate(dataStartDate.getDate() - 365)
const chartData = generateData(dataStartDate.toISOString(), 366, 1)
console.log("Chart Data:", chartData)

// Example usage:
const overviewData = getOverviewData(chartData)
console.log("Overview Data:", overviewData)

export default function Page() {
  // Extract unique stock names from chartData
  const uniqueStocks = Array.from(new Set(chartData.map(item => item.stock)));

  return (
    <div>
      <div className="border-b mb-4">
        <div className="flex h-16 items-center px-4">
          <MainNav className="mx-6" />
        </div>
      </div>
      <StocksOverview overviewData={overviewData}/>
      {uniqueStocks.map(stock => {
        const stockData = chartData.filter(item => item.stock === stock);
        return (
          <div key={stock} className="mb-8">
            <AreaChartInteractive chartData={stockData} stockNames={stock} />
          </div>
        );
      })}
    </div>
  )
}