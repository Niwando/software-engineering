"use client";
import { StocksOverview } from "@/components/dashboard/stocks-overview"
import { AreaChartInteractive } from "@/components/dashboard/area-chart-interactive"
import { MainNav } from "@/components/dashboard/main-nav"
import { useEffect, useState } from "react";
import { LoadingSpinner } from "@/components/loading-spinner"

// ----------------- RAW DATA GENERATION -----------------

function isTradingDay(date: Date) {
  const day = date.getDay()
  return day !== 0 && day !== 6
}

function generateData(startDate: string, numDays: number, intervalMinutes: number) {
  const data: { date: number; value: number; stock: string }[] = []
  const c = new Date(startDate)
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

export default function Page() {

  const [data, setData] = useState([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);

  // Funktion, die den Endpunkt abruft und das Mapping vornimmt
  const fetchStockData = async () => {
    try {
      // Passe den URL-Pfad an deinen Endpunkt an
      const response = await fetch("/api/database/data");
      if (!response.ok) {
        throw new Error("Network response was not ok");
      }
      const result = await response.json();

      // Transformation: Ersetze "close" durch "value"
      const transformedData = result.map((item) => ({
        value: item.close,
        stock: item.symbol,
        date: item.timestamp,
      })).sort((a, b) => new Date(a.date) - new Date(b.date));

      setData(transformedData);
    } catch (err) {
      console.error("Fehler beim Abrufen der Daten:", err);
      setError(err);
    } finally {
      setLoading(false);
    }
  };

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
      if (timestamps[i]! > maxTimestamp) {
        maxTimestamp = timestamps[i]!
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
        const opening = stockData[0]!.value
        const closing = stockData[stockData.length - 1]!.value
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
    const overallValue = Number(
      Object.values(stockStats).reduce((acc, cur) => acc + cur.closingValue, 0).toFixed(2)
    )
  
    // Determine best and worst performers, including their closing values
    let bestStock = { stock: "", performance: -Infinity, value: 0 }
    let worstStock = { stock: "", performance: Infinity, value: 0 }
    for (const stock in stockStats) {
      const stats = stockStats[stock]
      if (!stats) continue; // Überspringe den Fall, falls stats undefined ist
      const { performance, closingValue } = stats;
      
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
      perStockPerformance[stock] = stockStats[stock]!.performance
    }
  
    return {
      overallPerformance,
      overallValue,
      bestStock,
      worstStock,
      perStockPerformance,
    }
  }

  const overviewData = getOverviewData(data)

  useEffect(() => {
    fetchStockData();
  }, []);

  if (loading) {
    return <LoadingSpinner />
  }

  return (
    <div>
      <div className="border-b mb-4">
        <div className="flex h-16 items-center px-4">
          <MainNav className="mx-6" />
        </div>
      </div>
        <StocksOverview overviewData={overviewData}/>
        <AreaChartInteractive chartData={data}/>
    </div>
  )
}