"use client"
import { StocksOverview } from "@/components/predictions/stocks-overview"
import { AreaChartInteractive } from "@/components/predictions/area-chart-interactive"
import { MainNav } from "@/components/predictions/main-nav"
import { useEffect, useState } from "react";
import { LoadingSpinner } from "@/components/loading-spinner"


export default function Page() {
  const [data, setData] = useState([]);
    const [loading, setLoading] = useState(true);
    const [error, setError] = useState(null);

    function getOverviewData(
        chartData: { date: number; value: number; stock: string }[]
      ) {
        console.log("chartData", chartData[0])
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
        const timestamps = chartData.map((item) => new Date(item.date).getTime())
        console.log("timestamps", timestamps)
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
        console.log("maxDate", maxDate)
      
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
        
        const sum = Object.values(stockStats).reduce((acc, cur) => {
          return acc + parseFloat(cur.closingValue);
        }, 0);
        
        const overallValue = sum.toFixed(2); 
        console.log("overallValue", overallValue)
      
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

    // Funktion, die den Endpunkt abruft und das Mapping vornimmt
    const fetchStockData = async () => {
        try {
          // Passe den URL-Pfad an deinen Endpunkt an
          const response = await fetch("/api/database/prediction");
          if (!response.ok) {
            throw new Error("Network response was not ok");
          }
          const result = await response.json();
    
          // Transformation: Ersetze "close" durch "value"
          const transformedData = result.map((item) => ({
            value: item.predicted_close,
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
    useEffect(() => {
        fetchStockData();
      }, []);
    
      console.log("data", data.length)
    
      const overviewData = getOverviewData(data)
      const uniqueStocks = Array.from(new Set(data.map(item => item.stock)));
    
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
      {uniqueStocks.map(stock => {
        const stockData = data.filter(item => item.stock === stock);
        return (
          <div key={stock} className="mb-8">
            <AreaChartInteractive chartData={stockData} stockNames={stock} />
          </div>
        );
      })}
    </div>
  )
}