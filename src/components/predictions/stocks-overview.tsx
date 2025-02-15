"use client"

import {
  Card,
  CardContent,
  CardHeader,
  CardTitle,
} from "@/src/components/ui/card"
import { ArrowBigUp, ArrowBigDown } from "lucide-react"

export interface OverviewData {
  overallPerformance: number;
  overallValue: number;
  bestStock: { stock: string; performance: number; value: number };
  worstStock: { stock: string; performance: number; value: number };
  perStockPerformance: Record<string, number>;
}

export function StocksOverview({ overviewData }: { overviewData: OverviewData }) {

  const cardStyle = {
    background: `linear-gradient(to bottom, rgba(66, 164, 245, 0.45) 0%, rgba(0,0,255, 0) 70%)`
  }

  // Helper to render a filled arrow icon based on performance.
  const renderArrowIcon = (performance: number) => {
    return performance >= 0 ? (
      <ArrowBigUp size={20} fill="currentColor" className="inline-block fill-current text-blue-500" />
    ) : (
      <ArrowBigDown size={20} fill="currentColor" className="inline-block fill-current text-blue-500" />
    )
  }

  return (
    <div className="grid grid-cols-3 gap-4 p-8">
      <Card className="w-full" style={cardStyle}>
        <CardContent>
          <div className="h-full w-full flex justify-between items-center">
            <div className="flex flex-col">
              <div className="text-2xl font-bold">Market</div>
              <div className="text-2xl font-bold">{overviewData.overallValue} USD</div>
            </div>
            <div className="flex items-center text-2xl font-bold">
              {overviewData.overallPerformance}%
              <span className="ml-2">
                {renderArrowIcon(overviewData.overallPerformance)}
              </span>
            </div>
          </div>
        </CardContent>
      </Card>
      <Card className="w-full" style={cardStyle}>
        <CardContent>
          <div className="h-full w-full flex justify-between items-center">
            <div className="flex flex-col">
              <div className="text-2xl font-bold">{overviewData.bestStock.stock}</div>
              <div className="text-2xl font-bold">{overviewData.bestStock.value} USD</div>
            </div>
            <div className="flex items-center text-2xl font-bold">
              {overviewData.bestStock.performance}%
              <span className="ml-2">
                {renderArrowIcon(overviewData.bestStock.performance)}
              </span>
            </div>
          </div>
        </CardContent>
      </Card>
      <Card className="w-full" style={cardStyle}>
        <CardContent>
          <div className="h-full w-full flex justify-between items-center">
            <div className="flex flex-col">
              <div className="text-2xl font-bold">{overviewData.worstStock.stock}</div>
              <div className="text-2xl font-bold">{overviewData.worstStock.value} USD</div>
            </div>
            <div className="flex items-center text-2xl font-bold">
              {overviewData.worstStock.performance}%
              <span className="ml-2">
                {renderArrowIcon(overviewData.worstStock.performance)}
              </span>
            </div>
          </div>
        </CardContent>
      </Card>
    </div>
  )
}