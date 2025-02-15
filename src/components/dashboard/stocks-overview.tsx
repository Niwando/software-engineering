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
  // Determine gradient colors based on each performance value.
  const overallCardGradientRGB = overviewData.overallPerformance >= 0 ? "0,128,0" : "255,0,0";
  const bestCardGradientRGB = overviewData.bestStock.performance >= 0 ? "0,128,0" : "255,0,0";
  const worstCardGradientRGB = overviewData.worstStock.performance >= 0 ? "0,128,0" : "255,0,0";

  const overallOpacity = overviewData.overallPerformance >= 0 ? 0.45 : 0.35;
  const bestOpacity = overviewData.bestStock.performance >= 0 ? 0.45 : 0.35;
  const worstOpacity = overviewData.worstStock.performance >= 0 ? 0.45 : 0.35;

  const overallCardStyle = {
    background: `linear-gradient(to bottom, rgba(${overallCardGradientRGB}, ${overallOpacity}) 0%, rgba(${overallCardGradientRGB}, 0) 70%)`
  }
  const bestCardStyle = {
    background: `linear-gradient(to bottom, rgba(${bestCardGradientRGB}, ${bestOpacity}) 0%, rgba(${bestCardGradientRGB}, 0) 70%)`
  }
  const worstCardStyle = {
    background: `linear-gradient(to bottom, rgba(${worstCardGradientRGB}, ${worstOpacity}) 0%, rgba(${worstCardGradientRGB}, 0) 70%)`
  }

  // Helper to render a filled arrow icon based on performance.
  const renderArrowIcon = (performance: number) => {
    return performance >= 0 ? (
      <ArrowBigUp size={30} fill="currentColor" className="inline-block fill-current text-green-500" />
    ) : (
      <ArrowBigDown size={30} fill="currentColor" className="inline-block fill-current text-red-500" />
    )
  }

  return (
    <div className="grid grid-cols-1 min-[880px]:grid-cols-3 gap-4 p-8">
        <Card className="w-full" style={overallCardStyle}>
        <CardContent className="p-4 h-full">
          <div className="flex items-center w-full h-full">
            <div className="flex flex-col flex-1">
              <div className="text-2xl font-bold">Market</div>
              <div className="text-md font-bold">{overviewData.overallValue} USD</div>
            </div>
            <div className="flex items-center text-2xl font-bold">
              {overviewData.overallPerformance}%
              <span className="ml-2 pb-0.5">
                {renderArrowIcon(overviewData.overallPerformance)}
              </span>
            </div>
          </div>
        </CardContent>
      </Card>
      <Card className="w-full" style={bestCardStyle}>
        <CardContent className="p-4 h-full">
          <div className="flex items-center w-full h-full">
            <div className="flex flex-col flex-1">
              <div className="text-2xl font-bold">{overviewData.bestStock.stock}</div>
              <div className="text-md font-bold">{overviewData.bestStock.value} USD</div>
            </div>
            <div className="flex items-center text-2xl font-bold">
              {overviewData.bestStock.performance}%
              <span className="ml-2 pb-0.5">
                {renderArrowIcon(overviewData.bestStock.performance)}
              </span>
            </div>
          </div>
        </CardContent>
      </Card>
      <Card className="w-full" style={worstCardStyle}>
        <CardContent className="p-4 h-full">
          <div className="flex items-center w-full h-full">
            <div className="flex flex-col flex-1">
              <div className="text-2xl font-bold">{overviewData.worstStock.stock}</div>
              <div className="text-md font-bold">{overviewData.worstStock.value} USD</div>
            </div>
            <div className="flex items-center text-2xl font-bold">
              {overviewData.worstStock.performance}%
              <span className="ml-2 pb-0.5">
                {renderArrowIcon(overviewData.worstStock.performance)}
              </span>
            </div>
          </div>
        </CardContent>
      </Card>
    </div>
  )
}