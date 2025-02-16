import { NextRequest, NextResponse } from "next/server";
import { supabase } from "@/lib/supabase";

export async function GET(req: NextRequest) {
  try {
    const pageSize = 1000;
    let allData = [];
    let from = 0;
    let to = pageSize - 1;
    let keepFetching = true;

    while (keepFetching) {
      const { data, error } = await supabase
        .from("stock_data")
        .select("timestamp, symbol, close")
        .range(from, to);

      if (error) {
        console.error("Error fetching data:", error);
        return NextResponse.json(
          { error: "An error occurred while fetching data" },
          { status: 500 }
        );
      }
      const filteredData = data.map(item => {
        // Ziehe 5 Stunden vom Timestamp ab
        const adjustedDate = new Date(item.timestamp);
        adjustedDate.setHours(adjustedDate.getHours() - 5);
      
        return {
          ...item,
          // Aktualisierter Timestamp (als ISO-String)
          timestamp: adjustedDate.toISOString(),
          // Rundung von close auf zwei Nachkommastellen
          close: Number(item.close).toFixed(2)
        };
      });

      // Füge die gefilterten und angepassten Daten zu allData hinzu
      allData = allData.concat(filteredData);
      console.log(`Fetched ${allData.length} records`);

      // Falls weniger als pageSize Datensätze zurückgegeben wurden, sind keine weiteren vorhanden.
      if (data.length < pageSize) {
        keepFetching = false;
      } else {
        // Nächste Seite abrufen
        from += pageSize;
        to += pageSize;
      }
    }

    return NextResponse.json(allData, { status: 200 });
  } catch (error) {
    console.error("Unexpected Error:", error);
    return NextResponse.json(
      { error: "An unexpected error occurred" },
      { status: 500 }
    );
  }
}
