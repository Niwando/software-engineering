import { NextRequest, NextResponse } from "next/server";
import { supabase } from "@/lib/supabase";

export async function GET(req: NextRequest) {
  try {
    // 1) Aktuellstes Datum holen
    const { data: maxDateArray, error: maxDateError } = await supabase.rpc("get_current_max_date");
    if (maxDateError) {
      console.error("Error fetching max_date:", maxDateError);
      return NextResponse.json(
        { error: "An error occurred while fetching the max date" },
        { status: 500 }
      );
    }
    if (!maxDateArray?.length || !maxDateArray) {
      console.error("No max date returned from get_current_max_date()");
      return NextResponse.json({ error: "No data returned for max date" }, { status: 404 });
    }
    const p_max_date = maxDateArray;

    // 2) Chunking für den neuesten Tag
    let latestDayData: any[] = [];
    let keepFetchingLatest = true;
    let latestOffset = 0;
    let pageSize = 1000;

    while (keepFetchingLatest) {
      const { data: latestChunk, error } = await supabase.rpc("get_latest_day_data_chunked", {
        p_max_date,
        p_offset: latestOffset,
        p_limit: pageSize,
      });

      if (error) {
        console.error("Error fetching chunk of latest day data:", error);
        // Abbrechen oder weiter nach Bedarf
        break;
      }

      if (!latestChunk?.length) {
        keepFetchingLatest = false;
      } else {
        latestDayData = latestDayData.concat(latestChunk);
        latestOffset += pageSize;
      }
    }

    // 3) Beispiel: Stündliche Daten (5 Tage) – falls ebenfalls sehr groß, analoges Chunking
    const { data: fiveDayData, error: fiveDayError } = await supabase.rpc("get_five_day_hourly", {
        p_max_date,
      });
      if (fiveDayError) {
        console.error("Error fetching five-day data:", fiveDayError);
        return NextResponse.json(
          { error: "An error occurred while fetching five day data" },
          { status: 500 }
        );
      }

    // 4) Beispiel: Ältere Daten (Tagesende) – hast du schon chunking:
    let olderDayData: any[] = [];
    let keepFetchingOlder = true;
    let olderOffset = 0;
    pageSize = 500;

    while (keepFetchingOlder) {
      const { data: chunk, error } = await supabase.rpc("get_older_day_end_chunked", {
        p_max_date,
        p_offset: olderOffset,
        p_limit: pageSize,
      });

      if (error) {
        console.error("Error fetching chunk of older data:", error);
        break;
      }

      if (!chunk?.length) {
        keepFetchingOlder = false;
      } else {
        olderDayData = olderDayData.concat(chunk);
        olderOffset += pageSize;
      }
    }

    // 5) Zusammenfügen
    let combinedData = [...latestDayData, ...fiveDayData, ...olderDayData /* + ggf. fiveDayData usw. */];

    // 6) Sortieren und transformieren
    combinedData.sort((a, b) => new Date(a.timestamp).getTime() - new Date(b.timestamp).getTime());

    const transformedData = combinedData.map((item) => {
      const adjustedDate = new Date(item.timestamp);
      adjustedDate.setHours(adjustedDate.getHours() - 5);
      return {
        ...item,
        timestamp: adjustedDate.toISOString(),
        close: Number(item.close).toFixed(2),
      };
    });

    return NextResponse.json(transformedData, { status: 200 });
  } catch (error) {
    console.error("Unexpected Error:", error);
    return NextResponse.json({ error: "An unexpected error occurred" }, { status: 500 });
  }
}
