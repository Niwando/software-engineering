import { NextRequest, NextResponse } from "next/server";
import { supabase } from "@/lib/supabase";

export async function GET(req: NextRequest) {
  try {
    // 1) Aktuellstes Datum holen
    const { data: maxDateArray, error: maxDateError } = await supabase.rpc("get_current_max_date_predicted");
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
      const { data: latestChunk, error } = await supabase.rpc("get_latest_day_data_chunked_predicted", {
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

    // 6) Sortieren und transformieren
    latestDayData.sort((a, b) => new Date(a.timestamp).getTime() - new Date(b.timestamp).getTime());

    const transformedData = latestDayData.map((item) => {
      return {
        ...item,
        predicted_close: Number(item.predicted_close).toFixed(2),
      };
    });

    return NextResponse.json(transformedData, { status: 200 });
  } catch (error) {
    console.error("Unexpected Error:", error);
    return NextResponse.json({ error: "An unexpected error occurred" }, { status: 500 });
  }
}
