// src/tests/test_get.js

import { GET } from "app/api/alphavantage/yourGetFile";
import { supabase } from "@/lib/supabase";

// Wir mocken den Supabase-Client, damit rpc-Aufrufe kontrolliert beantwortet werden.
jest.mock("@/lib/supabase", () => ({
  supabase: {
    rpc: jest.fn(),
  },
}));

describe("GET function", () => {
  beforeEach(() => {
    jest.resetAllMocks();
  });

  test("should return transformed and sorted data with status 200", async () => {
    // Mock für get_current_max_date:
    supabase.rpc.mockImplementation((procName, params) => {
      if (procName === "get_current_max_date") {
        return Promise.resolve({ data: ["2024-01-01T00:00:00Z"], error: null });
      }
      if (procName === "get_latest_day_data_chunked") {
        // Falls p_offset == 0: liefere ein Chunk, ansonsten leeres Array.
        if (params.p_offset === 0) {
          return Promise.resolve({ data: [{ timestamp: "2024-01-01T10:00:00Z", close: "100" }], error: null });
        } else {
          return Promise.resolve({ data: [], error: null });
        }
      }
      if (procName === "get_five_day_hourly") {
        return Promise.resolve({ data: [{ timestamp: "2024-01-01T09:00:00Z", close: "90" }], error: null });
      }
      if (procName === "get_older_day_end_chunked") {
        if (params.p_offset === 0) {
          return Promise.resolve({ data: [{ timestamp: "2023-12-31T15:00:00Z", close: "80" }], error: null });
        } else {
          return Promise.resolve({ data: [], error: null });
        }
      }
      return Promise.resolve({ data: [], error: null });
    });

    // Dummy NextRequest (in diesem Fall nicht verwendet, also ein leeres Objekt)
    const dummyRequest = {};

    // Rufe die GET-Funktion auf.
    const response = await GET(dummyRequest);

    // Überprüfe, ob der Status 200 ist.
    expect(response.status).toBe(200);

    // NextResponse.json() gibt ein Response-Objekt zurück, aus dem wir die JSON-Daten extrahieren können.
    const json = await response.json();

    // Erwartete Transformation:
    // - Für ältere Daten: "2023-12-31T15:00:00Z" minus 5 Stunden -> "2023-12-31T10:00:00.000Z"
    // - Für 5-Tage-Daten: "2024-01-01T09:00:00Z" minus 5 Stunden -> "2024-01-01T04:00:00.000Z"
    // - Für neueste Daten: "2024-01-01T10:00:00Z" minus 5 Stunden -> "2024-01-01T05:00:00.000Z"
    //
    // Nach Sortierung (aufsteigend nach Timestamp) sollte die Reihenfolge sein:
    // 1) älter: "2023-12-31T10:00:00.000Z", close: "80.00"
    // 2) 5-Tage: "2024-01-01T04:00:00.000Z", close: "90.00"
    // 3) neueste: "2024-01-01T05:00:00.000Z", close: "100.00"

    const expected = [
      { timestamp: "2023-12-31T10:00:00.000Z", close: "80.00" },
      { timestamp: "2024-01-01T04:00:00.000Z", close: "90.00" },
      { timestamp: "2024-01-01T05:00:00.000Z", close: "100.00" },
    ];

    // Es wird erwartet, dass genau drei Datensätze zurückgegeben werden.
    expect(json.length).toBe(expected.length);

    // Vergleiche die relevanten Eigenschaften der transformierten Objekte.
    for (let i = 0; i < expected.length; i++) {
      expect(json[i].timestamp).toBe(expected[i].timestamp);
      expect(json[i].close).toBe(expected[i].close);
    }
  });
});
