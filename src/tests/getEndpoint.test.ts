// __tests__/getEndpoint.test.ts

import { GET } from "../app/api/database/data/route";
import { supabase } from "@/lib/supabase";
import { NextRequest } from "next/server";

// Mock the supabase module and its rpc method
jest.mock("@/lib/supabase", () => ({
  supabase: {
    rpc: jest.fn(),
  },
}));

describe("GET API Endpoint", () => {
  // Clear mocks after each test
  afterEach(() => {
    jest.clearAllMocks();
  });

  test("should return combined and transformed data sorted by timestamp", async () => {
    // Setup our mock implementation for a successful data retrieval
    (supabase.rpc as jest.Mock).mockImplementation((rpcName, params) => {
      switch (rpcName) {
        case "get_current_max_date":
          return Promise.resolve({ data: ["2023-01-02T00:00:00Z"], error: null });
        case "get_latest_day_data_chunked":
          // Simulate one chunk with data then an empty chunk to break the loop
          if (params.p_offset === 0) {
            return Promise.resolve({
              data: [{ timestamp: "2023-01-02T05:00:00.000Z", close: 10 }],
              error: null,
            });
          }
          return Promise.resolve({ data: [], error: null });
        case "get_five_day_hourly":
          return Promise.resolve({
            data: [{ timestamp: "2023-01-01T05:00:00.000Z", close: 20 }],
            error: null,
          });
        case "get_older_day_end_chunked":
          if (params.p_offset === 0) {
            return Promise.resolve({
              data: [{ timestamp: "2023-01-03T05:00:00.000Z", close: 30 }],
              error: null,
            });
          }
          return Promise.resolve({ data: [], error: null });
        default:
          return Promise.resolve({ data: [], error: null });
      }
    });

    const req = new NextRequest("http://localhost");
    const response = await GET(req);
    expect(response.status).toBe(200);

    const json = await response.json();
    // Expected transformation:
    // - Subtract 5 hours from the provided timestamps
    // - The five-day data (timestamp "2023-01-01T05:00:00.000Z") becomes "2023-01-01T00:00:00.000Z"
    // - The latest day data becomes "2023-01-02T00:00:00.000Z"
    // - The older day data becomes "2023-01-03T00:00:00.000Z"
    // Finally, the combined array is sorted in ascending order.
    expect(json).toEqual([
      { timestamp: "2023-01-01T00:00:00.000Z", close: "20.00" },
      { timestamp: "2023-01-02T00:00:00.000Z", close: "10.00" },
      { timestamp: "2023-01-03T00:00:00.000Z", close: "30.00" },
    ]);
  });

  test("should return error if get_current_max_date fails", async () => {
    (supabase.rpc as jest.Mock).mockImplementation((rpcName) => {
      if (rpcName === "get_current_max_date") {
        return Promise.resolve({ data: null, error: new Error("Test error") });
      }
      return Promise.resolve({ data: [], error: null });
    });

    const req = new NextRequest("http://localhost");
    const response = await GET(req);
    expect(response.status).toBe(500);
    const json = await response.json();
    expect(json.error).toBe("An error occurred while fetching the max date");
  });

  test("should return 404 if get_current_max_date returns empty array", async () => {
    (supabase.rpc as jest.Mock).mockImplementation((rpcName) => {
      if (rpcName === "get_current_max_date") {
        return Promise.resolve({ data: [], error: null });
      }
      return Promise.resolve({ data: [], error: null });
    });

    const req = new NextRequest("http://localhost");
    const response = await GET(req);
    expect(response.status).toBe(404);
    const json = await response.json();
    expect(json.error).toBe("No data returned for max date");
  });

  test("should return error if get_five_day_hourly fails", async () => {
    (supabase.rpc as jest.Mock).mockImplementation((rpcName) => {
      if (rpcName === "get_current_max_date") {
        return Promise.resolve({ data: ["2023-01-02T00:00:00Z"], error: null });
      }
      if (rpcName === "get_latest_day_data_chunked") {
        return Promise.resolve({ data: [], error: null });
      }
      if (rpcName === "get_five_day_hourly") {
        return Promise.resolve({ data: null, error: new Error("Five day error") });
      }
      if (rpcName === "get_older_day_end_chunked") {
        return Promise.resolve({ data: [], error: null });
      }
      return Promise.resolve({ data: [], error: null });
    });

    const req = new NextRequest("http://localhost");
    const response = await GET(req);
    expect(response.status).toBe(500);
    const json = await response.json();
    expect(json.error).toBe("An error occurred while fetching five day data");
  });

  test("should continue processing if an error occurs during chunk fetching", async () => {
    (supabase.rpc as jest.Mock).mockImplementation((rpcName, params) => {
      switch (rpcName) {
        case "get_current_max_date":
          return Promise.resolve({ data: ["2023-01-02T00:00:00Z"], error: null });
        case "get_latest_day_data_chunked":
          // Simulate an error during the chunk fetch
          return Promise.resolve({ data: null, error: new Error("Chunk error") });
        case "get_five_day_hourly":
          return Promise.resolve({
            data: [{ timestamp: "2023-01-01T05:00:00.000Z", close: 20 }],
            error: null,
          });
        case "get_older_day_end_chunked":
          return Promise.resolve({ data: [], error: null });
        default:
          return Promise.resolve({ data: [], error: null });
      }
    });

    const req = new NextRequest("http://localhost");
    const response = await GET(req);
    expect(response.status).toBe(200);
    const json = await response.json();

    // In this scenario the latest day data is skipped due to the error,
    // so only the five-day data is returned after transformation.
    expect(json).toEqual([
      { timestamp: "2023-01-01T00:00:00.000Z", close: "20.00" },
    ]);
  });

  test("should return 500 if an unexpected error occurs", async () => {
    // Force an unexpected error by having the rpc method throw
    (supabase.rpc as jest.Mock).mockImplementation(() => {
      throw new Error("Unexpected");
    });

    const req = new NextRequest("http://localhost");
    const response = await GET(req);
    expect(response.status).toBe(500);
    const json = await response.json();
    expect(json.error).toBe("An unexpected error occurred");
  });
});
