import { NextRequest } from "next/server";
import { GET } from "@/app/api/database/prediction/route";
import { supabase } from "@/lib/supabase";

// Mock the supabase module and its rpc method
jest.mock("@/lib/supabase", () => ({
  supabase: {
    rpc: jest.fn(),
  },
}));

describe("GET Predicted API Endpoint", () => {
  afterEach(() => {
    jest.clearAllMocks();
  });

  test("should return transformed predicted data sorted by timestamp", async () => {
    // Simulate a successful call to get_current_max_date_predicted
    // and a successful chunk fetch returning two records.
    (supabase.rpc as jest.Mock).mockImplementation((rpcName, params) => {
      if (rpcName === "get_current_max_date_predicted") {
        return Promise.resolve({ data: ["2023-01-01T00:00:00Z"], error: null });
      }
      if (rpcName === "get_latest_day_data_chunked_predicted") {
        // For the first call, return two records out of order.
        if (params.p_offset === 0) {
          return Promise.resolve({
            data: [
              { timestamp: "2023-01-02T12:00:00Z", predicted_close: 20 },
              { timestamp: "2023-01-02T08:00:00Z", predicted_close: 10 },
            ],
            error: null,
          });
        }
        // For subsequent calls, return an empty array to end the loop.
        return Promise.resolve({ data: [], error: null });
      }
      return Promise.resolve({ data: [], error: null });
    });

    const req = new NextRequest("http://localhost");
    const response = await GET(req);
    expect(response.status).toBe(200);
    const json = await response.json();

    // After sorting, the record with the earlier timestamp should come first.
    expect(json).toEqual([
      { timestamp: "2023-01-02T08:00:00Z", predicted_close: "10.00" },
      { timestamp: "2023-01-02T12:00:00Z", predicted_close: "20.00" },
    ]);
  });

  test("should return 500 if get_current_max_date_predicted fails", async () => {
    (supabase.rpc as jest.Mock).mockImplementation((rpcName) => {
      if (rpcName === "get_current_max_date_predicted") {
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

  test("should return 404 if get_current_max_date_predicted returns an empty array", async () => {
    (supabase.rpc as jest.Mock).mockImplementation((rpcName) => {
      if (rpcName === "get_current_max_date_predicted") {
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

  test("should break out of chunk loop if chunk fetching returns error", async () => {
    (supabase.rpc as jest.Mock).mockImplementation((rpcName, params) => {
      if (rpcName === "get_current_max_date_predicted") {
        return Promise.resolve({ data: ["2023-01-01T00:00:00Z"], error: null });
      }
      if (rpcName === "get_latest_day_data_chunked_predicted") {
        // Simulate an error on the chunk call.
        return Promise.resolve({ data: null, error: new Error("Chunk error") });
      }
      return Promise.resolve({ data: [], error: null });
    });

    const req = new NextRequest("http://localhost");
    const response = await GET(req);
    expect(response.status).toBe(200);
    const json = await response.json();
    // With a chunk error, no data is added so an empty array is returned.
    expect(json).toEqual([]);
  });

  test("should return 500 if an unexpected error occurs", async () => {
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
