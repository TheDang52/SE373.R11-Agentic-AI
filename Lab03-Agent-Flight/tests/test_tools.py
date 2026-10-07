from tools.airline_tools import (
    search_flights,
    check_flight,
    book_flight,
    cancel_booking,
)


def main():
    print("=" * 60)
    print("TEST 1 — SEARCH FLIGHTS")
    print("=" * 60)

    result = search_flights.invoke(
        {
            "origin": "SGN",
            "destination": "HAN",
            "date": "2026-10-20",
            "passengers": 1,
        }
    )

    print(result)

    print("\n" + "=" * 60)
    print("TEST 2 — CHECK FLIGHT")
    print("=" * 60)

    result = check_flight.invoke(
        {
            "flight_id": "VN123",
        }
    )

    print(result)

    print("\n" + "=" * 60)
    print("TEST 3 — BOOK WITHOUT CONFIRMATION")
    print("=" * 60)

    result = book_flight.invoke(
        {
            "flight_id": "VN123",
            "passenger_name": "Test User",
            "passengers": 1,
            "user_confirmed": False,
        }
    )

    print(result)

    print("\n" + "=" * 60)
    print("TEST 4 — BOOK WITH CONFIRMATION")
    print("=" * 60)

    result = book_flight.invoke(
        {
            "flight_id": "VN123",
            "passenger_name": "Test User",
            "passengers": 1,
            "user_confirmed": True,
        }
    )

    print(result)


if __name__ == "__main__":
    main()