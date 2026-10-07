from typing import Optional
from langchain_core.tools import tool


# ============================================================
# MOCK DATABASE
# ============================================================

FLIGHTS = [
    {
        "flight_id": "VN123",
        "airline": "Vietnam Airlines",
        "from": "SGN",
        "to": "HAN",
        "date": "2026-10-20",
        "departure": "08:00",
        "arrival": "10:10",
        "price": 1500000,
        "seats": 5,
    },
    {
        "flight_id": "VN456",
        "airline": "Vietnam Airlines",
        "from": "SGN",
        "to": "HAN",
        "date": "2026-10-20",
        "departure": "14:00",
        "arrival": "16:10",
        "price": 1800000,
        "seats": 2,
    },
    {
        "flight_id": "VJ789",
        "airline": "VietJet Air",
        "from": "SGN",
        "to": "HAN",
        "date": "2026-10-20",
        "departure": "19:30",
        "arrival": "21:40",
        "price": 1200000,
        "seats": 8,
    },
    {
        "flight_id": "VN222",
        "airline": "Vietnam Airlines",
        "from": "SGN",
        "to": "DAD",
        "date": "2026-10-21",
        "departure": "09:00",
        "arrival": "10:20",
        "price": 1100000,
        "seats": 4,
    },
    {
        "flight_id": "VJ333",
        "airline": "VietJet Air",
        "from": "SGN",
        "to": "DAD",
        "date": "2026-10-21",
        "departure": "15:30",
        "arrival": "16:50",
        "price": 950000,
        "seats": 6,
    },
]


BOOKINGS = []


# ============================================================
# TOOL 1 — SEARCH FLIGHTS
# ============================================================

@tool
def search_flights(
    origin: str,
    destination: str,
    date: str,
    passengers: int = 1,
) -> str:
    """
    Search available flights.

    Args:
        origin: Departure airport code, e.g. SGN.
        destination: Arrival airport code, e.g. HAN.
        date: Flight date in YYYY-MM-DD format.
        passengers: Number of passengers.
    """

    origin = origin.upper()
    destination = destination.upper()

    results = []

    for flight in FLIGHTS:
        if (
            flight["from"] == origin
            and flight["to"] == destination
            and flight["date"] == date
            and flight["seats"] >= passengers
        ):
            results.append(flight)

    if not results:
        return (
            f"No flights found from {origin} to {destination} "
            f"on {date} for {passengers} passenger(s)."
        )

    output = [f"Found {len(results)} flight(s):"]

    for flight in results:
        output.append(
            f"- {flight['flight_id']} | "
            f"{flight['airline']} | "
            f"{flight['departure']}-{flight['arrival']} | "
            f"{flight['price']:,} VND | "
            f"{flight['seats']} seats available"
        )

    return "\n".join(output)


# ============================================================
# TOOL 2 — CHECK FLIGHT
# ============================================================

@tool
def check_flight(flight_id: str) -> str:
    """
    Check detailed information and availability of a flight.

    Args:
        flight_id: Flight identifier such as VN123.
    """

    flight_id = flight_id.upper()

    for flight in FLIGHTS:
        if flight["flight_id"] == flight_id:

            if flight["seats"] <= 0:
                return f"Flight {flight_id} is sold out."

            return (
                f"Flight {flight_id} is available.\n"
                f"Airline: {flight['airline']}\n"
                f"Route: {flight['from']} -> {flight['to']}\n"
                f"Date: {flight['date']}\n"
                f"Departure: {flight['departure']}\n"
                f"Arrival: {flight['arrival']}\n"
                f"Price: {flight['price']:,} VND\n"
                f"Available seats: {flight['seats']}"
            )

    return f"Flight {flight_id} does not exist."


# ============================================================
# TOOL 3 — BOOK FLIGHT
# ============================================================

@tool
def book_flight(
    flight_id: str,
    passenger_name: str,
    passengers: int = 1,
    user_confirmed: bool = False,
) -> str:
    """
    Book a flight.

    Booking requires explicit user confirmation.

    Args:
        flight_id: Flight identifier.
        passenger_name: Passenger name.
        passengers: Number of passengers.
        user_confirmed: Must be True to complete booking.
    """

    flight_id = flight_id.upper()

    # Permission check at tool level
    if not user_confirmed:
        return (
            "BOOKING_BLOCKED: Explicit user confirmation "
            "is required before booking."
        )

    for flight in FLIGHTS:

        if flight["flight_id"] != flight_id:
            continue

        if flight["seats"] < passengers:
            return (
                f"BOOKING_FAILED: Only {flight['seats']} seat(s) "
                f"available on {flight_id}."
            )

        booking_id = f"BK{len(BOOKINGS) + 1:04d}"

        flight["seats"] -= passengers

        booking = {
            "booking_id": booking_id,
            "flight_id": flight_id,
            "passenger_name": passenger_name,
            "passengers": passengers,
        }

        BOOKINGS.append(booking)

        return (
            f"BOOKING_SUCCESS\n"
            f"Booking ID: {booking_id}\n"
            f"Flight: {flight_id}\n"
            f"Passenger: {passenger_name}\n"
            f"Passengers: {passengers}"
        )

    return f"BOOKING_FAILED: Flight {flight_id} does not exist."


# ============================================================
# TOOL 4 — CANCEL BOOKING
# ============================================================

@tool
def cancel_booking(booking_id: str) -> str:
    """
    Cancel an existing booking.

    Args:
        booking_id: Booking identifier such as BK0001.
    """

    booking_id = booking_id.upper()

    for booking in BOOKINGS:

        if booking["booking_id"] == booking_id:

            BOOKINGS.remove(booking)

            # Return seats to the flight
            for flight in FLIGHTS:
                if flight["flight_id"] == booking["flight_id"]:
                    flight["seats"] += booking["passengers"]
                    break

            return (
                f"CANCELLATION_SUCCESS\n"
                f"Booking {booking_id} has been cancelled."
            )

    return f"CANCELLATION_FAILED: Booking {booking_id} does not exist."


# ============================================================
# ALL TOOLS
# ============================================================

AIRLINE_TOOLS = [
    search_flights,
    check_flight,
    book_flight,
    cancel_booking,
]