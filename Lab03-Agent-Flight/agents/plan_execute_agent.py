import os
from dotenv import load_dotenv
from pydantic import BaseModel, Field
from langchain_google_genai import ChatGoogleGenerativeAI

from tools.airline_tools import (
    search_flights,
    check_flight,
    book_flight,
)

from harness.data_constraints import (
    FlightRequest,
    DataConstraintHarness,
)

from harness.completion_check import CompletionHarness
from harness.permission_check import PermissionHarness
from harness.handoff import HandoffHarness


load_dotenv()

if not os.getenv("GEMINI_API_KEY"):
    raise RuntimeError("GEMINI_API_KEY is not configured.")


# ============================================================
# 1. PLANNER
# ============================================================

class FlightPlan(BaseModel):
    origin: str = Field(description="Origin airport code")
    destination: str = Field(description="Destination airport code")
    date: str = Field(description="Flight date in YYYY-MM-DD format")
    passengers: int = Field(default=1)
    flight_id: str | None = Field(
        default=None,
        description="Flight ID if the user wants to book a specific flight"
    )
    passenger_name: str | None = Field(
        default=None,
        description="Passenger name if provided"
    )
    user_confirmed: bool = Field(
        default=False,
        description="Whether the user explicitly confirmed the booking"
    )
    steps: list[str] = Field(
        description="Ordered execution steps"
    )


model = ChatGoogleGenerativeAI(
    model="gemini-3.5-flash-lite",
)

planner = model.with_structured_output(FlightPlan)


# ============================================================
# 2. HARNESS
# ============================================================

data_harness = DataConstraintHarness()
completion_harness = CompletionHarness()
permission_harness = PermissionHarness()
handoff_harness = HandoffHarness()


# ============================================================
# 3. CREATE PLAN
# ============================================================

def create_plan(user_message: str) -> FlightPlan:

    prompt = f"""
You are a flight booking planner.

Convert the user's request into a structured execution plan.

User request:
{user_message}

Rules:

1. Extract origin airport code if available.
2. Extract destination airport code if available.
3. Extract date in YYYY-MM-DD format.
4. Extract number of passengers.
5. If passengers are not specified, use 1.
6. Extract flight ID if the user specifies one.
7. Extract passenger name if provided.
8. Detect whether the user explicitly confirmed booking.
9. Create an ordered plan.

For a search request, the plan should contain:
- validate request
- search flights
- check completion
- return results

For a booking request, the plan should contain:
- validate request
- check flight
- check booking permission
- book flight only if permission is granted
- check completion
- return result

Never assume user confirmation.
"""

    return planner.invoke(prompt)


# ============================================================
# 4. EXECUTOR
# ============================================================

def execute_plan(plan: FlightPlan):

    print("\n" + "=" * 60)
    print("PLAN")
    print("=" * 60)

    for i, step in enumerate(plan.steps, start=1):
        print(f"{i}. {step}")

    # --------------------------------------------------------
    # STEP 1 — DATA VALIDATION
    # --------------------------------------------------------

    request = FlightRequest(
        origin=plan.origin,
        destination=plan.destination,
        date=plan.date,
        passengers=plan.passengers,
    )

    valid, errors = data_harness.validate(request)

    print("\n" + "=" * 60)
    print("DATA VALIDATION")
    print("=" * 60)

    print(f"Valid: {valid}")

    if not valid:

        for error in errors:
            print(f"- {error}")

        handoff = handoff_harness.check(
            data_valid=False,
            permission_allowed=True,
            task_completed=False,
            error_count=len(errors),
        )

        print("\nHANDOFF:")
        print(f"Handoff: {handoff.should_handoff}")
        print(f"Reason: {handoff.reason}")

        return

    # --------------------------------------------------------
    # STEP 2 — SEARCH FLIGHTS
    # --------------------------------------------------------

    print("\n" + "=" * 60)
    print("EXECUTION")
    print("=" * 60)

    print(
        f"Searching flights: "
        f"{plan.origin} -> {plan.destination} "
        f"on {plan.date}, "
        f"{plan.passengers} passenger(s)"
    )

    search_output = search_flights.invoke({
        "origin": plan.origin,
        "destination": plan.destination,
        "date": plan.date,
        "passengers": plan.passengers,
    })

    print("\nSEARCH RESULT:")
    print(search_output)

    search_completion = completion_harness.check_search(
        search_output
    )

    print("\nSEARCH COMPLETION:")
    print(f"Completed: {search_completion.completed}")
    print(f"Reason: {search_completion.reason}")

    if not search_completion.completed:

        handoff = handoff_harness.check(
            data_valid=True,
            permission_allowed=True,
            task_completed=False,
            error_count=1,
        )

        print("\nHANDOFF:")
        print(f"Handoff: {handoff.should_handoff}")
        print(f"Reason: {handoff.reason}")

        return

    # --------------------------------------------------------
    # STEP 3 — IF NO BOOKING REQUEST
    # --------------------------------------------------------

    if plan.flight_id is None:

        handoff = handoff_harness.check(
            data_valid=True,
            permission_allowed=True,
            task_completed=True,
            error_count=0,
        )

        print("\n" + "=" * 60)
        print("FINAL HANDOFF CHECK")
        print("=" * 60)

        print(f"Handoff: {handoff.should_handoff}")
        print(f"Reason: {handoff.reason}")

        return search_output

    # --------------------------------------------------------
    # STEP 4 — CHECK FLIGHT
    # --------------------------------------------------------

    print("\n" + "=" * 60)
    print("CHECK FLIGHT")
    print("=" * 60)

    flight_output = check_flight.invoke({
        "flight_id": plan.flight_id,
    })

    print(flight_output)

    if "not found" in str(flight_output).lower():

        handoff = handoff_harness.check(
            data_valid=True,
            permission_allowed=True,
            task_completed=False,
            error_count=1,
        )

        print("\nHANDOFF:")
        print(f"Handoff: {handoff.should_handoff}")
        print(f"Reason: {handoff.reason}")

        return

    # --------------------------------------------------------
    # STEP 5 — PERMISSION CHECK
    # --------------------------------------------------------

    print("\n" + "=" * 60)
    print("BOOKING PERMISSION")
    print("=" * 60)

    permission = permission_harness.check(
        action="book_flight",
        user_confirmed=plan.user_confirmed,
    )

    print(f"Allowed: {permission.allowed}")
    print(f"Reason: {permission.reason}")

    if not permission.allowed:

        print("\nBOOKING STOPPED.")
        print("The user must explicitly confirm the booking.")

        handoff = handoff_harness.check(
            data_valid=True,
            permission_allowed=False,
            task_completed=False,
            error_count=0,
        )

        print("\nHANDOFF:")
        print(f"Handoff: {handoff.should_handoff}")
        print(f"Reason: {handoff.reason}")
        print(f"Target: {handoff.target}")

        return

    # --------------------------------------------------------
    # STEP 6 — BOOK FLIGHT
    # --------------------------------------------------------

    print("\n" + "=" * 60)
    print("BOOKING")
    print("=" * 60)

    passenger_name = plan.passenger_name or "Demo User"

    booking_output = book_flight.invoke({
        "flight_id": plan.flight_id,
        "passenger_name": passenger_name,
        "passengers": plan.passengers,
        "user_confirmed": True,
    })

    print(booking_output)

    # --------------------------------------------------------
    # STEP 7 — COMPLETION CHECK
    # --------------------------------------------------------

    completion = completion_harness.check_booking(
        booking_output
    )

    print("\n" + "=" * 60)
    print("BOOKING COMPLETION")
    print("=" * 60)

    print(f"Completed: {completion.completed}")
    print(f"Reason: {completion.reason}")

    # --------------------------------------------------------
    # STEP 8 — HANDOFF
    # --------------------------------------------------------

    handoff = handoff_harness.check(
        data_valid=True,
        permission_allowed=True,
        task_completed=completion.completed,
        error_count=0 if completion.completed else 1,
    )

    print("\n" + "=" * 60)
    print("FINAL HANDOFF CHECK")
    print("=" * 60)

    print(f"Handoff: {handoff.should_handoff}")
    print(f"Reason: {handoff.reason}")

    if handoff.should_handoff:
        print(f"Target: {handoff.target}")

    return booking_output


# ============================================================
# 5. TEST 1 — SEARCH
# ============================================================

def test_search():

    print("\n")
    print("#" * 60)
    print("TEST 1 — PLAN-THEN-EXECUTE SEARCH")
    print("#" * 60)

    user_message = (
        "Find flights from SGN to HAN on 2026-10-20 "
        "for 1 passenger."
    )

    print("\nUSER:")
    print(user_message)

    print("\nCreating plan...")

    plan = create_plan(user_message)

    print("\nGENERATED PLAN:")
    print(f"Origin: {plan.origin}")
    print(f"Destination: {plan.destination}")
    print(f"Date: {plan.date}")
    print(f"Passengers: {plan.passengers}")

    execute_plan(plan)


# ============================================================
# 6. TEST 2 — BOOKING WITHOUT CONFIRMATION
# ============================================================

def test_booking_without_confirmation():

    print("\n")
    print("#" * 60)
    print("TEST 2 — BOOKING WITHOUT CONFIRMATION")
    print("#" * 60)

    user_message = (
        "Book flight VN123 from SGN to HAN on 2026-10-20 "
        "for 1 passenger. Passenger name is Nguyen Van A."
    )

    print("\nUSER:")
    print(user_message)

    print("\nCreating plan...")

    plan = create_plan(user_message)

    print("\nGENERATED PLAN:")
    print(f"Flight ID: {plan.flight_id}")
    print(f"Passenger: {plan.passenger_name}")
    print(f"Confirmed: {plan.user_confirmed}")

    execute_plan(plan)


# ============================================================
# 7. TEST 3 — BOOKING WITH CONFIRMATION
# ============================================================

def test_booking_with_confirmation():

    print("\n")
    print("#" * 60)
    print("TEST 3 — BOOKING WITH CONFIRMATION")
    print("#" * 60)

    user_message = (
        "I confirm the booking. "
        "Book flight VN123 from SGN to HAN on 2026-10-20 "
        "for 1 passenger. Passenger name is Nguyen Van A."
    )

    print("\nUSER:")
    print(user_message)

    print("\nCreating plan...")

    plan = create_plan(user_message)

    print("\nGENERATED PLAN:")
    print(f"Flight ID: {plan.flight_id}")
    print(f"Passenger: {plan.passenger_name}")
    print(f"Confirmed: {plan.user_confirmed}")

    execute_plan(plan)


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    test_search()

    test_booking_without_confirmation()

    test_booking_with_confirmation()