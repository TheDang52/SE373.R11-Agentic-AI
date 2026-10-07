import os

from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain_google_genai import ChatGoogleGenerativeAI

from tools.airline_tools import (
    search_flights,
    check_flight,
    book_flight,
    cancel_booking,
)

from harness.data_constraints import (
    FlightRequest,
    DataConstraintHarness,
)

from harness.completion_check import (
    CompletionHarness,
)

from harness.permission_check import (
    PermissionHarness,
)

from harness.handoff import (
    HandoffHarness,
)


# ============================================================
# LOAD ENVIRONMENT
# ============================================================

load_dotenv()

if not os.getenv("GEMINI_API_KEY"):
    raise RuntimeError("GEMINI_API_KEY is not configured.")


# ============================================================
# GEMINI MODEL
# ============================================================

model = ChatGoogleGenerativeAI(
    model="gemini-3.5-flash-lite",
)


# ============================================================
# TOOLS
# ============================================================

tools = [
    search_flights,
    check_flight,
    book_flight,
    cancel_booking,
]


# ============================================================
# SYSTEM PROMPT
# ============================================================

SYSTEM_PROMPT = """
You are an airline booking assistant.

Your job is to help users search, inspect, book,
and cancel mock airline flights.

Available tools:
- search_flights
- check_flight
- book_flight
- cancel_booking

Rules:
1. Use tools whenever flight information is required.
2. Never invent flight information.
3. Before booking, make sure the user has explicitly
   confirmed the booking.
4. Never call book_flight with user_confirmed=True
   unless the user explicitly confirms the booking.
5. Clearly report tool results to the user.
6. If a request cannot be completed, explain why.
7. Follow the validation and permission rules enforced
   by the application.
"""


# ============================================================
# CREATE REACT AGENT
# ============================================================

agent = create_agent(
    model=model,
    tools=tools,
    system_prompt=SYSTEM_PROMPT,
)


# ============================================================
# HARNESS INSTANCES
# ============================================================

data_harness = DataConstraintHarness()
completion_harness = CompletionHarness()
permission_harness = PermissionHarness()
handoff_harness = HandoffHarness()


# ============================================================
# DATA VALIDATION
# ============================================================

def validate_flight_request(
    origin: str,
    destination: str,
    date: str,
    passengers: int = 1,
):
    request = FlightRequest(
        origin=origin,
        destination=destination,
        date=date,
        passengers=passengers,
    )

    return data_harness.validate(request)


# ============================================================
# PERMISSION CHECK
# ============================================================

def check_permission(
    action: str,
    user_confirmed: bool = False,
):
    return permission_harness.check(
        action=action,
        user_confirmed=user_confirmed,
    )


# ============================================================
# COMPLETION CHECK
# ============================================================

def check_completion(
    action: str,
    tool_output: str,
):
    if action == "search_flights":
        return completion_harness.check_search(tool_output)

    if action == "book_flight":
        return completion_harness.check_booking(tool_output)

    if action == "cancel_booking":
        return completion_harness.check_cancellation(tool_output)

    return None


# ============================================================
# HANDOFF CHECK
# ============================================================

def check_handoff(
    data_valid: bool,
    permission_allowed: bool,
    task_completed: bool,
    error_count: int = 0,
):
    return handoff_harness.check(
        data_valid=data_valid,
        permission_allowed=permission_allowed,
        task_completed=task_completed,
        error_count=error_count,
    )


# ============================================================
# RUN REACT AGENT
# ============================================================

def run_agent(user_message: str) -> str:

    result = agent.invoke(
        {
            "messages": [
                {
                    "role": "user",
                    "content": user_message,
                }
            ]
        }
    )

    messages = result["messages"]

    for message in reversed(messages):
        if getattr(message, "type", None) == "ai":
            return message.content

    return "No response generated."


# ============================================================
# TEST 1 — SEARCH
# ============================================================

def test_search():

    print("=" * 60)
    print("TEST 1 — REACT SEARCH")
    print("=" * 60)

    request = validate_flight_request(
        origin="SGN",
        destination="HAN",
        date="2026-10-20",
        passengers=1,
    )

    valid, errors = request

    print("\nDATA CONSTRAINT:")
    print("Valid:", valid)

    if errors:
        for error in errors:
            print("-", error)
        return

    query = (
        "Find flights from SGN to HAN "
        "on 2026-10-20 for 1 passenger."
    )

    print("\nUSER:")
    print(query)

    print("\nAGENT:")
    response = run_agent(query)
    print(response)


# ============================================================
# TEST 2 — PERMISSION
# ============================================================

def test_permission():

    print("\n" + "=" * 60)
    print("TEST 2 — BOOKING PERMISSION")
    print("=" * 60)

    permission = check_permission(
        action="book_flight",
        user_confirmed=False,
    )

    print("\nWITHOUT USER CONFIRMATION:")
    print("Allowed:", permission.allowed)
    print("Reason:", permission.reason)

    permission = check_permission(
        action="book_flight",
        user_confirmed=True,
    )

    print("\nWITH USER CONFIRMATION:")
    print("Allowed:", permission.allowed)
    print("Reason:", permission.reason)


# ============================================================
# TEST 3 — COMPLETION
# ============================================================

def test_completion():

    print("\n" + "=" * 60)
    print("TEST 3 — COMPLETION CHECK")
    print("=" * 60)

    success_output = (
        "BOOKING_SUCCESS: BK0001 | "
        "Flight VN123 | Passenger Nguyen Van A"
    )

    result = check_completion(
        action="book_flight",
        tool_output=success_output,
    )

    print("\nSUCCESS OUTPUT:")
    print("Completed:", result.completed)
    print("Reason:", result.reason)

    failed_output = (
        "BOOKING_BLOCKED: User confirmation required."
    )

    result = check_completion(
        action="book_flight",
        tool_output=failed_output,
    )

    print("\nFAILED OUTPUT:")
    print("Completed:", result.completed)
    print("Reason:", result.reason)


# ============================================================
# TEST 4 — HANDOFF
# ============================================================

def test_handoff():

    print("\n" + "=" * 60)
    print("TEST 4 — HANDOFF")
    print("=" * 60)

    result = check_handoff(
        data_valid=True,
        permission_allowed=True,
        task_completed=True,
    )

    print("\nCOMPLETED TASK:")
    print("Handoff:", result.should_handoff)
    print("Reason:", result.reason)

    result = check_handoff(
        data_valid=True,
        permission_allowed=False,
        task_completed=False,
    )

    print("\nPERMISSION DENIED:")
    print("Handoff:", result.should_handoff)
    print("Reason:", result.reason)
    print("Target:", result.target)


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    test_search()

    test_permission()

    test_completion()

    test_handoff()