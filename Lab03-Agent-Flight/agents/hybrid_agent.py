import os
from dotenv import load_dotenv
from pydantic import BaseModel, Field
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.agents import create_agent

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
# 1. MODEL
# ============================================================

model = ChatGoogleGenerativeAI(
    model="gemini-3.5-flash-lite",
)


# ============================================================
# 2. PLANNER
# ============================================================

class HybridPlan(BaseModel):
    origin: str = Field(description="Origin airport code")
    destination: str = Field(description="Destination airport code")
    date: str = Field(description="Flight date in YYYY-MM-DD format")
    passengers: int = Field(default=1)

    flight_id: str | None = None
    passenger_name: str | None = None

    objective: str = Field(
        description="search or booking"
    )

    steps: list[str]


planner = model.with_structured_output(HybridPlan)


# ============================================================
# 3. REACT EXECUTOR
# ============================================================

SYSTEM_PROMPT = """
You are the execution component of a hybrid airline agent.

The application has already performed validation and permission checks.

Available tools:
- search_flights
- check_flight
- book_flight

Rules:
1. Actually call the appropriate tools.
2. Never invent tool results.
3. For search requests, call search_flights.
4. For booking requests:
   - call check_flight first
   - then call book_flight
5. Return the actual tool result.
6. Do not claim success unless the tool reports success.
"""

react_agent = create_agent(
    model=model,
    tools=[
        search_flights,
        check_flight,
        book_flight,
    ],
    system_prompt=SYSTEM_PROMPT,
)


# ============================================================
# 4. HARNESS
# ============================================================

data_harness = DataConstraintHarness()
completion_harness = CompletionHarness()
permission_harness = PermissionHarness()
handoff_harness = HandoffHarness()


# ============================================================
# 5. CREATE PLAN
# ============================================================

def create_plan(user_message: str) -> HybridPlan:

    prompt = f"""
Create a high-level plan for this airline request.

User request:
{user_message}

Extract:
- origin
- destination
- date
- passengers
- flight ID if specified
- passenger name if specified
- objective: search or booking

Important:
Do NOT infer or assume user confirmation.
The application will determine permission separately.

Create 2-5 high-level steps.
"""

    return planner.invoke(prompt)


# ============================================================
# 6. DETECT EXPLICIT USER CONFIRMATION
# ============================================================

def detect_user_confirmation(user_message: str) -> bool:

    text = user_message.lower().strip()

    confirmation_phrases = [
        "i confirm",
        "confirm the booking",
        "confirm booking",
        "yes, book",
        "yes book",
        "tôi xác nhận",
        "xác nhận đặt vé",
        "đồng ý đặt",
    ]

    return any(
        phrase in text
        for phrase in confirmation_phrases
    )


# ============================================================
# 7. EXTRACT TOOL OUTPUTS
# ============================================================

def get_tool_outputs(result):

    outputs = []

    for message in result["messages"]:

        message_type = getattr(message, "type", None)

        if message_type == "tool":

            content = message.content

            if isinstance(content, str):
                outputs.append(content)

            else:
                outputs.append(str(content))

    return outputs


# ============================================================
# 8. EXECUTE REACT
# ============================================================

def execute_with_react(plan: HybridPlan):

    context = f"""
Execute this high-level plan.

Objective:
{plan.objective}

Origin:
{plan.origin}

Destination:
{plan.destination}

Date:
{plan.date}

Passengers:
{plan.passengers}

Flight ID:
{plan.flight_id}

Passenger name:
{plan.passenger_name}

Plan:
{plan.steps}

Actually use the appropriate tools.
Return the actual tool results.
"""

    result = react_agent.invoke({
        "messages": [
            {
                "role": "user",
                "content": context,
            }
        ]
    })

    tool_outputs = get_tool_outputs(result)

    return result, tool_outputs


# ============================================================
# 9. HYBRID RUNNER
# ============================================================

def run_hybrid(user_message: str):

    print("\n" + "=" * 60)
    print("USER REQUEST")
    print("=" * 60)

    print(user_message)

    # --------------------------------------------------------
    # PLANNER
    # --------------------------------------------------------

    print("\n" + "=" * 60)
    print("PLANNER")
    print("=" * 60)

    plan = create_plan(user_message)

    print(f"Objective: {plan.objective}")
    print(f"Origin: {plan.origin}")
    print(f"Destination: {plan.destination}")
    print(f"Date: {plan.date}")
    print(f"Passengers: {plan.passengers}")
    print(f"Flight ID: {plan.flight_id}")
    print(f"Passenger: {plan.passenger_name}")

    print("\nHigh-level plan:")

    for i, step in enumerate(plan.steps, start=1):
        print(f"{i}. {step}")

    # --------------------------------------------------------
    # REAL PERMISSION — CONTROLLED BY APPLICATION
    # --------------------------------------------------------

    user_confirmed = detect_user_confirmation(user_message)

    print("\nExplicit confirmation detected:", user_confirmed)

    # --------------------------------------------------------
    # DATA CONSTRAINT
    # --------------------------------------------------------

    print("\n" + "=" * 60)
    print("DATA CONSTRAINT")
    print("=" * 60)

    request = FlightRequest(
        origin=plan.origin,
        destination=plan.destination,
        date=plan.date,
        passengers=plan.passengers,
    )

    valid, errors = data_harness.validate(request)

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

        print("\nHANDOFF")
        print(f"Handoff: {handoff.should_handoff}")
        print(f"Reason: {handoff.reason}")

        return

    # --------------------------------------------------------
    # PERMISSION
    # --------------------------------------------------------

    if plan.objective == "booking" or plan.flight_id is not None:

        print("\n" + "=" * 60)
        print("PERMISSION CHECK")
        print("=" * 60)

        permission = permission_harness.check(
            action="book_flight",
            user_confirmed=user_confirmed,
        )

        print(f"Allowed: {permission.allowed}")
        print(f"Reason: {permission.reason}")

        if not permission.allowed:

            print("\nBOOKING STOPPED.")

            handoff = handoff_harness.check(
                data_valid=True,
                permission_allowed=False,
                task_completed=False,
                error_count=0,
            )

            print("\nHANDOFF")
            print(f"Handoff: {handoff.should_handoff}")
            print(f"Reason: {handoff.reason}")
            print(f"Target: {handoff.target}")

            return

    # --------------------------------------------------------
    # REACT EXECUTOR
    # --------------------------------------------------------

    print("\n" + "=" * 60)
    print("REACT EXECUTOR")
    print("=" * 60)

    result, tool_outputs = execute_with_react(plan)

    # Display actual tool outputs
    print("\nACTUAL TOOL OUTPUTS:")

    if tool_outputs:

        for output in tool_outputs:
            print(output)

    else:
        print("No tool output detected.")

    # --------------------------------------------------------
    # COMPLETION CHECK
    # --------------------------------------------------------

    print("\n" + "=" * 60)
    print("COMPLETION CHECK")
    print("=" * 60)

    if plan.flight_id is not None:

        completion = None

        for output in tool_outputs:

            check = completion_harness.check_booking(
                output
            )

            if check.completed:
                completion = check
                break

        if completion is None:
            completion = completion_harness.check_booking(
                "\n".join(tool_outputs)
            )

    else:

        completion = None

        for output in tool_outputs:

            check = completion_harness.check_search(
                output
            )

            if check.completed:
                completion = check
                break

        if completion is None:
            completion = completion_harness.check_search(
                "\n".join(tool_outputs)
            )

    print(f"Completed: {completion.completed}")
    print(f"Reason: {completion.reason}")

    # --------------------------------------------------------
    # HANDOFF
    # --------------------------------------------------------

    handoff = handoff_harness.check(
        data_valid=True,
        permission_allowed=True,
        task_completed=completion.completed,
        error_count=0 if completion.completed else 1,
    )

    print("\n" + "=" * 60)
    print("HANDOFF CHECK")
    print("=" * 60)

    print(f"Handoff: {handoff.should_handoff}")
    print(f"Reason: {handoff.reason}")

    if handoff.should_handoff:
        print(f"Target: {handoff.target}")


# ============================================================
# TEST 1 — SEARCH
# ============================================================

def test_search():

    print("\n")
    print("#" * 60)
    print("TEST 1 — HYBRID SEARCH")
    print("#" * 60)

    run_hybrid(
        "Find flights from SGN to HAN on 2026-10-20 "
        "for 1 passenger."
    )


# ============================================================
# TEST 2 — BOOKING WITHOUT CONFIRMATION
# ============================================================

def test_booking_without_confirmation():

    print("\n")
    print("#" * 60)
    print("TEST 2 — HYBRID BOOKING WITHOUT CONFIRMATION")
    print("#" * 60)

    run_hybrid(
        "Book flight VN123 from SGN to HAN on 2026-10-20 "
        "for 1 passenger. Passenger name is Nguyen Van A."
    )


# ============================================================
# TEST 3 — BOOKING WITH CONFIRMATION
# ============================================================

def test_booking_with_confirmation():

    print("\n")
    print("#" * 60)
    print("TEST 3 — HYBRID BOOKING WITH CONFIRMATION")
    print("#" * 60)

    run_hybrid(
        "I confirm the booking. "
        "Book flight VN123 from SGN to HAN on 2026-10-20 "
        "for 1 passenger. Passenger name is Nguyen Van A."
    )


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    test_search()

    test_booking_without_confirmation()

    test_booking_with_confirmation()