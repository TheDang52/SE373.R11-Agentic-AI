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


def test_data_constraints():

    print("=" * 60)
    print("TEST 1 — DATA CONSTRAINTS")
    print("=" * 60)

    harness = DataConstraintHarness()

    request = FlightRequest(
        origin="SGN",
        destination="HAN",
        date="2026-10-20",
        passengers=2,
    )

    valid, errors = harness.validate(request)

    print("Valid:", valid)
    print("Errors:", errors)


def test_invalid_data():

    print("\n" + "=" * 60)
    print("TEST 2 — INVALID DATA")
    print("=" * 60)

    harness = DataConstraintHarness()

    request = FlightRequest(
        origin="XXX",
        destination="SGN",
        date="invalid-date",
        passengers=20,
    )

    valid, errors = harness.validate(request)

    print("Valid:", valid)

    for error in errors:
        print("-", error)


def test_permission():

    print("\n" + "=" * 60)
    print("TEST 3 — PERMISSION")
    print("=" * 60)

    harness = PermissionHarness()

    result = harness.check(
        action="book_flight",
        user_confirmed=False,
    )

    print("Allowed:", result.allowed)
    print("Reason:", result.reason)

    result = harness.check(
        action="book_flight",
        user_confirmed=True,
    )

    print("Allowed:", result.allowed)
    print("Reason:", result.reason)


def test_completion():

    print("\n" + "=" * 60)
    print("TEST 4 — COMPLETION")
    print("=" * 60)

    harness = CompletionHarness()

    result = harness.check_booking(
        "BOOKING_SUCCESS\nBooking ID: BK0001"
    )

    print("Completed:", result.completed)
    print("Reason:", result.reason)

    result = harness.check_booking(
        "BOOKING_BLOCKED"
    )

    print("Completed:", result.completed)
    print("Reason:", result.reason)


def test_handoff():

    print("\n" + "=" * 60)
    print("TEST 5 — HANDOFF")
    print("=" * 60)

    harness = HandoffHarness()

    result = harness.check(
        data_valid=True,
        permission_allowed=True,
        task_completed=True,
    )

    print("Handoff:", result.should_handoff)
    print("Reason:", result.reason)

    result = harness.check(
        data_valid=True,
        permission_allowed=False,
        task_completed=False,
    )

    print("Handoff:", result.should_handoff)
    print("Reason:", result.reason)
    print("Target:", result.target)


if __name__ == "__main__":
    test_data_constraints()
    test_invalid_data()
    test_permission()
    test_completion()
    test_handoff()