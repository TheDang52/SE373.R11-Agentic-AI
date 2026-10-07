from dataclasses import dataclass
from typing import Optional


@dataclass
class CompletionResult:
    completed: bool
    reason: str


class CompletionHarness:
    """
    Determine whether an agent task is actually completed.

    Completion is checked using explicit programmatic criteria,
    not by asking the LLM whether it thinks the task is finished.
    """

    def check_search(
        self,
        tool_output: str,
    ) -> CompletionResult:

        if not tool_output:
            return CompletionResult(
                completed=False,
                reason="Search returned no output."
            )

        if "Found" in tool_output and "flight(s)" in tool_output:
            return CompletionResult(
                completed=True,
                reason="Flight search completed successfully."
            )

        return CompletionResult(
            completed=False,
            reason="No available flight was found."
        )

    def check_booking(
        self,
        tool_output: str,
    ) -> CompletionResult:

        if not tool_output:
            return CompletionResult(
                completed=False,
                reason="Booking returned no output."
            )

        if "BOOKING_SUCCESS" in tool_output:
            return CompletionResult(
                completed=True,
                reason="Booking completed successfully."
            )

        return CompletionResult(
            completed=False,
            reason="Booking was not completed."
        )

    def check_cancellation(
        self,
        tool_output: str,
    ) -> CompletionResult:

        if "CANCELLATION_SUCCESS" in tool_output:
            return CompletionResult(
                completed=True,
                reason="Cancellation completed successfully."
            )

        return CompletionResult(
            completed=False,
            reason="Cancellation was not completed."
        )