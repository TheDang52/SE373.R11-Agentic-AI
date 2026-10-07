from dataclasses import dataclass


@dataclass
class HandoffResult:
    should_handoff: bool
    reason: str
    target: str


class HandoffHarness:
    """
    Decide whether the task should be transferred
    to a human/operator.
    """

    def check(
        self,
        *,
        data_valid: bool,
        permission_allowed: bool,
        task_completed: bool,
        error_count: int = 0,
    ) -> HandoffResult:

        # ------------------------------------------------
        # Invalid data
        # ------------------------------------------------

        if not data_valid:
            return HandoffResult(
                should_handoff=True,
                reason="Required input data is invalid.",
                target="human_agent",
            )

        # ------------------------------------------------
        # Permission denied
        # ------------------------------------------------

        if not permission_allowed:
            return HandoffResult(
                should_handoff=True,
                reason="Agent does not have permission to continue.",
                target="human_agent",
            )

        # ------------------------------------------------
        # Too many failures
        # ------------------------------------------------

        if error_count >= 3:
            return HandoffResult(
                should_handoff=True,
                reason="Agent exceeded maximum allowed failures.",
                target="human_agent",
            )

        # ------------------------------------------------
        # Task not completed
        # ------------------------------------------------

        if not task_completed:
            return HandoffResult(
                should_handoff=True,
                reason="Task could not be completed automatically.",
                target="human_agent",
            )

        # ------------------------------------------------
        # Success
        # ------------------------------------------------

        return HandoffResult(
            should_handoff=False,
            reason="Task completed successfully.",
            target="none",
        )