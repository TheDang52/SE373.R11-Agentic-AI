from dataclasses import dataclass


@dataclass
class PermissionResult:
    allowed: bool
    reason: str


class PermissionHarness:
    """
    Control whether an agent is allowed to perform
    sensitive actions.
    """

    SENSITIVE_ACTIONS = {
        "book_flight",
        "cancel_booking",
    }

    def check(
        self,
        action: str,
        user_confirmed: bool = False,
    ) -> PermissionResult:

        # ------------------------------------------------
        # Read-only actions
        # ------------------------------------------------

        if action not in self.SENSITIVE_ACTIONS:
            return PermissionResult(
                allowed=True,
                reason="Action is read-only and does not require confirmation."
            )

        # ------------------------------------------------
        # Sensitive action
        # ------------------------------------------------

        if not user_confirmed:
            return PermissionResult(
                allowed=False,
                reason=(
                    f"Action '{action}' requires explicit "
                    "user confirmation."
                )
            )

        return PermissionResult(
            allowed=True,
            reason=(
                f"User confirmation granted for '{action}'."
            )
        )