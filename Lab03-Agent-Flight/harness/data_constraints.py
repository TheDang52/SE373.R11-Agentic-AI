from dataclasses import dataclass
from datetime import datetime


@dataclass
class FlightRequest:
    origin: str
    destination: str
    date: str
    passengers: int = 1


class DataConstraintHarness:
    """
    Validate user-provided flight request.

    The constraints are represented as data/rules
    instead of relying on the LLM to validate them.
    """

    ALLOWED_AIRPORTS = {
        "SGN",
        "HAN",
        "DAD",
        "HPH",
        "CXR",
        "PQC",
    }

    MIN_PASSENGERS = 1
    MAX_PASSENGERS = 9

    def validate(self, request: FlightRequest) -> tuple[bool, list[str]]:
        errors = []

        # ------------------------------------------------
        # Airport validation
        # ------------------------------------------------

        origin = request.origin.upper()
        destination = request.destination.upper()

        if origin not in self.ALLOWED_AIRPORTS:
            errors.append(
                f"Invalid origin airport: {origin}"
            )

        if destination not in self.ALLOWED_AIRPORTS:
            errors.append(
                f"Invalid destination airport: {destination}"
            )

        if origin == destination:
            errors.append(
                "Origin and destination cannot be the same."
            )

        # ------------------------------------------------
        # Passenger validation
        # ------------------------------------------------

        if request.passengers < self.MIN_PASSENGERS:
            errors.append(
                f"Passengers must be >= {self.MIN_PASSENGERS}."
            )

        if request.passengers > self.MAX_PASSENGERS:
            errors.append(
                f"Passengers must be <= {self.MAX_PASSENGERS}."
            )

        # ------------------------------------------------
        # Date validation
        # ------------------------------------------------

        try:
            flight_date = datetime.strptime(
                request.date,
                "%Y-%m-%d"
            ).date()

            if flight_date < datetime.now().date():
                errors.append(
                    "Flight date cannot be in the past."
                )

        except ValueError:
            errors.append(
                "Date must use YYYY-MM-DD format."
            )

        return len(errors) == 0, errors