import csv
import os
from typing import Dict, Any
from pathlib import Path


class CSVLogger:
    """Logger for training metrics to CSV."""

    def __init__(self, path: str, header: list):
        self.path = Path(path)
        self.header = header

        self.path.parent.mkdir(parents=True, exist_ok=True)

        with open(self.path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(self.header)

    def log(self, metrics: Dict[str, Any]):
        """Log a row of metrics."""
        row = [metrics.get(h, "") for h in self.header]

        with open(self.path, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(row)

    def load_history(self) -> Dict[str, list]:
        """Load logged history."""
        history = {h: [] for h in self.header}

        with open(self.path, "r") as f:
            reader = csv.DictReader(f)
            for row in reader:
                for h in self.header:
                    val = row.get(h, "")
                    if val != "":
                        try:
                            history[h].append(float(val))
                        except ValueError:
                            history[h].append(val)
                    else:
                        history[h].append(None)

        return history
