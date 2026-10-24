"""Wound cases, captures, and trends. In-memory in v0.1.0, like Weeks 1-2."""

from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field


@dataclass
class Capture:
    id: str
    taken_on: str            # ISO date the photo was taken
    sha256: str
    area_px: int | None = None

    def to_dict(self) -> dict:
        return {"id": self.id, "taken_on": self.taken_on,
                "sha256": self.sha256[:12], "area_px": self.area_px}


@dataclass
class WoundCase:
    id: str
    label: str                       # redacted before it reaches here
    created_at: float = field(default_factory=time.time)
    captures: list[Capture] = field(default_factory=list)
    comparisons: list[dict] = field(default_factory=list)
    _images: dict[str, bytes] = field(default_factory=dict)

    def add_capture(self, image_bytes: bytes, taken_on: str,
                    sha256: str) -> Capture:
        cap = Capture(id=uuid.uuid4().hex[:10], taken_on=taken_on,
                      sha256=sha256)
        self.captures.append(cap)
        self._images[cap.id] = image_bytes
        return cap

    def image_for(self, capture_id: str) -> bytes | None:
        return self._images.get(capture_id)

    def trend(self) -> dict:
        """Area series from comparisons, oldest first. The trend, not any
        single pair, is what a care team actually acts on."""
        points = []
        for cap in self.captures:
            if cap.area_px is not None:
                points.append({"taken_on": cap.taken_on,
                               "area_px": cap.area_px})
        change = None
        if len(points) >= 2 and points[0]["area_px"]:
            change = round(100.0 * (points[-1]["area_px"] - points[0]["area_px"])
                           / points[0]["area_px"], 2)
        return {"points": points, "total_change_pct": change,
                "captures": len(self.captures),
                "comparisons": len(self.comparisons)}

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "label": self.label,
            "created_at": self.created_at,
            "captures": [c.to_dict() for c in self.captures],
            "comparisons": self.comparisons,
            "trend": self.trend(),
        }


def new_case(label: str) -> WoundCase:
    return WoundCase(id=f"wnd-{uuid.uuid4().hex[:8]}", label=label)
