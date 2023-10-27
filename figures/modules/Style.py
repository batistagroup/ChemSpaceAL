from typing import Dict, Tuple, Callable

class Style:
    biscale: Dict[str, Tuple[Callable[[float], str], str]] = {
        "blue": (lambda a: f"rgba(3, 4, 94, {a})", "#023e8a"),
        "purple": (lambda a: f"rgba(123, 44, 191, {a})", "#c77dff"),
        "green": (lambda a: f"rgba(0, 128, 0, {a})", "#70e000"),
        "red": (lambda a: f"rgba(164, 19, 60, {a})", "#ff4d6d"),
        "orange": (lambda a: f"rgba(232, 93, 4, {a})", "#faa307"),
        "teal": (lambda a: f"rgba(0, 106, 113, {a})", "#48cae4"),
        "lime": (lambda a: f"rgba(103, 148, 54, {a})", "#aee833"),
        "gold": (lambda a: f"rgba(176, 141, 87, {a})", "#f0e442"),
        "beige": (lambda a: f"rgba(191, 165, 138, {a})", "#f5deb3"),
        "maroon": (lambda a: f"rgba(74, 0, 0, {a})", "#800000"),
        "dark_grey": (lambda a: f"rgba(0, 0, 0, {a})", "#2e2e2e"),
    }

    gaussian_opacity = 0.20
