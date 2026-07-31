"""A restrained visual style, applied consistently.

Charts here are read once, in a document, alongside prose that already says what
they mean. So they carry no ornament: no gridlines competing with the data, no
boxed frames, no colour used decoratively.

Colour is reserved for the one thing it has to encode — outcome severity — and
runs from grey through amber to red, which is the ordering a reader assumes
without being told. Everything else is greyscale.

The palette is checked for adequate contrast in greyscale as well as colour,
because a reader printing the document should not lose the ordering.

Examples
--------
>>> from vision_zero.viz.theme import SEVERITY_COLOURS, figure
>>> fig, ax = figure()
>>> SEVERITY_COLOURS[2]
'#b3272d'
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import matplotlib as mpl
import matplotlib.pyplot as plt

if TYPE_CHECKING:
    from matplotlib.axes import Axes
    from matplotlib.figure import Figure

__all__ = [
    "GREY",
    "MUTED",
    "SEVERITY_COLOURS",
    "SEVERITY_LABELS",
    "apply_theme",
    "figure",
    "severity_colour",
]

#: Increasing severity, increasing salience. Distinguishable in greyscale.
SEVERITY_COLOURS: dict[int, str] = {
    0: "#b8b8b3",
    1: "#d99a2b",
    2: "#b3272d",
}

SEVERITY_LABELS: dict[int, str] = {
    0: "Material damage only",
    1: "At least one minor injury",
    2: "Serious injury or fatality",
}

GREY = "#4a4a48"
MUTED = "#9a9a95"

_RC: dict[str, Any] = {
    "figure.dpi": 120,
    "savefig.dpi": 200,
    "savefig.bbox": "tight",
    "font.size": 9,
    "axes.titlesize": 10,
    "axes.titleweight": "normal",
    "axes.labelsize": 9,
    "axes.labelcolor": GREY,
    "axes.edgecolor": MUTED,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "axes.axisbelow": True,
    "grid.color": "#e6e6e2",
    "grid.linewidth": 0.6,
    "xtick.color": GREY,
    "ytick.color": GREY,
    "xtick.direction": "out",
    "ytick.direction": "out",
    "legend.frameon": False,
    "legend.fontsize": 8,
    "figure.facecolor": "white",
    "axes.facecolor": "white",
}


def apply_theme() -> None:
    """Apply the style to the global Matplotlib configuration.

    Called by :func:`figure`, so most callers need not invoke it. Useful when a
    document draws with a Matplotlib interface directly.
    """
    mpl.rcParams.update(_RC)  # type: ignore[arg-type]


def figure(width: float = 6.0, height: float = 3.6, **kwargs: Any) -> tuple[Figure, Axes]:
    """Create a themed figure and axes.

    Defaults are sized for a single column of a rendered document, so that text
    in the figure matches text around it without rescaling.

    Returns
    -------
    tuple
        The figure and its axes.
    """
    apply_theme()
    fig, ax = plt.subplots(figsize=(width, height), **kwargs)
    return fig, ax


def severity_colour(label: int) -> str:
    """Return the colour for one severity class, grey for anything unknown."""
    return SEVERITY_COLOURS.get(int(label), MUTED)
