"""
Read a summary file in tests: the rows of its tables and its last line.

The summary file is written for a reader (dice.io.results.write_summary_file):
a result block, a table with one row per statistic and one column per
estimator, notes, a table of the simulation parameters, and a last line with
the DICE version and the seed. Cells of a row are at least two spaces apart.
"""

from __future__ import annotations

import re

CELL_GAP = re.compile(r"\s{2,}")
NUMBER = re.compile(r"(?<![\^\w.])-?(?:\d+\.?\d*|\.\d+)(?:[eE][+-]?\d+)?|\binf\b|\bnan\b")


def summary_rows(summary: str) -> list[tuple[str, list[str]]]:
    """Each table row of a summary, in file order: its label and its cells.

    A line with a single cell (a heading, a note, a line of the result block)
    is not a row. The row of the column names has the first name as its label.
    """
    rows = []
    for line in summary.splitlines():
        cells = CELL_GAP.split(line.strip())
        if len(cells) >= 2:
            rows.append((cells[0], cells[1:]))
    return rows


def summary_values(summary: str) -> dict[str, list[str]]:
    """Each row label mapped to its cells; a label that recurs collects the cells of every row, in order."""
    values: dict[str, list[str]] = {}
    for label, cells in summary_rows(summary):
        values.setdefault(label, []).extend(cells)
    return values


def row_after(summary: str, label: str, occurrence: int = 0) -> tuple[str, list[str]]:
    """The row after a row with this label, for example the 95% interval of a statistic."""
    rows = summary_rows(summary)
    indices = [index for index, (row_label, _) in enumerate(rows) if row_label == label]
    return rows[indices[occurrence] + 1]


def result_block(summary: str) -> list[str]:
    """The lines of the result block: from the line that starts with 'Result' to the next empty line."""
    lines = summary.splitlines()
    start = next(index for index, line in enumerate(lines) if line.startswith("Result"))
    end = lines.index("", start)
    return lines[start:end]


def last_line(summary: str) -> str:
    """The last line of the summary: the DICE version and the seed."""
    return summary.splitlines()[-1]


def summary_skeleton(summary: str, version: str) -> list[str]:
    """The summary lines with every number masked as '#' and every gap between cells as two spaces."""
    skeleton = []
    for line in summary.splitlines():
        line = line.replace(f"DICE {version},", "DICE <version>,")
        line = NUMBER.sub("#", line)
        line = re.sub(r"^ {3,}", "  ", line)
        line = re.sub(r"(?<=\S) {2,}", "  ", line)
        skeleton.append(line)
    return skeleton
