"""LaTeX stripping and text normalization (Phase 2, run BEFORE any extraction).

arXiv abstracts carry raw LaTeX. Left in place it poisons every extractor:
`$\\alpha$` becomes a candidate, `\\cite{foo}` becomes a candidate, and inline
math fragments the n-gram stream. Stripping happens once, here, so that the
extractors and the dictionary re-scan see byte-identical text.
"""

from __future__ import annotations

import re

# Commands whose ARGUMENT is discarded along with the command: references,
# citations, labels, URLs. Keeping their contents would inject bibkeys.
_DROP_WITH_ARG = re.compile(
    r"\\(?:cite[a-zA-Z]*|ref|eqref|label|url|href|includegraphics|bibliography"
    r"|footnote|autoref|pageref|nocite)\s*(?:\[[^\]]*\])?\s*\{[^{}]*\}",
    re.I,
)

# Display and inline math. Order matters: $$ before $, and the escaped \$ is
# protected first so a literal dollar sign does not open a math span.
_ESCAPED_DOLLAR = re.compile(r"\\\$")
_MATH = re.compile(
    r"\$\$.*?\$\$"          # $$ ... $$
    r"|\$[^$]*?\$"          # $ ... $
    r"|\\\(.*?\\\)"         # \( ... \)
    r"|\\\[.*?\\\]",        # \[ ... \]
    re.S,
)

# Whole environments, math or otherwise (equation, align, tabular, ...).
_ENV = re.compile(r"\\begin\s*\{([^{}]*)\}.*?\\end\s*\{\1\}", re.S)

# A command with optional/required args where the argument IS text worth
# keeping: \textit{deep learning} -> deep learning.
_CMD_WITH_TEXT = re.compile(r"\\[a-zA-Z@]+\s*(?:\[[^\]]*\])?\s*\{([^{}]*)\}")

# A bare command with no argument: \alpha, \\, \newline.
_BARE_CMD = re.compile(r"\\[a-zA-Z@]+\*?|\\\\|\\[^a-zA-Z\s]")

_BRACES = re.compile(r"[{}]")
_WS = re.compile(r"\s+")


def strip_latex(text: str) -> str:
    """Remove math mode and LaTeX commands, preserving ordinary prose."""
    if not text:
        return ""
    t = _ESCAPED_DOLLAR.sub(" ", text)
    t = _ENV.sub(" ", t)
    t = _DROP_WITH_ARG.sub(" ", t)
    t = _MATH.sub(" ", t)
    # Unwrap text-bearing commands repeatedly, for nesting like \emph{\textbf{x}}.
    for _ in range(3):
        new = _CMD_WITH_TEXT.sub(r" \1 ", t)
        if new == t:
            break
        t = new
    t = _BARE_CMD.sub(" ", t)
    t = _BRACES.sub(" ", t)
    return _WS.sub(" ", t).strip()


def paper_text(title: str, abstract: str) -> str:
    """The unit of extraction: title and abstract, LaTeX-stripped.

    A period is inserted between them so the title cannot form an n-gram with
    the abstract's first word.
    """
    title = strip_latex(title or "").rstrip(" .")
    abstract = strip_latex(abstract or "")
    if title and abstract:
        return f"{title}. {abstract}"
    return title or abstract
