from dataclasses import dataclass


@dataclass
class Residual:
    text: str


def extract_residual(full_text: str, skeleton_text: str) -> Residual:
    """
    v0.1 residual: just keep everything except the skeleton header+goal
    (simple + gzip-friendly)
    """
    if full_text.startswith(skeleton_text):
        rest = full_text[len(skeleton_text):]
    else:
        rest = full_text
    return Residual(text=rest)
