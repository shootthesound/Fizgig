"""MOSS-Image's Clean-end share by epoch (its lownoise_pct box), read one way by the driver as it trains (driver.py
_share) and by the Slider card's plan (families/form.py clean_share_at). Plain Python: the form imports it without
torch."""


def clean_share(text, epoch, cycle=0):
    """The Clean-end share for 1-based `epoch` from the box as typed (a "%" dropped) and whether the epoch trains at
    half the LR: (share as text, half) - ("", False) for an empty box, the reference schedule. P: one share, as typed.
    Split "A:N:B": A% for the first N epochs, then B%. Slide "A:H:B:E": A% until epoch H, then sliding in a straight
    line to B% by epoch E (and B% after); "A:H:B:E:R": and after E, the higher of A and B, with every Rth epoch at the
    lower one, at half the LR (A or B first): those epochs reset structure, but at the full rate they cost likeness
    each time (Peter, 28 Oct 2026). `cycle`: a fine-tune's epochs a rotation (driver.ft_cycle), whose Split or Slide
    counts rotations. ValueError for a Split or a Slide it can't read."""
    pct = str(text or "").replace("%", "").strip()
    if ":" not in pct:
        return pct, False
    parts = [float(x) for x in pct.split(":")]
    if len(parts) not in (3, 4, 5):
        raise ValueError(f"Clean-end share {pct!r}: 3, 4 or 5 numbers")
    first, until, rest = parts[:3]
    end = parts[3] if len(parts) >= 4 else until
    every = int(parts[4]) if len(parts) == 5 else 0
    e = -(-int(epoch) // cycle) if cycle else int(epoch)      # a fine-tune: the rotation this epoch is in
    share = first if e <= until else rest if e >= end else first + (rest - first) * (e - until) / (end - until)
    half = False
    if every > 0 and e > end:
        refresh = (e - end) % every == 0          # the refresh epoch is at the lower share, the rest at the higher
        share = min(first, rest) if refresh else max(first, rest)
        half = first != rest and refresh
    return f"{round(share, 2):g}", half
