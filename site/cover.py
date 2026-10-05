"""Cover schematic for the frozen-critic page (Anthropic illustration style, no text).

A random actor (a box of scattered dots) writes a row of tokens; the frozen critic (a box with a snowflake) reads
them and returns one value per token (bars); three credit kernels, one per GAE lambda (long, short and single
decay), turn the values into advantages that travel back to the actor as a trail of dots.
"""
import random

IVORY, SLATE = "#F0EEE6", "#141413"
CLAY, OAT, KRAFT, MANILLA = "#D97757", "#E3DACC", "#D4A27F", "#EBDBBC"
W, H, OFF, SW = 1600, 1000, 12, 7


def cover_svg(fid="fc-wobble"):
    out = []
    f = f'filter="url(#{fid})"'

    def rect(x, y, w, h, rx, fill):
        out.append(f'<rect x="{x+OFF}" y="{y+OFF}" width="{w}" height="{h}" rx="{rx}" fill="{fill}"/>')
        out.append(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="{rx}" fill="none" stroke="{SLATE}" '
                   f'stroke-width="{SW}" stroke-linejoin="round" {f}/>')

    def circle(cx, cy, r, fill, stroke=True):
        out.append(f'<circle cx="{cx+OFF*0.6:.1f}" cy="{cy+OFF*0.6:.1f}" r="{r}" fill="{fill}"/>')
        if stroke:
            out.append(f'<circle cx="{cx}" cy="{cy}" r="{r}" fill="none" stroke="{SLATE}" stroke-width="{SW-2}" {f}/>')

    def line(d, width=SW):
        out.append(f'<path d="{d}" fill="none" stroke="{SLATE}" stroke-width="{width}" stroke-linecap="round" '
                   f'stroke-linejoin="round" {f}/>')

    out.append(f'<rect width="{W}" height="{H}" fill="{IVORY}"/>')
    # Actor: a box of randomly placed and sized dots, the random initialisation.
    rect(110, 300, 300, 320, 44, MANILLA)
    rng = random.Random(7)
    for i in range(5):
        for j in range(5):
            cx, cy = 155 + i * 52 + rng.uniform(-12, 12), 350 + j * 55 + rng.uniform(-12, 12)
            out.append(f'<circle cx="{cx:.1f}" cy="{cy:.1f}" r="{rng.uniform(6, 15):.1f}" '
                       f'fill="{rng.choice([SLATE, CLAY, KRAFT, OAT])}"/>')
    # Tokens the actor writes, travelling to the critic.
    fills = [OAT, KRAFT, CLAY, OAT, KRAFT, CLAY]
    xs = [480 + k * 96 for k in range(6)]
    for x, fill in zip(xs, fills):
        rect(x, 420, 70, 70, 16, fill)
    # Frozen critic: a large box with a snowflake.
    rect(1100, 240, 380, 360, 52, OAT)
    cx, cy, r = 1290, 420, 120
    import math
    for k in range(3):
        a = math.pi / 3 * k + math.pi / 2
        dx, dy = r * math.cos(a), r * math.sin(a)
        line(f"M{cx-dx:.1f} {cy-dy:.1f} L{cx+dx:.1f} {cy+dy:.1f}")
        for s in (1, -1):  # small branches near each tip
            tx, ty = cx + s * dx * 0.62, cy + s * dy * 0.62
            for b in (1, -1):
                ba = a + b * math.pi / 4 + (0 if s == 1 else math.pi)
                line(f"M{tx:.1f} {ty:.1f} L{tx + 38*math.cos(ba):.1f} {ty + 38*math.sin(ba):.1f}", SW - 1)
    # One value per token: bars under the tokens, falling along the response (a solution that goes wrong).
    base = 720
    for x, h in zip(xs, [150, 160, 150, 105, 70, 45]):
        rect(x + 10, base - h, 50, h, 10, CLAY)
    line(f"M{xs[0]-20} {base+6} L{xs[-1]+90} {base+6}", SW - 1)
    # Values leave the critic as a trail of dots toward the bars.
    for k in range(7):
        t = k / 6
        out.append(f'<circle cx="{1180 - 120*t:.1f}" cy="{620 + 70*t:.1f}" r="9" fill="{SLATE}"/>')
    # Three credit kernels, one per GAE lambda: slow decay, fast decay, a single step.
    rows = [[24, 22, 20, 18.5, 17, 15.5], [24, 13, 7, 4], [24]]
    for j, radii in enumerate(rows):
        y = 800 + j * 62
        for k, rr in enumerate(radii):
            circle(560 + k * 70, y, rr, KRAFT if j != 1 else MANILLA, stroke=(k == 0))
    # Advantages travel back to the actor.
    for k in range(9):
        t = k / 8
        x = 500 - 260 * t
        y = 862 - 190 * t + 120 * t * (1 - t)
        out.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="9" fill="{SLATE}"/>')
    defs = (f'<defs><filter id="{fid}" filterUnits="userSpaceOnUse" x="0" y="0" width="{W}" height="{H}">'
            f'<feTurbulence type="fractalNoise" baseFrequency="0.012" numOctaves="2" seed="9"/>'
            f'<feDisplacementMap in="SourceGraphic" scale="4.5"/></filter></defs>')
    return (f'<svg viewBox="0 0 {W} {H}" role="img" aria-label="A random actor writes tokens, a frozen critic scores '
            f'each token, and three credit kernels send advantages back to the actor" '
            f'style="width:100%;height:auto;display:block">' + defs + "".join(out) + "</svg>")


if __name__ == "__main__":
    print(cover_svg())
