"""Render the README infographic from saved results and a saved API response."""
import argparse
import json
import shutil
import subprocess
from pathlib import Path
from xml.sax.saxutils import escape

from nimble.paths import PROJECT_ROOT

# Bespoke Labs brand colors. MUTED is BLACK at about 70% on the card color.
TEAL, TAN, RED, BLUE, BLACK = '#0F484D', '#F8F3E6', '#F44029', '#B9C8DD', '#231F20'
WHITE, MINT, LIME = '#FFF9F9', '#D4E8E3', '#DAF767'
MUTED, LINE = '#5E5A57', '#E4DACA'

# These facts have no saved results file. The README states each one.
TRAINING_EXAMPLES, CATEGORIES = '2,676', 10
RECIPE = [['2,676 examples', 'LoRA rank 16', 'Learning rate 5e-5'],
          ['Batch size 8', '1 epoch', 'BF16', 'H100']]
H100_MEDIAN_MS = 106  # README latency table, Bespoke-Nimble-9B on H100

W, H = 1600, 1456
MARGIN, GAP = 48, 36
PW, PH = (W - 2 * MARGIN - GAP) // 2, 580
FONTS = """
.serif { font-family: 'GT Alpina', Georgia, 'Times New Roman', serif; }
.sans { font-family: Inter, system-ui, -apple-system, 'Segoe UI', Roboto, 'Helvetica Neue', Arial, sans-serif; }
.mono { font-family: 'MD IO', 'IBM Plex Mono', 'SF Mono', Menlo, Consolas, 'Liberation Mono', monospace; }
"""
CHROME = ['/Applications/Google Chrome.app/Contents/MacOS/Google Chrome',
          'google-chrome', 'google-chrome-stable', 'chromium', 'chromium-browser']


def attrs(**kw):
    return ' '.join(f'{k.rstrip("_").replace("_", "-")}="{v}"' for k, v in kw.items() if v is not None)


def text(x, y, s, size=21, fill=BLACK, weight=400, anchor='start', cls='sans', **kw):
    return (f'<text {attrs(x=x, y=y, font_size=size, fill=fill, font_weight=weight, text_anchor=anchor, class_=cls, **kw)}>'
            f'{escape(s)}</text>')


def spans(x, y, parts, anchor='start', cls='sans'):
    """One line of text with mixed styles: parts are (string, size, fill, weight)."""
    inner = ''.join(f'<tspan {attrs(font_size=size, fill=fill, font_weight=weight)}>{escape(s)}</tspan>'
                    for s, size, fill, weight in parts)
    return f'<text {attrs(x=x, y=y, text_anchor=anchor, class_=cls)}>{inner}</text>'


def rect(x, y, w, h, fill, rx=12, stroke=None, width=None, **kw):
    return f'<rect {attrs(x=x, y=y, width=w, height=h, rx=rx, fill=fill, stroke=stroke, stroke_width=width, **kw)}/>'


def arrow(x1, y1, x2, y2, color=MUTED):
    return f'<line {attrs(x1=x1, y1=y1, x2=x2, y2=y2, stroke=color, stroke_width=2, marker_end="url(#arrow)")}/>'


def width(s, size, weight=400):
    return len(s) * size * (0.6 if weight >= 600 else 0.55)


def chip(x, y, s, size=19, fill=WHITE, color=BLACK, weight=400, h=38, cls='sans'):
    w = round(width(s, size, weight) + 32)
    return w, [rect(x, y, w, h, fill, rx=h // 2, stroke=LINE if fill == WHITE else None, width=1.5),
               text(x + w / 2, y + h / 2 + size * .36, s, size, color, weight, 'middle', cls)]


def panel(x, y, number, title, subtitle):
    return [rect(x, y, PW, PH, WHITE, rx=22, stroke=LINE, width=1.5),
            f'<circle {attrs(cx=x + 56, cy=y + 58, r=22, fill=RED)}/>',
            text(x + 56, y + 66, str(number), 22, WHITE, 700, 'middle'),
            text(x + 94, y + 69, title, 34, TEAL, 400, cls='serif'),
            text(x + 94, y + 103, subtitle, 21, MUTED)]


def serving(x, y, example):
    """Prefill the shared prompt once, then read one answer token per question."""
    x0, out = x + 32, panel(x, y, 1, 'Serving', 'Read the prompt once, then one token per question')
    out += [rect(x0, y + 132, 350, 136, MINT), text(x0 + 20, y + 164, 'Text', 19, TEAL, 700),
            text(x0 + 20, y + 202, 'I was charged twice.'), text(x0 + 20, y + 232, 'Please refund the duplicate.'),
            rect(x0 + 370, y + 132, 300, 136, MINT), text(x0 + 390, y + 164, 'Questions', 19, TEAL, 700)]
    lanes = [('refund', 'Noul', ['false', 'true'], example['refund']),
             ('department', 'Choice', ['billing', 'technical'], example['department']),
             ('urgency', 'Score', ['Routine', 'Urgent', 'Emergency'], example['urgency'])]
    for i, (name, kind, _, _) in enumerate(lanes):
        out += [text(x0 + 390, y + 194 + 27 * i, name, 20, BLACK, 600),
                text(x0 + 650, y + 194 + 27 * i, kind, 19, MUTED, 400, 'end')]
    out += [arrow(x0 + 175, y + 268, x0 + 175, y + 292), arrow(x0 + 520, y + 268, x0 + 520, y + 292),
            rect(x0, y + 296, 670, 48, TEAL),
            text(x0 + 335, y + 327, 'Prefill the shared prompt once (KV cache)', 21, WHITE, 600, 'middle')]
    for i, (name, kind, labels, answer) in enumerate(lanes):
        lx = x0 + 228 * i
        if kind == 'Noul':
            probs = [1 - answer['noul'], answer['noul']]
        elif kind == 'Choice':
            probs = [answer['probabilities'][k] for k in labels]
        else:
            probs = [answer['probabilities'][str(k)] for k in range(len(labels))]
        best = probs.index(max(probs))
        out += [arrow(lx + 107, y + 344, lx + 107, y + 368),
                rect(lx, y + 372, 214, 150, WHITE, stroke=LINE, width=1.5),
                text(lx + 16, y + 402, name, 20, BLACK, 600)]
        for j, (label, p) in enumerate(zip(labels, probs)):
            row = y + 416 + 32 * j
            if 194 * p >= 6:
                out.append(rect(lx + 10, row, 194 * p, 26, RED if j == best else BLUE, rx=6,
                                fill_opacity=.22 if j == best else .45))
            out += [text(lx + 18, row + 19, f'{chr(65 + j)}', 16, MUTED, 600, cls='mono'),
                    text(lx + 40, row + 19, label, 18, BLACK, 600 if j == best else 400),
                    text(lx + 198, row + 19, f'{100 * p:.1f}%', 18, BLACK, 600 if j == best else 400, 'end')]
    out.append(text(x0, y + 556, f'No text to parse. Median {H100_MEDIAN_MS} ms per question on one H100.', 19, MUTED))
    return out


def curation(x, y):
    """A contrastive pair differs in one fact, and the label flips with it."""
    x0, out = x + 32, panel(x, y, 2, 'Data curation', 'Change one fact, and the correct answer flips')
    out += [rect(x0, y + 132, 670, 56, MINT), text(x0 + 20, y + 167, 'Rule', 19, TEAL, 700),
            text(x0 + 80, y + 167, 'Only Mira may authorize refunds for account 42.', 20)]
    for i, (name, answer) in enumerate([('Mira', 'true'), ('Noah', 'false')]):
        cx = x0 + 347 * i
        pill = round(width(name, 20, 700) + 26)
        out += [rect(cx, y + 204, 323, 152, WHITE, stroke=LINE, width=1.5),
                text(cx + 20, y + 234, f'Example {"AB"[i]}', 19, MUTED, 600),
                text(cx + 20, y + 266, 'The sole authorization', 20), text(cx + 20, y + 292, 'was signed by', 20),
                text(cx + 303, y + 292, 'Authorized?', 19, MUTED, 400, 'end'),
                rect(cx + 20, y + 304, pill, 34, LIME, rx=8), text(cx + 33, y + 328, name, 20, BLACK, 700),
                rect(cx + 219, y + 304, 84, 34, TEAL if answer == 'true' else RED, rx=17),
                text(cx + 261, y + 327, answer, 18, WHITE, 700, 'middle', 'mono')]
    steps = ['Extract a fact and flip one fact',
             'Check that removing a fact makes the answer unknown',
             'Keep pairs that pass every check']
    for i, step in enumerate(steps):
        row = y + 406 + 32 * i
        out += [f'<circle {attrs(cx=x0 + 12, cy=row - 7, r=12, fill=TEAL)}/>',
                text(x0 + 12, row - 1, str(i + 1), 15, WHITE, 700, 'middle'), text(x0 + 36, row, step, 20)]
    out += [f'<line {attrs(x1=x0, y1=y + 510, x2=x0 + 670, y2=y + 510, stroke=LINE, stroke_width=1.5)}/>',
            spans(x0, y + 550, [(TRAINING_EXAMPLES, 30, BLACK, 700), (' training examples across ', 20, MUTED, 400),
                                (str(CATEGORIES), 30, BLACK, 700), (' categories', 20, MUTED, 400)])]
    return out


def training(x, y):
    """LoRA on Qwen3.5-9B with cross-entropy over the allowed answer codes."""
    x0, out = x + 32, panel(x, y, 3, 'Training', 'Fine-tune Qwen3.5-9B on the answer tokens only')
    out += [rect(x0, y + 132, 214, 190, MINT), text(x0 + 20, y + 162, 'Prompt', 19, TEAL, 700)]
    for i, (indent, line) in enumerate([(0, 'context …'), (0, 'schema'), (20, 'A false'), (20, 'B true'),
                                        (0, 'field authorized')]):
        out.append(text(x0 + 20 + indent, y + 194 + 26 * i, line, 17, BLACK, 400, cls='mono'))
    out += [arrow(x0 + 220, y + 227, x0 + 244, y + 227),
            rect(x0 + 250, y + 132, 190, 190, BLUE, rx=14),
            text(x0 + 345, y + 184, 'Qwen3.5-9B', 21, BLACK, 700, 'middle'),
            text(x0 + 345, y + 212, 'base weights frozen', 18, BLACK, 400, 'middle'),
            rect(x0 + 263, y + 236, 164, 38, RED, rx=19),
            text(x0 + 345, y + 261, 'LoRA rank 16', 19, WHITE, 700, 'middle'),
            text(x0 + 345, y + 300, 'trained', 18, BLACK, 400, 'middle'),
            arrow(x0 + 446, y + 227, x0 + 476, y + 227),
            rect(x0 + 482, y + 132, 188, 190, WHITE, stroke=LINE, width=1.5),
            text(x0 + 576, y + 162, 'Answer logits', 19, TEAL, 700, 'middle'),
            rect(x0 + 520, y + 244, 44, 46, BLUE, rx=4), rect(x0 + 590, y + 196, 44, 94, RED, rx=4),
            f'<line {attrs(x1=x0 + 504, y1=y + 290, x2=x0 + 650, y2=y + 290, stroke=MUTED, stroke_width=1.5)}/>',
            text(x0 + 542, y + 312, 'A', 17, BLACK, 600, 'middle', 'mono'),
            text(x0 + 612, y + 312, 'B', 17, BLACK, 600, 'middle', 'mono'),
            text(x0 + 612, y + 186, 'label', 17, BLACK, 600, 'middle'),
            text(x0, y + 360, 'The loss is cross-entropy over the allowed answer codes only.', 20),
            text(x0, y + 390, 'Labels come from our checked data. We did not distill from Jev.', 20),
            text(x0, y + 434, 'Recipe', 19, TEAL, 700)]
    for r, row in enumerate(RECIPE):
        cx = x0
        for label in row:
            w, parts = chip(cx, y + 448 + 50 * r, label)
            out += parts
            cx += w + 12
    return out


def results(x, y, models):
    """Agreement with the reference labels on the shared 324-example holdout."""
    x0 = x + 32
    out = panel(x, y, 4, 'Results', 'Agreement with reference labels on 324 held-out examples')
    rows = sorted(models, key=lambda r: r['correct'], reverse=True)
    out.append(f'<line {attrs(x1=x0 + 228, y1=y + 132, x2=x0 + 228, y2=y + 412, stroke=LINE, stroke_width=1.5)}/>')
    for i, r in enumerate(rows):
        top, value = y + 140 + 38 * i, 100 * r['correct'] / r['count']
        nimble = r['name'] == 'Bespoke-Nimble-9B'
        fill = RED if nimble else BLUE
        length, weight = 3.6 * value, 700 if nimble else 400
        out += [text(x0 + 214, top + 17, r['name'], 20, BLACK, weight, 'end'),
                f'<path d="M{x0 + 228},{top} H{x0 + 224 + length} A4,4 0 0 1 {x0 + 228 + length},{top + 4} '
                f'V{top + 18} A4,4 0 0 1 {x0 + 224 + length},{top + 22} H{x0 + 228} Z" fill="{fill}"/>',
                text(x0 + 238 + length, top + 17, f'{value:.1f}%', 19, BLACK, weight)]
    score = {r['name']: 100 * r['correct'] / r['count'] for r in rows}
    nimble = score['Bespoke-Nimble-9B']
    callouts = [(f'+{nimble - score["Qwen3.5-9B"]:.0f}', 'over its base model,', 'Qwen3.5-9B'),
                (f'+{nimble - score["Qwen3.8-27B"]:.0f}', 'over Qwen3.8-27B,', 'a model 3× larger')]
    for i, (big, line1, line2) in enumerate(callouts):
        cx = x0 + 229 * i
        out += [rect(cx, y + 440, 40, 4, RED, rx=2),
                spans(cx, y + 492, [(big, 40, BLACK, 700), (' points', 21, BLACK, 600)]),
                text(cx, y + 522, line1, 19, MUTED), text(cx, y + 548, line2, 19, MUTED)]
    return out


def render(models, example):
    body = [rect(0, 0, W, H, TAN, rx=0),
            text(MARGIN, 96, 'Introducing Bespoke Nimble', 56, TEAL, 400, cls='serif'),
            text(MARGIN, 140, 'An open data, open model, open recipe for an open Jev', 24, MUTED),
            text(W - MARGIN, 96, 'Bespoke Labs', 24, RED, 700, 'end')]
    row2 = 180 + PH + GAP
    body += serving(MARGIN, 180, example) + curation(MARGIN + PW + GAP, 180)
    body += training(MARGIN, row2) + results(MARGIN + PW + GAP, row2, models)
    body += [text(MARGIN, H - 36, 'huggingface.co/bespokelabs/Bespoke-Nimble-9B  ·  github.com/bespokelabsai/nimble', 19, MUTED)]
    return (f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" viewBox="0 0 {W} {H}" '
            f'role="img" aria-labelledby="title">\n<title id="title">Introducing Bespoke Nimble</title>\n'
            f'<style>{FONTS}</style>\n<defs><marker id="arrow" viewBox="0 0 10 10" refX="8" refY="5" '
            f'markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path d="M0,0 L10,5 L0,10 z" '
            f'fill="{MUTED}"/></marker></defs>\n' + '\n'.join(body) + '\n</svg>\n')


def find_chrome(explicit):
    for candidate in [explicit] if explicit else CHROME:
        path = shutil.which(candidate) or (candidate if Path(candidate).exists() else None)
        if path:
            return path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--results', type=Path, default=PROJECT_ROOT / 'assets/evidence-324-results.json')
    parser.add_argument('--example', type=Path, default=PROJECT_ROOT / 'docs/assets/modal-public-smoke.json',
                        help='Saved public API response used in the serving panel.')
    parser.add_argument('--output-dir', type=Path, default=PROJECT_ROOT / 'assets/diagrams')
    parser.add_argument('--chrome', help='Chrome or Chromium binary for the PNG. Found automatically if omitted.')
    args = parser.parse_args()
    models = json.loads(args.results.read_text())['models']
    assert all(r['count'] == 324 and 0 <= r['correct'] <= r['count'] for r in models)
    example = json.loads(args.example.read_text())['response']['answers']
    args.output_dir.mkdir(parents=True, exist_ok=True)
    svg = args.output_dir / 'nimble-infographic.svg'
    svg.write_text(render(models, example))
    chrome = find_chrome(args.chrome)
    if not chrome:
        print(f'Wrote {svg}. No Chrome found, so the PNG was not updated.')
        return
    png = args.output_dir / 'nimble-infographic.png'
    subprocess.run([chrome, '--headless=new', '--disable-gpu', '--hide-scrollbars', f'--window-size={W},{H}',
                    '--force-device-scale-factor=2', f'--screenshot={png.resolve()}', svg.resolve().as_uri()],
                   check=True, capture_output=True)
    print(f'Wrote {svg} and {png}.')


if __name__ == '__main__':
    main()
