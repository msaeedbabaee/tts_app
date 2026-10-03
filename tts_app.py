"""
Unlimited Text-to-Speech (Streamlit + edge-tts)

Install & run:
    pip install streamlit edge-tts pypdf
    streamlit run tts_app.py

Features
  * Any language / gender / voice that edge-tts offers, speed / pitch / volume
  * Bilingual tab: English lines read by an English voice, Persian lines by a Persian voice, plus a
    study drill (English -> pause -> Persian -> English again, slower) for shadowing practice
  * PDF tab: upload a PDF, choose pages (e.g. 1-5, 8, 10-12), get one MP3 for the range / per page / every N pages
  * Voice gallery: listen to every voice of a language side by side, then pick one with a click
  * Single text tab: paste any amount of text (it is split into safe chunks automatically)
  * Batch tab: upload many .txt / .md files or a whole .zip (e.g. the `podcast/` folder),
    or point to a local folder. One MP3 per chapter or one MP3 per file. Everything is
    saved to a local folder and offered as one ZIP. Runs can be resumed (finished files are skipped).
  * Every tab can export, per section (chapter / page part) and/or for the whole topic, any mix of:
      MP3 audio, Markdown (.md) and PDF (Persian + English, right-to-left aware). Tick what you want.
      Files with the same name sit side by side: 001_Chapter_1.mp3 / .md / .pdf
  * The app itself has no character, file-count or duration limit.
    (Only the online Microsoft service behind edge-tts can throttle you; the app retries automatically.)
"""
import asyncio
import hashlib
import io
import re
import tempfile
import urllib.request
import zipfile
from collections import Counter
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path

import edge_tts
import streamlit as st

MAX_CHARS = 2500  # size of one request to the service; longer text is split automatically

FALLBACK_VOICES = [
    ("en-US-AriaNeural", "Female", "en-US", "English", "United States"),
    ("en-US-GuyNeural", "Male", "en-US", "English", "United States"),
    ("en-US-JennyNeural", "Female", "en-US", "English", "United States"),
    ("en-GB-SoniaNeural", "Female", "en-GB", "English", "United Kingdom"),
    ("en-GB-RyanNeural", "Male", "en-GB", "English", "United Kingdom"),
    ("fa-IR-DilaraNeural", "Female", "fa-IR", "Persian", "Iran"),
    ("fa-IR-FaridNeural", "Male", "fa-IR", "Persian", "Iran"),
]


# --------------------------------------------------------------------------- voices
@st.cache_data(ttl=86400, show_spinner="Loading voices…")
def load_voices():
    rows = []
    try:
        for v in asyncio.run(edge_tts.list_voices()):
            m = re.search(r"-\s*([^()]+?)\s*\((.+)\)\s*$", v.get("FriendlyName", ""))
            lang, region = (m.group(1), m.group(2)) if m else (v["Locale"], "")
            rows.append(dict(short=v["ShortName"], gender=v["Gender"], locale=v["Locale"], lang=lang, region=region))
    except Exception:
        rows = [dict(short=s, gender=g, locale=l, lang=la, region=r) for s, g, l, la, r in FALLBACK_VOICES]
        rows[0]["offline"] = True
    for r in rows:
        r["name"] = r["short"].split("-")[-1].replace("Neural", "")
        r["label"] = " - ".join(x for x in (r["lang"], r["region"], r["name"]) if x)
    return sorted(rows, key=lambda r: (r["lang"], r["region"], r["name"]))


# --------------------------------------------------------------------------- synthesis
def split_text(text, limit=MAX_CHARS):
    """Split long text on paragraph / sentence boundaries into chunks <= limit characters."""
    text = text.strip()
    if len(text) <= limit:
        return [text] if text else []
    pieces = []
    for para in re.split(r"\n\s*\n", text):
        para = para.strip()
        if not para:
            continue
        if len(para) <= limit:
            pieces.append(para)
        else:
            pieces.extend(s for s in re.split(r"(?<=[.!?…؟。])\s+", para) if s.strip())
    chunks, cur = [], ""
    for p in pieces:
        while len(p) > limit:  # a single monstrous sentence
            cut = p.rfind(" ", 0, limit)
            cut = cut if cut > 0 else limit
            if cur:
                chunks.append(cur); cur = ""
            chunks.append(p[:cut]); p = p[cut:].strip()
        if len(cur) + len(p) + 2 <= limit:
            cur = f"{cur}\n\n{p}" if cur else p
        else:
            chunks.append(cur); cur = p
    if cur:
        chunks.append(cur)
    return [c for c in chunks if c.strip()]


async def synth(text, cfg, retries=5):
    out = bytearray()
    for chunk in split_text(text):
        for attempt in range(retries):
            try:
                comm = edge_tts.Communicate(chunk, cfg["voice"], rate=cfg["rate"], pitch=cfg["pitch"], volume=cfg["volume"])
                data = bytearray()
                async for msg in comm.stream():
                    if msg["type"] == "audio":
                        data += msg["data"]
                if not data:
                    raise RuntimeError("no audio received")
                out += data
                break
            except Exception:
                if attempt == retries - 1:
                    raise
                await asyncio.sleep(1.5 * (attempt + 1))
    return bytes(out)


# --------------------------------------------------------------------------- parsing sources
def slug(s, n=60):
    return re.sub(r"[^\w]+", "_", s).strip("_")[:n] or "section"


def sections_from_txt(text):
    """[(title, [lines])] — splits at lines like 'Chapter 1.2 — …' / 'Topic 3 — …'."""
    secs, title, lines, seen = [], None, [], False
    for raw in text.splitlines():
        line = raw.strip()
        if re.match(r"^(Chapter|Topic)\s+\d", line):
            if lines:
                secs.append((title, lines))
            title, lines, seen = line.replace(" — ", ". "), [], True
        elif re.match(r"^(PART|QUICK-START)\b", line):
            continue
        elif line:
            lines.append(line)
    if lines:
        secs.append((title, lines))
    return secs


def sections_from_md(text, only_fences=True):
    secs, title, infence, buf = [], None, False, []
    for raw in text.splitlines():
        line = raw.rstrip()
        if line.strip().startswith("```"):
            if infence and buf:
                secs.append((title, buf)); buf = []
            infence = not infence
            continue
        if infence:
            if line.strip():
                buf.append(line.strip())
            continue
        m = re.match(r"^##\s+(\d+(?:\.\d+)*)\b", line)
        if m:
            title = f"Chapter {m.group(1)}"
        m = re.match(r"^#\s+.*?(\d+)\.\s+([A-Za-z].+)$", line)
        if m:
            title = f"Topic {m.group(1)}. {m.group(2)}"
    if secs or only_fences:
        return secs
    plain = [re.sub(r"[#>*_`|]", " ", l).strip() for l in text.splitlines()]
    return [(None, [l for l in plain if l])]


def read_sources(uploaded, folder):
    """-> list of (name, text)"""
    items = []
    for f in uploaded or []:
        data = f.getvalue()
        if f.name.lower().endswith(".zip"):
            with zipfile.ZipFile(io.BytesIO(data)) as z:
                for n in sorted(z.namelist()):
                    if n.lower().endswith((".txt", ".md")) and not n.endswith("/"):
                        items.append((n, z.read(n).decode("utf-8", "ignore")))
        else:
            items.append((f.name, data.decode("utf-8", "ignore")))
    if folder.strip():
        p = Path(folder.strip()).expanduser()
        if p.is_dir():
            for fp in sorted(list(p.rglob("*.txt")) + list(p.rglob("*.md"))):
                items.append((str(fp.relative_to(p)), fp.read_text(encoding="utf-8", errors="ignore")))
        else:
            st.warning(f"Folder not found: {p}")
    return items


def build_text(title, lines, repeat, announce):
    parts = []
    if announce and title:
        parts.append(title.rstrip(".") + ".")
    for s in lines:
        s = s.strip()
        if not s:
            continue
        if not re.search(r"[.?!…:;]$", s):
            s += "."
        parts.extend([s] * repeat)
    return "\n\n".join(parts)


# --------------------------------------------------------------------------- outputs model
# Every source (txt/md file, PDF range, pasted text) becomes a Group (= the whole topic / file)
# made of Sections (= chapters / page parts). A Section carries BOTH the text that is spoken and
# the document blocks that mirror it, so MP3 / Markdown / PDF are always built from the same data.
#   blocks: [("h", text) sub-heading | ("p", text) paragraph | ("qa", english, persian) pair]
@dataclass
class Section:
    name: str                                    # output path inside the folder, without extension
    title: str
    blocks: list = field(default_factory=list)
    texts: list = field(default_factory=list)    # plain texts for TTS (one voice)
    render: object = None                        # async callable(ctx) -> mp3 bytes (bilingual tab)


@dataclass
class Group:
    name: str
    title: str
    sections: list = field(default_factory=list)


@dataclass
class Task:
    path: Path
    run: object                                  # async callable(ctx) -> bytes
    kind: str = "mp3"
    phase: int = 0                               # phase 1 runs after phase 0 (merged MP3 re-uses chapter MP3s)
    skip_ok: bool = False                        # may be skipped when the file already exists


FORMAT_LABELS = {"mp3": "🔊 Audio (MP3)", "md": "📝 Markdown (.md)", "pdf": "📄 PDF"}
GRAN = {"section": "Per section", "file": "Whole topic", "both": "Both"}


def make_groups(items, repeat, announce, only_fences):
    """Batch tab: .txt / .md sources -> [Group]. One section per chapter."""
    groups = []
    for name, text in items:
        p = Path(name)
        stem = "__".join(p.with_suffix("").parts)
        secs = sections_from_md(text, only_fences) if name.lower().endswith(".md") else sections_from_txt(text)
        secs = [(t, l) for t, l in secs if l]
        if not secs:
            continue
        sections = [Section(name=f"{stem}/{i:03d}_{slug(t or 'section')}", title=t or f"Section {i}",
                            blocks=[("p", ln) for ln in lines], texts=[build_text(t, lines, repeat, announce)])
                    for i, (t, lines) in enumerate(secs, 1)]
        groups.append(Group(stem, p.stem, sections))
    return groups


# ---- building the individual files
def _md_line(t):
    t = t.strip().replace("\n", "  \n")
    return "\\" + t if t[:1] in "#>" else t


def render_md(title, sections, nested):
    """nested=True -> the whole topic ('# topic' / '## section'); False -> a single section file."""
    out = []

    def add(blocks, sub):
        for b in blocks:
            if b[0] == "h":
                out.extend([f"{'#' * sub} {b[1]}", ""])
            elif b[0] == "qa":
                out.extend([_md_line(b[1]), "", "> " + _md_line(b[2]).replace("  \n", " "), ""])
            else:
                out.extend([_md_line(b[1]), ""])

    if nested:
        out.extend([f"# {title}", ""])
        for s in sections:
            out.extend([f"## {s.title}", ""])
            add(s.blocks, 3)
    else:
        out.extend([f"# {sections[0].title}", ""])
        add(sections[0].blocks, 2)
    return ("\n".join(out).strip() + "\n").encode("utf-8")


FONT_DIR = Path(__file__).resolve().parent / "fonts"
FONT_URL = "https://raw.githubusercontent.com/rastikerdar/vazirmatn/master/fonts/ttf/{}"
SYSTEM_FONTS = [  # (regular, bold) — used only if the bundled Vazirmatn is missing and cannot be downloaded
    (r"C:\Windows\Fonts\tahoma.ttf", r"C:\Windows\Fonts\tahomabd.ttf"),
    (r"C:\Windows\Fonts\arial.ttf", r"C:\Windows\Fonts\arialbd.ttf"),
    ("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"),
    ("/usr/share/fonts/truetype/noto/NotoNaskhArabic-Regular.ttf", "/usr/share/fonts/truetype/noto/NotoNaskhArabic-Bold.ttf"),
    ("/System/Library/Fonts/Supplemental/Arial.ttf", "/System/Library/Fonts/Supplemental/Arial Bold.ttf"),
]


def resolve_pdf_font(custom=None):
    """-> (regular_path, bold_path) or (None, None). Order: uploaded font, fonts/Vazirmatn, download, system."""
    if custom:
        p = Path(tempfile.gettempdir()) / f"tts_app_font_{hashlib.md5(custom).hexdigest()[:10]}.ttf"
        if not p.exists():
            p.write_bytes(custom)
        return p, p
    reg, bold = FONT_DIR / "Vazirmatn-Regular.ttf", FONT_DIR / "Vazirmatn-Bold.ttf"
    if not (reg.exists() and bold.exists()):
        try:
            FONT_DIR.mkdir(exist_ok=True)
            for fp in (reg, bold):
                if not fp.exists():
                    with urllib.request.urlopen(FONT_URL.format(fp.name), timeout=20) as r:
                        fp.write_bytes(r.read())
        except Exception:  # noqa: BLE001
            pass
    if reg.exists():
        return reg, (bold if bold.exists() else reg)
    for r, b in SYSTEM_FONTS:
        if Path(r).exists():
            return Path(r), Path(b if Path(b).exists() else r)
    return None, None


@lru_cache(maxsize=8)
def _cmap(path):
    from fontTools.ttLib import TTFont
    return set(TTFont(path, lazy=True).getBestCmap())


def missing_glyphs(text, font_path):
    """Characters of `text` that the PDF font cannot draw (they would show as empty boxes)."""
    cm = _cmap(str(font_path))
    return sorted({c for c in text if ord(c) > 32 and ord(c) not in cm and c not in "\u200c\u200d\u200e\u200f\ufeff"})


def is_rtl(text):
    return FA_CH.search(text) is not None and len(FA_CH.findall(text)) >= len(LAT_CH.findall(text))


def render_pdf(title, sections, nested, font):
    """Bilingual-safe PDF: Persian is shaped + right-to-left, English left-to-right, line by line."""
    from fpdf import FPDF

    reg, bold = font

    class Doc(FPDF):
        def footer(self):
            ts = self.text_shaping
            self.set_y(-13)
            self.set_font("TTS", "", 9)
            self.set_text_color(130, 130, 130)
            self.set_text_shaping(True, direction="ltr")
            self.cell(0, 8, str(self.page_no()), align="C")
            self.text_shaping = ts

    pdf = Doc(format="A4")
    pdf.set_margins(18, 18, 18)
    pdf.set_auto_page_break(True, 20)
    pdf.add_font("TTS", "", str(reg))
    pdf.add_font("TTS", "B", str(bold))
    pdf.set_title(title)
    pdf.set_creator("Unlimited Text to Speech")
    pdf.add_page()

    def para(text, size=12, style="", color=(0, 0, 0), gap=2.5, keep=0):
        if keep and pdf.will_page_break(keep):
            pdf.add_page()
        for ln in text.split("\n"):
            ln = ln.strip()
            if not ln:
                continue
            rtl = is_rtl(ln)
            pdf.set_text_shaping(True, direction="rtl" if rtl else "ltr")
            pdf.set_font("TTS", style, size)
            pdf.set_text_color(*color)
            pdf.multi_cell(0, size * 0.62, ln, align="R" if rtl else "L", new_x="LMARGIN", new_y="NEXT")
        pdf.ln(gap)

    def add(blocks, sizes):
        for b in blocks:
            if b[0] == "h":
                para(b[1], sizes, "B", (90, 90, 90), 1.5, keep=25)
            elif b[0] == "qa":
                para(b[1], 12.5, "", (0, 0, 0), 0.5, keep=22)
                para(b[2], 12, "", (70, 70, 70), 3.5)
            else:
                para(b[1])

    if nested:
        para(title, 20, "B", gap=5)
        for s in sections:
            para(s.title, 15, "B", (30, 60, 130), 3, keep=35)
            add(s.blocks, 12.5)
    else:
        para(sections[0].title, 20, "B", gap=5)
        add(sections[0].blocks, 14)
    return bytes(pdf.output())


async def section_audio(ctx, secs, cfg, outdir, reuse):
    """MP3 bytes for sections (concatenated). reuse=True picks up finished per-section MP3s from disk."""
    async def one(s):
        p = outdir / f"{s.name}.mp3"
        if reuse and p.exists() and p.stat().st_size > 0:
            return p.read_bytes()
        if s.render:
            return await s.render(ctx)
        async with ctx["sem"]:
            return b"".join([await synth(t, cfg) for t in s.texts])
    return b"".join(await asyncio.gather(*[one(s) for s in secs]))


def plan_tasks(groups, outdir, fmts, cfg, font=None, reuse_existing=False):
    """fmts: {'mp3'|'md'|'pdf': 'section'|'file'|'both'} -> [Task]. Files sit side by side with the same base name."""
    tasks = []

    def mk(ext, path, secs, title, nested, phase=0, reuse=False):
        if ext == "mp3":
            async def run(ctx):
                return await section_audio(ctx, secs, cfg, outdir, reuse)
        elif ext == "md":
            async def run(ctx):
                return render_md(title, secs, nested)
        else:
            async def run(ctx):
                return render_pdf(title, secs, nested, font)
        return Task(path, run, ext, phase, skip_ok=(ext == "mp3"))

    for g in groups:
        for ext in ("mp3", "md", "pdf"):
            gran = fmts.get(ext)
            if not gran or not g.sections:
                continue
            want_sec, want_file = gran in ("section", "both"), gran in ("file", "both")
            if gran == "both" and len(g.sections) == 1:
                want_sec = False                       # a single section: both files would be identical
            if want_sec:
                for s in g.sections:
                    tasks.append(mk(ext, outdir / f"{s.name}.{ext}", [s], g.title, False))
            if want_file:
                reuse = ext == "mp3" and (want_sec or reuse_existing)
                tasks.append(mk(ext, outdir / f"{g.name}.{ext}", g.sections, g.title, True,
                                phase=1 if ext == "mp3" else 0, reuse=reuse))
    return tasks


def describe_tasks(tasks):
    n = Counter(t.kind for t in tasks)
    return " · ".join(f"{n[k]} {lab}" for k, lab in (("mp3", "MP3"), ("md", "Markdown"), ("pdf", "PDF")) if n[k]) or "nothing"


async def run_tasks(tasks, parallel, skip_existing, on_progress):
    sem = asyncio.Semaphore(parallel)           # how many files are built at once
    piece_sem = asyncio.Semaphore(parallel)     # how many requests to the speech service are in flight
    results = {"done": 0, "skipped": 0, "failed": []}

    async def work(t):
        async with sem:
            if t.skip_ok and skip_existing and t.path.exists() and t.path.stat().st_size > 0:
                return t, "skipped", None
            try:
                data = await t.run({"sem": piece_sem, "cache": {}})
                t.path.parent.mkdir(parents=True, exist_ok=True)
                t.path.write_bytes(data)
                return t, "done", None
            except Exception as e:  # noqa: BLE001
                return t, "failed", str(e)

    finished = 0
    for phase in sorted({t.phase for t in tasks}):
        pending = [asyncio.create_task(work(t)) for t in tasks if t.phase == phase]
        for fut in asyncio.as_completed(pending):
            t, status, err = await fut
            finished += 1
            if status == "failed":
                results["failed"].append((str(t.path), err))
            else:
                results[status] += 1
            on_progress(finished, len(tasks), t.path.name, status)
    return results


def zip_folder(files):
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_STORED) as z:  # mp3 is already compressed
        for f, root in files:
            z.write(f, str(f.relative_to(root)))
    return buf.getvalue()


# --------------------------------------------------------------------------- PDF
def parse_page_ranges(spec, total):
    """'1-5, 8, 10-' -> [1,2,3,4,5,8,10,...]; '' or 'all' -> every page. 1-based."""
    spec = spec.translate(str.maketrans("۰۱۲۳۴۵۶۷۸۹٠١٢٣٤٥٦٧٨٩", "01234567890123456789")).strip().lower()
    if spec in ("", "all", "*"):
        return list(range(1, total + 1))
    pages = []
    for part in re.split(r"[,\s;،]+", spec):
        if not part:
            continue
        m = re.fullmatch(r"(\d*)[-–—:](\d*)", part)
        if m and (m.group(1) or m.group(2)):
            a, b = int(m.group(1) or 1), int(m.group(2) or total)
            pages.extend(range(min(a, b), max(a, b) + 1))
        elif part.isdigit():
            pages.append(int(part))
        else:
            raise ValueError(f"Cannot understand '{part}'. Use something like 1-5, 8, 10-12")
    bad = sorted({p for p in pages if p < 1 or p > total})
    if bad:
        raise ValueError(f"Page(s) out of range (this PDF has {total} pages): {bad[:8]}")
    seen, out = set(), []
    for p in pages:
        if p not in seen:
            seen.add(p); out.append(p)
    return out


@st.cache_data(show_spinner=False)
def pdf_page_count(data):
    from pypdf import PdfReader
    return len(PdfReader(io.BytesIO(data)).pages)


@st.cache_data(show_spinner="Reading PDF…")
def extract_pdf_pages(data, pages):
    from pypdf import PdfReader
    reader = PdfReader(io.BytesIO(data))
    return [(p, reader.pages[p - 1].extract_text() or "") for p in pages]


FOREIGN_RE = re.compile(r"[\u0400-\u04FF\u0590-\u08FF\uFB1D-\uFDFF\uFE70-\uFEFF\u3040-\u30FF\u4E00-\u9FFF\uAC00-\uD7AF]")
PAGENUM_RE = re.compile(r"^\s*(page\s*)?\d{1,4}(\s*(of|/)\s*\d{1,4})?\s*$", re.I)


def clean_pdf_pages(pages, rm_headers=True, rm_pagenums=True, skip_foreign=False):
    """pages: [(no, raw_text)] -> [(no, clean_text)] with paragraphs re-flowed for speech."""
    norm = lambda l: re.sub(r"\d+", "#", l.strip().lower())  # noqa: E731
    page_lines = [(n, [l.rstrip() for l in t.splitlines()]) for n, t in pages]
    repeated = set()
    if rm_headers and len(page_lines) >= 3:
        from collections import Counter
        c = Counter()
        for _, ls in page_lines:
            ne = [l for l in ls if l.strip()]
            for l in {norm(x) for x in ne[:2] + ne[-2:]}:
                c[l] += 1
        repeated = {l for l, k in c.items() if k >= max(2, 0.5 * len(page_lines)) and len(l) > 1}
    out = []
    for n, ls in page_lines:
        keep = []
        for l in ls:
            if l.strip() and (norm(l) in repeated):
                continue
            if rm_pagenums and PAGENUM_RE.match(l):
                continue
            if skip_foreign and FOREIGN_RE.search(l):
                continue
            l = re.sub(r"^\s*[•▪●◦·]\s*", "", l)
            keep.append(l.strip())
        paras, cur = [], ""
        for l in keep:
            if not l:
                if cur: paras.append(cur); cur = ""
                continue
            if cur.endswith("-") and l[:1].islower():
                cur = cur[:-1] + l                      # de-hyphenate
            elif cur and not re.search(r"[.!?:;…\"”)]$", cur):
                cur += " " + l                          # wrapped line
            else:
                if cur: paras.append(cur)
                cur = l
        if cur:
            paras.append(cur)
        out.append((n, "\n\n".join(re.sub(r"\s+", " ", p).strip() for p in paras)))
    return out


PM_WHOLE, PM_PAGE, PM_EVERY = "Whole range as one part", "One part per page", "One part every N pages"


def pdf_groups(pages, stem, title, mode, every):
    """Cleaned PDF pages -> [Group]. Parts (= sections) are single pages, N pages, or the whole range."""
    pages = [(n, t) for n, t in pages if t.strip()]
    if not pages:
        return []
    step = len(pages) if mode == PM_WHOLE else 1 if mode == PM_PAGE else max(1, int(every))

    def label(a, b):
        return (f"page_{a:03d}", f"Page {a}") if a == b else (f"pages_{a:03d}-{b:03d}", f"Pages {a}\u2013{b}")

    sections = []
    for i in range(0, len(pages), step):
        grp = pages[i:i + step]
        key, ttl = label(grp[0][0], grp[-1][0])
        blocks = []
        for n, t in grp:
            if len(grp) > 1:
                blocks.append(("h", f"Page {n}"))
            blocks += [("p", p) for p in re.split(r"\n\s*\n", t) if p.strip()]
        sections.append(Section(f"{stem}_{key}", ttl, blocks, ["\n\n".join(t for _, t in grp)]))
    key, _ = label(pages[0][0], pages[-1][0])
    return [Group(f"{stem}_{key}", title, sections)]


# --------------------------------------------------------------------------- BILINGUAL
EMOJI_RE = re.compile("[\U0001F000-\U0001FAFF\u2600-\u27BF\u2B00-\u2BFF\uFE0F\u200D\u20E3]")
FA_CH = re.compile(r"[\u0600-\u06FF\u0750-\u077F\uFB50-\uFDFF\uFE70-\uFEFF]")
LAT_CH = re.compile(r"[A-Za-z]")
SKIP_TOKEN = re.compile(r"[—–→↔=|•·]+")


def split_runs(line):
    """One text line -> [(lang, text)], lang in {'en','fa'}. Words decide the language;
    a single English word inside a Persian sentence stays Persian (voices switch only for real phrases)."""
    toks = []
    for w in line.split():
        if SKIP_TOKEN.fullmatch(w):
            continue
        toks.append(["fa" if FA_CH.search(w) else "en" if LAT_CH.search(w) else "n", w])
    kinds = {k for k, _ in toks}
    if "fa" not in kinds:
        return [("en", " ".join(w for _, w in toks))] if "en" in kinds else []
    if "en" not in kinds:
        return [("fa", " ".join(w for _, w in toks))]
    last = None
    for t in toks:                       # neutrals (numbers, punctuation) follow the previous word
        if t[0] == "n":
            t[0] = last
        else:
            last = t[0]
    first = next(k for k, _ in toks if k)
    for t in toks:
        t[0] = t[0] or first
    runs = []
    for k, w in toks:
        if runs and runs[-1][0] == k:
            runs[-1][1].append(w)
        else:
            runs.append([k, [w]])
    for r in runs:                       # tiny English bits inside Persian -> Persian
        if r[0] == "en" and sum(1 for w in r[1] if LAT_CH.search(w)) < 2:
            r[0] = "fa"
    merged = []
    for k, ws in runs:
        if merged and merged[-1][0] == k:
            merged[-1][1].extend(ws)
        else:
            merged.append([k, list(ws)])
    return [(k, " ".join(ws)) for k, ws in merged]


def line_items(text):
    t = re.sub(r"[*_`~]+", "", text)
    t = EMOJI_RE.sub("", t).replace("\u2019", "'").strip()
    return split_runs(t) if t else []


def bilingual_sections(name, text, skip_fences=True):
    """[(title, [(lang, text), ...])]. Markdown files are split at '## 1.2 …' chapter headings."""
    is_md = name.lower().endswith(".md")
    chaptered = is_md and bool(re.search(r"(?m)^##\s+\d", text))
    started = not chaptered
    secs, title, items, infence = [], None, [], False

    def flush():
        if items:
            secs.append((title, list(items)))
            items.clear()

    for raw in text.splitlines():
        line = raw.rstrip()
        if line.strip().startswith("```"):
            infence = not infence
            continue
        if infence and skip_fences:
            continue
        m = re.match(r"^##\s+(\d+(?:\.\d+)*)\b", line) if is_md else None
        if m:
            flush(); title = f"Chapter {m.group(1)}"; started = True
            continue
        if not started or "English for Podcast" in line or re.match(r"^\s*(-{3,}|\*{3,}|_{3,})\s*$", line):
            continue
        if line.strip().startswith("|"):
            cells = [c.strip() for c in line.strip().strip("|").split("|")]
            if all(re.fullmatch(r":?-{2,}:?", c) for c in cells if c) or [c.lower() for c in cells] == ["english", "فارسی"]:
                continue
            for c in cells:
                items.extend(line_items(c))
            continue
        line = re.sub(r"^\s*(#{1,6}|>+|[-*+]|\d+[.)])\s+", "", line)
        items.extend(line_items(line))
    flush()
    return secs


def make_cards(items, max_fa=160):
    """Attach a Persian line that directly follows an English line to it as its translation."""
    cards = []
    for lang, text in items:
        if lang == "en":
            cards.append({"en": text, "fa": []})
        elif cards and cards[-1]["en"] and not cards[-1]["fa"] and (not max_fa or len(text) <= max_fa):
            cards[-1]["fa"].append(text)
        else:
            cards.append({"en": None, "fa": [text]})
    return cards


def tokens_for(items, mode, o):
    """-> [('say', lang, text, slow) | ('pause', seconds)]"""
    toks = []
    if mode.startswith("Study drill"):
        for c in make_cards(items, o["max_fa"]):
            if c["en"]:
                toks += [("say", "en", c["en"], False), ("pause", o["pause"])]
                if o["read_fa"] and c["fa"]:
                    toks += [("say", "fa", " ".join(c["fa"]), False), ("pause", o["pause"])]
                for _ in range(o["repeats"] - 1):
                    toks += [("say", "en", c["en"], True), ("pause", o["pause"])]
            elif o["read_expl"]:
                toks += [("say", "fa", " ".join(c["fa"]), False), ("pause", o["pause"])]
        return toks
    keep = [(l, t) for l, t in items if mode.startswith("Both") or (mode.startswith("English") and l == "en")
            or (mode.startswith("Persian") and l == "fa")]
    for lang, text in keep:
        if o["pause"] <= 0 and toks and toks[-1][0] == "say" and toks[-1][1] == lang:
            toks[-1] = ("say", lang, toks[-1][2] + "\n\n" + text, False)
        else:
            if toks and o["pause"] > 0:
                toks.append(("pause", o["pause"]))
            toks.append(("say", lang, text, False))
    return toks


# ---- silence without ffmpeg: build silent MP3 frames that match the header of the synthesized audio
_BR = {3: [0, 32, 40, 48, 56, 64, 80, 96, 112, 128, 160, 192, 224, 256, 320],
       2: [0, 8, 16, 24, 32, 40, 48, 56, 64, 80, 96, 112, 128, 144, 160],
       0: [0, 8, 16, 24, 32, 40, 48, 56, 64, 80, 96, 112, 128, 144, 160]}
_SR = {3: [44100, 48000, 32000], 2: [22050, 24000, 16000], 0: [11025, 12000, 8000]}


def mp3_silence(sample, seconds):
    hdr, kbps, sr, spf = b"\xff\xf3\x54\xc0", 48, 24000, 576          # defaults = edge-tts (24 kHz, 48 kbps, mono)
    try:
        i = 0
        if sample[:3] == b"ID3":
            i = 10 + ((sample[6] << 21) | (sample[7] << 14) | (sample[8] << 7) | sample[9])
        for j in range(i, min(len(sample) - 4, i + 4096)):
            b1, b2 = sample[j + 1], sample[j + 2]
            ver, layer = (b1 >> 3) & 3, (b1 >> 1) & 3
            if sample[j] == 0xFF and (b1 & 0xE0) == 0xE0 and ver != 1 and layer == 1 and 0 < (b2 >> 4) < 15 and ((b2 >> 2) & 3) < 3:
                kbps, sr = _BR[ver][b2 >> 4], _SR[ver][(b2 >> 2) & 3]
                spf = 1152 if ver == 3 else 576
                hdr = bytes([0xFF, b1 | 1, b2 & ~0x02, sample[j + 3]])  # no CRC, no padding
                break
    except Exception:  # noqa: BLE001
        pass
    length = int((144 if spf == 1152 else 72) * kbps * 1000 / sr)
    frame = hdr + b"\x00" * (length - 4)
    return frame * max(1, int(-(-seconds * sr // spf)))


def rate_str(mult):
    return f"{max(-75, min(100, round((mult - 1) * 100))):+d}%"


async def piece(ctx, text, voice, rate, pitch, volume):
    key = (text, voice, rate, pitch, volume)
    if key not in ctx["cache"]:
        async with ctx["sem"]:
            ctx["cache"][key] = await synth(text, dict(voice=voice, rate=rate, pitch=pitch, volume=volume))
    return ctx["cache"][key]


async def render_tokens(toks, ctx, voices, base):
    say = [t for t in toks if t[0] == "say"]
    if not say:
        return b""

    async def get(t):
        _, lang, text, slow = t
        speed = base["speed"] * (1 + base["slow"] / 100 if slow else 1)
        return await piece(ctx, text, voices[lang], rate_str(speed), base["pitch"], base["volume"])

    audios = await asyncio.gather(*[get(t) for t in say])
    it, out = iter(audios), bytearray()
    for t in toks:
        if t[0] == "say":
            out += next(it)
        elif t[1] > 0:
            out += mp3_silence(audios[0], t[1])
    return bytes(out)


def doc_blocks(items, mode, o):
    """Blocks for the .md / .pdf of a bilingual section: the same text the audio reads (without repeats)."""
    if mode.startswith("Study drill"):
        out = []
        for c in make_cards(items, o["max_fa"]):
            if c["en"]:
                out.append(("qa", c["en"], " ".join(c["fa"])) if (o["read_fa"] and c["fa"]) else ("p", c["en"]))
            elif o["read_expl"]:
                out.append(("p", " ".join(c["fa"])))
        return out
    return [("p", t) for l, t in items if mode.startswith("Both") or (mode.startswith("English") and l == "en")
            or (mode.startswith("Persian") and l == "fa")]


def bilingual_groups(items, mode, o, voices, base, announce, skip_fences):
    groups = []
    for name, text in items:
        stem = "__".join(Path(name).with_suffix("").parts)
        sections = []
        for title, its in bilingual_sections(name, text, skip_fences):
            toks = tokens_for(its, mode, o)
            if not toks:
                continue
            if announce and title:
                toks = [("say", "en", title + ".", False), ("pause", 0.5)] + toks

            def mk(toks):
                async def render(ctx):
                    return await render_tokens(toks, ctx, voices, base)
                return render

            i = len(sections) + 1
            sections.append(Section(name=f"{stem}/{i:03d}_{slug(title or 'section')}", title=title or f"Section {i}",
                                    blocks=doc_blocks(its, mode, o), render=mk(toks)))
        if sections:
            groups.append(Group(stem, Path(name).stem, sections))
    return groups


@st.cache_data(show_spinner=False, max_entries=500)
def preview_audio(voice, text, rate, pitch, volume):
    """Cached short sample so each voice is only generated once per setting."""
    return asyncio.run(synth(text, dict(voice=voice, rate=rate, pitch=pitch, volume=volume), retries=3))


async def _preview_many(shorts, text, cfg, parallel, on_progress):
    sem = asyncio.Semaphore(parallel)
    out, errors = {}, {}

    async def one(short):
        async with sem:
            try:
                out[short] = await synth(text, dict(cfg, voice=short), retries=3)
            except Exception as e:  # noqa: BLE001
                errors[short] = str(e)

    tasks = [asyncio.create_task(one(sh)) for sh in shorts]
    for i, t in enumerate(asyncio.as_completed(tasks), 1):
        await t
        on_progress(i, len(tasks))
    return out, errors


def use_voice(label):
    st.session_state["voice_label"] = label


def export_panel(key, per_label="Per section", whole_label="Whole topic", allow_gran=True):
    """Tick-boxes for MP3 / Markdown / PDF, each with its own granularity -> {'mp3': 'section'|'file'|'both', ...}"""
    st.markdown("**What do you want to get?**")
    labels = {"section": per_label, "file": whole_label, "both": f"Both ({per_label.lower()} + {whole_label.lower()})"}
    out = {}
    for col, ext in zip(st.columns(3), ("mp3", "md", "pdf")):
        with col:
            if st.checkbox(FORMAT_LABELS[ext], value=(ext == "mp3"), key=f"{key}_{ext}_on"):
                out[ext] = st.selectbox(f"{ext} granularity", list(labels), format_func=labels.get,
                                        key=f"{key}_{ext}_gran", label_visibility="collapsed") if allow_gran else "section"
    if not out:
        st.warning("Tick at least one output.")
    return out


def get_pdf_font(fmts, font_bytes):
    """None = PDF not requested, False = no usable font (error shown), else (regular, bold) paths."""
    if "pdf" not in fmts:
        return None
    font = resolve_pdf_font(font_bytes)
    if font[0] is None:
        st.error("No font available for the PDF. Put Vazirmatn-Regular.ttf and Vazirmatn-Bold.ttf in a `fonts/` folder "
                 "next to tts_app.py (or connect to the internet once so they download), or upload a .ttf in the sidebar.")
        return False
    return font


def doc_text(groups):
    return "\n".join(x for g in groups for s in g.sections for b in s.blocks for x in b[1:])


def warn_missing_glyphs(text, font):
    miss = missing_glyphs(text, font[0]) if font and text else []
    if miss:
        st.warning("The PDF font cannot draw these characters (they would appear as empty boxes): "
                   + " ".join(miss[:25]) + (" …" if len(miss) > 25 else "")
                   + "  — upload a font that supports your language in the sidebar (📄 PDF font).")


def show_results(res, tasks, outdir):
    st.success(f"Done: {res['done']} generated, {res['skipped']} skipped, {len(res['failed'])} failed. "
               f"Saved in: {outdir.resolve()}")
    if res["failed"]:
        with st.expander("Failed files (press Generate again to retry only the missing MP3s)"):
            for p, e in res["failed"]:
                st.write(f"`{p}` — {e}")
    return [(t.path, outdir) for t in tasks if t.path.exists()]


# --------------------------------------------------------------------------- UI
def main():
    st.set_page_config(page_title="Unlimited Text to Speech", page_icon="🔊", layout="wide")
    st.title("🔊 Unlimited Text to Speech")
    st.caption("Paste text or upload many files — no character, file-count or duration limit in this app.")

    voices = load_voices()
    if any(v.get("offline") for v in voices):
        st.warning("Could not reach the voice list online — showing a small built-in list. Check your internet connection.")

    with st.sidebar:
        st.header("Voice")
        langs = sorted({v["lang"] for v in voices})
        lang = st.selectbox("Language", langs, index=langs.index("English") if "English" in langs else 0)
        gender = st.radio("Gender", ["All", "Male", "Female"], horizontal=True)
        pool = [v for v in voices if v["lang"] == lang and gender in ("All", v["gender"])] or [v for v in voices if v["lang"] == lang]
        labels = [v["label"] for v in pool]
        default = next((i for i, v in enumerate(pool) if v["short"] == "en-US-AriaNeural"), 0)
        if st.session_state.get("voice_label") not in labels:
            st.session_state.pop("voice_label", None)
        if "voice_label" in st.session_state:  # set by a "Use" button or a previous run
            chosen = st.selectbox("Voice", labels, key="voice_label")
        else:
            chosen = st.selectbox("Voice", labels, index=default, key="voice_label")
        voice = pool[labels.index(chosen)]["short"]
        speed = st.select_slider("Speed", [0.5, 0.75, 0.9, 1.0, 1.1, 1.25, 1.5, 1.75, 2.0], value=1.0, format_func=lambda x: f"{x}x")
        pitch = st.slider("Pitch (Hz)", -50, 50, 0, 5)
        volume = st.slider("Volume (%)", -50, 50, 0, 5)
        cfg = dict(voice=voice, rate=f"{round((speed - 1) * 100):+d}%", pitch=f"{pitch:+d}Hz", volume=f"{volume:+d}%")

        st.divider()
        sample = st.text_input("Preview text", "Hello! This is a preview of this voice.")
        if st.button("▶ Preview voice"):
            try:
                st.audio(asyncio.run(synth(sample, cfg)), format="audio/mp3")
            except Exception as e:  # noqa: BLE001
                st.error(f"Preview failed: {e}")

        with st.expander("📄 PDF font"):
            font_up = st.file_uploader("Custom font (.ttf) — optional", type=["ttf"], key="pdf_font_up")
            st.caption("Default: Vazirmatn (Persian + English) from the `fonts/` folder. "
                       "Upload another .ttf if your text uses other scripts (Chinese, Cyrillic, …).")
        font_bytes = font_up.getvalue() if font_up else None

    tab1, tab2, tab_pdf, tab_bi, tab3 = st.tabs(["✍️ Single text", "📚 Batch files", "📄 PDF", "🌐 Bilingual", "🎙 Voice gallery"])

    # ---------------- single text
    with tab1:
        text = st.text_area("Text", height=320, placeholder="Enter text here… (any length)")
        st.caption(f"{len(text):,} characters")
        title1 = st.text_input("Title (used for the file names and as the heading)", "Speech", key="single_title")
        fm1 = export_panel("single", allow_gran=False)
        font1 = get_pdf_font(fm1, font_bytes)
        if font1:
            warn_missing_glyphs(text, font1)
        if st.button("Generate", type="primary", disabled=not text.strip() or not fm1 or font1 is False):
            base_name = slug(title1, 40) if title1.strip() else "speech"
            sec = Section(base_name, title1.strip() or "Speech",
                          [("p", p) for p in re.split(r"\n\s*\n", text.strip()) if p.strip()], [text])
            files = {}
            try:
                if "mp3" in fm1:
                    with st.spinner("Generating audio…"):
                        files[f"{base_name}.mp3"] = asyncio.run(synth(text, cfg))
                if "md" in fm1:
                    files[f"{base_name}.md"] = render_md(sec.title, [sec], False)
                if "pdf" in fm1:
                    files[f"{base_name}.pdf"] = render_pdf(sec.title, [sec], False, font1)
                st.session_state["single"] = files
            except Exception as e:  # noqa: BLE001
                st.error(f"Failed: {e}")
        for fname, data in (st.session_state.get("single") or {}).items():
            if fname.endswith(".mp3"):
                st.audio(data, format="audio/mp3")
            mime = {"mp3": "audio/mpeg", "md": "text/markdown", "pdf": "application/pdf"}[fname.rsplit(".", 1)[1]]
            st.download_button(f"⬇ Download {fname}", data, fname, mime, key=f"dl_single_{fname}")

    # ---------------- batch
    with tab2:
        st.write("Upload **.txt / .md files or a .zip** (for example the `podcast/` folder from the course), "
                 "or type a local folder path.")
        uploaded = st.file_uploader("Files", type=["txt", "md", "zip"], accept_multiple_files=True)
        folder = st.text_input("…or a local folder path (optional)", placeholder=r"C:\English-for-Shopping-Markdown\podcast")
        c2, c3 = st.columns(2)
        repeat = c2.number_input("Read each sentence N times", 1, 5, 1, help="Handy for shadowing practice (audio only).")
        parallel = c3.slider("Parallel requests", 1, 6, 3, help="Higher is faster but more likely to be throttled.")
        d1, d2, d3 = st.columns(3)
        announce = d1.checkbox("Read chapter titles aloud", True)
        skip = d2.checkbox("Skip MP3 files already generated", True)
        only_fences = d3.checkbox("In .md files read only the ```text podcast blocks", True)
        fmts = export_panel("batch", "Per chapter", "Whole file")
        outdir = Path(st.text_input("Save files to folder", "tts_output"))

        font = get_pdf_font(fmts, font_bytes)
        items = read_sources(uploaded, folder)
        groups = make_groups(items, repeat, announce, only_fences) if items else []
        tasks = plan_tasks(groups, outdir, fmts, cfg, font or None, skip) if groups and fmts and font is not False else []
        if items:
            chars = sum(len(t) for g in groups for s in g.sections for t in s.texts)
            st.info(f"{len(items)} source file(s) → {sum(len(g.sections) for g in groups)} chapter(s), about {chars:,} characters. "
                    f"Will create: {describe_tasks(tasks)}.")
            if font:
                warn_missing_glyphs(doc_text(groups), font)

        if st.button("Generate all", type="primary", disabled=not tasks):
            bar = st.progress(0.0)
            status = st.empty()

            def on_progress(done, total, name, st_):
                bar.progress(done / total)
                status.write(f"{done}/{total} — {name} ({st_})")

            res = asyncio.run(run_tasks(tasks, parallel, skip, on_progress))
            bar.progress(1.0)
            ok = show_results(res, tasks, outdir)
            st.session_state["zip"] = zip_folder(ok) if ok else None

        if st.session_state.get("zip"):
            st.download_button("⬇ Download everything as ZIP", st.session_state["zip"], "tts_output.zip", "application/zip")

    # ---------------- PDF
    with tab_pdf:
        st.write("Upload a PDF, choose the pages, and get audio, Markdown and/or a new clean PDF of the text. "
                 "Works with PDFs that contain real text (scanned images need OCR first). "
                 "Use a voice that matches the language of the text.")
        try:
            import pypdf  # noqa: F401
            have_pypdf = True
        except ImportError:
            have_pypdf = False
            st.error("pypdf is not installed. Run:  pip install pypdf")
        pdf = st.file_uploader("PDF file", type=["pdf"], key="pdf_file") if have_pypdf else None
        if pdf is not None:
            data = pdf.getvalue()
            try:
                total = pdf_page_count(data)
            except Exception as e:  # noqa: BLE001
                st.error(f"Could not open this PDF: {e}")
                total = 0
            if total:
                st.caption(f"{pdf.name} — {total} pages")
                spec = st.text_input("Pages", "1-5", help="Examples: 1-5 · 3, 7, 9-12 · 10- (from 10 to the end) · all")
                o1, o2, o3 = st.columns(3)
                rm_head = o1.checkbox("Remove repeated headers/footers", True)
                rm_num = o2.checkbox("Remove page numbers", True)
                skip_foreign = o3.checkbox("Skip Persian/Arabic/other non-Latin lines", False,
                                           help="Handy for bilingual PDFs: only the English lines are used.")
                m1, m2 = st.columns(2)
                pmode = m1.radio("Split the range into parts", [PM_WHOLE, PM_PAGE, PM_EVERY],
                                 help="A 'part' is the unit that gets its own file when you choose 'Per part' below.")
                every = m2.number_input("N (pages per part)", 1, 500, 5, disabled=pmode != PM_EVERY)
                pfmts = export_panel("pdf", "Per part", "Whole range")
                pout = Path(st.text_input("Save files to folder", "tts_output", key="pdf_out"))
                try:
                    wanted = parse_page_ranges(spec, total)
                except ValueError as e:
                    st.error(str(e)); wanted = []
                if wanted:
                    raw = extract_pdf_pages(data, tuple(wanted))
                    cleaned = clean_pdf_pages(raw, rm_head, rm_num, skip_foreign)
                    full = "\n\n".join(t for _, t in cleaned)
                    if not full.strip():
                        st.warning("No text found on these pages. The PDF may be scanned images (OCR needed) "
                                   "or every line was filtered out.")
                    else:
                        minutes = len(full) / 14 / 60 / max(0.25, float(speed))
                        st.info(f"{len(wanted)} page(s) selected · {len(full):,} characters · roughly {minutes:.0f} min of audio")
                        with st.expander("Preview extracted text"):
                            st.text(full[:6000] + ("\n…" if len(full) > 6000 else ""))
                        pfont = get_pdf_font(pfmts, font_bytes)
                        if pfont:
                            warn_missing_glyphs(full, pfont)
                        groups = pdf_groups(cleaned, slug(Path(pdf.name).stem, 40), Path(pdf.name).stem, pmode, every)
                        tasks = plan_tasks(groups, pout, pfmts, cfg, pfont or None, False) if pfmts and pfont is not False else []
                        if tasks:
                            st.caption(f"Will create: {describe_tasks(tasks)}.")
                        if st.button("Generate", type="primary", key="pdf_go", disabled=not tasks):
                            bar = st.progress(0.0)
                            status = st.empty()

                            def on_prog(done, total_, name, st_):
                                bar.progress(done / total_)
                                status.write(f"{done}/{total_} — {name} ({st_})")

                            with st.spinner("Generating… long ranges can take a while"):
                                res = asyncio.run(run_tasks(tasks, 3, False, on_prog))
                            bar.progress(1.0)
                            okf = show_results(res, tasks, pout)
                            st.session_state["pdf_zip"] = zip_folder(okf) if okf else None
                            if len(okf) == 1 and okf[0][0].suffix == ".mp3":
                                st.audio(okf[0][0].read_bytes(), format="audio/mp3")
        if st.session_state.get("pdf_zip"):
            st.download_button("⬇ Download PDF-tab output as ZIP", st.session_state["pdf_zip"], "pdf_output.zip", "application/zip")

    # ---------------- bilingual
    with tab_bi:
        st.write("For files that mix **English and Persian** (like this course): each language is read by its own voice. "
                 "The language is detected automatically, line by line.")
        en_pool = [v for v in voices if v["locale"].startswith("en")]
        fa_pool = [v for v in voices if v["locale"].startswith("fa")]
        if not fa_pool:
            st.warning("No Persian voice found in the voice list.")
        vc1, vc2 = st.columns(2)
        en_labels = [v["label"] for v in en_pool]
        fa_labels = [v["label"] for v in fa_pool]
        en_def = next((i for i, v in enumerate(en_pool) if v["short"] == (voice if voice.startswith("en") else "en-US-AriaNeural")), 0)
        en_v = en_pool[en_labels.index(vc1.selectbox("English voice", en_labels, index=en_def))]["short"] if en_pool else voice
        fa_def = next((i for i, v in enumerate(fa_pool) if v["short"] == "fa-IR-DilaraNeural"), 0)
        fa_v = fa_pool[fa_labels.index(vc2.selectbox("Persian voice", fa_labels, index=fa_def))]["short"] if fa_pool else voice
        voices_map = {"en": en_v, "fa": fa_v}

        bmode = st.radio("What to make", [
            "Study drill (English → Persian → English again)",
            "Both languages in original order",
            "English only",
            "Persian only"])
        o = dict(pause=0.0, repeats=1, read_fa=True, read_expl=False, max_fa=160)
        slow = -30
        if bmode.startswith("Study drill"):
            d1, d2, d3 = st.columns(3)
            o["repeats"] = d1.number_input("English repeats", 1, 3, 2, help="1 = once. Extra repeats are slower.")
            slow = d2.slider("Speed of repeats (%)", -60, 0, -30, 5)
            o["pause"] = d3.slider("Pause after each part (s)", 0.0, 6.0, 1.5, 0.5, help="Time to repeat aloud (shadowing).")
            e1, e2, e3 = st.columns(3)
            o["read_fa"] = e1.checkbox("Read the Persian translation", True)
            o["read_expl"] = e2.checkbox("Also read Persian explanations", False)
            o["max_fa"] = e3.number_input("Max length of a 'translation' (chars, 0 = any)", 0, 2000, 160,
                                          help="A Persian line right after an English line counts as its translation if it is this short.")
        else:
            o["pause"] = st.slider("Pause between lines (s)", 0.0, 6.0, 0.0, 0.5,
                                   help="0 = read continuously. Above 0 every line is a separate clip.")
        base = dict(speed=speed, slow=slow, pitch=cfg["pitch"], volume=cfg["volume"])

        src = st.radio("Source", ["Paste text", "Upload files (.txt / .md / .zip)"], horizontal=True)
        b_items = []
        if src == "Paste text":
            pasted = st.text_area("Bilingual text", height=220, key="bi_text",
                                  placeholder="Where is the bread?\nنان کجاست؟\n…")
            if pasted.strip():
                b_items = [("pasted.txt", pasted)]
        else:
            up = st.file_uploader("Files", type=["txt", "md", "zip"], accept_multiple_files=True, key="bi_files")
            fold = st.text_input("…or a local folder path (optional)", key="bi_folder", placeholder=r"C:\English-for-Shopping-Markdown\parts")
            b_items = read_sources(up, fold)
        f1, f2 = st.columns(2)
        skip_fences = f1.checkbox("Skip ```text podcast blocks (duplicates)", True)
        announce = f2.checkbox("Read chapter numbers aloud", True)
        bfmts = export_panel("bi", "Per chapter", "Whole file")

        parsed = [(n, bilingual_sections(n, t, skip_fences)) for n, t in b_items]
        all_items = [it for _, secs in parsed for _, its in secs for it in its]
        if all_items:
            n_en = sum(1 for l, _ in all_items if l == "en")
            st.info(f"Detected {n_en} English and {len(all_items) - n_en} Persian line(s) in "
                    f"{sum(len(s) for _, s in parsed)} section(s).")
            first = next((its for _, secs in parsed for _, its in secs if its), [])
            with st.expander("Check the language detection (first lines)"):
                st.dataframe([{"voice": "English" if l == "en" else "Persian", "text": t} for l, t in first[:40]],
                             hide_index=True)
            if st.button("🎧 Try the first few lines"):
                sample = tokens_for(first[:14], bmode, o)
                if sample:
                    with st.spinner("Generating…"):
                        try:
                            ctx = {"sem": asyncio.Semaphore(3), "cache": {}}
                            st.session_state["bi_try"] = asyncio.run(render_tokens(sample, ctx, voices_map, base))
                        except Exception as e:  # noqa: BLE001
                            st.error(f"Failed: {e}")
            if st.session_state.get("bi_try"):
                st.audio(st.session_state["bi_try"], format="audio/mp3")

        bout = Path(st.text_input("Save files to folder", "tts_output", key="bi_out"))
        g1, g2 = st.columns(2)
        bskip = g1.checkbox("Skip MP3 files already generated", True, key="bi_skip")
        bpar = g2.slider("Parallel requests", 1, 6, 3, key="bi_par")
        bfont = get_pdf_font(bfmts, font_bytes)
        bgroups = bilingual_groups(b_items, bmode, o, voices_map, base, announce, skip_fences) if b_items else []
        btasks = plan_tasks(bgroups, bout, bfmts, cfg, bfont or None, bskip) if bgroups and bfmts and bfont is not False else []
        if bgroups:
            st.caption(f"Will create: {describe_tasks(btasks)}.")
            if bfont:
                warn_missing_glyphs(doc_text(bgroups), bfont)
        if st.button("Generate", type="primary", key="bi_go", disabled=not btasks):
            bar = st.progress(0.0)
            status = st.empty()

            def on_bi(done, total_, name, st_):
                bar.progress(done / total_)
                status.write(f"{done}/{total_} — {name} ({st_})")

            with st.spinner("Generating… drills need several requests per line, so this takes longer"):
                res = asyncio.run(run_tasks(btasks, bpar, bskip, on_bi))
            bar.progress(1.0)
            okf = show_results(res, btasks, bout)
            st.session_state["bi_zip"] = zip_folder(okf) if okf else None
        if st.session_state.get("bi_zip"):
            st.download_button("⬇ Download bilingual output as ZIP", st.session_state["bi_zip"], "bilingual_output.zip", "application/zip")

    # ---------------- voice gallery
    with tab3:
        st.write(f"Listen to the **{len(pool)}** voice(s) for **{lang}** "
                 f"({'all genders' if gender == 'All' else gender.lower()}). Change language / gender in the sidebar. "
                 "Speed, pitch and volume from the sidebar are applied to the samples.")
        gtext = st.text_input("Sample text", "Hello! Where can I find the olive oil, please?", key="gallery_text")
        gallery = st.session_state.setdefault("gallery", {})
        key_now = (gtext, cfg["rate"], cfg["pitch"], cfg["volume"])
        if st.session_state.get("gallery_key") != key_now:  # settings changed -> old samples are stale
            gallery.clear()
            st.session_state["gallery_key"] = key_now

        if st.button(f"⏬ Generate samples for all {len(pool)} voices", disabled=not gtext.strip()):
            todo = [v["short"] for v in pool if v["short"] not in gallery]
            if todo:
                bar = st.progress(0.0)
                out, errors = asyncio.run(_preview_many(todo, gtext, cfg, 3, lambda i, n: bar.progress(i / n)))
                gallery.update(out)
                bar.empty()
                if errors:
                    st.warning(f"{len(errors)} sample(s) failed — press the button again to retry them.")

        last = st.session_state.pop("last_played", None)
        for v in pool:
            c_name, c_play, c_use, c_audio = st.columns([3.2, 1, 1, 4])
            c_name.markdown(f"{'♂' if v['gender'] == 'Male' else '♀'} **{v['label']}**  \n`{v['short']}`")
            if c_play.button("▶ Play", key=f"play_{v['short']}"):
                with st.spinner("…"):
                    try:
                        gallery[v["short"]] = preview_audio(v["short"], gtext, cfg["rate"], cfg["pitch"], cfg["volume"])
                        last = v["short"]
                    except Exception as e:  # noqa: BLE001
                        c_audio.error(f"Failed: {e}")
            c_use.button("✔ Use", key=f"use_{v['short']}", on_click=use_voice, args=(v["label"],),
                         type="primary" if v["short"] == voice else "secondary")
            if v["short"] in gallery:
                try:
                    c_audio.audio(gallery[v["short"]], format="audio/mp3", autoplay=(v["short"] == last))
                except TypeError:  # older Streamlit without autoplay
                    c_audio.audio(gallery[v["short"]], format="audio/mp3")
            st.divider()


main()
