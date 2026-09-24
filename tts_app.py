"""
Unlimited Text-to-Speech (Streamlit + edge-tts)

Install & run:
    pip install streamlit edge-tts pypdf
    streamlit run tts_app.py

Features
  * Any language / gender / voice that edge-tts offers, speed / pitch / volume
  * PDF tab: upload a PDF, choose pages (e.g. 1-5, 8, 10-12), get one MP3 for the range / per page / every N pages
  * Voice gallery: listen to every voice of a language side by side, then pick one with a click
  * Single text tab: paste any amount of text (it is split into safe chunks automatically)
  * Batch tab: upload many .txt / .md files or a whole .zip (e.g. the `podcast/` folder),
    or point to a local folder. One MP3 per chapter or one MP3 per file. Everything is
    saved to a local folder and offered as one ZIP. Runs can be resumed (finished files are skipped).
  * The app itself has no character, file-count or duration limit.
    (Only the online Microsoft service behind edge-tts can throttle you; the app retries automatically.)
"""
import asyncio
import io
import re
import zipfile
from dataclasses import dataclass, field
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
    return re.sub(r"[^A-Za-z0-9]+", "_", s).strip("_")[:n] or "section"


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


@dataclass
class Job:
    path: Path
    texts: list = field(default_factory=list)


def make_jobs(items, outdir, mode, repeat, announce, only_fences):
    jobs = []
    for name, text in items:
        p = Path(name)
        stem = "__".join(p.with_suffix("").parts)
        secs = sections_from_md(text, only_fences) if name.lower().endswith(".md") else sections_from_txt(text)
        secs = [(t, l) for t, l in secs if l]
        if not secs:
            continue
        texts = [(t, build_text(t, l, repeat, announce)) for t, l in secs]
        if mode == "One MP3 per chapter":
            for i, (t, body) in enumerate(texts, 1):
                jobs.append(Job(outdir / stem / f"{i:03d}_{slug(t or 'section')}.mp3", [body]))
        else:
            jobs.append(Job(outdir / f"{stem}.mp3", [b for _, b in texts]))
    return jobs


async def run_batch(jobs, cfg, parallel, skip_existing, on_progress):
    sem = asyncio.Semaphore(parallel)
    results = {"done": 0, "skipped": 0, "failed": []}

    async def work(job):
        async with sem:
            if skip_existing and job.path.exists() and job.path.stat().st_size > 0:
                return job, "skipped", None
            try:
                audio = b"".join([await synth(t, cfg) for t in job.texts])
                job.path.parent.mkdir(parents=True, exist_ok=True)
                job.path.write_bytes(audio)
                return job, "done", None
            except Exception as e:  # noqa: BLE001
                return job, "failed", str(e)

    tasks = [asyncio.create_task(work(j)) for j in jobs]
    finished = 0
    for fut in asyncio.as_completed(tasks):
        job, status, err = await fut
        finished += 1
        if status == "failed":
            results["failed"].append((str(job.path), err))
        else:
            results[status] += 1
        on_progress(finished, len(jobs), job.path.name, status)
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


def pdf_jobs(pages, stem, outdir, mode, every):
    pages = [(n, t) for n, t in pages if t.strip()]
    if not pages:
        return []
    if mode == "One MP3 per page":
        return [Job(outdir / f"{stem}_page_{n:03d}.mp3", [t]) for n, t in pages]
    step = len(pages) if mode == "One MP3 for the whole range" else max(1, int(every))
    jobs = []
    for i in range(0, len(pages), step):
        grp = pages[i:i + step]
        a, b = grp[0][0], grp[-1][0]
        name = f"{stem}_page_{a:03d}.mp3" if a == b else f"{stem}_pages_{a:03d}-{b:03d}.mp3"
        jobs.append(Job(outdir / name, ["\n\n".join(t for _, t in grp)]))
    return jobs


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

    tab1, tab2, tab_pdf, tab3 = st.tabs(["✍️ Single text", "📚 Batch files", "📄 PDF", "🎙 Voice gallery"])

    # ---------------- single text
    with tab1:
        text = st.text_area("Text", height=320, placeholder="Enter text here… (any length)")
        st.caption(f"{len(text):,} characters")
        if st.button("Generate audio", type="primary", disabled=not text.strip()):
            with st.spinner("Generating…"):
                try:
                    st.session_state["single"] = asyncio.run(synth(text, cfg))
                except Exception as e:  # noqa: BLE001
                    st.error(f"Failed: {e}")
        if st.session_state.get("single"):
            st.audio(st.session_state["single"], format="audio/mp3")
            st.download_button("⬇ Download MP3", st.session_state["single"], "speech.mp3", "audio/mpeg")

    # ---------------- batch
    with tab2:
        st.write("Upload **.txt / .md files or a .zip** (for example the `podcast/` folder from the course), "
                 "or type a local folder path.")
        uploaded = st.file_uploader("Files", type=["txt", "md", "zip"], accept_multiple_files=True)
        folder = st.text_input("…or a local folder path (optional)", placeholder=r"C:\English-for-Shopping-Markdown\podcast")
        c1, c2, c3 = st.columns(3)
        mode = c1.radio("Output", ["One MP3 per chapter", "One MP3 per file (chapters merged)"])
        repeat = c2.number_input("Read each sentence N times", 1, 5, 1, help="Handy for shadowing practice.")
        parallel = c3.slider("Parallel requests", 1, 6, 3, help="Higher is faster but more likely to be throttled.")
        d1, d2, d3 = st.columns(3)
        announce = d1.checkbox("Read chapter titles aloud", True)
        skip = d2.checkbox("Skip files already generated", True)
        only_fences = d3.checkbox("In .md files read only the ```text podcast blocks", True)
        outdir = Path(st.text_input("Save MP3 files to folder", "tts_output"))

        items = read_sources(uploaded, folder)
        jobs = make_jobs(items, outdir, mode, repeat, announce, only_fences) if items else []
        if items:
            chars = sum(len(t) for j in jobs for t in j.texts)
            st.info(f"{len(items)} source file(s) → {len(jobs)} MP3 file(s), about {chars:,} characters.")

        if st.button("Generate all", type="primary", disabled=not jobs):
            bar = st.progress(0.0)
            status = st.empty()
            log = st.container()

            def on_progress(done, total, name, st_):
                bar.progress(done / total)
                status.write(f"{done}/{total} — {name} ({st_})")

            res = asyncio.run(run_batch(jobs, cfg, parallel, skip, on_progress))
            bar.progress(1.0)
            ok = [(j.path, outdir) for j in jobs if j.path.exists()]
            st.success(f"Done: {res['done']} generated, {res['skipped']} skipped, {len(res['failed'])} failed. "
                       f"Saved in: {outdir.resolve()}")
            if res["failed"]:
                with log.expander("Failed files (press Generate all again to retry only these)"):
                    for p, e in res["failed"]:
                        st.write(f"`{p}` — {e}")
            if ok:
                st.session_state["zip"] = zip_folder(ok)

        if st.session_state.get("zip"):
            st.download_button("⬇ Download everything as ZIP", st.session_state["zip"], "tts_audio.zip", "application/zip")

    # ---------------- PDF
    with tab_pdf:
        st.write("Upload a PDF, choose the pages, and get audio. Works with PDFs that contain real text "
                 "(scanned images need OCR first). Use a voice that matches the language of the text.")
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
                                           help="Handy for bilingual PDFs: only the English lines are read.")
                m1, m2 = st.columns(2)
                pmode = m1.radio("Output", ["One MP3 for the whole range", "One MP3 per page", "One MP3 every N pages"])
                every = m2.number_input("N (pages per MP3)", 1, 500, 5, disabled=pmode != "One MP3 every N pages")
                pout = Path(st.text_input("Save MP3 files to folder", "tts_output", key="pdf_out"))
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
                        if st.button("Generate PDF audio", type="primary"):
                            stem = slug(Path(pdf.name).stem, 40)
                            jobs = pdf_jobs(cleaned, stem, pout, pmode, every)
                            bar = st.progress(0.0)
                            status = st.empty()

                            def on_prog(done, total_, name, st_):
                                bar.progress(done / total_)
                                status.write(f"{done}/{total_} — {name} ({st_})")

                            with st.spinner("Generating… long ranges can take a while"):
                                res = asyncio.run(run_batch(jobs, cfg, 3, False, on_prog))
                            bar.progress(1.0)
                            st.success(f"Done: {res['done']} file(s) generated, {len(res['failed'])} failed. "
                                       f"Saved in: {pout.resolve()}")
                            for pth, err in res["failed"]:
                                st.error(f"`{pth}` — {err}")
                            okf = [(j.path, pout) for j in jobs if j.path.exists()]
                            st.session_state["pdf_zip"] = zip_folder(okf) if okf else None
                            if len(okf) == 1:
                                st.audio(okf[0][0].read_bytes(), format="audio/mp3")
        if st.session_state.get("pdf_zip"):
            st.download_button("⬇ Download PDF audio as ZIP", st.session_state["pdf_zip"], "pdf_audio.zip", "application/zip")

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
