#!/usr/bin/env python3
"""
Transcribe Pro — a modernized, single‑file Tkinter app for audio transcription

Key upgrades over your original script
- Real progress (not simulated) based on audio duration and segment timestamps
- Multi‑file batch queue with per‑item status
- Cancel/Stop button and safe thread shutdowns (cancels show as Canceled, not Error)
- Choice of engine: Faster‑Whisper (default) or OpenAI Whisper (classic)
- Model picker (tiny…large‑v3), device/compute auto‑selection (CPU/GPU, float16/int8)
- VAD (voice activity detection) + word timestamps options
- Output formats: .txt, .srt, .vtt, .json (raw segments), .csv (segments)
- Language auto‑detect with manual override
- Automatic audio pre‑processing via ffmpeg/pydub; normalizes sample rate/mono
- Settings persisted to ~/.transcribe_pro.json (includes CSV)
- Rotating logs to ~/.transcribe_pro.log
- Minimal, clean ttk UI with dark mode toggle
- OpenAI-powered Meeting Distiller to generate structured notes (.md)
- NEW: Optional Meeting Distiller exports to **HTML (.html)** and **PDF (.pdf)**

Dependencies
    pip install faster-whisper pydub torch        # (torch optional if using only faster-whisper CPU)
    # For classic whisper path:
    pip install openai-whisper                    # or whisper from GitHub
    # For Meeting Distiller (OpenAI):
    pip install openai
    # For HTML/PDF export (optional; we'll try these in order if present):
    pip install markdown                          # Markdown → HTML (recommended)
    pip install pdfkit                            # HTML → PDF via wkhtmltopdf
    # and install wkhtmltopdf from your OS package manager, OR
    pip install weasyprint                        # alternative HTML → PDF (needs Cairo/Pango), OR
    pip install reportlab                         # simple text-only PDF fallback

Ensure ffmpeg is installed and on PATH.

Tested with Python 3.10+
"""

import os
import sys
import json
import time
import re
import textwrap
import queue
import threading
import logging
from logging.handlers import RotatingFileHandler
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any

import csv  # module-level import for export speed
import argparse
import tempfile

import tkinter as tk
from tkinter import ttk, filedialog, messagebox

from pydub import AudioSegment

# Optional imports guarded at runtime
try:
    import torch  # for classic whisper device check
except Exception:
    torch = None

# Optional HTML/PDF toolchain — we handle missing deps at runtime gracefully
try:
    import markdown as mdlib  # pip install markdown
except Exception:
    mdlib = None

try:
    import pdfkit  # pip install pdfkit  (requires wkhtmltopdf)
except Exception:
    pdfkit = None

try:
    from weasyprint import HTML as WeasyHTML  # pip install weasyprint
except Exception:
    WeasyHTML = None

try:
    from reportlab.lib.pagesizes import letter
    from reportlab.pdfgen import canvas as pdfcanvas
    from reportlab.lib.units import inch
except Exception:
    letter = None
    pdfcanvas = None
    inch = None

# Honor a user cache location for models if present
os.environ.setdefault("HF_HOME", os.path.join(os.path.expanduser("~"), ".cache", "huggingface"))

APP_NAME = "Transcribe Pro"
CFG_PATH = os.path.join(os.path.expanduser("~"), ".transcribe_pro.json")
LOG_PATH = os.path.join(os.path.expanduser("~"), ".transcribe_pro.log")
OUT_DEFAULT = os.path.join(os.path.expanduser("~"), "Transcripts")

# ----------------------------- Logging -------------------------------------
logger = logging.getLogger(APP_NAME)
logger.setLevel(logging.DEBUG)
handler = RotatingFileHandler(LOG_PATH, maxBytes=2_000_000, backupCount=3)
handler.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))
logger.addHandler(handler)

# --------------------------- Small utilities -------------------------------

def load_cfg() -> Dict[str, Any]:
    defaults = {
        "last_output": OUT_DEFAULT,
        "engine": "faster-whisper",           # or "whisper"
        "model": "base",                      # tiny|base|small|medium|large-v3
        "compute": "auto",                    # auto|float16|int8|int8_float16
        "device": "auto",                     # auto|cpu|gpu
        "vad_filter": True,
        "word_timestamps": False,
        "language": "auto",                   # auto or language code
        "dark": True,
        "formats": {"txt": True, "srt": True, "vtt": False, "json": False, "csv": False},
        "summarize": {
            "enabled": False,
            "model": "gpt-4o-mini",
            "max_output_tokens": 2000,
            "chunk_chars": 32000,
            "remember_key": False,
            "prompt": (
                "You will be provided meeting transcripts. You will provide in return detailed notes "
                "that include action items, deadlines, due dates, and as much detail as possible. "
                "Please identify any commitments made, including when beneficial specific quotes made "
                "by participants, any emails requested and what the contents of those emails should be "
                "and any parameters mentioned that I need to know in order to deliver the best possible job "
                "for their projects. Include everything a great project manager will need to move this project into greatness. "
                "Also any suggestions you have to make it better.\n\n"
                "Action Items & Owners\n"
                "Deadlines & Milestones\n"
                "Details on Commitments & Deliverables\n"
                "Emails Requested & Suggested Content\n"
                "Parameters & Requirements\n"
                "Prioritization & Next Steps\n"
                "Recommendations for Improvement"
            ),
            "html": False,   # NEW — export summary to HTML
            "pdf": False,    # NEW — export summary to PDF
        },
    }
    try:
        if os.path.exists(CFG_PATH):
            with open(CFG_PATH, "r", encoding="utf-8") as f:
                data = json.load(f)
                # merge nested dicts to preserve new keys
                if "formats" in data:
                    defaults["formats"].update(data.get("formats", {}))
                if "summarize" in data:
                    defaults["summarize"].update(data.get("summarize", {}))
                defaults.update({k: v for k, v in data.items() if k not in ("formats", "summarize")})
    except Exception as e:
        logger.exception("Failed to load config: %s", e)
    return defaults


def save_cfg(cfg: Dict[str, Any]):
    try:
        with open(CFG_PATH, "w", encoding="utf-8") as f:
            json.dump(cfg, f, indent=2)
    except Exception as e:
        logger.exception("Failed to save config: %s", e)


def human_time(sec: float) -> str:
    sec = max(0, int(sec))
    h, rem = divmod(sec, 3600)
    m, s = divmod(rem, 60)
    if h:
        return f"{h:d}:{m:02d}:{s:02d}"
    return f"{m:d}:{s:02d}"


def ts_to_srt(ts: float) -> str:
    h, rem = divmod(int(ts), 3600)
    m, s = divmod(rem, 60)
    ms = int((ts - int(ts)) * 1000)
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"

# ----------------------------- Data types ----------------------------------

@dataclass
class Job:
    path: str
    status: str = "Queued"  # Queued | Working | Done | Error | Canceled
    duration: float = 0.0
    progress: float = 0.0    # 0..1
    outdir: Optional[str] = None
    outputs: List[str] = field(default_factory=list)
    error: Optional[str] = None

# ----------------------------- Worker thread -------------------------------

class TranscribeWorker(threading.Thread):
    def __init__(self, jobs: List[Job], settings: Dict[str, Any], ui_queue: queue.Queue, cancel_event: threading.Event):
        super().__init__(daemon=True)
        self.jobs = jobs
        self.settings = settings
        self.ui_queue = ui_queue
        self.cancel_event = cancel_event

    def log_ui(self, kind: str, payload: Any):
        self.ui_queue.put((kind, payload))

    # ----------------- helpers -----------------
    def _cuda_available(self) -> bool:
        try:
            import torch
            return torch.cuda.is_available()
        except Exception:
            return False

    def _choose_compute(self, device: str, pref: str) -> str:
        if device == "cpu":
            return "int8" if pref in ("auto", "int8") else pref
        # GPU
        if pref == "auto":
            return "float16"
        return pref

    def _ensure_wav_mono16(self, src: str) -> str:
        """Convert to mono 16k wav with a unique temp filename; raise RuntimeError on failure."""
        try:
            audio = AudioSegment.from_file(src)
            audio = audio.set_channels(1).set_frame_rate(16000)
            tmp = f"{src}.{os.getpid()}.{int(time.time())}.tmp_mono16.wav"
            audio.export(tmp, format="wav")
            return tmp
        except Exception as e:
            logger.exception("Failed to preprocess audio %s: %s", src, e)
            raise RuntimeError(f"Audio preprocessing failed: {e}")

    def _make_outdir(self, root: str, src: str) -> str:
        base = os.path.splitext(os.path.basename(src))[0]
        return os.path.join(root, base)

    # ---------- Meeting Distiller helpers ----------
    def _split_into_chunks(self, text: str, max_chars: int) -> List[str]:
        """Greedy split on double-newline boundaries (fallback to words) to respect max_chars per chunk."""
        if len(text) <= max_chars:
            return [text]
        parts: List[str] = []
        buf = []
        cur = 0
        paras = re.split(r"\n{2,}", text)
        for p in paras:
            if cur + len(p) + 2 > max_chars and buf:
                parts.append("\n\n".join(buf))
                buf = [p]
                cur = len(p)
            else:
                buf.append(p)
                cur += len(p) + 2
        if buf:
            parts.append("\n\n".join(buf))
        # Second pass split by words if needed
        out: List[str] = []
        for chunk in parts:
            if len(chunk) <= max_chars:
                out.append(chunk)
            else:
                words = chunk.split()
                acc = []
                clen = 0
                for w in words:
                    if clen + len(w) + 1 > max_chars and acc:
                        out.append(" ".join(acc))
                        acc = [w]
                        clen = len(w)
                    else:
                        acc.append(w)
                        clen += len(w) + 1
                if acc:
                    out.append(" ".join(acc))
        return out

    def _extract_response_text(self, resp) -> str:
        """Be resilient to SDK changes: prefer response.output_text; else walk items/content."""
        try:
            txt = getattr(resp, "output_text", None)
            if txt:
                return str(txt)
        except Exception:
            pass
        try:
            texts: List[str] = []
            for item in getattr(resp, "output", []) or []:
                content = getattr(item, "content", None)
                if not content:
                    continue
                for c in content:
                    t = getattr(c, "text", None)
                    if t:
                        texts.append(str(t))
            return "\n".join(texts).strip()
        except Exception:
            return ""

    def _md_to_html(self, md_text: str, title: str = "Meeting Notes") -> str:
        """Convert Markdown to a standalone HTML string with minimal CSS; works without markdown lib."""
        CSS = (
            "body{font-family:system-ui,-apple-system,Segoe UI,Roboto,Inter,Arial,sans-serif;" \
            "line-height:1.5;margin:24px;color:#222;}" \
            "h1,h2,h3{margin-top:1.2em;border-bottom:1px solid #eee;padding-bottom:4px;}" \
            "code,pre{font-family:ui-monospace,Consolas,Monaco,monospace;background:#f6f8fa;}" \
            "pre{padding:12px;overflow:auto;} table{border-collapse:collapse;}" \
            "td,th{border:1px solid #ddd;padding:6px 8px;}" \
            "ul{margin-left:1.2em;} .small{color:#666;font-size:12px;}"
        )
        if mdlib is not None:
            try:
                body = mdlib.markdown(md_text, extensions=["fenced_code", "tables", "toc"])
            except Exception:
                body = f"<pre>{self._html_escape(md_text)}</pre>"
        else:
            body = f"<pre>{self._html_escape(md_text)}</pre>"
        html = f"""<!doctype html>
<html lang="en">
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>{self._html_escape(title)}</title>
<style>{CSS}</style>
<body>
<h1>{self._html_escape(title)}</h1>
{body}
<hr><div class="small">Generated by {APP_NAME}</div>
</body>
</html>"""
        return html

    def _html_escape(self, s: str) -> str:
        return (
            s.replace("&", "&amp;")
             .replace("<", "&lt;")
             .replace(">", "&gt;")
             .replace('"', "&quot;")
        )

    def _write_pdf_from_html(self, html_str: str, outpath: str):
        """Try pdfkit→weasyprint→reportlab; raise on total failure."""
        last_err = None
        # pdfkit (wkhtmltopdf)
        if pdfkit is not None:
            try:
                pdfkit.from_string(html_str, outpath)
                return
            except Exception as e:
                last_err = e
        # weasyprint
        if WeasyHTML is not None:
            try:
                WeasyHTML(string=html_str).write_pdf(outpath)
                return
            except Exception as e:
                last_err = e
        # reportlab fallback — render as text (no HTML styling)
        if pdfcanvas is not None and letter is not None and inch is not None:
            try:
                c = pdfcanvas.Canvas(outpath, pagesize=letter)
                width, height = letter
                x = 0.75 * inch
                y = height - 0.75 * inch
                max_width = width - 1.5 * inch
                # Extract rough text (strip tags)
                text = re.sub(r"<[^>]+>", "", html_str)
                for para in text.splitlines():
                    wrapped = textwrap.wrap(para, width=95)
                    for line in wrapped:
                        if y < 1 * inch:
                            c.showPage()
                            y = height - 0.75 * inch
                        c.drawString(x, y, line)
                        y -= 14
                    y -= 6
                c.save()
                return
            except Exception as e:
                last_err = e
        raise RuntimeError(f"PDF export failed: {last_err}")

    def _summarize_text(self, full_text: str, settings: Dict[str, Any]) -> str:
        """Summarize with OpenAI (Responses API), chunking if needed; fallback to Chat Completions if necessary."""
        if not full_text.strip():
            return ""
        try:
            from openai import OpenAI
        except Exception:
            raise RuntimeError("The 'openai' package is not installed. Run: pip install openai")

        api_key = settings.get("openai_api_key") or os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise RuntimeError("OpenAI API key not provided. Set OPENAI_API_KEY or enter it in the UI.")

        client = OpenAI(api_key=api_key)
        model = settings["summarize"]["model"]
        max_tok = int(settings["summarize"].get("max_output_tokens", 2000))
        max_chars = int(settings["summarize"].get("chunk_chars", 32000))
        prompt = settings["summarize"]["prompt"]

        chunks = self._split_into_chunks(full_text, max_chars)
        partials: List[str] = []

        try:
            if len(chunks) == 1:
                resp = client.responses.create(
                    model=model,
                    instructions=prompt,
                    input=chunks[0],
                    max_output_tokens=max_tok,
                )
                return self._extract_response_text(resp)
            else:
                # First pass: summarize each chunk
                for i, ch in enumerate(chunks, 1):
                    if self.cancel_event.is_set():
                        raise RuntimeError("Canceled")
                    self.log_ui("progress", min(0.95, i / max(1, len(chunks))))
                    pass_prompt = (
                        prompt
                        + "\n\nYou are summarizing CHUNK "
                        + f"{i}/{len(chunks)} of a long transcript. Follow the exact sections and be concise; do not reference chunk numbers in the output."
                    )
                    resp = client.responses.create(
                        model=model,
                        instructions=pass_prompt,
                        input=ch,
                        max_output_tokens=max_tok,
                    )
                    partials.append(self._extract_response_text(resp))
                # Second pass: consolidate
                consolidate_input = "\n\n---\n\n".join(partials)
                final_prompt = (
                    prompt
                    + "\n\nYou are given multiple partial summaries separated by '---'. Merge, deduplicate, and ensure the final output is comprehensive, actionable, and formatted into the same sections."
                )
                resp = client.responses.create(
                    model=model,
                    instructions=final_prompt,
                    input=consolidate_input,
                    max_output_tokens=max_tok,
                )
                return self._extract_response_text(resp)
        except Exception as e:
            # Fallback to Chat Completions for robustness
            try:
                chat = client.chat.completions.create(
                    model=model,
                    messages=[
                        {"role": "system", "content": prompt},
                        {"role": "user", "content": full_text if len(chunks) == 1 else "\n\n---\n\n".join(partials)},
                    ],
                )
                return chat.choices[0].message.content or ""
            except Exception as e2:
                raise RuntimeError(f"OpenAI summarization failed: {e2}") from e

    def _transcribe(self, engine: str, model, fast_model, in_path: str, language: str, vad: bool, word_ts: bool):
        """Return (segments_list, detected_lang). Each segment is a dict with start,end,text,optional words."""
        segments_list: List[Dict[str, Any]] = []
        detected_lang = None
        try:
            if engine == "faster-whisper":
                # Faster-Whisper streams segments; we update progress using segment end time
                total_dur = AudioSegment.from_file(in_path).duration_seconds or 1.0
                options = {
                    "vad_filter": vad,
                    "language": None if language == "auto" else language,
                    "word_timestamps": word_ts,
                }
                logger.info("FW transcribe %s with %s", os.path.basename(in_path), options)
                last_end = 0.0
                seg_iter, detected_lang = fast_model.transcribe(in_path, **options)
                for seg in seg_iter:
                    if self.cancel_event.is_set():
                        raise RuntimeError("Canceled")
                    item = {
                        "id": getattr(seg, "id", None),
                        "start": float(getattr(seg, "start", 0.0) or 0.0),
                        "end": float(getattr(seg, "end", 0.0) or 0.0),
                        "text": getattr(seg, "text", "").strip(),
                    }
                    if word_ts and getattr(seg, "words", None):
                        item["words"] = [
                            {
                                "start": float(getattr(w, "start", 0.0) or 0.0),
                                "end": float(getattr(w, "end", 0.0) or 0.0),
                                "word": getattr(w, "word", ""),
                            }
                            for w in seg.words
                        ]
                    segments_list.append(item)
                    last_end = item["end"] or last_end
                    self.log_ui("progress", min(0.99, last_end / max(1e-6, total_dur)))
                self.log_ui("progress", 1.0)
            else:
                # Classic OpenAI Whisper path
                import whisper
                options = {
                    "language": None if language == "auto" else language,
                    "word_timestamps": word_ts,
                    "fp16": self._cuda_available(),
                }
                logger.info("Whisper transcribe %s with %s", os.path.basename(in_path), options)
                result = model.transcribe(in_path, **options)
                detected_lang = result.get("language", "unknown")
                total = len(result.get("segments", [])) or 1
                for idx, seg in enumerate(result.get("segments", []), start=1):
                    if self.cancel_event.is_set():
                        raise RuntimeError("Canceled")
                    item = {
                        "id": seg.get("id"),
                        "start": float(seg.get("start", 0.0)),
                        "end": float(seg.get("end", 0.0)),
                        "text": seg.get("text", "").strip(),
                    }
                    if word_ts and seg.get("words"):
                        item["words"] = [
                            {
                                "start": float(w.get("start", 0.0)),
                                "end": float(w.get("end", 0.0)),
                                "word": w.get("word", ""),
                            }
                            for w in seg.get("words", [])
                        ]
                    segments_list.append(item)
                    # Approximate progress for OpenAI Whisper by segment count
                    self.log_ui("progress", min(0.99, idx / total))
                self.log_ui("progress", 1.0)
            return segments_list, detected_lang
        except Exception as e:
            logger.exception("Transcription failed for %s: %s", in_path, e)
            raise RuntimeError(f"Transcription failed: {e}")

    # ----------------- main worker loop -----------------
    def run(self):
        engine = self.settings["engine"]
        model_name = self.settings["model"]
        device_pref = self.settings["device"]
        compute = self.settings["compute"]
        vad_filter = self.settings["vad_filter"]
        word_ts = self.settings["word_timestamps"]
        language = self.settings["language"]
        formats = self.settings["formats"]
        summarize_cfg = self.settings.get("summarize", {})

        out_root = self.settings.get("last_output", OUT_DEFAULT)
        os.makedirs(out_root, exist_ok=True)

        model = None
        fast_model = None
        try:
            if engine == "faster-whisper":
                from faster_whisper import WhisperModel
                device = "cuda" if (device_pref in ("auto", "gpu") and self._cuda_available()) else "cpu"
                compute_type = self._choose_compute(device, compute)
                logger.info("Loading Faster-Whisper model=%s device=%s compute=%s", model_name, device, compute_type)
                fast_model = WhisperModel(model_name, device=device, compute_type=compute_type)
            else:
                import whisper
                device = "cuda" if (device_pref in ("auto", "gpu") and self._cuda_available()) else "cpu"
                logger.info("Loading Whisper model=%s device=%s", model_name, device)
                model = whisper.load_model(model_name, device=device)
        except Exception as e:
            logger.exception("Model load failed: %s", e)
            self.log_ui("fatal", f"Model load failed: {e}")
            return

        for job in self.jobs:
            if self.cancel_event.is_set():
                job.status = "Canceled"
                self.log_ui("job_update", job)
                break
            try:
                job.status = "Working"
                self.log_ui("job_update", job)
                # Preprocess/measure duration
                try:
                    audio = AudioSegment.from_file(job.path)
                    job.duration = audio.duration_seconds
                except Exception as e:
                    raise RuntimeError(f"Failed to load audio: {e}")
                self.log_ui("job_update", job)

                tmp_wav = None
                try:
                    if engine == "whisper":
                        tmp_wav = self._ensure_wav_mono16(job.path)
                        in_path = tmp_wav
                    else:
                        in_path = job.path

                    segments, language_out = self._transcribe(
                        engine, model, fast_model, in_path, language, vad_filter, word_ts
                    )
                    job.progress = 1.0

                    outdir = self._make_outdir(out_root, job.path)
                    job.outdir = outdir
                    os.makedirs(outdir, exist_ok=True)
                    base = os.path.splitext(os.path.basename(job.path))[0]

                    # Compose full text once to reuse for TXT + summarization
                    full_text = " ".join(s["text"] for s in segments).strip()

                    if formats.get("txt"):
                        path = os.path.join(outdir, f"{base}.txt")
                        with open(path, "w", encoding="utf-8") as f:
                            f.write(full_text + "\n")
                        job.outputs.append(path)
                    if formats.get("srt"):
                        path = os.path.join(outdir, f"{base}.srt")
                        with open(path, "w", encoding="utf-8") as f:
                            for i, s in enumerate(segments, 1):
                                f.write(f"{i}\n{ts_to_srt(s['start'])} --> {ts_to_srt(s['end'])}\n{s['text'].strip()}\n\n")
                        job.outputs.append(path)
                    if formats.get("vtt"):
                        path = os.path.join(outdir, f"{base}.vtt")
                        with open(path, "w", encoding="utf-8") as f:
                            f.write("WEBVTT\n\n")
                            for s in segments:
                                f.write(f"{ts_to_srt(s['start']).replace(',', '.')} --> {ts_to_srt(s['end']).replace(',', '.')}\n{s['text'].strip()}\n\n")
                        job.outputs.append(path)
                    if formats.get("json"):
                        path = os.path.join(outdir, f"{base}.json")
                        with open(path, "w", encoding="utf-8") as f:
                            json.dump({"language": language_out, "segments": segments}, f, ensure_ascii=False, indent=2)
                        job.outputs.append(path)
                    if formats.get("csv"):
                        path = os.path.join(outdir, f"{base}.csv")
                        with open(path, "w", newline="", encoding="utf-8") as f:
                            w = csv.writer(f)
                            w.writerow(["start", "end", "text"])
                            for s in segments:
                                w.writerow([f"{s['start']:.2f}", f"{s['end']:.2f}", s['text']])
                        job.outputs.append(path)

                    # Optional OpenAI summarization → Markdown (+ optional HTML/PDF)
                    if summarize_cfg.get("enabled"):
                        try:
                            self.log_ui("progress", 0.01)
                            self.log_ui("job_update", job)
                            summary_md = self._summarize_text(full_text, {**self.settings, "summarize": summarize_cfg})
                            spath_md = os.path.join(outdir, f"{base}_summary.md")
                            with open(spath_md, "w", encoding="utf-8") as f:
                                f.write(summary_md.strip() + "\n")
                            job.outputs.append(spath_md)

                            # HTML export
                            if summarize_cfg.get("html"):
                                html_str = self._md_to_html(summary_md, title=f"{base} — Meeting Notes")
                                spath_html = os.path.join(outdir, f"{base}_summary.html")
                                with open(spath_html, "w", encoding="utf-8") as f:
                                    f.write(html_str)
                                job.outputs.append(spath_html)

                            # PDF export
                            if summarize_cfg.get("pdf"):
                                try:
                                    # Prefer HTML conversion for richer styling
                                    html_str = self._md_to_html(summary_md, title=f"{base} — Meeting Notes")
                                    spath_pdf = os.path.join(outdir, f"{base}_summary.pdf")
                                    self._write_pdf_from_html(html_str, spath_pdf)
                                    job.outputs.append(spath_pdf)
                                except Exception as e:
                                    logger.exception("PDF export failed for %s: %s", job.path, e)
                        except Exception as e:
                            logger.exception("Summarization failed for %s: %s", job.path, e)
                            # Do not mark job as Error; continue with transcripts

                    job.status = "Done"
                    self.log_ui("job_update", job)
                finally:
                    if tmp_wav and os.path.exists(tmp_wav):
                        try:
                            os.remove(tmp_wav)
                        except Exception:
                            pass
            except RuntimeError as e:
                if str(e) == "Canceled":
                    job.status = "Canceled"
                    job.error = None
                else:
                    job.status = "Error"
                    job.error = str(e)
                    logger.exception("Job failed: %s", e)
                self.log_ui("job_update", job)
            except Exception as e:
                job.status = "Error"
                job.error = str(e)
                logger.exception("Job failed: %s", e)
                self.log_ui("job_update", job)
        self.log_ui("all_done", None)

# ----------------------------- UI ------------------------------------------

class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title(APP_NAME)
        self.geometry("1080x740")
        self.minsize(960, 620)
        self._cfg = load_cfg()
        self._jobs: List[Job] = []
        self._worker: Optional[TranscribeWorker] = None
        self._ui_queue: queue.Queue = queue.Queue()
        self._cancel = threading.Event()

        self._style = ttk.Style(self)
        self._apply_theme(self._cfg.get("dark", True))

        self._build_menu()
        self._build_toolbar()
        self._build_queue_panel()
        self._build_settings_panel()
        self._build_statusbar()

        # Output dir
        out = self._cfg.get("last_output", OUT_DEFAULT)
        os.makedirs(out, exist_ok=True)
        self.out_dir_var.set(out)

        # UI queue polling
        self.after(100, self._drain_ui_queue)

    # ---------- UI Builders ----------
    def _apply_theme(self, dark: bool):
        self._cfg["dark"] = dark
        theme = "clam"
        self._style.theme_use(theme)
        if dark:
            bg = "#1e1f22"; fg = "#e6e6e6"; acc = "#3a3d41"
        else:
            bg = "#f7f7f7"; fg = "#222"; acc = "#e6e6e6"
        self.configure(bg=bg)
        for w in ["TFrame", "TLabelframe", "TLabelframe.Label", "TLabel", "TNotebook", "TNotebook.Tab", "TButton", "TCheckbutton", "TRadiobutton", "TMenubutton", "Treeview", "TProgressbar"]:
            self._style.configure(w, background=bg, foreground=fg, fieldbackground=bg)
        self._style.map("TButton", background=[("active", acc)])

    def _build_menu(self):
        m = tk.Menu(self)
        filem = tk.Menu(m, tearoff=0)
        filem.add_command(label="Add Audio Files…", command=self.add_files)
        filem.add_command(label="Choose Output Folder…", command=self.choose_out)
        filem.add_separator()
        filem.add_command(label="Open Output Folder", command=self.open_out)
        filem.add_separator()
        filem.add_command(label="Exit", command=self.destroy)
        m.add_cascade(label="File", menu=filem)

        viewm = tk.Menu(m, tearoff=0)
        viewm.add_command(label="Toggle Dark Mode", command=self.toggle_dark)
        m.add_cascade(label="View", menu=viewm)

        helpm = tk.Menu(m, tearoff=0)
        helpm.add_command(label="Open Log", command=lambda: self._open_path(LOG_PATH))
        helpm.add_command(label="About", command=self._about)
        m.add_cascade(label="Help", menu=helpm)

        self.config(menu=m)

    def _build_toolbar(self):
        bar = ttk.Frame(self)
        bar.pack(fill="x", padx=8, pady=6)
        ttk.Button(bar, text="Add Files", command=self.add_files).pack(side="left")
        ttk.Button(bar, text="Remove Selected", command=self.remove_selected).pack(side="left", padx=6)
        ttk.Separator(bar, orient="vertical").pack(side="left", fill="y", padx=6)
        self.start_btn = ttk.Button(bar, text="Start", command=self.start)
        self.start_btn.pack(side="left")
        self.stop_btn = ttk.Button(bar, text="Stop", command=self.stop, state="disabled")
        self.stop_btn.pack(side="left", padx=6)
        ttk.Separator(bar, orient="vertical").pack(side="left", fill="y", padx=6)
        ttk.Button(bar, text="Open Output", command=self.open_out).pack(side="left")
        ttk.Label(bar, text="Output:").pack(side="left", padx=(12,4))
        self.out_dir_var = tk.StringVar()
        out_entry = ttk.Entry(bar, textvariable=self.out_dir_var, width=56)
        out_entry.pack(side="left")
        ttk.Button(bar, text="…", width=3, command=self.choose_out).pack(side="left", padx=(4,0))

    def _build_queue_panel(self):
        frm = ttk.Frame(self)
        frm.pack(fill="both", expand=True, padx=8)
        cols = ("file", "status", "dur", "prog")
        self.tree = ttk.Treeview(frm, columns=cols, show="headings", selectmode="extended")
        self.tree.heading("file", text="File")
        self.tree.heading("status", text="Status")
        self.tree.heading("dur", text="Duration")
        self.tree.heading("prog", text="Progress")
        self.tree.column("file", width=620)
        self.tree.column("status", width=120, anchor="center")
        self.tree.column("dur", width=90, anchor="center")
        self.tree.column("prog", width=120, anchor="center")
        self.tree.pack(fill="both", expand=True, side="left")

        vsb = ttk.Scrollbar(frm, orient="vertical", command=self.tree.yview)
        self.tree.configure(yscroll=vsb.set)
        vsb.pack(side="right", fill="y")

    def _build_settings_panel(self):
        nb = ttk.Notebook(self)
        nb.pack(fill="x", padx=8, pady=6)

        # Engine tab
        t1 = ttk.Frame(nb); nb.add(t1, text="Engine")
        self.engine_var = tk.StringVar(value=self._cfg.get("engine", "faster-whisper"))
        ttk.Radiobutton(t1, text="Faster‑Whisper (recommended)", variable=self.engine_var, value="faster-whisper").grid(row=0, column=0, sticky="w", padx=6, pady=4)
        ttk.Radiobutton(t1, text="OpenAI Whisper (classic)", variable=self.engine_var, value="whisper").grid(row=1, column=0, sticky="w", padx=6, pady=4)

        ttk.Label(t1, text="Model size").grid(row=0, column=1, sticky="e", padx=6)
        self.model_var = tk.StringVar(value=self._cfg.get("model", "base"))
        ttk.Combobox(t1, textvariable=self.model_var, values=["tiny","base","small","medium","large-v3"], width=12, state="readonly").grid(row=0, column=2, padx=6)

        ttk.Label(t1, text="Device").grid(row=1, column=1, sticky="e", padx=6)
        self.device_var = tk.StringVar(value=self._cfg.get("device", "auto"))
        ttk.Combobox(t1, textvariable=self.device_var, values=["auto","cpu","gpu"], width=10, state="readonly").grid(row=1, column=2, padx=6)

        ttk.Label(t1, text="Compute type (FW)").grid(row=2, column=1, sticky="e", padx=6)
        self.compute_var = tk.StringVar(value=self._cfg.get("compute", "auto"))
        ttk.Combobox(t1, textvariable=self.compute_var, values=["auto","float16","int8","int8_float16"], width=12, state="readonly").grid(row=2, column=2, padx=6)

        # Options tab
        t2 = ttk.Frame(nb); nb.add(t2, text="Options")
        self.vad_var = tk.BooleanVar(value=self._cfg.get("vad_filter", True))
        self.words_var = tk.BooleanVar(value=self._cfg.get("word_timestamps", False))
        ttk.Checkbutton(t2, text="VAD filter (reduce silence)", variable=self.vad_var).grid(row=0, column=0, sticky="w", padx=6, pady=4)
        ttk.Checkbutton(t2, text="Word timestamps (slower, detailed)", variable=self.words_var).grid(row=1, column=0, sticky="w", padx=6, pady=4)

        ttk.Label(t2, text="Language").grid(row=0, column=1, sticky="e", padx=6)
        self.lang_var = tk.StringVar(value=self._cfg.get("language", "auto"))
        ttk.Combobox(t2, textvariable=self.lang_var, values=["auto","en","es","fr","de","pt","it","ru","zh","ja","ko"], width=12, state="readonly").grid(row=0, column=2, padx=6)

        # Formats tab
        t3 = ttk.Frame(nb); nb.add(t3, text="Formats")
        self.f_txt = tk.BooleanVar(value=self._cfg.get("formats",{}).get("txt", True))
        self.f_srt = tk.BooleanVar(value=self._cfg.get("formats",{}).get("srt", True))
        self.f_vtt = tk.BooleanVar(value=self._cfg.get("formats",{}).get("vtt", False))
        self.f_json= tk.BooleanVar(value=self._cfg.get("formats",{}).get("json", False))
        self.f_csv = tk.BooleanVar(value=self._cfg.get("formats",{}).get("csv", False))
        ttk.Checkbutton(t3, text=".txt", variable=self.f_txt).grid(row=0, column=0, sticky="w", padx=6, pady=4)
        ttk.Checkbutton(t3, text=".srt", variable=self.f_srt).grid(row=0, column=1, sticky="w", padx=6, pady=4)
        ttk.Checkbutton(t3, text=".vtt", variable=self.f_vtt).grid(row=0, column=2, sticky="w", padx=6, pady=4)
        ttk.Checkbutton(t3, text=".json (segments)", variable=self.f_json).grid(row=0, column=3, sticky="w", padx=6, pady=4)
        ttk.Checkbutton(t3, text=".csv (segments)", variable=self.f_csv).grid(row=0, column=4, sticky="w", padx=6, pady=4)

        # Summarize tab
        t4 = ttk.Frame(nb); nb.add(t4, text="Meeting Distiller")
        self.sum_enable = tk.BooleanVar(value=self._cfg.get("summarize",{}).get("enabled", False))
        ttk.Checkbutton(t4, text="Generate meeting notes with OpenAI", variable=self.sum_enable).grid(row=0, column=0, sticky="w", padx=6, pady=6, columnspan=4)

        ttk.Label(t4, text="Model").grid(row=1, column=0, sticky="e", padx=6)
        self.sum_model = tk.StringVar(value=self._cfg.get("summarize",{}).get("model", "gpt-4o-mini"))
        ttk.Combobox(t4, textvariable=self.sum_model, values=["gpt-4o-mini","gpt-4o"], width=16, state="readonly").grid(row=1, column=1, padx=6, sticky="w")

        ttk.Label(t4, text="Max output tokens").grid(row=1, column=2, sticky="e", padx=6)
        self.sum_max_tokens = tk.IntVar(value=int(self._cfg.get("summarize",{}).get("max_output_tokens", 2000)))
        ttk.Entry(t4, textvariable=self.sum_max_tokens, width=8).grid(row=1, column=3, padx=6, sticky="w")

        ttk.Label(t4, text="Chunk size (chars)").grid(row=1, column=4, sticky="e", padx=6)
        self.sum_chunk_chars = tk.IntVar(value=int(self._cfg.get("summarize",{}).get("chunk_chars", 32000)))
        ttk.Entry(t4, textvariable=self.sum_chunk_chars, width=10).grid(row=1, column=5, padx=6, sticky="w")

        ttk.Label(t4, text="OpenAI API Key").grid(row=2, column=0, sticky="e", padx=6)
        self.api_key_var = tk.StringVar(value=(self._cfg.get("summarize",{}).get("api_key", "") if self._cfg.get("summarize",{}).get("remember_key") else ""))
        ttk.Entry(t4, textvariable=self.api_key_var, width=48, show="*").grid(row=2, column=1, columnspan=3, sticky="w", padx=6)
        self.sum_remember = tk.BooleanVar(value=bool(self._cfg.get("summarize",{}).get("remember_key", False)))
        ttk.Checkbutton(t4, text="Remember key (stored in plain text config)", variable=self.sum_remember).grid(row=2, column=4, columnspan=2, sticky="w", padx=6)

        # Output types for Meeting Distiller
        self.sum_html = tk.BooleanVar(value=self._cfg.get("summarize",{}).get("html", False))
        self.sum_pdf  = tk.BooleanVar(value=self._cfg.get("summarize",{}).get("pdf", False))
        ttk.Checkbutton(t4, text="Also save HTML (.html)", variable=self.sum_html).grid(row=3, column=0, sticky="w", padx=6, pady=(4,6))
        ttk.Checkbutton(t4, text="Also save PDF (.pdf)",  variable=self.sum_pdf ).grid(row=3, column=1, sticky="w", padx=6, pady=(4,6))

        ttk.Label(t4, text="Prompt").grid(row=4, column=0, sticky="ne", padx=6, pady=(8,6))
        self.prompt_text = tk.Text(t4, width=120, height=10, wrap="word")
        self.prompt_text.grid(row=4, column=1, columnspan=5, sticky="we", padx=6, pady=(8,6))
        self.prompt_text.insert("1.0", self._cfg.get("summarize",{}).get("prompt", ""))

    def _build_statusbar(self):
        bar = ttk.Frame(self)
        bar.pack(fill="x", side="bottom")
        self.pb = ttk.Progressbar(bar, orient="horizontal", mode="determinate")
        self.pb.pack(fill="x", expand=True, padx=8, pady=6)
        self.status_var = tk.StringVar(value="Ready")
        ttk.Label(bar, textvariable=self.status_var).pack(side="right", padx=8)

    # ---------- Actions ----------
    def toggle_dark(self):
        self._apply_theme(not self._cfg.get("dark", True))
        save_cfg(self._cfg)

    def add_files(self):
        paths = filedialog.askopenfilenames(title="Select audio files", filetypes=[("Audio", "*.wav *.mp3 *.m4a *.mp4 *.aac *.flac *.ogg")])
        if not paths: return
        for p in paths:
            if p in (self.tree.get_children()):
                continue
            job = Job(path=p)
            self._jobs.append(job)
            self.tree.insert("", "end", iid=p, values=(os.path.basename(p), job.status, "—", "0%"))

    def remove_selected(self):
        for iid in self.tree.selection():
            self.tree.delete(iid)
            self._jobs = [j for j in self._jobs if j.path != iid]

    def choose_out(self):
        d = filedialog.askdirectory(initialdir=self.out_dir_var.get() or OUT_DEFAULT)
        if not d: return
        self.out_dir_var.set(d)
        self._cfg["last_output"] = d
        save_cfg(self._cfg)

    def open_out(self):
        self._open_path(self.out_dir_var.get() or OUT_DEFAULT)

    def _open_path(self, path: str):
        try:
            if sys.platform.startswith("win"):
                os.startfile(path)
            elif sys.platform == "darwin":
                os.system(f"open '{path}'")
            else:
                os.system(f"xdg-open '{path}'")
        except Exception as e:
            messagebox.showerror("Open", str(e))

    def start(self):
        if self._worker and self._worker.is_alive():
            return
        if not self._jobs:
            messagebox.showwarning("No files", "Add one or more audio files to the queue.")
            return
        # Update config from UI
        self._cfg.update({
            "engine": self.engine_var.get(),
            "model": self.model_var.get(),
            "device": self.device_var.get(),
            "compute": self.compute_var.get(),
            "vad_filter": bool(self.vad_var.get()),
            "word_timestamps": bool(self.words_var.get()),
            "language": self.lang_var.get(),
            "formats": {"txt": bool(self.f_txt.get()), "srt": bool(self.f_srt.get()), "vtt": bool(self.f_vtt.get()), "json": bool(self.f_json.get()), "csv": bool(self.f_csv.get())},
            "last_output": self.out_dir_var.get() or OUT_DEFAULT,
        })
        self._cfg["summarize"]= {
            "enabled": bool(self.sum_enable.get()),
            "model": self.sum_model.get(),
            "max_output_tokens": int(self.sum_max_tokens.get()),
            "chunk_chars": int(self.sum_chunk_chars.get()),
            "remember_key": bool(self.sum_remember.get()),
            "prompt": self.prompt_text.get("1.0", "end-1c"),
            "html": bool(self.sum_html.get()),
            "pdf": bool(self.sum_pdf.get()),
        }
        if self.sum_remember.get():
            # store key in config only if user explicitly allows; warn it is plain text
            self._cfg["summarize"]["api_key"] = self.api_key_var.get()
        else:
            self._cfg.get("summarize", {}).pop("api_key", None)
        save_cfg(self._cfg)

        # Reset UI
        for j in self._jobs:
            self.tree.set(j.path, "status", "Queued")
            self.tree.set(j.path, "dur", "—")
            self.tree.set(j.path, "prog", "0%")
        self.pb["value"] = 0
        self.status_var.set("Loading model…")
        self._cancel.clear()

        # Start worker
        settings = dict(self._cfg)
        # pass API key for this session (not persisted unless remember=true)
        settings["openai_api_key"] = self.api_key_var.get().strip() or self._cfg.get("summarize", {}).get("api_key") or os.getenv("OPENAI_API_KEY")
        self._worker = TranscribeWorker(self._jobs, settings, self._ui_queue, self._cancel)
        self._worker.start()
        self.start_btn.configure(state="disabled")
        self.stop_btn.configure(state="normal")

    def stop(self):
        if self._worker and self._worker.is_alive():
            self._cancel.set()
            self.status_var.set("Canceling…")

    # ---------- UI queue pump ----------
    def _drain_ui_queue(self):
        try:
            while True:
                kind, payload = self._ui_queue.get_nowait()
                if kind == "fatal":
                    self.status_var.set(str(payload))
                    self.start_btn.configure(state="normal")
                    self.stop_btn.configure(state="disabled")
                elif kind == "job_update":
                    j: Job = payload
                    if j.duration:
                        self.tree.set(j.path, "dur", human_time(j.duration))
                    self.tree.set(j.path, "status", j.status)
                    if j.status == "Done":
                        self.tree.set(j.path, "prog", "100%")
                    elif j.status in ("Error", "Canceled"):
                        self.tree.set(j.path, "prog", "—")
                        if j.error:
                            logger.error("Job error for %s: %s", j.path, j.error)
                elif kind == "progress":
                    frac = float(payload)
                    self.pb["value"] = int(frac * 100)
                    # Update the single active job's progress text
                    # (Find first item not Done/Error)
                    for j in self._jobs:
                        st = self.tree.set(j.path, "status")
                        if st in ("Working", "Queued"):
                            self.tree.set(j.path, "prog", f"{int(frac*100)}%")
                            break
                elif kind == "all_done":
                    self.status_var.set("Ready")
                    self.start_btn.configure(state="normal")
                    self.stop_btn.configure(state="disabled")
        except queue.Empty:
            pass
        self.after(100, self._drain_ui_queue)

    def _about(self):
        messagebox.showinfo(APP_NAME, f"{APP_NAME}\nA streamlined Tkinter frontend for Faster‑Whisper/Whisper.\nLogs: {LOG_PATH}")

# ----------------------------- Self‑tests ----------------------------------

def run_self_tests() -> int:
    """Basic smoke tests for timestamp formatting, chunking, and exporters (no model/API needed)."""
    failures = 0

    # ts_to_srt
    if ts_to_srt(1.234) != "00:00:01,234":
        logger.error("ts_to_srt failed: %s", ts_to_srt(1.234))
        failures += 1

    # chunk splitter
    long_text = ("para1\n\n" + "word "*10000).strip()
    # 2000 chars should force multiple chunks
    def _chunks(txt):
        w = TranscribeWorker([], {}, queue.Queue(), threading.Event())
        return w._split_into_chunks(txt, 2000)
    chunks = _chunks(long_text)
    if not chunks or any(len(c) > 2000 for c in chunks):
        logger.error("Chunk split failed: lengths=%s", [len(c) for c in chunks])
        failures += 1

    # HTML conversion test
    w = TranscribeWorker([], {}, queue.Queue(), threading.Event())
    html = w._md_to_html("# Title\n\n**Bold** text.")
    if "<html" not in html or "Bold" not in html:
        logger.error("HTML export failed: %r", html[:80])
        failures += 1

    # fake segments
    segments = [
        {"start": 0.0, "end": 1.0, "text": "Hello"},
        {"start": 1.0, "end": 2.5, "text": "world!"},
    ]

    with tempfile.TemporaryDirectory() as d:
        base = os.path.join(d, "test")
        # TXT
        p_txt = base + ".txt"
        with open(p_txt, "w", encoding="utf-8") as f:
            f.write(" ".join(s["text"] for s in segments).strip() + "\n")
        with open(p_txt, "r", encoding="utf-8") as f:
            content = f.read()
            if not content.endswith("\n") or "Hello world!" not in content:
                logger.error("TXT export failed: %r", content)
                failures += 1
        # SRT
        p_srt = base + ".srt"
        with open(p_srt, "w", encoding="utf-8") as f:
            for i, s in enumerate(segments, 1):
                f.write(f"{i}\n{ts_to_srt(s['start'])} --> {ts_to_srt(s['end'])}\n{s['text'].strip()}\n\n")
        with open(p_srt, "r", encoding="utf-8") as f:
            srt_text = f.read()
            if srt_text.count("--> ") != 2 or "\n\n" not in srt_text:
                logger.error("SRT export failed: %r", srt_text)
                failures += 1
        # VTT
        p_vtt = base + ".vtt"
        with open(p_vtt, "w", encoding="utf-8") as f:
            f.write("WEBVTT\n\n")
            for s in segments:
                f.write(f"{ts_to_srt(s['start']).replace(',', '.')} --> {ts_to_srt(s['end']).replace(',', '.')}\n{s['text'].strip()}\n\n")
        with open(p_vtt, "r", encoding="utf-8") as f:
            vtt_text = f.read()
            if not vtt_text.startswith("WEBVTT\n\n") or vtt_text.count("--> ") != 2:
                logger.error("VTT export failed: %r", vtt_text)
                failures += 1
        # CSV
        p_csv = base + ".csv"
        with open(p_csv, "w", newline="", encoding="utf-8") as f:
            wcsv = csv.writer(f)
            wcsv.writerow(["start", "end", "text"])
            for s in segments:
                wcsv.writerow([f"{s['start']:.2f}", f"{s['end']:.2f}", s['text']])
        with open(p_csv, "r", encoding="utf-8") as f:
            rows = f.read().strip().splitlines()
            if len(rows) != 3 or rows[0] != "start,end,text":
                logger.error("CSV export failed: %r", rows)
                failures += 1

    return failures

# ----------------------------- main ----------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=APP_NAME)
    parser.add_argument("--self-test", action="store_true", help="Run quick exporter tests and exit")
    args, _ = parser.parse_known_args()

    if args.self_test:
        rc = run_self_tests()
        print(f"Self-tests {'PASSED' if rc == 0 else 'FAILED'} (failures={rc})")
        sys.exit(0 if rc == 0 else 1)

    try:
        app = App()
        app.mainloop()
    except Exception as e:
        logger.exception("Fatal error: %s", e)
        raise
