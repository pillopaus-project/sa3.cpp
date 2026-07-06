#!/usr/bin/env python3
import argparse
import json
import os
import re
import shutil
import subprocess
import tempfile
from dataclasses import dataclass
from datetime import datetime, timezone
from difflib import SequenceMatcher
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SONG_BOOK = Path.home() / "repos" / "bako" / "song-book"
DATASET = DEFAULT_SONG_BOOK / "datasets" / "mnesia-audio-v1"
DEFAULT_OUTPUT = DEFAULT_SONG_BOOK / "data" / "lyric-timesheets" / "mnesia-audio-v1"
DEFAULT_WHISPER_REPO = Path.home() / "repos" / "inference" / "speech2text"
DEFAULT_WHISPER_CLI = DEFAULT_WHISPER_REPO / "backend" / "whisper-cli"
DEFAULT_WHISPER_MODEL = DEFAULT_WHISPER_REPO / "backend" / "ggml-base.en.bin"
DEFAULT_LIBRARY_PATH = ":".join(
    [
        str(DEFAULT_WHISPER_REPO / "backend" / "whisper.cpp" / "build" / "src"),
        str(DEFAULT_WHISPER_REPO / "backend" / "whisper.cpp" / "build" / "ggml" / "src"),
    ]
)


SECTION_RE = re.compile(r"^##\s*(?:\[(?P<bracket>[^\]]+)\]|(?P<plain>.+?))\s*$")
TITLE_RE = re.compile(r"^#\s+(.+?)\s*$")
WORD_RE = re.compile(r"[a-z0-9]+")
SPECIAL_TOKEN_RE = re.compile(r"^\[_.*?\]$")


@dataclass
class LyricSection:
    index: int
    label: str
    lines: list[str]

    @property
    def text(self) -> str:
        return "\n".join(self.lines).strip()

    @property
    def tokens(self) -> list[str]:
        return normalize_words(self.text)


@dataclass
class LyricWord:
    index: int
    line_index: int
    word_index: int
    text: str
    normalized: str


def normalize_words(text: str) -> list[str]:
    text = text.lower().replace("'", "")
    return WORD_RE.findall(text)


def parse_lyrics(path: Path) -> tuple[str | None, list[LyricSection]]:
    title = None
    sections: list[LyricSection] = []
    current_label = "unsectioned"
    current_lines: list[str] = []

    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.rstrip()
        title_match = TITLE_RE.match(line)
        if title_match and title is None:
            title = title_match.group(1).strip()
            continue

        section_match = SECTION_RE.match(line)
        if section_match:
            if any(item.strip() for item in current_lines):
                sections.append(
                    LyricSection(
                        index=len(sections),
                        label=current_label,
                        lines=current_lines,
                    )
                )
            current_label = (section_match.group("bracket") or section_match.group("plain") or "").strip()
            current_lines = []
            continue

        if line.startswith("#"):
            continue

        current_lines.append(line)

    if any(item.strip() for item in current_lines):
        sections.append(
            LyricSection(
                index=len(sections),
                label=current_label,
                lines=current_lines,
            )
        )

    return title, sections


def seconds_from_offsets(item: dict, key: str) -> float | None:
    offsets = item.get("offsets")
    if not isinstance(offsets, dict):
        return None
    value = offsets.get(key)
    if value is None:
        return None
    return round(float(value) / 1000.0, 3)


def clean_word_text(text: str) -> str:
    stripped = text.strip()
    if not stripped or SPECIAL_TOKEN_RE.match(stripped):
        return ""
    return stripped.strip("♪[](){}<>\"“”‘’.,!?;:")


def lyric_words(section: LyricSection) -> list[LyricWord]:
    words: list[LyricWord] = []
    for line_index, line in enumerate(section.lines):
        word_index = 0
        for match in WORD_RE.finditer(line):
            text = match.group(0)
            words.append(
                LyricWord(
                    index=len(words),
                    line_index=line_index,
                    word_index=word_index,
                    text=text,
                    normalized=normalize_words(text)[0],
                )
            )
            word_index += 1
    return words


def parse_whisper_json(path: Path) -> tuple[list[dict], list[dict]]:
    data = json.loads(path.read_text(encoding="utf-8", errors="replace"))
    transcription = data.get("transcription") or []
    segments: list[dict] = []
    words: list[dict] = []

    for segment_index, segment in enumerate(transcription):
        text = (segment.get("text") or "").strip()
        segments.append(
            {
                "index": segment_index,
                "start_time": seconds_from_offsets(segment, "from"),
                "end_time": seconds_from_offsets(segment, "to"),
                "text": text,
            }
        )

        for token in segment.get("tokens") or []:
            cleaned = clean_word_text(token.get("text") or "")
            normalized = normalize_words(cleaned)
            if not normalized:
                continue
            words.append(
                {
                    "index": len(words),
                    "segment_index": segment_index,
                    "start_time": seconds_from_offsets(token, "from"),
                    "end_time": seconds_from_offsets(token, "to"),
                    "text": cleaned,
                    "normalized": normalized[0],
                    "confidence": token.get("p"),
                }
            )

    return segments, words


def best_local_window(lyric_tokens: list[str], transcript: list[str], cursor: int) -> tuple[int | None, int | None, float]:
    if not lyric_tokens or cursor >= len(transcript):
        return None, None, 0.0

    lyric_len = len(lyric_tokens)
    min_span = max(1, lyric_len // 2)
    max_span = max(lyric_len + 1, lyric_len * 2)
    best_start = None
    best_end = None
    best_score = 0.0
    best_weighted_score = 0.0
    lyric_set = set(lyric_tokens)

    for start in range(cursor, len(transcript)):
        if transcript[start] not in lyric_set:
            continue
        end_floor = min(len(transcript), start + min_span)
        end_ceiling = min(len(transcript), start + max_span)
        for end in range(end_floor, end_ceiling + 1):
            score = SequenceMatcher(None, transcript[start:end], lyric_tokens, autojunk=False).ratio()
            distance = max(0, start - cursor)
            weighted_score = score / (1.0 + (distance / max(1, lyric_len)))
            if weighted_score > best_weighted_score:
                best_start = start
                best_end = end
                best_score = score
                best_weighted_score = weighted_score

    return best_start, best_end, best_score


def distribute_section_times(
    section_words: list[LyricWord],
    transcript_words: list[dict],
    window_start: int | None,
    window_end: int | None,
) -> list[dict]:
    timed: list[dict] = []
    if window_start is None or window_end is None or not section_words:
        for lyric_word in section_words:
            timed.append(
                {
                    "index": lyric_word.index,
                    "line_index": lyric_word.line_index,
                    "word_index": lyric_word.word_index,
                    "text": lyric_word.text,
                    "normalized": lyric_word.normalized,
                    "start_time": None,
                    "end_time": None,
                    "source": "lyric_sheet",
                    "timing_source": "unmatched",
                }
            )
        return timed

    window_words = transcript_words[window_start:window_end]
    transcript_tokens = [word["normalized"] for word in window_words]
    lyric_tokens = [word.normalized for word in section_words]
    matcher = SequenceMatcher(None, transcript_tokens, lyric_tokens, autojunk=False)
    matched_times: dict[int, tuple[float | None, float | None]] = {}

    for opcode, transcript_a, _transcript_b, lyric_a, lyric_b in matcher.get_opcodes():
        if opcode != "equal":
            continue
        for offset, lyric_index in enumerate(range(lyric_a, lyric_b)):
            transcript_index = transcript_a + offset
            source_word = window_words[transcript_index]
            matched_times[lyric_index] = (source_word["start_time"], source_word["end_time"])

    known_indexes = sorted(matched_times)
    for lyric_word in section_words:
        timing_source = "matched"
        start_time = None
        end_time = None

        if lyric_word.index in matched_times:
            start_time, end_time = matched_times[lyric_word.index]
        elif window_words:
            timing_source = "interpolated"
            left = [idx for idx in known_indexes if idx < lyric_word.index]
            right = [idx for idx in known_indexes if idx > lyric_word.index]
            left_idx = left[-1] if left else None
            right_idx = right[0] if right else None

            if left_idx is not None and right_idx is not None:
                left_time = matched_times[left_idx][1]
                right_time = matched_times[right_idx][0]
                if left_time is not None and right_time is not None and right_idx != left_idx:
                    fraction = (lyric_word.index - left_idx) / (right_idx - left_idx)
                    start_time = round(left_time + (right_time - left_time) * fraction, 3)
                    end_time = start_time
            elif left_idx is not None:
                start_time = matched_times[left_idx][1]
                end_time = start_time
            elif right_idx is not None:
                start_time = matched_times[right_idx][0]
                end_time = start_time

        timed.append(
            {
                "index": lyric_word.index,
                "line_index": lyric_word.line_index,
                "word_index": lyric_word.word_index,
                "text": lyric_word.text,
                "normalized": lyric_word.normalized,
                "start_time": start_time,
                "end_time": end_time,
                "source": "lyric_sheet",
                "timing_source": timing_source,
            }
        )

    return timed


def align_sections(sections: list[LyricSection], words: list[dict]) -> list[dict]:
    transcript = [word["normalized"] for word in words]
    cursor = 0
    aligned: list[dict] = []

    for section in sections:
        words_from_sheet = lyric_words(section)
        lyric_tokens = [word.normalized for word in words_from_sheet]
        result = {
            "index": section.index,
            "label": section.label,
            "start_time": None,
            "end_time": None,
            "alignment_confidence": 0.0,
            "matched_word_start": None,
            "matched_word_end": None,
            "text": section.text,
            "lines": section.lines,
            "timed_words": distribute_section_times(words_from_sheet, words, None, None),
        }

        if not lyric_tokens or cursor >= len(transcript):
            aligned.append(result)
            continue

        start, end_exclusive, score = best_local_window(lyric_tokens, transcript, cursor)

        if start is not None and end_exclusive is not None:
            end = end_exclusive - 1
            result.update(
                {
                    "start_time": words[start]["start_time"],
                    "end_time": words[end]["end_time"],
                    "alignment_confidence": round(score, 3),
                    "matched_word_start": start,
                    "matched_word_end": end,
                    "timed_words": distribute_section_times(words_from_sheet, words, start, end_exclusive),
                }
            )
            cursor = end + 1

        aligned.append(result)

    return aligned


def load_metadata() -> dict[str, dict]:
    metadata_path = DATASET / "metadata.jsonl"
    if not metadata_path.exists():
        return {}
    metadata: dict[str, dict] = {}
    for line in metadata_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        item = json.loads(line)
        metadata[item["id"]] = item
    return metadata


def whisper_environment(existing_env: dict[str, str]) -> dict[str, str]:
    env = dict(existing_env)
    current = env.get("LD_LIBRARY_PATH")
    env["LD_LIBRARY_PATH"] = DEFAULT_LIBRARY_PATH if not current else f"{DEFAULT_LIBRARY_PATH}:{current}"
    return env


def run_whisper(audio_path: Path, whisper_cli: Path, model_path: Path, work_dir: Path) -> Path:
    output_prefix = work_dir / audio_path.stem
    subprocess.run(
        [
            str(whisper_cli),
            "-m",
            str(model_path),
            "-f",
            str(audio_path),
            "-l",
            "en",
            "-oj",
            "-ojf",
            "-of",
            str(output_prefix),
        ],
        cwd=ROOT,
        env=whisper_environment(os.environ),
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
        text=True,
        check=True,
    )
    return output_prefix.with_suffix(".json")


def generate_track(
    audio_path: Path,
    lyrics_path: Path | None,
    output_path: Path,
    whisper_cli: Path,
    model_path: Path,
    metadata: dict[str, dict],
) -> dict:
    with tempfile.TemporaryDirectory(prefix="songbook-lyric-times-") as tmp:
        whisper_json = run_whisper(audio_path, whisper_cli, model_path, Path(tmp))

        segments, words = parse_whisper_json(whisper_json)

    title = None
    lyric_sections: list[LyricSection] = []
    warnings: list[str] = []
    if lyrics_path and lyrics_path.exists():
        title, lyric_sections = parse_lyrics(lyrics_path)
    else:
        warnings.append("missing lyrics file")

    aligned_sections = align_sections(lyric_sections, words)
    item_metadata = metadata.get(audio_path.stem, {})

    payload = {
        "schema_version": 1,
        "track_id": audio_path.stem,
        "title": item_metadata.get("title") or title,
        "artist": item_metadata.get("artist"),
        "audio_path": str(audio_path.relative_to(DATASET)),
        "lyrics_path": str(lyrics_path.relative_to(DATASET)) if lyrics_path else None,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "generator": {
            "script": "scripts/generate-lyric-timesheets.py",
            "engine": "whisper.cpp",
            "model": str(model_path),
            "options": ["-l", "en", "-oj", "-ojf"],
        },
        "warnings": warnings,
        "lyric_sections": aligned_sections,
        "timed_lyrics": aligned_sections,
        "transcript_segments": segments,
        "words": words,
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return {
        "track_id": audio_path.stem,
        "output_path": str(output_path.relative_to(ROOT)),
        "lyrics_sections": len(aligned_sections),
        "words": len(words),
        "warnings": warnings,
    }


def build_index(output_dir: Path, items: list[dict]) -> None:
    payload = {
        "schema_version": 1,
        "dataset": "mnesia-audio-v1",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "track_count": len(items),
        "tracks": items,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "index.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate lyric timesheets from MNESIA audio and lyrics.")
    parser.add_argument("--audio-dir", type=Path, default=DATASET / "audio")
    parser.add_argument("--lyrics-dir", type=Path, default=DATASET / "lyrics")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--whisper-cli", type=Path, default=DEFAULT_WHISPER_CLI)
    parser.add_argument("--model", type=Path, default=DEFAULT_WHISPER_MODEL)
    parser.add_argument("--track", action="append", help="Only generate the named track id. Can be repeated.")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    if not args.whisper_cli.exists():
        raise SystemExit(f"missing whisper CLI: {args.whisper_cli}")
    if not args.model.exists():
        raise SystemExit(f"missing whisper model: {args.model}")
    if not args.audio_dir.exists():
        raise SystemExit(f"missing audio directory: {args.audio_dir}")

    tracks = sorted(args.audio_dir.glob("*.mp3"))
    if args.track:
        requested = set(args.track)
        tracks = [track for track in tracks if track.stem in requested]
        missing = sorted(requested - {track.stem for track in tracks})
        if missing:
            raise SystemExit(f"missing requested audio tracks: {', '.join(missing)}")

    metadata = load_metadata()
    generated: list[dict] = []

    for audio_path in tracks:
        output_path = args.output_dir / f"{audio_path.stem}.json"
        lyrics_path = args.lyrics_dir / f"{audio_path.stem}.txt"

        if output_path.exists() and not args.force:
            generated.append(
                {
                    "track_id": audio_path.stem,
                    "output_path": str(output_path.relative_to(ROOT)),
                    "skipped": True,
                }
            )
            print(f"skip {audio_path.stem}", flush=True)
            continue

        print(f"generate {audio_path.stem}", flush=True)
        generated.append(
            generate_track(
                audio_path=audio_path,
                lyrics_path=lyrics_path if lyrics_path.exists() else None,
                output_path=output_path,
                whisper_cli=args.whisper_cli,
                model_path=args.model,
                metadata=metadata,
            )
        )

    build_index(args.output_dir, generated)
    print(f"wrote {len(generated)} lyric timesheets to {args.output_dir.relative_to(ROOT)}", flush=True)


if __name__ == "__main__":
    main()
