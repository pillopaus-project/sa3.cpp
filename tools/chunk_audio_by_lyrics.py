#!/usr/bin/env python3
import argparse
import json
import re
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SONG_BOOK = Path.home() / "repos" / "bako" / "song-book"
DATASET = DEFAULT_SONG_BOOK / "datasets" / "mnesia-audio-v1"
DEFAULT_TIMESHEETS = DEFAULT_SONG_BOOK / "data" / "lyric-timesheets" / "mnesia-audio-v1"
DEFAULT_OUTPUT = DATASET / "lyric-section-chunks"
MIN_CHUNK_SECONDS = 30.0
CUT_PREROLL_SECONDS = 0.05


@dataclass
class Section:
    index: int
    label: str
    start_time: float | None
    text: str
    lines: list[str]


@dataclass
class ChunkPlan:
    index: int
    start_time: float
    end_time: float
    sections: list[Section]
    warnings: list[str]

    @property
    def duration(self) -> float:
        return self.end_time - self.start_time


def slugify(text: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")
    return slug or "chunk"


def run_json(command: list[str]) -> dict:
    completed = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, check=True)
    return json.loads(completed.stdout)


def audio_duration(path: Path) -> float:
    data = run_json(
        [
            "ffprobe",
            "-v",
            "error",
            "-show_entries",
            "format=duration",
            "-of",
            "json",
            str(path),
        ]
    )
    return float(data["format"]["duration"])


def first_word_start(section: dict) -> float | None:
    for word in section.get("timed_words") or []:
        value = word.get("start_time")
        if value is not None:
            return float(value)
    value = section.get("start_time")
    if value is not None:
        return float(value)
    return None


def load_sections(timesheet: dict) -> list[Section]:
    sections: list[Section] = []
    for item in timesheet.get("timed_lyrics") or timesheet.get("lyric_sections") or []:
        sections.append(
            Section(
                index=int(item["index"]),
                label=item.get("label") or f"Section {item['index'] + 1}",
                start_time=first_word_start(item),
                text=item.get("text") or "",
                lines=item.get("lines") or [],
            )
        )
    return sections


def plan_chunks(sections: list[Section], duration: float) -> list[ChunkPlan]:
    if not sections:
        return [ChunkPlan(index=0, start_time=0.0, end_time=duration, sections=[], warnings=["no lyric sections"])]

    anchored = [section for section in sections if section.start_time is not None]
    if not anchored:
        return [
            ChunkPlan(
                index=0,
                start_time=0.0,
                end_time=duration,
                sections=sections,
                warnings=["no section start timings; emitted full-track chunk"],
            )
        ]

    section_starts = {section.index: section.start_time for section in anchored if section.index > 0}
    groups: list[list[Section]] = []
    current: list[Section] = []

    for section in sections:
        if section.index in section_starts and current:
            groups.append(current)
            current = []
        current.append(section)
    if current:
        groups.append(current)

    chunks: list[ChunkPlan] = []
    pending: list[Section] = []
    chunk_start: float | None = 0.0

    for group in groups:
        group_start = next((section_starts.get(section.index) for section in group if section.index in section_starts), None)
        if group_start is None:
            pending.extend(group)
            continue

        if not pending:
            pending = list(group)
            continue

        if chunk_start is None:
            chunk_start = 0.0

        prospective_duration = group_start - chunk_start
        if prospective_duration >= MIN_CHUNK_SECONDS:
            chunks.append(
                ChunkPlan(
                    index=len(chunks),
                    start_time=max(0.0, chunk_start - CUT_PREROLL_SECONDS),
                    end_time=max(0.0, group_start - CUT_PREROLL_SECONDS),
                    sections=pending,
                    warnings=[],
                )
            )
            pending = list(group)
            chunk_start = group_start
        else:
            pending.extend(group)

    if pending:
        if chunk_start is None:
            chunk_start = 0.0
        chunks.append(
            ChunkPlan(
                index=len(chunks),
                start_time=max(0.0, chunk_start - CUT_PREROLL_SECONDS),
                end_time=duration,
                sections=pending,
                warnings=[],
            )
        )

    if len(chunks) > 1 and chunks[-1].duration < MIN_CHUNK_SECONDS:
        tail = chunks.pop()
        previous = chunks[-1]
        chunks[-1] = ChunkPlan(
            index=previous.index,
            start_time=previous.start_time,
            end_time=tail.end_time,
            sections=previous.sections + tail.sections,
            warnings=previous.warnings + ["merged trailing chunk under 30s"],
        )

    normalized: list[ChunkPlan] = []
    for index, chunk in enumerate(chunks):
        warnings = list(chunk.warnings)
        if chunk.duration < MIN_CHUNK_SECONDS:
            warnings.append("chunk is under 30s because source interval is under 30s")
        normalized.append(
            ChunkPlan(
                index=index,
                start_time=round(chunk.start_time, 3),
                end_time=round(chunk.end_time, 3),
                sections=chunk.sections,
                warnings=warnings,
            )
        )

    return normalized


def chunk_lyrics_text(chunk: ChunkPlan) -> str:
    parts: list[str] = []
    for section in chunk.sections:
        parts.append(section.text.strip())
    return "\n\n".join(part for part in parts if part).strip() + "\n"


def export_audio(audio_path: Path, output_path: Path, start_time: float, duration: float) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        [
            "ffmpeg",
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            "-ss",
            f"{start_time:.3f}",
            "-i",
            str(audio_path),
            "-t",
            f"{duration:.3f}",
            "-map",
            "0:a:0",
            "-c:a",
            "libmp3lame",
            "-q:a",
            "2",
            str(output_path),
        ],
        check=True,
    )


def clean_output(output_dir: Path) -> None:
    for child in ["audio", "lyrics"]:
        path = output_dir / child
        if path.exists():
            shutil.rmtree(path)
    for child in ["metadata.jsonl", "index.json"]:
        path = output_dir / child
        if path.exists():
            path.unlink()


def main() -> None:
    parser = argparse.ArgumentParser(description="Chunk MNESIA audio by timed lyric section boundaries.")
    parser.add_argument("--timesheet-dir", type=Path, default=DEFAULT_TIMESHEETS)
    parser.add_argument("--audio-dir", type=Path, default=DATASET / "audio")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    if not args.timesheet_dir.exists():
        raise SystemExit(f"missing timesheet directory: {args.timesheet_dir}")
    if not args.audio_dir.exists():
        raise SystemExit(f"missing audio directory: {args.audio_dir}")

    if args.force:
        clean_output(args.output_dir)

    audio_out = args.output_dir / "audio"
    lyrics_out = args.output_dir / "lyrics"
    records: list[dict] = []

    for timesheet_path in sorted(args.timesheet_dir.glob("*.json")):
        if timesheet_path.name == "index.json":
            continue

        timesheet = json.loads(timesheet_path.read_text(encoding="utf-8"))
        track_id = timesheet["track_id"]
        audio_path = args.audio_dir / f"{track_id}.mp3"
        if not audio_path.exists():
            records.append({"track_id": track_id, "warnings": ["missing source audio"]})
            continue

        if not timesheet.get("lyrics_path"):
            records.append({"track_id": track_id, "warnings": ["missing lyrics file; skipped chunk export"]})
            print(f"skip {track_id}: missing lyrics", flush=True)
            continue

        print(f"chunk {track_id}", flush=True)
        duration = audio_duration(audio_path)
        sections = load_sections(timesheet)
        chunks = plan_chunks(sections, duration)

        for chunk in chunks:
            labels = "-".join(slugify(section.label) for section in chunk.sections[:2])
            chunk_id = f"{track_id}__chunk-{chunk.index + 1:02d}"
            if labels:
                chunk_id = f"{chunk_id}-{labels}"

            chunk_audio = audio_out / f"{chunk_id}.mp3"
            chunk_lyrics = lyrics_out / f"{chunk_id}.txt"
            export_audio(audio_path, chunk_audio, chunk.start_time, chunk.duration)
            chunk_lyrics.parent.mkdir(parents=True, exist_ok=True)
            chunk_lyrics.write_text(chunk_lyrics_text(chunk), encoding="utf-8")

            records.append(
                {
                    "id": chunk_id,
                    "track_id": track_id,
                    "title": timesheet.get("title"),
                    "artist": timesheet.get("artist"),
                    "audio_path": str(chunk_audio.relative_to(args.output_dir)),
                    "lyrics_path": str(chunk_lyrics.relative_to(args.output_dir)),
                    "source_audio_path": str(audio_path.relative_to(DATASET)),
                    "start_time": chunk.start_time,
                    "end_time": chunk.end_time,
                    "duration_seconds": round(chunk.duration, 3),
                    "section_indexes": [section.index for section in chunk.sections],
                    "section_labels": [section.label for section in chunk.sections],
                    "warnings": chunk.warnings,
                }
            )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    metadata_path = args.output_dir / "metadata.jsonl"
    with metadata_path.open("w", encoding="utf-8") as handle:
        for record in records:
            if "id" in record:
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")

    index = {
        "schema_version": 1,
        "dataset": "mnesia-audio-v1-lyric-section-chunks",
        "source_timesheets": str(args.timesheet_dir.relative_to(ROOT)),
        "min_chunk_seconds": MIN_CHUNK_SECONDS,
        "chunk_count": sum(1 for record in records if "id" in record),
        "skipped_tracks": [record for record in records if "id" not in record],
        "chunks": [record for record in records if "id" in record],
    }
    (args.output_dir / "index.json").write_text(json.dumps(index, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"wrote {index['chunk_count']} chunks to {args.output_dir.relative_to(ROOT)}", flush=True)


if __name__ == "__main__":
    main()
