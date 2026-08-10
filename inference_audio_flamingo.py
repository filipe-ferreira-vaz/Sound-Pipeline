#!/usr/bin/env python3
"""
Run inference on a local directory of audio files against an Audio Flamingo 3
model served by vLLM (OpenAI-compatible API).

Expected folder layout:
    Output_n/background-x-[event_1]-y-[event_2]-z-(...)-[event_n]-k/[SNR]/<audio>
"""

import argparse
import base64
import json
import logging
import re
import sys
from collections import defaultdict
from pathlib import Path

from openai import OpenAI

log = logging.getLogger("inference")

PATH_PATTERN = re.compile(
    r"(?:Output_(?P<num_events>\d+)/)?"
    r"background-\d+-(?P<events>[a-z_]+-\d+(?:-[a-z_]+-\d+)*)"
    r"/(?P<snr>\d+(?:\.\d+)?)(?:/|$)"
)
EVENT_PAIR = re.compile(r"([a-z_]+)-(\d+)")
AUDIO_EXTS = {".wav", ".mp3", ".flac", ".ogg", ".m4a", ".aac", ".wma"}

MIME_TYPES = {
    ".wav": "audio/wav",
    ".mp3": "audio/mpeg",
    ".flac": "audio/flac",
    ".ogg": "audio/ogg",
    ".m4a": "audio/mp4",
    ".aac": "audio/aac",
    ".wma": "audio/x-ms-wma",
}

PROMPTS = [
    {
        "title": "Simultaneous_Check",
        "template": (
            "<audio>\n"
            "Analyze the provided audio.\n"
            "\n"
            "Task:\n"
            "Determine if all of the following sound events occur "
            "simultaneously (overlap temporally) at any point in the audio.\n"
            "\n"
            "Target Events:\n"
            "[sound events]\n"
            "\n"
            "Requirements:\n"
            "1. Return true ONLY if all listed events are present AND share "
            "at least one temporal intersection.\n"
            "2. Return false if any listed event is missing, or if they do "
            "not all overlap.\n"
            "\n"
            "Strict Output Constraints:\n"
            "1. Output exactly one JSON object.\n"
            "2. The JSON object must contain exactly one key: "
            '"simultaneous".\n'
            "3. The value must be a boolean (true or false).\n"
            "4. DO NOT wrap the output in markdown code blocks (e.g., "
            "```json or ```python).\n"
            "5. DO NOT output conversational text, explanations, scripts, "
            "or arrays.\n"
            "6. Your entire response must be exactly one of the two "
            "allowed outputs below.\n"
            "\n"
            "Allowed Output 1:\n"
            '{"simultaneous": true}\n'
            "\n"
            "Allowed Output 2:\n"
            '{"simultaneous": false}'
        ),
    },
    {
        "title": "Event_Identification",
        "template": (
            "<audio>\n"
            "Analyze the provided audio. It contains background noise and "
            "multiple overlapping sound events.\n"
            "\n"
            "Task:\n"
            "Identify every distinct sound event present in the audio.\n"
            "\n"
            "Requirements:\n"
            "1. Isolate and identify overlapping sounds explicitly.\n"
            "2. Exclude general, unstructured background noise from the "
            "final list.\n"
            "3. Classify each sound event as specifically as possible.\n"
            "\n"
            "Output Format:\n"
            "Provide the identified sound events strictly as a JSON array "
            "of strings. Do not include conversational text or "
            "explanations.\n"
            "\n"
            "Expected Output Example:\n"
            '["dog bark", "car horn", "airplane flyby"]'
        ),
    },
    {
        "title": "Event_Identification_modified",
        "template": (
            "<audio>\n"
            "Analyze the provided audio. It contains background noise and "
            "multiple overlapping sound events.\n"
            "\n"
            "Task:\n"
            "Identify every distinct sound event present in the audio.\n"
            "\n"
            "Requirements:\n"
            "1. Isolate and identify overlapping sounds explicitly.\n"
            "2. Exclude general, unstructured background noise from the "
            "final list.\n"
            "3. Classify each sound event as specifically as possible.\n"
            "\n"
            "Output Format:\n"
            "Provide the identified sound events strictly as a JSON array "
            "of strings. Do not include conversational text or "
            "explanations.\n"
            "\n"
            "Expected Output Example:\n"
            '["sound event 1", "sound event 2", "sound event 3"]'
        ),
    },
]


def parse_path(audio_path: Path):
    match = PATH_PATTERN.search(str(audio_path))
    if not match:
        raise ValueError(
            f"Path does not match expected layout: {audio_path}\n"
            "Expected: Output_n/background-x-[event_1]-y-[event_2]-.../[SNR]/<audio>"
        )
    events = [name for name, _ in EVENT_PAIR.findall(match.group("events"))]
    num_events = match.group("num_events")
    if num_events is None:
        num_events = len(events)
    return int(num_events), events, match.group("snr")


def make_data_url(audio_path: Path) -> str:
    audio_bytes = audio_path.read_bytes()
    mime = MIME_TYPES.get(audio_path.suffix.lower(), "audio/wav")
    return f"data:{mime};base64,{base64.b64encode(audio_bytes).decode('ascii')}"


def call_model(client: OpenAI, model: str, audio_path: Path, prompt: str,
               temperature: float, max_tokens: int) -> str:
    data_url = make_data_url(audio_path)
    completion = client.chat.completions.create(
        model=model,
        temperature=temperature,
        max_tokens=max_tokens,
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "audio_url", "audio_url": {"url": data_url}},
                    {"type": "text", "text": prompt},
                ],
            }
        ],
    )
    return completion.choices[0].message.content


def sanitize_token(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.]+", "_", value).strip("_")


def build_output_filename(snr: str, prompt_title: str) -> str:
    safe_title = sanitize_token(prompt_title)
    return f"results_SNR_{snr}_{safe_title}.json"


def build_run_dir_name(num_events, snrs, prompt_titles) -> str:
    events_part = "-".join(str(n) for n in sorted(set(num_events)))
    snr_part = "_".join(sanitize_token(s) for s in sorted(set(snrs), key=float))
    prompts_part = "_".join(sanitize_token(t) for t in prompt_titles)
    return f"results_{events_part}-sound-events_{snr_part}_{prompts_part}"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run Audio Flamingo 3 (vLLM) inference on local audio files."
    )
    parser.add_argument("root", nargs="?", default=".",
                        help="Root folder to traverse (e.g. the Output_n folder).")
    parser.add_argument("--base-url", default="http://localhost:8123/v1",
                        help="vLLM OpenAI-compatible API base URL.")
    parser.add_argument("--api-key", default="EMPTY",
                        help="API key expected by the vLLM server.")
    parser.add_argument("--model", default="nvidia/audio-flamingo-3-hf",
                        help="Model name served by vLLM.")
    parser.add_argument("--output-dir", default="results",
                        help="Directory where the per-run result folder is "
                             "created.")
    parser.add_argument("--prompts", nargs="+",
                        choices=[p["title"] for p in PROMPTS],
                        default=[p["title"] for p in PROMPTS],
                        help="Prompt(s) to run (default: all).")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max-tokens", type=int, default=256)
    parser.add_argument("--limit", type=int, default=0,
                        help="Only process the first N audio files (0 = all).")
    parser.add_argument("--quiet", action="store_true",
                        help="Only log the debug lines (no progress/counters).")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logging.basicConfig(
        level=logging.WARNING if args.quiet else logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        datefmt="%H:%M:%S",
    )

    prompts = [p for p in PROMPTS if p["title"] in args.prompts]

    root = Path(args.root).resolve()
    audio_files = [
        p for p in root.rglob("*")
        if p.is_file() and p.suffix.lower() in AUDIO_EXTS
    ]
    if not audio_files:
        log.error("No audio files found under %s", root)
        sys.exit(1)
    if args.limit > 0:
        audio_files = audio_files[: args.limit]
    log.info("Discovered %d audio file(s) under %s", len(audio_files), root)

    num_events = set()
    snrs = set()
    for audio_path in audio_files:
        try:
            n, _, snr = parse_path(audio_path)
        except ValueError:
            continue
        num_events.add(n)
        snrs.add(snr)
    if not snrs:
        log.error("No audio files match the expected folder layout.")
        sys.exit(1)

    run_dir = Path(args.output_dir) / build_run_dir_name(
        num_events, snrs, [p["title"] for p in prompts]
    )
    run_dir.mkdir(parents=True, exist_ok=True)
    log.info("Results will be written to %s", run_dir)

    client = OpenAI(api_key=args.api_key, base_url=args.base_url)
    results_by_key = defaultdict(list)
    failures = 0

    for idx, audio_path in enumerate(audio_files, start=1):
        try:
            num_events, sound_events, snr = parse_path(audio_path)
        except ValueError as exc:
            log.error("SKIP %s: %s", audio_path.name, exc)
            continue

        for prompt in prompts:
            raw_title = prompt["title"]
            filled_prompt = prompt["template"].replace(
                "[sound events]", ", ".join(sound_events)
            )

            log.info("=" * 80)
            log.info("[%d/%d] Audio: %s", idx, len(audio_files), audio_path.name)
            log.info("Prompt (%s):\n%s", raw_title, filled_prompt)

            try:
                model_output = call_model(
                    client,
                    args.model,
                    audio_path,
                    filled_prompt,
                    args.temperature,
                    args.max_tokens,
                )
            except Exception as exc:
                failures += 1
                model_output = f"[ERROR] {type(exc).__name__}: {exc}"
                log.error("API call failed for %s: %s", audio_path.name, exc)
            else:
                log.info("Raw output:\n%s", model_output)

            results_by_key[(snr, raw_title)].append(
                {
                    "filename": audio_path.name,
                    "num_sound_events": num_events,
                    "sound_events": sound_events,
                    "prompt_title": raw_title,
                    "snr": snr,
                    "model_output": model_output,
                }
            )

    if not results_by_key:
        log.error("No valid audio files were processed.")
        sys.exit(1)

    for (snr, prompt_title), records in sorted(results_by_key.items()):
        out_path = run_dir / build_output_filename(snr, prompt_title)
        with out_path.open("w", encoding="utf-8") as fh:
            json.dump(records, fh, indent=2, ensure_ascii=False)
            fh.write("\n")
        log.info("Wrote %d result(s) to %s", len(records), out_path)

    log.info("Done. Successful calls: %d, failed calls: %d",
             sum(len(r) for r in results_by_key.values()) - failures, failures)


if __name__ == "__main__":
    main()