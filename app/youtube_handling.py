
import re
import subprocess
from pathlib import Path

# Directory to store subtitles and cleaned transcripts
TEMP_DIR = Path("static/temp")
TEMP_DIR.mkdir(parents=True, exist_ok=True)


def clean_old_temp_files():
    """Delete all VTT and TXT files from the temp folder."""
    for file in TEMP_DIR.glob("*"):
        if file.suffix in [".vtt", ".txt"]:
            try:
                file.unlink()
            except Exception as e:
                print(f"Warning: Couldn't delete {file}: {e}")


def extract_video_id(url: str) -> str:
    # Extract the video ID from various YouTube URL formats
    match = re.search(r'(?:v=|youtu\.be/|embed/|watch\?v=)([\w-]+)', url)
    if not match:
        raise ValueError("Invalid YouTube URL")

    video_id = match.group(1)

    # Reconstruct the clean URL
    clean_url = f"https://www.youtube.com/watch?v={video_id}"
    return clean_url


def download_subtitles(url, langs=('en', 'hi')) -> Path:
    """
    Downloads subtitles using yt-dlp for the given YouTube URL.
    Tries the languages in order. Returns the path to the saved .vtt file.
    """
    clean_url = extract_video_id(url)  # Get the cleaned URL
    video_id_match = re.search(r'(?:v=|youtu\.be/|embed/)([\w-]+)', clean_url)
    if not video_id_match:
        raise ValueError("Could not extract video ID from cleaned URL")
    video_id = video_id_match.group(1)
    output_basename = TEMP_DIR / video_id

    for lang in langs:
        try:
            subprocess.run([
                'yt-dlp',
                '--write-auto-sub',
                '--sub-lang', lang,
                '--skip-download',
                '--output', str(output_basename),
                clean_url  # Use the cleaned URL here
            ], check=True)

            # Look for .vtt file
            for file in TEMP_DIR.glob(f"{video_id}*.vtt"):
                return file

        except subprocess.CalledProcessError:
            continue

    raise FileNotFoundError("No subtitles found in the given languages.")


# def extract_transcript_with_timestamps(vtt_file: Path) -> list[dict]:
#     segments = []
#     current_segment = None
    
#     # Regex to match VTT timestamp lines: 00:00:00.000 --> 00:00:05.000
#     timestamp_pattern = re.compile(r'(\d{2}:\d{2}:\d{2}\.\d{3}) --> (\d{2}:\d{2}:\d{2}\.\d{3})')

#     with open(vtt_file, 'r', encoding='utf-8') as f:
#         for line in f:
#             line = line.strip()
#             if not line:
#                 continue
            
#             if 'WEBVTT' in line:
#                 continue
            
#             # Skip simple numeric lines (sequence numbers)
#             if line.isdigit():
#                 continue

#             match = timestamp_pattern.search(line)
#             if match:
#                 if current_segment and current_segment['text'].strip():
#                     segments.append(current_segment)
#                 current_segment = {'start': match.group(1), 'end': match.group(2), 'text': ''}
#             elif current_segment:
#                 clean_text = re.sub(r'<[^>]+>', '', line)
#                 current_segment['text'] += clean_text + " "

#     if current_segment and current_segment['text'].strip():
#         segments.append(current_segment)
        
#     return segments

import re
from datetime import timedelta, datetime

def parse_time(t: str) -> float:
    # Convert VTT timestamp to seconds (00:00:12.345 → seconds)
    h, m, s = t.split(":")
    s, ms = s.split(".")
    return int(h) * 3600 + int(m) * 60 + int(s) + int(ms) / 1000


def format_time(seconds: float) -> str:
    td = timedelta(seconds=seconds)
    # Format back to HH:MM:SS.mmm
    total_seconds = int(td.total_seconds())
    ms = int((seconds - total_seconds) * 1000)
    return str(timedelta(seconds=total_seconds)) + f".{ms:03d}"


def extract_transcript_with_timestamps(vtt_file: Path) -> list[dict]:
    segments = []

    timestamp_pattern = re.compile(
        r'(\d{2}:\d{2}:\d{2}\.\d{3}) --> (\d{2}:\d{2}:\d{2}\.\d{3})'
    )

    captions = []

    # -------- Parse VTT normally first --------
    with open(vtt_file, 'r', encoding='utf-8') as f:
        current = None
        for line in f:
            line = line.strip()
            if not line or 'WEBVTT' in line or line.isdigit():
                continue

            match = timestamp_pattern.search(line)
            if match:
                if current and current['text'].strip():
                    captions.append(current)

                current = {
                    "start": parse_time(match.group(1)),
                    "end": parse_time(match.group(2)),
                    "text": ""
                }
            elif current:
                clean = re.sub(r'<[^>]+>', '', line)
                current["text"] += clean + " "

        if current and current["text"].strip():
            captions.append(current)

    # -------- Now merge into fixed 1-minute windows --------
    if not captions:
        return []

    video_start = captions[0]["start"]
    video_end = captions[-1]["end"]

    window_start = video_start
    window_end = window_start + 60

    text_buffer = ""

    for cap in captions:
        while cap["start"] >= window_end:
            # Close current window first
            if text_buffer.strip():
                segments.append({
                    "start": format_time(window_start),
                    "end": format_time(window_end),
                    "text": text_buffer.strip()
                })
            text_buffer = ""
            window_start = window_end
            window_end += 60

        text_buffer += cap["text"]

    # Flush final chunk
    if text_buffer.strip():
        segments.append({
            "start": format_time(window_start),
            "end": format_time(window_end),
            "text": text_buffer.strip()
        })

    return segments

# def extract_clean_text(vtt_file: Path) -> str:
#     # Kept for backward compatibility if needed, but we prefer structured data now
#     segments = extract_transcript_with_timestamps(vtt_file)
#     full_text = " ".join([seg['text'].strip() for seg in segments])
    
#     txt_file = vtt_file.with_suffix('.txt')
#     with open(txt_file, 'w', encoding='utf-8') as f:
#         f.write(full_text)
        
#     return full_text

def extract_full_text_from_youtube(vtt_file: Path) -> str:
    """
    Extracts the full text from a YouTube VTT file as a single string.
    This is used for LLM-based chunking which prefers the whole document context.
    """
    segments = extract_transcript_with_timestamps(vtt_file)
    # Combine all segment texts
    full_text = " ".join([seg['text'].strip() for seg in segments])
    return full_text
