"""Helpers for turning supported viewing pages into readable video sources."""

from __future__ import annotations

from urllib.parse import urlparse


def is_youtube_url(source: str) -> bool:
    """Return whether a URL belongs to YouTube's supported public domains."""
    parsed = urlparse(source)
    host = (parsed.hostname or "").lower().rstrip(".")
    return parsed.scheme in {"http", "https"} and (
        host == "youtu.be" or host == "youtube.com" or host.endswith(".youtube.com")
    )


def resolve_youtube_stream(source: str) -> str:
    """Resolve a public YouTube video/live page to a temporary media URL."""
    if not is_youtube_url(source):
        raise ValueError("This is not a supported YouTube URL.")

    try:
        import yt_dlp
    except ImportError as exc:
        raise RuntimeError("YouTube links need yt-dlp. Install project dependencies with: python -m pip install -r requirements.txt") from exc

    options = {
        "quiet": True,
        "no_warnings": True,
        "noplaylist": True,
        "format": "best",
    }
    try:
        with yt_dlp.YoutubeDL(options) as downloader:
            info = downloader.extract_info(source, download=False)
    except Exception as exc:
        raise RuntimeError(f"Could not open this YouTube stream: {exc}") from exc

    stream_url = info.get("url") if isinstance(info, dict) else None
    if not stream_url:
        raise RuntimeError("YouTube did not provide a playable stream for this video.")
    return stream_url
