"""Time helpers. Instants are ISO-8601 with offset; local dates use the company IANA zone
(pinned tzdata wheel; set PYTHONTZPATH="" so host tz files are never consulted)."""
from __future__ import annotations

import zoneinfo
from datetime import date, datetime, timedelta, timezone
from zoneinfo import ZoneInfo

zoneinfo.reset_tzpath([])  # only the pinned tzdata wheel, never host tz files


def parse(ts: str) -> datetime:
    dt = datetime.fromisoformat(ts)
    if dt.tzinfo is None:
        raise ValueError("timestamp without offset")
    return dt


def local_date(ts: str, tz: str) -> date:
    return parse(ts).astimezone(ZoneInfo(tz)).date()


def iso(dt: datetime) -> str:
    return dt.isoformat(timespec="seconds")


def local_instant(d: date, hour: int, minute: int, tz: str) -> datetime:
    return datetime(d.year, d.month, d.day, hour, minute, tzinfo=ZoneInfo(tz))


def in_utc(dt: datetime) -> datetime:
    return dt.astimezone(timezone.utc)


def daterange(start: date, end_exclusive: date):
    d = start
    while d < end_exclusive:
        yield d
        d += timedelta(days=1)


def is_business_day(d: date) -> bool:
    return d.weekday() < 5
