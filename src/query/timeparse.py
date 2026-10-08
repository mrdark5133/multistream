import re
import datetime
from typing import Optional, Tuple


def parse_time_expression(
    query_text: str,
    now_ref: Optional[datetime.datetime] = None
) -> Tuple[Optional[datetime.datetime], Optional[datetime.datetime], str]:
    """
    Parse relative or absolute time expressions from natural language queries.
    Returns (start_dt, end_dt, matched_span_text).
    """
    if now_ref is None:
        now_ref = datetime.datetime.now()

    text = query_text.lower()

    # 1. "in the last/past X minutes/mins" or "last X minutes"
    m_min = re.search(r"\b(?:in\s+the\s+)?(?:last|past)\s+(\d+)\s*(?:minutes|mins|minute|min)\b", text)
    if m_min:
        mins = int(m_min.group(1))
        start_dt = now_ref - datetime.timedelta(minutes=mins)
        return start_dt, now_ref, m_min.group(0)

    # 2. "in the last/past X hours/hrs" or "last X hours"
    m_hr = re.search(r"\b(?:in\s+the\s+)?(?:last|past)\s+(\d+)\s*(?:hours|hrs|hour|hr)\b", text)
    if m_hr:
        hrs = int(m_hr.group(1))
        start_dt = now_ref - datetime.timedelta(hours=hrs)
        return start_dt, now_ref, m_hr.group(0)

    # 3. "in the last/past hour" or "last hour"
    m_1hr = re.search(r"\b(?:in\s+the\s+)?(?:last|past)\s+hour\b", text)
    if m_1hr:
        start_dt = now_ref - datetime.timedelta(hours=1)
        return start_dt, now_ref, m_1hr.group(0)

    # 4. "today"
    m_today = re.search(r"\btoday\b", text)
    if m_today:
        start_dt = now_ref.replace(hour=0, minute=0, second=0, microsecond=0)
        return start_dt, now_ref, m_today.group(0)

    # 5. "yesterday"
    m_yest = re.search(r"\byesterday\b", text)
    if m_yest:
        yesterday_date = (now_ref - datetime.timedelta(days=1)).date()
        start_dt = datetime.datetime.combine(yesterday_date, datetime.time.min)
        end_dt = datetime.datetime.combine(yesterday_date, datetime.time.max)
        return start_dt, end_dt, m_yest.group(0)

    # 6. "this morning"
    m_morn = re.search(r"\b(?:this\s+morning|in\s+the\s+morning)\b", text)
    if m_morn:
        base_date = now_ref.date()
        start_dt = datetime.datetime.combine(base_date, datetime.time(6, 0))
        end_dt = datetime.datetime.combine(base_date, datetime.time(12, 0))
        return start_dt, end_dt, m_morn.group(0)

    # 7. "this afternoon"
    m_aft = re.search(r"\b(?:this\s+afternoon|in\s+the\s+afternoon)\b", text)
    if m_aft:
        base_date = now_ref.date()
        start_dt = datetime.datetime.combine(base_date, datetime.time(12, 0))
        end_dt = datetime.datetime.combine(base_date, datetime.time(18, 0))
        return start_dt, end_dt, m_aft.group(0)

    # 8. "this evening"
    m_eve = re.search(r"\b(?:this\s+evening|in\s+the\s+evening)\b", text)
    if m_eve:
        base_date = now_ref.date()
        start_dt = datetime.datetime.combine(base_date, datetime.time(18, 0))
        end_dt = datetime.datetime.combine(base_date, datetime.time(23, 59, 59))
        return start_dt, end_dt, m_eve.group(0)

    # 9. "between X and Y AM/PM" (12-hour clock)
    m_ampm = re.search(r"\bbetween\s+(\d{1,2})(?::(\d{2}))?\s*(am|pm)?\s+and\s+(\d{1,2})(?::(\d{2}))?\s*(am|pm)\b", text)
    if m_ampm:
        h1, min1, ap1, h2, min2, ap2 = m_ampm.groups()
        h1, h2 = int(h1), int(h2)
        min1 = int(min1) if min1 else 0
        min2 = int(min2) if min2 else 0
        ap1 = (ap1 or ap2).lower()
        ap2 = ap2.lower()
        if ap1 == "pm" and h1 < 12:
            h1 += 12
        elif ap1 == "am" and h1 == 12:
            h1 = 0
        if ap2 == "pm" and h2 < 12:
            h2 += 12
        elif ap2 == "am" and h2 == 12:
            h2 = 0
        base_date = now_ref.date()
        start_dt = datetime.datetime.combine(base_date, datetime.time(h1, min1))
        end_dt = datetime.datetime.combine(base_date, datetime.time(h2, min2))
        return start_dt, end_dt, m_ampm.group(0)

    # 10. "between HH:MM and HH:MM" (24-hour clock)
    m_between = re.search(r"\bbetween\s+(\d{1,2}):(\d{2})\s+and\s+(\d{1,2}):(\d{2})\b", text)
    if m_between:
        h1, m1, h2, m2 = map(int, m_between.groups())
        base_date = now_ref.date()
        start_dt = datetime.datetime.combine(base_date, datetime.time(h1, m1))
        end_dt = datetime.datetime.combine(base_date, datetime.time(h2, m2))
        return start_dt, end_dt, m_between.group(0)

    # 7. Absolute ISO timestamp extraction: YYYY-MM-DDTHH:MM:SS
    m_iso = re.search(r"(\d{4}-\d{2}-\d{2}[T\s]\d{2}:\d{2}:\d{2})", query_text)
    if m_iso:
        dt = datetime.datetime.fromisoformat(m_iso.group(1).replace(" ", "T"))
        # Window of +/- 5 minutes around point timestamp
        return dt - datetime.timedelta(minutes=5), dt + datetime.timedelta(minutes=5), m_iso.group(0)

    return None, None, ""
