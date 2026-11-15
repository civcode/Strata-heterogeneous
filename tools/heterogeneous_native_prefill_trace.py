"""Join successful native-prefix commits to optional host round-trip timings."""


def join_native_prefill(markers, timings):
    """Retain legacy untimed commits, but reject ambiguous or orphan timings."""
    commits = {}
    for marker in markers:
        request = marker["request"]
        if request in commits:
            raise ValueError("duplicate native prefill marker")
        commits[request] = dict(marker)
    seen = set()
    for timing in timings:
        request = timing["request"]
        if request in seen or request not in commits:
            raise ValueError("duplicate or orphan native prefill timing")
        seen.add(request)
        marker = commits[request]
        start, end = timing["start_ns"], timing["end_ns"]
        if timing["rows"] != marker["rows"] or not 0 <= start <= end:
            raise ValueError("invalid native prefill timing")
        marker.update(start_ns=start, end_ns=end, roundtrip_ns=end-start)
    return list(commits.values())


def native_timing_issue(marker, bounds, first_window_start_ns):
    """Check that a timed prefix is in the same prefill phase, before window 2."""
    if "start_ns" not in marker:
        return None
    start, end = marker["start_ns"], marker["end_ns"]
    if not (bounds["request_start"] <= start / 1e9 <= end / 1e9 <= bounds["prefill_end"]
            and end <= first_window_start_ns):
        return "native prefill timing outside benchmark phase or overlapping ordinary window"
    return None
