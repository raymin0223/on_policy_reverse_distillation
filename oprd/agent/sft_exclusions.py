"""One explicitly user-approved exclusion, never a general corrupt-file filter."""

import hashlib

REVISION = "294604adbcd977f0ab27593ebd8fadb968a589db"
ARCHIVE = "cold_start_part2.zip"
SHA256 = "a9a6a72f73970490f9eb6df2b1ca8da2e121b172ffe3df004eeee8585a2193a1"
MEMBER = "GPT4Scene/14621/images/image_5.png"
DATASET = "GPT4Scene"
ROW_INDEX = 2784
QUESTION_ID = 14621


def archive_exclusions(path, revision):
    if revision != REVISION:
        raise ValueError("Corrupt-sample exception is only valid for the audited revision")
    if path.name != ARCHIVE:
        return ()
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            h.update(block)
    if h.hexdigest() != SHA256:
        raise ValueError("Corrupt-sample exception cannot mask a changed/download-damaged archive")
    return (MEMBER,)


def filter_approved_row(name, rows):
    if name != DATASET:
        return rows, []
    row = rows[ROW_INDEX]
    paths = [p.removeprefix("./").removeprefix("cold_start/") for p in row["images"]]
    if row.get("question_id") != QUESTION_ID or MEMBER not in paths:
        raise ValueError("Audited corrupt SFT row identity changed")
    hits = [
        i
        for i, r in enumerate(rows)
        if any(p.removeprefix("./").removeprefix("cold_start/") == MEMBER for p in r["images"])
    ]
    if hits != [ROW_INDEX]:
        raise ValueError("More than the one approved trajectory references the damaged image")
    return rows[:ROW_INDEX] + rows[ROW_INDEX + 1 :], [
        {
            "dataset": name,
            "source_row": ROW_INDEX,
            "question_id": QUESTION_ID,
            "image": MEMBER,
            "archive": ARCHIVE,
            "archive_sha256": SHA256,
            "reason": "Released PNG and ZIP CRC damaged; exact full trajectory exclusion approved by user",
        }
    ]
