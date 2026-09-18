"""One module per public dataset, discovered by nimble.datasets.public_benchmarks.

Each module defines NAME, SOURCE_URL, LICENSE, NOTE, RELEASE_YEAR, SUBSETS,
rows(path, subset="") yielding raw upstream dicts from a downloaded local file, and
record(raw, subset="") returning a record built with
nimble.datasets.public_records.record_for, or None to skip the row.
"""
