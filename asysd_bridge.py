"""Optional genuine ASySD R-package deduplication bridge.

Never call this ASySD unless Rscript and the ASySD package actually run.
On hosts without R/ASySD (e.g. default Streamlit Community Cloud), the
conservative Python engine remains available and an ASySD-ready input CSV
and local R script can be downloaded.

Round-trip safety: all original UIDs must be accounted for exactly once,
conflicting DOIs must not be auto-merged, and original metadata/provenance
must be retained. A bridge failure leaves records unchanged.
"""
from __future__ import annotations

import csv
import io
import re
import shutil
import subprocess
import tempfile
from collections import Counter
from pathlib import Path

from review_core import ReviewRecord, MergeResult


class ASySDError(RuntimeError):
    pass


BRIDGE_SCRIPT = Path(__file__).with_name("asysd_bridge.R")


def asysd_available() -> bool:
    """Presence of Rscript is necessary, but not sufficient for ASySD."""
    return shutil.which("Rscript") is not None and BRIDGE_SCRIPT.exists()


def asysd_csv(records: list[ReviewRecord]) -> bytes:
    out = io.StringIO(newline="")
    cols = ["record_id", "title", "author", "year", "journal", "doi",
            "abstract", "source", "pmid"]
    writer = csv.DictWriter(out, fieldnames=cols)
    writer.writeheader()
    for record in records:
        writer.writerow({
            "record_id": record.uid,
            "title": record.title,
            "author": record.authors,
            "year": record.year,
            "journal": record.journal,
            "doi": record.doi,
            "abstract": record.abstract,
            "source": "; ".join(record.sources),
            "pmid": record.pmid,
        })
    return out.getvalue().encode("utf-8-sig")


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def _uids_from_asysd(row: dict[str, str]) -> list[str]:
    # ASySD with merge_citations=TRUE produces merged record_ids;
    # singletons may also expose a record_id or duplicate_id.
    value = row.get("record_ids") or row.get("record_id") or row.get("duplicate_id") or ""
    return list(dict.fromkeys(x.strip() for x in re.split(r"\s*,\s*", value) if x.strip()))


def _reconstruct(
    originals: list[ReviewRecord],
    unique_rows: list[dict[str, str]],
    candidate_rows: list[dict[str, str]],
) -> MergeResult:
    by_uid = {r.uid: r for r in originals}
    if len(by_uid) != len(originals):
        raise ASySDError("Non-unique input UIDs. Cannot audit ASySD assignments.")
    seen: set[str] = set()
    records: list[ReviewRecord] = []
    audit: list[dict] = []

    for row in unique_rows:
        ids = _uids_from_asysd(row)
        if not ids or any(x not in by_uid for x in ids):
            raise ASySDError("ASySD unique output lacks recognized record IDs.")
        if any(x in seen for x in ids):
            raise ASySDError("ASySD assigned a record to more than one group.")
        seen.update(ids)
        group = [by_uid[x] for x in ids]
        distinct_dois = {x.doi for x in group if x.doi}
        if len(distinct_dois) > 1:
            raise ASySDError(
                "ASySD suggested merging records with different nonempty DOIs. "
                "Review these conflicts manually; no records were removed."
            )
        preferred = max(
            group, key=lambda r: (
                bool(r.doi), bool(r.pmid),
                len(r.abstract), len(r.title), len(r.authors), r.uid
            )
        )
        def best(field: str) -> str:
            return max((str(getattr(r, field) or "") for r in group),
                       key=len, default="")
        merged = ReviewRecord(
            uid=min(ids), title=best("title"), abstract=best("abstract"),
            year=preferred.year or best("year"),
            doi=preferred.doi or best("doi"),
            pmid=preferred.pmid or best("pmid"),
            authors=best("authors"), journal=best("journal"),
            source=preferred.source, source_id=preferred.source_id,
            sources=sorted({src for rec in group for src in rec.sources}),
            identifiers=sorted({src for rec in group for src in rec.identifiers}),
        )
        records.append(merged)
        if len(group) > 1:
            audit.append({
                "master_uid": merged.uid, "group_size": len(group),
                "member_uids": ids, "sources": merged.sources,
                "source_identifiers": merged.identifiers,
                "doi": merged.doi, "pmid": merged.pmid,
                "matching_evidence": "ASySD R dedup_citations; auto-merge",
            })
    if seen != set(by_uid):
        missing = len(set(by_uid) - seen)
        raise ASySDError(
            f"ASySD export omitted {missing} original record(s). "
            "No deduplicated result was accepted."
        )

    records.sort(key=lambda x: (x.title.casefold(), x.uid))
    fuzzy = []
    for row in candidate_rows:
        fuzzy.append({
            **row, "status": "MANUAL REVIEW - ASySD candidate NOT merged"
        })
    return MergeResult(
        records=records, counts=dict(sorted(Counter(r.source for r in originals).items())),
        duplicates_removed=len(originals) - len(records),
        audit=audit, fuzzy_review=fuzzy, input_count=len(originals),
    )


def run_asysd(records: list[ReviewRecord], *, timeout: int = 180) -> MergeResult:
    """Execute the REAL ASySD R function; no fallback masquerading as ASySD."""
    if not asysd_available():
        raise ASySDError("Rscript not installed. Run ASySD locally or install R on the server.")
    if not records:
        raise ASySDError("ASySD requires at least one citation.")
    with tempfile.TemporaryDirectory(prefix="searchbridge_asysd_") as tmp:
        folder = Path(tmp)
        infile, unique, candidate = (
            folder / "citations.csv", folder / "unique.csv", folder / "manual.csv"
        )
        infile.write_bytes(asysd_csv(records))
        try:
            p = subprocess.run(
                ["Rscript", str(BRIDGE_SCRIPT), str(infile), str(unique), str(candidate)],
                capture_output=True, text=True, timeout=timeout, check=False,
            )
        except (OSError, subprocess.TimeoutExpired) as exc:
            raise ASySDError("ASySD R process unavailable or timed out.") from exc
        if p.returncode:
            raise ASySDError(
                "ASySD R deduplication failed. Install the official "
                "camaradesuk/ASySD package and its R dependencies. "
                + (p.stderr or p.stdout)[-700:]
            )
        if not unique.exists() or not candidate.exists():
            raise ASySDError("ASySD did not create both expected output CSVs.")
        return _reconstruct(records, _read_csv(unique), _read_csv(candidate))
