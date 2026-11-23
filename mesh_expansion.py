"""Resolve exact NLM MeSH descriptor explosion (root plus all narrower descriptors).

Uses the public NLM MeSH RDF Lookup and SPARQL APIs. This module does NOT
resolve APA PsycINFO or Emtree explosions. No LLM and no authentication.
"""
from __future__ import annotations

import csv
import io
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Callable

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from translator_core import Node, heading_details, parse_ovid


LOOKUP = "https://id.nlm.nih.gov/mesh/lookup/descriptor"
SPARQL = "https://id.nlm.nih.gov/mesh/sparql"
MESH_BASE = "http://id.nlm.nih.gov/mesh/"
ID_PATTERN = re.compile(r"^D\d{6,9}$")
PAGE_SIZE = 500
MAX_POSITIONS = 10000


class MeSHError(RuntimeError):
    """NLM unavailable, ambiguous match, or response cannot be validated."""


@dataclass(frozen=True)
class ExplodedHeading:
    name: str
    qualifier: str = ""

    @property
    def label(self) -> str:
        return f"exp {self.name}/" + (self.qualifier if self.qualifier else "")


@dataclass(frozen=True)
class DescriptorPosition:
    descriptor: str
    mesh_id: str
    tree_number: str
    root_tree: str
    level: int
    is_root: bool

    @property
    def mesh_url(self) -> str:
        return f"https://id.nlm.nih.gov/mesh/{self.mesh_id}"


@dataclass(frozen=True)
class MeSHExpansion:
    heading: ExplodedHeading
    resolved_name: str
    mesh_id: str
    positions: tuple[DescriptorPosition, ...]
    fetched_utc: str

    @property
    def children_count(self) -> int:
        return len({p.mesh_id for p in self.positions if not p.is_root})

    @property
    def descriptor_count(self) -> int:
        return len({p.mesh_id for p in self.positions})

    @property
    def root_trees(self) -> tuple[str, ...]:
        return tuple(sorted({p.root_tree for p in self.positions if p.is_root}))


def find_exp_headings(ovid_strategy: str) -> list[ExplodedHeading]:
    """Read *parsed* Ovid heading nodes rather than scanning free-text with regex."""
    node = parse_ovid(ovid_strategy)
    found: dict[tuple[str, str], ExplodedHeading] = {}

    def visit(current: Node | None) -> None:
        if current is None:
            return
        if current.kind == "heading":
            name, exploded, qualifier = heading_details(current.value)
            if exploded:
                key = name.casefold(), qualifier.casefold()
                found.setdefault(key, ExplodedHeading(name, qualifier))
        else:
            visit(current.left)
            visit(current.right)
    visit(node)
    return list(found.values())


def _session() -> requests.Session:
    s = requests.Session()
    s.headers.update({"User-Agent": "SearchBridge/0.4 (MeSH RDF descriptor explorer)",
                      "Accept": "application/json"})
    retry = Retry(total=2, connect=2, read=2, backoff_factor=0.8,
                  status_forcelist=[429, 500, 502, 503, 504],
                  allowed_methods=["GET"], respect_retry_after_header=True)
    s.mount("https://", HTTPAdapter(max_retries=retry))
    return s


def _json_get(s: requests.Session, url: str, params: dict) -> object:
    try:
        result = s.get(url, params=params, timeout=(12, 80))
        result.raise_for_status()
        return result.json()
    except (requests.RequestException, ValueError) as exc:
        raise MeSHError(
            f"MeSH RDF request failed ({url.rsplit('/',1)[-1]}): {type(exc).__name__}. "
            "Please retry or check NLM service availability."
        ) from exc


def _mesh_id(resource: str) -> str:
    """Accept only official descriptor IDs; cannot interpolate arbitrary URLs."""
    found = resource.rstrip("/").rsplit("/", 1)[-1]
    if not ID_PATTERN.fullmatch(found):
        raise MeSHError(f"Unexpected MeSH descriptor identifier: {resource!r}")
    return found


def lookup_exact_descriptor(
    label: str, session: requests.Session | None = None
) -> tuple[str, str]:
    """Resolve by exact preferred MeSH label; never guess nearest neighbour."""
    session = session or _session()
    if not label or len(label) > 300:
        raise MeSHError("Descriptor label missing or too long.")
    response = _json_get(session, LOOKUP,
                         {"label": label, "match": "exact", "year": "current", "limit": 50})
    if not isinstance(response, list):
        raise MeSHError("NLM lookup response was not an array.")
    exact = [x for x in response
             if isinstance(x, dict)
             and str(x.get("label", "")).casefold() == label.casefold()]
    ids: dict[str, str] = {}
    for item in exact:
        ids[_mesh_id(str(item.get("resource", "")))] = str(item["label"])
    if not ids:
        raise MeSHError(
            f"'{label}' is not an exact preferred MeSH descriptor label. "
            "Check the spelling or look up the preferred name in the NLM MeSH Browser."
        )
    if len(ids) != 1:
        raise MeSHError(
            f"'{label}' has multiple exact matches ({', '.join(ids)}). "
            "The expansion was not attempted because the descriptor is ambiguous."
        )
    return next(iter(ids.items()))


def _descendants_query(mesh_id: str, page_size: int, offset: int) -> str:
    """Official NLM pattern: parentTreeNumber+ rather than broaderDescriptor+."""
    if not ID_PATTERN.fullmatch(mesh_id):
        raise MeSHError("Invalid MeSH ID")
    return f"""PREFIX meshv: <http://id.nlm.nih.gov/mesh/vocab#>
PREFIX mesh: <http://id.nlm.nih.gov/mesh/>
PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
SELECT DISTINCT ?descriptor ?label ?rootTree ?childTree
FROM <http://id.nlm.nih.gov/mesh>
WHERE {{
    mesh:{mesh_id} meshv:treeNumber ?rootTree .
    ?childTree meshv:parentTreeNumber+ ?rootTree .
    ?descriptor meshv:treeNumber ?childTree .
    ?descriptor meshv:active 1 .
    ?descriptor rdfs:label ?label .
}}
ORDER BY ?rootTree ?childTree ?descriptor
LIMIT {page_size} OFFSET {offset}"""


def _value(binding: dict, key: str) -> str:
    val = binding.get(key)
    if not isinstance(val, dict) or not isinstance(val.get("value"), str):
        raise MeSHError(f"Missing '{key}' in NLM SPARQL binding.")
    return val["value"]


def _tree_segment(uri: str) -> str:
    seg = uri.rstrip("/").rsplit("/", 1)[-1]
    if not re.fullmatch(r"[A-Z]\d+(?:\.\d+)*", seg):
        raise MeSHError(f"Unexpected MeSH tree number: {uri!r}")
    return seg


def _parse_position(binding: dict) -> DescriptorPosition:
    descriptor_uri = _value(binding, "descriptor")
    mesh_id = _mesh_id(descriptor_uri)
    label = _value(binding, "label")
    root = _tree_segment(_value(binding, "rootTree"))
    child = _tree_segment(_value(binding, "childTree"))
    if not child.startswith(root + "."):
        raise MeSHError("Returned MeSH tree number does not descend from its root.")
    depth = child.count(".") - root.count(".")
    return DescriptorPosition(descriptor=label, mesh_id=mesh_id,
                              tree_number=child, root_tree=root, level=depth,
                              is_root=False)


def expand_mesh_heading(
    heading: ExplodedHeading,
    session: requests.Session | None = None,
    progress: Callable[[int], None] | None = None,
    page_size: int = PAGE_SIZE,
    max_positions: int = MAX_POSITIONS,
) -> MeSHExpansion:
    """Return root and all descendant tree positions; fail on incomplete pagination."""
    if page_size < 1 or max_positions < 1:
        raise ValueError("Page size and max positions must be positive.")
    session = session or _session()
    mesh_id, resolved_name = lookup_exact_descriptor(heading.name, session)
    rows: dict[tuple[str, str], DescriptorPosition] = {}
    offset = 0
    while True:
        data = _json_get(session, SPARQL,
                         {"query": _descendants_query(mesh_id, page_size, offset),
                          "format": "JSON", "limit": page_size})
        try:
            raw_rows = data["results"]["bindings"]
        except (TypeError, KeyError) as exc:
            raise MeSHError("Unexpected NLM SPARQL response (results.bindings missing).") from exc
        if not isinstance(raw_rows, list):
            raise MeSHError("Unexpected NLM SPARQL bindings.")
        if len(raw_rows) > page_size:
            raise MeSHError("NLM returned more rows than the requested page size.")
        for raw in raw_rows:
            position = _parse_position(raw)
            rows[position.mesh_id, position.tree_number] = position
        offset += len(raw_rows)
        if progress:
            progress(len(rows))
        if len(raw_rows) < page_size:
            break
        if offset >= max_positions:
            raise MeSHError(
                f"MeSH expansion exceeds the {max_positions:,}-position safety limit. "
                "Cannot show an incomplete hierarchy as if it were complete."
            )

    root_paths = sorted({position.root_tree for position in rows.values()})
    # For headings with no descendants, obtain their root tree nodes separately.
    # Even when descendants are present, this request ensures every root path is shown.
    query = f"""PREFIX meshv: <http://id.nlm.nih.gov/mesh/vocab#>
PREFIX mesh: <http://id.nlm.nih.gov/mesh/>
SELECT DISTINCT ?rootTree
FROM <http://id.nlm.nih.gov/mesh>
WHERE {{ mesh:{mesh_id} meshv:treeNumber ?rootTree . }}
ORDER BY ?rootTree"""
    response = _json_get(session, SPARQL,
                         {"query": query, "format": "JSON", "limit": 1000})
    try:
        root_bindings = response["results"]["bindings"]
    except (TypeError, KeyError) as exc:
        raise MeSHError("NLM root-tree lookup response was malformed.") from exc
    if not isinstance(root_bindings, list):
        raise MeSHError("NLM root-tree lookup returned no binding array.")
    for entry in root_bindings:
        tree = _tree_segment(_value(entry, "rootTree"))
        if tree not in root_paths:
            root_paths.append(tree)
    if not root_paths:
        # Descriptors without tree numbers: retain the resolved descriptor
        # without inventing any narrower terms.
        root_paths = [""]
    positions = list(rows.values())
    for tree in root_paths:
        positions.append(DescriptorPosition(
            descriptor=resolved_name, mesh_id=mesh_id,
            tree_number=tree, root_tree=tree, level=0, is_root=True
        ))
    positions.sort(key=lambda p: (p.root_tree, p.tree_number, p.level, p.descriptor.casefold()))
    return MeSHExpansion(
        heading=heading, resolved_name=resolved_name, mesh_id=mesh_id,
        positions=tuple(positions),
        fetched_utc=datetime.now(timezone.utc).isoformat(),
    )


def display_rows(expansion: MeSHExpansion) -> list[dict]:
    return [
        {
            "Level": row.level,
            "Descriptor": row.descriptor,
            "MeSH ID": row.mesh_id,
            "Root tree": row.root_tree,
            "Tree number": row.tree_number,
            "Relation": "Source descriptor" if row.is_root else "Narrower descriptor",
            "NLM": row.mesh_url,
        }
        for row in expansion.positions
    ]


def csv_export(expansion: MeSHExpansion) -> bytes:
    """Export ALL tree positions (including multiple placements), UTF-8 BOM."""
    f = io.StringIO(newline="")
    columns = ["Level", "Descriptor", "MeSH ID", "Root tree",
               "Tree number", "Relation", "NLM"]
    writer = csv.DictWriter(f, fieldnames=columns)
    writer.writeheader()
    writer.writerows(display_rows(expansion))
    return f.getvalue().encode("utf-8-sig")


def labels_as_text(expansion: MeSHExpansion) -> str:
    """One preferred descriptor label per distinct ID, root first."""
    seen = {expansion.mesh_id}
    result = [expansion.resolved_name]
    for entry in expansion.positions:
        if entry.mesh_id not in seen:
            seen.add(entry.mesh_id)
            result.append(entry.descriptor)
    return "\n".join(result)
