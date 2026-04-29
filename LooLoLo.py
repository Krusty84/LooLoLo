#!/usr/bin/env python3
"""
LooLoLo - MLIR source location analyzer

This tool focuses on one reliable question:
which source locations are present in one MLIR stage and then disappear in
later stages?

Important design choices:
- File locations are taken from MLIR itself. The tool does not invent a
  synthetic "ground truth" set of AST token coordinates and compare MLIR
  against that.
- For Python sources, the AST is used only to identify the source scope being
  lowered and to report source-line coverage gaps inside that scope.
- Named locations are tracked stage-to-stage, because they are compiler-chosen
  metadata and cannot be treated as an exact source-language ground truth.

Created by Sedoykin Alexey on 29/04/2026.
"""

from __future__ import annotations

import argparse
import ast
import bisect
from collections import defaultdict
import hashlib
import html
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import DefaultDict, Dict, Iterable, List, Optional, Sequence, Set, Tuple


class AnalysisError(RuntimeError):
    """Raised when the input cannot be analyzed safely."""


class Colors:
    RESET = "\033[0m"
    BOLD = "\033[1m"
    GREEN = "\033[32m"
    RED = "\033[31m"
    YELLOW = "\033[33m"
    BLUE = "\033[34m"
    CYAN = "\033[36m"


def c(text: str, color: str = "", bold: bool = False, enable: Optional[bool] = None) -> str:
    if enable is None:
        enable = sys.stdout.isatty()
    if not enable:
        return text
    prefix = Colors.BOLD if bold else ""
    return f"{prefix}{color}{text}{Colors.RESET}"


@dataclass(frozen=True)
class SourceSpan:
    file: str
    start_line: int
    start_col: Optional[int] = None
    end_line: Optional[int] = None
    end_col: Optional[int] = None

    def normalized_end_line(self) -> int:
        return self.end_line if self.end_line is not None else self.start_line

    def normalized_end_col(self) -> Optional[int]:
        return self.end_col if self.end_col is not None else self.start_col

    def line_numbers(self) -> Set[int]:
        return set(range(self.start_line, self.normalized_end_line() + 1))

    def sort_key(self) -> Tuple[str, int, int, int, int]:
        return (
            Path(self.file).name,
            self.start_line,
            -1 if self.start_col is None else self.start_col,
            self.normalized_end_line(),
            -1 if self.normalized_end_col() is None else self.normalized_end_col(),
        )

    def display(self, basename_only: bool = True) -> str:
        file_label = Path(self.file).name if basename_only else self.file
        if self.start_col is None:
            if self.end_line and self.end_line != self.start_line:
                return f"{file_label}:{self.start_line} to {self.end_line}"
            return f"{file_label}:{self.start_line}"

        if self.end_line is None and self.end_col is None:
            return f"{file_label}:{self.start_line}:{self.start_col}"

        end_line = self.normalized_end_line()
        end_col = self.normalized_end_col()
        if end_line == self.start_line:
            if end_col == self.start_col:
                return f"{file_label}:{self.start_line}:{self.start_col}"
            return f"{file_label}:{self.start_line}:{self.start_col} to :{end_col}"
        return f"{file_label}:{self.start_line}:{self.start_col} to {end_line}:{end_col}"


@dataclass
class LocationExpr:
    kind: str
    raw: str
    name: Optional[str] = None
    span: Optional[SourceSpan] = None
    ref: Optional[str] = None
    children: List["LocationExpr"] = field(default_factory=list)


@dataclass
class ResolvedBundle:
    names: Set[str] = field(default_factory=set)
    spans: Set[SourceSpan] = field(default_factory=set)
    contains_unknown: bool = False

    def absorb(self, other: "ResolvedBundle") -> None:
        self.names.update(other.names)
        self.spans.update(other.spans)
        self.contains_unknown = self.contains_unknown or other.contains_unknown


@dataclass(frozen=True)
class PythonScope:
    name: str
    qualname: str
    kind: str
    start_line: int
    end_line: int

    def span_size(self) -> int:
        return self.end_line - self.start_line

    def line_numbers(self) -> Set[int]:
        return set(range(self.start_line, self.end_line + 1))


@dataclass
class ParsedStage:
    path: Path
    attachment_count: int
    resolved_attachments: List[ResolvedBundle]
    attachment_occurrences: List["LocationOccurrence"]
    all_spans: Set[SourceSpan]
    symbol_names: Set[str]


@dataclass(frozen=True)
class MlirSnippet:
    line_number: int
    text: str


@dataclass(frozen=True)
class LocationOccurrence:
    expr: LocationExpr
    line_number: int
    line_text: str


@dataclass(frozen=True)
class MlirOpRecord:
    stage_name: str
    op_name: str
    line_number: int
    text: str
    location_status: str
    source_lines: Tuple[int, ...]
    source_spans: Tuple[SourceSpan, ...]
    result_names: Tuple[str, ...]
    operand_names: Tuple[str, ...]
    semantic_fingerprint: str
    status_reason: str


@dataclass
class StageAnalysis:
    path: Path
    attachment_count: int
    matched_attachment_count: int
    matched_unknown_attachment_count: int
    source_spans: Set[SourceSpan]
    source_lines: Set[int]
    source_names: Set[str]
    symbol_names: Set[str]
    source_line_snippets: Dict[int, Tuple[MlirSnippet, ...]]
    ops: Tuple[MlirOpRecord, ...]
    ops_by_source_line: Dict[int, Tuple[MlirOpRecord, ...]]


@dataclass
class PipelineAnalysis:
    source_file: Path
    mlir_source_path: str
    stages: List[StageAnalysis]
    order_inferred: bool
    reference_stage: StageAnalysis
    python_scopes: List[PythonScope]
    scope_code_lines: Set[int]
    scope_gap_lines: Set[int]


@dataclass(frozen=True)
class LineReportEntry:
    line_number: int
    source_text: str
    present_in: Tuple[str, ...]
    lost_in: Tuple[str, ...]
    present_evidence: Tuple[Tuple[str, Tuple[MlirSnippet, ...]], ...]


@dataclass(frozen=True)
class ReportSummary:
    total_original_lines_analyzed: int
    lines_fully_preserved_until_end: int
    lines_present_in_final_stage: int
    lines_partially_preserved: int
    lines_completely_lost: int
    average_preservation_rate: float
    final_stage_preservation_rate: float
    generated_unknown_ops: int


@dataclass(frozen=True)
class DangerousPass:
    transition: str
    lost_line_count: int
    lost_op_count: int
    degraded_op_count: int
    fingerprint_match_count: int
    likely_reason: str


@dataclass(frozen=True)
class SourceLineInfo:
    line_number: int
    text: str
    function_name: Optional[str]
    ast_columns: Tuple[int, ...]
    in_lowered_scope: bool


def find_matching(text: str, open_index: int, open_char: str, close_char: str) -> int:
    depth = 0
    in_string = False
    escaped = False

    for index in range(open_index, len(text)):
        char = text[index]
        if in_string:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                in_string = False
            continue

        if char == '"':
            in_string = True
        elif char == open_char:
            depth += 1
        elif char == close_char:
            depth -= 1
            if depth == 0:
                return index

    raise AnalysisError(f"Unbalanced {open_char}{close_char} in MLIR text")


def skip_ranges(index: int, ranges: Sequence[Tuple[int, int]], cursor: int) -> Tuple[int, int]:
    while cursor < len(ranges) and index >= ranges[cursor][1]:
        cursor += 1
    if cursor < len(ranges):
        start, end = ranges[cursor]
        if start <= index < end:
            return end, cursor
    return index, cursor


def find_next_loc_start(text: str, start: int) -> int:
    in_string = False
    escaped = False
    index = start

    while index < len(text) - 3:
        char = text[index]
        if in_string:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                in_string = False
            index += 1
            continue

        if char == '"':
            in_string = True
            index += 1
            continue

        if text.startswith("loc(", index):
            if index == 0 or not (text[index - 1].isalnum() or text[index - 1] in "._$"):
                return index
        index += 1

    return -1


class LocationParser:
    def __init__(self, text: str):
        self.text = text
        self.index = 0

    def parse(self) -> LocationExpr:
        expr = self.parse_location()
        self.skip_ws()
        if self.index != len(self.text):
            raise AnalysisError(f"Unexpected trailing location text: {self.text[self.index:]}")
        return expr

    def skip_ws(self) -> None:
        while self.index < len(self.text) and self.text[self.index].isspace():
            self.index += 1

    def peek(self) -> Optional[str]:
        if self.index >= len(self.text):
            return None
        return self.text[self.index]

    def consume(self, expected: str) -> None:
        if not self.text.startswith(expected, self.index):
            raise AnalysisError(
                f"Expected {expected!r} at offset {self.index} in {self.text!r}"
            )
        self.index += len(expected)

    def parse_location(self) -> LocationExpr:
        self.skip_ws()
        start = self.index
        char = self.peek()

        if char == "#":
            ref = self.parse_ref()
            return LocationExpr(kind="ref", raw=self.text[start:self.index], ref=ref)

        if char == "?":
            self.index += 1
            return LocationExpr(kind="unknown", raw=self.text[start:self.index])

        if self.text.startswith("unknown", self.index):
            self.consume("unknown")
            return LocationExpr(kind="unknown", raw=self.text[start:self.index])

        if self.text.startswith("callsite(", self.index):
            return self.parse_callsite()

        if self.text.startswith("fused", self.index):
            return self.parse_fused()

        if char == '"':
            return self.parse_string_based()

        raise AnalysisError(f"Unsupported location form: {self.text[start:]}")

    def parse_ref(self) -> str:
        start = self.index
        self.consume("#")
        while self.index < len(self.text):
            char = self.text[self.index]
            if char.isalnum() or char in "._$-":
                self.index += 1
                continue
            break
        return self.text[start:self.index]

    def parse_string(self) -> str:
        if self.peek() != '"':
            raise AnalysisError(f"Expected string literal in {self.text!r}")
        self.index += 1
        chunks: List[str] = []
        escaped = False
        while self.index < len(self.text):
            char = self.text[self.index]
            self.index += 1
            if escaped:
                chunks.append(char)
                escaped = False
                continue
            if char == "\\":
                escaped = True
                continue
            if char == '"':
                return "".join(chunks)
            chunks.append(char)
        raise AnalysisError(f"Unterminated string literal in {self.text!r}")

    def parse_int(self) -> int:
        start = self.index
        while self.index < len(self.text) and self.text[self.index].isdigit():
            self.index += 1
        if start == self.index:
            raise AnalysisError(f"Expected integer in {self.text!r}")
        return int(self.text[start:self.index])

    def parse_optional_int(self) -> Optional[int]:
        if self.peek() is None or not self.peek().isdigit():
            return None
        return self.parse_int()

    def parse_string_based(self) -> LocationExpr:
        start = self.index
        value = self.parse_string()
        self.skip_ws()

        if self.peek() == "(":
            self.consume("(")
            child = self.parse_location()
            self.skip_ws()
            self.consume(")")
            return LocationExpr(
                kind="name",
                raw=self.text[start:self.index],
                name=value,
                children=[child],
            )

        if self.peek() == ":":
            span = self.parse_file_span(value)
            return LocationExpr(kind="file", raw=self.text[start:self.index], span=span)

        return LocationExpr(kind="name", raw=self.text[start:self.index], name=value)

    def parse_file_span(self, file_name: str) -> SourceSpan:
        self.consume(":")
        start_line = self.parse_int()
        start_col: Optional[int] = None
        end_line: Optional[int] = None
        end_col: Optional[int] = None

        if self.peek() == ":":
            self.consume(":")
            start_col = self.parse_optional_int()

        self.skip_ws()
        if self.text.startswith("to", self.index):
            self.consume("to")
            self.skip_ws()
            explicit_end_line = self.parse_optional_int()
            self.consume(":")
            explicit_end_col = self.parse_optional_int()
            end_line = explicit_end_line if explicit_end_line is not None else start_line
            end_col = explicit_end_col

        return SourceSpan(
            file=file_name,
            start_line=start_line,
            start_col=start_col,
            end_line=end_line,
            end_col=end_col,
        )

    def parse_callsite(self) -> LocationExpr:
        start = self.index
        self.consume("callsite(")
        callee = self.parse_location()
        self.skip_ws()
        self.consume("at")
        caller = self.parse_location()
        self.skip_ws()
        self.consume(")")
        return LocationExpr(
            kind="callsite",
            raw=self.text[start:self.index],
            children=[callee, caller],
        )

    def parse_fused(self) -> LocationExpr:
        start = self.index
        self.consume("fused")
        self.skip_ws()
        if self.peek() == "<":
            end = find_matching(self.text, self.index, "<", ">")
            self.index = end + 1
            self.skip_ws()

        self.consume("[")
        children: List[LocationExpr] = []
        while True:
            self.skip_ws()
            if self.peek() == "]":
                break
            children.append(self.parse_location())
            self.skip_ws()
            if self.peek() == ",":
                self.consume(",")
                continue
            if self.peek() == "]":
                break
            raise AnalysisError(f"Malformed fused location list: {self.text!r}")
        self.consume("]")
        return LocationExpr(kind="fused", raw=self.text[start:self.index], children=children)


ALIAS_DEF_RE = re.compile(r"(?m)^\s*(#[A-Za-z0-9_.$-]+)\s*=\s*loc\(")
MLIR_SYMBOL_RE = re.compile(
    r"\b(?:func\.func|tt\.func)\b(?:\s+(?:public|private|nested))?\s+@([A-Za-z0-9_.$-]+)"
)


def parse_loc_expression(fragment: str) -> LocationExpr:
    return LocationParser(fragment).parse()


def parse_aliases(content: str) -> Tuple[Dict[str, LocationExpr], List[Tuple[int, int]]]:
    aliases: Dict[str, LocationExpr] = {}
    ranges: List[Tuple[int, int]] = []

    for match in ALIAS_DEF_RE.finditer(content):
        alias_name = match.group(1)
        loc_start = match.end() - 1
        loc_end = find_matching(content, loc_start, "(", ")")
        fragment = content[loc_start + 1:loc_end]
        aliases[alias_name] = parse_loc_expression(fragment)
        ranges.append((match.start(), loc_end + 1))

    ranges.sort()
    return aliases, ranges


def parse_non_alias_locations(content: str, alias_ranges: Sequence[Tuple[int, int]]) -> List[LocationOccurrence]:
    locations: List[LocationOccurrence] = []
    index = 0
    range_cursor = 0
    line_starts = [0]
    for match in re.finditer(r"\n", content):
        line_starts.append(match.end())
    content_lines = content.splitlines()

    while True:
        loc_start = find_next_loc_start(content, index)
        if loc_start == -1:
            break

        advanced, range_cursor = skip_ranges(loc_start, alias_ranges, range_cursor)
        if advanced != loc_start:
            index = advanced
            continue

        open_index = loc_start + len("loc")
        close_index = find_matching(content, open_index, "(", ")")
        fragment = content[open_index + 1:close_index]
        expr = parse_loc_expression(fragment)
        line_index = bisect.bisect_right(line_starts, loc_start) - 1
        line_number = line_index + 1
        line_text = content_lines[line_index] if line_index < len(content_lines) else ""
        locations.append(
            LocationOccurrence(
                expr=expr,
                line_number=line_number,
                line_text=line_text,
            )
        )
        index = close_index + 1

    return locations


def resolve_location(
    expr: LocationExpr,
    aliases: Dict[str, LocationExpr],
    trail: Optional[Set[str]] = None,
) -> ResolvedBundle:
    trail = set() if trail is None else trail

    if expr.kind == "ref":
        if expr.ref not in aliases:
            raise AnalysisError(f"Unknown location alias reference: {expr.ref}")
        if expr.ref in trail:
            raise AnalysisError(f"Recursive location alias reference: {expr.ref}")
        return resolve_location(aliases[expr.ref], aliases, trail | {expr.ref})

    bundle = ResolvedBundle()

    if expr.kind == "unknown":
        bundle.contains_unknown = True
        return bundle

    if expr.kind == "file":
        if expr.span is None:
            raise AnalysisError("File location is missing its span")
        bundle.spans.add(expr.span)
        return bundle

    if expr.kind == "name":
        if expr.name is not None:
            bundle.names.add(expr.name)
        for child in expr.children:
            bundle.absorb(resolve_location(child, aliases, trail))
        return bundle

    if expr.kind in {"callsite", "fused"}:
        for child in expr.children:
            bundle.absorb(resolve_location(child, aliases, trail))
        return bundle

    raise AnalysisError(f"Unsupported resolved location kind: {expr.kind}")


SSA_VALUE_RE = re.compile(r"%[A-Za-z0-9_.$-]+")
MLIR_OP_RE = re.compile(r"^([A-Za-z_][A-Za-z0-9_.$-]*)\b")
MLIR_RESULT_ASSIGN_RE = re.compile(
    r"^\s*((?:%[A-Za-z0-9_.$-]+(?:\s*,\s*)?)+)\s*="
)


def location_contains_kind(expr: LocationExpr, kind: str) -> bool:
    if expr.kind == kind:
        return True
    return any(location_contains_kind(child, kind) for child in expr.children)


def strip_mlir_location_suffix(text: str) -> str:
    loc_index = find_next_loc_start(text, 0)
    if loc_index == -1:
        return text.strip()
    return text[:loc_index].rstrip()


def extract_mlir_op_name(line_text: str) -> Optional[str]:
    stripped = line_text.strip()
    if not stripped or stripped.startswith("#"):
        return None
    if stripped.startswith("}") or stripped.startswith(")"):
        return None

    before_loc = strip_mlir_location_suffix(stripped)
    assignment = MLIR_RESULT_ASSIGN_RE.match(before_loc)
    candidate = before_loc[assignment.end():].strip() if assignment else before_loc

    match = MLIR_OP_RE.match(candidate)
    if not match:
        return None
    return match.group(1)


def extract_mlir_result_names(line_text: str) -> Tuple[str, ...]:
    before_loc = strip_mlir_location_suffix(line_text.strip())
    assignment = MLIR_RESULT_ASSIGN_RE.match(before_loc)
    if not assignment:
        return ()
    return tuple(SSA_VALUE_RE.findall(assignment.group(1)))


def extract_mlir_operand_names(line_text: str) -> Tuple[str, ...]:
    before_loc = strip_mlir_location_suffix(line_text.strip())
    assignment = MLIR_RESULT_ASSIGN_RE.match(before_loc)
    if assignment:
        before_loc = before_loc[assignment.end():]
    return tuple(SSA_VALUE_RE.findall(before_loc))


def semantic_fingerprint(op_name: str, line_text: str) -> str:
    normalized = strip_mlir_location_suffix(line_text)
    assignment = MLIR_RESULT_ASSIGN_RE.match(normalized)
    if assignment:
        normalized = normalized[assignment.end():]
    normalized = SSA_VALUE_RE.sub("%", normalized)
    normalized = re.sub(r"\s+", " ", normalized).strip()
    digest = hashlib.sha1(f"{op_name}|{normalized}".encode("utf-8")).hexdigest()
    return digest[:12]


def classify_location_status(
    exprs: Sequence[LocationExpr],
    bundles: Sequence[ResolvedBundle],
    mlir_source_path: str,
) -> Tuple[str, Set[SourceSpan], str]:
    source_spans = {
        span
        for bundle in bundles
        for span in bundle.spans
        if span.file == mlir_source_path
    }
    has_unknown = any(bundle.contains_unknown for bundle in bundles)
    has_any_span = any(bundle.spans for bundle in bundles)
    has_fused = any(location_contains_kind(expr, "fused") for expr in exprs)
    has_callsite = any(location_contains_kind(expr, "callsite") for expr in exprs)
    has_missing_columns = any(
        span.start_col is None
        for span in source_spans
    )

    if not source_spans:
        if has_unknown:
            return "GENERATED", set(), "generated or compiler-introduced op with unknown location"
        if has_any_span:
            return "LOST", set(), "location points outside the selected source file"
        return "GENERATED", set(), "generated or compiler-introduced op with no original source span"

    if has_unknown or has_fused or has_callsite or has_missing_columns:
        reasons: List[str] = []
        if has_unknown:
            reasons.append("contains unknown")
        if has_fused:
            reasons.append("uses fused location")
        if has_callsite:
            reasons.append("uses callsite location")
        if has_missing_columns:
            reasons.append("missing column precision")
        return "DEGRADED", source_spans, ", ".join(reasons)

    return "OK", source_spans, "points to original source"


def build_mlir_op_records(
    stage_path: Path,
    occurrences: Sequence[LocationOccurrence],
    bundles: Sequence[ResolvedBundle],
    mlir_source_path: str,
) -> List[MlirOpRecord]:
    grouped: DefaultDict[int, List[Tuple[LocationOccurrence, ResolvedBundle]]] = defaultdict(list)
    for occurrence, bundle in zip(occurrences, bundles):
        grouped[occurrence.line_number].append((occurrence, bundle))

    records: List[MlirOpRecord] = []
    for line_number in sorted(grouped):
        occurrence_bundles = grouped[line_number]
        line_text = occurrence_bundles[0][0].line_text
        op_name = extract_mlir_op_name(line_text)
        if op_name is None:
            continue

        exprs = [occurrence.expr for occurrence, _ in occurrence_bundles]
        resolved_bundles = [bundle for _, bundle in occurrence_bundles]
        status, source_spans, reason = classify_location_status(
            exprs,
            resolved_bundles,
            mlir_source_path,
        )
        source_lines = sorted({
            line
            for span in source_spans
            for line in span.line_numbers()
        })
        records.append(
            MlirOpRecord(
                stage_name=stage_path.name,
                op_name=op_name,
                line_number=line_number,
                text=line_text,
                location_status=status,
                source_lines=tuple(source_lines),
                source_spans=tuple(sorted(source_spans, key=SourceSpan.sort_key)),
                result_names=extract_mlir_result_names(line_text),
                operand_names=extract_mlir_operand_names(line_text),
                semantic_fingerprint=semantic_fingerprint(op_name, line_text),
                status_reason=reason,
            )
        )

    return records


def parse_mlir_stage(path: Path) -> ParsedStage:
    content = path.read_text(encoding="utf-8")
    aliases, alias_ranges = parse_aliases(content)
    occurrences = parse_non_alias_locations(content, alias_ranges)
    resolved = [resolve_location(occurrence.expr, aliases) for occurrence in occurrences]
    all_spans = {span for bundle in resolved for span in bundle.spans}
    symbol_names = set(MLIR_SYMBOL_RE.findall(content))

    return ParsedStage(
        path=path,
        attachment_count=len(occurrences),
        resolved_attachments=resolved,
        attachment_occurrences=occurrences,
        all_spans=all_spans,
        symbol_names=symbol_names,
    )


def choose_mlir_source_path(
    source_file: Path,
    parsed_stages: Sequence[ParsedStage],
    explicit_path: Optional[str],
) -> str:
    mlir_paths = sorted({span.file for stage in parsed_stages for span in stage.all_spans})
    if not mlir_paths:
        raise AnalysisError("No file locations were found in the provided MLIR stages")

    if explicit_path is not None:
        exact_matches = [path for path in mlir_paths if path == explicit_path]
        if exact_matches:
            return exact_matches[0]
        suffix_matches = [path for path in mlir_paths if path.endswith(explicit_path)]
        if len(suffix_matches) == 1:
            return suffix_matches[0]
        raise AnalysisError(
            f"MLIR source path {explicit_path!r} was not found uniquely. Candidates: {', '.join(mlir_paths)}"
        )

    exact_matches = [path for path in mlir_paths if path == source_file.as_posix()]
    if len(exact_matches) == 1:
        return exact_matches[0]

    basename_matches = sorted({path for path in mlir_paths if Path(path).name == source_file.name})
    if len(basename_matches) == 1:
        return basename_matches[0]
    if len(basename_matches) > 1:
        raise AnalysisError(
            "Multiple MLIR source paths share the same basename. "
            "Use --mlir-source-path to choose one explicitly: "
            + ", ".join(basename_matches)
        )

    raise AnalysisError(
        f"No MLIR file locations match source basename {source_file.name!r}. "
        f"Available MLIR source paths: {', '.join(mlir_paths)}"
    )


def build_stage_analysis(stage: ParsedStage, mlir_source_path: str) -> StageAnalysis:
    matched_bundles: List[ResolvedBundle] = []
    source_line_snippets: Dict[int, Set[MlirSnippet]] = {}
    ops = build_mlir_op_records(
        stage.path,
        stage.attachment_occurrences,
        stage.resolved_attachments,
        mlir_source_path,
    )
    ops_by_source_line: Dict[int, List[MlirOpRecord]] = {}
    for op in ops:
        for line_number in op.source_lines:
            ops_by_source_line.setdefault(line_number, []).append(op)

    for occurrence, bundle in zip(stage.attachment_occurrences, stage.resolved_attachments):
        matched_spans_for_occurrence = [
            span for span in bundle.spans
            if span.file == mlir_source_path
        ]
        if not matched_spans_for_occurrence:
            continue
        matched_bundles.append(bundle)
        snippet = MlirSnippet(
            line_number=occurrence.line_number,
            text=occurrence.line_text,
        )
        for span in matched_spans_for_occurrence:
            for line_number in span.line_numbers():
                source_line_snippets.setdefault(line_number, set()).add(snippet)

    matched_spans = {
        span
        for bundle in matched_bundles
        for span in bundle.spans
        if span.file == mlir_source_path
    }
    matched_lines = {line for span in matched_spans for line in span.line_numbers()}
    matched_names = {name for bundle in matched_bundles for name in bundle.names}
    matched_unknown = sum(1 for bundle in matched_bundles if bundle.contains_unknown)

    return StageAnalysis(
        path=stage.path,
        attachment_count=stage.attachment_count,
        matched_attachment_count=len(matched_bundles),
        matched_unknown_attachment_count=matched_unknown,
        source_spans=matched_spans,
        source_lines=matched_lines,
        source_names=matched_names,
        symbol_names=stage.symbol_names,
        source_line_snippets={
            line_number: tuple(
                sorted(
                    snippets,
                    key=lambda snippet: (snippet.line_number, snippet.text),
                )
            )
            for line_number, snippets in source_line_snippets.items()
        },
        ops=tuple(ops),
        ops_by_source_line={
            line_number: tuple(
                sorted(records, key=lambda op: (op.line_number, op.op_name, op.text))
            )
            for line_number, records in ops_by_source_line.items()
        },
    )


def infer_stage_order(stages: Sequence[StageAnalysis]) -> List[StageAnalysis]:
    return sorted(
        stages,
        key=lambda stage: (
            -len(stage.source_spans),
            -len(stage.source_lines),
            -len(stage.source_names),
            stage.path.name,
        ),
    )


def collect_python_scopes(tree: ast.AST) -> List[PythonScope]:
    scopes: List[PythonScope] = []

    def visit(node: ast.AST, parents: List[str]) -> None:
        for child in ast.iter_child_nodes(node):
            if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                qualname = ".".join(parents + [child.name]) if parents else child.name
                scopes.append(
                    PythonScope(
                        name=child.name,
                        qualname=qualname,
                        kind=type(child).__name__,
                        start_line=child.lineno,
                        end_line=child.end_lineno or child.lineno,
                    )
                )
                visit(child, parents + [child.name])
            else:
                visit(child, parents)

    visit(tree, [])
    return scopes


def choose_python_scopes(
    source_file: Path,
    reference_stage: StageAnalysis,
    explicit_scope: Optional[str],
) -> List[PythonScope]:
    if source_file.suffix.lower() != ".py":
        return []

    tree = ast.parse(source_file.read_text(encoding="utf-8"), filename=str(source_file))
    scopes = collect_python_scopes(tree)
    if not scopes:
        return []

    if explicit_scope is not None:
        selected = [
            scope for scope in scopes
            if scope.name == explicit_scope or scope.qualname == explicit_scope
        ]
        if not selected:
            raise AnalysisError(f"Python scope {explicit_scope!r} was not found in {source_file}")
        return sorted(selected, key=lambda scope: (scope.span_size(), scope.qualname))

    symbol_matches = [
        scope for scope in scopes
        if scope.name in reference_stage.symbol_names or scope.qualname in reference_stage.symbol_names
    ]
    if len(symbol_matches) == 1:
        return symbol_matches

    if reference_stage.source_lines:
        covering = [
            scope for scope in scopes
            if reference_stage.source_lines.issubset(scope.line_numbers())
        ]
        if covering:
            best = min(covering, key=lambda scope: (scope.span_size(), scope.qualname))
            return [best]

        overlapping = [
            scope for scope in scopes
            if reference_stage.source_lines & scope.line_numbers()
        ]
        if overlapping:
            return sorted(overlapping, key=lambda scope: (scope.span_size(), scope.qualname))

    return []


def scope_code_lines(source_file: Path, scopes: Sequence[PythonScope]) -> Set[int]:
    if not scopes:
        return set()
    source_lines = source_file.read_text(encoding="utf-8").splitlines()
    selected_lines: Set[int] = set()
    for scope in scopes:
        for line_number in range(scope.start_line, scope.end_line + 1):
            line_text = source_lines[line_number - 1]
            stripped = line_text.strip()
            if stripped and not stripped.startswith("#"):
                selected_lines.add(line_number)
    return selected_lines


def build_pipeline_analysis(
    source_file: Path,
    mlir_paths: Sequence[Path],
    infer_stage_order_from_richness: bool = False,
    mlir_source_path: Optional[str] = None,
    scope_name: Optional[str] = None,
) -> PipelineAnalysis:
    parsed_stages = [parse_mlir_stage(path) for path in mlir_paths]
    selected_mlir_source_path = choose_mlir_source_path(source_file, parsed_stages, mlir_source_path)
    analyzed_stages = [
        build_stage_analysis(stage, selected_mlir_source_path)
        for stage in parsed_stages
    ]

    if not mlir_paths:
        raise AnalysisError("No MLIR stages were provided")

    order_inferred = False
    if infer_stage_order_from_richness and len(mlir_paths) > 1:
        ordered_stages = infer_stage_order(analyzed_stages)
        order_inferred = True
    else:
        stage_by_path = {stage.path: stage for stage in analyzed_stages}
        ordered_stages = [stage_by_path[path] for path in mlir_paths]

    reference_stage = ordered_stages[0]
    python_scopes = choose_python_scopes(source_file, reference_stage, scope_name)
    covered_scope_lines = scope_code_lines(source_file, python_scopes)
    scope_gap_lines = covered_scope_lines - reference_stage.source_lines

    return PipelineAnalysis(
        source_file=source_file,
        mlir_source_path=selected_mlir_source_path,
        stages=ordered_stages,
        order_inferred=order_inferred,
        reference_stage=reference_stage,
        python_scopes=python_scopes,
        scope_code_lines=covered_scope_lines,
        scope_gap_lines=scope_gap_lines,
    )


def is_empty_source_line(source_text: str) -> bool:
    return source_text.strip() == ""


def build_line_report_entries(
    analysis: PipelineAnalysis,
    ignore_empty_lines: bool = False,
    lowered_scope_only: bool = False,
) -> List[LineReportEntry]:
    stage_names = tuple(stage.path.name for stage in analysis.stages)
    source_lines = analysis.source_file.read_text(encoding="utf-8").splitlines()
    entries: List[LineReportEntry] = []
    allowed_lines = analysis.scope_code_lines if lowered_scope_only else None

    for line_number, source_text in enumerate(source_lines, start=1):
        if ignore_empty_lines and is_empty_source_line(source_text):
            continue
        if allowed_lines is not None and line_number not in allowed_lines:
            continue

        present_evidence = tuple(
            (
                stage.path.name,
                stage.source_line_snippets.get(line_number, ()),
            )
            for stage in analysis.stages
            if line_number in stage.source_lines
        )
        present_in = tuple(stage_name for stage_name, _ in present_evidence)
        lost_in = tuple(
            stage_name for stage_name in stage_names
            if stage_name not in present_in
        )
        entries.append(
            LineReportEntry(
                line_number=line_number,
                source_text=source_text,
                present_in=present_in,
                lost_in=lost_in,
                present_evidence=present_evidence,
            )
        )

    return entries


def display_source_line(source_text: str) -> str:
    if source_text == "":
        return "[blank line]"
    if source_text.strip() == "":
        return f"[whitespace-only line: {len(source_text)} spaces]"
    return source_text


def format_stage_list(stage_names: Sequence[str]) -> str:
    if not stage_names:
        return "none"
    return ", ".join(stage_names)


def build_report_summary(
    entries: Sequence[LineReportEntry],
    stage_count: int,
    final_stage_name: Optional[str],
    generated_unknown_ops: int = 0,
) -> ReportSummary:
    total_lines = len(entries)
    lines_fully_preserved = sum(1 for entry in entries if len(entry.present_in) == stage_count)
    lines_present_in_final_stage = sum(
        1
        for entry in entries
        if final_stage_name is not None and final_stage_name in entry.present_in
    )
    lines_completely_lost = sum(1 for entry in entries if not entry.present_in)
    lines_partially_preserved = total_lines - lines_fully_preserved - lines_completely_lost
    average_preservation_rate = 0.0
    final_stage_preservation_rate = 0.0
    if total_lines and stage_count:
        average_preservation_rate = (
            sum(len(entry.present_in) / stage_count for entry in entries) / total_lines * 100.0
        )
        final_stage_preservation_rate = (
            lines_present_in_final_stage
            / total_lines
            * 100.0
        )

    return ReportSummary(
        total_original_lines_analyzed=total_lines,
        lines_fully_preserved_until_end=lines_fully_preserved,
        lines_present_in_final_stage=lines_present_in_final_stage,
        lines_partially_preserved=lines_partially_preserved,
        lines_completely_lost=lines_completely_lost,
        average_preservation_rate=average_preservation_rate,
        final_stage_preservation_rate=final_stage_preservation_rate,
        generated_unknown_ops=generated_unknown_ops,
    )


def render_mlir_snippet(snippet: MlirSnippet) -> str:
    return f"{snippet.line_number}: {snippet.text}"


def render_line_report(
    analysis: PipelineAnalysis,
    ignore_empty_lines: bool = False,
    lowered_scope_only: bool = False,
    enable_color: Optional[bool] = None,
) -> str:
    if enable_color is None:
        enable_color = sys.stdout.isatty()

    entries = build_line_report_entries(
        analysis,
        ignore_empty_lines=ignore_empty_lines,
        lowered_scope_only=lowered_scope_only,
    )
    final_stage_name = analysis.stages[-1].path.name if analysis.stages else None
    summary = build_report_summary(entries, len(analysis.stages), final_stage_name)
    rendered: List[str] = []
    present_label = c("Present", Colors.GREEN, bold=True, enable=enable_color)
    lost_label = c("Lost", Colors.RED, bold=True, enable=enable_color)

    for index, entry in enumerate(entries):
        rendered.append(c(display_source_line(entry.source_text), Colors.CYAN, bold=True, enable=enable_color))
        rendered.append(f"→ {present_label} in: {format_stage_list(entry.present_in)}")
        for stage_name, snippets in entry.present_evidence:
            for snippet in snippets:
                rendered.append(f"   - {stage_name}: {render_mlir_snippet(snippet)}")
        rendered.append(f"→ {lost_label} in: {format_stage_list(entry.lost_in)}")
        if index != len(entries) - 1:
            rendered.append("")

    rendered.append("")
    rendered.append(c("Summary", Colors.BLUE, bold=True, enable=enable_color))
    rendered.append(f"Total original lines analyzed: {summary.total_original_lines_analyzed}")
    rendered.append(f"Lines fully preserved until the end: {summary.lines_fully_preserved_until_end}")
    rendered.append(f"Lines present in final stage: {summary.lines_present_in_final_stage}")
    rendered.append(f"Lines partially preserved: {summary.lines_partially_preserved}")
    rendered.append(f"Lines completely lost: {summary.lines_completely_lost}")
    rendered.append(f"Average preservation rate across all stages: {summary.average_preservation_rate:.2f}%")
    rendered.append(f"Final-stage preservation rate: {summary.final_stage_preservation_rate:.2f}%")
    if ignore_empty_lines:
        rendered.append("Empty and whitespace-only source lines were ignored.")
    if lowered_scope_only:
        rendered.append("Only inferred lowered source scope lines were analyzed.")

    return "\n".join(rendered)


def build_source_line_info(
    source_file: Path,
    analysis: Optional[PipelineAnalysis] = None,
    ignore_empty_lines: bool = False,
    lowered_scope_only: bool = False,
) -> List[SourceLineInfo]:
    source_lines = source_file.read_text(encoding="utf-8").splitlines()
    function_by_line: Dict[int, Optional[str]] = {}
    columns_by_line: Dict[int, Set[int]] = {}

    if source_file.suffix.lower() == ".py":
        tree = ast.parse(source_file.read_text(encoding="utf-8"), filename=str(source_file))
        scopes = sorted(collect_python_scopes(tree), key=lambda scope: scope.span_size())
        for node in ast.walk(tree):
            if hasattr(node, "lineno") and hasattr(node, "col_offset"):
                columns_by_line.setdefault(node.lineno, set()).add(node.col_offset)
        for line_number in range(1, len(source_lines) + 1):
            matching = [
                scope for scope in scopes
                if scope.start_line <= line_number <= scope.end_line
            ]
            function_by_line[line_number] = matching[0].qualname if matching else None

    lowered_scope_lines = analysis.scope_code_lines if analysis is not None else set()
    infos: List[SourceLineInfo] = []
    for line_number, text in enumerate(source_lines, start=1):
        if ignore_empty_lines and is_empty_source_line(text):
            continue
        in_lowered_scope = line_number in lowered_scope_lines if analysis is not None else True
        if lowered_scope_only and not in_lowered_scope:
            continue
        infos.append(
            SourceLineInfo(
                line_number=line_number,
                text=text,
                function_name=function_by_line.get(line_number),
                ast_columns=tuple(sorted(columns_by_line.get(line_number, set()))),
                in_lowered_scope=in_lowered_scope,
            )
        )
    return infos


def escape_table_cell(text: str) -> str:
    return text.replace("|", "\\|").replace("\n", " ")


def format_status(status: str, enable_color: bool) -> str:
    if status == "OK":
        return c("OK", Colors.GREEN, bold=True, enable=enable_color)
    if status == "LOST":
        return c("LOST", Colors.RED, bold=True, enable=enable_color)
    if status == "DEGRADED":
        return c("DEGRADED", Colors.YELLOW, bold=True, enable=enable_color)
    if status == "GENERATED":
        return c("GENERATED", Colors.BLUE, bold=True, enable=enable_color)
    return status


def source_location_summary(op: MlirOpRecord) -> str:
    if not op.source_spans:
        return "-"
    return ", ".join(span.display() for span in op.source_spans)


def source_column_summary(info: SourceLineInfo) -> str:
    if not info.ast_columns:
        return "-"
    return ",".join(str(column) for column in info.ast_columns)


def build_lost_line_events(
    line_infos: Sequence[SourceLineInfo],
    stages: Sequence[StageAnalysis],
) -> List[Tuple[SourceLineInfo, str]]:
    events: List[Tuple[SourceLineInfo, str]] = []
    if not stages:
        return events

    for info in line_infos:
        present_stage_indexes = [
            index
            for index, stage in enumerate(stages)
            if info.line_number in stage.source_lines
        ]
        if not present_stage_indexes:
            events.append((info, f"before {stages[0].path.name}"))
            continue

        for index in range(present_stage_indexes[0] + 1, len(stages)):
            if info.line_number not in stages[index].source_lines:
                events.append((info, f"{stages[index - 1].path.name} -> {stages[index].path.name}"))
                break
    return events


def summarize_dangerous_passes(stages: Sequence[StageAnalysis]) -> List[DangerousPass]:
    summaries: List[DangerousPass] = []
    for previous, current in zip(stages, stages[1:]):
        lost_lines = previous.source_lines - current.source_lines
        current_lines = current.source_lines
        lost_ops = sum(
            1
            for op in previous.ops
            if op.source_lines and not (set(op.source_lines) & current_lines)
        )
        degraded_ops = sum(1 for op in current.ops if op.location_status != "OK")
        previous_fingerprints = {op.semantic_fingerprint for op in previous.ops}
        current_fingerprints = {op.semantic_fingerprint for op in current.ops}
        fingerprint_matches = len(previous_fingerprints & current_fingerprints)
        transition = f"{previous.path.name} -> {current.path.name}"
        reasons: List[str] = []
        if lost_lines or lost_ops:
            reasons.append("replacement/lowering likely failed to propagate original locs")
        if degraded_ops:
            reasons.append("new or fused locations need review")
        if not reasons:
            reasons.append("no source-line loss detected")
        summaries.append(
            DangerousPass(
                transition=transition,
                lost_line_count=len(lost_lines),
                lost_op_count=lost_ops,
                degraded_op_count=degraded_ops,
                fingerprint_match_count=fingerprint_matches,
                likely_reason="; ".join(reasons),
            )
        )

    return sorted(
        summaries,
        key=lambda item: (item.lost_line_count, item.lost_op_count, item.degraded_op_count),
        reverse=True,
    )[:5]


def render_operation_report(
    analysis: PipelineAnalysis,
    ignore_empty_lines: bool = False,
    lowered_scope_only: bool = True,
    enable_color: Optional[bool] = None,
) -> str:
    if enable_color is None:
        enable_color = sys.stdout.isatty()

    line_infos = build_source_line_info(
        analysis.source_file,
        analysis=analysis,
        ignore_empty_lines=ignore_empty_lines,
        lowered_scope_only=lowered_scope_only,
    )
    lost_line_events = build_lost_line_events(line_infos, analysis.stages)
    dangerous_passes = summarize_dangerous_passes(analysis.stages)
    rendered: List[str] = []

    rendered.append(c("Operation Location Report", Colors.BLUE, bold=True, enable=enable_color))
    rendered.append(f"Source: {analysis.source_file}")
    rendered.append(f"MLIR source path: {analysis.mlir_source_path}")
    if analysis.order_inferred:
        rendered.append("Stage order: inferred from source-location richness; pass attribution is approximate.")
    else:
        rendered.append("Stage order: explicit.")
    rendered.append("")

    rendered.append("Table: original_line -> MLIR_op -> stage -> location_status")
    rendered.append("| original_line | source_ast_cols | function | MLIR_op | stage | location_status | source_loc | fingerprint | MLIR line |")
    rendered.append("| --- | --- | --- | --- | --- | --- | --- | --- | --- |")

    for info in line_infos:
        source_label = f"{info.line_number}: {display_source_line(info.text)}"
        for stage in analysis.stages:
            ops = stage.ops_by_source_line.get(info.line_number, ())
            if not ops:
                rendered.append(
                    "| "
                    + " | ".join(
                        [
                            escape_table_cell(c(source_label, Colors.CYAN, bold=True, enable=enable_color)),
                            source_column_summary(info),
                            escape_table_cell(info.function_name or "-"),
                            "-",
                            stage.path.name,
                            format_status("LOST", enable_color),
                            "-",
                            "-",
                            "line not present in this stage",
                        ]
                    )
                    + " |"
                )
                continue

            for op in ops:
                rendered.append(
                    "| "
                    + " | ".join(
                        [
                            escape_table_cell(c(source_label, Colors.CYAN, bold=True, enable=enable_color)),
                            source_column_summary(info),
                            escape_table_cell(info.function_name or "-"),
                            escape_table_cell(op.op_name),
                            stage.path.name,
                            format_status(op.location_status, enable_color),
                            escape_table_cell(source_location_summary(op)),
                            op.semantic_fingerprint,
                            escape_table_cell(f"{op.line_number}: {op.text}"),
                        ]
                    )
                    + " |"
                )

    rendered.append("")
    rendered.append(c("Lost Original Lines", Colors.RED, bold=True, enable=enable_color))
    if lost_line_events:
        for info, disappeared_at in lost_line_events:
            function_label = f" [{info.function_name}]" if info.function_name else ""
            rendered.append(
                f"- line {info.line_number}{function_label}: {display_source_line(info.text)} -> {disappeared_at}"
            )
    else:
        rendered.append("- none")

    rendered.append("")
    rendered.append(c("Top Dangerous Pass Transitions", Colors.YELLOW, bold=True, enable=enable_color))
    if dangerous_passes:
        for index, item in enumerate(dangerous_passes, start=1):
            rendered.append(
                f"{index}. {item.transition}: lost_lines={item.lost_line_count}, "
                f"lost_ops={item.lost_op_count}, degraded_or_unanchored_ops={item.degraded_op_count}; "
                f"semantic_fingerprint_matches={item.fingerprint_match_count}; "
                f"likely_reason={item.likely_reason}"
            )
    else:
        rendered.append("1. No adjacent stage transition available.")

    generated_ops = [
        op
        for stage in analysis.stages
        for op in stage.ops
        if op.location_status == "GENERATED"
    ]
    rendered.append("")
    rendered.append(c("Generated / Unanchored MLIR Ops", Colors.BLUE, bold=True, enable=enable_color))
    if generated_ops:
        for op in generated_ops[:20]:
            rendered.append(
                f"- {op.stage_name}:{op.line_number} {op.op_name} -> {op.status_reason}: {op.text}"
            )
        if len(generated_ops) > 20:
            rendered.append(f"- ... and {len(generated_ops) - 20} more")
    else:
        rendered.append("- none")

    line_entries = build_line_report_entries(
        analysis,
        ignore_empty_lines=ignore_empty_lines,
        lowered_scope_only=lowered_scope_only,
    )
    summary = build_report_summary(
        line_entries,
        len(analysis.stages),
        analysis.stages[-1].path.name if analysis.stages else None,
        generated_unknown_ops=sum(
            1
            for stage in analysis.stages
            for op in stage.ops
            if op.location_status == "GENERATED"
        ),
    )
    rendered.append("")
    rendered.append(c("Summary", Colors.BLUE, bold=True, enable=enable_color))
    rendered.append(f"Total original lines analyzed: {summary.total_original_lines_analyzed}")
    rendered.append(f"Lines fully preserved until the end: {summary.lines_fully_preserved_until_end}")
    rendered.append(f"Lines present in final stage: {summary.lines_present_in_final_stage}")
    rendered.append(f"Lines partially preserved: {summary.lines_partially_preserved}")
    rendered.append(f"Lines completely lost: {summary.lines_completely_lost}")
    rendered.append(f"Generated/unanchored ops with unknown source: {summary.generated_unknown_ops}")
    rendered.append(f"Average preservation rate across all stages: {summary.average_preservation_rate:.2f}%")
    rendered.append(f"Final-stage preservation rate: {summary.final_stage_preservation_rate:.2f}%")

    rendered.append("")
    rendered.append(c("Fix Suggestions", Colors.GREEN, bold=True, enable=enable_color))
    rendered.append("- In custom lowering/rewrite passes, create replacement ops with the source op location or an explicit FusedLoc that includes the original source FileLineColLoc.")
    rendered.append("- For generated helper ops, use a NameLoc/FusedLoc rooted in the original op instead of loc(unknown).")
    rendered.append("- Add pass-level assertions or tests that reject newly created operations with unknown locations after each major lowering boundary.")
    rendered.append("- Keep explicit stage dumps in pass order; automatic pass attribution from filenames is approximate.")
    if ignore_empty_lines:
        rendered.append("- Empty and whitespace-only source lines were ignored.")
    if lowered_scope_only:
        rendered.append("- Only inferred lowered source scope lines were analyzed.")

    return "\n".join(rendered)


def html_escape(value: object) -> str:
    return html.escape(str(value), quote=True)


def status_class(status: str) -> str:
    return status.lower()


def default_html_report_path(source_file: Path) -> Path:
    return source_file.with_name(f"{source_file.stem}.loololo.html")


def build_summary_for_analysis(
    analysis: PipelineAnalysis,
    ignore_empty_lines: bool = False,
    lowered_scope_only: bool = False,
) -> ReportSummary:
    entries = build_line_report_entries(
        analysis,
        ignore_empty_lines=ignore_empty_lines,
        lowered_scope_only=lowered_scope_only,
    )
    generated_unknown_ops = sum(
        1
        for stage in analysis.stages
        for op in stage.ops
        if op.location_status == "GENERATED"
    )
    return build_report_summary(
        entries,
        len(analysis.stages),
        analysis.stages[-1].path.name if analysis.stages else None,
        generated_unknown_ops=generated_unknown_ops,
    )


def render_html_report(
    analysis: PipelineAnalysis,
    ignore_empty_lines: bool = False,
    lowered_scope_only: bool = True,
) -> str:
    line_infos = build_source_line_info(
        analysis.source_file,
        analysis=analysis,
        ignore_empty_lines=ignore_empty_lines,
        lowered_scope_only=lowered_scope_only,
    )
    lost_line_events = build_lost_line_events(line_infos, analysis.stages)
    dangerous_passes = summarize_dangerous_passes(analysis.stages)
    generated_ops = [
        op
        for stage in analysis.stages
        for op in stage.ops
        if op.location_status == "GENERATED"
    ]
    summary = build_summary_for_analysis(
        analysis,
        ignore_empty_lines=ignore_empty_lines,
        lowered_scope_only=lowered_scope_only,
    )

    stage_order_note = (
        "Stage order inferred from source-location richness; pass attribution is approximate."
        if analysis.order_inferred
        else "Stage order is explicit."
    )

    rows: List[str] = []
    for info in line_infos:
        source_label = f"{info.line_number}: {display_source_line(info.text)}"
        for stage in analysis.stages:
            ops = stage.ops_by_source_line.get(info.line_number, ())
            if not ops:
                rows.append(
                    "<tr>"
                    f"<td><code>{html_escape(source_label)}</code></td>"
                    f"<td>{html_escape(source_column_summary(info))}</td>"
                    f"<td>{html_escape(info.function_name or '-')}</td>"
                    "<td>-</td>"
                    f"<td>{html_escape(stage.path.name)}</td>"
                    '<td><span class="badge lost">LOST</span></td>'
                    "<td>-</td>"
                    "<td>-</td>"
                    "<td>line not present in this stage</td>"
                    "</tr>"
                )
                continue

            for op in ops:
                rows.append(
                    "<tr>"
                    f"<td><code>{html_escape(source_label)}</code></td>"
                    f"<td>{html_escape(source_column_summary(info))}</td>"
                    f"<td>{html_escape(info.function_name or '-')}</td>"
                    f"<td><code>{html_escape(op.op_name)}</code></td>"
                    f"<td>{html_escape(stage.path.name)}</td>"
                    f'<td><span class="badge {status_class(op.location_status)}">{html_escape(op.location_status)}</span></td>'
                    f"<td>{html_escape(source_location_summary(op))}</td>"
                    f"<td><code>{html_escape(op.semantic_fingerprint)}</code></td>"
                    f"<td><code>{html_escape(f'{op.line_number}: {op.text}')}</code></td>"
                    "</tr>"
                )

    lost_items = [
        (
            f"<li><strong>line {info.line_number}</strong>"
            f"{' [' + html_escape(info.function_name) + ']' if info.function_name else ''}: "
            f"<code>{html_escape(display_source_line(info.text))}</code> "
            f"<span>lost at {html_escape(disappeared_at)}</span></li>"
        )
        for info, disappeared_at in lost_line_events
    ]
    if not lost_items:
        lost_items = ["<li>none</li>"]

    dangerous_items = [
        (
            f"<li><strong>{html_escape(item.transition)}</strong>: "
            f"lost lines {item.lost_line_count}, lost ops {item.lost_op_count}, "
            f"degraded or unanchored ops {item.degraded_op_count}, "
            f"fingerprint matches {item.fingerprint_match_count}. "
            f"{html_escape(item.likely_reason)}</li>"
        )
        for item in dangerous_passes
    ]
    if not dangerous_items:
        dangerous_items = ["<li>No adjacent stage transition available.</li>"]

    generated_items = [
        (
            f"<li><strong>{html_escape(op.stage_name)}:{op.line_number} {html_escape(op.op_name)}</strong>: "
            f"{html_escape(op.status_reason)}<br><code>{html_escape(op.text)}</code></li>"
        )
        for op in generated_ops[:50]
    ]
    if len(generated_ops) > 50:
        generated_items.append(f"<li>... and {len(generated_ops) - 50} more</li>")
    if not generated_items:
        generated_items = ["<li>none</li>"]

    empty_note = (
        "<p class=\"note\">Empty and whitespace-only source lines were ignored.</p>"
        if ignore_empty_lines
        else ""
    )
    scope_note = (
        "<p class=\"note\">Only inferred lowered source scope lines were analyzed.</p>"
        if lowered_scope_only
        else "<p class=\"note\">Full source file was analyzed.</p>"
    )

    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>LooLoLo Location Report - {html_escape(analysis.source_file.name)}</title>
  <style>
    :root {{
      --bg: #f6f7f2;
      --panel: #ffffff;
      --ink: #17201b;
      --muted: #5a665f;
      --line: #d9ded6;
      --ok: #137a3f;
      --lost: #b42318;
      --degraded: #946200;
      --generated: #315f8c;
      --code: #102018;
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      background: var(--bg);
      color: var(--ink);
      font: 14px/1.45 ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
    }}
    header {{
      padding: 28px 32px 20px;
      border-bottom: 1px solid var(--line);
      background: #eef1ea;
    }}
    h1 {{ margin: 0 0 8px; font-size: 24px; }}
    h2 {{ margin: 28px 0 12px; font-size: 18px; }}
    main {{ padding: 0 32px 40px; }}
    code {{
      color: var(--code);
      font: 12px/1.4 ui-monospace, SFMono-Regular, Menlo, Consolas, monospace;
      white-space: pre-wrap;
    }}
    .meta, .note {{ color: var(--muted); margin: 4px 0; }}
    .summary {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(190px, 1fr));
      gap: 12px;
      margin: 20px 0 8px;
    }}
    .metric {{
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 6px;
      padding: 12px;
    }}
    .metric strong {{ display: block; font-size: 20px; margin-bottom: 2px; }}
    .table-wrap {{
      overflow-x: auto;
      border: 1px solid var(--line);
      background: var(--panel);
      border-radius: 6px;
    }}
    table {{ width: 100%; border-collapse: collapse; min-width: 1200px; }}
    th, td {{
      padding: 8px 10px;
      border-bottom: 1px solid var(--line);
      text-align: left;
      vertical-align: top;
    }}
    th {{
      position: sticky;
      top: 0;
      background: #e7ebe3;
      z-index: 1;
      font-size: 12px;
      text-transform: uppercase;
      color: var(--muted);
    }}
    .badge {{
      display: inline-block;
      min-width: 72px;
      padding: 2px 8px;
      border-radius: 999px;
      color: #fff;
      font-weight: 700;
      text-align: center;
      font-size: 12px;
    }}
    .ok {{ background: var(--ok); }}
    .lost {{ background: var(--lost); }}
    .degraded {{ background: var(--degraded); }}
    .generated {{ background: var(--generated); }}
    ul {{ padding-left: 20px; }}
    li {{ margin: 7px 0; }}
  </style>
</head>
<body>
  <header>
    <h1>LooLoLo Operation Location Report</h1>
    <p class="meta">Source: {html_escape(analysis.source_file)}</p>
    <p class="meta">MLIR source path: {html_escape(analysis.mlir_source_path)}</p>
    <p class="meta">{html_escape(stage_order_note)}</p>
  </header>
  <main>
    <section class="summary">
      <div class="metric"><strong>{summary.total_original_lines_analyzed}</strong>Total original lines analyzed</div>
      <div class="metric"><strong>{summary.lines_fully_preserved_until_end}</strong>Lines fully preserved until the end</div>
      <div class="metric"><strong>{summary.lines_present_in_final_stage}</strong>Lines present in final stage</div>
      <div class="metric"><strong>{summary.lines_completely_lost}</strong>Lines completely lost</div>
      <div class="metric"><strong>{summary.generated_unknown_ops}</strong>Generated/unanchored unknown ops</div>
      <div class="metric"><strong>{summary.average_preservation_rate:.2f}%</strong>Average preservation rate</div>
      <div class="metric"><strong>{summary.final_stage_preservation_rate:.2f}%</strong>Final-stage preservation rate</div>
    </section>
    {empty_note}
    {scope_note}

    <h2>Operation Table</h2>
    <div class="table-wrap">
      <table>
        <thead>
          <tr>
            <th>Original Line</th>
            <th>AST Cols</th>
            <th>Function</th>
            <th>MLIR Op</th>
            <th>Stage</th>
            <th>Status</th>
            <th>Source Loc</th>
            <th>Fingerprint</th>
            <th>MLIR Line</th>
          </tr>
        </thead>
        <tbody>
          {''.join(rows)}
        </tbody>
      </table>
    </div>

    <h2>Lost Original Lines</h2>
    <ul>{''.join(lost_items)}</ul>

    <h2>Top Dangerous Pass Transitions</h2>
    <ol>{''.join(dangerous_items)}</ol>

    <h2>Generated / Unanchored MLIR Ops</h2>
    <ul>{''.join(generated_items)}</ul>

    <h2>Fix Suggestions</h2>
    <ul>
      <li>In custom lowering/rewrite passes, create replacement ops with the source op location or an explicit FusedLoc that includes the original source FileLineColLoc.</li>
      <li>For generated helper ops, use a NameLoc/FusedLoc rooted in the original op instead of loc(unknown).</li>
      <li>Add pass-level assertions or tests that reject newly created operations with unknown locations after each major lowering boundary.</li>
      <li>Keep explicit stage dumps in pass order; automatic pass attribution from filenames is approximate.</li>
    </ul>
  </main>
</body>
</html>
"""


def render_terminal_summary(
    analysis: PipelineAnalysis,
    html_report_path: Path,
    ignore_empty_lines: bool = False,
    lowered_scope_only: bool = True,
) -> str:
    summary = build_summary_for_analysis(
        analysis,
        ignore_empty_lines=ignore_empty_lines,
        lowered_scope_only=lowered_scope_only,
    )
    dangerous = summarize_dangerous_passes(analysis.stages)
    top_transition = dangerous[0].transition if dangerous else "n/a"
    return "\n".join(
        [
            "LooLoLo summary",
            f"Total original lines analyzed: {summary.total_original_lines_analyzed}",
            f"Lines fully preserved until the end: {summary.lines_fully_preserved_until_end}",
            f"Lines completely lost: {summary.lines_completely_lost}",
            f"Generated/unanchored ops with unknown source: {summary.generated_unknown_ops}",
            f"Average preservation rate across all stages: {summary.average_preservation_rate:.2f}%",
            f"Top dangerous transition: {top_transition}",
            f"HTML report: {html_report_path}",
        ]
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Analyze source-location preservation across MLIR stages"
    )
    parser.add_argument("source_file", type=Path, help="Original source file")
    parser.add_argument(
        "--mlir-dir",
        type=Path,
        default=None,
        help="Directory to scan for *.mlir files when --stage is not provided",
    )
    parser.add_argument(
        "--stage",
        type=Path,
        action="append",
        default=[],
        help="MLIR stage file in pipeline order. Repeat to define the full pipeline.",
    )
    parser.add_argument(
        "--mlir-source-path",
        type=str,
        default=None,
        help="Exact or suffix MLIR source path to match when the local path differs",
    )
    parser.add_argument(
        "--scope",
        type=str,
        default=None,
        help="Python function/class scope to analyze explicitly",
    )
    parser.add_argument(
        "--ignore-empty-lines",
        action="store_true",
        help="Skip blank and whitespace-only lines from the original source report",
    )
    parser.add_argument(
        "--full-source",
        action="store_true",
        help="Analyze the full source file instead of only the inferred lowered scope",
    )
    parser.add_argument(
        "--line-report",
        action="store_true",
        help="Print the legacy source-line presence report instead of the operation table",
    )
    parser.add_argument(
        "--text-report",
        action="store_true",
        help="Print the operation report as text instead of generating HTML",
    )
    parser.add_argument(
        "--html-report",
        type=Path,
        default=None,
        help="Path for the generated HTML report. Defaults to <source>.loololo.html next to the source file.",
    )
    return parser.parse_args()


def resolve_mlir_paths(args: argparse.Namespace, source_file: Path) -> List[Path]:
    if args.stage:
        mlir_paths = [path.resolve() for path in args.stage]
    else:
        mlir_dir = args.mlir_dir.resolve() if args.mlir_dir else source_file.parent
        mlir_paths = sorted(mlir_dir.glob("*.mlir"))

    if not mlir_paths:
        raise AnalysisError("No MLIR stage files were found")

    missing = [path for path in mlir_paths if not path.exists()]
    if missing:
        raise AnalysisError("Missing MLIR stage files: " + ", ".join(str(path) for path in missing))

    return mlir_paths


def main() -> int:
    args = parse_args()
    source_file = args.source_file.resolve()
    if not source_file.exists():
        print(c(f"Error: source file not found: {source_file}", Colors.RED))
        return 1

    try:
        mlir_paths = resolve_mlir_paths(args, source_file)
        analysis = build_pipeline_analysis(
            source_file=source_file,
            mlir_paths=mlir_paths,
            infer_stage_order_from_richness=not bool(args.stage),
            mlir_source_path=args.mlir_source_path,
            scope_name=args.scope,
        )
        if args.line_report:
            print(
                render_line_report(
                    analysis,
                    ignore_empty_lines=args.ignore_empty_lines,
                    lowered_scope_only=not args.full_source,
                )
            )
        elif args.text_report:
            print(
                render_operation_report(
                    analysis,
                    ignore_empty_lines=args.ignore_empty_lines,
                    lowered_scope_only=not args.full_source,
                )
            )
        else:
            html_report_path = (
                args.html_report.resolve()
                if args.html_report is not None
                else default_html_report_path(source_file)
            )
            html_report_path.parent.mkdir(parents=True, exist_ok=True)
            html_report_path.write_text(
                render_html_report(
                    analysis,
                    ignore_empty_lines=args.ignore_empty_lines,
                    lowered_scope_only=not args.full_source,
                ),
                encoding="utf-8",
            )
            print(
                render_terminal_summary(
                    analysis,
                    html_report_path,
                    ignore_empty_lines=args.ignore_empty_lines,
                    lowered_scope_only=not args.full_source,
                )
            )
    except AnalysisError as exc:
        print(c(f"Error: {exc}", Colors.RED))
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
