# Architecture

## Overview

LooLoLo is a single-file Python CLI for analyzing how MLIR source-location
metadata changes across transformation stages. It reads an original source file
and one or more text `.mlir` files, extracts MLIR `loc(...)` attachments, maps
them back to source lines, and renders reports that show preserved, lost,
degraded, and generated/unanchored locations.

The implementation is intentionally evidence-based: MLIR locations are treated
as the source of truth for preservation. Python AST data is used to identify
source scopes and AST columns, not to invent expected MLIR locations.

## Repository Structure

- `LooLoLo.py` contains the CLI, parsers, analysis model, report generation, and entry point.
- `tests/test_loololo.py` verifies location parsing, stage analysis, scoped reporting, and report rendering.
- `demo/test.py` is the Python/Triton demo source.
- `demo/kernel.ttir.mlir` and `demo/kernel.ttadapter.mlir` are demo MLIR stages.
- `demo/test.loololo.html` and `demo/report.txt` are generated demo reports.

## Main Runtime Components

### CLI

`parse_args()` defines the command-line interface. `main()` resolves inputs,
builds a `PipelineAnalysis`, then renders one of three report modes:

- HTML report, which is the default.
- Text operation report with `--text-report`.
- Legacy source-line presence report with `--line-report`.

The default terminal output is intentionally short: summary statistics plus the
generated HTML path.

### MLIR Location Parser

The MLIR parser handles textual `loc(...)` forms used by the demo and tests:

- Alias definitions such as `#loc0 = loc(...)`.
- Inline non-alias `loc(...)` attachments.
- File locations like `"file.py":13:24`.
- Name locations.
- `fused[...]` locations.
- `callsite(...)` locations.
- Unknown locations.

The parser resolves aliases into `ResolvedBundle` values containing source
spans, names, and an `contains_unknown` flag.

### MLIR Operation Extraction

`build_mlir_op_records()` groups location attachments by MLIR line, extracts the
operation name, result SSA names, operand SSA names, and a normalized semantic
fingerprint.

Each operation receives a location status:

- `OK` when it points cleanly to the selected original source.
- `DEGRADED` when source evidence remains but precision is weakened.
- `LOST` when the location points outside the selected source.
- `GENERATED` when no original-source span is available, usually because the op is compiler-generated or unanchored.

### Python Source Scope Analysis

For Python sources, `ast` is used to collect function and class scopes and AST
column offsets. Scope detection is used to reduce false positives: by default,
operation and HTML reports analyze only the inferred lowered scope rather than
the full file.

If the user passes `--full-source`, every source line is included. If the user
passes `--scope`, the selected Python function or class is used explicitly.

### Pipeline Analysis

`build_pipeline_analysis()` coordinates the core analysis:

1. Parse each MLIR stage.
2. Choose the MLIR source path matching the local source file.
3. Build per-stage `StageAnalysis` records.
4. Preserve explicit stage order when `--stage` is used.
5. Infer order from source-location richness only when scanning a directory.
6. Select the Python lowered scope and compute source-scope coverage gaps.

`PipelineAnalysis` is the central object consumed by all report renderers.

### Report Renderers

There are three report paths:

- `render_html_report()` builds the main interactive-friendly report.
- `render_operation_report()` builds the text operation table.
- `render_line_report()` builds the legacy source-line presence report.

Summary values are produced through `build_summary_for_analysis()` and
`build_report_summary()`.

## Data Flow

```text
source file + MLIR stage files
        |
        v
parse MLIR loc aliases and inline loc attachments
        |
        v
resolve locations into source spans, names, and unknown flags
        |
        v
select MLIR source path matching the local source file
        |
        v
build per-stage source-line, snippet, and operation records
        |
        v
collect Python scopes and choose lowered source scope
        |
        v
build PipelineAnalysis
        |
        v
render HTML, text operation report, or line report
```

## Key Design Decisions

- MLIR location metadata is the primary evidence. The tool does not assume that every AST node must have a corresponding MLIR op.
- Unknown unanchored ops are classified as `GENERATED`, not automatically as `LOST`, to avoid false source-loss reports for compiler-created helper ops.
- Scoped analysis is the default for operation and HTML reports because full-file analysis can incorrectly mark imports, wrappers, and host code as lost.
- Explicit stage order is preferred. Inferred order is based on location richness and should be treated as approximate.
- Semantic fingerprints normalize SSA value names to support coarse cross-stage comparison without claiming full SSA lineage tracking.

## External Dependencies and Integrations

The current implementation uses only the Python standard library:

- `argparse` for CLI handling.
- `ast` for Python scope and column extraction.
- `re` for MLIR text scanning.
- `hashlib` for operation fingerprints.
- `html` for HTML escaping.
- `dataclasses` and `pathlib` for data modeling and filesystem paths.

There is no dependency on MLIR Python bindings. This keeps the tool easy to run,
but it also means parsing is textual and intentionally limited to the location
syntax the tool understands.

## Build and Validation Notes

Run syntax validation:

```bash
python3 -m py_compile LooLoLo.py tests/test_loololo.py
```

Run tests:

```bash
python3 -m unittest discover -s tests -v
```

Regenerate the demo HTML report:

```bash
python3 LooLoLo.py demo/test.py \
  --stage demo/kernel.ttir.mlir \
  --stage demo/kernel.ttadapter.mlir \
  --ignore-empty-lines
```

Regenerate the demo text report:

```bash
python3 LooLoLo.py demo/test.py \
  --stage demo/kernel.ttir.mlir \
  --stage demo/kernel.ttadapter.mlir \
  --ignore-empty-lines \
  --text-report > demo/report.txt
```

## Known Constraints

- MLIR parsing is textual. Unsupported or unusual location syntax can raise `AnalysisError`.
- The tool tracks explicit source-location evidence, not full semantic equivalence of lowered programs.
- SSA lineage tracking is shallow. Result names, operand names, and fingerprints are collected, but there is no full dataflow graph reconstruction yet.
- Pass attribution depends on the provided stage boundaries. If a stage dump is missing, the tool can only report the transition between available files.
- Inferred stage order is approximate. Use repeated `--stage` arguments for reliable pass ordering.
- Non-Python source files can still be analyzed through MLIR file locations, but Python-specific scope inference and AST columns are not available.
