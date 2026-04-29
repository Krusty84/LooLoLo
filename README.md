# LooLoLo

LooLoLo is a command-line analyzer for tracking source-location metadata across
MLIR transformation stages. It answers a practical question: which original
source lines still have MLIR locations at each stage, and where do those
locations disappear, degrade, or become unanchored?

The current implementation is optimized for reliability over speed. It uses the
locations already emitted in MLIR as evidence, avoids inventing synthetic source
truth, and separates compiler-generated unknown-location operations from true
source-line loss.

## What It Reports

- Original source lines and whether they are present or lost in each MLIR stage.
- MLIR operations associated with each source line.
- Location status per operation: `OK`, `LOST`, `DEGRADED`, or `GENERATED`.
- Exact MLIR snippets where a source location is preserved.
- Top stage transitions where source lines or operation locations disappear.
- Summary statistics, including preservation rates and generated/unanchored ops.
- HTML report by default, with a short CLI summary.

## Repository Layout

- `LooLoLo.py` - main CLI and analysis implementation.
- `demo/test.py` - Python/Triton demo source.
- `demo/*.mlir` - demo MLIR stages.
- `demo/test.loololo.html` - generated demo HTML report.
- `demo/report.txt` - generated demo text report.
- `tests/test_loololo.py` - unit tests for parsing, analysis, and reporting.

## Requirements

LooLoLo currently uses only the Python standard library.

Tested with Python 3.12 in this workspace.

## Quick Start

Generate the demo HTML report:

```bash
python3 LooLoLo.py demo/test.py \
  --stage demo/kernel.ttir.mlir \
  --stage demo/kernel.ttadapter.mlir \
  --ignore-empty-lines
```

The CLI prints only a summary and the generated HTML path:

```text
LooLoLo summary
Total original lines analyzed: 14
Lines fully preserved until the end: 6
Lines completely lost: 6
Generated/unanchored ops with unknown source: 1
Average preservation rate across all stages: 50.00%
Top dangerous transition: kernel.ttir.mlir -> kernel.ttadapter.mlir
HTML report: demo/test.loololo.html
```

Generate a text operation report instead:

```bash
python3 LooLoLo.py demo/test.py \
  --stage demo/kernel.ttir.mlir \
  --stage demo/kernel.ttadapter.mlir \
  --ignore-empty-lines \
  --text-report
```

Generate the legacy source-line presence report:

```bash
python3 LooLoLo.py demo/test.py \
  --stage demo/kernel.ttir.mlir \
  --stage demo/kernel.ttadapter.mlir \
  --ignore-empty-lines \
  --line-report
```

## CLI Options

- `source_file` - original source file to analyze.
- `--stage PATH` - MLIR stage file in pipeline order. Repeat for multiple stages.
- `--mlir-dir PATH` - scan a directory for `*.mlir` files when `--stage` is not provided.
- `--mlir-source-path PATH` - exact or suffix source path as it appears inside MLIR locations.
- `--scope NAME` - analyze a specific Python function or class scope.
- `--ignore-empty-lines` - skip blank and whitespace-only source lines.
- `--full-source` - analyze the entire source file instead of only the inferred lowered scope.
- `--text-report` - print the operation report instead of generating HTML.
- `--line-report` - print the legacy line-presence report.
- `--html-report PATH` - choose the output HTML path.

## Default Scope Behavior

By default, operation and HTML reports analyze only the inferred lowered source
scope. For the demo, this means the Triton kernel body is analyzed instead of
also counting imports, host wrapper code, and `__main__` setup as lost MLIR
locations.

Use `--full-source` when you intentionally want every non-empty source line to be
included in the statistics.

## Location Statuses

- `OK` - the operation has a location pointing to the selected source file with usable line and column information.
- `DEGRADED` - the operation still references the source, but the location is fused, callsite-based, partially unknown, or missing column precision.
- `LOST` - the line or operation no longer points to the selected original source.
- `GENERATED` - the operation has `loc(unknown)` or no resolvable original-source span and is treated as compiler-generated or unanchored, not as direct evidence of a lost source line.

## Validation

Run the test suite:

```bash
python3 -m unittest discover -s tests -v
```

Run syntax validation:

```bash
python3 -m py_compile LooLoLo.py tests/test_loololo.py
```

## Current Limits

LooLoLo is evidence-based. It is reliable for detecting disappearance of MLIR
locations that explicitly reference the selected source file, but it does not yet
perform full SSA lineage reconstruction, dominance-aware value tracking, or
deep semantic matching across arbitrary lowerings.

When stage files are not passed explicitly with `--stage`, ordering is inferred
from source-location richness. That is useful for demos, but explicit stage order
is more reliable for pass attribution.
