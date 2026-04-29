import unittest
from pathlib import Path

import LooLoLo


ROOT = Path(__file__).resolve().parents[1]
DEMO = ROOT / "demo"


class LocationParsingTests(unittest.TestCase):
    def test_non_alias_scan_ignores_alloc_substrings(self) -> None:
        text = """
#loc0 = loc("/tmp/test.py":1:0)
%0 = memref.alloc() : memref<1xi32> loc(#loc0)
""".strip()

        aliases, alias_ranges = LooLoLo.parse_aliases(text)
        locations = LooLoLo.parse_non_alias_locations(text, alias_ranges)

        self.assertEqual(len(locations), 1)
        bundle = LooLoLo.resolve_location(locations[0].expr, aliases)
        self.assertEqual(
            bundle.spans,
            {LooLoLo.SourceSpan(file="/tmp/test.py", start_line=1, start_col=0)},
        )

    def test_fused_and_callsite_locations_are_flattened(self) -> None:
        expr = LooLoLo.parse_loc_expression(
            'fused["a.cc":1:2, callsite("callee"("b.cc":3:4) at "c.cc":5:6)]'
        )

        bundle = LooLoLo.resolve_location(expr, aliases={})

        self.assertEqual(bundle.names, {"callee"})
        self.assertEqual(
            bundle.spans,
            {
                LooLoLo.SourceSpan(file="a.cc", start_line=1, start_col=2),
                LooLoLo.SourceSpan(file="b.cc", start_line=3, start_col=4),
                LooLoLo.SourceSpan(file="c.cc", start_line=5, start_col=6),
            },
        )


class DemoPipelineTests(unittest.TestCase):
    def test_demo_pipeline_explicit_order(self) -> None:
        analysis = LooLoLo.build_pipeline_analysis(
            source_file=DEMO / "test.py",
            mlir_paths=[DEMO / "kernel.ttir.mlir", DEMO / "kernel.ttadapter.mlir"],
            infer_stage_order_from_richness=False,
        )

        self.assertFalse(analysis.order_inferred)
        self.assertEqual([stage.path.name for stage in analysis.stages], ["kernel.ttir.mlir", "kernel.ttadapter.mlir"])
        self.assertEqual(analysis.reference_stage.path.name, "kernel.ttir.mlir")
        self.assertEqual([scope.qualname for scope in analysis.python_scopes], ["add_kernel"])
        self.assertEqual(analysis.scope_gap_lines, {7, 8, 9, 10, 11, 12})

        reference = analysis.reference_stage
        adapter = analysis.stages[1]
        mlir_source = "/workspace/tests/test.py"

        self.assertEqual(reference.source_lines, {6, 13, 15, 16, 18, 19, 21, 23})
        self.assertEqual(reference.source_names - adapter.source_names, {"offsets", "pid"})
        self.assertEqual(
            reference.source_spans - adapter.source_spans,
            {
                LooLoLo.SourceSpan(file=mlir_source, start_line=13, start_col=24),
                LooLoLo.SourceSpan(file=mlir_source, start_line=16, start_col=28),
                LooLoLo.SourceSpan(file=mlir_source, start_line=16, start_col=41),
                LooLoLo.SourceSpan(file=mlir_source, start_line=23, start_col=4),
            },
        )

    def test_demo_pipeline_infers_richer_stage_as_reference(self) -> None:
        analysis = LooLoLo.build_pipeline_analysis(
            source_file=DEMO / "test.py",
            mlir_paths=[DEMO / "kernel.ttadapter.mlir", DEMO / "kernel.ttir.mlir"],
            infer_stage_order_from_richness=True,
        )

        self.assertTrue(analysis.order_inferred)
        self.assertEqual([stage.path.name for stage in analysis.stages], ["kernel.ttir.mlir", "kernel.ttadapter.mlir"])
        self.assertEqual(analysis.reference_stage.path.name, "kernel.ttir.mlir")

    def test_demo_line_report_entries(self) -> None:
        analysis = LooLoLo.build_pipeline_analysis(
            source_file=DEMO / "test.py",
            mlir_paths=[DEMO / "kernel.ttir.mlir", DEMO / "kernel.ttadapter.mlir"],
            infer_stage_order_from_richness=False,
        )

        entries = LooLoLo.build_line_report_entries(analysis)

        self.assertEqual(entries[5].source_text, "def add_kernel(")
        self.assertEqual(entries[5].present_in, ("kernel.ttir.mlir", "kernel.ttadapter.mlir"))
        self.assertEqual(entries[5].lost_in, ())
        self.assertEqual(
            tuple(stage_name for stage_name, _ in entries[5].present_evidence),
            ("kernel.ttir.mlir", "kernel.ttadapter.mlir"),
        )

        self.assertEqual(entries[12].source_text, "    pid = tl.program_id(axis = 0)")
        self.assertEqual(entries[12].present_in, ("kernel.ttir.mlir",))
        self.assertEqual(entries[12].lost_in, ("kernel.ttadapter.mlir",))
        self.assertEqual(entries[12].present_evidence[0][1][0].line_number, 9)
        self.assertIn("tt.get_program_id", entries[12].present_evidence[0][1][0].text)

        self.assertEqual(entries[25].source_text, "def vector_add(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:")
        self.assertEqual(entries[25].present_in, ())
        self.assertEqual(entries[25].lost_in, ("kernel.ttir.mlir", "kernel.ttadapter.mlir"))

    def test_demo_line_report_rendering(self) -> None:
        analysis = LooLoLo.build_pipeline_analysis(
            source_file=DEMO / "test.py",
            mlir_paths=[DEMO / "kernel.ttir.mlir", DEMO / "kernel.ttadapter.mlir"],
            infer_stage_order_from_richness=False,
        )

        rendered = LooLoLo.render_line_report(analysis)

        self.assertIn(
            "def add_kernel(\n→ Present in: kernel.ttir.mlir, kernel.ttadapter.mlir\n   - kernel.ttir.mlir: 7:   tt.func public @add_kernel(",
            rendered,
        )
        self.assertIn(
            "    pid = tl.program_id(axis = 0)\n→ Present in: kernel.ttir.mlir\n   - kernel.ttir.mlir: 9:     %pid = tt.get_program_id x : i32 loc(#loc18)\n→ Lost in: kernel.ttadapter.mlir",
            rendered,
        )
        self.assertIn(
            "def vector_add(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:\n→ Present in: none\n→ Lost in: kernel.ttir.mlir, kernel.ttadapter.mlir",
            rendered,
        )
        self.assertIn("Summary\nTotal original lines analyzed: 52", rendered)
        self.assertIn("Average preservation rate across all stages:", rendered)

    def test_demo_line_report_ignores_empty_lines_and_computes_summary(self) -> None:
        analysis = LooLoLo.build_pipeline_analysis(
            source_file=DEMO / "test.py",
            mlir_paths=[DEMO / "kernel.ttir.mlir", DEMO / "kernel.ttadapter.mlir"],
            infer_stage_order_from_richness=False,
        )

        entries = LooLoLo.build_line_report_entries(
            analysis,
            ignore_empty_lines=True,
            lowered_scope_only=False,
        )
        summary = LooLoLo.build_report_summary(entries, len(analysis.stages), analysis.stages[-1].path.name)

        self.assertEqual(len(entries), 35)
        self.assertTrue(all(not LooLoLo.is_empty_source_line(entry.source_text) for entry in entries))
        self.assertEqual(summary.total_original_lines_analyzed, 35)
        self.assertEqual(summary.lines_fully_preserved_until_end, 6)
        self.assertEqual(summary.lines_present_in_final_stage, 6)
        self.assertEqual(summary.lines_partially_preserved, 2)
        self.assertEqual(summary.lines_completely_lost, 27)
        self.assertAlmostEqual(summary.average_preservation_rate, 20.0)
        self.assertAlmostEqual(summary.final_stage_preservation_rate, 17.142857142857142)

    def test_demo_lowered_scope_summary_reduces_false_source_losses(self) -> None:
        analysis = LooLoLo.build_pipeline_analysis(
            source_file=DEMO / "test.py",
            mlir_paths=[DEMO / "kernel.ttir.mlir", DEMO / "kernel.ttadapter.mlir"],
            infer_stage_order_from_richness=False,
        )

        entries = LooLoLo.build_line_report_entries(
            analysis,
            ignore_empty_lines=True,
            lowered_scope_only=True,
        )
        generated_ops = sum(
            1
            for stage in analysis.stages
            for op in stage.ops
            if op.location_status == "GENERATED"
        )
        summary = LooLoLo.build_report_summary(
            entries,
            len(analysis.stages),
            analysis.stages[-1].path.name,
            generated_unknown_ops=generated_ops,
        )

        self.assertEqual([entry.line_number for entry in entries], [6, 7, 8, 9, 10, 11, 12, 13, 15, 16, 18, 19, 21, 23])
        self.assertEqual(summary.total_original_lines_analyzed, 14)
        self.assertEqual(summary.lines_fully_preserved_until_end, 6)
        self.assertEqual(summary.lines_present_in_final_stage, 6)
        self.assertEqual(summary.lines_partially_preserved, 2)
        self.assertEqual(summary.lines_completely_lost, 6)
        self.assertAlmostEqual(summary.average_preservation_rate, 50.0)
        self.assertAlmostEqual(summary.final_stage_preservation_rate, 42.857142857142854)
        self.assertEqual(summary.generated_unknown_ops, 1)

    def test_demo_operation_records_capture_status_and_fingerprints(self) -> None:
        analysis = LooLoLo.build_pipeline_analysis(
            source_file=DEMO / "test.py",
            mlir_paths=[DEMO / "kernel.ttir.mlir", DEMO / "kernel.ttadapter.mlir"],
            infer_stage_order_from_richness=False,
        )

        ttir = analysis.stages[0]
        adapter = analysis.stages[1]
        pid_ops = ttir.ops_by_source_line[13]
        unknown_ops = [op for op in ttir.ops if op.op_name == "arith.constant"]

        self.assertEqual(pid_ops[0].op_name, "tt.get_program_id")
        self.assertEqual(pid_ops[0].location_status, "OK")
        self.assertEqual(pid_ops[0].source_lines, (13,))
        self.assertEqual(len(pid_ops[0].semantic_fingerprint), 12)
        self.assertNotIn(13, adapter.ops_by_source_line)
        self.assertEqual(unknown_ops[0].location_status, "GENERATED")
        self.assertEqual(unknown_ops[0].source_lines, ())

    def test_demo_operation_report_rendering(self) -> None:
        analysis = LooLoLo.build_pipeline_analysis(
            source_file=DEMO / "test.py",
            mlir_paths=[DEMO / "kernel.ttir.mlir", DEMO / "kernel.ttadapter.mlir"],
            infer_stage_order_from_richness=False,
        )

        rendered = LooLoLo.render_operation_report(analysis, ignore_empty_lines=True)

        self.assertIn("Table: original_line -> MLIR_op -> stage -> location_status", rendered)
        self.assertIn("| original_line | source_ast_cols | function | MLIR_op | stage | location_status | source_loc | fingerprint | MLIR line |", rendered)
        self.assertIn("13:     pid = tl.program_id(axis = 0) | 4,10,24,31 | add_kernel | tt.get_program_id | kernel.ttir.mlir | OK", rendered)
        self.assertIn("13:     pid = tl.program_id(axis = 0) | 4,10,24,31 | add_kernel | - | kernel.ttadapter.mlir | LOST", rendered)
        self.assertIn("Top Dangerous Pass Transitions", rendered)
        self.assertIn("kernel.ttir.mlir -> kernel.ttadapter.mlir: lost_lines=2", rendered)
        self.assertIn("semantic_fingerprint_matches=", rendered)
        self.assertIn("Generated / Unanchored MLIR Ops", rendered)
        self.assertIn("kernel.ttir.mlir:8 arith.constant", rendered)

    def test_demo_html_report_and_terminal_summary(self) -> None:
        analysis = LooLoLo.build_pipeline_analysis(
            source_file=DEMO / "test.py",
            mlir_paths=[DEMO / "kernel.ttir.mlir", DEMO / "kernel.ttadapter.mlir"],
            infer_stage_order_from_richness=False,
        )
        html = LooLoLo.render_html_report(analysis, ignore_empty_lines=True)
        summary = LooLoLo.render_terminal_summary(
            analysis,
            DEMO / "test.loololo.html",
            ignore_empty_lines=True,
        )

        self.assertIn("<!doctype html>", html)
        self.assertIn("LooLoLo Operation Location Report", html)
        self.assertIn("<th>Original Line</th>", html)
        self.assertIn("tt.get_program_id", html)
        self.assertIn("Top Dangerous Pass Transitions", html)
        self.assertIn("HTML report:", summary)
        self.assertIn("Total original lines analyzed: 14", summary)
        self.assertIn("Generated/unanchored ops with unknown source: 1", summary)
        self.assertNotIn("tt.get_program_id", summary)


if __name__ == "__main__":
    unittest.main()
