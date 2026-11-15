from dataclasses import asdict
import io
import json
import logging
import math
import os
from pathlib import Path
import tempfile
import unittest
import unittest.mock

from matplotlib.colors import LogNorm
import matplotlib.pyplot as plt
import numpy as np
from streamlit.testing.v1 import AppTest

from src.agents.cad_canvas import apply_canvas_action, build_canvas_payload
from src.agents.csg import CADModel
from src.agents.fea import FEAResult, FEASolver
from src.agents.problem import (
    HARBOR_TASK_BUILDERS,
    make_harbor_cantilever,
    make_harbor_l_bracket,
    make_harbor_suspension_rocker,
)
from src.agents.solutions import (
    _locate_design_json,
    build_comparison_rows,
    discover_harbor_runs,
    evaluate_cad_verification,
    evaluate_replay_step,
    get_cached_reference_simp,
    load_design_payload_solution,
    load_design_solution,
    load_harbor_trial,
    reconstruct_cad_step,
    replay_cutout_history,
    resolve_problem_for_design,
    run_in_repo_solver,
)
from src.agents.verifier import score_cad_submission
from src.agents.visualizer import (
    compute_load_case_fea,
    compute_replay_field_ranges,
    render_fea_field_figure,
    render_model_geometry_figure,
    render_replay_trajectory_figure,
    render_solution_views,
)


def _assert_figure_renders_png(testcase: unittest.TestCase, fig) -> None:
    buf = io.BytesIO()
    fig.savefig(buf, format="png")
    testcase.assertGreater(buf.tell(), 200)


class TestSolutionLoadingAndComparison(unittest.TestCase):
    def test_harbor_trial_discovery_and_loading_across_all_three_tasks(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            jobs_root = Path(tmp) / "jobs"
            self.assertEqual(discover_harbor_runs(jobs_root), [])

            job_dir = jobs_root / "2026-11-14__12-00-00"
            job_dir.mkdir(parents=True)
            (job_dir / "result.json").write_text(json.dumps({"n_trials": 3}))

            expected_targets = {
                "cantilever-lightweight": 0.45,
                "l-bracket-stress": 0.36,
                "suspension-rocker-multiload": 0.40,
                "aerospace-rib-obstacles": 0.35,
            }

            for idx, (task_id, builder) in enumerate(HARBOR_TASK_BUILDERS.items()):
                problem = builder()
                model = CADModel(problem)
                model.cut_circle(problem.width * 0.5, problem.height * 0.5, 3.2)

                trial_dir = job_dir / f"{task_id}__trial{idx}"
                if idx == 0:
                    design_rel = Path("artifacts") / "custom" / "design.json"
                    design_file = trial_dir / design_rel
                    design_file.parent.mkdir(parents=True)
                    model.save_json(design_file)
                    (trial_dir / "artifacts" / "manifest.json").write_text(
                        json.dumps(
                            [
                                {
                                    "source": "/app/design.json",
                                    "destination": str(design_rel),
                                }
                            ]
                        )
                    )
                elif idx == 1:
                    design_file = trial_dir / "artifacts" / "app" / "design.json"
                    design_file.parent.mkdir(parents=True)
                    raw = model.to_dict()
                    raw["problem_name"] = ""
                    design_file.write_text(json.dumps(raw))
                else:
                    design_file = trial_dir / "design.json"
                    trial_dir.mkdir(parents=True)
                    raw = model.to_dict()
                    raw["problem_name"] = ""
                    design_file.write_text(json.dumps(raw))

                (trial_dir / "result.json").write_text(
                    json.dumps(
                        {
                            "task_name": f"junction26/{task_id}",
                            "trial_name": trial_dir.name,
                            "config": {
                                "agent": {
                                    "name": "opencode",
                                    "model_name": "gemini/gemini-2.5-pro",
                                }
                            },
                            "verifier_result": {"rewards": {"reward": 0.8125}},
                        }
                    )
                )

                if idx == 0:
                    ver_dir = trial_dir / "verifier"
                    ver_dir.mkdir(parents=True)
                    live_ver = evaluate_cad_verification(
                        problem, model, simp_iterations=15
                    )
                    payload = asdict(live_ver)
                    payload["agent_compliance"] = -1.0
                    (ver_dir / "metrics.json").write_text(json.dumps(payload))
                    (ver_dir / "reward.txt").write_text("0.7777\n")

            discovered = discover_harbor_runs(jobs_root)
            self.assertEqual(len(discovered), 4)
            self.assertEqual(len(discover_harbor_runs(job_dir)), 4)
            self.assertEqual(len(discover_harbor_runs(discovered[0])), 1)

            loaded_records = [load_harbor_trial(p) for p in discovered]
            loaded_by_task = {rec.task_id: rec for rec in loaded_records}
            self.assertEqual(set(loaded_by_task.keys()), set(expected_targets.keys()))

            for task_id, target_vol in expected_targets.items():
                rec = loaded_by_task[task_id]
                self.assertEqual(rec.source_kind, "harbor_run")
                self.assertAlmostEqual(
                    rec.problem.target_volume_fraction, target_vol, places=5
                )
                self.assertEqual(rec.solver_name, "opencode (gemini/gemini-2.5-pro)")
                self.assertIsNotNone(rec.cad_model)
                self.assertIsNotNone(rec.verification)
                expected_fea = FEASolver(rec.problem).solve(rec.density, penalty=1.0)
                self.assertAlmostEqual(
                    rec.fea_result.compliance, expected_fea.compliance, places=6
                )

            cant_rec = loaded_by_task["cantilever-lightweight"]
            assert cant_rec.verification is not None
            self.assertTrue(math.isinf(cant_rec.verification.agent_compliance))
            self.assertAlmostEqual(cant_rec.harbor_reward or 0.0, 0.7777, places=4)

            rows = build_comparison_rows(loaded_records)
            self.assertEqual(len(rows), 4)

    def test_existing_workspace_harbor_jobs_load_when_present(self) -> None:
        workspace_trials = discover_harbor_runs("jobs")
        for trial_path in workspace_trials:
            rec = load_harbor_trial(trial_path)
            self.assertEqual(rec.source_kind, "harbor_run")
            self.assertIsNotNone(rec.cad_model)
            self.assertIsNotNone(rec.harbor_reward)

    def test_problem_resolution_priority_and_harbor_presets(self) -> None:
        for name in (
            "Cantilever Lightweight Bracket",
            "Stress-Constrained L-Bracket",
            "Multi-Load Suspension Rocker",
            "Aerospace Rib with Obstacle Tunnels",
        ):
            prob, _ = resolve_problem_for_design(design_data={"problem_name": name})
            self.assertEqual(prob.name, name)

        with tempfile.TemporaryDirectory() as tmp:
            colocated_prob = make_harbor_l_bracket()
            colocated_prob.target_volume_fraction = 0.31
            colocated_prob.save_json(Path(tmp) / "problem.json")
            resolved, _ = resolve_problem_for_design(
                design_data={"problem_name": "Cantilever Lightweight Bracket"},
                search_dir=tmp,
            )
            self.assertAlmostEqual(resolved.target_volume_fraction, 0.31, places=5)

    def test_all_in_repo_solvers_and_live_verifier_parity(self) -> None:
        problem = make_harbor_cantilever()
        ref_simp = get_cached_reference_simp(problem, max_iterations=15)

        simp_rec = run_in_repo_solver(
            problem, "simp", max_iterations=15, simp_iterations=15
        )
        oracle_rec = run_in_repo_solver(
            problem, "oracle_solver", reference_simp=ref_simp
        )

        self.assertIsNone(simp_rec.cad_model)
        self.assertIsNotNone(simp_rec.verification)
        self.assertIsNotNone(simp_rec.harbor_reward)
        self.assertEqual(simp_rec.density.shape, (problem.ny, problem.nx))

        self.assertIsNotNone(oracle_rec.cad_model)
        self.assertIsNotNone(oracle_rec.verification)
        self.assertIsNotNone(oracle_rec.harbor_reward)
        direct_fea = FEASolver(problem).solve(oracle_rec.density, penalty=1.0)
        self.assertAlmostEqual(
            oracle_rec.fea_result.compliance, direct_fea.compliance, places=6
        )

        with tempfile.TemporaryDirectory() as tmp:
            design_path = Path(tmp) / "oracle_design.json"
            assert oracle_rec.cad_model is not None
            oracle_rec.cad_model.save_json(design_path)
            loaded_design = load_design_solution(design_path, reference_simp=ref_simp)
            official_ver = score_cad_submission(
                problem, design_path, simp_iterations=15
            )
            assert loaded_design.verification is not None
            self.assertAlmostEqual(
                loaded_design.verification.reward, official_ver.reward, places=4
            )

        rows = build_comparison_rows([simp_rec, oracle_rec])
        self.assertEqual(len(rows), 2)
        self.assertNotEqual(rows[0]["Harbor Reward"], "N/A (Non-CAD)")
        self.assertGreater(float(rows[0]["Harbor Reward"]), 0.0)

    def test_multi_regime_verifier_parity_and_defensive_loading(self) -> None:
        cantilever = make_harbor_cantilever()
        l_bracket = make_harbor_l_bracket()
        rocker = make_harbor_suspension_rocker()

        uncut_model = CADModel(cantilever)

        invalid_feat_model = CADModel(cantilever)
        self.assertTrue(invalid_feat_model.cut_circle(25.0, 15.0, 6.0))
        self.assertTrue(invalid_feat_model.cut_circle(42.0, 15.0, 1.0))

        over_stress_model = CADModel(l_bracket)
        self.assertTrue(over_stress_model.cut_rect(10.0, 10.0, 10.0, 10.0))

        thin_wall_model = CADModel(cantilever)
        self.assertTrue(thin_wall_model.cut_rect(5.0, 4.0, 45.0, 4.0))
        self.assertTrue(thin_wall_model.cut_rect(5.0, 8.6, 45.0, 4.0))
        self.assertTrue(thin_wall_model.cut_rect(5.0, 13.2, 45.0, 4.0))
        self.assertTrue(thin_wall_model.cut_rect(5.0, 17.8, 45.0, 4.0))
        self.assertTrue(thin_wall_model.cut_rect(5.0, 22.4, 45.0, 3.6))

        disconnected_model = CADModel(cantilever)
        self.assertTrue(disconnected_model.cut_rect(22.0, -1.0, 10.0, 32.0))

        multiload_model = CADModel(rocker)
        self.assertTrue(multiload_model.cut_slot(26.0, 20.0, 52.0, 20.0, 4.0))

        regimes = [
            ("uncut", cantilever, uncut_model),
            ("invalid_feature", cantilever, invalid_feat_model),
            ("over_stress", l_bracket, over_stress_model),
            ("thin_wall", cantilever, thin_wall_model),
            ("disconnected", cantilever, disconnected_model),
            ("multiload", rocker, multiload_model),
        ]

        with tempfile.TemporaryDirectory() as tmp:
            tmp_dir = Path(tmp)
            results_by_name = {}
            for name, prob, cad in regimes:
                d_file = tmp_dir / f"{name}.json"
                cad.save_json(d_file)
                reloaded_cad = CADModel.load_json(d_file, prob)
                ref_simp = get_cached_reference_simp(prob, max_iterations=15)
                live_ver = evaluate_cad_verification(
                    prob, reloaded_cad, reference_simp=ref_simp, simp_iterations=15
                )
                official_ver = score_cad_submission(prob, d_file, simp_iterations=15)
                self.assertEqual(asdict(live_ver), asdict(official_ver), msg=name)
                results_by_name[name] = live_ver

            self.assertFalse(results_by_name["uncut"].valid)
            self.assertFalse(results_by_name["invalid_feature"].valid)
            self.assertTrue(results_by_name["over_stress"].valid)
            self.assertLess(results_by_name["over_stress"].stress_factor, 1.0)
            self.assertTrue(results_by_name["thin_wall"].valid)
            self.assertFalse(results_by_name["thin_wall"].wall_thickness_ok)
            self.assertLess(results_by_name["thin_wall"].manufacturability_factor, 1.0)
            self.assertFalse(results_by_name["disconnected"].valid)
            self.assertTrue(results_by_name["multiload"].valid)
            self.assertEqual(len(results_by_name["multiload"].case_compliances), 2)

            slash_trial = tmp_dir / "cantilever-lightweight__slash"
            slash_design = slash_trial / "artifacts" / "app" / "design.json"
            slash_design.parent.mkdir(parents=True)
            multiload_or_cant = CADModel(cantilever)
            multiload_or_cant.cut_circle(25.0, 15.0, 4.0)
            multiload_or_cant.save_json(slash_design)
            (slash_trial / "artifacts" / "manifest.json").write_text(
                json.dumps(
                    [{"source": "/app/design.json", "destination": "/app/design.json"}]
                )
            )
            slash_rec = load_harbor_trial(
                slash_trial,
                reference_simp=get_cached_reference_simp(cantilever, max_iterations=15),
            )
            self.assertEqual(
                Path(slash_rec.metadata["design_path"]).resolve(),
                slash_design.resolve(),
            )

            for bad_json in ("[1, 2, 3]", "null"):
                bad_file = tmp_dir / "bad_design.json"
                bad_file.write_text(bad_json)
                with self.assertRaises(ValueError):
                    load_design_solution(bad_file, problem=cantilever)
                with self.assertRaises(ValueError):
                    load_design_payload_solution(
                        json.loads(bad_json), problem=cantilever
                    )
                bad_trial = tmp_dir / f"cantilever-lightweight__bad_{len(bad_json)}"
                bad_trial_design = bad_trial / "artifacts" / "app" / "design.json"
                bad_trial_design.parent.mkdir(parents=True, exist_ok=True)
                bad_trial_design.write_text(bad_json)
                with self.assertRaises(ValueError):
                    load_harbor_trial(bad_trial, problem=cantilever)

    def test_symlink_containment_and_cycle_deduplication(self) -> None:
        cantilever = make_harbor_cantilever()
        ref_simp = get_cached_reference_simp(cantilever, max_iterations=15)

        with tempfile.TemporaryDirectory() as tmp:
            tmp_dir = Path(tmp)
            outside_dir = tmp_dir / "outside_secret"
            (outside_dir / "app").mkdir(parents=True)
            outside_model = CADModel(cantilever)
            outside_model.cut_circle(25.0, 15.0, 4.0)
            outside_design = outside_dir / "design.json"
            outside_model.save_json(outside_design)
            outside_model.save_json(outside_dir / "app" / "design.json")

            t_app = tmp_dir / "jobs_esc" / "cantilever-lightweight__esc_app"
            (t_app / "artifacts" / "app").mkdir(parents=True)
            os.symlink(outside_design, t_app / "artifacts" / "app" / "design.json")
            (t_app / "artifacts" / "manifest.json").write_text(
                json.dumps(
                    [{"source": "/app/design.json", "destination": "/app/design.json"}]
                )
            )

            t_custom = tmp_dir / "jobs_esc" / "cantilever-lightweight__esc_custom"
            (t_custom / "artifacts" / "custom").mkdir(parents=True)
            os.symlink(
                outside_design, t_custom / "artifacts" / "custom" / "design.json"
            )
            (t_custom / "artifacts" / "manifest.json").write_text(
                json.dumps(
                    [
                        {
                            "source": "/app/design.json",
                            "destination": "custom/design.json",
                        }
                    ]
                )
            )

            t_root = tmp_dir / "jobs_esc" / "cantilever-lightweight__esc_root"
            t_root.mkdir(parents=True)
            os.symlink(outside_design, t_root / "design.json")

            t_artdir = tmp_dir / "jobs_esc" / "cantilever-lightweight__esc_artdir"
            t_artdir.mkdir(parents=True)
            os.symlink(outside_dir, t_artdir / "artifacts")

            for esc_trial in (t_app, t_custom, t_root, t_artdir):
                self.assertIsNone(_locate_design_json(esc_trial))
                with self.assertRaises(FileNotFoundError):
                    load_harbor_trial(esc_trial, reference_simp=ref_simp)
            self.assertEqual(discover_harbor_runs(tmp_dir / "jobs_esc"), [])

            t_internal = tmp_dir / "jobs_int" / "cantilever-lightweight__internal"
            (t_internal / "artifacts" / "real").mkdir(parents=True)
            (t_internal / "artifacts" / "latest").mkdir(parents=True)
            real_design = t_internal / "artifacts" / "real" / "design.json"
            outside_model.save_json(real_design)
            os.symlink(
                Path("..") / "real" / "design.json",
                t_internal / "artifacts" / "latest" / "design.json",
            )
            (t_internal / "artifacts" / "manifest.json").write_text(
                json.dumps(
                    [
                        {
                            "source": "/app/design.json",
                            "destination": "latest/design.json",
                        }
                    ]
                )
            )
            located_internal = _locate_design_json(t_internal)
            self.assertIsNotNone(located_internal)
            assert located_internal is not None
            self.assertTrue(
                located_internal.resolve().is_relative_to(t_internal.resolve())
            )
            internal_rec = load_harbor_trial(t_internal, reference_simp=ref_simp)
            self.assertEqual(internal_rec.source_kind, "harbor_run")

            cycle_jobs = tmp_dir / "jobs_cycle"
            valid_trial = cycle_jobs / "job1" / "cantilever-lightweight__valid"
            (valid_trial / "artifacts" / "app").mkdir(parents=True)
            outside_model.save_json(valid_trial / "artifacts" / "app" / "design.json")
            failed_trial = cycle_jobs / "job1" / "a_failed_trial"
            failed_trial.mkdir(parents=True)
            os.symlink(Path(".."), failed_trial / "cycle_link")
            discovered_cycle = discover_harbor_runs(cycle_jobs)
            self.assertEqual(discovered_cycle, [valid_trial])

            outside_jobs = (
                tmp_dir / "outside_jobs_target" / "cantilever-lightweight__ext"
            )
            (outside_jobs / "artifacts" / "app").mkdir(parents=True)
            outside_model.save_json(outside_jobs / "artifacts" / "app" / "design.json")
            ext_root = tmp_dir / "jobs_ext_only"
            ext_root.mkdir(parents=True)
            os.symlink(outside_jobs.parent, ext_root / "ext_job_link")
            self.assertEqual(discover_harbor_runs(ext_root), [])


class TestModelAndStressStrainVisualization(unittest.TestCase):
    def test_single_load_and_multiload_plots_match_fea_values_without_pyplot_leaks(
        self,
    ) -> None:
        baseline_fignums = list(plt.get_fignums())

        for problem in (make_harbor_cantilever(), make_harbor_l_bracket()):
            model = CADModel(problem)
            model.cut_circle(20.0, 15.0, 3.5)
            density = model.to_density_grid()
            expected_fea = FEASolver(problem).solve(density, penalty=1.0)

            model_fig = render_model_geometry_figure(problem, model, density)
            stress_fig = render_fea_field_figure(
                problem, density, expected_fea, field="stress", cad_model=model
            )
            strain_fig = render_fea_field_figure(
                problem,
                density,
                expected_fea,
                field="strain_energy",
                cad_model=model,
                log_scale=True,
            )

            for fig in (model_fig, stress_fig, strain_fig):
                _assert_figure_renders_png(self, fig)

            solid_mask = density > 1e-3
            void_mask = ~solid_mask
            stress_arr = stress_fig.axes[0].images[0].get_array()
            strain_arr = strain_fig.axes[0].images[0].get_array()
            assert stress_arr is not None
            assert strain_arr is not None

            self.assertTrue(np.all(np.ma.getmaskarray(stress_arr)[void_mask]))
            self.assertTrue(np.all(np.ma.getmaskarray(strain_arr)[void_mask]))
            np.testing.assert_allclose(
                np.asarray(stress_arr)[solid_mask],
                expected_fea.von_mises_stress[solid_mask],
                rtol=0.0,
                atol=1e-12,
            )
            np.testing.assert_allclose(
                np.asarray(strain_arr)[solid_mask],
                expected_fea.element_strain_energies[solid_mask],
                rtol=0.0,
                atol=1e-12,
            )

        rocker = make_harbor_suspension_rocker()
        simp_sol = run_in_repo_solver(
            rocker, "simp", max_iterations=10, simp_iterations=10
        )
        cad_model = CADModel(rocker)
        cad_model.cut_slot(28.0, 20.0, 50.0, 20.0, 4.0)
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / "rocker.json"
            cad_model.save_json(p)
            rocker_sol = load_design_solution(
                p,
                problem=rocker,
                reference_simp=get_cached_reference_simp(rocker, max_iterations=10),
            )

        for sol in (simp_sol, rocker_sol):
            for case_idx in (None, 0, 1):
                views = render_solution_views(
                    sol, load_case_index=case_idx, log_strain=(case_idx == 1)
                )
                for fig in (
                    views.model_figure,
                    views.stress_figure,
                    views.strain_figure,
                ):
                    _assert_figure_renders_png(self, fig)

                expected = compute_load_case_fea(
                    rocker, sol.density, load_case_index=case_idx, penalty=1.0
                )
                solid_mask = sol.density > 1e-3
                plotted_stress = views.stress_figure.axes[0].images[0].get_array()
                plotted_strain = views.strain_figure.axes[0].images[0].get_array()
                np.testing.assert_allclose(
                    np.asarray(plotted_stress)[solid_mask],
                    expected.von_mises_stress[solid_mask],
                    rtol=0.0,
                    atol=1e-12,
                )
                np.testing.assert_allclose(
                    np.asarray(plotted_strain)[solid_mask],
                    expected.element_strain_energies[solid_mask],
                    rtol=0.0,
                    atol=1e-12,
                )

            env_fea = compute_load_case_fea(rocker, sol.density, load_case_index=None)
            c0_fea = compute_load_case_fea(rocker, sol.density, load_case_index=0)
            c1_fea = compute_load_case_fea(rocker, sol.density, load_case_index=1)
            self.assertAlmostEqual(
                env_fea.compliance,
                c0_fea.compliance + c1_fea.compliance,
                places=6,
            )
            np.testing.assert_allclose(
                env_fea.von_mises_stress,
                np.maximum(c0_fea.von_mises_stress, c1_fea.von_mises_stress),
                rtol=1e-10,
                atol=1e-12,
            )

        with self.assertRaises(ValueError):
            compute_load_case_fea(rocker, rocker_sol.density, load_case_index=2)

        self.assertEqual(list(plt.get_fignums()), baseline_fignums)

    def test_degenerate_zero_and_uniform_fields_render_cleanly(self) -> None:
        problem = make_harbor_cantilever()
        density = np.ones((problem.ny, problem.nx), dtype=float)
        zero_fea = FEAResult(
            displacements=np.zeros(2 * (problem.nx + 1) * (problem.ny + 1)),
            compliance=0.0,
            element_strain_energies=np.zeros((problem.ny, problem.nx), dtype=float),
            von_mises_stress=np.zeros((problem.ny, problem.nx), dtype=float),
            case_compliances=[0.0],
            case_max_stresses=[0.0],
        )
        for log_flag in (False, True):
            fig_s = render_fea_field_figure(
                problem, density, zero_fea, field="stress", log_scale=log_flag
            )
            fig_e = render_fea_field_figure(
                problem, density, zero_fea, field="strain_energy", log_scale=log_flag
            )
            _assert_figure_renders_png(self, fig_s)
            _assert_figure_renders_png(self, fig_e)

        micro_fea = FEAResult(
            displacements=np.zeros(2 * (problem.nx + 1) * (problem.ny + 1)),
            compliance=1e-15 * problem.nx * problem.ny,
            element_strain_energies=np.full((problem.ny, problem.nx), 1e-15),
            von_mises_stress=np.full((problem.ny, problem.nx), 1e-15),
            case_compliances=[1e-15 * problem.nx * problem.ny],
            case_max_stresses=[1e-15],
        )
        fig_micro = render_fea_field_figure(
            problem, density, micro_fea, field="strain_energy", log_scale=True
        )
        micro_norm = fig_micro.axes[0].images[0].norm
        self.assertIsInstance(micro_norm, LogNorm)
        assert isinstance(micro_norm.vmin, float) and isinstance(micro_norm.vmax, float)
        self.assertGreater(micro_norm.vmin, 0.0)
        self.assertLess(micro_norm.vmin, micro_norm.vmax)
        _assert_figure_renders_png(self, fig_micro)

        simp_sol = run_in_repo_solver(
            problem, "simp", max_iterations=8, simp_iterations=8
        )
        reused_views = render_solution_views(
            simp_sol, problem=simp_sol.problem, load_case_index=None
        )
        self.assertIs(reused_views.fea_result, simp_sol.fea_result)


class TestCutoutHistoryReplay(unittest.TestCase):
    def test_step_by_step_cutout_replay_from_saved_design_json(self) -> None:
        l_bracket = make_harbor_l_bracket()
        source_model = CADModel(l_bracket)
        self.assertTrue(source_model.cut_circle(10.0, 35.0, 2.5))
        self.assertTrue(source_model.cut_rect(5.0, 5.0, 8.0, 6.0, fillet_radius=1.2))
        self.assertTrue(source_model.cut_slot(18.0, 8.0, 32.0, 8.0, 2.0))
        self.assertTrue(
            source_model.cut_polygon(
                [(6.0, 16.0), (14.0, 16.0), (14.0, 26.0), (6.0, 24.0)],
                fillet_radius=1.0,
            )
        )
        self.assertEqual(len(source_model.history), 4)

        cantilever = make_harbor_cantilever()
        mirror_model = CADModel(cantilever)
        self.assertTrue(
            mirror_model.cut_circle(18.0, 8.0, 2.5, mirror_x=True, mirror_y=True)
        )
        self.assertTrue(mirror_model.cut_rect(26.0, 11.0, 8.0, 8.0))

        with tempfile.TemporaryDirectory() as tmp:
            lb_path = Path(tmp) / "l_bracket_design.json"
            source_model.save_json(lb_path)
            loaded_lb = CADModel.load_json(lb_path, l_bracket)

            frames = replay_cutout_history(loaded_lb)
            self.assertEqual(len(frames), 5)
            self.assertIsNone(frames[0].action)
            self.assertAlmostEqual(frames[0].volume_fraction, 0.64, places=4)

            for idx, frame in enumerate(frames):
                self.assertEqual(frame.step, idx)
                self.assertEqual(len(frame.cad_model.history), idx)
                if idx > 0:
                    self.assertEqual(frame.action, loaded_lb.history[idx - 1])
                    self.assertLess(
                        frame.volume_fraction, frames[idx - 1].volume_fraction
                    )
                views = render_solution_views(frame, problem=l_bracket)
                _assert_figure_renders_png(self, views.model_figure)
                _assert_figure_renders_png(self, views.stress_figure)
                _assert_figure_renders_png(self, views.strain_figure)

            traj_fig = render_replay_trajectory_figure(frames, l_bracket, active_step=2)
            _assert_figure_renders_png(self, traj_fig)

            final_frame = frames[-1]
            self.assertTrue(
                final_frame.cad_model.current_geometry.equals(
                    loaded_lb.current_geometry
                )
            )
            self.assertTrue(
                np.array_equal(final_frame.density, loaded_lb.to_density_grid())
            )

            mir_path = Path(tmp) / "mirror_design.json"
            mirror_model.save_json(mir_path)
            loaded_mir = CADModel.load_json(mir_path, cantilever)
            step1 = evaluate_replay_step(loaded_mir, 1)
            self.assertLess(
                step1.volume_fraction, 1.0 - 3.0 * (np.pi * 2.5**2) / 1800.0
            )
            views_mir = render_solution_views(step1, problem=cantilever)
            _assert_figure_renders_png(self, views_mir.model_figure)

            uncut_path = Path(tmp) / "uncut.json"
            CADModel(l_bracket).save_json(uncut_path)
            uncut_loaded = CADModel.load_json(uncut_path, l_bracket)
            uncut_frames = replay_cutout_history(uncut_loaded)
            self.assertEqual(len(uncut_frames), 1)
            self.assertEqual(uncut_frames[0].step, 0)
            self.assertAlmostEqual(uncut_frames[0].volume_fraction, 0.64, places=4)

            with self.assertRaises(ValueError):
                reconstruct_cad_step(loaded_lb, -1)
            with self.assertRaises(ValueError):
                reconstruct_cad_step(loaded_lb, len(loaded_lb.history) + 1)

            replayed_step2 = reconstruct_cad_step(loaded_lb, 2)
            self.assertIsNot(replayed_step2.history[0], loaded_lb.history[0])
            replayed_step2.history[0].mirror_x = True
            replayed_step2.history[0].description = "mutated"
            self.assertFalse(loaded_lb.history[0].mirror_x)
            self.assertNotEqual(loaded_lb.history[0].description, "mutated")

            frame_step1 = evaluate_replay_step(loaded_lb, 1)
            assert frame_step1.action is not None
            frame_step1.action.mirror_y = True
            self.assertFalse(loaded_lb.history[0].mirror_y)
            frame_views = render_solution_views(
                frame_step1, problem=loaded_lb.problem, load_case_index=None
            )
            self.assertIs(frame_views.fea_result, frame_step1.fea_result)


class TestStreamlitFrontendApp(unittest.TestCase):
    def test_headless_app_startup_solvers_multiload_and_replay(self) -> None:
        app_path = Path(__file__).resolve().parents[1] / "src" / "agents" / "app.py"
        at = AppTest.from_file(app_path)
        at.run(timeout=30)
        self.assertEqual(len(at.exception), 0)
        self.assertTrue(any("Target Vol: 0.45" in c.value for c in at.sidebar.caption))

        at.sidebar.selectbox(key="preset_key").set_value("l-bracket-stress").run(
            timeout=30
        )
        self.assertEqual(len(at.exception), 0)
        self.assertTrue(any("Target Vol: 0.36" in c.value for c in at.sidebar.caption))

        at.sidebar.button(key="run_solvers_btn").click().run(timeout=60)
        self.assertEqual(len(at.exception), 0)
        self.assertGreaterEqual(len(at.table), 1)
        simp_id = next(
            opt
            for opt in at.selectbox(key="replay_solution_selector").options
            if "simp" in opt.lower()
        )
        at.selectbox(key="replay_solution_selector").set_value(simp_id).run(timeout=30)
        self.assertTrue(
            any("Continuous SIMP density" in info.value for info in at.info)
        )

        at.sidebar.selectbox(key="preset_key").set_value(
            "suspension-rocker-multiload"
        ).run(timeout=30)
        self.assertEqual(len(at.exception), 0)
        self.assertTrue(any("Target Vol: 0.40" in c.value for c in at.sidebar.caption))

        at.sidebar.button(key="run_solvers_btn").click().run(timeout=60)
        self.assertEqual(len(at.exception), 0)

        at.sidebar.checkbox(key="log_strain").check().run(timeout=30)
        self.assertEqual(len(at.exception), 0)

        with tempfile.TemporaryDirectory() as tmp:
            trial_dir = Path(tmp) / "cantilever-lightweight__smoke"
            art_dir = trial_dir / "artifacts" / "app"
            art_dir.mkdir(parents=True)
            prob = make_harbor_cantilever()
            m = CADModel(prob)
            m.cut_circle(25.0, 15.0, 4.0)
            m.cut_rect(35.0, 10.0, 8.0, 8.0, fillet_radius=1.5)
            m.save_json(art_dir / "design.json")
            (trial_dir / "verifier").mkdir(parents=True)
            (trial_dir / "verifier" / "reward.txt").write_text("0.8800\n")

            with unittest.mock.patch(
                "src.agents.solutions.discover_harbor_runs", return_value=[trial_dir]
            ):
                at.run(timeout=30)
                at.sidebar.multiselect(key="selected_harbor_trials").set_value(
                    [str(trial_dir)]
                ).run(timeout=30)
                at.sidebar.button(key="load_harbor_btn").click().run(timeout=45)
                self.assertEqual(len(at.exception), 0)

            harbor_sol_id = f"harbor:{trial_dir.resolve()}"
            at.selectbox(key="replay_solution_selector").set_value(harbor_sol_id).run(
                timeout=30
            )
            self.assertEqual(len(at.exception), 0)

            at.slider(key="replay_step_slider").set_value(0).run(timeout=30)
            self.assertEqual(len(at.exception), 0)
            at.slider(key="replay_step_slider").set_value(1).run(timeout=30)
            self.assertEqual(len(at.exception), 0)

            uncut_file = Path(tmp) / "uncut_design.json"
            CADModel(prob).save_json(uncut_file)
            uncut_rec = load_design_solution(uncut_file)
            at.session_state["loaded_solutions"][uncut_rec.solution_id] = uncut_rec
            at.run(timeout=30)
            self.assertEqual(len(at.exception), 0)

            uncut_id = uncut_rec.solution_id
            at.selectbox(key="replay_solution_selector").set_value(uncut_id).run(
                timeout=30
            )
            self.assertEqual(len(at.exception), 0)
            self.assertTrue(any("Step 0 / 0" in info.value for info in at.info))

    def test_human_canvas_replay_sync_clear_and_cross_task_load_case(self) -> None:
        app_path = Path(__file__).resolve().parents[1] / "src" / "agents" / "app.py"
        at = AppTest.from_file(app_path)
        at.run(timeout=30)
        self.assertEqual(len(at.exception), 0)
        self.assertEqual(
            at.selectbox(key="replay_solution_selector").options,
            ["Human Designer (0 cutouts)"],
        )

        at.radio(key="human_cut_type").set_value("Circular Hole").run(timeout=30)
        at.button(key="human_cut_circle_btn").click().run(timeout=30)
        self.assertEqual(len(at.exception), 0)
        self.assertNotIn("human_fea_snapshot", at.session_state)
        self.assertEqual(
            at.selectbox(key="replay_solution_selector").options,
            ["Human Designer (1 cutouts)"],
        )

        at.button(key="human_refresh_fea_btn").click().run(timeout=30)
        self.assertEqual(len(at.exception), 0)
        self.assertIn("human_fea_snapshot", at.session_state)

        at.radio(key="human_cut_type").set_value("Rectangular Cutout").run(timeout=30)
        self.assertEqual(len(at.exception), 0)
        at.button(key="human_cut_rect_btn").click().run(timeout=30)
        self.assertEqual(len(at.exception), 0)
        self.assertEqual(
            at.selectbox(key="replay_solution_selector").options,
            ["Human Designer (2 cutouts)"],
        )

        at.radio(key="human_cut_type").set_value("Slot Cutout").run(timeout=30)
        self.assertEqual(len(at.exception), 0)
        at.button(key="human_cut_slot_btn").click().run(timeout=30)
        self.assertEqual(len(at.exception), 0)
        self.assertEqual(
            at.selectbox(key="replay_solution_selector").options,
            ["Human Designer (3 cutouts)"],
        )

        at.slider(key="human_sx2").set_value(15.0).run(timeout=30)
        at.button(key="human_cut_slot_btn").click().run(timeout=30)
        self.assertEqual(len(at.exception), 0)
        self.assertGreaterEqual(len(at.warning), 1)
        self.assertEqual(
            at.selectbox(key="replay_solution_selector").options,
            ["Human Designer (3 cutouts)"],
        )

        at.sidebar.checkbox(key="log_strain").check().run(timeout=30)
        self.assertEqual(len(at.exception), 0)

        at.button(key="human_undo_btn").click().run(timeout=30)
        self.assertEqual(len(at.exception), 0)
        self.assertEqual(
            at.selectbox(key="replay_solution_selector").options,
            ["Human Designer (2 cutouts)"],
        )

        at.button(key="human_reset_btn").click().run(timeout=30)
        self.assertEqual(len(at.exception), 0)
        self.assertEqual(
            at.selectbox(key="replay_solution_selector").options,
            ["Human Designer (0 cutouts)"],
        )

        with tempfile.TemporaryDirectory() as tmp:
            rocker_trial = Path(tmp) / "suspension-rocker-multiload__cross"
            rocker_art = rocker_trial / "artifacts" / "app"
            rocker_art.mkdir(parents=True)
            rocker_prob = make_harbor_suspension_rocker()
            rocker_cad = CADModel(rocker_prob)
            rocker_cad.cut_slot(28.0, 20.0, 50.0, 20.0, 4.0)
            rocker_cad.save_json(rocker_art / "design.json")
            (rocker_trial / "verifier").mkdir(parents=True)
            (rocker_trial / "verifier" / "reward.txt").write_text("0.9100\n")

            with unittest.mock.patch(
                "src.agents.solutions.discover_harbor_runs",
                return_value=[rocker_trial],
            ):
                at.run(timeout=30)
                at.sidebar.multiselect(key="selected_harbor_trials").set_value(
                    [str(rocker_trial)]
                ).run(timeout=30)
                at.sidebar.button(key="load_harbor_btn").click().run(timeout=45)
                self.assertEqual(len(at.exception), 0)

            rocker_sol_id = f"harbor:{rocker_trial.resolve()}"
            at.selectbox(key="replay_solution_selector").set_value(rocker_sol_id).run(
                timeout=30
            )
            self.assertEqual(len(at.exception), 0)
            self.assertIn("active_replay_model", at.session_state)

            at.sidebar.button(key="run_solvers_btn").click().run(timeout=60)
            self.assertEqual(len(at.exception), 0)
            simp_ids = [
                sid
                for sid in at.session_state["loaded_solutions"]
                if sid.startswith("solver:simp:")
            ]
            self.assertEqual(len(simp_ids), 1)
            at.selectbox(key="replay_solution_selector").set_value(simp_ids[0]).run(
                timeout=30
            )
            self.assertEqual(len(at.exception), 0)
            self.assertNotIn("active_replay_model", at.session_state)
            self.assertNotIn("active_replay_label", at.session_state)

            at.selectbox(key="replay_solution_selector").set_value(rocker_sol_id).run(
                timeout=30
            )
            self.assertEqual(len(at.exception), 0)
            self.assertIn("active_replay_model", at.session_state)

    def test_multi_preset_same_solver_export_disambiguation_and_clean_slider_sync(
        self,
    ) -> None:
        log_messages: list[str] = []

        class _ListHandler(logging.Handler):
            def emit(self, record: logging.LogRecord) -> None:
                log_messages.append(record.getMessage())

        st_logger = logging.getLogger("streamlit")
        handler = _ListHandler()
        st_logger.addHandler(handler)
        try:
            app_path = Path(__file__).resolve().parents[1] / "src" / "agents" / "app.py"
            at = AppTest.from_file(app_path)
            at.run(timeout=30)
            self.assertEqual(len(at.exception), 0)

            at.radio(key="human_cut_type").set_value("Circular Hole").run(timeout=30)
            at.button(key="human_cut_circle_btn").click().run(timeout=30)
            at.radio(key="human_cut_type").set_value("Rectangular Cutout").run(
                timeout=30
            )
            at.button(key="human_cut_rect_btn").click().run(timeout=30)
            self.assertEqual(at.slider(key="replay_step_slider").value, 2)
            at.button(key="human_undo_btn").click().run(timeout=30)
            self.assertEqual(len(at.exception), 0)
            self.assertEqual(at.slider(key="replay_step_slider").value, 1)

            at.sidebar.button(key="run_solvers_btn").click().run(timeout=60)
            self.assertEqual(len(at.exception), 0)

            at.sidebar.selectbox(key="preset_key").set_value("l-bracket-stress").run(
                timeout=30
            )
            at.sidebar.button(key="run_solvers_btn").click().run(timeout=60)
            self.assertEqual(len(at.exception), 0)

            self.assertFalse(
                any("replay_step_slider" in msg for msg in log_messages),
                msg=f"Unexpected replay_step_slider warnings: {log_messages}",
            )
        finally:
            st_logger.removeHandler(handler)

    def test_problem_geometry_updates_across_successive_tries(self) -> None:
        app_path = Path(__file__).resolve().parents[1] / "src" / "agents" / "app.py"
        at = AppTest.from_file(app_path)
        at.run(timeout=30)
        self.assertEqual(len(at.exception), 0)
        self.assertTrue(
            any("Cantilever Lightweight Bracket" in s.value for s in at.subheader)
        )

        # Try 1: run in-repo solvers on default cantilever-lightweight
        at.sidebar.button(key="run_solvers_btn").click().run(timeout=60)
        self.assertEqual(len(at.exception), 0)
        try1_headings = [m.value for m in at.markdown if m.value.startswith("#### ")]
        self.assertEqual(
            try1_headings,
            [
                "#### Reference Solution (`Cantilever Lightweight Bracket`)",
                "#### SIMP Topology Optimum (Continuous) (`Cantilever Lightweight Bracket`)",
            ],
        )
        try1_replay_id = at.selectbox(key="replay_solution_selector").value
        self.assertTrue(str(try1_replay_id).startswith("solver:oracle_solver:"))

        # Switch Design Problem to l-bracket-stress before running solvers:
        # Comparison & Replay must immediately switch away from Cantilever
        at.sidebar.selectbox(key="preset_key").set_value("l-bracket-stress").run(
            timeout=30
        )
        self.assertEqual(len(at.exception), 0)
        switched_headings = [
            m.value for m in at.markdown if m.value.startswith("#### ")
        ]
        self.assertEqual(switched_headings, [])
        self.assertTrue(
            any("Stress-Constrained L-Bracket" in s.value for s in at.subheader)
        )
        switched_replay_id = at.selectbox(key="replay_solution_selector").value
        self.assertNotEqual(switched_replay_id, try1_replay_id)
        self.assertTrue(str(switched_replay_id).startswith("human:"))

        # Try 2: run in-repo solvers on l-bracket-stress
        at.sidebar.button(key="run_solvers_btn").click().run(timeout=60)
        self.assertEqual(len(at.exception), 0)
        try2_headings = [m.value for m in at.markdown if m.value.startswith("#### ")]
        self.assertEqual(
            try2_headings,
            [
                "#### Reference Solution (`Stress-Constrained L-Bracket`)",
                "#### SIMP Topology Optimum (Continuous) (`Stress-Constrained L-Bracket`)",
            ],
        )
        try2_replay_id = at.selectbox(key="replay_solution_selector").value
        self.assertNotEqual(try2_replay_id, try1_replay_id)
        self.assertTrue(str(try2_replay_id).startswith("solver:oracle_solver:"))
        self.assertEqual(
            at.session_state["loaded_solutions"][try2_replay_id].problem.name,
            "Stress-Constrained L-Bracket",
        )

    def test_cad_canvas_payload_and_polygon_cutout(self) -> None:
        prob = make_harbor_cantilever()
        model = CADModel(prob)
        payload = build_canvas_payload(
            prob,
            model,
            tool="polygon",
            fillet_radius=0.5,
            slot_radius=2.5,
            grid_snap=0.5,
            problem_key="cantilever",
        )
        self.assertEqual(payload["tool"], "polygon")
        self.assertEqual(len(payload["polygons"]), 1)
        self.assertGreaterEqual(len(payload["supports"]), 1)
        self.assertGreaterEqual(len(payload["loads"]), 1)

        ok, err = apply_canvas_action(
            model,
            {
                "type": "polygon",
                "points": [[20.0, 10.0], [32.0, 10.0], [26.0, 20.0]],
                "fillet_radius": 0.5,
            },
        )
        self.assertTrue(ok)
        self.assertIsNone(err)
        self.assertEqual(len(model.history), 1)
        self.assertEqual(model.history[0].action_type, "polygon")

        updated_payload = build_canvas_payload(
            prob,
            model,
            tool="circle",
            fillet_radius=0.0,
            slot_radius=2.5,
            grid_snap=0.5,
            problem_key="cantilever",
        )
        self.assertEqual(len(updated_payload["polygons"][0]["holes"]), 1)

        bad_ok, bad_err = apply_canvas_action(
            model,
            {
                "type": "polygon",
                "points": [[1.0, 1.0], [2.0, 2.0]],
                "fillet_radius": 0.0,
            },
        )
        self.assertFalse(bad_ok)
        self.assertIsNotNone(bad_err)

    def test_replay_heatmap_scale_is_constant_across_steps(self) -> None:
        prob = make_harbor_cantilever()
        model = CADModel(prob)
        model.cut_circle(25.0, 15.0, 4.0)
        model.cut_rect(35.0, 10.0, 8.0, 8.0, fillet_radius=1.0)
        frames = replay_cutout_history(model)
        stress_range, strain_range = compute_replay_field_ranges(
            frames, log_strain=False
        )
        v0 = render_solution_views(
            frames[0],
            problem=prob,
            stress_range=stress_range,
            strain_range=strain_range,
        )
        v_last = render_solution_views(
            frames[-1],
            problem=prob,
            stress_range=stress_range,
            strain_range=strain_range,
        )
        self.assertEqual(
            v0.stress_figure.axes[0].images[0].get_clim(),
            v_last.stress_figure.axes[0].images[0].get_clim(),
        )
        self.assertEqual(
            v0.strain_figure.axes[0].images[0].get_clim(),
            v_last.strain_figure.axes[0].images[0].get_clim(),
        )

    def test_replay_trajectory_figure_has_clear_background_and_white_text(
        self,
    ) -> None:
        from matplotlib.image import imread

        prob = make_harbor_cantilever()
        model = CADModel(prob)
        model.cut_circle(25.0, 15.0, 4.0)
        frames = replay_cutout_history(model)
        fig = render_replay_trajectory_figure(frames, prob, active_step=1)
        self.assertEqual(fig.patch.get_alpha(), 0.0)
        for ax in fig.axes:
            self.assertEqual(ax.patch.get_alpha(), 0.0)
            self.assertEqual(ax.title.get_color(), "#ffffff")
            self.assertEqual(ax.xaxis.label.get_color(), "#ffffff")

        buf = io.BytesIO()
        fig.savefig(buf, format="png")
        buf.seek(0)
        rgba = imread(buf)
        self.assertEqual(rgba.shape[2], 4)
        self.assertEqual(float(rgba[0, 0, 3]), 0.0)


if __name__ == "__main__":
    unittest.main()
