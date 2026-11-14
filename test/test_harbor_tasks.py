from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

from src.agents.cad_cli import main as cad_cli_main
from src.agents.csg import CADModel
from src.agents.fea import FEASolver
from src.agents.harbor_sync import REPO_ROOT, sync_harbor_tasks
from src.agents.oracle_solver import ParametricCADOptimizer
from src.agents.problem import (
    HARBOR_TASK_BUILDERS,
    make_harbor_cantilever,
    make_harbor_l_bracket,
    make_harbor_suspension_rocker,
    make_suspension_arm,
)
from src.agents.verifier import score_cad_submission


class TestAdvancedCSGAndMultiLoadFEA(unittest.TestCase):
    def test_scaled_suspension_arm_fits_domain(self) -> None:
        problem = make_suspension_arm(width=60.0, height=30.0, nx=40, ny=20)
        for ko in problem.keepouts:
            cx, cy, r = ko.params
            self.assertLessEqual(cx + r, problem.width)
            self.assertLessEqual(cy + r, problem.height)

    def test_polygon_fillet_mirror_and_serialization(self) -> None:
        problem = make_harbor_cantilever()
        model = CADModel(problem)
        self.assertTrue(model.is_structurally_connected())

        ok_poly = model.cut_polygon(
            [(22.0, 26.0), (54.0, 26.0), (54.0, 12.0)], fillet_radius=1.6
        )
        self.assertTrue(ok_poly)
        ok_rect = model.cut_rect(10.0, 10.0, 10.0, 8.0, fillet_radius=2.0)
        self.assertTrue(ok_rect)
        ok_mirror = model.cut_circle(28.0, 8.0, 2.0, mirror_y=True)
        self.assertTrue(ok_mirror)

        self.assertEqual(model.check_feature_constraints(), [])
        self.assertTrue(model.is_structurally_connected())

        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / "design.json"
            model.save_json(p)
            reloaded = CADModel.load_json(p, problem)
            self.assertAlmostEqual(
                model.get_volume_fraction(), reloaded.get_volume_fraction(), places=5
            )
            self.assertEqual(len(reloaded.history), 3)

    def test_severed_beam_detected_by_connectivity(self) -> None:
        problem = make_harbor_cantilever()
        model = CADModel(problem)
        # Vertical slot slicing across the entire beam height severs load path
        model.cut_rect(25.0, -2.0, 6.0, problem.height + 4.0)
        self.assertFalse(model.is_structurally_connected())

    def test_multiload_fea_solve(self) -> None:
        problem = make_harbor_suspension_rocker()
        solver = FEASolver(problem)
        model = CADModel(problem)
        res = solver.solve(model.to_density_grid(), penalty=1.0)
        self.assertEqual(len(res.case_compliances), 2)
        self.assertGreater(res.case_compliances[0], 0.0)
        self.assertGreater(res.case_compliances[1], 0.0)
        self.assertAlmostEqual(res.compliance, sum(res.case_compliances), places=6)


class TestHarborVerifierAndTasks(unittest.TestCase):
    def test_verifier_zero_on_nop_severed_and_pinhole(self) -> None:
        problem = make_harbor_l_bracket()
        with tempfile.TemporaryDirectory() as tmp:
            d_path = Path(tmp) / "design.json"

            # 1. Missing file -> 0.0
            r_missing = score_cad_submission(problem, d_path, simp_iterations=15)
            self.assertEqual(r_missing.reward, 0.0)

            # 2. Uncut stock plate (nop) -> 0.0
            stock = CADModel(problem)
            stock.save_json(d_path)
            r_nop = score_cad_submission(problem, d_path, simp_iterations=15)
            self.assertEqual(r_nop.reward, 0.0)

            # 3. Illegal pinhole (< min_feature_radius) -> 0.0
            pinhole = CADModel(problem)
            pinhole.cut_circle(10.0, 10.0, 5.0)
            pinhole.cut_circle(12.0, 25.0, 0.4)
            pinhole.save_json(d_path)
            r_pin = score_cad_submission(problem, d_path, simp_iterations=15)
            self.assertEqual(r_pin.reward, 0.0)

            # 4. Severed load path -> 0.0
            severed = CADModel(problem)
            severed.cut_rect(-1.0, 18.0, 25.0, 5.0)
            severed.save_json(d_path)
            r_sev = score_cad_submission(problem, d_path, simp_iterations=15)
            self.assertEqual(r_sev.reward, 0.0)

    def test_partial_credit_and_oracle_scores(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            for tid, builder in HARBOR_TASK_BUILDERS.items():
                problem = builder()
                d_path = Path(tmp) / f"{tid}_oracle.json"
                oracle_model = ParametricCADOptimizer(problem).optimize()
                oracle_model.save_json(d_path)
                res = score_cad_submission(problem, d_path, simp_iterations=30)
                self.assertTrue(res.valid, msg=f"{tid} invalid: {res.reason}")
                self.assertGreater(
                    res.reward, 0.85, msg=f"{tid} oracle reward too low: {res}"
                )
                self.assertLessEqual(res.reward, 1.0)

            # Partial-credit design gets smooth reward in (0, oracle)
            p_cant = make_harbor_cantilever()
            partial_model = ParametricCADOptimizer(p_cant).optimize()
            partial_model.undo()
            partial_model.undo()
            h_path = Path(tmp) / "partial.json"
            partial_model.save_json(h_path)
            r_partial = score_cad_submission(p_cant, h_path, simp_iterations=25)
            self.assertGreater(r_partial.reward, 0.15)
            self.assertLess(r_partial.reward, 0.92)

    def test_synced_harbor_layout_and_sandbox_isolation(self) -> None:
        task_dirs = sync_harbor_tasks()
        self.assertEqual(len(task_dirs), 3)

        for t_dir in task_dirs:
            self.assertTrue((t_dir / "task.toml").exists())
            self.assertTrue((t_dir / "instruction.md").exists())
            self.assertTrue((t_dir / "environment" / "Dockerfile").exists())
            self.assertTrue((t_dir / "tests" / "test.sh").exists())
            self.assertTrue((t_dir / "solution" / "solve.sh").exists())

            # Anti-cheat check: simp.py, verifier.py, oracle_solver.py must NOT be in environment/
            cad_env_files = {
                p.name for p in (t_dir / "environment" / "cad_env").iterdir()
            }
            self.assertNotIn("simp.py", cad_env_files)
            self.assertNotIn("verifier.py", cad_env_files)
            self.assertNotIn("oracle_solver.py", cad_env_files)

        # Verify CLI & container module execution end-to-end in a temp workspace
        sample_task = REPO_ROOT / "tasks" / "cantilever-lightweight"
        with tempfile.TemporaryDirectory() as tmp:
            prob_json = Path(tmp) / "problem.json"
            des_json = Path(tmp) / "design.json"
            rew_txt = Path(tmp) / "reward.txt"
            met_json = Path(tmp) / "metrics.json"
            prob_json.write_text(
                (sample_task / "environment" / "problem.json").read_text()
            )

            rc = cad_cli_main(
                [
                    "--problem",
                    str(prob_json),
                    "--design",
                    str(des_json),
                    "cut-circle",
                    "--x",
                    "30.0",
                    "--y",
                    "16.0",
                    "--r",
                    "4.0",
                ]
            )
            self.assertEqual(rc, 0)

            # Run synced oracle_pkg + verifier_pkg subprocesses just like Harbor containers
            env_ora = {
                "PYTHONPATH": f"{sample_task / 'solution'}:{sample_task / 'environment'}"
            }
            subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "oracle_pkg.oracle_solver",
                    "--problem",
                    str(prob_json),
                    "--design",
                    str(des_json),
                ],
                env=env_ora,
                check=True,
            )

            env_ver = {"PYTHONPATH": str(sample_task / "tests")}
            subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "verifier_pkg.verifier",
                    "--problem",
                    str(prob_json),
                    "--design",
                    str(des_json),
                    "--reward-out",
                    str(rew_txt),
                    "--metrics-out",
                    str(met_json),
                ],
                env=env_ver,
                check=True,
            )
            reward_val = float(rew_txt.read_text().strip())
            self.assertGreater(reward_val, 0.85)


if __name__ == "__main__":
    unittest.main()
