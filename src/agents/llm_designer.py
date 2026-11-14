from dataclasses import dataclass, field
import json
import os
import re
import numpy as np

from .csg import CADModel
from .fea import FEASolver
from .problem import DesignProblem


@dataclass
class LLMStepRecord:
    step: int
    thought: str
    action: str
    compliance: float
    volume_fraction: float


@dataclass
class LLMDesignResult:
    cad_model: CADModel
    transcript: list[LLMStepRecord] = field(default_factory=list)


class LLMCADDesigner:
    """ReAct LLM agent that designs 2D CAD components using structural FEA feedback."""

    def __init__(self, problem: DesignProblem):
        self.problem = problem
        self.solver = FEASolver(problem)

    def _generate_state_prompt(
        self,
        model: CADModel,
        step: int,
        max_steps: int,
    ) -> tuple[str, list[tuple[float, float]]]:
        """Formats the current CAD state and FEA stress concentration for the LLM."""
        density = model.to_density_grid()
        fea_res = self.solver.solve(density, penalty=1.0)
        curr_vol = model.get_volume_fraction()
        target_vol = self.problem.target_volume_fraction

        dx = self.problem.width / self.problem.nx
        dy = self.problem.height / self.problem.ny
        xs = (np.arange(self.problem.nx) + 0.5) * dx
        ys = (np.arange(self.problem.ny) + 0.5) * dy

        # Locate stress hotspots and low-stress regions
        stress = fea_res.von_mises_stress
        solid_mask = density > 0.5

        # Hotspots (top stress among solid material)
        masked_high = np.where(solid_mask, stress, -np.inf)
        top_high_idx = np.argsort(masked_high.flatten())[-3:][::-1]
        hotspots = [
            (
                float(xs[idx % self.problem.nx]),
                float(ys[idx // self.problem.nx]),
                float(masked_high.flatten()[idx]),
            )
            for idx in top_high_idx
            if masked_high.flatten()[idx] > -np.inf
        ]

        # Low-stress candidates (lowest stress among solid material)
        masked_low = np.where(solid_mask, stress, np.inf)
        top_low_idx = np.argsort(masked_low.flatten())[:5]
        coldspots = [
            (
                float(xs[idx % self.problem.nx]),
                float(ys[idx // self.problem.nx]),
                float(masked_low.flatten()[idx]),
            )
            for idx in top_low_idx
            if masked_low.flatten()[idx] < np.inf
        ]

        prompt = f"""--- STEP {step + 1}/{max_steps} ---
Design Problem: {self.problem.name} ({self.problem.width:.0f}x{self.problem.height:.0f})
Target Volume Fraction: {target_vol:.2f} (Current: {curr_vol:.2f})
Current Compliance: {fea_res.compliance:.4f} (Lower = stiffer)
Max Stress: {np.max(stress):.2f}

High-Stress Hotspots (CRITICAL LOAD PATHS - DO NOT CUT HERE):
{", ".join([f"({x:.1f}, {y:.1f}) [stress={s:.2f}]" for x, y, s in hotspots])}

Low-Stress Coldspots (UNDERUTILIZED MATERIAL - IDEAL FOR CUTOUTS):
{", ".join([f"({x:.1f}, {y:.1f}) [stress={s:.2f}]" for x, y, s in coldspots])}

History of actions taken:
{[a.description for a in model.history] if model.history else "None yet (stock plate)"}

AVAILABLE TOOLS:
- `cut_circle(x, y, radius)`
- `cut_slot(x1, y1, x2, y2, radius)`
- `stop()`

Respond in JSON format:
{{
  "thought": "Your structural reasoning analyzing why and where to remove weight...",
  "tool": "cut_circle",
  "args": {{"x": 30.0, "y": 15.0, "radius": 4.0}}
}}
"""
        cold_coords = [(x, y) for x, y, _ in coldspots]
        return prompt, cold_coords

    def _simulated_llm_response(
        self,
        model: CADModel,
        cold_coords: list[tuple[float, float]],
    ) -> dict:
        """Heuristic rule-based LLM response simulator for offline demos and testing."""
        curr_vol = model.get_volume_fraction()
        target_vol = self.problem.target_volume_fraction

        if curr_vol <= target_vol + 0.02:
            return {
                "thought": f"Target volume fraction ({target_vol:.2f}) achieved. Preserving structural integrity.",
                "tool": "stop",
                "args": {},
            }

        # Pick first coldspot not too close to previous cuts
        for cx, cy in cold_coords:
            # Check proximity to supports
            if any(np.hypot(cx - s.x, cy - s.y) < 5.0 for s in self.problem.supports):
                continue
            # Check proximity to loads
            if any(np.hypot(cx - l.x, cy - l.y) < 5.0 for l in self.problem.loads):
                continue

            remaining_vol = curr_vol - target_vol
            domain_area = self.problem.width * self.problem.height
            target_cut_area = (remaining_vol * domain_area) / 3.0
            max_r = min(self.problem.width, self.problem.height) / 4.0
            radius = float(np.clip(np.sqrt(target_cut_area / np.pi), 3.0, max_r))
            return {
                "thought": f"The region around ({cx:.1f}, {cy:.1f}) exhibits minimal von Mises stress. Cutting a hole of radius {radius:.1f} sheds unnecessary mass while preserving primary load paths.",
                "tool": "cut_circle",
                "args": {
                    "x": round(cx, 1),
                    "y": round(cy, 1),
                    "radius": round(radius, 1),
                },
            }

        return {
            "thought": "No safe coldspots identified without compromising boundary conditions. Stopping.",
            "tool": "stop",
            "args": {},
        }

    def design(
        self,
        max_steps: int = 8,
        use_live_api: bool = False,
    ) -> LLMDesignResult:
        """Runs the ReAct design loop with LLM reasoning."""
        model = CADModel(self.problem)
        transcript: list[LLMStepRecord] = []

        for step in range(max_steps):
            prompt, cold_coords = self._generate_state_prompt(model, step, max_steps)

            # In hackathon demo, if an API key is configured and requested, call LLM; else use built-in simulator
            if use_live_api and (
                os.environ.get("OPENAI_API_KEY") or os.environ.get("GEMINI_API_KEY")
            ):
                # Placeholder for live LLM API call
                response = self._simulated_llm_response(model, cold_coords)
            else:
                response = self._simulated_llm_response(model, cold_coords)

            tool = response.get("tool")
            thought = response.get("thought", "")
            args = response.get("args", {})

            if tool == "stop":
                fea_res = self.solver.solve(model.to_density_grid(), penalty=1.0)
                transcript.append(
                    LLMStepRecord(
                        step=step + 1,
                        thought=thought,
                        action="stop()",
                        compliance=fea_res.compliance,
                        volume_fraction=model.get_volume_fraction(),
                    )
                )
                break

            action_desc = f"{tool}({args})"
            if tool == "cut_circle":
                model.cut_circle(
                    args.get("x", 0.0),
                    args.get("y", 0.0),
                    args.get("radius", 3.0),
                )
            elif tool == "cut_slot":
                model.cut_slot(
                    args.get("x1", 0.0),
                    args.get("y1", 0.0),
                    args.get("x2", 0.0),
                    args.get("y2", 0.0),
                    args.get("radius", 2.5),
                )

            fea_res = self.solver.solve(model.to_density_grid(), penalty=1.0)

            # Safeguard: if cut caused structural disconnect/blowup, undo
            last_comp = transcript[-1].compliance if transcript else 100.0
            if fea_res.compliance > last_comp * 8.0:
                model.undo()
                thought += " [Load path severed; reverting action]"
                action_desc += " [REVERTED]"
                fea_res = self.solver.solve(model.to_density_grid(), penalty=1.0)
            transcript.append(
                LLMStepRecord(
                    step=step + 1,
                    thought=thought,
                    action=action_desc,
                    compliance=fea_res.compliance,
                    volume_fraction=model.get_volume_fraction(),
                )
            )

            if model.get_volume_fraction() <= self.problem.target_volume_fraction:
                break

        return LLMDesignResult(cad_model=model, transcript=transcript)
