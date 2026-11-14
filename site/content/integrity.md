## Reward Integrity

Built for Harbor v1.4 and RL training, verification runs in **isolated mode (`environment_mode = "separate"`)** so agents cannot inspect `simp.py`, `oracle_solver.py`, or `verifier.py` inside the sandbox.

### Strict Manufacturing Guardrails
- **Disconnected Load Paths**: Floating or severed material earns `0.0`—agents cannot delete the part or isolate load bosses.
- **Unmachinable Slivers**: Minimum wall-thickness checks penalize razor-thin webs and degenerate micro-slivers.
- **Keep-Out Zones & Feature Limits**: Mounting bosses and load/support regions are strictly protected, and cutout counts are bounded.
- **Stress & Volume Gates**: Exceeding `target_volume` or `max_stress` caps reward; doing nothing (`vol = 1.0`) or stopping early earns `0.0`.

> Teaching AI to think in forces, geometry, and steel means completing the structural task well is the **only** way to score.
