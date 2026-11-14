from dataclasses import dataclass, field
from enum import Enum
import json
from pathlib import Path
from typing import Any


class KeepOutType(Enum):
    SOLID = "solid"
    VOID = "void"


@dataclass(frozen=True)
class Support:
    x: float
    y: float
    fixed_x: bool = True
    fixed_y: bool = True
    tolerance: float = 1.0


@dataclass(frozen=True)
class Load:
    x: float
    y: float
    fx: float = 0.0
    fy: float = -1.0


@dataclass(frozen=True)
class KeepOut:
    shape: str  # "circle" or "rect"
    params: tuple[float, ...]
    keepout_type: KeepOutType = KeepOutType.SOLID


@dataclass
class DesignProblem:
    name: str
    width: float
    height: float
    nx: int
    ny: int
    supports: list[Support] = field(default_factory=list)
    loads: list[Load] = field(default_factory=list)
    load_cases: list[list[Load]] = field(default_factory=list)
    keepouts: list[KeepOut] = field(default_factory=list)
    target_volume_fraction: float = 0.40
    youngs_modulus: float = 1.0
    poissons_ratio: float = 0.3
    penalty: float = 3.0
    filter_radius: float = 1.5
    max_stress_limit: float | None = None
    min_feature_radius: float = 1.0
    min_wall_thickness: float = 1.2
    max_cutouts: int = 24
    description: str = ""

    @property
    def effective_load_cases(self) -> list[list[Load]]:
        if self.load_cases:
            return self.load_cases
        if self.loads:
            return [self.loads]
        return []

    @property
    def all_loads(self) -> list[Load]:
        seen: dict[tuple[float, float, float, float], Load] = {}
        for case in self.effective_load_cases:
            for load in case:
                seen[(load.x, load.y, load.fx, load.fy)] = load
        return list(seen.values())

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "width": self.width,
            "height": self.height,
            "nx": self.nx,
            "ny": self.ny,
            "supports": [
                {
                    "x": s.x,
                    "y": s.y,
                    "fixed_x": s.fixed_x,
                    "fixed_y": s.fixed_y,
                    "tolerance": s.tolerance,
                }
                for s in self.supports
            ],
            "loads": [{"x": l.x, "y": l.y, "fx": l.fx, "fy": l.fy} for l in self.loads],
            "load_cases": [
                [{"x": l.x, "y": l.y, "fx": l.fx, "fy": l.fy} for l in case]
                for case in self.load_cases
            ],
            "keepouts": [
                {
                    "shape": k.shape,
                    "params": list(k.params),
                    "keepout_type": k.keepout_type.value,
                }
                for k in self.keepouts
            ],
            "target_volume_fraction": self.target_volume_fraction,
            "youngs_modulus": self.youngs_modulus,
            "poissons_ratio": self.poissons_ratio,
            "penalty": self.penalty,
            "filter_radius": self.filter_radius,
            "max_stress_limit": self.max_stress_limit,
            "min_feature_radius": self.min_feature_radius,
            "min_wall_thickness": self.min_wall_thickness,
            "max_cutouts": self.max_cutouts,
            "description": self.description,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "DesignProblem":
        supports = [Support(**s) for s in data.get("supports", [])]
        loads = [Load(**l) for l in data.get("loads", [])]
        load_cases = [[Load(**l) for l in case] for case in data.get("load_cases", [])]
        keepouts = [
            KeepOut(
                shape=k["shape"],
                params=tuple(float(v) for v in k["params"]),
                keepout_type=KeepOutType(k["keepout_type"]),
            )
            for k in data.get("keepouts", [])
        ]
        return cls(
            name=data["name"],
            width=float(data["width"]),
            height=float(data["height"]),
            nx=int(data["nx"]),
            ny=int(data["ny"]),
            supports=supports,
            loads=loads,
            load_cases=load_cases,
            keepouts=keepouts,
            target_volume_fraction=float(data.get("target_volume_fraction", 0.40)),
            youngs_modulus=float(data.get("youngs_modulus", 1.0)),
            poissons_ratio=float(data.get("poissons_ratio", 0.3)),
            penalty=float(data.get("penalty", 3.0)),
            filter_radius=float(data.get("filter_radius", 1.5)),
            max_stress_limit=(
                float(data["max_stress_limit"])
                if data.get("max_stress_limit") is not None
                else None
            ),
            min_feature_radius=float(data.get("min_feature_radius", 1.0)),
            min_wall_thickness=float(data.get("min_wall_thickness", 1.2)),
            max_cutouts=int(data.get("max_cutouts", 24)),
            description=str(data.get("description", "")),
        )

    def save_json(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(self.to_dict(), indent=2))

    @classmethod
    def load_json(cls, path: str | Path) -> "DesignProblem":
        return cls.from_dict(json.loads(Path(path).read_text()))


def make_cantilever_beam(
    width: float = 60.0,
    height: float = 30.0,
    nx: int = 60,
    ny: int = 30,
    target_volume_fraction: float = 0.40,
) -> DesignProblem:
    """Fixed at left wall (x=0), downward load at bottom-right corner."""
    supports = [
        Support(x=0.0, y=y, fixed_x=True, fixed_y=True)
        for y in [i * (height / ny) for i in range(ny + 1)]
    ]
    loads = [Load(x=width, y=0.0, fx=0.0, fy=-1.0)]
    return DesignProblem(
        name="Cantilever Beam",
        width=width,
        height=height,
        nx=nx,
        ny=ny,
        supports=supports,
        loads=loads,
        target_volume_fraction=target_volume_fraction,
    )


def make_mbb_beam(
    width: float = 60.0,
    height: float = 20.0,
    nx: int = 60,
    ny: int = 20,
    target_volume_fraction: float = 0.50,
) -> DesignProblem:
    """Pinned at bottom-left corner, roller at bottom-right corner, load at center top."""
    supports = [
        Support(x=0.0, y=0.0, fixed_x=True, fixed_y=True),
        Support(x=width, y=0.0, fixed_x=False, fixed_y=True),
    ]
    loads = [Load(x=width / 2.0, y=height, fx=0.0, fy=-1.0)]
    return DesignProblem(
        name="MBB Beam",
        width=width,
        height=height,
        nx=nx,
        ny=ny,
        supports=supports,
        loads=loads,
        target_volume_fraction=target_volume_fraction,
    )


def make_l_bracket(
    width: float = 50.0,
    height: float = 50.0,
    nx: int = 50,
    ny: int = 50,
    target_volume_fraction: float = 0.40,
) -> DesignProblem:
    """L-bracket with fixed top edge, downward load at right tip, and top-right void keepout."""
    supports = [
        Support(x=x, y=height, fixed_x=True, fixed_y=True)
        for x in [i * (width / nx) for i in range(int(nx * 0.4) + 1)]
    ]
    loads = [Load(x=width, y=height * 0.2, fx=0.0, fy=-1.0)]
    keepouts = [
        KeepOut(
            shape="rect",
            params=(width * 0.4, height * 0.4, width * 0.6, height * 0.6),
            keepout_type=KeepOutType.VOID,
        )
    ]
    return DesignProblem(
        name="L-Bracket",
        width=width,
        height=height,
        nx=nx,
        ny=ny,
        supports=supports,
        loads=loads,
        keepouts=keepouts,
        target_volume_fraction=target_volume_fraction,
    )


def make_suspension_arm(
    width: float = 80.0,
    height: float = 40.0,
    nx: int = 80,
    ny: int = 40,
    target_volume_fraction: float = 0.35,
) -> DesignProblem:
    """Automotive suspension rocker with 3 solid bolt bosses scaled to domain dimensions."""
    sx = width / 80.0
    sy = height / 40.0
    sr = min(sx, sy)

    h1 = (10.0 * sx, 20.0 * sy, 5.0 * sr)
    h2 = (70.0 * sx, 32.0 * sy, 4.0 * sr)
    h3 = (70.0 * sx, 8.0 * sy, 4.0 * sr)

    supports = [
        Support(x=h1[0], y=h1[1], fixed_x=True, fixed_y=True, tolerance=5.0 * sr)
    ]
    loads = [
        Load(x=h2[0], y=h2[1], fx=0.0, fy=-1.0),
        Load(x=h3[0], y=h3[1], fx=-0.5, fy=0.5),
    ]
    keepouts = [
        KeepOut(shape="circle", params=h1, keepout_type=KeepOutType.SOLID),
        KeepOut(shape="circle", params=h2, keepout_type=KeepOutType.SOLID),
        KeepOut(shape="circle", params=h3, keepout_type=KeepOutType.SOLID),
    ]
    return DesignProblem(
        name="Suspension Arm",
        width=width,
        height=height,
        nx=nx,
        ny=ny,
        supports=supports,
        loads=loads,
        keepouts=keepouts,
        target_volume_fraction=target_volume_fraction,
    )


def make_harbor_cantilever() -> DesignProblem:
    """Task 1 (Easy/Medium): Tip-loaded cantilever plate with mounting flange & load boss."""
    width, height = 60.0, 30.0
    nx, ny = 60, 30
    supports = [
        Support(x=0.0, y=y, fixed_x=True, fixed_y=True, tolerance=1.1)
        for y in [i * (height / ny) for i in range(ny + 1)]
    ]
    load_pt = (57.0, 4.0)
    loads = [Load(x=load_pt[0], y=load_pt[1], fx=0.0, fy=-1.0)]
    keepouts = [
        KeepOut(
            shape="rect", params=(0.0, 0.0, 2.5, height), keepout_type=KeepOutType.SOLID
        ),
        KeepOut(
            shape="circle",
            params=(load_pt[0], load_pt[1], 3.0),
            keepout_type=KeepOutType.SOLID,
        ),
    ]
    return DesignProblem(
        name="Cantilever Lightweight Bracket",
        width=width,
        height=height,
        nx=nx,
        ny=ny,
        supports=supports,
        loads=loads,
        keepouts=keepouts,
        target_volume_fraction=0.45,
        filter_radius=1.8,
        max_stress_limit=1.40,
        min_feature_radius=1.5,
        min_wall_thickness=1.5,
        max_cutouts=20,
        description=(
            "Lightweight a 60x30 cantilever bracket to <=45% volume fraction while "
            "maximizing stiffness between the fixed left wall and the bottom-right load boss."
        ),
    )


def make_harbor_l_bracket() -> DesignProblem:
    """Task 2 (Medium/Hard): Stress-constrained L-bracket with sharp re-entrant void corner."""
    width, height = 50.0, 50.0
    nx, ny = 50, 50
    supports = [
        Support(x=x, y=height, fixed_x=True, fixed_y=True, tolerance=1.1)
        for x in [i * (width / nx) for i in range(int(nx * 0.4) + 1)]
    ]
    load_pt = (46.0, 10.0)
    loads = [Load(x=load_pt[0], y=load_pt[1], fx=0.0, fy=-1.0)]
    keepouts = [
        KeepOut(
            shape="rect",
            params=(20.0, 20.0, 30.0, 30.0),
            keepout_type=KeepOutType.VOID,
        ),
        KeepOut(
            shape="rect",
            params=(0.0, 47.5, 20.0, 2.5),
            keepout_type=KeepOutType.SOLID,
        ),
        KeepOut(
            shape="circle",
            params=(load_pt[0], load_pt[1], 3.0),
            keepout_type=KeepOutType.SOLID,
        ),
    ]
    return DesignProblem(
        name="Stress-Constrained L-Bracket",
        width=width,
        height=height,
        nx=nx,
        ny=ny,
        supports=supports,
        loads=loads,
        keepouts=keepouts,
        target_volume_fraction=0.36,
        filter_radius=1.6,
        max_stress_limit=1.85,
        min_feature_radius=1.5,
        min_wall_thickness=1.6,
        max_cutouts=22,
        description=(
            "Design a 50x50 L-bracket (<=36% domain area; stock L-shape starts at 64% area) "
            "that resists tip bending without exceeding the von Mises stress limit around "
            "the re-entrant (20, 20) inner corner."
        ),
    )


def make_harbor_suspension_rocker() -> DesignProblem:
    """Task 3 (Hard): Multi-load suspension rocker arm with 3 mounting bosses."""
    width, height = 80.0, 40.0
    nx, ny = 64, 32
    pivot = (12.0, 20.0, 5.0)
    damper = (68.0, 31.0, 4.0)
    pushrod = (68.0, 9.0, 4.0)

    supports = [
        Support(x=pivot[0], y=pivot[1], fixed_x=True, fixed_y=True, tolerance=5.0)
    ]
    case_bump = [
        Load(x=damper[0], y=damper[1], fx=0.0, fy=-1.4),
        Load(x=pushrod[0], y=pushrod[1], fx=-0.8, fy=0.6),
    ]
    case_braking = [
        Load(x=damper[0], y=damper[1], fx=-0.7, fy=-0.35),
        Load(x=pushrod[0], y=pushrod[1], fx=0.5, fy=-0.75),
    ]
    keepouts = [
        KeepOut(shape="circle", params=pivot, keepout_type=KeepOutType.SOLID),
        KeepOut(shape="circle", params=damper, keepout_type=KeepOutType.SOLID),
        KeepOut(shape="circle", params=pushrod, keepout_type=KeepOutType.SOLID),
    ]
    return DesignProblem(
        name="Multi-Load Suspension Rocker",
        width=width,
        height=height,
        nx=nx,
        ny=ny,
        supports=supports,
        loads=case_bump,
        load_cases=[case_bump, case_braking],
        keepouts=keepouts,
        target_volume_fraction=0.40,
        filter_radius=1.8,
        max_stress_limit=1.40,
        min_feature_radius=1.5,
        min_wall_thickness=1.8,
        max_cutouts=24,
        description=(
            "Design an 80x40 motorsport suspension rocker arm (<=40% volume fraction) "
            "connecting 3 solid mounting bosses across 2 independent load cases "
            "(vertical bump + lateral braking shear)."
        ),
    )


def make_harbor_aerospace_rib() -> DesignProblem:
    """Task 4 (Hard/Expert): 3-load-case aerospace rib with 3 solid lugs + 2 void pipe tunnels."""
    width, height = 80.0, 40.0
    nx, ny = 64, 32

    supports = [
        Support(x=0.0, y=y, fixed_x=True, fixed_y=True, tolerance=1.3)
        for y in [i * (height / ny) for i in range(ny + 1)]
    ]

    lug_top = (74.0, 32.0, 4.0)
    lug_bot = (74.0, 8.0, 4.0)
    lug_mid = (42.0, 35.0, 3.5)

    # 3 distinct flight/landing load cases: tip bending, axial+shear, and mid-spar torsion
    case_pullup = [
        Load(x=lug_top[0], y=lug_top[1], fx=0.0, fy=-1.1),
        Load(x=lug_bot[0], y=lug_bot[1], fx=0.0, fy=-0.7),
    ]
    case_gust = [
        Load(x=lug_top[0], y=lug_top[1], fx=-0.8, fy=0.9),
        Load(x=lug_mid[0], y=lug_mid[1], fx=0.4, fy=-1.0),
    ]
    case_shear = [
        Load(x=lug_bot[0], y=lug_bot[1], fx=0.7, fy=1.1),
        Load(x=lug_mid[0], y=lug_mid[1], fx=-0.6, fy=0.8),
    ]

    keepouts = [
        KeepOut(
            shape="rect", params=(0.0, 0.0, 3.0, height), keepout_type=KeepOutType.SOLID
        ),
        KeepOut(shape="circle", params=lug_top, keepout_type=KeepOutType.SOLID),
        KeepOut(shape="circle", params=lug_bot, keepout_type=KeepOutType.SOLID),
        KeepOut(shape="circle", params=lug_mid, keepout_type=KeepOutType.SOLID),
        # 2 void hydraulic/avionics pass-through tunnels that truss chords must route around
        KeepOut(
            shape="circle", params=(30.0, 20.0, 4.0), keepout_type=KeepOutType.VOID
        ),
        KeepOut(
            shape="circle", params=(54.0, 20.0, 4.0), keepout_type=KeepOutType.VOID
        ),
    ]

    return DesignProblem(
        name="Aerospace Rib with Obstacle Tunnels",
        width=width,
        height=height,
        nx=nx,
        ny=ny,
        supports=supports,
        loads=case_pullup,
        load_cases=[case_pullup, case_gust, case_shear],
        keepouts=keepouts,
        target_volume_fraction=0.35,
        filter_radius=1.8,
        max_stress_limit=2.05,
        min_feature_radius=1.5,
        min_wall_thickness=1.6,
        max_cutouts=22,
        description=(
            "Design an 80x40 aerospace wing rib (<=35% volume fraction) routing "
            "triangulated load paths around 2 void pass-through tunnels to 3 solid "
            "mounting lugs across 3 independent load cases."
        ),
    )


HARBOR_TASK_BUILDERS = {
    "cantilever-lightweight": make_harbor_cantilever,
    "l-bracket-stress": make_harbor_l_bracket,
    "suspension-rocker-multiload": make_harbor_suspension_rocker,
    "aerospace-rib-obstacles": make_harbor_aerospace_rib,
}
