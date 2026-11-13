from dataclasses import dataclass, field
from enum import Enum


class KeepOutType(Enum):
    SOLID = "solid"  # Must remain solid material (e.g. mounting boss)
    VOID = "void"  # Must remain empty/void (e.g. clearance zone)


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
    params: tuple[float, ...]  # (x, y, r) or (x, y, w, h)
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
    keepouts: list[KeepOut] = field(default_factory=list)
    target_volume_fraction: float = 0.40
    youngs_modulus: float = 1.0
    poissons_ratio: float = 0.3
    penalty: float = 3.0
    filter_radius: float = 1.5


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
    """L-bracket with fixed top edge, downward load at right edge tip, and top-right void keepout."""
    supports = [
        Support(x=x, y=height, fixed_x=True, fixed_y=True)
        for x in [i * (width / nx) for i in range(int(nx * 0.4) + 1)]
    ]
    loads = [Load(x=width, y=height * 0.2, fx=0.0, fy=-1.0)]
    # Void keepout for top-right corner of the bounding box to form L-shape
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
    """Automotive/aerospace suspension rocker with 3 bolt holes (keepouts)."""
    h1 = (10.0, 20.0, 5.0)  # Main pivot bushing (fixed)
    h2 = (70.0, 32.0, 4.0)  # Damper connection (loaded)
    h3 = (70.0, 8.0, 4.0)  # Wheel carrier link (loaded)

    supports = [Support(x=h1[0], y=h1[1], fixed_x=True, fixed_y=True, tolerance=5.0)]
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
