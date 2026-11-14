from shapely.geometry import MultiPolygon, Polygon

from .csg import CADModel


def export_cad_to_svg(model: CADModel) -> str:
    """Exports CADModel geometry to an SVG string."""
    return model.to_svg()


def export_cad_to_dxf(model: CADModel) -> str:
    """Generates an ASCII DXF (AutoCAD R12) file representing the 2D CAD boundary."""
    lines = [
        "0",
        "SECTION",
        "2",
        "HEADER",
        "0",
        "ENDSEC",
        "0",
        "SECTION",
        "2",
        "ENTITIES",
    ]

    geom = model.current_geometry
    polygons: list[Polygon] = []
    if isinstance(geom, Polygon):
        polygons = [geom]
    elif isinstance(geom, MultiPolygon):
        polygons = list(geom.geoms)

    for poly in polygons:
        # Exterior boundary polyline
        coords = list(poly.exterior.coords)
        lines.extend(["0", "POLYLINE", "8", "OUTLINE", "66", "1", "70", "1"])
        for x, y in coords:
            lines.extend(
                [
                    "0",
                    "VERTEX",
                    "8",
                    "OUTLINE",
                    "10",
                    f"{x:.3f}",
                    "20",
                    f"{y:.3f}",
                    "30",
                    "0.0",
                ]
            )
        lines.extend(["0", "SEQEND"])

        # Interior cutouts
        for interior in poly.interiors:
            icoords = list(interior.coords)
            lines.extend(["0", "POLYLINE", "8", "HOLES", "66", "1", "70", "1"])
            for x, y in icoords:
                lines.extend(
                    [
                        "0",
                        "VERTEX",
                        "8",
                        "HOLES",
                        "10",
                        f"{x:.3f}",
                        "20",
                        f"{y:.3f}",
                        "30",
                        "0.0",
                    ]
                )
            lines.extend(["0", "SEQEND"])

    lines.extend(["0", "ENDSEC", "0", "EOF"])
    return "\n".join(lines)
