from __future__ import annotations

import argparse
import math
from dataclasses import dataclass
import os
from pathlib import Path
import subprocess
import sys
from typing import Iterable

import numpy as np


USD_LIBS_ROOT = Path("/root/isaacsim/extscache/omni.usd.libs-1.0.1+8131b85d.lx64.r.cp311")
ISAAC_PYTHON = Path("/root/isaacsim/python.sh")


def _ensure_usd_imports():
    if str(USD_LIBS_ROOT) not in sys.path:
        sys.path.insert(0, str(USD_LIBS_ROOT))
    usd_bin_dir = USD_LIBS_ROOT / "bin"
    ld_library_path = os.environ.get("LD_LIBRARY_PATH", "")
    if str(usd_bin_dir) not in ld_library_path.split(":"):
        os.environ["LD_LIBRARY_PATH"] = f"{usd_bin_dir}:{ld_library_path}" if ld_library_path else str(usd_bin_dir)

    try:
        from pxr import Gf, Usd, UsdGeom  # noqa: F401
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise RuntimeError(
            "Failed to import USD Python bindings. Run this tool in the Isaac Sim environment, "
            "or make sure pxr and the USD shared libraries are available."
        ) from exc

    from pxr import Gf, Usd, UsdGeom

    return Gf, Usd, UsdGeom


def _run_with_isaac_python() -> int:
    env = os.environ.copy()
    python_path_parts = [str(USD_LIBS_ROOT)]
    if env.get("PYTHONPATH"):
        python_path_parts.append(env["PYTHONPATH"])
    env["PYTHONPATH"] = ":".join(python_path_parts)

    usd_bin_dir = str(USD_LIBS_ROOT / "bin")
    ld_library_parts = [usd_bin_dir]
    if env.get("LD_LIBRARY_PATH"):
        ld_library_parts.append(env["LD_LIBRARY_PATH"])
    env["LD_LIBRARY_PATH"] = ":".join(ld_library_parts)
    env["ROBOT_NAV_USD_REEXEC"] = "1"

    command = [str(ISAAC_PYTHON), str(Path(__file__).resolve()), *sys.argv[1:]]
    completed = subprocess.run(command, env=env, check=False)
    return int(completed.returncode)


@dataclass(frozen=True)
class MapConfig:
    resolution: float
    padding_m: float
    wall_normal_z_max: float
    min_height_m: float
    dilation_radius_m: float


def _build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate a 2D occupancy map from a USD scene.")
    parser.add_argument("--usd-path", required=True, help="Path to the source USD file.")
    parser.add_argument("--output-yaml", required=True, help="Path to the output Nav2 map YAML.")
    parser.add_argument("--output-pgm", default="", help="Path to the output PGM. Defaults next to YAML.")
    parser.add_argument("--resolution", type=float, default=0.05, help="Map resolution in meters per cell.")
    parser.add_argument("--padding-m", type=float, default=0.6, help="Extra border around occupied geometry.")
    parser.add_argument(
        "--wall-normal-z-max",
        type=float,
        default=0.35,
        help="Faces with |normal.z| <= this are treated as vertical obstacles.",
    )
    parser.add_argument(
        "--min-height-m",
        type=float,
        default=0.08,
        help="Ignore faces whose highest vertex is below this Z threshold.",
    )
    parser.add_argument(
        "--dilation-radius-m",
        type=float,
        default=0.08,
        help="Inflate occupied cells by this radius after rasterization.",
    )
    parser.add_argument(
        "--exclude-substring",
        action="append",
        default=[],
        help="Skip prims whose path contains this substring. Can be passed multiple times.",
    )
    return parser


def _default_exclusions() -> tuple[str, ...]:
    return (
        "/Looks",
        "/DomeLight",
        "/SphereLight",
        "/Camera",
        "/GroundPlane",
        "/PhysicsMaterial",
        "/Mobie_grasper2",
        "/Bottle",
        "/Cube",
        "/SmallPallet",
        "/BigPallet",
    )


def _triangulate(face_counts: np.ndarray, face_indices: np.ndarray) -> Iterable[tuple[int, int, int]]:
    cursor = 0
    for count in face_counts.tolist():
        if count < 3:
            cursor += count
            continue
        polygon = face_indices[cursor : cursor + count]
        cursor += count
        anchor = int(polygon[0])
        for offset in range(1, count - 1):
            yield anchor, int(polygon[offset]), int(polygon[offset + 1])


def _world_points(mesh_prim, xform_cache, UsdGeom) -> np.ndarray:
    mesh = UsdGeom.Mesh(mesh_prim)
    points_attr = mesh.GetPointsAttr()
    points = points_attr.Get()
    if not points:
        return np.zeros((0, 3), dtype=np.float64)

    world_transform = xform_cache.GetLocalToWorldTransform(mesh_prim)
    transformed = []
    for point in points:
        world_point = world_transform.Transform(point)
        transformed.append((float(world_point[0]), float(world_point[1]), float(world_point[2])))
    return np.asarray(transformed, dtype=np.float64)


def _collect_vertical_triangles(stage, config: MapConfig, excluded_substrings: tuple[str, ...]):
    Gf, _, UsdGeom = _ensure_usd_imports()
    del Gf

    xform_cache = UsdGeom.XformCache()
    triangles: list[np.ndarray] = []

    for prim in stage.Traverse():
        if not prim.IsActive():
            continue
        if prim.GetTypeName() != "Mesh":
            continue

        prim_path = str(prim.GetPath())
        if any(token in prim_path for token in excluded_substrings):
            continue

        mesh = UsdGeom.Mesh(prim)
        face_counts = np.asarray(mesh.GetFaceVertexCountsAttr().Get() or [], dtype=np.int32)
        face_indices = np.asarray(mesh.GetFaceVertexIndicesAttr().Get() or [], dtype=np.int32)
        if face_counts.size == 0 or face_indices.size == 0:
            continue

        points = _world_points(prim, xform_cache, UsdGeom)
        if points.size == 0:
            continue

        for index_a, index_b, index_c in _triangulate(face_counts, face_indices):
            triangle = points[[index_a, index_b, index_c], :]
            edge_ab = triangle[1] - triangle[0]
            edge_ac = triangle[2] - triangle[0]
            normal = np.cross(edge_ab, edge_ac)
            norm = float(np.linalg.norm(normal))
            if norm <= 1e-9:
                continue
            normal /= norm
            if abs(float(normal[2])) > config.wall_normal_z_max:
                continue
            if float(np.max(triangle[:, 2])) < config.min_height_m:
                continue
            triangles.append(triangle)

    return triangles


def _map_bounds_from_triangles(triangles: list[np.ndarray], padding_m: float) -> tuple[float, float, float, float]:
    if not triangles:
        raise RuntimeError("No obstacle triangles were found in the USD scene after filtering.")

    all_points = np.concatenate(triangles, axis=0)
    min_x = float(np.min(all_points[:, 0])) - padding_m
    min_y = float(np.min(all_points[:, 1])) - padding_m
    max_x = float(np.max(all_points[:, 0])) + padding_m
    max_y = float(np.max(all_points[:, 1])) + padding_m
    return min_x, min_y, max_x, max_y


def _point_in_triangle(px: float, py: float, a: np.ndarray, b: np.ndarray, c: np.ndarray) -> bool:
    v0 = c - a
    v1 = b - a
    v2 = np.array([px, py], dtype=np.float64) - a

    dot00 = float(np.dot(v0, v0))
    dot01 = float(np.dot(v0, v1))
    dot02 = float(np.dot(v0, v2))
    dot11 = float(np.dot(v1, v1))
    dot12 = float(np.dot(v1, v2))
    denom = dot00 * dot11 - dot01 * dot01
    if abs(denom) < 1e-12:
        return False
    inv = 1.0 / denom
    u = (dot11 * dot02 - dot01 * dot12) * inv
    v = (dot00 * dot12 - dot01 * dot02) * inv
    return u >= -1e-9 and v >= -1e-9 and (u + v) <= 1.0 + 1e-9


def _rasterize_triangles(
    triangles: list[np.ndarray],
    resolution: float,
    origin_x: float,
    origin_y: float,
    width: int,
    height: int,
) -> np.ndarray:
    occupancy = np.zeros((height, width), dtype=np.uint8)

    for triangle in triangles:
        tri_xy = triangle[:, :2]
        min_x = float(np.min(tri_xy[:, 0]))
        min_y = float(np.min(tri_xy[:, 1]))
        max_x = float(np.max(tri_xy[:, 0]))
        max_y = float(np.max(tri_xy[:, 1]))
        start_x = max(0, int(math.floor((min_x - origin_x) / resolution)))
        end_x = min(width - 1, int(math.ceil((max_x - origin_x) / resolution)))
        start_y = max(0, int(math.floor((min_y - origin_y) / resolution)))
        end_y = min(height - 1, int(math.ceil((max_y - origin_y) / resolution)))
        a, b, c = tri_xy[0], tri_xy[1], tri_xy[2]

        for grid_y in range(start_y, end_y + 1):
            world_y = origin_y + (grid_y + 0.5) * resolution
            image_y = height - 1 - grid_y
            for grid_x in range(start_x, end_x + 1):
                world_x = origin_x + (grid_x + 0.5) * resolution
                if _point_in_triangle(world_x, world_y, a, b, c):
                    occupancy[image_y, grid_x] = 1

    return occupancy


def _dilate(mask: np.ndarray, radius_cells: int) -> np.ndarray:
    if radius_cells <= 0:
        return mask

    dilated = mask.copy()
    occupied = np.argwhere(mask > 0)
    offsets = []
    for dy in range(-radius_cells, radius_cells + 1):
        for dx in range(-radius_cells, radius_cells + 1):
            if dx * dx + dy * dy <= radius_cells * radius_cells:
                offsets.append((dy, dx))

    for row, col in occupied.tolist():
        for dy, dx in offsets:
            rr = row + dy
            cc = col + dx
            if 0 <= rr < mask.shape[0] and 0 <= cc < mask.shape[1]:
                dilated[rr, cc] = 1
    return dilated


def _write_pgm(path: Path, occupancy_mask: np.ndarray) -> None:
    image = np.where(occupancy_mask > 0, 0, 254).astype(np.uint8)
    with path.open("wb") as file_obj:
        header = f"P5\n{image.shape[1]} {image.shape[0]}\n255\n".encode("ascii")
        file_obj.write(header)
        file_obj.write(image.tobytes())


def _write_yaml(path: Path, image_name: str, resolution: float, origin_x: float, origin_y: float) -> None:
    text = (
        f"image: {image_name}\n"
        "mode: trinary\n"
        f"resolution: {resolution:.6f}\n"
        f"origin: [{origin_x:.6f}, {origin_y:.6f}, 0.0]\n"
        "negate: 0\n"
        "occupied_thresh: 0.65\n"
        "free_thresh: 0.196\n"
    )
    path.write_text(text, encoding="utf-8")


def generate_map(
    usd_path: Path,
    output_yaml: Path,
    output_pgm: Path,
    config: MapConfig,
    excluded_substrings: tuple[str, ...],
) -> dict[str, float]:
    _, Usd, _ = _ensure_usd_imports()

    stage = Usd.Stage.Open(str(usd_path))
    if stage is None:
        raise RuntimeError(f"Failed to open USD stage: {usd_path}")

    triangles = _collect_vertical_triangles(stage, config, excluded_substrings)
    min_x, min_y, max_x, max_y = _map_bounds_from_triangles(triangles, config.padding_m)
    width = int(math.ceil((max_x - min_x) / config.resolution))
    height = int(math.ceil((max_y - min_y) / config.resolution))

    occupancy = _rasterize_triangles(triangles, config.resolution, min_x, min_y, width, height)
    radius_cells = max(int(math.ceil(config.dilation_radius_m / config.resolution)), 0)
    occupancy = _dilate(occupancy, radius_cells)

    output_yaml.parent.mkdir(parents=True, exist_ok=True)
    output_pgm.parent.mkdir(parents=True, exist_ok=True)
    _write_pgm(output_pgm, occupancy)
    _write_yaml(output_yaml, output_pgm.name, config.resolution, min_x, min_y)

    return {
        "width": float(width),
        "height": float(height),
        "origin_x": min_x,
        "origin_y": min_y,
        "occupied_cells": float(int(np.count_nonzero(occupancy))),
    }


def main() -> int:
    if os.environ.get("ROBOT_NAV_USD_REEXEC") != "1":
        return _run_with_isaac_python()

    args = _build_argparser().parse_args()
    output_yaml = Path(args.output_yaml).resolve()
    output_pgm = Path(args.output_pgm).resolve() if args.output_pgm else output_yaml.with_suffix(".pgm")
    config = MapConfig(
        resolution=float(args.resolution),
        padding_m=float(args.padding_m),
        wall_normal_z_max=float(args.wall_normal_z_max),
        min_height_m=float(args.min_height_m),
        dilation_radius_m=float(args.dilation_radius_m),
    )
    excluded_substrings = tuple(_default_exclusions()) + tuple(args.exclude_substring)
    stats = generate_map(
        usd_path=Path(args.usd_path).resolve(),
        output_yaml=output_yaml,
        output_pgm=output_pgm,
        config=config,
        excluded_substrings=excluded_substrings,
    )
    print(
        "Generated occupancy map:",
        f"yaml={output_yaml}",
        f"pgm={output_pgm}",
        f"size={int(stats['width'])}x{int(stats['height'])}",
        f"origin=({stats['origin_x']:.3f}, {stats['origin_y']:.3f})",
        f"occupied_cells={int(stats['occupied_cells'])}",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
