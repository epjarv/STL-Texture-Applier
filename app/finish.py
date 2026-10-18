"""Carve a height map into a triangle mesh and export an STL."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import trimesh
from trimesh.geometry import faces_to_edges

MAX_FACES = 1_400_000
MIN_REPEATS = 0.5
MAX_REPEATS = 16.0
# Tuned so the on-screen triangle estimate matches a subdivided surface.
_AREA_FACTOR = 0.145


@dataclass(frozen=True)
class FinishSettings:
    depth: float
    scale: float
    detail: float
    contrast: float
    rotation: float
    offset_x: float
    offset_y: float
    invert: bool
    centered: bool


@dataclass
class PreparedSurface:
    vertices: np.ndarray
    faces: np.ndarray
    normals: np.ndarray
    clamped: bool


@dataclass
class FinishResult:
    vertices: np.ndarray
    faces: np.ndarray
    triangles: int
    clamped: bool


def load_mesh(path: str) -> trimesh.Trimesh:
    loaded = trimesh.load(path, force="mesh", process=True)
    if isinstance(loaded, trimesh.Scene):
        parts = [g for g in loaded.geometry.values() if isinstance(g, trimesh.Trimesh)]
        if not parts:
            raise ValueError("That file did not contain a triangle mesh.")
        loaded = trimesh.util.concatenate(parts)
    if not isinstance(loaded, trimesh.Trimesh):
        raise ValueError("That file did not contain a triangle mesh.")
    if loaded.faces.size == 0 or loaded.vertices.size == 0:
        raise ValueError("That mesh is empty.")
    if loaded.is_watertight and float(loaded.volume) < 0:
        loaded.invert()
    return loaded


def sample_cube() -> trimesh.Trimesh:
    return trimesh.creation.box(extents=(20.0, 20.0, 20.0))


def edge_target(size: float, area: float, scale: float, detail: float) -> tuple[float, bool]:
    """Edge length that keeps the pattern readable without an enormous file."""
    size = max(float(size), 1e-6)
    repeats = float(np.clip(scale, MIN_REPEATS, MAX_REPEATS))
    samples = 14.0 + float(np.clip(detail, 0.0, 1.0)) * 46.0
    max_edge = (size / repeats) / samples
    clamped = False
    triangle_area = max(max_edge * max_edge * _AREA_FACTOR, 1e-12)
    estimate = area / triangle_area
    while estimate > MAX_FACES and max_edge < size:
        max_edge *= 1.18
        triangle_area = max_edge * max_edge * _AREA_FACTOR
        estimate = area / triangle_area
        clamped = True
    max_edge = max(max_edge, size / 900.0)
    return float(max_edge), clamped


def estimate_triangles(area: float, max_edge: float, original_faces: int) -> int:
    estimate = int(area / max(max_edge * max_edge * _AREA_FACTOR, 1e-12))
    return max(estimate, original_faces)


def _longest_edge(vertices: np.ndarray, faces: np.ndarray) -> float:
    if len(faces) == 0:
        return 0.0
    a = vertices[faces[:, 0]]
    b = vertices[faces[:, 1]]
    c = vertices[faces[:, 2]]
    lengths = np.maximum(
        np.linalg.norm(a - b, axis=1),
        np.maximum(np.linalg.norm(b - c, axis=1), np.linalg.norm(c - a, axis=1)),
    )
    return float(lengths.max())


def _vertex_normals(vertices: np.ndarray, faces: np.ndarray) -> np.ndarray:
    tris = vertices[faces]
    face_normals = np.cross(tris[:, 1] - tris[:, 0], tris[:, 2] - tris[:, 0])
    normals = np.zeros_like(vertices)
    for corner in range(3):
        np.add.at(normals, faces[:, corner], face_normals)
    lengths = np.linalg.norm(normals, axis=1, keepdims=True)
    return normals / np.clip(lengths, 1e-12, None)


def prepare_surface(
    vertices: np.ndarray,
    faces: np.ndarray,
    max_edge: float,
    clamped: bool,
) -> PreparedSurface:
    vertices = np.asarray(vertices, dtype=np.float64)
    faces = np.asarray(faces, dtype=np.int64)
    if len(faces) > 0:
        vertices, faces, hit_cap = _refine_mesh(vertices, faces, max_edge)
        clamped = clamped or hit_cap
    normals = _vertex_normals(vertices, faces)
    return PreparedSurface(
        vertices=np.ascontiguousarray(vertices, dtype=np.float64),
        faces=np.ascontiguousarray(faces, dtype=np.int64),
        normals=np.ascontiguousarray(normals, dtype=np.float64),
        clamped=clamped,
    )


def _bisect_long_edges(
    vertices: np.ndarray,
    faces: np.ndarray,
    max_edge: float,
) -> tuple[np.ndarray, np.ndarray, bool]:
    """Bisect every edge longer than max_edge. Shared midpoints keep the mesh closed."""
    edges = np.sort(faces_to_edges(faces), axis=1)
    unique, inverse = trimesh.grouping.unique_rows(edges)
    edges_unique = edges[unique]
    lengths = np.linalg.norm(
        vertices[edges_unique[:, 0]] - vertices[edges_unique[:, 1]],
        axis=1,
    )
    long_edge = lengths > max_edge
    if not np.any(long_edge):
        return vertices, faces, False

    midpoint = np.full(len(unique), -1, dtype=np.int64)
    midpoint[long_edge] = np.arange(int(long_edge.sum())) + len(vertices)
    vertices = np.vstack((vertices, vertices[edges_unique[long_edge]].mean(axis=1)))
    face_mid = midpoint[inverse].reshape((-1, 3))
    split = face_mid >= 0
    count = split.sum(axis=1)
    parts: list[np.ndarray] = [faces[count == 0]]

    one = count == 1
    if np.any(one):
        choice = np.argmax(split[one], axis=1)
        row = np.arange(len(choice))
        tri, mid = faces[one], face_mid[one]
        a = tri[row, choice]
        b = tri[row, (choice + 1) % 3]
        c = tri[row, (choice + 2) % 3]
        point = mid[row, choice]
        parts.append(np.column_stack((a, point, c, point, b, c)).reshape((-1, 3)))

    two = count == 2
    if np.any(two):
        choice = (np.argmin(split[two], axis=1) + 1) % 3
        row = np.arange(len(choice))
        tri, mid = faces[two], face_mid[two]
        a = tri[row, choice]
        b = tri[row, (choice + 1) % 3]
        c = tri[row, (choice + 2) % 3]
        left = mid[row, choice]
        right = mid[row, (choice + 1) % 3]
        use_short = (
            np.sum((vertices[a] - vertices[right]) ** 2, axis=1)
            <= np.sum((vertices[left] - vertices[c]) ** 2, axis=1)
        )[:, None]
        corner = np.column_stack((left, b, right))
        quad_a = np.where(use_short, np.column_stack((a, left, right)), np.column_stack((a, left, c)))
        quad_b = np.where(use_short, np.column_stack((a, right, c)), np.column_stack((left, right, c)))
        parts.append(np.vstack((corner, quad_a, quad_b)))

    three = count == 3
    if np.any(three):
        tri, mid = faces[three], face_mid[three]
        v0, v1, v2 = tri.T
        m0, m1, m2 = mid.T
        parts.append(
            np.column_stack((v0, m0, m2, m0, v1, m1, m2, m1, v2, m0, m1, m2)).reshape((-1, 3))
        )

    faces = np.vstack([part for part in parts if len(part)]).astype(np.int64, copy=False)
    return vertices, faces, True


def _refine_mesh(
    vertices: np.ndarray,
    faces: np.ndarray,
    max_edge: float,
) -> tuple[np.ndarray, np.ndarray, bool]:
    """Split edges that are too long to hold the pattern, and stop near the face cap.

    Each pass only bisects edges that are still too long, so a dense hull stays
    put while a coarse deck or cabin picks up vertices.
    """
    clamped = False
    for _ in range(8):
        if len(faces) >= MAX_FACES:
            clamped = True
            break
        longest = _longest_edge(vertices, faces)
        if longest <= max_edge * 1.001:
            break
        vertices, faces, changed = _bisect_long_edges(vertices, faces, max_edge)
        if not changed:
            break
        if len(faces) >= MAX_FACES:
            clamped = True
            break
    else:
        if _longest_edge(vertices, faces) > max_edge * 1.05:
            clamped = True
    return vertices, faces, clamped


def _sample(image: np.ndarray, u: np.ndarray, v: np.ndarray) -> np.ndarray:
    """Tile the height map. v grows upward and the image stays upright."""
    height, width = image.shape
    u = np.mod(u, 1.0)
    v = np.mod(-v, 1.0)
    x = u * width
    y = v * height
    x0 = np.floor(x).astype(np.int32) % width
    y0 = np.floor(y).astype(np.int32) % height
    x1 = (x0 + 1) % width
    y1 = (y0 + 1) % height
    tx = (x - np.floor(x)).astype(np.float32)
    ty = (y - np.floor(y)).astype(np.float32)
    h00 = image[y0, x0]
    h10 = image[y0, x1]
    h01 = image[y1, x0]
    h11 = image[y1, x1]
    return h00 * (1 - tx) * (1 - ty) + h10 * tx * (1 - ty) + h01 * (1 - tx) * ty + h11 * tx * ty


def _smoothstep(edge0: float, edge1: float, value: np.ndarray) -> np.ndarray:
    t = np.clip((value - edge0) / (edge1 - edge0), 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)


def _wrap_part(
    points: np.ndarray,
    normals: np.ndarray,
    image: np.ndarray,
    p: np.ndarray,
    per_unit: float,
    center: np.ndarray,
    angle: float,
    ox: float,
    oy: float,
) -> np.ndarray:
    """Planar on decks and floors, continuous wrap around walls and curves.

    The wrap uses one angular coordinate, so the pattern keeps the same spacing
    as it turns the stern instead of smearing two flat projections together.
    An integer number of repeats makes the seam meet.
    """
    midpoint = (points.min(axis=0) + points.max(axis=0)) * 0.5
    rel = points - midpoint
    radius = np.hypot(rel[:, 0], rel[:, 1])
    radius_ref = max(float(np.percentile(radius, 98)), 1e-6)
    repeats_around = max(1, int(round(2.0 * np.pi * radius_ref * per_unit)))
    around = np.arctan2(rel[:, 1], rel[:, 0]) / (2.0 * np.pi) * repeats_around
    height = rel[:, 2] * per_unit
    u_side, v_side = _rotate(around, height, angle)
    h_side = _sample(image, u_side + ox, v_side + oy)

    u_cap, v_cap = _rotate(p[:, 0] - center[0], p[:, 1] - center[1], angle)
    h_cap = _sample(image, u_cap + ox, v_cap + oy)

    # Sides and curved corners stay on the wrap. Decks and bottoms use the flat map.
    on_cap = _smoothstep(0.38, 0.8, np.abs(normals[:, 2]))
    return h_side * (1.0 - on_cap) + h_cap * on_cap


def _rotate(u: np.ndarray, v: np.ndarray, degrees: float) -> tuple[np.ndarray, np.ndarray]:
    if abs(degrees) < 1e-6:
        return u, v
    theta = np.deg2rad(degrees)
    cos_t = np.cos(theta)
    sin_t = np.sin(theta)
    return u * cos_t - v * sin_t, u * sin_t + v * cos_t


def _project(
    points: np.ndarray,
    normals: np.ndarray,
    image: np.ndarray,
    settings: FinishSettings,
    bounds_min: np.ndarray,
    bounds_max: np.ndarray,
) -> np.ndarray:
    size = max(float(np.max(bounds_max - bounds_min)), 1e-6)
    per_unit = float(np.clip(settings.scale, MIN_REPEATS, MAX_REPEATS)) / size
    p = (points - bounds_min) * per_unit
    center = (bounds_max - bounds_min) * per_unit * 0.5
    return _wrap_part(
        points,
        normals,
        image,
        p,
        per_unit,
        center,
        settings.rotation,
        settings.offset_x,
        settings.offset_y,
    )


def _base_keep(vertices: np.ndarray, normals: np.ndarray) -> np.ndarray:
    """Leave the underside untouched so the part still sits flat on a print bed.

    Faces aimed downward get no relief. Vertices on the lowest plane stay put
    even where the side wall shares that edge.
    """
    z = vertices[:, 2]
    span = max(float(z.max() - z.min()), 1e-6)
    on_bed = z <= (z.min() + max(span * 5e-4, 1e-4))
    # 0 when the surface faces down, 1 once it has turned up toward the side.
    keep = _smoothstep(-0.62, -0.22, normals[:, 2])
    keep = np.where(on_bed, 0.0, keep)
    return keep.astype(np.float64)


def displace(
    surface: PreparedSurface,
    image: np.ndarray,
    settings: FinishSettings,
    bounds_min: np.ndarray,
    bounds_max: np.ndarray,
) -> FinishResult:
    if settings.depth <= 0:
        return FinishResult(
            vertices=np.array(surface.vertices, copy=True),
            faces=surface.faces,
            triangles=int(len(surface.faces)),
            clamped=surface.clamped,
        )
    height = _project(surface.vertices, surface.normals, image, settings, bounds_min, bounds_max)
    height = np.clip((height - 0.5) * settings.contrast + 0.5, 0.0, 1.0)
    if settings.invert:
        height = 1.0 - height
    if settings.centered:
        amount = (height - 0.5) * 2.0 * settings.depth
    else:
        amount = height * settings.depth
    amount *= _base_keep(surface.vertices, surface.normals)
    vertices = surface.vertices + surface.normals * amount[:, None]
    bed = float(surface.vertices[:, 2].min())
    vertices[:, 2] = np.maximum(vertices[:, 2], bed)
    return FinishResult(
        vertices=vertices,
        faces=surface.faces,
        triangles=int(len(surface.faces)),
        clamped=surface.clamped,
    )


def export_stl(path: str, vertices: np.ndarray, faces: np.ndarray) -> None:
    mesh = trimesh.Trimesh(vertices, faces, process=False)
    mesh.export(path, file_type="stl")
