from __future__ import annotations

import argparse
import contextlib
import csv
import hashlib
import heapq
import io
import math
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

try:
    import numpy as np
    from scipy.optimize import least_squares
    from scipy.spatial import ConvexHull, cKDTree
except ImportError as exc:
    raise SystemExit(
        "Required packages are missing. Run: "
        "python -m pip install numpy==2.4.4 scipy==1.17.1"
    ) from exc


PROGRAM_VERSION = "1.0.0"
SAMPLE1_PROFILE = {
    "pca_trim_percent": [1.8, 0.55, 0.45],
    "geometry_calibration_scale": 1.0561673547373531,
    "pca_parameter_origin": "training-fold selection; held-out fold 1 of 5",
    "geometry_scale_origin": "recorded preprocessing calibration_scale for data.ply",
}
SAMPLE1_INPUT_SHA256 = '9a75ccdb05976550c9fbc5f90cbd3f3c91dd0c1dc142eea97a416795a35212cf'
SOURCE_PROVENANCE = {
    "step10_gai3.py": "bfa2008ab9b963a8d32b9cb656aee89ab1521d4c802d6edbad6e8160453d2566",
    "triaxial_method_comparison.py": "1552fb06c3bdacfb9cb06898e6eea01c4212b18213d02fecc5d072149a926787",
    "surface_area_method_comparison.py": "194362eefffdcdc965d2efa52ec465abde10b8e876b813ce1893ad74e4023634",
    "volume_method_comparison.py": "01a234cfc2270ed44dc3c9b0640db68aba5b826f80fd89e23113c0cfdac5abb8"
}

PLY_SCALAR_TYPES: Dict[str, str] = {
    "char": "i1",
    "int8": "i1",
    "uchar": "u1",
    "uint8": "u1",
    "short": "i2",
    "int16": "i2",
    "ushort": "u2",
    "uint16": "u2",
    "int": "i4",
    "int32": "i4",
    "uint": "u4",
    "uint32": "u4",
    "float": "f4",
    "float32": "f4",
    "double": "f8",
    "float64": "f8",
}

@dataclass
class BerryParameters:

    knn: int = 30
    radius_min: float = 0.0075
    radius_max: float = 0.018
    vote_radius_steps: int = 10
    vote_voxel_size: float = 0.0025
    vote_curvature_quantile: float = 0.72
    vote_min_weight: float = 25.0
    vote_min_count: int = 25
    vote_nms_distance: float = 0.0065
    max_vote_peaks: int = 240
    max_vote_samples: int = 50000
    peak_fit_max_shift: float = 0.008
    peak_prefilter_normal_angle: float = 65.0
    ransac_iterations: int = 180
    inlier_threshold: float = 0.0019
    min_inliers: int = 45
    min_inlier_ratio: float = 0.15
    min_coverage_angle: float = 45.0
    normal_angle: float = 50.0
    min_normal_agreement: float = 0.52
    connectivity_edge_distance: float = 0.0035
    candidate_merge_distance: float = 0.008
    candidate_merge_radius_fraction: float = 0.35
    candidate_overlap_threshold: float = 0.7
    candidate_min_separation_fraction: float = 0.55
    core_residual: float = 0.0013
    core_normal_angle: float = 42.0
    core_ambiguity_margin: float = 0.0003
    core_cost_margin: float = 0.12
    min_core_points: int = 35
    grow_angle: float = 45.0
    grow_normal_angle: float = 55.0
    grow_edge_distance: float = 0.0032
    growth_knn: int = 12
    adaptive_edge_factor: float = 3.0
    grow_soft_curvature: float = 0.12
    grow_hard_curvature: float = 0.28
    grow_sphere_residual: float = 0.0028
    max_path_cost: float = 18.0
    competition_margin: float = 0.7
    boundary_sphere_residual: float = 0.0025
    boundary_residual_margin: float = 0.0004
    assignment_rounds: int = 2
    final_assignment_rounds: int = 1
    assignment_knn: int = 24
    assignment_distance: float = 0.0045
    min_vote_fraction: float = 0.62
    assignment_residual: float = 0.0023
    assignment_normal_angle: float = 50.0
    assignment_ambiguity_margin: float = 0.00035
    refit_support_threshold: float = 0.0022
    max_center_shift: float = 0.0035
    final_min_label_points: int = 45
    final_min_inliers: int = 42
    final_min_coverage_angle: float = 42.0
    final_min_normal_agreement: float = 0.5
    final_merge_distance: float = 0.008
    final_merge_radius_fraction: float = 0.35
    final_overlap_threshold: float = 0.7
    final_min_separation_fraction: float = 0.55
    random_seed: int = 42


@dataclass
class SphereCandidate:
    """一个果粒球候选及其几何质量指标。"""

    candidate_id: int
    seed_index: int
    patch_size: int
    center: np.ndarray
    radius: float
    inlier_indices: np.ndarray = field(repr=False)
    inlier_count: int = 0
    inlier_ratio: float = 0.0
    rms: float = math.inf
    coverage_angle: float = 0.0
    normal_agreement: float = 0.0
    score: float = 0.0
    status: str = "accepted"
    reason: str = ""


def read_vertex_ply(path: Path) -> Tuple[np.ndarray, np.ndarray]:
    """读取仅含标量 vertex 属性的 ASCII 或二进制 PLY。"""
    with path.open("rb") as handle:
        first = handle.readline().decode("ascii", errors="strict").strip()
        if first != "ply":
            raise ValueError(f"{path} 不是有效的 PLY 文件")

        file_format = None
        vertex_count = None
        vertex_properties: List[Tuple[str, str]] = []
        current_element = None

        while True:
            raw_line = handle.readline()
            if not raw_line:
                raise ValueError("PLY 头部缺少 end_header")
            line = raw_line.decode("ascii", errors="strict").strip()
            if not line or line.startswith("comment") or line.startswith("obj_info"):
                continue
            parts = line.split()
            if parts[0] == "format":
                file_format = parts[1]
            elif parts[0] == "element":
                current_element = parts[1]
                if current_element == "vertex":
                    vertex_count = int(parts[2])
            elif parts[0] == "property" and current_element == "vertex":
                if parts[1] == "list":
                    raise ValueError("本脚本不支持 vertex 的 list 属性")
                vertex_properties.append((parts[2], parts[1]))
            elif parts[0] == "end_header":
                break

        if file_format is None or vertex_count is None:
            raise ValueError("PLY 头部缺少 format 或 vertex 数量")
        if not vertex_properties:
            raise ValueError("PLY 没有 vertex 属性")

        property_names = [name for name, _ in vertex_properties]
        for coordinate in ("x", "y", "z"):
            if coordinate not in property_names:
                raise ValueError(f"PLY 缺少坐标属性 {coordinate}")

        if file_format == "ascii":
            raw = np.loadtxt(handle, max_rows=vertex_count, ndmin=2)
            if len(raw) != vertex_count:
                raise ValueError(f"PLY 声明 {vertex_count} 个点，实际读取 {len(raw)} 个")
            columns = {name: raw[:, index] for index, name in enumerate(property_names)}
        elif file_format in {"binary_little_endian", "binary_big_endian"}:
            endian = "<" if file_format == "binary_little_endian" else ">"
            dtype_fields = []
            for name, type_name in vertex_properties:
                if type_name not in PLY_SCALAR_TYPES:
                    raise ValueError(f"不支持的 PLY 属性类型: {type_name}")
                dtype_fields.append((name, endian + PLY_SCALAR_TYPES[type_name]))
            raw = np.fromfile(handle, dtype=np.dtype(dtype_fields), count=vertex_count)
            if len(raw) != vertex_count:
                raise ValueError(f"PLY 声明 {vertex_count} 个点，实际读取 {len(raw)} 个")
            columns = {name: raw[name] for name in property_names}
        else:
            raise ValueError(f"不支持的 PLY 格式: {file_format}")

    points = np.column_stack([columns["x"], columns["y"], columns["z"]]).astype(np.float64)
    if all(name in columns for name in ("red", "green", "blue")):
        colors = np.column_stack([columns["red"], columns["green"], columns["blue"]])
        if np.issubdtype(colors.dtype, np.floating) and colors.size and np.nanmax(colors) <= 1.0:
            colors = colors * 255.0
        colors = np.clip(np.rint(colors), 0, 255).astype(np.uint8)
    else:
        colors = np.full((len(points), 3), 180, dtype=np.uint8)

    finite = np.isfinite(points).all(axis=1)
    if not np.all(finite):
        points = points[finite]
        colors = colors[finite]
    return points, colors


def estimate_normals_and_curvature(
    points: np.ndarray,
    tree: cKDTree,
    knn: int,
    chunk_size: int = 5000,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """通过局部 PCA 估计法向、曲率，并返回 KNN 距离和索引。"""
    knn = min(max(knn, 6), len(points))
    distances, neighbor_indices = tree.query(points, k=knn, workers=-1)
    if knn == 1:
        distances = distances[:, None]
        neighbor_indices = neighbor_indices[:, None]

    normals = np.zeros_like(points)
    curvature = np.zeros(len(points), dtype=np.float64)
    for start in range(0, len(points), chunk_size):
        end = min(start + chunk_size, len(points))
        neighborhoods = points[neighbor_indices[start:end]]
        centered = neighborhoods - neighborhoods.mean(axis=1, keepdims=True)
        covariance = np.einsum("nki,nkj->nij", centered, centered)
        covariance /= max(knn - 1, 1)
        eigenvalues, eigenvectors = np.linalg.eigh(covariance)
        normals[start:end] = eigenvectors[:, :, 0]
        curvature[start:end] = eigenvalues[:, 0] / np.maximum(eigenvalues.sum(axis=1), 1e-15)

    # 用指向整串点云外部的方向消除 PCA 法向的正负二义性。
    cloud_center = np.median(points, axis=0)
    radial = points - cloud_center
    flip = np.einsum("ij,ij->i", normals, radial) < 0
    normals[flip] *= -1.0
    return normals, curvature, distances, neighbor_indices


def sphere_from_four(sample: np.ndarray) -> Optional[Tuple[np.ndarray, float]]:
    """由四个非共面点计算球心和半径。"""
    reference = sample[0]
    matrix = 2.0 * (sample[1:] - reference)
    rhs = np.sum(sample[1:] ** 2, axis=1) - np.sum(reference ** 2)
    try:
        center = np.linalg.solve(matrix, rhs)
    except np.linalg.LinAlgError:
        return None
    radius = float(np.linalg.norm(reference - center))
    if not np.isfinite(radius) or not np.all(np.isfinite(center)):
        return None
    return center, radius


def coverage_angle_degrees(directions: np.ndarray, max_samples: int = 180) -> float:
    """用球面支持方向的最大夹角近似球面覆盖程度。"""
    if len(directions) < 2:
        return 0.0
    if len(directions) > max_samples:
        sample_indices = np.linspace(0, len(directions) - 1, max_samples).astype(int)
        directions = directions[sample_indices]
    minimum_dot = float(np.min(np.clip(directions @ directions.T, -1.0, 1.0)))
    return math.degrees(math.acos(minimum_dot))


def sphere_intersection_fraction(
    center_a: np.ndarray,
    radius_a: float,
    center_b: np.ndarray,
    radius_b: float,
) -> float:
    """返回两球交集体积占较小球体积的比例。"""
    distance = float(np.linalg.norm(center_a - center_b))
    small = min(radius_a, radius_b)
    large = max(radius_a, radius_b)
    if distance >= radius_a + radius_b:
        return 0.0
    if distance <= large - small or distance <= 1e-12:
        return 1.0
    intersection = (
        math.pi
        * (radius_a + radius_b - distance) ** 2
        * (
            distance ** 2
            + 2.0 * distance * (radius_a + radius_b)
            - 3.0 * (radius_a - radius_b) ** 2
        )
        / (12.0 * distance)
    )
    small_volume = 4.0 * math.pi * small ** 3 / 3.0
    return float(np.clip(intersection / small_volume, 0.0, 1.0))


@dataclass
class VoteAccumulator:
    """多尺度球心投票累加结果。"""

    voter_indices: np.ndarray
    voxel_centers: np.ndarray
    voxel_weights: np.ndarray
    voxel_counts: np.ndarray
    peak_centers: np.ndarray
    peak_weights: np.ndarray
    peak_counts: np.ndarray
    sampled_votes: np.ndarray
    sampled_vote_weights: np.ndarray


@dataclass
class CompetitiveGrowthResult:
    """果粒级竞争区域生长状态。"""

    raw_labels: np.ndarray
    labels: np.ndarray
    best_cost: np.ndarray
    second_cost: np.ndarray
    second_labels: np.ndarray
    ambiguity_mask: np.ndarray
    unreachable_mask: np.ndarray


@dataclass
class AssignmentStats:
    """一轮待定点补分统计。"""

    round_id: int
    pending_before: int
    newly_assigned: int
    pending_after: int
    rejected_no_neighbor: int
    rejected_vote: int
    rejected_sphere: int
    rejected_normal: int
    rejected_ambiguous: int


def normal_coherence(normals: np.ndarray, neighbor_indices: np.ndarray, use_knn: int = 10) -> np.ndarray:
    """计算点法向与邻域法向的无向一致性。"""
    use_knn = min(max(use_knn, 2), neighbor_indices.shape[1])
    neighbors = neighbor_indices[:, 1:use_knn]
    dots = np.abs(np.einsum("nij,nj->ni", normals[neighbors], normals))
    return np.mean(dots, axis=1)


def multiscale_vote_accumulator(
    points: np.ndarray,
    normals: np.ndarray,
    curvature: np.ndarray,
    coherence: np.ndarray,
    radii: np.ndarray,
    voxel_size: float,
    curvature_quantile: float,
    min_weight: float,
    min_count: int,
    nms_distance: float,
    max_peaks: int,
    max_vote_samples: int,
    rng: np.random.Generator,
) -> VoteAccumulator:
    """双向多尺度球心投票，并在体素累加器上执行三维非极大值抑制。"""
    eligible = np.ones(len(points), dtype=bool)
    source_curvature = curvature[eligible]
    if not len(source_curvature):
        empty3 = np.empty((0, 3), dtype=np.float64)
        empty1 = np.empty(0, dtype=np.float64)
        return VoteAccumulator(
            np.empty(0, dtype=np.int64), empty3, empty1, np.empty(0, dtype=np.int64),
            empty3, empty1, np.empty(0, dtype=np.int64), empty3, empty1,
        )
    curvature_limit = float(np.quantile(source_curvature, curvature_quantile))
    eligible &= curvature <= curvature_limit
    voter_indices = np.where(eligible)[0]
    if not len(voter_indices):
        empty3 = np.empty((0, 3), dtype=np.float64)
        empty1 = np.empty(0, dtype=np.float64)
        return VoteAccumulator(
            voter_indices, empty3, empty1, np.empty(0, dtype=np.int64),
            empty3, empty1, np.empty(0, dtype=np.int64), empty3, empty1,
        )

    curvature_scale = max(float(np.percentile(curvature[voter_indices], 75)), 1e-6)
    base_weights = np.exp(-curvature[voter_indices] / curvature_scale)
    base_weights *= 0.25 + 0.75 * coherence[voter_indices]
    primary_votes = (
        points[voter_indices, None, :]
        - normals[voter_indices, None, :] * radii[None, :, None]
    ).reshape(-1, 3)
    reverse_votes = (
        points[voter_indices, None, :]
        + normals[voter_indices, None, :] * radii[None, :, None]
    ).reshape(-1, 3)
    repeated_weights = np.repeat(base_weights, len(radii))
    votes = np.vstack([primary_votes, reverse_votes])
    weights = np.concatenate([
        repeated_weights,
        repeated_weights * 0.72,
    ])

    voxel_keys = np.floor(votes / voxel_size).astype(np.int32)
    _, inverse = np.unique(voxel_keys, axis=0, return_inverse=True)
    voxel_weights = np.bincount(inverse, weights=weights)
    voxel_counts = np.bincount(inverse).astype(np.int64)
    denominator = np.maximum(voxel_weights, 1e-12)
    voxel_centers = np.column_stack([
        np.bincount(inverse, weights=weights * votes[:, axis], minlength=len(voxel_weights))
        / denominator
        
        for axis in range(3)
    ])

    peak_indices: List[int] = []
    for voxel_index in np.argsort(voxel_weights)[::-1]:
        if voxel_weights[voxel_index] < min_weight:
            break
        if voxel_counts[voxel_index] < min_count:
            continue
        center = voxel_centers[voxel_index]
        if peak_indices:
            kept_centers = voxel_centers[np.asarray(peak_indices, dtype=np.int64)]
            if np.min(np.linalg.norm(kept_centers - center, axis=1)) < nms_distance:
                continue
        peak_indices.append(int(voxel_index))
        adaptive_peak_limit = min(
            360,
            max(max_peaks, int(math.ceil(len(voter_indices) / 230.0))),
        )
        if len(peak_indices) >= adaptive_peak_limit:
            break

    peak_array = np.asarray(peak_indices, dtype=np.int64)
    if len(votes) > max_vote_samples:
        probabilities = weights / np.sum(weights)
        sample_indices = rng.choice(len(votes), max_vote_samples, replace=False, p=probabilities)
    else:
        sample_indices = np.arange(len(votes))
    return VoteAccumulator(
        voter_indices=voter_indices,
        voxel_centers=voxel_centers,
        voxel_weights=voxel_weights,
        voxel_counts=voxel_counts,
        peak_centers=voxel_centers[peak_array] if len(peak_array) else np.empty((0, 3)),
        peak_weights=voxel_weights[peak_array] if len(peak_array) else np.empty(0),
        peak_counts=voxel_counts[peak_array] if len(peak_array) else np.empty(0, dtype=np.int64),
        sampled_votes=votes[sample_indices],
        sampled_vote_weights=weights[sample_indices],
    )


def largest_inlier_component(
    indices: np.ndarray,
    point_count: int,
    neighbor_indices: np.ndarray,
    neighbor_distances: np.ndarray,
    max_edge_distance: float,
) -> np.ndarray:
    """保留球面内点在原 KNN 图上的最大连通分量，抑制拼接多个果粒的假球。"""
    if not len(indices):
        return indices
    membership = np.zeros(point_count, dtype=bool)
    membership[indices] = True
    visited = np.zeros(point_count, dtype=bool)
    best_component: List[int] = []
    for start in indices:
        start = int(start)
        if visited[start]:
            continue
        visited[start] = True
        stack = [start]
        component: List[int] = []
        while stack:
            current = stack.pop()
            component.append(current)
            valid = neighbor_distances[current, 1:] <= max_edge_distance
            for neighbor in neighbor_indices[current, 1:][valid]:
                neighbor = int(neighbor)
                if membership[neighbor] and not visited[neighbor]:
                    visited[neighbor] = True
                    stack.append(neighbor)
        if len(component) > len(best_component):
            best_component = component
    return np.asarray(best_component, dtype=np.int64)


def rejected_candidate(candidate_id: int, seed_index: int, reason: str) -> SphereCandidate:
    return SphereCandidate(
        candidate_id=candidate_id,
        seed_index=seed_index,
        patch_size=0,
        center=np.full(3, np.nan),
        radius=math.nan,
        inlier_indices=np.empty(0, dtype=np.int64),
        status="rejected",
        reason=reason,
    )


def fit_vote_peak_sphere(
    candidate_id: int,
    peak_center: np.ndarray,
    points: np.ndarray,
    normals: np.ndarray,
    tree: cKDTree,
    neighbor_indices: np.ndarray,
    neighbor_distances: np.ndarray,
    rng: np.random.Generator,
    args: BerryParameters,
) -> SphereCandidate:
    """围绕一个投票峰执行带中心先验的 RANSAC 球拟合。"""
    _, nearest = tree.query(peak_center, k=1)
    seed_index = int(nearest)
    search_radius = args.radius_max + args.peak_fit_max_shift + args.inlier_threshold
    patch_indices = np.asarray(tree.query_ball_point(peak_center, search_radius), dtype=np.int64)
    if len(patch_indices) < args.min_inliers:
        return rejected_candidate(candidate_id, seed_index, "peak_patch_too_small")

    patch_points = points[patch_indices]
    delta = patch_points - peak_center
    center_distances = np.linalg.norm(delta, axis=1)
    radial = delta / np.maximum(center_distances[:, None], 1e-12)
    normal_alignment = np.abs(np.einsum("ij,ij->i", normals[patch_indices], radial))
    usable = (
        (center_distances >= args.radius_min - args.peak_fit_max_shift)
        & (center_distances <= args.radius_max + args.peak_fit_max_shift)
        & (normal_alignment >= math.cos(math.radians(args.peak_prefilter_normal_angle)))
    )
    patch_indices = patch_indices[usable]
    patch_points = points[patch_indices]
    if len(patch_indices) < args.min_inliers:
        return rejected_candidate(candidate_id, seed_index, "few_prefilter_points")

    best_center: Optional[np.ndarray] = None
    best_radius = math.nan
    best_local_inliers = np.empty(0, dtype=np.int64)
    best_residual_sum = math.inf
    for _ in range(args.ransac_iterations):
        sphere = sphere_from_four(patch_points[rng.choice(len(patch_points), 4, replace=False)])
        if sphere is None:
            continue
        center, radius = sphere
        if not args.radius_min <= radius <= args.radius_max:
            continue
        if np.linalg.norm(center - peak_center) > args.peak_fit_max_shift:
            continue
        residuals = np.abs(np.linalg.norm(patch_points - center, axis=1) - radius)
        local_inliers = np.where(residuals <= args.inlier_threshold)[0]
        residual_sum = float(np.sum(residuals[local_inliers])) if len(local_inliers) else math.inf
        if len(local_inliers) > len(best_local_inliers) or (
            len(local_inliers) == len(best_local_inliers) and residual_sum < best_residual_sum
        ):
            best_center = center
            best_radius = float(radius)
            best_local_inliers = local_inliers
            best_residual_sum = residual_sum

    if best_center is None or len(best_local_inliers) < args.min_inliers:
        return rejected_candidate(candidate_id, seed_index, "ransac_failed")

    initial_global = largest_inlier_component(
        patch_indices[best_local_inliers],
        len(points),
        neighbor_indices,
        neighbor_distances,
        args.connectivity_edge_distance,
    )
    if len(initial_global) < args.min_inliers:
        return rejected_candidate(candidate_id, seed_index, "disconnected_support")

    lower = np.r_[peak_center - args.peak_fit_max_shift, args.radius_min]
    upper = np.r_[peak_center + args.peak_fit_max_shift, args.radius_max]

    def residual_function(parameters: np.ndarray, indices: np.ndarray) -> np.ndarray:
        return np.linalg.norm(points[indices] - parameters[:3], axis=1) - parameters[3]

    parameters = np.r_[best_center, best_radius]
    try:
        result = least_squares(
            lambda value: residual_function(value, initial_global),
            parameters,
            bounds=(lower, upper),
            loss="soft_l1",
            f_scale=args.inlier_threshold,
            max_nfev=160,
        )
        parameters = result.x
    except (ValueError, np.linalg.LinAlgError):
        pass

    center = parameters[:3]
    radius = float(parameters[3])
    residuals = np.abs(np.linalg.norm(patch_points - center, axis=1) - radius)
    global_inliers = largest_inlier_component(
        patch_indices[residuals <= args.inlier_threshold],
        len(points),
        neighbor_indices,
        neighbor_distances,
        args.connectivity_edge_distance,
    )
    if len(global_inliers) >= 4:
        try:
            result = least_squares(
                lambda value: residual_function(value, global_inliers),
                parameters,
                bounds=(lower, upper),
                loss="soft_l1",
                f_scale=args.inlier_threshold,
                max_nfev=160,
            )
            center = result.x[:3]
            radius = float(result.x[3])
        except (ValueError, np.linalg.LinAlgError):
            pass

    all_residuals = np.abs(np.linalg.norm(points[global_inliers] - center, axis=1) - radius)
    global_inliers = global_inliers[all_residuals <= args.inlier_threshold]
    inlier_count = len(global_inliers)
    inlier_ratio = inlier_count / max(len(patch_indices), 1)
    rms = (
        float(np.sqrt(np.mean((np.linalg.norm(points[global_inliers] - center, axis=1) - radius) ** 2)))
        if inlier_count else math.inf
    )
    if inlier_count:
        directions = points[global_inliers] - center
        directions /= np.maximum(np.linalg.norm(directions, axis=1, keepdims=True), 1e-12)
        coverage = coverage_angle_degrees(directions)
        normal_dot = np.abs(np.einsum("ij,ij->i", normals[global_inliers], directions))
        normal_agreement = float(
            np.mean(normal_dot >= math.cos(math.radians(args.normal_angle)))
        )
    else:
        coverage = 0.0
        normal_agreement = 0.0

    score = (
        inlier_count
        * (0.5 + inlier_ratio)
        * (0.5 + normal_agreement)
        * max(coverage / 180.0, 0.05)
        / (1.0 + rms / max(args.inlier_threshold, 1e-12))
    )
    candidate = SphereCandidate(
        candidate_id=candidate_id,
        seed_index=seed_index,
        patch_size=len(patch_indices),
        center=center,
        radius=radius,
        inlier_indices=np.unique(global_inliers),
        inlier_count=inlier_count,
        inlier_ratio=inlier_ratio,
        rms=rms,
        coverage_angle=coverage,
        normal_agreement=normal_agreement,
        score=score,
        status="accepted",
        reason="initial_vote_peak",
    )

    reasons: List[str] = []
    if inlier_count < args.min_inliers:
        reasons.append("few_inliers")
    if inlier_ratio < args.min_inlier_ratio:
        reasons.append("low_inlier_ratio")
    if coverage < args.min_coverage_angle:
        reasons.append("low_coverage")
    if normal_agreement < args.min_normal_agreement:
        reasons.append("normal_mismatch")
    if not args.radius_min <= radius <= args.radius_max:
        reasons.append("radius_out_of_range")
    if np.linalg.norm(center - peak_center) > args.peak_fit_max_shift + 1e-9:
        reasons.append("center_shift_too_large")
    if reasons:
        candidate.status = "rejected"
        candidate.reason = "+".join(reasons)
    return candidate


def fit_vote_peaks(
    accumulator: VoteAccumulator,
    points: np.ndarray,
    normals: np.ndarray,
    tree: cKDTree,
    neighbor_indices: np.ndarray,
    neighbor_distances: np.ndarray,
    rng: np.random.Generator,
    args: BerryParameters,
) -> Tuple[List[SphereCandidate], List[SphereCandidate]]:
    fitted: List[SphereCandidate] = []
    accepted: List[SphereCandidate] = []
    total = len(accumulator.peak_centers)
    for peak_id, center in enumerate(accumulator.peak_centers, start=1):
        candidate = fit_vote_peak_sphere(
            peak_id,
            center,
            points,
            normals,
            tree,
            neighbor_indices,
            neighbor_distances,
            rng,
            args,
        )
        fitted.append(candidate)
        if candidate.status == "accepted":
            accepted.append(candidate)
        if peak_id % 25 == 0 or peak_id == total:
            print(f"  已拟合 {peak_id}/{total} 个投票峰，通过 {len(accepted)}")
    return fitted, accepted


def candidates_are_duplicate(
    first: SphereCandidate,
    second: SphereCandidate,
    center_distance: float,
    radius_fraction: float,
    overlap_threshold: float,
    min_separation_fraction: float = 0.0,
    second_support_mask: Optional[np.ndarray] = None,
) -> bool:
    """优先按共享球面内点判重；可用第二候选的布尔掩码加速交集计数。"""
    distance = float(np.linalg.norm(first.center - second.center))
    radius_sum = max(first.radius + second.radius, 1e-12)
    mean_radius = radius_sum * 0.5
    radius_difference = abs(first.radius - second.radius) / mean_radius

    first_indices = np.asarray(first.inlier_indices, dtype=np.int64)
    second_indices = np.asarray(second.inlier_indices, dtype=np.int64)
    smaller_support = min(len(first_indices), len(second_indices))
    if smaller_support:
        if second_support_mask is not None:
            shared_count = int(np.count_nonzero(second_support_mask[first_indices]))
        else:
            shared_count = np.intersect1d(
                first_indices,
                second_indices,
                assume_unique=True,
            ).size
        shared_support = float(shared_count / smaller_support)
    else:
        shared_support = 0.0

    if shared_support >= 0.52:
        return True
    if (
        shared_support >= 0.30
        and radius_difference <= max(radius_fraction, 0.38)
        and distance <= 0.90 * radius_sum
    ):
        return True

    strict_center_limit = min(center_distance * 0.55, 0.25 * radius_sum)
    if (
        distance <= strict_center_limit
        and radius_difference <= min(radius_fraction, 0.28)
    ):
        return True

    overlap = sphere_intersection_fraction(
        first.center,
        first.radius,
        second.center,
        second.radius,
    )
    return (
        shared_support >= 0.22
        and overlap >= max(overlap_threshold, 0.72)
        and radius_difference <= max(radius_fraction, 0.35)
    )


def consolidate_candidates(
    candidates: Sequence[SphereCandidate],
    center_distance: float,
    radius_fraction: float,
    overlap_threshold: float,
    min_separation_fraction: float,
    point_count: int,
) -> Tuple[List[SphereCandidate], List[SphereCandidate]]:
    """按评分保留候选；代表候选使用布尔支持掩码加速共享内点判重。"""
    kept: List[SphereCandidate] = []
    kept_support_masks: List[np.ndarray] = []
    duplicates: List[SphereCandidate] = []
    duplicate_representatives: Dict[int, SphereCandidate] = {}
    for candidate in sorted(candidates, key=lambda item: item.score, reverse=True):
        matched: Optional[SphereCandidate] = None
        for existing, existing_mask in zip(kept, kept_support_masks):
            if candidates_are_duplicate(
                candidate,
                existing,
                center_distance,
                radius_fraction,
                overlap_threshold,
                min_separation_fraction,
                second_support_mask=existing_mask,
            ):
                matched = existing
                break
        if matched is None:
            kept.append(candidate)
            support_mask = np.zeros(point_count, dtype=bool)
            support_mask[candidate.inlier_indices] = True
            kept_support_masks.append(support_mask)
        else:
            duplicate_representatives[id(candidate)] = matched
            candidate.status = "duplicate"
            duplicates.append(candidate)

    kept.sort(key=lambda item: tuple(item.center))
    for index, candidate in enumerate(kept, start=1):
        candidate.candidate_id = index
        candidate.status = "initial_unique"
        candidate.reason = f"vote_candidate_{index}"
    for candidate in duplicates:
        representative = duplicate_representatives[id(candidate)]
        candidate.reason = f"duplicate_of_candidate_{representative.candidate_id}"
    return kept, duplicates


def sphere_point_metrics(
    points: np.ndarray,
    normals: np.ndarray,
    centers: np.ndarray,
    radii: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """返回点对多个球的球面残差和法向一致性。"""
    delta = points[:, None, :] - centers[None, :, :]
    distances = np.linalg.norm(delta, axis=2)
    residuals = np.abs(distances - radii[None, :])
    directions = delta / np.maximum(distances[:, :, None], 1e-12)
    alignments = np.abs(np.einsum("nkj,nj->nk", directions, normals))
    return residuals, alignments


def build_high_confidence_cores(
    points: np.ndarray,
    normals: np.ndarray,
    candidates: Sequence[SphereCandidate],
    residual_threshold: float,
    normal_angle: float,
    ambiguity_margin: float,
    cost_margin: float,
    allowed_labels: Optional[np.ndarray] = None,
    chunk_size: int = 4000,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """由候选球生成互斥的多点核心；模糊点保持标签 0。"""
    labels = np.zeros(len(points), dtype=np.int32)
    best_residuals = np.full(len(points), np.inf, dtype=np.float64)
    ambiguous = np.zeros(len(points), dtype=bool)
    if not candidates:
        return labels, best_residuals, ambiguous
    centers = np.vstack([item.center for item in candidates])
    radii = np.asarray([item.radius for item in candidates], dtype=np.float64)
    cosine = math.cos(math.radians(normal_angle))
    normal_scale = max(1.0 - cosine, 1e-6)

    for start in range(0, len(points), chunk_size):
        end = min(start + chunk_size, len(points))
        residuals, alignments = sphere_point_metrics(
            points[start:end], normals[start:end], centers, radii
        )
        valid = (residuals <= residual_threshold) & (alignments >= cosine)
        if allowed_labels is not None:
            allowed = allowed_labels[start:end]
            valid &= (allowed[:, None] == 0) | (
                np.arange(1, len(candidates) + 1, dtype=np.int32)[None, :] == allowed[:, None]
            )
        costs = residuals / max(residual_threshold, 1e-12)
        costs += 0.25 * (1.0 - alignments) / normal_scale
        costs[~valid] = np.inf
        order = np.argsort(costs, axis=1)[:, : min(2, len(candidates))]
        row = np.arange(end - start)
        best_slot = order[:, 0]
        best_cost = costs[row, best_slot]
        best_residual = residuals[row, best_slot]
        if len(candidates) > 1:
            second_slot = order[:, 1]
            second_cost = costs[row, second_slot]
            second_residual = residuals[row, second_slot]
        else:
            second_cost = np.full(end - start, np.inf)
            second_residual = np.full(end - start, np.inf)
        finite_best = np.isfinite(best_cost)
        finite_second = np.isfinite(second_cost)
        separated = np.ones(end - start, dtype=bool)
        both = finite_best & finite_second
        separated[both] = (
            ((second_residual[both] - best_residual[both]) >= ambiguity_margin)
            & ((second_cost[both] - best_cost[both]) >= cost_margin)
        )
        clear = finite_best & separated
        local_labels = np.zeros(end - start, dtype=np.int32)
        local_labels[clear] = best_slot[clear] + 1
        labels[start:end] = local_labels
        best_residuals[start:end] = best_residual
        ambiguous[start:end] = np.isfinite(best_cost) & np.isfinite(second_cost) & ~clear
    return labels, best_residuals, ambiguous


def filter_candidates_by_core_support(
    labels: np.ndarray,
    candidates: Sequence[SphereCandidate],
    minimum_points: int,
) -> Tuple[np.ndarray, List[SphereCandidate], List[SphereCandidate]]:
    """删除没有足够独立核心点的候选，并压缩标签。"""
    if not candidates:
        return labels, [], []
    counts = np.bincount(labels, minlength=len(candidates) + 1)
    kept: List[SphereCandidate] = []
    rejected: List[SphereCandidate] = []
    mapping = np.zeros(len(candidates) + 1, dtype=np.int32)
    for old_label, candidate in enumerate(candidates, start=1):
        if counts[old_label] >= minimum_points:
            mapping[old_label] = len(kept) + 1
            kept.append(candidate)
        else:
            candidate.status = "rejected"
            candidate.reason = f"few_unique_core_points_{int(counts[old_label])}"
            rejected.append(candidate)
    new_labels = mapping[labels]
    for new_label, candidate in enumerate(kept, start=1):
        candidate.candidate_id = new_label
    return new_labels, kept, rejected


def update_two_best_paths(
    point_index: int,
    label: int,
    cost: float,
    best_label: np.ndarray,
    best_cost: np.ndarray,
    second_label: np.ndarray,
    second_cost: np.ndarray,
) -> bool:
    """为一个点维护来自不同果粒标签的最优和次优传播路径。"""
    epsilon = 1e-12
    if label == best_label[point_index]:
        if cost + epsilon < best_cost[point_index]:
            best_cost[point_index] = cost
            return True
        return False
    if label == second_label[point_index]:
        if cost + epsilon < second_cost[point_index]:
            second_cost[point_index] = cost
            if second_cost[point_index] + epsilon < best_cost[point_index]:
                best_cost[point_index], second_cost[point_index] = (
                    second_cost[point_index], best_cost[point_index]
                )
                best_label[point_index], second_label[point_index] = (
                    second_label[point_index], best_label[point_index]
                )
            return True
        return False
    if cost + epsilon < best_cost[point_index]:
        second_cost[point_index] = best_cost[point_index]
        second_label[point_index] = best_label[point_index]
        best_cost[point_index] = cost
        best_label[point_index] = label
        return True
    if cost + epsilon < second_cost[point_index]:
        second_cost[point_index] = cost
        second_label[point_index] = label
        return True
    return False


def competitive_sphere_growing(
    points: np.ndarray,
    normals: np.ndarray,
    curvature: np.ndarray,
    neighbor_indices: np.ndarray,
    neighbor_distances: np.ndarray,
    local_spacing: np.ndarray,
    core_labels: np.ndarray,
    candidates: Sequence[SphereCandidate],
    args: BerryParameters,
) -> CompetitiveGrowthResult:
    """
    以每个候选球的全部核心点为多源起点执行竞争式区域生长。
    """
    count = len(points)
    best_label = np.zeros(count, dtype=np.int32)
    second_label = np.zeros(count, dtype=np.int32)
    best_cost = np.full(count, np.inf, dtype=np.float64)
    second_cost = np.full(count, np.inf, dtype=np.float64)
    queue: List[Tuple[float, int, int]] = []
    core_indices = np.where(core_labels > 0)[0]
    best_label[core_indices] = core_labels[core_indices]
    best_cost[core_indices] = 0.0
    for point_index in core_indices:
        heapq.heappush(queue, (0.0, int(core_labels[point_index]), int(point_index)))

    centers = np.vstack([item.center for item in candidates])
    radii = np.asarray([item.radius for item in candidates], dtype=np.float64)
    smooth_cosine = math.cos(math.radians(args.grow_angle))
    sphere_cosine = math.cos(math.radians(args.grow_normal_angle))
    smooth_scale = max(1.0 - smooth_cosine, 1e-6)
    growth_neighbor_count = min(
        args.growth_knn,
        neighbor_indices.shape[1] - 1,
    )

    while queue:
        path_cost, label, current = heapq.heappop(queue)
        valid_best = label == best_label[current] and path_cost <= best_cost[current] + 1e-12
        valid_second = label == second_label[current] and path_cost <= second_cost[current] + 1e-12
        if not (valid_best or valid_second) or path_cost > args.max_path_cost:
            continue
        center = centers[label - 1]
        radius = radii[label - 1]
        current_normal = normals[current]
        for slot in range(1, growth_neighbor_count + 1):
            neighbor = int(neighbor_indices[current, slot])
            edge_distance = float(neighbor_distances[current, slot])
            edge_limit = max(
                args.grow_edge_distance,
                args.adaptive_edge_factor * max(local_spacing[current], local_spacing[neighbor]),
            )
            if edge_distance > edge_limit:
                continue
            if curvature[neighbor] > args.grow_hard_curvature:
                continue
            surface_normal_dot = abs(float(np.dot(current_normal, normals[neighbor])))
            if surface_normal_dot < smooth_cosine:
                continue
            delta = points[neighbor] - center
            center_distance = float(np.linalg.norm(delta))
            sphere_residual = abs(center_distance - radius)
            if sphere_residual > args.grow_sphere_residual:
                continue
            direction = delta / max(center_distance, 1e-12)
            sphere_normal_dot = abs(float(np.dot(normals[neighbor], direction)))
            if sphere_normal_dot < sphere_cosine:
                continue
            edge_cost = (
                0.20 * edge_distance / max(edge_limit, 1e-12)
                + 0.20 * (1.0 - surface_normal_dot) / smooth_scale
                + 0.45 * sphere_residual / max(args.grow_sphere_residual, 1e-12)
                + 0.15 * min(curvature[neighbor] / max(args.grow_soft_curvature, 1e-12), 2.0)
            )
            new_cost = path_cost + max(edge_cost, 1e-6)
            if new_cost > args.max_path_cost:
                continue
            if update_two_best_paths(
                neighbor,
                label,
                new_cost,
                best_label,
                best_cost,
                second_label,
                second_cost,
            ):
                heapq.heappush(queue, (new_cost, label, neighbor))

    raw_labels = best_label.copy()
    labels = raw_labels.copy()
    finite_paths = (
        (best_label > 0)
        & (second_label > 0)
        & np.isfinite(best_cost)
        & np.isfinite(second_cost)
    )
    cost_gap = np.full(count, np.inf, dtype=np.float64)
    cost_gap[finite_paths] = second_cost[finite_paths] - best_cost[finite_paths]
    ambiguity_mask = np.zeros(count, dtype=bool)
    indices = np.where(finite_paths & (cost_gap <= args.competition_margin))[0]
    if len(indices):
        best_centers = centers[best_label[indices] - 1]
        second_centers = centers[second_label[indices] - 1]
        best_radii = radii[best_label[indices] - 1]
        second_radii = radii[second_label[indices] - 1]
        best_residual = np.abs(np.linalg.norm(points[indices] - best_centers, axis=1) - best_radii)
        second_residual = np.abs(
            np.linalg.norm(points[indices] - second_centers, axis=1) - second_radii
        )
        truly_ambiguous = (
            (best_residual <= args.boundary_sphere_residual)
            & (second_residual <= args.boundary_sphere_residual)
            & (np.abs(second_residual - best_residual) <= args.boundary_residual_margin)
        )
        ambiguity_mask[indices[truly_ambiguous]] = True
    # 高置信度核心永不因边界竞争而清零。
    ambiguity_mask[core_indices] = False
    labels[ambiguity_mask] = 0
    unreachable_mask = raw_labels == 0
    return CompetitiveGrowthResult(
        raw_labels=raw_labels,
        labels=labels,
        best_cost=best_cost,
        second_cost=second_cost,
        second_labels=second_label,
        ambiguity_mask=ambiguity_mask,
        unreachable_mask=unreachable_mask,
    )


def pending_assignment_round(
    round_id: int,
    points: np.ndarray,
    normals: np.ndarray,
    labels: np.ndarray,
    candidates: Sequence[SphereCandidate],
    args: BerryParameters,
) -> Tuple[np.ndarray, AssignmentStats]:
    """用 KNN 多数票、球面残差和法向一致性补分一轮标签 0 点。"""
    new_labels = labels.copy()
    pending_indices = np.where(labels == 0)[0]
    labeled_indices = np.where(labels > 0)[0]
    counters = {"no_neighbor": 0, "vote": 0, "sphere": 0, "normal": 0, "ambiguous": 0}
    if not len(pending_indices) or not len(labeled_indices) or not candidates:
        stats = AssignmentStats(
            round_id, len(pending_indices), 0, len(pending_indices),
            len(pending_indices), 0, 0, 0, 0,
        )
        return new_labels, stats

    tree = cKDTree(points[labeled_indices])
    query_k = min(max(args.assignment_knn, 1), len(labeled_indices))
    distances, local_neighbors = tree.query(points[pending_indices], k=query_k, workers=-1)
    if query_k == 1:
        distances = distances[:, None]
        local_neighbors = local_neighbors[:, None]
    neighbor_point_indices = labeled_indices[local_neighbors]
    neighbor_labels = labels[neighbor_point_indices]
    centers = np.vstack([item.center for item in candidates])
    radii = np.asarray([item.radius for item in candidates])
    normal_cosine = math.cos(math.radians(args.assignment_normal_angle))
    assigned = 0

    for row, point_index in enumerate(pending_indices):
        valid = distances[row] <= args.assignment_distance
        if not np.any(valid):
            counters["no_neighbor"] += 1
            continue
        local_labels = neighbor_labels[row, valid]
        unique, counts = np.unique(local_labels[local_labels > 0], return_counts=True)
        if not len(unique):
            counters["no_neighbor"] += 1
            continue
        order = np.argsort(counts)[::-1]
        unique = unique[order]
        counts = counts[order]
        vote_fraction = counts / max(np.sum(counts), 1)
        if vote_fraction[0] < args.min_vote_fraction:
            counters["vote"] += 1
            continue

        candidate_indices = unique.astype(np.int64) - 1
        delta = points[point_index][None, :] - centers[candidate_indices]
        center_distances = np.linalg.norm(delta, axis=1)
        residuals = np.abs(center_distances - radii[candidate_indices])
        sphere_order = np.argsort(residuals)
        best_slot = int(sphere_order[0])
        best_candidate = int(candidate_indices[best_slot])
        best_residual = float(residuals[best_slot])
        if best_residual > args.assignment_residual:
            counters["sphere"] += 1
            continue
        direction = delta[best_slot] / max(center_distances[best_slot], 1e-12)
        if abs(float(np.dot(normals[point_index], direction))) < normal_cosine:
            counters["normal"] += 1
            continue
        if len(sphere_order) > 1:
            second_residual = float(residuals[int(sphere_order[1])])
            if second_residual - best_residual < args.assignment_ambiguity_margin:
                counters["ambiguous"] += 1
                continue
        new_labels[point_index] = best_candidate + 1
        assigned += 1

    stats = AssignmentStats(
        round_id=round_id,
        pending_before=len(pending_indices),
        newly_assigned=assigned,
        pending_after=len(pending_indices) - assigned,
        rejected_no_neighbor=counters["no_neighbor"],
        rejected_vote=counters["vote"],
        rejected_sphere=counters["sphere"],
        rejected_normal=counters["normal"],
        rejected_ambiguous=counters["ambiguous"],
    )
    return new_labels, stats


def refit_candidates_from_labels(
    labels: np.ndarray,
    candidates: Sequence[SphereCandidate],
    points: np.ndarray,
    normals: np.ndarray,
    args: BerryParameters,
) -> List[SphereCandidate]:
    """使用当前互斥标签点稳健复拟合所有候选球。"""
    refined: List[SphereCandidate] = []
    cosine = math.cos(math.radians(args.assignment_normal_angle))
    for label, candidate in enumerate(candidates, start=1):
        indices = np.where(labels == label)[0]
        if len(indices) < 4:
            candidate.status = "rejected"
            candidate.reason = "too_few_labeled_points"
            refined.append(candidate)
            continue
        label_points = points[indices]
        delta0 = label_points - candidate.center
        distances0 = np.linalg.norm(delta0, axis=1)
        residual0 = np.abs(distances0 - candidate.radius)
        directions0 = delta0 / np.maximum(distances0[:, None], 1e-12)
        normal0 = np.abs(np.einsum("ij,ij->i", normals[indices], directions0))
        support = (
            (residual0 <= args.refit_support_threshold * 1.5)
            & (normal0 >= cosine)
        )
        if np.count_nonzero(support) < 4:
            candidate.status = "rejected"
            candidate.reason = "too_few_refit_support_points"
            refined.append(candidate)
            continue

        initial = np.r_[candidate.center, candidate.radius]
        lower = np.r_[candidate.center - args.max_center_shift, args.radius_min]
        upper = np.r_[candidate.center + args.max_center_shift, args.radius_max]

        def residual_function(parameters: np.ndarray) -> np.ndarray:
            return np.linalg.norm(label_points[support] - parameters[:3], axis=1) - parameters[3]

        try:
            result = least_squares(
                residual_function,
                initial,
                bounds=(lower, upper),
                loss="soft_l1",
                f_scale=args.refit_support_threshold,
                max_nfev=180,
            )
            center = result.x[:3]
            radius = float(result.x[3])
        except (ValueError, np.linalg.LinAlgError):
            center = candidate.center.copy()
            radius = float(candidate.radius)

        delta = label_points - center
        distances = np.linalg.norm(delta, axis=1)
        residuals = np.abs(distances - radius)
        local_inliers = np.where(residuals <= args.refit_support_threshold)[0]
        global_inliers = np.unique(indices[local_inliers])
        inlier_count = len(global_inliers)
        inlier_ratio = inlier_count / max(len(indices), 1)
        rms = (
            float(np.sqrt(np.mean(residuals[local_inliers] ** 2)))
            if inlier_count else math.inf
        )
        if inlier_count:
            directions = delta[local_inliers]
            directions /= np.maximum(np.linalg.norm(directions, axis=1, keepdims=True), 1e-12)
            coverage = coverage_angle_degrees(directions)
            normal_dot = np.abs(np.einsum("ij,ij->i", normals[global_inliers], directions))
            normal_agreement = float(
                np.mean(normal_dot >= math.cos(math.radians(args.normal_angle)))
            )
        else:
            coverage = 0.0
            normal_agreement = 0.0
        score = (
            inlier_count
            * (0.5 + inlier_ratio)
            * (0.5 + normal_agreement)
            * max(coverage / 180.0, 0.05)
            / (1.0 + rms / max(args.refit_support_threshold, 1e-12))
        )
        refined.append(SphereCandidate(
            candidate_id=label,
            seed_index=candidate.seed_index,
            patch_size=len(indices),
            center=center,
            radius=radius,
            inlier_indices=global_inliers,
            inlier_count=inlier_count,
            inlier_ratio=inlier_ratio,
            rms=rms,
            coverage_angle=coverage,
            normal_agreement=normal_agreement,
            score=score,
            status="refined",
            reason=f"refined_label_{label}",
        ))
    return refined


def prune_and_merge_labeled_candidates(
    labels: np.ndarray,
    candidates: Sequence[SphereCandidate],
    args: BerryParameters,
) -> Tuple[np.ndarray, List[SphereCandidate], List[SphereCandidate]]:
    """按独立支持和几何质量淘汰候选，并将收敛到同一球的标签合并。"""
    counts = np.bincount(labels, minlength=len(candidates) + 1)
    eligible: List[Tuple[int, SphereCandidate]] = []
    removed: List[SphereCandidate] = []
    for old_label, candidate in enumerate(candidates, start=1):
        reasons = []
        if counts[old_label] < args.final_min_label_points:
            reasons.append("few_label_points")
        if candidate.inlier_count < args.final_min_inliers:
            reasons.append("few_refined_inliers")
        if candidate.coverage_angle < args.final_min_coverage_angle:
            reasons.append("low_refined_coverage")
        if candidate.normal_agreement < args.final_min_normal_agreement:
            reasons.append("low_refined_normal_agreement")
        if not args.radius_min <= candidate.radius <= args.radius_max:
            reasons.append("refined_radius_out_of_range")
        if reasons:
            candidate.status = "rejected"
            candidate.reason = "+".join(reasons)
            removed.append(candidate)
        else:
            eligible.append((old_label, candidate))

    representatives: List[Tuple[int, SphereCandidate]] = []
    representative_support_masks: List[np.ndarray] = []
    old_to_representative: Dict[int, SphereCandidate] = {}
    for old_label, candidate in sorted(eligible, key=lambda pair: pair[1].score, reverse=True):
        duplicate_of = None
        for (_, existing), existing_mask in zip(
            representatives,
            representative_support_masks,
        ):
            if candidates_are_duplicate(
                candidate,
                existing,
                args.final_merge_distance,
                args.final_merge_radius_fraction,
                args.final_overlap_threshold,
                args.final_min_separation_fraction,
                second_support_mask=existing_mask,
            ):
                duplicate_of = existing
                break
        if duplicate_of is None:
            representatives.append((old_label, candidate))
            support_mask = np.zeros(len(labels), dtype=bool)
            support_mask[candidate.inlier_indices] = True
            representative_support_masks.append(support_mask)
            old_to_representative[old_label] = candidate
        else:
            old_to_representative[old_label] = duplicate_of
            candidate.status = "duplicate"
            candidate.reason = f"merged_after_refit_with_{duplicate_of.candidate_id}"
            removed.append(candidate)

    representatives.sort(key=lambda pair: tuple(pair[1].center))
    new_candidates = [candidate for _, candidate in representatives]
    new_id_by_object = {id(candidate): new_id for new_id, candidate in enumerate(new_candidates, start=1)}
    mapping = np.zeros(len(candidates) + 1, dtype=np.int32)
    for old_label, representative in old_to_representative.items():
        mapping[old_label] = new_id_by_object[id(representative)]
    new_labels = mapping[labels]
    for new_id, candidate in enumerate(new_candidates, start=1):
        candidate.candidate_id = new_id
        candidate.status = "kept_after_refit"
        candidate.reason = f"berry_{new_id}"
    return new_labels, new_candidates, removed


def validate_arguments(args: BerryParameters) -> None:
    if not 0 < args.radius_min < args.radius_max:
        raise ValueError("必须满足 0 < radius-min < radius-max")
    if args.vote_radius_steps < 2:
        raise ValueError("vote-radius-steps 至少为 2")
    if args.knn < 6:
        raise ValueError("knn 至少为 6")
    if not 1 <= args.growth_knn < args.knn:
        raise ValueError("growth-knn 必须至少为 1，且小于 knn")
    for name in (
        "vote_curvature_quantile", "min_inlier_ratio", "min_normal_agreement",
        "candidate_merge_radius_fraction", "candidate_overlap_threshold",
        "candidate_min_separation_fraction",
        "min_vote_fraction", "final_min_normal_agreement",
        "final_merge_radius_fraction", "final_overlap_threshold",
        "final_min_separation_fraction",
    ):
        value = getattr(args, name)
        if not 0 <= value <= 1:
            raise ValueError(f"{name} 必须位于 0～1")
    if args.assignment_rounds < 0 or args.final_assignment_rounds < 0:
        raise ValueError("补分轮数不能小于 0")


def count_berries(
    points: np.ndarray,
    args: BerryParameters,
) -> Tuple[int, List[SphereCandidate], np.ndarray, dict]:
    """Compute berry count, fitted models, labels and stage diagnostics from XYZ."""
    started = time.perf_counter()
    rng = np.random.default_rng(args.random_seed)
    radii_grid = np.linspace(args.radius_min, args.radius_max, args.vote_radius_steps)

    if len(points) < args.knn:
        raise ValueError(f"点云点数不足，至少需要 {args.knn} 点")

    maximum_extent = float(np.max(np.ptp(points, axis=0)))
    if maximum_extent > 1.0:
        raise ValueError(
            f"点云跨度为 {maximum_extent:.3f}，疑似仍为毫米/厘米尺度原始点云；"
            "Supply a preprocessed grape point cloud with coordinates in metres."
        )

    tree = cKDTree(points)
    normals, curvature, neighbor_distances, neighbor_indices = (
        estimate_normals_and_curvature(points, tree, args.knn)
    )
    spacing_slots = min(6, neighbor_distances.shape[1])
    local_spacing = np.median(
        neighbor_distances[:, 1:spacing_slots],
        axis=1,
    )
    coherence = normal_coherence(normals, neighbor_indices)

    vote_state = multiscale_vote_accumulator(
        points,
        normals,
        curvature,
        coherence,
        radii_grid,
        args.vote_voxel_size,
        args.vote_curvature_quantile,
        args.vote_min_weight,
        args.vote_min_count,
        args.vote_nms_distance,
        args.max_vote_peaks,
        args.max_vote_samples,
        rng,
    )
    _, accepted = fit_vote_peaks(
        vote_state,
        points,
        normals,
        tree,
        neighbor_indices,
        neighbor_distances,
        rng,
        args,
    )
    initial_candidates, initial_duplicates = consolidate_candidates(
        accepted,
        args.candidate_merge_distance,
        args.candidate_merge_radius_fraction,
        args.candidate_overlap_threshold,
        args.candidate_min_separation_fraction,
        len(points),
    )
    if not initial_candidates:
        raise RuntimeError("没有通过几何验证的初始果粒候选")
    diagnostics = {
        "initial_vote_peaks": len(vote_state.peak_centers),
        "initial_fit_accepted": len(accepted),
        "initial_unique_candidates": len(initial_candidates),
        "initial_duplicates_removed": len(initial_duplicates),
    }

    initial_core_labels, _, _ = build_high_confidence_cores(
        points,
        normals,
        initial_candidates,
        args.core_residual,
        args.core_normal_angle,
        args.core_ambiguity_margin,
        args.core_cost_margin,
    )
    initial_core_labels, initial_candidates, _ = (
        filter_candidates_by_core_support(
            initial_core_labels,
            initial_candidates,
            args.min_core_points,
        )
    )
    if not initial_candidates:
        raise RuntimeError("所有候选均缺少独立核心支持")
    diagnostics["core_candidates"] = len(initial_candidates)

    first_growth = competitive_sphere_growing(
        points,
        normals,
        curvature,
        neighbor_indices,
        neighbor_distances,
        local_spacing,
        initial_core_labels,
        initial_candidates,
        args,
    )

    working_labels = first_growth.labels.copy()
    working_candidates = list(initial_candidates)
    for round_id in range(1, args.assignment_rounds + 1):
        working_labels, _ = pending_assignment_round(
            round_id,
            points,
            normals,
            working_labels,
            working_candidates,
            args,
        )

    if args.assignment_rounds > 0:
        working_candidates = refit_candidates_from_labels(
            working_labels,
            working_candidates,
            points,
            normals,
            args,
        )

    working_labels, working_candidates, _ = (
        prune_and_merge_labeled_candidates(
            working_labels,
            working_candidates,
            args,
        )
    )
    working_candidates = refit_candidates_from_labels(
        working_labels,
        working_candidates,
        points,
        normals,
        args,
    )
    if not working_candidates:
        raise RuntimeError("复拟合质控后没有保留果粒候选")
    diagnostics["post_growth_candidates"] = len(working_candidates)

    final_labels = working_labels.copy()
    final_candidates = list(working_candidates)

    for offset in range(1, args.final_assignment_rounds + 1):
        round_id = args.assignment_rounds + offset
        final_labels, _ = pending_assignment_round(
            round_id,
            points,
            normals,
            final_labels,
            final_candidates,
            args,
        )

    if args.final_assignment_rounds > 0:
        final_candidates = refit_candidates_from_labels(
            final_labels,
            final_candidates,
            points,
            normals,
            args,
        )

    final_labels, final_candidates, _ = prune_and_merge_labeled_candidates(
        final_labels,
        final_candidates,
        args,
    )
    # 最终质控之后只读取候选数量，不再执行不会参与后续决策的复拟合。
    diagnostics["final_candidates"] = len(final_candidates)
    diagnostics["runtime_seconds"] = time.perf_counter() - started
    diagnostics["labeled_points"] = int(np.count_nonzero(final_labels))
    diagnostics["unassigned_points"] = int(np.count_nonzero(final_labels == 0))
    return len(final_candidates), final_candidates, final_labels, diagnostics

def _signed_polygon_area(points_2d: np.ndarray) -> float:
    following = np.roll(points_2d, -1, axis=0)
    return 0.5 * float(
        np.sum(points_2d[:, 0] * following[:, 1] - following[:, 0] * points_2d[:, 1])
    )


def _resample_closed_polygon(points_2d: np.ndarray, count: int) -> np.ndarray:
    """按周长等弧长重采样闭合多边形，保留原有方向。"""
    if count < 3:
        raise ValueError("轮廓重采样点数必须至少为 3")
    following = np.roll(points_2d, -1, axis=0)
    lengths = np.linalg.norm(following - points_2d, axis=1)
    keep = lengths > np.finfo(float).eps
    points_2d = points_2d[keep]
    if len(points_2d) < 3:
        raise ValueError("凸包轮廓退化")
    following = np.roll(points_2d, -1, axis=0)
    lengths = np.linalg.norm(following - points_2d, axis=1)
    perimeter = float(np.sum(lengths))
    if not np.isfinite(perimeter) or perimeter <= 0:
        raise ValueError("凸包轮廓周长无效")
    cumulative = np.concatenate(([0.0], np.cumsum(lengths)))
    distances = np.arange(count, dtype=float) * perimeter / count
    segments = np.searchsorted(cumulative, distances, side="right") - 1
    segments = np.minimum(segments, len(points_2d) - 1)
    fractions = (distances - cumulative[segments]) / lengths[segments]
    return points_2d[segments] + (
        following[segments] - points_2d[segments]
    ) * fractions[:, np.newaxis]


def _cyclically_align(reference: np.ndarray, moving: np.ndarray) -> np.ndarray:
    """保持两个轮廓的环绕方向，寻找最小对应距离的循环起点。"""
    costs = np.asarray(
        [np.sum((reference - np.roll(moving, shift, axis=0)) ** 2) for shift in range(len(moving))]
    )
    return np.roll(moving, int(np.argmin(costs)), axis=0)


def contour_slice_convex_hull_surface_cm2(
    points: np.ndarray,
    slice_thickness_m: float,
    contour_points: int = 64,
    axis_mode: str = "pca",
) -> tuple[float, dict]:
    
    points = np.asarray(points, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError("点云坐标必须是 N×3 数组")
    if len(points) < 4 or not np.all(np.isfinite(points)):
        raise ValueError("点云点数不足或包含非有限坐标")
    if slice_thickness_m <= 0:
        raise ValueError("切片厚度必须大于 0")
    if contour_points < 3:
        raise ValueError("每个轮廓的重采样点数必须至少为 3")

    centered = points - np.mean(points, axis=0)
    if axis_mode == "pca":
        covariance = np.cov(centered, rowvar=False)
        eigenvalues, eigenvectors = np.linalg.eigh(covariance)
        order = np.argsort(eigenvalues)[::-1]
        eigenvalues = eigenvalues[order]
        eigenvectors = eigenvectors[:, order]
        transformed = centered @ eigenvectors
        heights = transformed[:, 0]
        planar = transformed[:, 1:3]
        height_axis = eigenvectors[:, 0]
    elif axis_mode == "z":
        eigenvalues = np.full(3, np.nan, dtype=float)
        heights = centered[:, 2]
        planar = centered[:, :2]
        height_axis = np.asarray([0.0, 0.0, 1.0])
    else:
        raise ValueError("axis_mode 只能是 'pca' 或 'z'")

    minimum_height = float(np.min(heights))
    maximum_height = float(np.max(heights))
    cloud_height = maximum_height - minimum_height
    if cloud_height <= 0:
        raise ValueError("点云在切片方向上的高度为 0")
    slice_count = max(1, int(math.ceil(cloud_height / slice_thickness_m)))
    slice_indices = np.floor(
        (heights - minimum_height) / slice_thickness_m
    ).astype(np.int64)
    slice_indices = np.clip(slice_indices, 0, slice_count - 1)

    rings: list[dict] = []
    skipped_sparse = 0
    skipped_degenerate = 0
    point_counts: list[int] = []
    for slice_index in range(slice_count):
        mask = slice_indices == slice_index
        current = np.unique(planar[mask], axis=0)
        point_counts.append(int(np.count_nonzero(mask)))
        if len(current) < 3:
            skipped_sparse += 1
            continue
        if np.linalg.matrix_rank(current - current[0]) < 2:
            skipped_degenerate += 1
            continue
        try:
            hull = ConvexHull(current)
        except Exception:
            try:
                hull = ConvexHull(current, qhull_options="QJ")
            except Exception:
                skipped_degenerate += 1
                continue
        polygon = current[hull.vertices]
        if _signed_polygon_area(polygon) < 0:
            polygon = polygon[::-1]
        ring = _resample_closed_polygon(polygon, contour_points)
        plane_height = minimum_height + (slice_index + 0.5) * slice_thickness_m
        plane_height = min(plane_height, maximum_height)
        rings.append(
            {
                "index": slice_index,
                "height": plane_height,
                "points": ring,
                "cap_area_m2": float(hull.volume),
            }
        )

    if len(rings) < 2:
        raise ValueError("有效二维凸包切片少于 2，无法连接表面")

    side_area_m2 = 0.0
    gap_count = 0
    maximum_gap = 0
    for lower, upper in zip(rings, rings[1:]):
        gap = int(upper["index"] - lower["index"])
        if gap > 1:
            gap_count += 1
        maximum_gap = max(maximum_gap, gap)
        lower_2d = np.asarray(lower["points"])
        upper_2d = _cyclically_align(lower_2d, np.asarray(upper["points"]))
        lower_3d = np.column_stack(
            (np.full(contour_points, lower["height"]), lower_2d)
        )
        upper_3d = np.column_stack(
            (np.full(contour_points, upper["height"]), upper_2d)
        )
        lower_next = np.roll(lower_3d, -1, axis=0)
        upper_next = np.roll(upper_3d, -1, axis=0)
        first_triangles = 0.5 * np.linalg.norm(
            np.cross(upper_3d - lower_3d, lower_next - lower_3d), axis=1
        )
        second_triangles = 0.5 * np.linalg.norm(
            np.cross(upper_3d - lower_next, upper_next - lower_next), axis=1
        )
        side_area_m2 += float(np.sum(first_triangles) + np.sum(second_triangles))

    cap_area_m2 = float(rings[0]["cap_area_m2"] + rings[-1]["cap_area_m2"])
    total_area_cm2 = (side_area_m2 + cap_area_m2) * 1e4
    if not np.isfinite(total_area_cm2) or total_area_cm2 <= 0:
        raise ValueError("轮廓切片凸包表面积无效")
    return total_area_cm2, {
        "切片坐标系": "PCA第一主成分" if axis_mode == "pca" else "原始Z轴",
        "切片厚度(m)": float(slice_thickness_m),
        "每层轮廓重采样点数": int(contour_points),
        "切片方向X": float(height_axis[0]),
        "切片方向Y": float(height_axis[1]),
        "切片方向Z": float(height_axis[2]),
        "切片方向高度(m)": float(cloud_height),
        "理论切片数": int(slice_count),
        "有效二维凸包切片数": int(len(rings)),
        "侧面连接带数": int(len(rings) - 1),
        "含空缺层的连接数": int(gap_count),
        "最大切片序号间隔": int(maximum_gap),
        "稀疏跳过切片数": int(skipped_sparse),
        "退化跳过切片数": int(skipped_degenerate),
        "每层点数最小值": int(min(point_counts)),
        "每层点数最大值": int(max(point_counts)),
        "轮廓侧面积(cm²)": float(side_area_m2 * 1e4),
        "轮廓顶底端盖面积(cm²)": float(cap_area_m2 * 1e4),
        "PCA特征值1": float(eigenvalues[0]),
        "PCA特征值2": float(eigenvalues[1]),
        "PCA特征值3": float(eigenvalues[2]),
    }


def multiscale_contour_slice_surface_cm2(
    points: np.ndarray,
    slice_thicknesses_m: tuple[float, ...],
    contour_points: int = 64,
    axis_mode: str = "z",
) -> tuple[float, dict]:
    if not slice_thicknesses_m:
        raise ValueError("多尺度模式至少需要一个切片厚度")
    thicknesses = tuple(float(value) for value in slice_thicknesses_m)
    if any(not np.isfinite(value) or value <= 0 for value in thicknesses):
        raise ValueError("多尺度切片厚度必须全部是有限正数")
    if len(set(thicknesses)) != len(thicknesses):
        raise ValueError("多尺度切片厚度不能重复")

    areas: list[float] = []
    scale_records: list[dict] = []
    for thickness in thicknesses:
        area, record = contour_slice_convex_hull_surface_cm2(
            points,
            slice_thickness_m=thickness,
            contour_points=contour_points,
            axis_mode=axis_mode,
        )
        areas.append(float(area))
        scale_records.append(record)

    combined = float(np.mean(areas))
    # 用较厚尺度的切片质量记录作为主记录，同时保留每个尺度的面积和有效层数。
    representative_index = int(np.argmax(thicknesses))
    combined_record = dict(scale_records[representative_index])
    combined_record.update(
        {
            "轮廓计算模式": "多尺度算术平均",
            "多尺度数量": int(len(thicknesses)),
            "多尺度切片厚度(m)": ",".join(f"{value:.8g}" for value in thicknesses),
            "多尺度表面积均值(cm²)": combined,
            "多尺度表面积标准差(cm²)": float(np.std(areas)),
            "多尺度表面积变异系数(%)": (
                float(np.std(areas) / combined * 100.0) if combined > 0 else math.nan
            ),
        }
    )
    for index, (thickness, area, record) in enumerate(
        zip(thicknesses, areas, scale_records), start=1
    ):
        prefix = f"尺度{index}"
        combined_record[f"{prefix}切片厚度(m)"] = thickness
        combined_record[f"{prefix}表面积(cm²)"] = area
        combined_record[f"{prefix}有效切片数"] = int(record["有效二维凸包切片数"])
        combined_record[f"{prefix}稀疏跳过切片数"] = int(record["稀疏跳过切片数"])
    return combined, combined_record


# ---------------------------------------------------------------------------
# Volume: fixed-thickness integration of two-dimensional convex-hull areas
# ---------------------------------------------------------------------------
def contour_slice_convex_hull_volume_cm3(
    points: np.ndarray,
    slice_thickness_m: float,
    axis_mode: str = "pca",
) -> tuple[float, dict]:
  
    points = np.asarray(points, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError("点云坐标必须是 N×3 数组")
    if len(points) < 4:
        raise ValueError("轮廓切片凸包计算所需点数不足")
    if not np.all(np.isfinite(points)):
        raise ValueError("点云包含 NaN 或无穷大坐标")
    if slice_thickness_m <= 0:
        raise ValueError("切片厚度必须大于 0")

    centered = points - np.mean(points, axis=0)
    if axis_mode == "pca":
        covariance = np.cov(centered, rowvar=False)
        eigenvalues, eigenvectors = np.linalg.eigh(covariance)
        order = np.argsort(eigenvalues)[::-1]
        eigenvalues = eigenvalues[order]
        eigenvectors = eigenvectors[:, order]
        transformed = centered @ eigenvectors
        height_coordinates = transformed[:, 0]
        planar_points = transformed[:, 1:3]
        height_axis = eigenvectors[:, 0]
    elif axis_mode == "z":
        eigenvalues = np.full(3, np.nan, dtype=np.float64)
        height_coordinates = centered[:, 2]
        planar_points = centered[:, :2]
        height_axis = np.array([0.0, 0.0, 1.0], dtype=np.float64)
    else:
        raise ValueError("axis_mode 只能是 'pca' 或 'z'")

    height_min = float(np.min(height_coordinates))
    height_max = float(np.max(height_coordinates))
    height = height_max - height_min
    if height <= 0:
        raise ValueError("点云在切片方向上的高度为 0")

    slice_count = max(1, int(math.ceil(height / slice_thickness_m)))
    # 将恰好位于最高边界的点归入最后一层，避免因右开区间漏掉最高点。
    slice_indices = np.floor(
        (height_coordinates - height_min) / slice_thickness_m
    ).astype(np.int64)
    slice_indices = np.clip(slice_indices, 0, slice_count - 1)

    slice_areas_m2: list[float] = []
    slice_point_counts: list[int] = []
    skipped_too_sparse = 0
    skipped_degenerate = 0
    for slice_index in range(slice_count):
        current = planar_points[slice_indices == slice_index]
        slice_point_counts.append(int(len(current)))
        if len(current) < 3:
            skipped_too_sparse += 1
            continue

        # 重复二维坐标不会增加轮廓，只会使 Qhull 更易出现退化错误。
        current = np.unique(current, axis=0)
        if len(current) < 3 or np.linalg.matrix_rank(current - current[0]) < 2:
            skipped_degenerate += 1
            continue
        try:
            hull = ConvexHull(current)
        except Exception:
            # QJ 仅在数值近退化时加入确定性的微小扰动，不改变计算模型。
            try:
                hull = ConvexHull(current, qhull_options="QJ")
            except Exception:
                skipped_degenerate += 1
                continue
        # SciPy 二维 ConvexHull.volume 表示面积，area 表示周长。
        slice_areas_m2.append(float(hull.volume))

    if not slice_areas_m2:
        raise ValueError("所有切片均点数不足或二维凸包退化")

    volume_m3 = float(np.sum(slice_areas_m2) * slice_thickness_m)
    if not np.isfinite(volume_m3) or volume_m3 <= 0:
        raise ValueError("轮廓切片凸包体积无效")

    return volume_m3 * 1e6, {
        "切片坐标系": "PCA第一主成分" if axis_mode == "pca" else "原始Z轴",
        "切片厚度(m)": float(slice_thickness_m),
        "切片方向X": float(height_axis[0]),
        "切片方向Y": float(height_axis[1]),
        "切片方向Z": float(height_axis[2]),
        "切片方向高度(m)": float(height),
        "理论切片数": int(slice_count),
        "有效二维凸包切片数": int(len(slice_areas_m2)),
        "稀疏跳过切片数": int(skipped_too_sparse),
        "退化跳过切片数": int(skipped_degenerate),
        "每层点数最小值": int(min(slice_point_counts)),
        "每层点数最大值": int(max(slice_point_counts)),
        "PCA特征值1": float(eigenvalues[0]),
        "PCA特征值2": float(eigenvalues[1]),
        "PCA特征值3": float(eigenvalues[2]),
    }


def pca_dimensions_cm(
    points: np.ndarray, trim_percent: Sequence[float]
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    trims = np.asarray(trim_percent, dtype=np.float64)
    if trims.shape != (3,) or not np.all(np.isfinite(trims)):
        raise ValueError("Three finite PCA trimming percentages are required.")
    if np.any(trims < 0) or np.any(trims >= 50):
        raise ValueError("Each PCA trimming percentage must be in [0, 50).")
    centered = points - np.mean(points, axis=0)
    eigenvalues, eigenvectors = np.linalg.eigh(np.cov(centered.T))
    order = np.argsort(eigenvalues)[::-1]
    if eigenvalues[order[-1]] <= max(eigenvalues[order[0]], 1e-30) * 1e-12:
        raise ValueError("The input must span three dimensions.")
    axes = eigenvectors[:, order]
    projected = centered @ axes
    extrema_cm = np.ptp(projected, axis=0) * 100.0
    robust_cm = np.array([
        np.percentile(projected[:, axis], 100.0 - trims[axis])
        - np.percentile(projected[:, axis], trims[axis])
        for axis in range(3)
    ]) * 100.0
    return robust_cm, extrema_cm, axes


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_phenotype_results(output_path: Path, traits: dict) -> None:
    """Write the single result CSV without creating any output directory."""
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(traits))
        writer.writeheader()
        writer.writerow(traits)


def build_parser() -> argparse.ArgumentParser:
    directory = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(
        description="Measure grape bunch dimensions, surface area, volume and berry count.",
        epilog="Writes only phenotype_results.csv in the same directory as this script.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--input", type=Path, default=directory / "data.ply",
                        help="One preprocessed PLY point cloud; XYZ required, RGB optional")
    parser.add_argument("--profile", choices=("sample1", "custom"), default="sample1",
                        help="Sample 1 reproduction or explicitly configured other data")
    parser.add_argument("--input-unit", choices=("m", "cm", "mm"), default="m",
                        help="Physical unit of the supplied PLY coordinates")
    parser.add_argument("--pca-trim-percent", nargs=3, type=float, metavar=("H", "W", "T"),
                        help="One-sided symmetric trimming for PC1/PC2/PC3, in percent")
    parser.add_argument("--geometry-calibration-scale", type=float,
                        help="Divide coordinates by this factor for area/volume only")
    parser.add_argument("--surface-slices-mm", type=float, nargs="+", default=[0.5, 0.75],
                        help="Contour thicknesses for arithmetic-mean surface area")
    parser.add_argument("--surface-axis", choices=("z", "pca"), default="z",
                        help="Surface contour slicing axis")
    parser.add_argument("--contour-points", type=int, default=64,
                        help="Equal-arclength samples per surface contour")
    parser.add_argument("--volume-slice-mm", type=float, default=1.0,
                        help="Thickness of volume slices")
    parser.add_argument("--volume-axis", choices=("pca", "z"), default="pca",
                        help="Volume slicing axis")
    parser.add_argument("--random-seed", type=int, default=42,
                        help="Seed for voting diagnostics and RANSAC")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.profile == "custom" and (
        args.pca_trim_percent is None or args.geometry_calibration_scale is None
    ):
        parser.error("--profile custom requires --pca-trim-percent H W T and "
                     "--geometry-calibration-scale SCALE.")
    trims = (args.pca_trim_percent if args.pca_trim_percent is not None
             else SAMPLE1_PROFILE["pca_trim_percent"])
    geometry_scale = (args.geometry_calibration_scale
                      if args.geometry_calibration_scale is not None
                      else SAMPLE1_PROFILE["geometry_calibration_scale"])
    if not math.isfinite(geometry_scale) or geometry_scale <= 0:
        parser.error("--geometry-calibration-scale must be finite and positive.")
    if any(not math.isfinite(v) or v < 0 or v >= 50 for v in trims):
        parser.error("Each PCA trimming percentage must be finite and in [0, 50).")
    if any(not math.isfinite(v) or v <= 0 for v in args.surface_slices_mm):
        parser.error("Surface slice thicknesses must be finite and positive.")
    if len(set(args.surface_slices_mm)) != len(args.surface_slices_mm):
        parser.error("Surface slice thicknesses must be distinct.")
    if not math.isfinite(args.volume_slice_mm) or args.volume_slice_mm <= 0:
        parser.error("--volume-slice-mm must be finite and positive.")
    if args.contour_points < 3:
        parser.error("--contour-points must be at least 3.")
    if args.random_seed < 0:
        parser.error("--random-seed must be nonnegative.")

    input_path = args.input.expanduser().resolve()
    output_path = Path(__file__).resolve().with_name("phenotype_results.csv")
    if not input_path.is_file():
        parser.error(f"PLY file not found: {input_path}. Place data.ply next to this "
                     "script or specify --input /path/to/data.ply.")
    if input_path == output_path:
        parser.error("The input file cannot be phenotype_results.csv.")

    print("[1/4] Reading preprocessed point cloud...", flush=True)
    input_hash = sha256_file(input_path)
    points, _ = read_vertex_ply(input_path)
    points = points * {"m": 1.0, "cm": 0.01, "mm": 0.001}[args.input_unit]
    if len(points) < 30:
        raise ValueError("At least 30 finite points are required for berry counting.")
    if float(np.max(np.ptp(points, axis=0))) > 1.0:
        raise ValueError("Cloud extent exceeds 1 m. Check --input-unit and "
                         "provide a preprocessed, reference-ball-removed grape bunch.")
    if args.profile == "sample1" and input_hash != SAMPLE1_INPUT_SHA256:
        print("Note: this file differs from the documented Sample 1 PLY. "
              "The displayed Sample 1 parameter settings will still be used. "
              "Use --profile custom with explicit settings for other data.",
              file=sys.stderr)

    print("[2/4] Measuring PCA dimensions...", flush=True)
    dimensions, _, _ = pca_dimensions_cm(points, trims)
    print("[3/4] Measuring contour surface area and volume...", flush=True)
    # Preserve the scale conventions of the existing experiments explicitly.
    # This factor is from preprocessing; it is never inferred from true traits.
    geometry_points = points * (1.0 / geometry_scale)
    surface_area, _ = multiscale_contour_slice_surface_cm2(
        geometry_points,
        tuple(value * 0.001 for value in args.surface_slices_mm),
        contour_points=args.contour_points,
        axis_mode=args.surface_axis,
    )
    volume, _ = contour_slice_convex_hull_volume_cm3(
        geometry_points, args.volume_slice_mm * 0.001, axis_mode=args.volume_axis,
    )
    print("[4/4] Fitting and counting individual berries...", flush=True)
    berry_parameters = BerryParameters(random_seed=args.random_seed)
    validate_arguments(berry_parameters)
    with contextlib.redirect_stdout(io.StringIO()):
        berry_count, _, _, _ = count_berries(
            points, berry_parameters,
        )

    traits = {
        "input_file": input_path.name,
        "point_count": int(len(points)),
        "height_cm": float(dimensions[0]),
        "width_cm": float(dimensions[1]),
        "thickness_cm": float(dimensions[2]),
        "surface_area_cm2": float(surface_area),
        "volume_cm3": float(volume),
        "berry_count": int(berry_count),
    }
    if not all(math.isfinite(traits[key]) and traits[key] > 0 for key in (
        "height_cm", "width_cm", "thickness_cm", "surface_area_cm2", "volume_cm3"
    )):
        raise RuntimeError("An invalid phenotype measurement was computed.")
    write_phenotype_results(output_path, traits)
    print("\nComputed phenotype measurements")
    print(f"  Height:       {dimensions[0]:.6f} cm")
    print(f"  Width:        {dimensions[1]:.6f} cm")
    print(f"  Thickness:    {dimensions[2]:.6f} cm")
    print(f"  Surface area: {surface_area:.6f} cm^2")
    print(f"  Volume:       {volume:.6f} cm^3")
    print(f"  Berry count:  {berry_count}")
    print(f"  Point count:  {len(points):,}")
    print(f"  Results:      {output_path}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, ValueError, RuntimeError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        raise SystemExit(1)
