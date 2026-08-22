"""Viser measure tools for LingBot Map — pick closest point, distance, area."""

from __future__ import annotations

from typing import Any

import numpy as np


def _gather_points(viewer: Any, max_points: int = 400_000) -> np.ndarray:
    """Flatten visible point clouds (finite XYZ only), optionally subsample."""
    chunks: list[np.ndarray] = []
    steps = getattr(viewer, "all_steps", None) or list(getattr(viewer, "pcs", {}).keys())
    thr = float(getattr(getattr(viewer, "vis_threshold_slider", None), "value", 0.0) or 0.0)
    # Prefer constructor threshold if slider not ready
    if thr <= 0:
        thr = float(getattr(viewer, "vis_threshold", 1.5) or 1.5)
    down = int(getattr(getattr(viewer, "downsample_slider", None), "value", 0) or 0)
    if down <= 0:
        down = int(getattr(viewer, "downsample_factor", 10) or 10)
    down = max(1, down)

    for step in steps:
        entry = viewer.pcs.get(step)
        if not entry:
            continue
        pc = np.asarray(entry["pc"], dtype=np.float32).reshape(-1, 3)
        if pc.size == 0:
            continue
        conf = entry.get("conf")
        if conf is not None:
            conf = np.asarray(conf, dtype=np.float32).reshape(-1)
            if conf.shape[0] == pc.shape[0]:
                pc = pc[conf >= thr]
        if down > 1 and pc.shape[0] > 0:
            pc = pc[::down]
        finite = np.isfinite(pc).all(axis=1)
        pc = pc[finite]
        if pc.shape[0]:
            chunks.append(pc)

    if not chunks:
        return np.zeros((0, 3), dtype=np.float32)
    pts = np.concatenate(chunks, axis=0)
    if pts.shape[0] > max_points:
        idx = np.linspace(0, pts.shape[0] - 1, max_points, dtype=np.int64)
        pts = pts[idx]
    return pts


def _closest_point_on_ray(
    points: np.ndarray,
    origin: np.ndarray,
    direction: np.ndarray,
) -> tuple[np.ndarray | None, float]:
    """Return nearest point to the ray and perpendicular distance."""
    if points is None or points.shape[0] == 0:
        return None, float("inf")
    o = np.asarray(origin, dtype=np.float64).reshape(3)
    d = np.asarray(direction, dtype=np.float64).reshape(3)
    n = np.linalg.norm(d)
    if n < 1e-12:
        return None, float("inf")
    d = d / n
    # Reject points behind the camera (t < 0)
    rel = points.astype(np.float64) - o
    t = rel @ d
    mask = t > 0.0
    if not np.any(mask):
        return None, float("inf")
    cand = points[mask]
    rel = rel[mask]
    t = t[mask]
    closest_on_ray = o + t[:, None] * d
    dist = np.linalg.norm(cand.astype(np.float64) - closest_on_ray, axis=1)
    i = int(np.argmin(dist))
    return cand[i].astype(np.float32), float(dist[i])


def _triangle_area(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> float:
    return 0.5 * float(np.linalg.norm(np.cross(b - a, c - a)))


def _fmt_xyz(p: np.ndarray) -> str:
    return "({:.4f}, {:.4f}, {:.4f})".format(float(p[0]), float(p[1]), float(p[2]))


def install_measure_tools(viewer: Any) -> None:
    """Add Measure folder + click-to-pick closest point on the point cloud."""
    import viser

    if getattr(viewer, "_olares_measure_installed", False):
        return
    viewer._olares_measure_installed = True

    state: dict[str, Any] = {
        "points": _gather_points(viewer),
        "picks": [],  # list[np.ndarray]
        "markers": [],  # scene handles
        "lines": [],
        "enabled": True,
    }

    with viewer.server.gui.add_folder("Measure (Olares)"):
        info = viewer.server.gui.add_markdown(
            "**Click scene** to snap to nearest cloud point.\n"
            "2+ picks → segment + path length. 3+ → triangle area of last 3.\n"
            "Cloud pts cached: **{}**".format(state["points"].shape[0])
        )
        enable = viewer.server.gui.add_checkbox(
            "Enable pick",
            initial_value=True,
            hint="When on, left-click ray finds closest reconstructed point.",
        )
        refresh_btn = viewer.server.gui.add_button(
            "Refresh point cache",
            hint="Rebuild from current conf/downsample filters (after slider changes).",
        )
        undo_btn = viewer.server.gui.add_button("Undo last pick")
        clear_btn = viewer.server.gui.add_button("Clear all picks")
        readout = viewer.server.gui.add_markdown("_No picks yet._")

    def _marker_radius() -> float:
        try:
            _, scale = viewer._compute_scene_center_and_scale()
            return float(max(0.005, min(0.2, scale * 0.01)))
        except Exception:
            return 0.02

    def _clear_markers() -> None:
        for h in state["markers"]:
            try:
                h.remove()
            except Exception:
                pass
        for h in state["lines"]:
            try:
                h.remove()
            except Exception:
                pass
        state["markers"] = []
        state["lines"] = []

    def _redraw() -> None:
        _clear_markers()
        picks = state["picks"]
        colors = [
            (1.0, 0.2, 0.2),
            (0.2, 1.0, 0.3),
            (0.2, 0.5, 1.0),
            (1.0, 0.85, 0.1),
            (1.0, 0.4, 0.9),
        ]
        for i, p in enumerate(picks):
            c = colors[i % len(colors)]
            h = viewer.server.scene.add_icosphere(
                name="/olares_measure/pt_{}".format(i),
                radius=_marker_radius(),
                color=c,
                position=(float(p[0]), float(p[1]), float(p[2])),
            )
            state["markers"].append(h)
        for i in range(len(picks) - 1):
            a, b = picks[i], picks[i + 1]
            lh = viewer.server.scene.add_spline_catmull_rom(
                name="/olares_measure/seg_{}".format(i),
                positions=(
                    (float(a[0]), float(a[1]), float(a[2])),
                    (float(b[0]), float(b[1]), float(b[2])),
                ),
                color=(1.0, 0.9, 0.2),
                line_width=3.0,
            )
            state["lines"].append(lh)

        lines = ["**Picks:** {}".format(len(picks))]
        for i, p in enumerate(picks):
            lines.append("- P{} {}".format(i + 1, _fmt_xyz(p)))
        if len(picks) >= 2:
            segs = []
            total = 0.0
            for i in range(len(picks) - 1):
                d = float(np.linalg.norm(picks[i + 1] - picks[i]))
                segs.append("P{}–P{} = {:.4f}".format(i + 1, i + 2, d))
                total += d
            lines.append("**Distances:** " + "; ".join(segs))
            lines.append("**Path length:** {:.4f}".format(total))
        if len(picks) >= 3:
            a, b, c = picks[-3], picks[-2], picks[-1]
            area = _triangle_area(a, b, c)
            lines.append(
                "**Triangle area (last 3):** {:.6f} (half cross-product; same units as coords)".format(
                    area
                )
            )
        if len(picks) == 0:
            readout.content = "_No picks yet. Click in the 3D view._"
        else:
            readout.content = "\n".join(lines)

    @enable.on_update
    def _(_) -> None:
        state["enabled"] = bool(enable.value)

    @refresh_btn.on_click
    def _(_) -> None:
        state["points"] = _gather_points(viewer)
        info.content = (
            "**Click scene** to snap to nearest cloud point.\n"
            "2+ picks → segment + path length. 3+ → triangle area of last 3.\n"
            "Cloud pts cached: **{}**".format(state["points"].shape[0])
        )

    @undo_btn.on_click
    def _(_) -> None:
        if state["picks"]:
            state["picks"].pop()
            _redraw()

    @clear_btn.on_click
    def _(_) -> None:
        state["picks"] = []
        _redraw()

    def _on_click(event: Any) -> None:
        if not state["enabled"]:
            return
        pts = state["points"]
        if pts.shape[0] == 0:
            state["points"] = _gather_points(viewer)
            pts = state["points"]
        if pts.shape[0] == 0:
            readout.content = "_No points in cache — reconstruct first / refresh cache._"
            return
        origin = getattr(event, "ray_origin", None)
        direction = getattr(event, "ray_direction", None)
        if origin is None or direction is None:
            readout.content = "_Click event missing ray — try again / update viser._"
            return
        origin = np.asarray(origin, dtype=np.float64)
        direction = np.asarray(direction, dtype=np.float64)
        hit, ray_dist = _closest_point_on_ray(pts, origin, direction)
        if hit is None:
            readout.content = "_No point near ray (try another angle)._"
            return
        state["picks"].append(hit)
        _redraw()
        # Append ray miss distance hint
        extra = " _(ray→point {:.4f})_".format(ray_dist)
        readout.content = readout.content + extra

    # Register for future clients; wrap existing connect hook
    prev_connect = getattr(viewer, "_connect_client", None)

    def _connect_client(client: Any) -> None:
        if callable(prev_connect):
            prev_connect(client)
        try:
            # Prefer scene pointer click API (viser ≥0.2)
            @client.scene.on_click
            def _(event: Any) -> None:
                _on_click(event)
        except Exception:
            try:

                @client.scene.on_pointer_event  # type: ignore[attr-defined]
                def _(event: Any) -> None:
                    et = getattr(event, "event_type", None) or getattr(event, "type", "")
                    if str(et).lower() in ("click", "pointerdown", ""):
                        _on_click(event)
            except Exception as exc:
                readout.content = "_Measure click hook failed: {}_".format(exc)

    viewer._connect_client = _connect_client  # type: ignore[method-assign]
    # Hot-wire already connected clients
    try:
        for client in viewer.server.get_clients().values():
            try:

                @client.scene.on_click
                def _(event: Any, _c=client) -> None:
                    _on_click(event)
            except Exception:
                pass
    except Exception:
        pass

    # Keep a reference so GC doesn't drop closures
    viewer._olares_measure_state = state
