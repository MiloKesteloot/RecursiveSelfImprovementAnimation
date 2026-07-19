"""Visual style sandbox: 6 small (8-node) networks in a 3x2 grid, each
built and animated with a different visual treatment, for comparing
options against the look already used in recursive_self_improvement.py
before committing to one for the main piece.

Cell layout (row, col):
  (0,0) Glow Halo   -- the current recursive_self_improvement.py look,
                        reused directly from that module: layered halo
                        rings per node, a blurred silhouette glow blob
                        underneath, gentle per-node drift.
  (0,1) Pulse       -- flat core dot + a glow ring that breathes
                        (radius/opacity oscillate), each node out of
                        phase with its neighbors.
  (0,2) Wireframe   -- unfilled ring nodes, dashed edges, the whole net
                        slowly rotating as one rigid body.
  (1,0) Particle    -- tiny bright node cores in a soft halo, plus a
                        scattered field of small twinkling dust
                        particles behind the net.
  (1,1) Circuit     -- rounded-square "chip" nodes, right-angle elbowed
                        traces instead of straight edges, and bright
                        pulses that travel the traces on a loop.
  (1,2) Rainbow     -- nodes colored across the hue wheel, edges
                        gradient-colored between their endpoints, the
                        whole palette slowly cycling hue over time.

Run with:
    manim -pql visual_tests.py VisualStyleTests

Set HOLD_SECONDS to change how long the grid sits still (idle
animations keep running the whole time) before it fades out, e.g. for a
quick look:
    HOLD_SECONDS=3 manim -pql visual_tests.py VisualStyleTests
"""

import colorsys
import math
import os
import random

import numpy as np
from manim import (
    DOWN,
    Circle,
    Dot,
    DashedLine,
    FadeIn,
    FadeOut,
    Group,
    LaggedStart,
    Line,
    RoundedRectangle,
    Scene,
    Text,
    VGroup,
    VMobject,
    config,
)

from recursive_self_improvement import (
    BACKDROP_GLOW_COLOR,  # noqa: F401 (kept for parity with the source module)
    BACKGROUND_COLOR,
    GREEN_EDGE,
    GREEN_PALETTE,
    make_backdrop,
    make_edge,
    make_glow_blob,
    make_node,
    nearest_neighbor_edges,
    sample_cloud,
)

config.pixel_width = 1920
config.pixel_height = 1080
config.frame_width = config.frame_height * config.pixel_width / config.pixel_height

HOLD_SECONDS = float(os.environ.get("HOLD_SECONDS", "10"))

N_NODES = 8
K_NEIGHBORS = 2
CLOUD_RADIUS = 1.05
NODE_RADIUS = 0.11

COLS_X = [-4.5, 0.0, 4.5]
ROWS_Y = [2.1, -2.1]

PULSE_CORE = "#FFD9A0"
PULSE_GLOW = "#FF8C42"

WIREFRAME_COLOR = "#8FE3FF"

PARTICLE_NODE_RADIUS = 0.075
PARTICLE_CORE = "#FFFFFF"
PARTICLE_GLOW = "#8FB8FF"

CIRCUIT_NODE_SIZE = 0.16
CIRCUIT_CORE = "#B6FFB0"
CIRCUIT_GLOW = "#2FBE6A"


def hue_to_hex(hue, sat=0.75, val=1.0):
    r, g, b = colorsys.hsv_to_rgb(hue % 1.0, sat, val)
    return "#%02x%02x%02x" % (int(r * 255), int(g * 255), int(b * 255))


# --- Style 1: Glow Halo (the existing recursive_self_improvement.py look) --


def build_glow_halo(center, seed):
    points = sample_cloud(N_NODES, CLOUD_RADIUS, seed)
    world_points = [center + p for p in points]
    nodes_list = [make_node(p, NODE_RADIUS, GREEN_PALETTE) for p in world_points]
    nodes = VGroup(*nodes_list)
    edge_idx = nearest_neighbor_edges(points, K_NEIGHBORS)
    edges = VGroup(*[make_edge(nodes_list[i], nodes_list[j], GREEN_EDGE) for i, j in edge_idx])
    glow = make_glow_blob(points, edge_idx, CLOUD_RADIUS, NODE_RADIUS)
    glow.move_to(center)
    return Group(glow, edges, nodes)


# --- Style 2: Pulse -- breathing glow rings, out of phase per node --------


def add_pulse(node, ring, pos, phase, amplitude=0.32, speed=2.0):
    state = {"t": random.uniform(0, 2 * math.pi), "scale": 1.0}

    def updater(mob, dt):
        state["t"] += dt * speed
        wave = 0.5 + 0.5 * math.sin(state["t"] + phase)
        target_scale = 1.0 + amplitude * wave
        ring.scale(target_scale / state["scale"], about_point=pos)
        ring.set_stroke(opacity=0.3 + 0.5 * wave)
        ring.set_fill(opacity=0.08 + 0.24 * wave)
        state["scale"] = target_scale

    node.add_updater(updater)


def build_pulse(center, seed):
    points = sample_cloud(N_NODES, CLOUD_RADIUS, seed)
    world_points = [center + p for p in points]
    nodes = VGroup()
    for p in world_points:
        core = Circle(radius=NODE_RADIUS * 0.5, stroke_width=0, fill_color=PULSE_CORE, fill_opacity=1).move_to(p)
        ring = Circle(
            radius=NODE_RADIUS * 1.5, stroke_color=PULSE_GLOW, stroke_width=2, fill_color=PULSE_GLOW, fill_opacity=0.18
        ).move_to(p)
        node = VGroup(ring, core)
        add_pulse(node, ring, p, phase=random.uniform(0, 2 * math.pi))
        nodes.add(node)
    edge_idx = nearest_neighbor_edges(points, K_NEIGHBORS)
    edges = VGroup(
        *[
            Line(world_points[i], world_points[j], stroke_color=PULSE_GLOW, stroke_width=1.6, stroke_opacity=0.5)
            for i, j in edge_idx
        ]
    )
    return VGroup(edges, nodes)


# --- Style 3: Wireframe -- outline nodes, dashed edges, rigid rotation ----


def make_wireframe_node(pos, radius, color):
    ring = Circle(radius=radius * 1.3, stroke_color=color, stroke_width=1.6, fill_opacity=0).move_to(pos)
    dot = Circle(radius=radius * 0.28, stroke_width=0, fill_color=color, fill_opacity=0.9).move_to(pos)
    return VGroup(ring, dot)


def build_wireframe(center, seed):
    points = sample_cloud(N_NODES, CLOUD_RADIUS, seed)
    world_points = [center + p for p in points]
    nodes = VGroup(*[make_wireframe_node(p, NODE_RADIUS, WIREFRAME_COLOR) for p in world_points])
    edge_idx = nearest_neighbor_edges(points, K_NEIGHBORS)
    edges = VGroup(
        *[
            DashedLine(
                world_points[i], world_points[j], stroke_color=WIREFRAME_COLOR, stroke_width=1.4, dash_length=0.08
            )
            for i, j in edge_idx
        ]
    )
    group = VGroup(edges, nodes)
    group.add_updater(lambda m, dt: m.rotate(dt * 0.35, about_point=center))
    return group


# --- Style 4: Particle -- tight bright cores + a twinkling dust field ----


def make_dust(center, radius, count, color):
    dust = VGroup()
    for _ in range(count):
        r = radius * random.uniform(0.3, 1.15)
        theta = random.uniform(0, 2 * math.pi)
        p = center + np.array([r * math.cos(theta), r * math.sin(theta), 0])
        dot = Dot(point=p, radius=random.uniform(0.006, 0.018), color=color)
        base_op = random.uniform(0.15, 0.5)
        dot.set_opacity(base_op)
        phase = random.uniform(0, 2 * math.pi)
        speed = random.uniform(0.8, 2.0)
        state = {"t": 0.0}

        def updater(mob, dt, phase=phase, speed=speed, base_op=base_op, state=state):
            state["t"] += dt * speed
            mob.set_opacity(base_op * (0.3 + 0.7 * (0.5 + 0.5 * math.sin(state["t"] + phase))))

        dot.add_updater(updater)
        dust.add(dot)
    return dust


def build_particle(center, seed):
    points = sample_cloud(N_NODES, CLOUD_RADIUS, seed)
    world_points = [center + p for p in points]
    nodes_list = []
    for p in world_points:
        core = Circle(
            radius=PARTICLE_NODE_RADIUS * 0.6, stroke_width=0, fill_color=PARTICLE_CORE, fill_opacity=1
        ).move_to(p)
        halo = Circle(
            radius=PARTICLE_NODE_RADIUS * 3.2, stroke_width=0, fill_color=PARTICLE_GLOW, fill_opacity=0.22
        ).move_to(p)
        nodes_list.append(VGroup(halo, core))
    nodes = VGroup(*nodes_list)
    edge_idx = nearest_neighbor_edges(points, K_NEIGHBORS)
    edges = VGroup(
        *[
            Line(world_points[i], world_points[j], stroke_color=PARTICLE_GLOW, stroke_width=1.0, stroke_opacity=0.25)
            for i, j in edge_idx
        ]
    )
    dust = make_dust(center, CLOUD_RADIUS * 1.3, 45, PARTICLE_GLOW)
    return VGroup(edges, nodes, dust)


# --- Style 5: Circuit -- chip nodes, elbowed traces, traveling pulses ----


def make_chip(pos, size, color_core, color_glow):
    outer = RoundedRectangle(
        corner_radius=0.03, width=size * 2.2, height=size * 2.2, stroke_width=0, fill_color=color_glow, fill_opacity=0.18
    ).move_to(pos)
    body = RoundedRectangle(
        corner_radius=0.02, width=size * 1.3, height=size * 1.3, stroke_color=color_core, stroke_width=1.5,
        fill_color=color_glow, fill_opacity=0.6
    ).move_to(pos)
    return VGroup(outer, body)


def make_trace(a, b, color):
    mid = np.array([b[0], a[1], 0])
    path = VMobject(stroke_color=color, stroke_width=2)
    path.set_points_as_corners([a, mid, b])
    glow = path.copy().set_stroke(width=6, opacity=0.15)
    return VGroup(glow, path)


def add_traveling_pulse(path, color, period, phase):
    dot = Dot(radius=0.05, color=color)
    state = {"t": phase}

    def updater(mob, dt):
        state["t"] += dt
        prop = (state["t"] % period) / period
        mob.move_to(path.point_from_proportion(prop))

    dot.add_updater(updater)
    return dot


def build_circuit(center, seed):
    points = sample_cloud(N_NODES, CLOUD_RADIUS, seed)
    world_points = [center + p for p in points]
    chips = VGroup(*[make_chip(p, CIRCUIT_NODE_SIZE, CIRCUIT_CORE, CIRCUIT_GLOW) for p in world_points])
    edge_idx = nearest_neighbor_edges(points, K_NEIGHBORS)
    traces = VGroup()
    pulses = VGroup()
    for i, j in edge_idx:
        trace = make_trace(world_points[i], world_points[j], CIRCUIT_GLOW)
        traces.add(trace)
        pulses.add(add_traveling_pulse(trace[1], CIRCUIT_CORE, period=random.uniform(2.2, 3.6), phase=random.uniform(0, 3)))
    return VGroup(traces, chips, pulses)


# --- Style 6: Rainbow -- hue-cycling nodes, gradient edges ----------------


def make_rainbow_node(pos, radius, index, count):
    halo = Circle(radius=radius * 1.9, stroke_width=0, fill_opacity=0.22).move_to(pos)
    core = Circle(radius=radius * 0.55, stroke_width=1, fill_opacity=1).move_to(pos)
    node = VGroup(halo, core)
    state = {"t": random.uniform(0, 2 * math.pi)}

    def updater(mob, dt):
        state["t"] += dt
        color = hue_to_hex((index / count + state["t"] * 0.05) % 1.0)
        core.set_fill(color=color, opacity=1)
        core.set_stroke(color=color)
        halo.set_fill(color=color, opacity=0.22)

    node.add_updater(updater)
    return node, core


def build_rainbow(center, seed):
    points = sample_cloud(N_NODES, CLOUD_RADIUS, seed)
    world_points = [center + p for p in points]
    count = len(world_points)
    nodes_list = []
    cores = []
    for idx, p in enumerate(world_points):
        node, core = make_rainbow_node(p, NODE_RADIUS, idx, count)
        nodes_list.append(node)
        cores.append(core)
    nodes = VGroup(*nodes_list)
    edge_idx = nearest_neighbor_edges(points, K_NEIGHBORS)
    edges = VGroup()
    for i, j in edge_idx:
        line = Line(world_points[i], world_points[j], stroke_width=3)
        edge = VGroup(line)

        def updater(mob, i=i, j=j, line=line):
            line.set_color([cores[i].get_fill_color(), cores[j].get_fill_color()])

        edge.add_updater(updater)
        edges.add(edge)
    return VGroup(edges, nodes)


class VisualStyleTests(Scene):
    def construct(self):
        self.camera.background_color = BACKGROUND_COLOR
        backdrop = make_backdrop()
        self.add(backdrop)

        cells = [
            (ROWS_Y[0], COLS_X[0], "Glow Halo", build_glow_halo),
            (ROWS_Y[0], COLS_X[1], "Pulse", build_pulse),
            (ROWS_Y[0], COLS_X[2], "Wireframe", build_wireframe),
            (ROWS_Y[1], COLS_X[0], "Particle", build_particle),
            (ROWS_Y[1], COLS_X[1], "Circuit", build_circuit),
            (ROWS_Y[1], COLS_X[2], "Rainbow", build_rainbow),
        ]

        nets = []
        labels = []
        for seed_offset, (y, x, name, builder) in enumerate(cells):
            center = np.array([x, y, 0])
            nets.append(builder(center, seed=100 + seed_offset))
            label = Text(name, font="Consolas", color="#B9C7DE").scale(0.34)
            label.move_to(center + DOWN * 1.55)
            labels.append(label)

        self.play(
            LaggedStart(*[FadeIn(n, scale=0.6) for n in nets], lag_ratio=0.08, run_time=1.6),
            LaggedStart(*[FadeIn(l) for l in labels], lag_ratio=0.08, run_time=1.6),
        )

        self.wait(HOLD_SECONDS)

        self.play(
            *[FadeOut(n) for n in nets],
            *[FadeOut(l) for l in labels],
            FadeOut(backdrop),
            run_time=1.2,
        )
