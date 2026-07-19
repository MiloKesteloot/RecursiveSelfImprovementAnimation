"""Visual style sandbox, take 2: 6 small (8-node) networks in a 3x2 grid,
all built from the *same* glowing-circle node/edge language already used
in recursive_self_improvement.py (layered halo rings, a blurred glow
blob underneath, thin lit edges) -- unlike the first pass at this file,
none of these swap in a different node shape or drop the halo/blur look.
Each cell instead just dials a different knob on that same look: a
bigger/softer bloom, a breathing pulse, expanding sonar rings, a
traveling comet, or an organic flicker.

Cell layout (row, col):
  (0,0) Glow Halo    -- unchanged baseline from recursive_self_improvement.py:
                         layered halo rings, blurred glow blob, gentle drift.
  (0,1) Soft Bloom   -- same nodes plus one extra big, very soft bloom
                         circle behind each one, and a larger/softer
                         glow blob -- a turned-up, dreamier version of
                         the same halo.
  (0,2) Pulse Core   -- same nodes, but each one's outer halo ring
                         breathes (grows/brightens and shrinks/dims) on
                         its own slow cycle, out of phase with its
                         neighbors, while the core stays steady.
  (1,0) Sonar Rings  -- same nodes, plus each one periodically sends out
                         an expanding, fading ring -- like a soft radar
                         ping -- staggered per node.
  (1,1) Comet Edges  -- same nodes and edges, plus a small glowing comet
                         that continuously travels each edge from one
                         node to the other, fading in and out at the ends.
  (1,2) Flicker Aura -- same nodes, but each halo's brightness flickers
                         unevenly (summed random-phase sine waves, not a
                         clean pulse) with occasional brighter flares --
                         an organic, candle-like glow instead of a steady one.

Run with:
    manim -pql visual_tests.py VisualStyleTests

Set HOLD_SECONDS to change how long the grid sits still (idle animations
keep running the whole time) before it fades out, e.g. for a quick look:
    HOLD_SECONDS=3 manim -pql visual_tests.py VisualStyleTests
"""

import math
import os
import random

import numpy as np
from manim import DOWN, Circle, FadeIn, FadeOut, Group, LaggedStart, Scene, Text, VGroup, config

from recursive_self_improvement import (
    BACKGROUND_COLOR,
    BLUE_EDGE,
    BLUE_PALETTE,
    GREEN_EDGE,
    GREEN_PALETTE,
    MAGENTA_EDGE,
    MAGENTA_PALETTE,
    PURPLE_EDGE,
    PURPLE_PALETTE,
    RED_EDGE,
    RED_PALETTE,
    TEAL_EDGE,
    TEAL_PALETTE,
    build_net,
    make_backdrop,
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


# --- Soft Bloom: one extra, big, very soft circle behind each node -------


def add_soft_bloom(node, radius, glow_color):
    pos = node.get_center()
    bloom = Circle(radius=radius * 7.0, stroke_width=0, fill_color=glow_color, fill_opacity=0.045).move_to(pos)
    node.add_to_back(bloom)


# --- Pulse Core: the halo ring breathes, the core stays steady -----------


def add_pulse_breathe(node, amplitude=0.32, speed=1.6):
    _outer, halo, _mid, _core = node
    state = {"t": random.uniform(0, 2 * math.pi), "scale": 1.0}

    def updater(mob, dt):
        state["t"] += dt * speed
        wave = 0.5 + 0.5 * math.sin(state["t"])
        target_scale = 1.0 + amplitude * wave
        halo.scale(target_scale / state["scale"], about_point=node.get_center())
        halo.set_fill(opacity=0.14 + 0.3 * wave)
        state["scale"] = target_scale

    node.add_updater(updater)


# --- Sonar Rings: each node periodically pings an expanding, fading ring -


def add_sonar_rings(node, base_radius, color, extras, count=2, period=2.6):
    max_growth = base_radius * 4.2
    for k in range(count):
        ring = Circle(radius=base_radius, stroke_color=color, stroke_width=2, fill_opacity=0)
        state = {"t": k * (period / count)}

        def updater(mob, dt, state=state):
            state["t"] += dt
            progress = (state["t"] % period) / period
            mob.become(
                Circle(radius=base_radius + max_growth * progress, stroke_color=color, stroke_width=2, fill_opacity=0)
            )
            mob.move_to(node.get_center())
            mob.set_stroke(opacity=max(0.0, 1.0 - progress) * 0.55)

        ring.add_updater(updater)
        extras.add(ring)


# --- Comet Edges: a small glowing dot travels each edge on a loop --------


def add_edge_comet(edge, color, extras, period, phase):
    dot = Circle(radius=0.045, stroke_width=0, fill_color=color, fill_opacity=0.95)
    glow = Circle(radius=0.1, stroke_width=0, fill_color=color, fill_opacity=0.35)
    comet = VGroup(glow, dot)
    state = {"t": phase}

    def updater(mob, dt):
        state["t"] += dt
        prop = (state["t"] % period) / period
        a, b = edge.node_a.get_center(), edge.node_b.get_center()
        mob.move_to(a + (b - a) * prop)
        fade = min(1.0, prop * 6, (1 - prop) * 6)
        glow.set_fill(opacity=0.35 * fade)
        dot.set_fill(opacity=0.95 * fade)

    comet.add_updater(updater)
    extras.add(comet)


# --- Flicker Aura: organic, uneven brightness with occasional flares -----


def add_flicker(node, speed=1.0):
    outer, halo, _mid, _core = node
    freqs = [random.uniform(0.5, 1.4) for _ in range(3)]
    phases = [random.uniform(0, 2 * math.pi) for _ in range(3)]
    state = {"t": random.uniform(0, 10), "flare": 0.0}

    def updater(mob, dt):
        state["t"] += dt * speed
        wobble = sum(math.sin(state["t"] * f + p) for f, p in zip(freqs, phases)) / 3
        level = 0.6 + 0.4 * wobble
        state["flare"] = max(0.0, state["flare"] - dt * 1.4)
        if random.random() < dt * 0.15:
            state["flare"] = 1.0
        boost = 1.0 + 1.6 * state["flare"]
        outer.set_fill(opacity=min(1.0, 0.10 * level * boost))
        halo.set_fill(opacity=min(1.0, 0.28 * level * boost))

    node.add_updater(updater)


def build_cell(center, seed, palette, edge_color, style):
    net, nodes, edges, glow = build_net(
        n_nodes=N_NODES,
        cloud_radius=CLOUD_RADIUS,
        k_neighbors=K_NEIGHBORS,
        node_radius=NODE_RADIUS,
        seed=seed,
        center=center,
        palette=palette,
        edge_color=edge_color,
    )
    extras = VGroup()

    if style == "baseline":
        pass
    elif style == "bloom":
        glow.scale(1.35)
        for node in nodes:
            add_soft_bloom(node, NODE_RADIUS, palette[2])
    elif style == "pulse":
        for node in nodes:
            add_pulse_breathe(node)
    elif style == "sonar":
        for node in nodes:
            add_sonar_rings(node, NODE_RADIUS, edge_color, extras)
    elif style == "comet":
        for edge in edges:
            add_edge_comet(edge, palette[0], extras, period=random.uniform(1.8, 3.0), phase=random.uniform(0, 3))
    elif style == "flicker":
        for node in nodes:
            add_flicker(node)

    return Group(net, extras)


class VisualStyleTests(Scene):
    def construct(self):
        self.camera.background_color = BACKGROUND_COLOR
        backdrop = make_backdrop()
        self.add(backdrop)

        cells = [
            (ROWS_Y[0], COLS_X[0], "Glow Halo", GREEN_PALETTE, GREEN_EDGE, "baseline"),
            (ROWS_Y[0], COLS_X[1], "Soft Bloom", TEAL_PALETTE, TEAL_EDGE, "bloom"),
            (ROWS_Y[0], COLS_X[2], "Pulse Core", BLUE_PALETTE, BLUE_EDGE, "pulse"),
            (ROWS_Y[1], COLS_X[0], "Sonar Rings", PURPLE_PALETTE, PURPLE_EDGE, "sonar"),
            (ROWS_Y[1], COLS_X[1], "Comet Edges", MAGENTA_PALETTE, MAGENTA_EDGE, "comet"),
            (ROWS_Y[1], COLS_X[2], "Flicker Aura", RED_PALETTE, RED_EDGE, "flicker"),
        ]

        nets = []
        labels = []
        for seed_offset, (y, x, name, palette, edge_color, style) in enumerate(cells):
            center = np.array([x, y, 0])
            nets.append(build_cell(center, 100 + seed_offset, palette, edge_color, style))
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
