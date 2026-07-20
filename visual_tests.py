"""Visual style sandbox, take 2: 6 small (8-node) networks in a 3x2 grid,
all built from the *same* glowing-circle node/edge language already used
in recursive_self_improvement.py (layered halo rings, a blurred glow
blob underneath, thin lit edges) -- unlike the first pass at this file,
none of these swap in a different node shape or drop the halo/blur look.
Each cell instead just dials a different knob on that same look: a
bigger/softer bloom, a breathing pulse, expanding sonar rings, chains of
pulses hopping along edges, or bloom that itself breathes.

Unlike recursive_self_improvement.py, nodes here sit perfectly still --
that source file gives every node a small idle wander to read as
"alive," but for a *neural* net specifically, that same wiggle reads as
unsettling rather than lively, so it's stripped off right after
build_net for every cell (see the comment in build_cell).

Cell layout (row, col):
  (0,0) Glow Halo     -- unchanged baseline from recursive_self_improvement.py
                          (minus the drift): layered halo rings, blurred glow blob.
  (0,1) Soft Bloom    -- the node's own flat halo rings replaced with a
                          real blurred glow (gaussian raster, not stacked
                          opacity circles) behind a plain bright core.
  (0,2) Pulse Core    -- same nodes, but each one's outer halo ring
                          breathes (grows/brightens and shrinks/dims) on
                          its own slow cycle, out of phase with its
                          neighbors, while the core stays steady.
  (1,0) Sonar Rings   -- same nodes, plus each one periodically sends out
                          an expanding, fading ring -- like a soft radar
                          ping -- staggered per node.
  (1,1) Bloom Pulse   -- Soft Bloom's node/glow treatment, plus every so
                          often a chain of 3-5 pulses fires: one node
                          pings, then an adjacent node (following an
                          actual edge) 0.2s later, then another, hopping
                          across the net instead of pinging independently.
  (1,2) Bloom Breathe -- Soft Bloom's node/glow treatment, but the glow
                          discs themselves continuously grow and shrink
                          -- Pulse Core's breathing motion applied to the
                          blobs instead of to a ring outline.

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
from manim import DOWN, Circle, FadeIn, FadeOut, Group, ImageMobject, LaggedStart, Scene, Text, config

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
    _hex_to_rgb,
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


# --- Soft Bloom: a true blurred glow behind each node, not stacked rings -
#
# A stack of flat, low-opacity circles (what this used to be) always
# shows a visible ring at every layer's radius, since alpha-compositing
# discs is still a step function -- see make_glow_blob's own docstring
# in recursive_self_improvement.py for the same reasoning. Only a real
# per-pixel gradient, rendered as a raster image, reads as an actual
# glow rather than a target/bullseye of rings. So each bloom here is a
# small gaussian falloff rasterized once with numpy, same as
# make_glow_blob's technique -- just one soft blob per node instead of
# one shared image for the whole net. It's tracked to that node's live
# (drifting) center every frame via its own updater rather than folded
# into the node's own VGroup, since ImageMobject can't live inside a
# VGroup (VGroup only accepts VMobjects) -- unlike the earlier, broken
# attempt at this (scaling one shared image of the whole net's
# silhouette, which desynced every node but the one nearest its center),
# a separate per-node image tracked individually can't drift out of
# alignment with the node it belongs to.


def make_glow_disc(span_radius, sigma_radius, color, peak_alpha):
    resolution = 96
    scale = resolution / (span_radius * 2)
    cx = cy = resolution / 2
    ys, xs = np.mgrid[0:resolution, 0:resolution]
    dist = np.sqrt((xs - cx) ** 2 + (ys - cy) ** 2)
    sigma_px = sigma_radius * scale
    alpha = peak_alpha * np.exp(-(dist**2) / (2 * sigma_px**2))

    rgba = np.zeros((resolution, resolution, 4), dtype=np.uint8)
    rgba[..., 0:3] = _hex_to_rgb(color).astype(np.uint8)
    rgba[..., 3] = np.clip(alpha, 0, 255).astype(np.uint8)

    image = ImageMobject(rgba)
    span = span_radius * 2
    image.stretch_to_fit_width(span)
    image.stretch_to_fit_height(span)
    return image


def strip_halo_rings(node):
    """Remove the node's own flat halo rings (outer, halo) -- with those
    still layered on top of the blurred glow discs below, the node reads
    as double-glowing. Left with just mid/core, the node is a plain
    bright dot and the gaussian discs become the only source of glow
    around it."""
    outer, halo, _mid, _core = node
    node.remove(outer, halo)


def add_soft_bloom(node, radius, glow_color, extras):
    for span_mult, sigma_mult, peak_alpha in ((6.5, 2.2, 60), (3.6, 1.0, 115)):
        disc = make_glow_disc(radius * span_mult, radius * sigma_mult, glow_color, peak_alpha)
        disc.move_to(node.get_center())
        disc.add_updater(lambda mob: mob.move_to(node.get_center()))
        extras.add(disc)


# --- Bloom Breathe: Soft Bloom's static glow, plus a new crisp ring
#     breathing on top -- exactly Pulse Core's halo mechanic, just as an
#     added ring rather than reusing the node's own (already-stripped)
#     halo child. The blurry bloom discs themselves are untouched here
#     and never animate; only this new solid-stroke ring pulses.


def add_bloom_breathe(node, radius, glow_color, extras, amplitude=0.32, speed=1.6):
    add_soft_bloom(node, radius, glow_color, extras)

    ring = Circle(radius=radius * 1.5, stroke_color=glow_color, stroke_width=2, fill_color=glow_color, fill_opacity=0.18)
    ring.move_to(node.get_center())
    node.add(ring)

    state = {"t": random.uniform(0, 2 * math.pi), "scale": 1.0}

    def updater(mob, dt):
        state["t"] += dt * speed
        wave = 0.5 + 0.5 * math.sin(state["t"])
        target_scale = 1.0 + amplitude * wave
        ring.scale(target_scale / state["scale"], about_point=node.get_center())
        ring.set_fill(opacity=0.14 + 0.3 * wave)
        ring.set_stroke(opacity=0.35 + 0.5 * wave)
        state["scale"] = target_scale

    node.add_updater(updater)


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


# --- Bloom Pulse: soft bloom nodes, plus occasional chains of pulses --
#     one node pings, then an adjacent node (following an actual edge),
#     then another, 3-5 hops long with a fixed stagger between each hop
#     -- rather than every node independently, randomly pinging on its
#     own. Each ping spawns a brand-new ring mobject rather than
#     reusing one ring per node, so a node that gets hit again while its
#     last ring is still expanding gets a second, independent ring
#     instead of the first one being cut off. Each ping also flashes
#     that node's own mid ring white-hot for an instant before it fades
#     back to its resting color, so the node itself visibly "fires" too.


def _lerp_hex(hex_a, hex_b, t):
    a, b = _hex_to_rgb(hex_a), _hex_to_rgb(hex_b)
    rgb = np.clip(a + (b - a) * t, 0, 255).astype(int)
    return "#%02x%02x%02x" % tuple(rgb)


def spawn_pulse_ring(node, base_radius, color, extras, max_growth_mult=1.6, duration=2.0, peak_opacity=0.425):
    ring = Circle(radius=base_radius, stroke_color=color, stroke_width=2.4, fill_opacity=0)
    ring.move_to(node.get_center())
    max_growth = base_radius * max_growth_mult
    state = {"t": 0.0}

    def updater(mob, dt):
        state["t"] += dt
        progress = state["t"] / duration
        if progress >= 1.0:
            extras.remove(mob)
            mob.clear_updaters()
            return
        mob.become(
            Circle(radius=base_radius + max_growth * progress, stroke_color=color, stroke_width=2.4, fill_opacity=0)
        )
        mob.move_to(node.get_center())
        mob.set_stroke(opacity=(1.0 - progress) * peak_opacity)

    ring.add_updater(updater)
    extras.add(ring)


def add_node_flash(node, rest_color, flash_color, duration=0.15):
    """Snap the node's mid ring (index 0, after strip_halo_rings leaves
    just [mid, core]) to flash_color the instant this fires, then ease
    it back to rest_color over `duration` -- a quick white-hot flash
    rather than a slow fade."""
    mid = node[0]
    state = {"active": False, "t": 0.0}

    def updater(mob, dt):
        if not state["active"]:
            return
        state["t"] += dt
        progress = min(1.0, state["t"] / duration)
        color = _lerp_hex(flash_color, rest_color, progress)
        mid.set_fill(color=color)
        mid.set_stroke(color=color)
        if progress >= 1.0:
            state["active"] = False

    def trigger():
        state["active"] = True
        state["t"] = 0.0

    node.add_updater(updater)
    return trigger


def add_pulse_chains(nodes, edges, base_radius, palette, extras, chain_stagger=0.2):
    core_color, mid_color = palette[0], palette[1]
    node_list = list(nodes)
    index_of = {id(n): i for i, n in enumerate(node_list)}
    adjacency = {i: [] for i in range(len(node_list))}
    for edge in edges:
        i, j = index_of[id(edge.node_a)], index_of[id(edge.node_b)]
        adjacency[i].append(j)
        adjacency[j].append(i)

    flash_triggers = [add_node_flash(node, mid_color, core_color) for node in node_list]

    def fire(idx):
        spawn_pulse_ring(node_list[idx], base_radius, core_color, extras)
        flash_triggers[idx]()

    # queue holds [time_remaining, node_index] for hops still waiting to
    # fire; cooldown counts down to the next chain once the queue drains.
    state = {"cooldown": random.uniform(1.5, 6.0), "queue": []}

    def scheduler(mob, dt):
        pending = []
        for delay, idx in state["queue"]:
            delay -= dt
            if delay <= 0:
                fire(idx)
            else:
                pending.append([delay, idx])
        state["queue"] = pending

        if state["queue"]:
            return
        state["cooldown"] -= dt
        if state["cooldown"] > 0:
            return

        length = random.randint(3, 5)
        current = random.randrange(len(node_list))
        path = [current]
        visited = {current}
        for _ in range(length - 1):
            neighbors = adjacency[current]
            if not neighbors:
                break
            unvisited = [n for n in neighbors if n not in visited]
            current = random.choice(unvisited) if unvisited else random.choice(neighbors)
            path.append(current)
            visited.add(current)

        state["queue"] = [[i * chain_stagger, idx] for i, idx in enumerate(path)]
        state["cooldown"] = 0.5  # next chain starts 0.5s after this one's last hop fires

    extras.add_updater(scheduler)


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
    # build_net wires up a slow idle drift/wiggle on every node (meant to
    # read as "alive" in recursive_self_improvement.py) -- dropped here
    # since for a neural net specifically that reads as unsettling rather
    # than lively. This is the only updater on a node at this point (edge
    # updaters live on the edges, not the nodes), so clearing here can't
    # catch any style-specific updater added below.
    for node in nodes:
        node.clear_updaters()

    extras = Group()

    if style == "baseline":
        pass
    elif style == "bloom":
        for node in nodes:
            strip_halo_rings(node)
            add_soft_bloom(node, NODE_RADIUS, palette[2], extras)
        net.add_to_back(extras)
        return net
    elif style == "pulse":
        for node in nodes:
            add_pulse_breathe(node)
    elif style == "sonar":
        for node in nodes:
            add_sonar_rings(node, NODE_RADIUS, edge_color, extras)
    elif style == "bloom_pulse":
        for node in nodes:
            strip_halo_rings(node)
            add_soft_bloom(node, NODE_RADIUS, palette[2], extras)
        add_pulse_chains(nodes, edges, NODE_RADIUS, palette, extras)
        net.add_to_back(extras)
        return net
    elif style == "bloom_breathe":
        for node in nodes:
            strip_halo_rings(node)
            add_bloom_breathe(node, NODE_RADIUS, palette[2], extras)
        net.add_to_back(extras)
        return net

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
            (ROWS_Y[1], COLS_X[1], "Bloom Pulse", MAGENTA_PALETTE, MAGENTA_EDGE, "bloom_pulse"),
            (ROWS_Y[1], COLS_X[2], "Bloom Breathe", RED_PALETTE, RED_EDGE, "bloom_breathe"),
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

        # Bloom Pulse's chain scheduler keeps spawning/removing ring
        # mobjects for as long as it has updaters -- if that's still
        # happening *during* the FadeOut below, the animation's family
        # snapshot (taken once at the start) stops matching the live
        # mobject partway through and manim's zip() crashes. Freezing
        # every updater first (recursively, so it reaches the scheduler
        # on `extras` and any ring/flash updaters still in flight) stops
        # that churn before the fade begins.
        for n in nets:
            n.clear_updaters()

        self.play(
            *[FadeOut(n) for n in nets],
            *[FadeOut(l) for l in labels],
            FadeOut(backdrop),
            run_time=1.2,
        )
