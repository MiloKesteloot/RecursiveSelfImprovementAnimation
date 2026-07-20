"""Visual style sandbox: a single 8-node network using the "Bloom Pulse"
treatment -- the survivor of a wider comparison across several glow
styles, all built on the same glowing-circle node/edge language used in
recursive_self_improvement.py (layered halo rings, a blurred glow blob,
thin lit edges).

Bloom Pulse itself: each node's own flat halo rings are stripped and
replaced with a real blurred glow (gaussian raster, not stacked opacity
circles) behind a plain bright core. On top of that, chains of 3-5
pulses fire every so often -- one node pings, then an adjacent node
(following an actual edge) 0.2s later, then another, hopping across the
net -- each ping spawning its own independent ring (so a node hit twice
in quick succession shows two rings, not one cut short) and flashing
that node's own ring white-hot for an instant. 0.5s after a chain's
last hop fires, another chain starts from a random node.

The net appears instantly (no fade-in) and sits still -- no idle
drift/wiggle, which read as unsettling rather than lively for something
meant to be a neural net -- with pulses starting almost immediately
rather than after the long pause the scheduler used when this was part
of a larger comparison grid.

Run with:
    manim -pql visual_tests.py VisualStyleTests

Set HOLD_SECONDS to change how long the scene runs, e.g. for a quick look:
    HOLD_SECONDS=5 manim -pql visual_tests.py VisualStyleTests
"""

import os
import random

import numpy as np
from manim import Circle, Group, ImageMobject, Scene, config

from recursive_self_improvement import (
    BACKGROUND_COLOR,
    MAGENTA_EDGE,
    MAGENTA_PALETTE,
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


# --- Soft Bloom base: a true blurred glow behind each node, not stacked
#     rings -- a stack of flat, low-opacity circles always shows a
#     visible ring at every layer's radius, since alpha-compositing
#     discs is still a step function -- see make_glow_blob's own
#     docstring in recursive_self_improvement.py for the same reasoning.
#     Only a real per-pixel gradient, rendered as a raster image, reads
#     as an actual glow rather than a target/bullseye of rings. Each
#     bloom here is a small gaussian falloff rasterized once with numpy,
#     tracked to its node's live center every frame via its own updater
#     rather than folded into the node's own VGroup, since ImageMobject
#     can't live inside a VGroup (VGroup only accepts VMobjects).


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


# --- Bloom Pulse chains: one node pings, then an adjacent node
#     (following an actual edge), then another, 3-5 hops long with a
#     fixed stagger between each hop -- rather than every node
#     independently, randomly pinging on its own. Each ping spawns a
#     brand-new ring mobject rather than reusing one ring per node, so a
#     node that gets hit again while its last ring is still expanding
#     gets a second, independent ring instead of the first one being cut
#     off. Each ping also flashes that node's own mid ring white-hot for
#     an instant before it fades back to its resting color.


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


def add_node_flash(node, rest_color, flash_color, attack=0.05, release=0.6):
    """Ramp the node's mid ring (index 0, after strip_halo_rings leaves
    just [mid, core]) up to flash_color over `attack`, then ease it back
    down to rest_color over `release` -- a quick rise-then-fade, rather
    than snapping straight to flash_color on the first frame and only
    fading from there (which reads as a flat white pop, not a flash)."""
    mid = node[0]
    duration = attack + release
    state = {"active": False, "t": 0.0}

    def updater(mob, dt):
        if not state["active"]:
            return
        state["t"] += dt
        if state["t"] <= attack:
            color = _lerp_hex(rest_color, flash_color, state["t"] / attack)
        else:
            progress = min(1.0, (state["t"] - attack) / release)
            color = _lerp_hex(flash_color, rest_color, progress)
        mid.set_fill(color=color)
        mid.set_stroke(color=color)
        if state["t"] >= duration:
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

    # The flash fires slightly after the ring spawns -- simultaneous
    # felt backwards, since the ring starts at the node's own radius and
    # only becomes visibly bigger a beat later, while the flash (being
    # fast) is already done by then. Firing the flash FLASH_DELAY after
    # the ring gives the ring a moment's head start.
    FLASH_DELAY = 0.1

    def fire(idx):
        spawn_pulse_ring(node_list[idx], base_radius, core_color, extras)
        state["flash_queue"].append([FLASH_DELAY, idx])

    # queue holds [time_remaining, node_index] for hops still waiting to
    # fire; cooldown counts down to the next chain once the queue drains.
    # flash_queue holds the same shape for flashes waiting on FLASH_DELAY.
    # Initial cooldown is short -- the net appears instantly (no
    # fade-in), so the first chain should follow almost right away
    # rather than after the multi-second pause this used when it was
    # one cell among several others still fading in.
    state = {"cooldown": random.uniform(0.2, 0.6), "queue": [], "flash_queue": []}

    def scheduler(mob, dt):
        pending_flash = []
        for delay, idx in state["flash_queue"]:
            delay -= dt
            if delay <= 0:
                flash_triggers[idx]()
            else:
                pending_flash.append([delay, idx])
        state["flash_queue"] = pending_flash

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


def build_bloom_pulse_net(center, seed, palette, edge_color):
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
    # catch any updater added below.
    for node in nodes:
        node.clear_updaters()

    extras = Group()
    for node in nodes:
        strip_halo_rings(node)
        add_soft_bloom(node, NODE_RADIUS, palette[2], extras)
    add_pulse_chains(nodes, edges, NODE_RADIUS, palette, extras)
    net.add_to_back(extras)
    return net


class VisualStyleTests(Scene):
    def construct(self):
        self.camera.background_color = BACKGROUND_COLOR
        self.add(make_backdrop())
        self.add(build_bloom_pulse_net(np.array([0.0, 0.0, 0.0]), seed=104, palette=MAGENTA_PALETTE, edge_color=MAGENTA_EDGE))
        self.wait(HOLD_SECONDS)
