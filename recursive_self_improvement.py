"""Recursive self-improvement, for a documentary segment (16:9).

Styled after assets/glow-net.png: a glowing net on the left, and a
thick, chunky arrow on each side of a bracketed code block in between
-- one feeding the net into the code, one feeding the code into the
next net, which grows in on the right. The bracket is built from two
straight segments and two quarter-circle arcs (see make_bracket),
matching the reference's angular brace rather than manim's own Brace.
The code itself is small and dense -- more a wall of real-looking code
scrolling by than something the viewer is meant to actually read.
Every node drifts a little in place the whole time it's on screen, so
the nets read as alive rather than static diagrams. Each net in the
chain keeps its own fixed color for its whole lifetime -- green, teal,
blue, purple, magenta -- rather than cooling to a uniform "current"
color, so the chain of successors stays visually distinct at a glance.
The left net and the code vanish while the right net slides into the
left slot, becoming "current" for the next lap. The code itself climbs
a ladder of real-looking ML syntax lap over lap: an ordinary training
step, then architecture search, then meta-gradients tuning the
learning process itself, then the net patching its own source. Each
lap is also faster than the last (SPEED_MULTIPLIERS), slow and
deliberate at first, compressing into a blur by the final lap -- the
pacing itself is the takeoff curve.

The final net is red, and isn't flat: grown the same way as the others
(an arrow, code writing itself behind a brace, another arrow), what
appears in its place is a regular icosahedron (12 vertices, 30 edges) --
the same glowing node/edge language, just not flattened. It's brought to
center, the camera tilts to reveal its depth, and it spins for a while
before the scene fades out.

Pure visual, no on-screen narration/labels -- meant to sit under a
voiceover.

Run with:
    manim -pql recursive_self_improvement.py RecursiveSelfImprovement

For fast iteration, set FAST_PREVIEW=1 to clamp every idle hold (including
the 14s icosahedron spin) down to a short beat, cutting render time without
touching any actual grow/write/shift animation:
    FAST_PREVIEW=1 manim -pql recursive_self_improvement.py RecursiveSelfImprovement
"""

import math
import os
import random

import numpy as np
from scipy.ndimage import gaussian_filter
from manim import (
    DEGREES,
    DOWN,
    LEFT,
    ORIGIN,
    PI,
    Arc,
    Arrow,
    Circle,
    Dot3D,
    FadeIn,
    FadeOut,
    Group,
    GrowArrow,
    GrowFromCenter,
    ImageMobject,
    LaggedStart,
    Line,
    Line3D,
    Text,
    ThreeDScene,
    VGroup,
    Write,
    config,
)

config.pixel_width = 1920
config.pixel_height = 1080
config.frame_width = config.frame_height * config.pixel_width / config.pixel_height

# Set FAST_PREVIEW=1 in the environment for quick iteration: every "hold"
# (a self.wait() where nothing is growing/writing/shifting -- just sitting
# there being looked at) gets clamped to a short beat instead of its full
# scripted length, and the 14s ambient icosahedron spin shortens to a few
# seconds. Actual grow/write/shift animations are untouched either way, so
# fast-preview still shows every beat of the scene, just without paying
# render time for the parts that are pure dead air. Leave FAST_PREVIEW unset
# (the default) for the real, full-paced final render.
FULL_TIMING = os.environ.get("FAST_PREVIEW", "0") != "1"
IDLE_HOLD = 0.06
SPIN_HOLD = 3.0

# Set INCLUDE_FINALE=1 to render the icosahedron reveal + 3D camera spin
# after the flat-net chain. That finale is the overwhelming majority of
# render time: true 3D mobjects (Dot3D/Line3D) under a continuously
# rotating camera force a full re-projection of the whole shape on every
# single frame, unlike the flat 2D nets/lines everywhere else in the
# scene, which stayed cheap even across a full render. Off by default so
# the rest of the animation renders fast while it's still being iterated
# on; flip it on when the finale itself needs work.
INCLUDE_FINALE = os.environ.get("INCLUDE_FINALE", "0") == "1"

BACKGROUND_COLOR = "#0A1830"
BACKDROP_GLOW_COLOR = "#1C3D66"

GREEN_CORE = "#EAFFF6"
GREEN_MID = "#7FF0C0"
GREEN_GLOW = "#2FBE86"
GREEN_EDGE = "#4FD8A8"
GREEN_PALETTE = (GREEN_CORE, GREEN_MID, GREEN_GLOW)

BLUE_CORE = "#EAF4FF"
BLUE_MID = "#8FCBFF"
BLUE_GLOW = "#2E86D6"
BLUE_EDGE = "#5AAEEF"
BLUE_PALETTE = (BLUE_CORE, BLUE_MID, BLUE_GLOW)

TEAL_CORE = "#EAFFFB"
TEAL_MID = "#7FF0E0"
TEAL_GLOW = "#2FBEA8"
TEAL_EDGE = "#4FD8C8"
TEAL_PALETTE = (TEAL_CORE, TEAL_MID, TEAL_GLOW)

PURPLE_CORE = "#F5EAFF"
PURPLE_MID = "#C89BFF"
PURPLE_GLOW = "#8A4FD6"
PURPLE_EDGE = "#A46AEF"
PURPLE_PALETTE = (PURPLE_CORE, PURPLE_MID, PURPLE_GLOW)

MAGENTA_CORE = "#FFEAF8"
MAGENTA_MID = "#FF9BE0"
MAGENTA_GLOW = "#D63AAE"
MAGENTA_EDGE = "#EF5AC8"
MAGENTA_PALETTE = (MAGENTA_CORE, MAGENTA_MID, MAGENTA_GLOW)

RED_CORE = "#FFECEC"
RED_MID = "#FF9B9B"
RED_GLOW = "#D63A3A"
RED_EDGE = "#EF5A5A"
RED_PALETTE = (RED_CORE, RED_MID, RED_GLOW)

# Each net in the chain keeps its own fixed color for its whole lifetime,
# in this order, rather than cooling to a uniform "current" color. The
# hue creeps steadily toward red across the chain so the finale's red
# icosahedron reads as the endpoint of a trend, not an arbitrary switch.
NET_PALETTES = [GREEN_PALETTE, TEAL_PALETTE, BLUE_PALETTE, PURPLE_PALETTE, MAGENTA_PALETTE]
NET_EDGE_COLORS = [GREEN_EDGE, TEAL_EDGE, BLUE_EDGE, PURPLE_EDGE, MAGENTA_EDGE]

CODE_COLOR = "#7FE8C0"
ARROW_COLOR = "#8FA6C2"

LEFT_X = -4.6
MID_X = 0.0
RIGHT_X = 4.6

# How far (and how fast) each node idly wanders from its own home spot.
DRIFT_AMPLITUDE = 0.035

# Each transition's pseudocode reads like a step up the real
# self-improvement ladder: an ordinary training step, then architecture
# search, then meta-gradients tuning the learning process itself, then
# the net patching its own source, ending in a line that -- deliberately
# -- doesn't have a peaceful reading. Each block runs long and small
# (see make_code_block) -- dense enough that it reads as "real code
# scrolling by," not as something the viewer is meant to actually read.
CODE_STAGE_1 = [
    "import torch",
    "import torch.nn.functional as F",
    "for x_batch, y in loader:",
    "    logits = net(x_batch)",
    "    loss = F.cross_entropy(logits, y)",
    "    optimizer.zero_grad()",
    "    loss.backward()",
    "    optimizer.step()",
    "    scheduler.step()",
]

CODE_STAGE_2 = [
    "search_space = NAS.default_space()",
    "successor = NAS(net, search_space).sample()",
    "successor.load_state_dict(net.sd, strict=False)",
    "successor.to(device)",
    "opt = AdamW(successor.parameters(), lr=3e-4)",
    "for step in range(budget * 4):",
    "    successor.train_step(next(loader))",
    "net = successor if successor.val_acc > net.val_acc else net",
]

CODE_STAGE_3 = [
    "meta_opt = SGD(net.hyperparams, lr=meta_lr)",
    "for outer_step in range(meta_steps):",
    "    g = meta_grad(meta_loss, net.hyperparams)",
    "    meta_opt.step(g)",
    "    net.hp -= meta_lr * g",
    "    net.rewrite_training_loop()",
    "    net.validate(held_out)",
]

CODE_STAGE_4 = [
    "critique = net.self_evaluate(net.source)",
    "patch = net.propose_patch(critique)",
    "sandbox = Sandbox(net.clone())",
    "sandbox.apply_patch(patch)",
    "if sandbox.score > net.score:",
    "    net = net.apply_patch(patch)",
    "    net.log_patch(patch)",
    "assert net.score > net.prev_score",
]

CODE_STAGE_FINAL = [
    "successor = population.best()",
    "if successor.score > net.score:",
    "    net = successor",
    "    depth += 1",
    "checkpoint(net, depth)",
    "if depth > SAFE_LIMIT:",
    "    alert_operators()",
    "    break_containment()",
]

# Growth chain: net 0 spawns on the left, deliberately tiny (4 nodes) so
# the whole chain reads as starting small -- each following flat net
# spawns bigger and denser on the right, then slides into the left
# slot. cloud_radius scales with sqrt(n_nodes) so node density (and
# therefore the per-node glow blur, see make_glow_blob) stays visually
# consistent across a wide range of net sizes rather than one stage
# reading as noticeably denser or sparser than its neighbors.
STAGES = [
    dict(n_nodes=4, cloud_radius=0.74, k_neighbors=1, node_radius=0.11, seed=1),
    dict(n_nodes=9, cloud_radius=1.11, k_neighbors=2, node_radius=0.09, seed=2),
    dict(n_nodes=16, cloud_radius=1.48, k_neighbors=2, node_radius=0.078, seed=3),
    dict(n_nodes=24, cloud_radius=1.81, k_neighbors=3, node_radius=0.065, seed=4),
    dict(n_nodes=34, cloud_radius=2.15, k_neighbors=3, node_radius=0.055, seed=5),
]
CODE_STAGES = [CODE_STAGE_1, CODE_STAGE_2, CODE_STAGE_3, CODE_STAGE_4]

# The mini-movie 1 intro's end net: a plain "larger net appears" beat, not
# part of the main STAGES growth chain, so it gets its own stage dict
# rather than being shoehorned into the existing 4/9/16/24/34 sequence.
# cloud_radius follows the same cloud_radius = 0.37 * sqrt(n_nodes) scaling
# STAGES uses, so node density matches its neighbors in that chain.
INTRO_END_STAGE = dict(n_nodes=10, cloud_radius=1.17, k_neighbors=2, node_radius=0.088, seed=11)

# One multiplier per STAGES[1:] transition: iterations start slow and
# deliberate, then compress -- the same shape as the takeoff curves this
# scene is about -- so the chain visibly accelerates into the finale
# rather than ticking along at a uniform clip.
SPEED_MULTIPLIERS = [2.2, 1.45, 0.95, 0.65]
FINAL_CODE_MULT = 0.45

# The 12 vertices and 20 triangular faces of a regular icosahedron (same
# coordinates manim's own Icosahedron mobject uses), kept here as plain
# data so the finale can be built as a bare glowing vertex/edge graph
# rather than manim's filled-face Polyhedron mesh.
_ICO_EDGE_LENGTH = 3.4
_UA = _ICO_EDGE_LENGTH * ((1 + 5**0.5) / 4)
_UB = _ICO_EDGE_LENGTH * 0.5
ICO_VERTICES = [
    np.array([0, _UB, _UA]),
    np.array([0, -_UB, _UA]),
    np.array([0, _UB, -_UA]),
    np.array([0, -_UB, -_UA]),
    np.array([_UB, _UA, 0]),
    np.array([_UB, -_UA, 0]),
    np.array([-_UB, _UA, 0]),
    np.array([-_UB, -_UA, 0]),
    np.array([_UA, 0, _UB]),
    np.array([_UA, 0, -_UB]),
    np.array([-_UA, 0, _UB]),
    np.array([-_UA, 0, -_UB]),
]
ICO_FACES = [
    (1, 8, 0), (1, 5, 7), (8, 5, 1), (7, 3, 5), (5, 9, 3),
    (8, 9, 5), (3, 2, 9), (9, 4, 2), (8, 4, 9), (0, 4, 8),
    (6, 4, 0), (6, 2, 4), (11, 2, 6), (3, 11, 2), (0, 6, 10),
    (10, 1, 0), (10, 7, 1), (11, 7, 3), (10, 11, 7), (10, 11, 6),
]
ICO_EDGES = sorted({tuple(sorted((f[i], f[(i + 1) % 3]))) for f in ICO_FACES for i in range(3)})


def _hex_to_rgb(hex_color):
    hex_color = hex_color.lstrip("#")
    return np.array([int(hex_color[i : i + 2], 16) for i in (0, 2, 4)], dtype=np.float64)


def make_backdrop():
    """A soft, centered glow behind everything, faking the radial gradient
    in the reference image on top of the flat navy background. Stacking
    concentric circles (even dozens, at tiny opacity each) always shows a
    visible ring at every boundary radius, since alpha-compositing discs
    is still a step function -- only a true per-pixel gradient, rendered
    as a raster image here, is actually smooth."""
    resolution = 480
    aspect = config.frame_width / config.frame_height
    width_px = int(resolution * aspect)
    ys, xs = np.mgrid[0:resolution, 0:width_px]
    cx, cy = width_px / 2, resolution / 2
    scale = config.frame_height / resolution
    dist = np.sqrt(((xs - cx) * scale) ** 2 + ((ys - cy) * scale) ** 2)

    max_dist = 9.5
    t = np.clip(1 - dist / max_dist, 0, 1) ** 1.6

    bg = _hex_to_rgb(BACKGROUND_COLOR)
    glow = _hex_to_rgb(BACKDROP_GLOW_COLOR)
    rgb = (bg + (glow - bg) * t[:, :, None]).astype(np.uint8)
    alpha = np.full((*t.shape, 1), 255, dtype=np.uint8)
    rgba = np.concatenate([rgb, alpha], axis=2)

    image = ImageMobject(rgba)
    image.stretch_to_fit_width(config.frame_width)
    image.stretch_to_fit_height(config.frame_height)
    image.move_to(ORIGIN)
    image.set_z_index(-10)
    return image


def add_drift(node):
    """A small, cheap idle wander around the node's own spot -- two sine
    waves with random per-node phase/frequency, so a whole net's worth of
    nodes don't wobble in lockstep. Stored on the node itself (.home,
    .drift_t) so the updater has no outside state to manage.

    .home is solved for so that home + offset_at(drift_t) equals the
    node's actual current position, rather than snapping .home to the
    current position and then adding a fresh random offset on top of it.
    The naive version doesn't just cause the "grows in, then teleports"
    glitch GrowFromCenter/FadeIn would otherwise produce (they suspend
    updaters for their own duration, so the first offset would only
    apply the instant they resume) -- it also visibly jumps an
    already-displayed node every time this is called, which matters
    because resume_drift() calls this again on nodes that are already
    fully visible and in place, right after a net finishes sliding into
    its slot."""
    node.drift_t = random.uniform(0, 2 * math.pi)
    phase_y = random.uniform(0, 2 * math.pi)
    freq_x = random.uniform(0.6, 1.3)
    freq_y = random.uniform(0.6, 1.3)

    def offset_at(t):
        return np.array(
            [
                DRIFT_AMPLITUDE * math.sin(t * freq_x),
                DRIFT_AMPLITUDE * math.cos(t * freq_y + phase_y),
                0,
            ]
        )

    node.home = node.get_center().copy() - offset_at(node.drift_t)

    def updater(mob, dt):
        mob.drift_t += dt
        mob.move_to(mob.home + offset_at(mob.drift_t))

    node.add_updater(updater)


def stop_drift(nodes_group, edges_group):
    """Fully remove every node's and edge's updater rather than merely
    suspending them -- .suspend_updating() turned out not to be enough:
    even suspended, an attached updater somehow still left a following
    net.animate.shift() silently unable to move the mobject (proven by
    an isolated test: identical shift, .suspend_updating() -> no
    movement, .clear_updaters() -> moves correctly). Call before any
    animation that repositions a net."""
    for node in nodes_group:
        node.clear_updaters()
    for edge in edges_group:
        edge.clear_updaters()


def resume_drift(nodes_group, edges_group):
    """Re-attach fresh updaters -- node drift anchored on wherever it
    just ended up, edges tracking their nodes again -- call right after
    any animation that moves a net (e.g. the slide into the left slot)."""
    for node in nodes_group:
        add_drift(node)
    for edge in edges_group:
        add_edge_tracking(edge)


def make_node(pos, radius, palette):
    """A glowing node: three soft, low-opacity halo rings of increasing
    brightness behind a bright core, faking a bloom without a real
    shader -- closer to the reference's blown-out node highlights than a
    single flat-filled circle."""
    core_c, mid_c, glow_c = palette
    outer = Circle(radius=radius * 4.2, stroke_width=0, fill_color=glow_c, fill_opacity=0.10)
    halo = Circle(radius=radius * 2.3, stroke_width=0, fill_color=glow_c, fill_opacity=0.28)
    mid = Circle(radius=radius * 1.15, stroke_color=mid_c, stroke_width=1, fill_color=mid_c, fill_opacity=0.9)
    core = Circle(radius=radius * 0.5, stroke_color=core_c, stroke_width=1, fill_color=core_c, fill_opacity=1)
    node = VGroup(outer, halo, mid, core)
    node.move_to(pos)
    add_drift(node)
    return node


def make_edge(node_a, node_b, color):
    """A glowing connection: a wide, faint pass beneath a thin, brighter
    one, so the line reads as lit rather than a flat stroke. Tracks the
    two nodes' live (drifting) centers every frame rather than a pair of
    fixed points, so it stays attached as its endpoints wander."""
    a, b = node_a.get_center(), node_b.get_center()
    wide = Line(a, b, stroke_color=color, stroke_width=4.5, stroke_opacity=0.12)
    thin = Line(a, b, stroke_color=color, stroke_width=1.3, stroke_opacity=0.6)
    edge = VGroup(wide, thin)
    edge.node_a = node_a
    edge.node_b = node_b
    add_edge_tracking(edge)
    return edge


def add_edge_tracking(edge):
    wide, thin = edge

    def updater(mob):
        a, b = edge.node_a.get_center(), edge.node_b.get_center()
        wide.put_start_and_end_on(a, b)
        thin.put_start_and_end_on(a, b)

    edge.add_updater(updater)


def make_glow_blob(local_points, edge_indices, cloud_radius, node_radius):
    """A real gaussian-blurred white silhouette of this net's own
    node/edge layout, rasterized once and placed underneath the crisp
    vector net. The per-node halo rings in make_node fake a bloom per
    node, but they stay as discrete overlapping rings; where several
    nodes sit close together this instead blurs into one continuous
    soft wash, which is the part a stack of flat-opacity circles can't
    reproduce. Static once built -- fine here since the blur radius
    dwarfs the ~0.035-unit idle node drift, so it doesn't need to track
    each node's individual jitter to still look attached."""
    span = cloud_radius * 2 + 1.2
    resolution = 220
    scale = resolution / span
    cx = cy = resolution / 2
    ys, xs = np.mgrid[0:resolution, 0:resolution]

    alpha = np.zeros((resolution, resolution), dtype=np.float64)

    node_px_r = max(node_radius * scale * 1.6, 3.0)
    for p in local_points:
        px, py = cx + p[0] * scale, cy - p[1] * scale
        dist = np.sqrt((xs - px) ** 2 + (ys - py) ** 2)
        alpha = np.maximum(alpha, np.clip(255 * (1 - dist / node_px_r), 0, 255))

    line_half_w = max(node_radius * scale * 0.5, 1.5)
    for i, j in edge_indices:
        a, b = local_points[i], local_points[j]
        ax, ay = cx + a[0] * scale, cy - a[1] * scale
        bx, by = cx + b[0] * scale, cy - b[1] * scale
        seg = np.array([bx - ax, by - ay])
        seg_len2 = seg @ seg + 1e-9
        t = np.clip(((xs - ax) * seg[0] + (ys - ay) * seg[1]) / seg_len2, 0, 1)
        proj_x, proj_y = ax + t * seg[0], ay + t * seg[1]
        dist = np.sqrt((xs - proj_x) ** 2 + (ys - proj_y) ** 2)
        alpha = np.maximum(alpha, np.clip(180 * (1 - dist / line_half_w), 0, 180))

    alpha = gaussian_filter(alpha, sigma=resolution * 0.025) * 0.9

    rgba = np.zeros((resolution, resolution, 4), dtype=np.uint8)
    rgba[..., 0:3] = 255
    rgba[..., 3] = np.clip(alpha, 0, 255).astype(np.uint8)

    image = ImageMobject(rgba)
    image.stretch_to_fit_width(span)
    image.stretch_to_fit_height(span)
    return image


def sample_cloud(n, radius, seed, min_dist_factor=0.32):
    """n points scattered inside a disc of the given radius, rejecting
    anything too close to a point already placed -- rough blue-noise
    spacing, so nodes read as an organic cluster rather than a grid."""
    rng = random.Random(seed)
    points = []
    tries = 0
    while len(points) < n and tries < 4000:
        tries += 1
        x, y = rng.uniform(-1, 1), rng.uniform(-1, 1)
        if x * x + y * y > 1:
            continue
        candidate = np.array([x * radius, y * radius, 0])
        if all(np.linalg.norm(candidate - p) > radius * min_dist_factor for p in points):
            points.append(candidate)
    return points


def nearest_neighbor_edges(points, k):
    """Each point connects to its k nearest neighbors (deduplicated) --
    a dense but locally-organized mesh, matching the reference nets'
    web-like (not fully-connected) look."""
    edges = set()
    for i, p in enumerate(points):
        order = sorted(range(len(points)), key=lambda j: np.linalg.norm(points[j] - p))
        count = 0
        for j in order:
            if j == i:
                continue
            edges.add(tuple(sorted((i, j))))
            count += 1
            if count >= k:
                break
    return edges


def build_net(n_nodes, cloud_radius, k_neighbors, node_radius, seed, center, palette, edge_color):
    local_points = sample_cloud(n_nodes, cloud_radius, seed)
    points = [center + p for p in local_points]

    nodes_list = [make_node(p, node_radius, palette) for p in points]
    nodes_group = VGroup(*nodes_list)
    edge_indices = nearest_neighbor_edges(local_points, k_neighbors)
    edges_group = VGroup(*[make_edge(nodes_list[i], nodes_list[j], edge_color) for i, j in edge_indices])

    glow = make_glow_blob(local_points, edge_indices, cloud_radius, node_radius)
    glow.move_to(center)

    # Group rather than VGroup: it needs to hold the glow's ImageMobject
    # alongside the VMobject-based edges/nodes, which VGroup rejects.
    # glow goes first so it paints as the bottom layer, underneath the
    # crisp edges and nodes -- a member of net itself, not tracked
    # separately, so the existing FadeOut/.animate.shift() calls on the
    # whole net move and fade it right along with everything else
    # without any extra code.
    net = Group(glow, edges_group, nodes_group)
    return net, nodes_group, edges_group, glow


def make_code_block(lines, center):
    # Small and dense rather than legible-sized: this is meant to read as
    # a wall of real-looking code scrolling behind the bracket, not as
    # captions the viewer is expected to actually read line by line.
    rows = VGroup(*[Text(line, font="Consolas", color=CODE_COLOR).scale(0.155) for line in lines])
    rows.arrange(DOWN, aligned_edge=LEFT, buff=0.075)
    rows.move_to(center)
    return rows


def make_bracket(target, buff=0.25, color=ARROW_COLOR, stroke_width=4):
    """A brace built from two straight segments and two quarter-circle
    arcs (line - arc - arc - line, meeting at a tip pointing at the
    incoming arrow) rather than manim's own Brace -- matching the
    angular, geometric bracket in assets/glow-net.png instead of the
    smoother font-derived curve Brace renders."""
    cy = target.get_center()[1]
    height = target.height + 0.3
    r = min(height * 0.16, 0.32)
    baseline_x = target.get_left()[0] - buff
    tip_x = baseline_x - r

    top = np.array([baseline_x, cy + height / 2, 0])
    top_line_end = np.array([baseline_x, cy + r, 0])
    tip = np.array([tip_x, cy, 0])
    bottom_line_start = np.array([baseline_x, cy - r, 0])
    bottom = np.array([baseline_x, cy - height / 2, 0])

    top_line = Line(top, top_line_end)
    top_arc = Arc(radius=r, start_angle=0, angle=-PI / 2, arc_center=np.array([tip_x, cy + r, 0]))
    bottom_arc = Arc(radius=r, start_angle=PI / 2, angle=-PI / 2, arc_center=np.array([tip_x, cy - r, 0]))
    bottom_line = Line(bottom_line_start, bottom)

    bracket = VGroup(top_line, top_arc, bottom_arc, bottom_line)
    bracket.set_stroke(color=color, width=stroke_width)
    return bracket


def make_arrow(start_x, end_x):
    # Thick shaft and a wide, chunky head -- a flat "block arrow" look
    # like the reference image's, rather than a thin stroked line with a
    # small triangle tip. Both stroke_width (shaft thickness) and
    # tip_length (arrowhead size) default to scaling down proportionally
    # to the arrow's own length, and this scene's arrows span a wide
    # range of lengths (net radius and code/brace width both change every
    # lap) -- left alone, that made some laps' arrows read as a proper
    # wide arrowhead and others as a stubby point stuck on an unchanged
    # thick rectangle, since the two didn't shrink at the same rate. Both
    # ratio ceilings are set well past any length this scene actually
    # produces, so tip_length and stroke_width stay pinned at their fixed
    # values and only the shaft's length -- the part actually free to
    # change -- responds to the available space.
    return Arrow(
        start=np.array([start_x, 0, 0]),
        end=np.array([end_x, 0, 0]),
        color=ARROW_COLOR,
        stroke_width=22,
        buff=0,
        tip_length=0.35,
        max_tip_length_to_length_ratio=5.0,
        max_stroke_width_to_length_ratio=100,
    )


def flow_arrows(left_net_edge, right_net_edge, brace, code):
    """The left (net -> code) and right (code -> net) arrows for one lap.
    Recomputed every time from the actual current geometry (net radii and
    code/brace width both change every cycle) so each arrow keeps equal
    padding on both of its own sides, rather than reusing fixed
    coordinates that only lined up for one particular net size."""
    brace_left_edge = brace.get_left()[0]
    code_right_edge = code.get_right()[0]
    left_margin = (brace_left_edge - left_net_edge) * 0.28
    right_margin = (right_net_edge - code_right_edge) * 0.28
    left_arrow = make_arrow(left_net_edge + left_margin, brace_left_edge - left_margin)
    right_arrow = make_arrow(code_right_edge + right_margin, right_net_edge - right_margin)
    return left_arrow, right_arrow


class RecursiveSelfImprovement(ThreeDScene):
    def hold(self, duration):
        """A pause where nothing is being grown, written, or shifted --
        just a beat to let the current frame sit. Clamped to a short
        stand-in length in fast-preview mode (see FULL_TIMING) since these
        contribute render time without any visual change to show for it."""
        self.wait(duration if FULL_TIMING else min(duration, IDLE_HOLD))

    def grow_in(self, nodes_group, edges_group, glow, run_time=1.1):
        # Nodes and edges both start at t=0 within this single self.play()
        # call -- rather than nodes finishing before edges even begin --
        # so the net reads as growing in all at once.
        #
        # suspend_mobject_updating=False on both the LaggedStart itself
        # and each animation inside it: Animation suspends a mobject's
        # updaters for its own duration by default, and LaggedStart only
        # resumes them once the whole group ends -- without the outer
        # flag too, it suspends every node's drift updater for the entire
        # grow-in regardless of the inner flags. Left suspended, each node
        # sits frozen at its pre-grow position the whole time, then jumps
        # the instant updating resumes and the updater computes a
        # now-stale drift offset for the first time -- the "grows in,
        # then teleports" glitch. Keeping drift live throughout avoids
        # that discontinuity entirely rather than patching around it.
        self.play(
            FadeIn(glow),
            LaggedStart(
                *[GrowFromCenter(n, suspend_mobject_updating=False) for n in nodes_group],
                lag_ratio=0.04,
                suspend_mobject_updating=False,
            ),
            LaggedStart(
                *[FadeIn(e, suspend_mobject_updating=False) for e in edges_group],
                lag_ratio=0.02,
                suspend_mobject_updating=False,
            ),
            run_time=run_time,
        )

    def construct(self):
        self.camera.background_color = BACKGROUND_COLOR
        self.set_camera_orientation(phi=0 * DEGREES, theta=-90 * DEGREES)

        # Mini-movie 1: a short, standalone taste of the same growth
        # pattern (net -> code -> bigger net) used throughout mini-movie
        # 2, then a beat of plain blank background as a hard cut between
        # the two, then mini-movie 2 -- the full chain -- runs unchanged.
        self.construct_intro()
        self.hold(1.0)
        self.construct_main()

    def construct_intro(self):
        """Mini-movie 1: one simple lap -- a small blue net grows in,
        writes an ordinary training step, and a bigger red net grows in
        response -- then everything clears for the blank cut before
        mini-movie 2. Blue and red rather than the main chain's own
        green-to-magenta progression, since this intro isn't actually
        part of that chain -- it's a self-contained preview of the same
        beat, not its first lap in disguise."""
        backdrop = make_backdrop()
        self.add_fixed_in_frame_mobjects(backdrop)

        m = SPEED_MULTIPLIERS[0]

        net0, nodes0, edges0, glow0 = build_net(
            center=np.array([LEFT_X, 0, 0]), palette=BLUE_PALETTE, edge_color=BLUE_EDGE, **STAGES[0]
        )
        self.grow_in(nodes0, edges0, glow0, run_time=1.1 * m)
        self.hold(0.4 * m)

        code = make_code_block(CODE_STAGE_1, np.array([MID_X, 0, 0]))
        brace = make_bracket(code, buff=0.25, color=ARROW_COLOR)
        left_arrow, right_arrow = flow_arrows(
            LEFT_X + STAGES[0]["cloud_radius"], RIGHT_X - INTRO_END_STAGE["cloud_radius"], brace, code
        )

        self.play(GrowArrow(left_arrow), run_time=0.5 * m)
        self.play(
            LaggedStart(
                *[Write(row) for row in code], lag_ratio=0.3, run_time=(0.5 + 0.16 * len(CODE_STAGE_1)) * m
            ),
            GrowFromCenter(brace, run_time=0.35 * m),
        )
        self.hold(0.4 * m)
        self.play(GrowArrow(right_arrow), run_time=0.5 * m)

        net1, nodes1, edges1, glow1 = build_net(
            center=np.array([RIGHT_X, 0, 0]), palette=RED_PALETTE, edge_color=RED_EDGE, **INTRO_END_STAGE
        )
        self.grow_in(nodes1, edges1, glow1, run_time=max(1.3 * m, 0.5))
        self.hold(0.5 * m)

        self.play(
            FadeOut(net0),
            FadeOut(net1),
            FadeOut(code),
            FadeOut(brace),
            FadeOut(left_arrow),
            FadeOut(right_arrow),
            FadeOut(backdrop),
            run_time=max(1.0 * m, 0.45),
        )

    def construct_main(self):
        # Fixed in frame rather than plain self.add(): a flat, ORIGIN-
        # centered circle is part of the 3D scene like anything else, so
        # once the camera tilts for the icosahedron reveal it would be
        # seen edge-on and warp into an ellipse instead of staying a
        # screen-space vignette. But fixed-in-frame content composites
        # incorrectly (paints over, not under, everything else) the
        # moment the camera actually moves -- a Manim/Cairo limitation,
        # not a visual choice -- so it must be faded out no later than
        # the start of the first camera move (see the FadeOut(backdrop)
        # below) rather than left in place through the finale.
        backdrop = make_backdrop()
        self.add_fixed_in_frame_mobjects(backdrop)

        # Net 0 spawns on the left, green for its whole lifetime -- grown
        # in at the same unhurried pace as the first loop iteration below.
        current_net, current_nodes, current_edges, current_glow = build_net(
            center=np.array([LEFT_X, 0, 0]), palette=NET_PALETTES[0], edge_color=NET_EDGE_COLORS[0], **STAGES[0]
        )
        self.grow_in(current_nodes, current_edges, current_glow, run_time=1.1 * SPEED_MULTIPLIERS[0])
        self.hold(0.4 * SPEED_MULTIPLIERS[0])
        current_radius = STAGES[0]["cloud_radius"]

        # Left net -> left arrow -> code typing in behind a brace -> right
        # arrow -> a bigger net (its own fixed color) grows on the right
        # -> the left net, code, and arrows vanish while the right net
        # slides into the left slot, becoming "current" for the next lap.
        # Every beat in the lap is scaled by that lap's own multiplier, so
        # early laps linger and later laps snap by increasingly fast.
        for i, (stage, code_lines, m) in enumerate(zip(STAGES[1:], CODE_STAGES, SPEED_MULTIPLIERS), start=1):
            code = make_code_block(code_lines, np.array([MID_X, 0, 0]))
            brace = make_bracket(code, buff=0.25, color=ARROW_COLOR)
            left_arrow, right_arrow = flow_arrows(
                LEFT_X + current_radius, RIGHT_X - stage["cloud_radius"], brace, code
            )

            self.play(GrowArrow(left_arrow), run_time=0.5 * m)
            self.play(
                LaggedStart(
                    *[Write(row) for row in code], lag_ratio=0.3, run_time=(0.5 + 0.16 * len(code_lines)) * m
                ),
                GrowFromCenter(brace, run_time=0.35 * m),
            )
            self.hold(0.4 * m)

            self.play(GrowArrow(right_arrow), run_time=0.5 * m)

            next_net, next_nodes, next_edges, next_glow = build_net(
                center=np.array([RIGHT_X, 0, 0]), palette=NET_PALETTES[i], edge_color=NET_EDGE_COLORS[i], **stage
            )
            self.grow_in(next_nodes, next_edges, next_glow, run_time=max(1.3 * m, 0.5))
            self.hold(0.5 * m)

            stop_drift(next_nodes, next_edges)
            self.play(
                FadeOut(current_net),
                FadeOut(code),
                FadeOut(brace),
                FadeOut(left_arrow),
                FadeOut(right_arrow),
                next_net.animate.shift(np.array([LEFT_X - RIGHT_X, 0, 0])),
                run_time=max(1.0 * m, 0.45),
            )
            resume_drift(next_nodes, next_edges)
            current_net, current_nodes, current_edges, current_glow = next_net, next_nodes, next_edges, next_glow
            current_radius = stage["cloud_radius"]

        self.hold(0.4 * SPEED_MULTIPLIERS[-1])

        # Final lap: the magenta net writes its own code just like the
        # others -- fastest of all, the takeoff now nearly instantaneous --
        # but what grows in its place isn't another flat net -- it's the
        # icosahedron, red, still viewed face-on so it grows in at the
        # same right-hand spot the others did.
        code = make_code_block(CODE_STAGE_FINAL, np.array([MID_X, 0, 0]))
        brace = make_bracket(code, buff=0.25, color=ARROW_COLOR)
        ico_footprint_radius = 2.0
        left_arrow, right_arrow = flow_arrows(
            LEFT_X + current_radius, RIGHT_X - ico_footprint_radius, brace, code
        )

        self.play(GrowArrow(left_arrow), run_time=0.5 * FINAL_CODE_MULT)
        self.play(
            LaggedStart(
                *[Write(row) for row in code],
                lag_ratio=0.3,
                run_time=(0.5 + 0.16 * len(CODE_STAGE_FINAL)) * FINAL_CODE_MULT,
            ),
            GrowFromCenter(brace, run_time=0.35 * FINAL_CODE_MULT),
        )
        self.hold(0.4 * FINAL_CODE_MULT)
        self.play(GrowArrow(right_arrow), run_time=0.5 * FINAL_CODE_MULT)

        if INCLUDE_FINALE:
            ico_offset = np.array([RIGHT_X, 0, 0])
            ico_vertices = VGroup(
                *[
                    Dot3D(point=ico_offset + v, radius=0.11, color=RED_CORE, resolution=(12, 12))
                    for v in ICO_VERTICES
                ]
            )
            ico_glow = VGroup(
                *[Dot3D(point=ico_offset + v, radius=0.22, color=RED_GLOW, resolution=(8, 8)) for v in ICO_VERTICES]
            )
            ico_glow.set_opacity(0.35)
            ico_edges = VGroup(
                *[
                    Line3D(
                        ico_offset + ICO_VERTICES[i], ico_offset + ICO_VERTICES[j], thickness=0.025, color=RED_EDGE
                    )
                    for i, j in ICO_EDGES
                ]
            )

            self.play(
                LaggedStart(*[GrowFromCenter(v) for v in ico_glow], lag_ratio=0.05),
                LaggedStart(*[GrowFromCenter(v) for v in ico_vertices], lag_ratio=0.05),
                LaggedStart(*[FadeIn(e) for e in ico_edges], lag_ratio=0.03),
                run_time=1.8,
            )
            self.hold(2.0)

            # The last flat net and its code fade away while the icosahedron
            # is brought to center -- then the camera tilts to reveal it was
            # never flat to begin with, and it spins as the closing shot.
            ico_group = VGroup(ico_glow, ico_vertices, ico_edges)
            self.play(
                FadeOut(current_net),
                FadeOut(code),
                FadeOut(brace),
                FadeOut(left_arrow),
                FadeOut(right_arrow),
                FadeOut(backdrop),
                ico_group.animate.shift(np.array([-RIGHT_X, 0, 0])),
                run_time=1.2,
            )

            self.move_camera(phi=65 * DEGREES, theta=-50 * DEGREES, run_time=2.2)

            self.begin_ambient_camera_rotation(rate=0.18)
            self.wait(14.0 if FULL_TIMING else SPIN_HOLD)
            self.stop_ambient_camera_rotation()

            self.play(FadeOut(ico_group), run_time=1.5)
            self.hold(0.5)
        else:
            # Finale skipped (INCLUDE_FINALE=0, the default): end on a
            # clean fade of the last flat net and its code rather than
            # cutting off mid-frame, so this stays a watchable clip on its
            # own while the expensive 3D reveal is off.
            self.hold(0.4 * FINAL_CODE_MULT)
            self.play(
                FadeOut(current_net),
                FadeOut(code),
                FadeOut(brace),
                FadeOut(left_arrow),
                FadeOut(right_arrow),
                FadeOut(backdrop),
                run_time=1.0,
            )
