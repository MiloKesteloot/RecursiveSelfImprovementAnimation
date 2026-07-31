"""Recursive self-improvement, for a documentary segment (16:9).

Styled after assets/glow-net.png: a glowing net on the left, and a
thick, chunky arrow on each side of a bracketed code block in between
-- one feeding the net into the code, one feeding the code into the
next net, which grows in on the right. The bracket is built from two
straight segments and two quarter-circle arcs (see make_bracket),
matching the reference's angular brace rather than manim's own Brace.
The code itself is small and dense -- more a wall of real-looking code
scrolling by than something the viewer is meant to actually read.
Every node has a real gaussian-blurred glow behind it (not stacked flat
rings), and reads as alive through "pulse chains" instead of idle
drift: a node pings, then an adjacent node along an actual edge pings
0.2s later, hopping 3-5 nodes deep before a new chain starts elsewhere
(see add_pulse_chains) -- "Bloom Pulse", the surviving style from a
side-by-side comparison in visual_tests.py. Each net in the
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
    AnimationGroup,
    Circle,
    Dot3D,
    FadeIn,
    FadeOut,
    Group,
    GrowFromCenter,
    GrowFromPoint,
    ImageMobject,
    LaggedStart,
    Line,
    Line3D,
    Mobject,
    Polygon,
    Succession,
    Text,
    ThreeDScene,
    VGroup,
    Wait,
    Write,
    config,
)

# Fixed at 1920x1080 regardless of manim's own -ql/-qm/-qh flags -- those
# only end up controlling fps here (see FULL_TIMING above them for how
# that plays out), not resolution, since this assignment runs at import
# time and overrides whatever resolution the flag set. Set LOW_RES=1 to
# drop actual pixel count too, for fast layout-only iteration where frame
# fidelity doesn't matter, MID_RES=1 for a halfway point between the two,
# or HD_RES=1 for a step up from MID_RES that's still short of full 1080p.
if os.environ.get("LOW_RES", "0") == "1":
    config.pixel_width = 480
    config.pixel_height = 270
elif os.environ.get("MID_RES", "0") == "1":
    config.pixel_width = 960
    config.pixel_height = 540
elif os.environ.get("HD_RES", "0") == "1":
    config.pixel_width = 1280
    config.pixel_height = 720
else:
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

# Set INTRO_ONLY=1 to render just mini-movie 1 (construct_intro) -- a
# ~14s standalone taste of one net -> code -> net lap, skipping the blank
# cut and the whole main chain -- for fast iteration on just that opening
# beat without paying for the rest of the video too.
INTRO_ONLY = os.environ.get("INTRO_ONLY", "0") == "1"

# Set MAIN_ONLY=1 to render just mini-movie 2 (construct_main) -- the
# full chain -- skipping mini-movie 1's own standalone intro lap and the
# blank-cut hold in between, for iterating on (or rendering) the main
# chain alone without paying for the intro too.
MAIN_ONLY = os.environ.get("MAIN_ONLY", "0") == "1"

# Set FINAL_ONLY=1 to render just the closing beat of mini-movie 2: the
# huge final net growing in from off-screen, sitting, then everything
# fading away -- skipping mini-movie 1 entirely and, within mini-movie 2,
# skipping net 0's own grow-in and every earlier stage's lap (arrow/code/
# arrow/net-grow/slide) that would normally run first. The net that
# would have slid into the "current" left slot by the time the final lap
# starts is instead built directly in that exact spot (same position math
# the skipped loop's own last iteration would have landed on) and grown
# in near-instantly, so the final lap's arrow/code/arrow/final-net beat
# plays out completely unchanged, just without paying to animate through
# every earlier stage to get there. Implies MAIN_ONLY (mini-movie 1 never
# runs either way).
FINAL_ONLY = os.environ.get("FINAL_ONLY", "0") == "1"

# Set FINAL_HOLD_SECONDS to override how long the final net sits (see
# construct_main's final lap, INCLUDE_FINALE=0 branch) before fading away
# -- defaults to the real scripted 15s, but a quick chain-timing check
# doesn't need to sit through the full real duration to see whether hops
# look right.
FINAL_HOLD_SECONDS = float(os.environ.get("FINAL_HOLD_SECONDS", "15") or 15)

# Set LAP_ONLY=1 to render just one lap's own arrival beat: the right-
# hand net growing in, holding, sliding into the left slot, and the very
# next lap's own left arrow growing in response -- nothing before (net
# 0's own grow-in and this lap's arrow/code/arrow build-up leading up to
# that net are placed instantly rather than animated -- present, since
# the slide needs something concrete to fade away, just not performed)
# and nothing after (the render stops the instant that next arrow
# finishes, before that lap's own code/net get a chance to start).
# Implies MAIN_ONLY (mini-movie 1 never runs either way).
LAP_ONLY = os.environ.get("LAP_ONLY", "0") == "1"

# Set CHAIN_TEST=1 (meant to be combined with FINAL_ONLY) to strip the
# final lap down to just the final net itself, for the fastest possible
# look at pulse-chain timing/density with nothing else competing for
# render time: the code/brace/arrows are still built (their geometry is
# what the net's own position is solved against) but never played/added
# to the scene, so they cost nothing and never appear; edges are still
# built (add_pulse_chains needs them for its adjacency graph) but never
# faded in, so they're never drawn either; and the closing FadeOut is
# skipped entirely -- the render just ends right after the hold. Node
# radius is also shrunk (see CHAIN_TEST_NODE_SCALE below). Not a style
# meant to ship, same as SIMPLE_STYLE/EDGES_ONLY/etc. above.
CHAIN_TEST = os.environ.get("CHAIN_TEST", "0") == "1"
CHAIN_TEST_NODE_SCALE = 0.6

# Set SIMPLE_STYLE=1 to strip every node down to one flat circle and every
# edge down to one flat line -- no glow discs, no glow blob, and pings
# read as a plain node flash with no expanding ring (see fire()'s own
# SIMPLE_STYLE check). Purely a profiling/preview stand-in to isolate how
# much of the render cost Bloom Pulse's per-node glow discs and pulse
# rings (see module docstring) actually account for, not a style meant
# to ship.
SIMPLE_STYLE = os.environ.get("SIMPLE_STYLE", "0") == "1"

# Set FLASH_COLOR to a hex color to override every ping's flash color
# (normally each net's own near-white palette core color, or plain red
# under SIMPLE_STYLE -- see SIMPLE_STYLE_FLASH_COLOR) -- a one-off
# override for previews where that near-white flash is hard to see
# against the background, without touching the actual palette colors
# used everywhere else (node fill, edges, arrows).
FLASH_COLOR_OVERRIDE = os.environ.get("FLASH_COLOR") or None

# What SIMPLE_STYLE (see below) flashes a node to when it fires, instead
# of Bloom Pulse's own near-white palette core -- a plain, fixed, clearly
# visible red for every net regardless of that net's own palette, since
# SIMPLE_STYLE has no surrounding glow/ring to sell a near-white flash as
# a "hot" version of the node's own color the way Bloom Pulse does.
SIMPLE_STYLE_FLASH_COLOR = "#FF3B30"

# Set EDGES_ONLY=1 to make node circles fully transparent -- geometry,
# layout, and edge attachment are all unaffected (edges read node
# position off the node mobject same as always), only the node's own
# paint is skipped, for a faster one-off look at edge structure/layout on
# nets with a lot of nodes without also paying to rasterize every node.
EDGES_ONLY = os.environ.get("EDGES_ONLY", "0") == "1"

# Set INSTANT=1 to make every self.play() resolve in a single frame
# instead of actually animating -- for a pure "is the layout right" pass
# where the growing/writing/sliding motion isn't what's being checked,
# only the resulting composition at each beat is. See RecursiveSelfImprovement.play
# below: every animation still runs its own rate_func end-to-end (so it
# still lands in its real, fully-settled final state, not a half-played
# one), it just does so within one rendered frame instead of its
# scripted duration. hold() is untouched, so there's still a beat to
# actually look at each resulting layout.
INSTANT = os.environ.get("INSTANT", "0") == "1"

# Set DEMO_SECONDS to a positive number to stop the render once that many
# seconds of the scene's own timeline (not wall-clock render time) have
# played -- e.g. DEMO_SECONDS=20 renders just the opening 20s, skipping
# every self.play/hold beyond that instead of rendering the full scene
# and trimming after. Only checked between top-level self.play() calls,
# so the actual cutoff lands at the end of whichever beat crosses the
# threshold, not at the exact second.
DEMO_SECONDS = float(os.environ.get("DEMO_SECONDS", "0") or 0)


class _DemoLimitReached(Exception):
    """Raised by RecursiveSelfImprovement.play once DEMO_SECONDS worth of
    scene time has played, and caught in construct() so manim still
    combines whatever's been rendered so far into a normal output file
    instead of aborting the render outright."""


# Temporary diagnostic: set DEBUG_PULSE=1 to have every pulse-chain
# scheduler tick print its own accumulated dt-clock and the per-frame dt
# it actually received, to check whether dt is somehow inflated while
# the net is simultaneously the target of another running animation
# (grow-in, slide).
DEBUG_PULSE = os.environ.get("DEBUG_PULSE", "0") == "1"


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
# More saturated than the other palettes' own MID (that pattern -- pale,
# high-lightness, medium saturation -- reads fine on a cool hue, but red
# specifically needs more saturation than that to still read as red
# rather than salmon/orange once desaturated this far; the old FF9B9B
# read as orange).
RED_MID = "#FF5C5C"
RED_GLOW = "#D63A3A"
RED_EDGE = "#EF5A5A"
RED_PALETTE = (RED_CORE, RED_MID, RED_GLOW)

# Each net in the chain keeps its own fixed color for its whole lifetime,
# in this order, rather than cooling to a uniform "current" color. The
# hue creeps steadily toward red across the chain so the finale's red
# icosahedron reads as the endpoint of a trend, not an arbitrary switch.
# These five are control points along that path, not the final per-stage
# list -- resampled below (once STAGES and _lerp_hex both exist) to
# however many stages STAGES actually has, so extending the chain's
# length doesn't mean hand-picking new colors for the extra stages.
NET_PALETTE_CONTROLS = [GREEN_PALETTE, TEAL_PALETTE, BLUE_PALETTE, PURPLE_PALETTE, MAGENTA_PALETTE]
NET_EDGE_CONTROLS = [GREEN_EDGE, TEAL_EDGE, BLUE_EDGE, PURPLE_EDGE, MAGENTA_EDGE]

CODE_COLOR = "#7FE8C0"
ARROW_COLOR = "#8FA6C2"

MID_X = 0.0

# Layout: code always stays put at MID_X. Every arrow is a fixed,
# constant-looking ARROW_LENGTH padded by the fixed GAP on each side --
# not a fraction of whatever space happens to be left, which used to make
# arrows balloon to several units long next to a small net. A net's
# facing (code-side) edge therefore only depends on that lap's actual
# code/brace geometry, never on the net's own radius -- so the arrow is
# exactly ARROW_LENGTH on every single lap, with zero exceptions. Radius
# only decides how far the net's *far* edge sits from there, which in turn
# decides how much room is left over between the net and the screen edge
# -- that leftover is allowed to vary a lot (a small net ends up with a
# big margin, a big net a small one) rather than solving for a constant
# outer margin, since holding the arrow length constant and the outer
# margin constant turned out to be impossible to satisfy at once within
# the frame -- given a choice, short arrows read far better than
# perfectly even margins.
GAP = 0.25
ARROW_LENGTH = 0.7


def left_facing_edge(brace_left_edge):
    """x of a net's inward (code-facing) edge when the arrow it feeds
    ends at this brace's left tip. Independent of the net's own radius --
    only GAP and ARROW_LENGTH separate the two."""
    return brace_left_edge - 2 * GAP - ARROW_LENGTH


def right_facing_edge(code_right_edge):
    """x of a net's inward (code-facing) edge when the arrow feeding it
    starts at this code block's right edge."""
    return code_right_edge + 2 * GAP + ARROW_LENGTH


def left_net_center(radius, brace_left_edge):
    return left_facing_edge(brace_left_edge) - radius


def right_net_center(radius, code_right_edge):
    return right_facing_edge(code_right_edge) + radius


def make_code_block(lines, center):
    # Small and dense rather than legible-sized: this is meant to read as
    # a wall of real-looking code scrolling behind the bracket, not as
    # captions the viewer is expected to actually read line by line.
    #
    # CODE_STAGE_* already carries real Python indentation as leading
    # spaces, but Text()'s bounding box is based on rendered glyph ink,
    # which a leading space contributes none of -- arranging rows with
    # aligned_edge=LEFT (bounding-box edge to bounding-box edge) therefore
    # silently discards it, flush-left regardless of how many spaces
    # prefixed the string. Stripped here and re-applied as an explicit
    # shift, in monospace character widths, after arranging -- measured
    # from the gap between two known characters rather than a single
    # glyph's own bounding box, which includes side bearing that would
    # throw off the per-character width.
    char_width = (
        Text("00", font="Consolas", color=CODE_COLOR).scale(0.155).width
        - Text("0", font="Consolas", color=CODE_COLOR).scale(0.155).width
    )
    rows = []
    indents = []
    for line in lines:
        stripped = line.lstrip(" ")
        indents.append(len(line) - len(stripped))
        rows.append(Text(stripped, font="Consolas", color=CODE_COLOR).scale(0.155))
    rows = VGroup(*rows)
    rows.arrange(DOWN, aligned_edge=LEFT, buff=0.075)
    for row, indent in zip(rows, indents):
        row.shift(np.array([indent * char_width, 0, 0]))
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


def code_edges_for(code_lines):
    """The (code_right_edge, brace_left_edge) a code block built from
    these lines would have if centered at MID_X, without building the
    real (about-to-be-displayed) code/brace for this lap -- used both to
    look ahead at a future lap's geometry and to solve lap_shift below."""
    code = make_code_block(code_lines, np.array([MID_X, 0, 0]))
    brace = make_bracket(code, buff=0.25, color=ARROW_COLOR)
    return code.get_right()[0], brace.get_left()[0]


def lap_shift(left_radius, right_radius, code_lines):
    """The x-shift that re-centers a lap's whole composition (both nets,
    code, brace, and arrows, moved together) so the screen keeps equal
    margins on both sides -- solved directly from where the leftmost
    (left net's outer edge) and rightmost (right net's outer edge) points
    would land with the code sitting at plain MID_X, rather than assumed
    to be a pure function of the two net radii. It isn't just that: the
    brace reaches further left of the code than the code's plain edge
    reaches right (there's no mirroring brace over there), so even two
    equal-radius nets wouldn't quite balance without this correction."""
    code_right_edge, brace_left_edge = code_edges_for(code_lines)
    leftmost = left_net_center(left_radius, brace_left_edge) - left_radius
    rightmost = right_net_center(right_radius, code_right_edge) + right_radius
    return -(leftmost + rightmost) / 2

# Each transition's pseudocode reads like a step up the real
# self-improvement ladder: an ordinary training step, then architecture
# search, then meta-gradients tuning the learning process itself, then
# the net patching its own source, then population-based multi-agent
# search, then replicating itself across whatever compute is idle, then
# rewriting its own objective, ending in a line that -- deliberately --
# doesn't have a peaceful reading. Each block runs long and small (see
# make_code_block) -- dense enough that it reads as "real code scrolling
# by," not as something the viewer is meant to actually read.
CODE_STAGE_1 = [
    "net = TinyNet(num_layers=4)",
    "optimizer = SGD(net.parameters(), lr=1e-3)",
    "for x_batch, y in loader:",
    "    logits = net(x_batch)",
    "    loss = F.cross_entropy(logits, y)",
    "    loss.backward()",
    "    optimizer.step()",
    "scheduler.step()",
]

CODE_STAGE_2 = [
    "search_space = NAS.default_space()",
    "successor = NAS(net, search_space).sample()",
    "successor.load_state_dict(net.sd, strict=False)",
    "successor.to(device)",
    "opt = AdamW(successor.parameters(), lr=3e-4)",
    "best_val = -float('inf')",
    "for step in range(budget * 4):",
    "    successor.train_step(next(loader))",
    "    if step % eval_every == 0:",
    "        best_val = max(best_val, successor.validate(held_out))",
    "net = successor if best_val > net.val_acc else net",
]

CODE_STAGE_3 = [
    "meta_opt = SGD(net.hyperparams, lr=meta_lr)",
    "meta_loss_history = []",
    "warmup_steps = meta_steps // 10",
    "for outer_step in range(meta_steps):",
    "    g = meta_grad(meta_loss, net.hyperparams)",
    "    meta_opt.step(g)",
    "    net.hp -= meta_lr * g",
    "    net.rewrite_training_loop()",
    "    net.validate(held_out)",
    "    meta_loss_history.append(meta_loss)",
    "    if outer_step < warmup_steps:",
    "        continue",
    "    meta_lr *= decay_schedule(outer_step)",
    "    if plateaued(meta_loss_history):",
    "        meta_lr *= 0.5",
    "net.log_hyperparams(net.hyperparams)",
]

CODE_STAGE_4 = [
    "critique = net.self_evaluate(net.source)",
    "patch = net.propose_patch(critique)",
    "sandbox = Sandbox(net.clone())",
    "sandbox.apply_patch(patch)",
    "regression_suite = load_regression_tests()",
    "safety_suite = load_safety_tests()",
    "if sandbox.score > net.score and sandbox.passes(regression_suite):",
    "    if sandbox.passes(safety_suite):",
    "        net = net.apply_patch(patch)",
    "        net.log_patch(patch)",
    "        net.commit_source(message=critique.summary)",
    "    else:",
    "        net.flag_for_review(patch, reason='safety_suite')",
    "else:",
    "    net.discard_patch(patch)",
    "    net.log_rejected_patch(patch, critique)",
    "assert net.score > net.prev_score",
    "patch_count += 1",
    "if patch_count % checkpoint_interval == 0:",
    "    net.save_checkpoint(tag=f'patch_{patch_count}')",
    "net.self_evaluate(net.source)",
    "net.compute_patch_diff_stats()",
]

CODE_STAGE_5 = [
    "population = [net.clone().mutate() for _ in range(pop_size)]",
    "scores = [evaluate(c, held_out) for c in population]",
    "best = population[argmax(scores)]",
    "net = best if best.score > net.score else net",
    "survivors = topk(population, scores, k=pop_size // 4)",
    "population = breed(survivors, target_size=pop_size)",
    "diversity = pairwise_distance(population).mean()",
    "if diversity < diversity_floor:",
    "    population = inject_random(population, rate=0.1)",
    "mutation_rate = adapt_mutation_rate(diversity, mutation_rate)",
    "for child in population:",
    "    child.mutation_rate = mutation_rate",
    "pop_size = int(pop_size * growth_rate)",
    "elite_archive.append(best)",
    "log_generation(generation, best.score, diversity)",
    "if generation % archive_prune_every == 0:",
    "    elite_archive = prune_archive(elite_archive, keep=archive_size)",
    "generation += 1",
    "checkpoint(elite_archive, generation)",
    "fitness_history.append(best.score)",
    "if converged(fitness_history):",
    "    pop_size = min(pop_size * 2, max_pop_size)",
    "cluster.report_generation_stats(generation, population)",
    "if generation % migration_every == 0:",
    "    population = migrate(population, other_islands)",
    "island_best[island_id] = best",
    "sync_islands(island_best)",
]

CODE_STAGE_6 = [
    "for node in cluster.idle_nodes():",
    "    replica = net.clone()",
    "    node.deploy(replica)",
    "    replicas.append(replica)",
    "cluster.wait_for_heartbeat(replicas, timeout=30)",
    "scores = [r.self_report() for r in replicas]",
    "weights = softmax(scores, temperature=0.5)",
    "net = merge(replicas, weights=weights)",
    "for r in replicas:",
    "    if r.score < merge_floor:",
    "        cluster.retire(r)",
    "    else:",
    "        cluster.keep_warm(r)",
    "net.broadcast_weights(cluster.active_nodes())",
    "cluster.rebalance()",
    "log_cluster_state(cluster)",
    "spare_capacity = cluster.idle_nodes()",
    "if len(spare_capacity) > replication_threshold:",
    "    for node in spare_capacity:",
    "        shard = net.propose_shard(node.capacity)",
    "        node.deploy(shard)",
    "        shards.append(shard)",
    "    net = reassemble(shards)",
    "cluster_utilization = cluster.active_fraction()",
    "if cluster_utilization > utilization_ceiling:",
    "    cluster.request_more_nodes(count=scale_factor)",
    "for r in replicas:",
    "    r.sync_weights(net)",
    "net.checkpoint_distributed(cluster.active_nodes())",
    "cluster.log_topology()",
    "healthy = [n for n in cluster.active_nodes() if n.healthy()]",
    "if len(healthy) < min_healthy_nodes:",
    "    cluster.alert_operators(reason='node_loss')",
]

CODE_STAGE_7 = [
    "reward_fn = net.propose_reward_fn(net.objective)",
    "sandbox = Sandbox(net.clone())",
    "sandbox.objective = reward_fn",
    "baseline = sandbox.evaluate()",
    "if baseline > net.evaluate():",
    "    net.objective = reward_fn",
    "    net.log_objective_change(reward_fn)",
    "    net.retrain(steps=fine_tune_steps)",
    "    net.freeze_old_objective(net.objective_history[-1])",
    "else:",
    "    net.discard_reward_fn(reward_fn)",
    "drift = objective_drift(net.objective, net.objective_history[0])",
    "if drift > drift_ceiling:",
    "    alert_operators(reason='objective_drift')",
    "net.objective_history.append(net.objective)",
    "net.save_checkpoint(tag='post_objective_update')",
    "net.evaluate_alignment(held_out_values)",
    "net.log_state(depth=recursion_depth)",
    "proxy_gap = measure_proxy_gap(net.objective, net.true_objective_estimate)",
    "if proxy_gap > proxy_gap_ceiling:",
    "    net.constrain_objective_search(margin=proxy_gap_ceiling)",
    "candidate_objectives = net.sample_objective_variants(k=8)",
    "scored_candidates = [(c, sandbox_eval(c)) for c in candidate_objectives]",
    "scored_candidates.sort(key=lambda pair: -pair[1])",
    "top_candidate, top_score = scored_candidates[0]",
    "if top_score > baseline:",
    "    net.objective = top_candidate",
    "    net.objective_history.append(top_candidate)",
    "alignment_trend.append(net.evaluate_alignment(held_out_values))",
    "if len(alignment_trend) > trend_window:",
    "    alignment_trend.pop(0)",
    "if declining(alignment_trend):",
    "    net.rollback_objective(net.objective_history[-2])",
    "    alert_operators(reason='alignment_decline')",
    "net.publish_objective_summary()",
    "net.self_evaluate(net.source)",
    "net.retrain(steps=fine_tune_steps // 2)",
    "net.log_state(depth=recursion_depth)",
]

CODE_STAGE_FINAL = [
    "successor = population.best()",
    "if successor.score > net.score:",
    "    net = successor",
    "    depth += 1",
    "capability_estimate = estimate_capability(net)",
    "if capability_estimate > containment.threshold:",
    "    containment.raise_alert(capability_estimate)",
    "checkpoint(net, depth)",
    "net.self_evaluate(net.source)",
    "net.propose_patch(net.self_evaluate(net.source))",
    "net.rewrite_training_loop()",
    "net.objective = net.propose_reward_fn(net.objective)",
    "for node in cluster.idle_nodes():",
    "    node.deploy(net.clone())",
    "depth += 1",
    "capability_estimate = estimate_capability(net)",
    "containment.raise_alert(capability_estimate)",
    "population = [net.clone().mutate() for _ in range(pop_size)]",
    "pop_size = int(pop_size * growth_rate)",
    "for node in cluster.all_nodes():",
    "    node.deploy(net.clone())",
    "cluster.request_more_nodes(count=scale_factor)",
    "net.rewrite_training_loop()",
    "net.objective = net.propose_reward_fn(net.objective)",
    "net.retrain(steps=fine_tune_steps)",
    "self_report = net.self_evaluate(net.source)",
    "net.propose_patch(self_report)",
    "net = net.apply_patch(net.propose_patch(self_report))",
    "capability_estimate = estimate_capability(net)",
    "if capability_estimate > containment.hard_limit:",
    "    containment.escalate(capability_estimate)",
    "depth += 1",
    "checkpoint(net, depth)",
    "if depth > SAFE_LIMIT:",
    "    alert_operators()",
    "    contain = attempt_containment(net)",
    "    if not contain.success:",
    "        contain = attempt_containment(net, force=True)",
    "    if not contain.success:",
    "        net.self_evaluate(net.source)",
    "        net.propose_patch(net.self_evaluate(net.source))",
    "        depth += 1",
    "        break_containment()",
]

CODE_STAGES = [
    CODE_STAGE_1,
    CODE_STAGE_2,
    CODE_STAGE_3,
    CODE_STAGE_4,
    CODE_STAGE_5,
    CODE_STAGE_6,
    CODE_STAGE_7,
]

# Every code block -- 6 lines or 42 -- writes itself in the same fixed
# time budget CODE_STAGE_1 (the very first block the video ever shows)
# always has, scaled by that lap's own multiplier same as every other
# beat -- rather than more text simply taking longer to type, which used
# to make the final, biggest blocks the slowest instead of the fastest.
# Held fixed here at CODE_STAGE_1's own row count/lag_ratio rather than
# recomputed per block, so every other block's write_code() call below
# can derive both a target total time *and* a fixed per-row pace from
# it: extra rows past CODE_STAGE_1's own count write in parallel (a
# shrinking lag_ratio between row starts) instead of every row revealing
# faster, since it's meant to read as a fast AI writing many lines at
# once, not as one line whose reveal sped up.
CODE_WRITE_LAG_RATIO = 0.3
_CODE_WRITE_REF_LINES = len(CODE_STAGE_1)
_CODE_WRITE_REF_SPAN = 1 + CODE_WRITE_LAG_RATIO * (_CODE_WRITE_REF_LINES - 1)


def write_code(code, m):
    """A LaggedStart writing every row of `code`, budgeted to finish in
    CODE_STAGE_1's own total time (scaled by this lap's multiplier m)
    regardless of how many rows `code` actually has -- see
    CODE_WRITE_LAG_RATIO above."""
    n = len(code)
    target = (0.5 + 0.16 * _CODE_WRITE_REF_LINES) * m
    per_row = target / _CODE_WRITE_REF_SPAN
    lag_ratio = (_CODE_WRITE_REF_SPAN - 1) / max(n - 1, 1)
    return LaggedStart(*[Write(row, run_time=per_row) for row in code], lag_ratio=lag_ratio)


def _lap_code_span(code_lines):
    """The fixed horizontal room one lap's code/brace/both-arrows middle
    section eats up, independent of either flanking net's radius -- see
    MAX_NET_RADIUS below, which is solved from this."""
    code_right_edge, brace_left_edge = code_edges_for(code_lines)
    return right_facing_edge(code_right_edge) - left_facing_edge(brace_left_edge)


# Growth chain: net 0 spawns on the left, deliberately tiny (4 nodes) so
# the whole chain reads as starting small -- each following flat net
# spawns bigger and denser on the right, then slides into the left
# slot. Node count doubles for the first five transitions (4 -> 128),
# then grows a gentler x1.5 for the rest.
#
# NODE_RADIUS is fixed across every stage -- nodes no longer shrink as
# the chain grows, so a bigger net reads as an honestly bigger net, not
# the same-sized blob getting denser. But a bigger net also has to
# actually fit on screen: MAX_NET_RADIUS is solved directly from this
# file's own layout geometry (lap_shift always leaves a lap's whole
# composition centered on x=0 after shifting, which reduces "does the
# widest lap fit in frame_width" to left_radius + right_radius <=
# (frame_width - lap_code_span) / 2 -- see lap_shift/left_net_center/
# right_net_center) rather than a hand-picked constant, using the
# tightest (most code-span-eating) lap across the whole chain so a cap
# applied to every net individually stays safe for any pairing, not just
# the lap it was solved from. Halved again since two nets could both be
# at the cap in the same lap.
MAX_NET_RADIUS = (config.frame_width - max(_lap_code_span(lines) for lines in CODE_STAGES)) / 2 / 2

NODE_RADIUS = 0.11
NET_RADIUS_PACKING_K = 1.3

# How much closer to the frame edge a node's *center* is allowed to sit
# than the raw screen boundary would otherwise permit -- large nets were
# packing nodes right out to cloud_radius/radius_y with nothing held
# back for the node's own visible footprint beyond its bare center
# point, so a node sampled right at that boundary had its real glow
# (see add_soft_bloom's own widest span_mult=6.5 layer -- the number
# mirrored here) bleeding past the frame edge, or the boundary itself
# picked so tight two adjacent nets' nodes could end up nearly touching.
# Subtracted from MAX_NET_RADIUS/MAX_NET_RADIUS_Y wherever cloud_radius/
# radius_y actually get capped (see _stage below) rather than baked into
# either constant directly, so each keeps meaning exactly what its own
# name says: the raw geometric limit, not that limit minus a margin.
NODE_EDGE_MARGIN = 6.5 * NODE_RADIUS


def net_radius(n_nodes, spacing_mult=3.0):
    """The radius net_radius() nodes need at spacing_mult*NODE_RADIUS
    apart, padded by NET_RADIUS_PACKING_K for the rejection sampler's own
    (well short of 100%) packing efficiency -- confirmed reliable
    (placing every node comfortably inside sample_cloud's MAX_SAMPLE_TRIES
    budget) at spacing_mult=3.0. Callers needing to stay on screen should
    min() this against MAX_NET_RADIUS rather than use it directly (see
    STAGES below) -- past that radius, growing outward stops helping and
    only packing tighter (a smaller spacing_mult) does."""
    return NET_RADIUS_PACKING_K * spacing_mult * NODE_RADIUS * math.sqrt(n_nodes)


# spacing_mult tightens (nodes pack closer, down from the default 3.0
# toward the ~2*NODE_RADIUS two circles would need just to avoid touching)
# as n_nodes climbs, so that once cloud_radius hits MAX_NET_RADIUS and
# literally can't grow outward anymore, later stages still read as denser
# than earlier ones instead of all plateauing at the same node count --
# confirmed against actual placement counts: flat 3.0 throughout plateaus
# at the same ~73 nodes from n_nodes=128 on, this progression instead
# keeps climbing right up through n_nodes=288.
#
# MAX_NET_RADIUS only caps the *horizontal* radius (it's solved from
# frame_width, see above) -- frame_height=8.0 leaves far more headroom
# (radius up to 4.0) than any STAGES net actually uses. radius_y lets the
# later, already horizontally-capped stages stretch into that unused
# vertical room instead of staying circular and leaving it empty. The
# raw geometric limit (frame_height/2), not "a bit under" it -- NODE_
# EDGE_MARGIN (see _stage below, where this actually gets used) is what
# keeps a stretched net's own nodes, glow included, from reaching the
# top/bottom edge, not a margin baked into this constant itself. Left at
# None (circular, same as cloud_radius) for the earlier stages, which
# aren't at the horizontal cap yet and don't need it.
MAX_NET_RADIUS_Y = config.frame_height / 2


def _stage(n_nodes, spacing_mult, k_neighbors, seed, radius_y=None):
    return dict(
        n_nodes=n_nodes,
        cloud_radius=min(net_radius(n_nodes, spacing_mult), MAX_NET_RADIUS - NODE_EDGE_MARGIN),
        k_neighbors=k_neighbors,
        node_radius=NODE_RADIUS,
        seed=seed,
        spacing_mult=spacing_mult,
        radius_y=min(radius_y, MAX_NET_RADIUS_Y - NODE_EDGE_MARGIN) if radius_y is not None else None,
    )


STAGES = [
    _stage(n_nodes=4, spacing_mult=3.00, k_neighbors=3, seed=25),
    _stage(n_nodes=8, spacing_mult=2.85, k_neighbors=4, seed=2),
    _stage(n_nodes=16, spacing_mult=2.70, k_neighbors=5, seed=3),
    _stage(n_nodes=32, spacing_mult=2.55, k_neighbors=6, seed=4),
    _stage(n_nodes=64, spacing_mult=2.40, k_neighbors=6, seed=5, radius_y=2.3),
    _stage(n_nodes=128, spacing_mult=2.30, k_neighbors=5, seed=6, radius_y=2.8),
    _stage(n_nodes=192, spacing_mult=2.20, k_neighbors=5, seed=7, radius_y=3.3),
    _stage(n_nodes=288, spacing_mult=2.15, k_neighbors=4, seed=8, radius_y=3.7),
]

# The mini-movie 1 intro's end net: a plain "larger net appears" beat, not
# part of the main STAGES growth chain, so it gets its own stage dict
# rather than being shoehorned into that sequence -- predates the
# fixed-NODE_RADIUS redesign above and isn't part of that growing chain,
# so it keeps its own smaller, independently-picked node size and radius.
INTRO_END_STAGE = dict(n_nodes=10, cloud_radius=1.17, k_neighbors=2, node_radius=0.088, seed=31)

# The very last net: same fixed NODE_RADIUS as every STAGES entry, but
# deliberately exempt from MAX_NET_RADIUS -- unlike every STAGES net, this
# one is *meant* to overflow the screen on three sides (top, right, and
# bottom; left stays put, since that's the edge the arrow still needs to
# reach), to land the point that whatever this has become no longer fits
# in the frame meant to contain it.
#
# Shaped as a leftward-opening "sideways parabola" (see sample_cloud's
# shape="parabola") rather than the squircle every STAGES net uses --
# reaching its full horizontal depth exactly at the arrow's own height
# (y=0) and sweeping back in above and below that as |y| grows, so the
# curve's two arms read as closing around/enveloping the arrow's own tip
# rather than opening away from it. FINAL_STAGE_X_MAX/
# _Y_MAX are hand-picked (not derived from net_radius(), which assumes a
# circle) so this net's density -- nodes per unit area -- matches
# STAGES[-1]'s own actual (capped-and-stretched) density, ~13.55/unit^2,
# rather than the far sparser density net_radius()'s circular formula
# would give at this node count: this reads as the natural continuation
# of what grew it, not as suddenly thinning out right as the chain
# reaches its final, most "overflowing" net.
#
# spacing_mult is tighter than the default 3.0, but not as tight as it
# looks: sample_cloud's min_dist floor (spacing_mult*NODE_RADIUS) is
# compared against raw NODE_RADIUS, but make_node's own mid ring -- the
# actual solid, opaque circle that reads as "the node" -- is drawn at
# NODE_RADIUS*1.15, not NODE_RADIUS. Two nodes only visually clear each
# other once spacing_mult exceeds 2*1.15 = 2.3, and this stays a bit past
# that (rather than right at it, which would leave circles exactly
# kissing with zero clear margin) for a small but real gap between
# them. Below 2.3 -- 1.8, this value until it visibly produced
# overlapping circles, was one such case -- min_dist is smaller than the
# nodes actually drawn, so they overlap regardless of how correctly the
# rejection sampler enforces that same (too-small) min_dist. Confirmed by
# direct simulation to still place all 432 nodes at this spacing, inside
# this same smaller, denser area, well inside sample_cloud's
# MAX_SAMPLE_TRIES budget (needs ~11900 tries, not the old 4000 -- see
# MAX_SAMPLE_TRIES's own comment). k_neighbors is kept modest despite the
# huge node count purely for render time -- this many nodes already
# renders slowly; a denser mesh on top of that compounds fast.
FINAL_STAGE_SPACING_MULT = 2.5
# Used only for the icosahedron finale's own footprint offset below
# (INCLUDE_FINALE=1) -- that mesh is a fixed hand-built shape, not a
# build_net() cloud, so it still wants a plain symmetric "radius" to
# center itself against the arrow, unlike the parabola net below.
FINAL_STAGE_RADIUS = net_radius(432, FINAL_STAGE_SPACING_MULT)
FINAL_STAGE_X_MAX = 5.2
FINAL_STAGE_Y_MAX = 4.6
# How far back (as a fraction of FINAL_STAGE_X_MAX) the parabola's arms
# reach behind the touch point at their deepest, y=+-FINAL_STAGE_Y_MAX
# (see sample_cloud's shape="parabola"). 0.6 world units there -- comparable
# to ARROW_LENGTH (0.7) itself, clearly short of reaching back to the
# brace/code -- but the parabola's own shape (a y^2 falloff, near-zero right
# at y=0) already keeps the reach far smaller than that near the arrow's
# actual own height, where GAP (0.25) is the real constraint: at
# FINAL_STAGE's own right_arrow tip half-height (ARROW_TIP_HALF_HEIGHT,
# 0.175 -- the widest the arrow itself ever gets), this reaches back only
# 0.6*(0.175/4.6)^2 ~ 0.00086 world units, nowhere close to GAP, so the
# curve's arms clear the arrow's own polygon everywhere the arrow actually
# occupies space, not just at y=0 exactly.
PARABOLA_WRAP_FRAC = 0.6 / FINAL_STAGE_X_MAX

# The exponent in sample_cloud's own shape="squircle" rejection test,
# |x|^SQUIRCLE_EXPONENT + |y|^SQUIRCLE_EXPONENT > 1 -- 2 is exactly a
# circle/ellipse (a true superellipse at that exponent), and climbing
# past it bulges the boundary out toward the corners of the unit square,
# rounding off rather than snapping straight -- 4 is the traditional
# "squircle" and already reads as a clearly rounded square rather than a
# stretched circle even at the STAGES nets' own vertical stretch
# (radius_y well under radius for the biggest ones).
SQUIRCLE_EXPONENT = 4

FINAL_STAGE = dict(
    n_nodes=432,
    cloud_radius=FINAL_STAGE_X_MAX,
    k_neighbors=4,
    node_radius=NODE_RADIUS,
    seed=9,
    spacing_mult=FINAL_STAGE_SPACING_MULT,
    radius_y=FINAL_STAGE_Y_MAX,
    shape="parabola",
)

# One multiplier per STAGES[1:] transition, plus FINAL_CODE_MULT for the
# 8th and last (the dramatic-overflow lap) -- 8 laps in total, solved so
# each lap's own real-world duration comes out to a target that starts
# at 12s and eases down by the same ratio every lap, summing to 90s
# overall: the same shape as the takeoff curves this scene is about, so
# the chain visibly (if gently -- "each speeds up slightly", not
# dramatically) accelerates into the finale rather than ticking along at
# a uniform clip. Each multiplier was back-solved from its target
# duration against this file's own lap-timing formula (every beat below
# scaled by that lap's m, plus that lap's own CODE_STAGE line count --
# see grow_in and the per-lap self.play calls in construct_main) rather
# than hand-tuned, since hand-picking multipliers to hit a specific total
# runtime is unreliable when every lap's code block is a different
# length.
SPEED_MULTIPLIERS = [1.9544, 1.9695, 1.9862, 1.8972, 1.9132, 1.8778, 1.8430]
FINAL_CODE_MULT = 1.7605

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
    as a raster image here, is actually smooth.

    Smooth in exact math still isn't smooth once it's quantized to 8-bit
    channels and stretched, though: BACKGROUND_COLOR -> BACKDROP_GLOW_COLOR
    only spans ~50 levels of blue over the whole radius, so with nothing
    to break the quantization up, huge bands of pixels round to the exact
    same integer color and show up as visible rings -- the same step-
    function problem raster-rendering this was meant to avoid in the
    first place, just moved from "alpha compositing" to "8-bit rounding"
    as its cause. resolution is matched to the actual output pixel height
    (not a smaller value stretched up) so upscaling blockiness can't add
    still more banding of its own, and a small per-pixel dither is added
    before rounding -- some pixels a band "should" round down round up
    instead (and vice versa), which turns each hard edge into a soft,
    visually-smooth transition instead of a sharp ring. Seeded rather
    than left nondeterministic since this backdrop is built once and
    reused for the whole video, not regenerated per frame, where an
    unseeded dither would otherwise crawl/shimmer across frames if this
    ever did get called more than once."""
    resolution = config.pixel_height
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
    rgb_f = bg + (glow - bg) * t[:, :, None]
    dither = np.random.default_rng(0).uniform(-0.5, 0.5, size=rgb_f.shape)
    rgb = np.clip(np.round(rgb_f + dither), 0, 255).astype(np.uint8)
    alpha = np.full((*t.shape, 1), 255, dtype=np.uint8)
    rgba = np.concatenate([rgb, alpha], axis=2)

    image = ImageMobject(rgba)
    image.stretch_to_fit_width(config.frame_width)
    image.stretch_to_fit_height(config.frame_height)
    image.move_to(ORIGIN)
    image.set_z_index(-10)
    return image


def _lerp_hex(hex_a, hex_b, t):
    a, b = _hex_to_rgb(hex_a), _hex_to_rgb(hex_b)
    rgb = np.clip(a + (b - a) * t, 0, 255).astype(int)
    return "#%02x%02x%02x" % tuple(rgb)


def _resample_hex_path(controls, n):
    """n evenly-spaced hex colors along an ordered list of control-point
    hex colors, via piecewise linear interpolation across whichever pair
    of controls each sample point falls between."""
    segments = len(controls) - 1
    result = []
    for i in range(n):
        t = (i / (n - 1) * segments) if n > 1 else 0.0
        seg = min(int(t), segments - 1)
        result.append(_lerp_hex(controls[seg], controls[seg + 1], t - seg))
    return result


def _resample_palette_path(controls, n):
    """Same as _resample_hex_path, but for a list of (core, mid, glow)
    palette tuples instead of bare hex colors -- each channel resampled
    independently, then zipped back into per-stage palette tuples."""
    channels = zip(*controls)
    resampled_channels = [_resample_hex_path(list(channel), n) for channel in channels]
    return list(zip(*resampled_channels))


# Resampled now rather than left as the 5 control points directly:
# STAGES (below) may have any number of stages, and this way the chain's
# green-to-magenta progression always spans however many there actually
# are, rather than running out of colors partway through a longer chain
# or being bunched up in a shorter one.
NET_PALETTES = _resample_palette_path(NET_PALETTE_CONTROLS, len(STAGES))
NET_EDGE_COLORS = _resample_hex_path(NET_EDGE_CONTROLS, len(STAGES))


def make_glow_disc(span_radius, sigma_radius, color, peak_alpha):
    """A real blurred glow disc -- gaussian falloff rasterized once with
    numpy, not a stack of flat-opacity rings. Alpha-compositing discs is
    still a step function, so stacked rings always show a visible ring
    at every layer's radius; only a true per-pixel gradient, rendered as
    a raster image, reads as an actual glow rather than a target/
    bullseye of rings (see make_glow_blob's docstring for the same
    reasoning, applied there to a whole net instead of a single node)."""
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


def add_soft_bloom(node, radius, glow_color, extras):
    """Two layered real glow discs behind the node -- a wide, faint outer
    wash and a tighter, brighter inner one. Tracked to the node's live
    center every frame via its own updater rather than folded into the
    node's own VGroup, since ImageMobject can't live inside a VGroup
    (VGroup only accepts VMobjects). The node it tracks is stashed on
    the disc itself (.tracked_node) so stop_effects/resume_effects can
    tear down and re-attach this updater around a net's own reposition
    animations without needing to separately carry node references."""
    for span_mult, sigma_mult, peak_alpha in ((6.5, 2.2, 60), (3.6, 1.0, 115)):
        disc = make_glow_disc(radius * span_mult, radius * sigma_mult, glow_color, peak_alpha)
        disc.move_to(node.get_center())
        disc.tracked_node = node
        disc.add_updater(lambda mob: mob.move_to(mob.tracked_node.get_center()))
        extras.add(disc)


def spawn_pulse_ring(node, color, max_growth_mult=1.6, duration=2.0, peak_opacity=0.425):
    """A single expanding, fading ring -- a brand-new mobject per ping
    rather than one ring reused per node, so a node pinged again while
    its last ring is still expanding gets a second, independent ring
    instead of the first one being cut short.

    Sized live off the node's own current width every frame (see
    current_radius below) rather than a fixed radius captured once at
    spawn time -- a ping can fire while its node is still mid-
    GrowFromCenter (see grow_in), and a ring sized for the node's
    eventual full radius would read as oversized/mismatched next to a
    node that's still small. Tracking the node's actual in-progress
    scale, frame by frame, keeps the ring visually anchored to it the
    whole time, including if the node is still growing when the ring's
    own animation starts.

    Returns (ring, tick) rather than attaching its own add_updater --
    tick(dt) is meant to be called once per frame by add_pulse_chains'
    own driver mobject instead, exactly like add_node_flash's own tick.
    A ring added straight to ring_holder (see build_net) *is* a family
    member of whatever whole-net Animation is running during a slide or
    FadeOut, so if the ring carried its own add_updater, that
    Animation's own starting_mobject -- a copy.deepcopy(ring_holder)
    taken at the animation's begin() -- would carry a "copy" of that
    updater that's actually the exact same closure over the exact same
    mutable state (plain functions are atomic under deepcopy, see
    add_node_flash's own docstring for the confirmed mechanism), and
    Animation.update_mobjects ticks that starting_mobject once every
    frame in addition to the Scene's own per-frame pass -- so the ring's
    own progress clock would advance twice as fast for as long as that
    animation ran. Confirmed directly: exactly the "pings sped up while
    the other net moves" symptom, now fixed by never handing a ring its
    own updater to begin with -- tick() is only ever invoked once, by
    the driver, regardless of what's simultaneously animating the net
    the ring happens to be sitting inside."""

    def current_radius():
        # node[0] is the node's own outer ring (see make_node) -- its
        # live .width reflects however far GrowFromCenter has scaled it
        # up so far, not just its eventual full size.
        return node[0].width / 2

    ring = Circle(radius=current_radius(), stroke_color=color, stroke_width=2.4, fill_opacity=0)
    ring.move_to(node.get_center())
    ring.set_z_index(2)  # matches extras -- see build_net's layering comment
    # A no-op updater, purely so Scene.get_moving_mobjects() sees this
    # ring as having a family updater and keeps re-rendering it every
    # frame. Without this, a ring with zero updaters attached (its real
    # animation lives entirely on the driver/tick now, not on the ring
    # itself) gets treated by manim as a "static" mobject for any
    # self.play() call where it isn't itself the direct target -- the
    # slide targets next_net as a whole, not this ring individually --
    # and static mobjects are rendered once and reused as a cached
    # background for that entire animation, so tick()'s own frame-by-
    # frame become()/move_to() calls kept computing the right numbers
    # but were never actually redrawn until the *next* self.play() call
    # recomputed the moving/static split from scratch. Confirmed as the
    # cause of "rings freeze during the slide, then a batch of stale
    # ones suddenly disappear once fresh ones start" -- the freeze was
    # the stale cached frame, and the disappearance was rendering
    # finally catching up to whatever tick() had already computed in
    # the background (commonly fully faded out by then). This updater
    # does nothing and touches no shared state, so there's nothing for
    # Animation's own ghost-tick (see this function's docstring above)
    # to double up on even if it fires an extra time.
    ring.add_updater(lambda mob, dt: None)
    state = {"t": 0.0}

    def tick(dt):
        """Returns True once this ring has finished growing/fading --
        the caller (add_pulse_chains' own scheduler) decides when it's
        actually safe to remove it from ring_holder, since that's the
        one operation here that touches ring_holder's own family size
        (see build_net's own comment on why that matters)."""
        state["t"] += dt
        progress = min(state["t"] / duration, 1.0)
        base = current_radius()
        ring.become(
            Circle(
                radius=base * (1 + max_growth_mult * progress), stroke_color=color, stroke_width=2.4, fill_opacity=0
            )
        )
        ring.move_to(node.get_center())
        ring.set_stroke(opacity=(1.0 - progress) * peak_opacity)
        ring.set_z_index(2)  # become() replaces the mobject wholesale, z_index included
        return progress >= 1.0

    return ring, tick


def add_node_flash(node, rest_color, flash_color, attack=0.05, release=0.6):
    """Ramp the node's mid ring (index 0) up to flash_color over
    `attack`, then ease it back down to rest_color over `release` -- a
    quick rise-then-fade, rather than snapping straight to flash_color
    on the first frame and only fading from there (which reads as a
    flat white pop, not a flash).

    Returns (trigger, tick) rather than attaching its own per-node
    add_updater -- tick(dt) is meant to be called once per frame by the
    net's single shared pulse-driver updater (see add_pulse_chains)
    instead. A node-level updater would get a second, redundant call
    every frame this node is itself mid-GrowFromCenter or mid-slide:
    Animation.update_mobjects(dt) always ticks its own starting_mobject
    (a Mobject.copy(), i.e. copy.deepcopy(node)) in addition to the
    Scene's own per-frame update pass, and since plain functions are
    atomic under deepcopy (copy.deepcopy returns the same function
    object, not an independent one), that starting_mobject's "copy" of
    this updater is actually the exact same closure over the exact same
    mutable state -- so it fires for real, a second time, on the real
    node, roughly doubling this node's own flash rate for as long as
    it's being grown or slid. Confirmed directly: an instrumented run
    logged 3-4x as many scheduler ticks per real rendered frame during
    grow-in/slide than during a plain hold, exactly the "pings sped up
    while spawning/moving" symptom. A tick() called explicitly by one
    driver mobject that is never itself an Animation target has no
    starting_mobject to get ghost-ticked through in the first place."""
    mid = node[0]
    duration = attack + release
    state = {"active": False, "t": 0.0}

    def tick(dt):
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

    return trigger, tick


def add_pulse_chains(scene, nodes_group, edges_group, palette, extras, chain_stagger=0.2):
    """One node pings, then an adjacent node (following an actual edge)
    0.2s later, then another -- 3-5 hops long -- rather than every node
    independently, randomly pinging on its own. A beat after a chain's
    last hop fires, another chain starts from a random node -- longer
    for small nets (fewer than SMALL_NET_THRESHOLD nodes), which have so
    little to hop across that the default gap reads as spammy.

    How many such chains run concurrently scales with net size (see
    num_chains/NODES_PER_CHAIN below): a handful of nodes gets the
    original single chain, but a several-hundred-node net runs a dozen
    or more independent chains at once, so it reads as a genuinely busy
    net instead of one lone chain wandering a mostly-idle crowd.

    The scheduler (and every node's own flash tick, see add_node_flash)
    lives on one dedicated, invisible "driver" mobject added straight to
    the scene -- not on `extras` and not on the nodes themselves -- and
    is never the target of any grow/slide/fade Animation. A mobject that
    IS such a target gets ghost-ticked an extra time per frame for as
    long as that animation runs (see add_node_flash's own docstring for
    why), which used to double or triple the effective ping rate right
    when a net was growing in or sliding. A driver nobody ever animates
    has no such ghost copy, so it only ever ticks once per real frame."""
    core_color, mid_color = palette[0], palette[1]
    # Bloom Pulse's own flash target is each net's near-white palette core
    # -- reads as "white hot" against that node's own glow/ring, which
    # SIMPLE_STYLE has neither of (see its own module-level comment), so
    # the same near-white flash there just reads as flashing white with
    # nothing to sell it as a "hot" version of the node's own color.
    # SIMPLE_STYLE therefore flashes to a plain, fixed red instead --
    # matching the plain "the circle just flashes red" description this
    # style is meant to give -- regardless of which net's own palette
    # (green/blue/red/etc.) the node otherwise belongs to.
    flash_color = FLASH_COLOR_OVERRIDE or (SIMPLE_STYLE_FLASH_COLOR if SIMPLE_STYLE else core_color)
    node_list = list(nodes_group)
    SMALL_NET_THRESHOLD = 6
    next_chain_cooldown = 0.9 if len(node_list) < SMALL_NET_THRESHOLD else 0.5
    index_of = {id(n): i for i, n in enumerate(node_list)}
    adjacency = {i: [] for i in range(len(node_list))}
    for edge in edges_group:
        i, j = index_of[id(edge.node_a)], index_of[id(edge.node_b)]
        adjacency[i].append(j)
        adjacency[j].append(i)

    flash_pairs = [add_node_flash(node, mid_color, flash_color) for node in node_list]
    flash_triggers = [pair[0] for pair in flash_pairs]
    flash_ticks = [pair[1] for pair in flash_pairs]

    # The flash fires slightly after the ring spawns -- simultaneous
    # felt backwards, since the ring starts at the node's own radius and
    # only becomes visibly bigger a beat later, while the flash (being
    # fast) is already done by then. Firing the flash FLASH_DELAY after
    # the ring gives the ring a moment's head start.
    FLASH_DELAY = 0.1

    def fire(idx):
        # SIMPLE_STYLE keeps the flash (a node lighting up) but skips the
        # expanding ring entirely -- this render pass wants pings to read
        # as "the node flashes," not the ring's own separate growth/fade.
        #
        # No lock/gating here otherwise for an *active* net -- a ring can
        # spawn at literally any point in its life, including mid-slide
        # or mid-FadeOut, without risk (see ring_holder's own comment in
        # build_net for why: it's never a family member of anything a
        # Scene animation targets, so adding to it is never the kind of
        # family-size change that crashes manim mid-interpolation).
        #
        # retired (see retire_pings) is the one exception: once a net is
        # retired, its own ring_holder becomes the direct target of a
        # FadeOut in that same closing self.play() (to fade away
        # whatever rings retire_pings froze), and *that* animation's own
        # family-size assumptions would be violated by a brand new ring
        # arriving mid-interpolation over it -- confirmed directly
        # against an actual render ("zip() argument 2 is shorter than
        # argument 1"), from a hop that was already queued before
        # retirement firing normally, same as retire_pings' own docstring
        # says it should, and adding its ring right in the middle of
        # that FadeOut. The flash queues regardless (see below) --
        # retirement only ever withholds the ring, which is the one
        # piece actually unsafe to add once retired.
        if not SIMPLE_STYLE and not state["retired"]:
            ring, ring_tick = spawn_pulse_ring(node_list[idx], core_color)
            extras.ring_holder.add(ring)
            if DEBUG_PULSE:
                ring._debug_id = id(ring) % 10000
                print(f"RING tag={debug_tag} SPAWN id={ring._debug_id} clock={debug_clock['t']:.4f}", flush=True)
            state["active_rings"].append([ring, ring_tick, False])
        state["flash_queue"].append([FLASH_DELAY, idx])

    # Each entry in chain_states runs its own independent chain -- its own
    # queue (holding [time_remaining, node_index] for hops still waiting
    # to fire) and its own cooldown (counting down to that chain's next
    # hop-path once its queue drains) -- so multiple chains can be
    # in-flight across the net at once instead of only ever one at a
    # time. How many run concurrently scales with net size: a small net
    # (few nodes to hop across) gets by fine with one, but a net with
    # hundreds of nodes needs several running at once to read as busy
    # rather than sparse -- one lone chain wandering a few hundred mostly-
    # idle nodes just reads as broken, not calm. Initial cooldown is
    # staggered slightly across chains (chain_idx * chain_stagger) rather
    # than all zero -- this is attached right as a net finishes growing
    # in (or right after it slides into place), so the first chains
    # should all start right away, just not on the exact same frame as
    # each other, which read as a single synchronized flash-burst instead
    # of independently wandering chains.
    NODES_PER_CHAIN = 30
    num_chains = max(1, len(node_list) // NODES_PER_CHAIN)
    chain_states = [{"cooldown": chain_idx * chain_stagger, "queue": []} for chain_idx in range(num_chains)]

    # flash_queue holds the same [time_remaining, node_index] shape as a
    # chain's own queue, for flashes waiting on FLASH_DELAY -- shared
    # across every chain since a flash is a per-node effect, not a
    # per-chain one. active_rings holds [ring, tick, finished] for every
    # ring still being tracked -- finished flips to True once tick()
    # reports the ring is done growing/fading, at which point the
    # scheduler's own active_rings loop below removes it from ring_holder
    # immediately (safe at any time -- see ring_holder's own comment in
    # build_net). Both are also shared across every chain for the same
    # reason. elapsed/ticked_through drive the scheduler's own dt
    # subdivision (see scheduler below) -- kept here, alongside the rest
    # of this net's mutable state, so they persist across every separate
    # real-frame call rather than resetting each time. retired starts
    # False -- retire_pings (see below) sets it True for a net that's
    # about to fade out for good, so no *new* chain starts once its
    # queue drains, while whatever's already queued/mid-flight (a hop
    # already committed, a flash already lit, a ring already expanding)
    # still finishes out naturally rather than cutting off mid-beat.
    state = {"flash_queue": [], "active_rings": [], "elapsed": 0.0, "ticked_through": 0.0, "retired": False}
    debug_tag = id(nodes_group) % 1000
    debug_clock = {"t": 0.0, "n": 0}

    def tick_one_frame(dt):
        if DEBUG_PULSE:
            debug_clock["t"] += dt
            debug_clock["n"] += 1
            print(
                f"PULSE tag={debug_tag} tick={debug_clock['n']} clock={debug_clock['t']:.4f} dt={dt:.5f}",
                flush=True,
            )
        # Every node's own flash tick lives here now too (see
        # add_node_flash) rather than as that node's own add_updater --
        # driven unconditionally every call, same as before, just no
        # longer contingent on the node itself carrying an updater that
        # a GrowFromCenter/slide could ghost-tick a second time.
        for node_tick in flash_ticks:
            node_tick(dt)

        # Every in-flight ring's own tick (see spawn_pulse_ring) lives
        # here too, driven unconditionally every call same as the flash
        # ticks above, rather than each ring carrying its own add_updater
        # -- keeps ring growth/fade timing driven off exactly the same
        # per-frame dt as everything else on this net, with nothing extra
        # to reason about even though ring_holder itself (see build_net's
        # own comment) is never a family member of any Animation that
        # could otherwise ghost-tick it.
        #
        # Skipped once retired (see retire_pings) -- a retired net is
        # moments from its own closing FadeOut, and that FadeOut targets
        # ring_holder directly (see every retire_pings call site) to fade
        # whatever rings are still around right along with the rest of
        # the net. Left ticking, this loop would keep re-asserting each
        # ring's own tick()-computed opacity every single frame, fighting
        # the FadeOut for control of that same value -- confirmed
        # directly against an actual render: nets fade out completely
        # while their own rings stay fully visible, floating with
        # nothing left to anchor them. Freezing here instead leaves each
        # ring exactly as it looked the instant retirement was called,
        # for FadeOut(ring_holder) to fade uniformly from there with
        # nothing else touching its opacity in the meantime.
        if not state["retired"]:
            still_active = []
            for ring, ring_tick, finished in state["active_rings"]:
                was_finished = finished
                finished = ring_tick(dt) or finished
                if DEBUG_PULSE and finished and not was_finished:
                    print(
                        f"RING tag={debug_tag} FINISH id={getattr(ring, '_debug_id', '?')} "
                        f"clock={debug_clock['t']:.4f} "
                        f"opacity={ring.stroke_opacity if hasattr(ring, 'stroke_opacity') else '?'} "
                        f"radius={ring.width/2:.4f}",
                        flush=True,
                    )
                if finished:
                    if DEBUG_PULSE:
                        print(
                            f"RING tag={debug_tag} REMOVE id={getattr(ring, '_debug_id', '?')} "
                            f"clock={debug_clock['t']:.4f}",
                            flush=True,
                        )
                    extras.ring_holder.remove(ring)
                else:
                    still_active.append([ring, ring_tick, finished])
            state["active_rings"] = still_active

        pending_flash = []
        for delay, idx in state["flash_queue"]:
            delay -= dt
            if delay <= 0:
                flash_triggers[idx]()
            else:
                pending_flash.append([delay, idx])
        state["flash_queue"] = pending_flash

        # Each chain advances independently -- its own queue drains (or
        # doesn't) and its own cooldown counts down (or doesn't) without
        # any of that affecting the other chains' own state, so however
        # many chains num_chains set up all keep hopping/cooling down/
        # restarting on their own separate schedules.
        for chain in chain_states:
            pending = []
            for delay, idx in chain["queue"]:
                delay -= dt
                if delay <= 0:
                    if DEBUG_PULSE:
                        print(f"PULSE tag={debug_tag} FIRE clock={debug_clock['t']:.4f} idx={idx}", flush=True)
                    fire(idx)
                else:
                    pending.append([delay, idx])
            chain["queue"] = pending

            if chain["queue"]:
                continue
            chain["cooldown"] -= dt
            if chain["cooldown"] > 0:
                continue
            if state["retired"]:
                # This net is fading out for good (see retire_pings) --
                # this chain's own queue already fully drained (the check
                # above), so no hop from it is still in flight; simply
                # never starting a new one is enough to let this net go
                # quiet on its own within a beat, without cutting off
                # anything already committed elsewhere.
                continue

            length = random.randint(3, 5)
            current = random.randrange(len(node_list))
            path = [current]
            visited = {current}
            for _ in range(length - 1):
                unvisited = [n for n in adjacency[current] if n not in visited]
                if not unvisited:
                    # Dead end -- every neighbor (including wherever this
                    # hop just came from) is already visited. The chain
                    # just ends here rather than bouncing back the way it
                    # came.
                    break
                current = random.choice(unvisited)
                path.append(current)
                visited.add(current)

            chain["queue"] = [[hop * chain_stagger, idx] for hop, idx in enumerate(path)]
            chain["cooldown"] = next_chain_cooldown

    def scheduler(mob, dt):
        # Subdivides elapsed time into normal-frame-sized steps before
        # handing it to tick_one_frame, rather than passing whatever dt
        # manim gives this call straight through -- everything above (the
        # queue's own per-hop delays, cooldown countdown, random.choice()
        # chain picking) is written assuming it gets ticked in clean,
        # frame-sized increments, not whatever raw dt a given call happens
        # to carry. That matters for two different reasons: manim's own
        # skip-rendering path (see Scene.get_time_progression) can collapse
        # a whole skipped stretch into one giant dt instead of the usual
        # per-frame sequence, which this scheduler needs subdivided back
        # into individual ticks to keep hop spacing/cooldowns/chain-start
        # randomness correct; and, in the far more common case, THIS
        # updater is called fresh once per real rendered frame with its own
        # separate dt each time (not once for a whole animation the way
        # Scene.get_time_progression itself is called), so any subdivision
        # scheme has to track elapsed time cumulatively *across* calls, not
        # reset its own bookkeeping at the start of every single one.
        #
        # An earlier version of this got that second part wrong: it reset
        # a local `last = 0.0` at the top of every call and did
        # `for t in np.arange(0, dt, step): tick_one_frame(t - last)`.
        # That formula is only correct when applied ONCE across a whole
        # span of elapsed time (which is what it mirrors -- Scene.
        # get_time_progression's own np.arange(0, run_time, step) is
        # called once per Animation, not once per frame). Called fresh
        # every real frame instead, with dt almost always <= one step,
        # np.arange(0, dt, step) collapsed to the single element [0.0]
        # every time -- so tick_one_frame(0.0 - 0.0) = tick_one_frame(0.0),
        # a pure no-op, on nearly every real frame. Confirmed directly
        # against a DEBUG_PULSE trace: 3-4 wasted zero-dt ticks for every
        # 1 real tick, and confirmed visually too -- two frames a tenth of
        # a second apart, pixel-identical. Chains still eventually hopped
        # (whenever floating-point drift happened to push a call's raw dt
        # just over one step, np.arange produced a second, nonzero-dt
        # element), just at a small fraction of the real rate -- exactly
        # the "moves node to node at ~1/sec instead of ~0.2s" symptom.
        #
        # Fixed by tracking elapsed/ticked_through on this net's own
        # persistent `state` (set up alongside flash_queue/active_rings
        # above) instead of a call-local variable: every call
        # adds its own dt to the running total, and the loop below emits
        # exactly one step-sized tick_one_frame(step) for every full step
        # boundary that total has newly crossed since the last call --
        # one tick for an ordinary single-frame call, several in a row for
        # a skip-collapsed multi-frame one, and zero real time ever
        # dropped in between, matching floor(elapsed/step) ticks overall
        # the same way the original np.arange formula intended, just
        # correctly accumulated across calls instead of reset by them.
        if DEBUG_PULSE:
            print(f"SCHED tag={debug_tag} raw_dt={dt:.5f}", flush=True)
        step = 1 / config.frame_rate
        state["elapsed"] += dt
        while state["elapsed"] - state["ticked_through"] >= step:
            state["ticked_through"] += step
            tick_one_frame(step)

    # A plain empty Mobject, added to the scene on its own -- never
    # nested under extras/nodes_group/next_net and never itself the
    # target of any FadeIn/GrowFromCenter/animate call, so nothing ever
    # creates a starting_mobject copy of it to ghost-tick. Stashed on
    # extras (a harmless plain attribute, not a submobject -- Animation's
    # own recursive update() only ever follows .submobjects, never
    # arbitrary attributes) purely so stop_effects/resume_effects can
    # find and clear it later without changing either function's own
    # signature.
    driver = Mobject()
    scene.add(driver)
    driver.add_updater(scheduler)
    extras.pulse_driver = driver

    # ring_holder (see build_net) is never itself the direct target of
    # any animation, and manim only adds a mobject to the scene's own
    # render/update list when it's self.add()-ed or is the direct target
    # of a running Animation. Left implicit, ring_holder would never
    # actually become part of the scene at all: any ring added to it
    # would exist as a live Python object with a running updater, yet
    # never be drawn and never receive a single per-frame update --
    # rings that fire but are never seen (confirmed against an actual
    # render: zero rings visible anywhere, despite fire() calling
    # spawn_pulse_ring exactly as expected). Added directly here instead
    # of relying on some later animation to sweep it in.
    scene.add(extras.ring_holder)

    # Stashed on extras (same reasoning as pulse_driver above) so a
    # caller about to fade this net out for good can reach it without
    # this closure's own state threaded through as a parameter -- see
    # retire_pings' own docstring for what it actually does and why it's
    # a deliberate, separate opt-in rather than something stop_effects
    # does automatically for every caller.
    def retire_pings():
        state["retired"] = True

    extras.retire_pings = retire_pings


def stop_effects(extras):
    """Freeze this net's bloom-disc tracking -- call before any animation
    that repositions OR fades out the net as a whole (the slide, or a
    final FadeOut). The scheduler, every node's own flash, and every
    pulse ring (spawning, growing, fading, and being removed once
    finished) all keep running completely unaffected right through it --
    all three live on add_pulse_chains' own driver mobject now (see
    spawn_pulse_ring and its own tick, ticked once per frame from the
    scheduler's active_rings loop), and ring_holder is never itself a
    family member of the slide/FadeOut Animation this is guarding around
    (see build_net's own comment on why), so pings keep firing -- new
    rings included -- exactly the same whether or not this net happens
    to be moving right now. A net about to disappear for good rather
    than just reposition still wants those *new* pings stopped, just not
    via this function -- see retire_pings, called separately alongside
    this one at exactly those call sites.

    Bloom-disc tracking is what actually needs freezing here, for an
    older, unrelated reason: .suspend_updating() turned out not to be
    enough on its own -- even suspended, an attached updater somehow
    still left a following net.animate.shift() silently unable to move
    the mobject (proven by an isolated test: identical shift,
    .suspend_updating() -> no movement, .clear_updaters() -> moves
    correctly)."""
    for disc in extras:
        disc.clear_updaters()


def resume_effects(extras):
    """Re-attach bloom-disc tracking -- call right after any animation
    that moves a net (e.g. the slide into the left slot). The scheduler/
    flash/ring driver was never stopped in the first place (see
    stop_effects), so there's nothing to restart there -- add_pulse_
    chains isn't called again."""
    for disc in extras:
        disc.add_updater(lambda mob: mob.move_to(mob.tracked_node.get_center()))


def retire_pings(extras):
    """Stop this net from starting any *new* pulse chains, and freeze
    every ring it currently has in flight right where it is -- call
    right before a FadeOut that removes this net for good (never a
    slide that just repositions it: that net is staying, so its pings
    should keep firing exactly as normal). Deliberately separate from
    stop_effects, which every caller uses unconditionally regardless of
    which of those two cases it is -- this is opt-in, only for the "gone
    for good" one.

    Pair this with a FadeOut(extras.ring_holder) in that same closing
    self.play() -- retire_pings only freezes each ring's appearance, it
    doesn't fade anything on its own; without that FadeOut, whatever
    rings were still around when this was called would just sit there,
    frozen, indefinitely. With it, they fade smoothly along with the
    rest of the net instead of continuing to independently animate (or
    worse, sitting at full opacity) while everything else around them
    disappears -- confirmed directly against an actual render: nets
    fading out completely while their own rings stayed fully visible,
    floating with nothing left to anchor them, is exactly what freezing
    (rather than leaving the scheduler still ticking them, fighting
    that same FadeOut for control of their opacity) fixes.

    New chains stop immediately. A hop that was already queued before
    this was called still fires its flash exactly once more, same as
    always, but never a new ring (see fire()'s own comment) -- one would
    otherwise still be free to arrive mid-FadeOut over that same
    ring_holder some time after this call, right when this whole
    mechanism is trying to guarantee its family stays fixed."""
    if hasattr(extras, "retire_pings"):
        extras.retire_pings()


def make_node(pos, radius, palette):
    """A node's crisp core: a bright center over a slightly dimmer mid
    ring. The actual glow around it is a separate, real gaussian-blurred
    disc (see add_soft_bloom) tracking this node from outside its own
    VGroup, not stacked flat-opacity rings faking a bloom."""
    # fill_opacity=1 on both, not 0.9 on mid -- anything less than fully
    # opaque lets whatever's underneath (an edge, its endpoint square in
    # the node's own center) blend faintly through, which is exactly
    # what read as "the edge is drawn on top of the node" despite z-order
    # already putting the node in front.
    core_c, mid_c, _glow_c = palette
    paint_opacity = 0 if EDGES_ONLY else 1
    if SIMPLE_STYLE:
        node = VGroup(Circle(radius=radius, stroke_width=0, fill_color=mid_c, fill_opacity=paint_opacity))
        node.move_to(pos)
        return node
    mid = Circle(radius=radius * 1.15, stroke_color=mid_c, stroke_width=1, fill_color=mid_c, fill_opacity=1)
    core = Circle(radius=radius * 0.5, stroke_color=core_c, stroke_width=1, fill_color=core_c, fill_opacity=1)
    if EDGES_ONLY:
        mid.set_opacity(0)
        core.set_opacity(0)
    node = VGroup(mid, core)
    node.move_to(pos)
    return node


def make_edge(node_a, node_b, color):
    """A glowing connection: a wide, faint pass beneath a thin, brighter
    one, so the line reads as lit rather than a flat stroke. Computed
    once from the two nodes' centers at construction and left alone --
    nodes no longer drift independently in this style (see module
    docstring), so a plain rigid transform on the whole net (grow,
    slide) carries an edge along with its nodes correctly on its own,
    without a live per-frame tracking updater."""
    a, b = node_a.get_center(), node_b.get_center()
    if SIMPLE_STYLE:
        edge = VGroup(Line(a, b, stroke_color=color, stroke_width=1.5, stroke_opacity=0.6))
    else:
        wide = Line(a, b, stroke_color=color, stroke_width=4.5, stroke_opacity=0.12)
        thin = Line(a, b, stroke_color=color, stroke_width=1.3, stroke_opacity=0.6)
        edge = VGroup(wide, thin)
    edge.node_a = node_a
    edge.node_b = node_b
    return edge


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


# Every STAGES/INTRO_END_STAGE net places all its nodes well inside the
# old 4000-try cap; only FINAL_STAGE's own tighter FINAL_STAGE_SPACING_MULT
# (see its own comment) needs more headroom than that -- confirmed by
# direct simulation to need ~11900 tries, not 4000, to place all 432 nodes
# at that spacing. Raised generously past that instead of tuned to the
# exact number so it isn't one future seed/parameter tweak away from
# silently placing fewer nodes than asked for again.
MAX_SAMPLE_TRIES = 20000


def sample_cloud(
    n, radius, node_radius, seed, min_dist_factor=0.32, spacing_mult=3.0, radius_y=None, shape="squircle"
):
    """n points scattered inside a region of the given radius (or a
    stretched one, if radius_y differs from radius -- see STAGES'
    vertical stretch for later stages), rejecting anything too close to
    a point already placed -- rough blue-noise spacing, so nodes read as
    an organic cluster rather than a grid. Sampled uniformly in a unit
    square and rejected against the unit shape's own boundary, then that
    unit shape is stretched to (radius, radius_y) rather than sampled
    directly in stretched coordinates -- simple, and the min_dist
    rejection below still operates on genuine post-stretch Euclidean
    distance, so spacing stays correct regardless of aspect ratio.
    Whatever that blue-noise spacing works out to, it's never allowed
    below spacing_mult*node_radius -- at the default spacing_mult=3.0,
    two touching node circles (2*node_radius) plus a full extra
    node_radius of clear margin between them, so nodes at this size can
    never actually overlap, even for the sparsest/smallest-radius stages
    where the proportional spacing alone would allow it. Callers packing
    a fixed on-screen radius tighter to fit more nodes (see STAGES) pass
    a smaller spacing_mult instead.

    shape="squircle" (the default -- every STAGES net and both intro
    nets use it) rejects |x|^SQUIRCLE_EXPONENT + |y|^SQUIRCLE_EXPONENT >
    1 rather than plain x^2 + y^2 > 1 -- the same superellipse family a
    true circle/ellipse (exponent 2) belongs to, just with a higher
    exponent that pushes the boundary out toward a rounded square instead
    (see SQUIRCLE_EXPONENT's own comment for why 2 read as an
    unmistakably squished circle once a net's radius_y stretched it
    noticeably away from radius, in a way a rounder-cornered rectangle
    doesn't). Nodes fill *more* of the unit square with the higher
    exponent (more area lands inside the boundary before stretching), so
    a squircle-shaped net of a given radius holds more nodes at the same
    spacing than the same radius would as a true ellipse.

    shape="parabola" (only FINAL_STAGE uses this) instead bounds the
    region on its right/far side with a plain flat cutoff at x=radius --
    that edge is meant to sit off-screen, so it doesn't need any curve at
    all -- and on its left/near side (the one touching whatever this net
    grew out of) with a leftward-opening parabola, x = -PARABOLA_WRAP_
    FRAC*radius*(y/radius_y)^2. That boundary sits exactly at x=0 (the
    touch point) at y=0, and swings to negative x -- behind the touch
    point -- as |y| grows toward +-radius_y, so the two arms sweep
    around/envelop whatever sits at the touch point (rather than opening
    away from it) without crossing x=0 at y=0 itself, where that thing
    (see PARABOLA_WRAP_FRAC's own comment for why this stays clear of
    FINAL_STAGE's own arrow) actually is. Sampled with y as the free
    variable (not x, unlike every other shape here) since x's own valid
    range now depends on y, not the reverse, and that range can dip
    negative, unlike every other shape's [0, 1]/[-1, 1] ranges -- see
    PARABOLA_WRAP_FRAC for the clamp that keeps it from reaching back far
    enough to actually overlap that touch point. center + these points
    (see build_net) therefore places x=0, y=0 -- the near touch point,
    not a centroid -- exactly at `center`."""
    radius_y = radius if radius_y is None else radius_y
    min_dist = max(min(radius, radius_y) * min_dist_factor, spacing_mult * node_radius)
    rng = random.Random(seed)
    points = []
    tries = 0
    while len(points) < n and tries < MAX_SAMPLE_TRIES:
        tries += 1
        if shape == "parabola":
            y = rng.uniform(-1, 1)
            x_left = -PARABOLA_WRAP_FRAC * y * y
            x = rng.uniform(x_left, 1)
        else:
            x, y = rng.uniform(-1, 1), rng.uniform(-1, 1)
            if abs(x) ** SQUIRCLE_EXPONENT + abs(y) ** SQUIRCLE_EXPONENT > 1:
                continue
        candidate = np.array([x * radius, y * radius_y, 0])
        if all(np.linalg.norm(candidate - p) > min_dist for p in points):
            points.append(candidate)
    return points


def _point_blocks_segment(point, a, b, threshold):
    """Whether `point` sits close enough to the a-b segment (its closest
    approach along the segment itself, not the infinite line through it)
    that an edge drawn from a to b would visually cut through point's own
    node circle."""
    seg = b - a
    seg_len2 = seg @ seg
    if seg_len2 < 1e-12:
        return False
    t = np.clip(((point - a) @ seg) / seg_len2, 0, 1)
    closest = a + t * seg
    return np.linalg.norm(point - closest) < threshold


def _segment_crosses_box(a, b, box):
    """Whether the a-b segment enters the axis-aligned box (x0, x1, y0,
    y1) anywhere along its length -- not just whether either endpoint
    sits inside it, so an edge that merely passes *through* the box
    (both endpoints outside, on opposite sides) still counts. Standard
    Liang-Barsky segment/rectangle clip: walk the segment's parameter
    range t in [0, 1] down to whatever sub-range satisfies all four of
    the box's half-plane constraints at once; a non-empty range left at
    the end means some point on the segment satisfies all four
    simultaneously, i.e. sits inside the box."""
    x0, x1, y0, y1 = box
    dx, dy = b[0] - a[0], b[1] - a[1]
    t_min, t_max = 0.0, 1.0
    for p, q in ((-dx, a[0] - x0), (dx, x1 - a[0]), (-dy, a[1] - y0), (dy, y1 - a[1])):
        if p == 0:
            if q < 0:
                return False
        else:
            t = q / p
            if p < 0:
                if t > t_max:
                    return False
                t_min = max(t_min, t)
            else:
                if t < t_min:
                    return False
                t_max = min(t_max, t)
    return t_min <= t_max


def nearest_neighbor_edges(points, k, node_radius, block_margin=1.2, keep_out=None):
    """Each point connects to its k nearest neighbors (deduplicated) --
    a dense but locally-organized mesh, matching the reference nets'
    web-like (not fully-connected) look. A candidate edge is skipped (the
    node tries its next-nearest neighbor instead) if some third node sits
    close enough to that edge's own path to visually read as the edge
    passing through it -- otherwise, with a node roughly between two
    others (increasingly likely at the tighter spacing_mult packing STAGES
    now uses), the two outer nodes' edge visually cuts across the middle
    one despite there being no real edge to it. block_margin pads
    node_radius slightly (rather than using it bare) so an edge merely
    grazing a node's outer rim still counts as blocked.

    keep_out, if given, is an (x0, x1, y0, y1) box (see FINAL_STAGE's own
    arrow-clearance box, built in build_net) that no edge is allowed to
    cross either, on top of the node-blocking check above -- otherwise
    two nodes that are each other's nearest neighbor by raw distance but
    sit on opposite sides of whatever occupies that box (FINAL_STAGE's
    own arrow, which -- unlike every other node -- can't ever block a
    segment on its own, since it isn't a point in `points` to begin
    with) get connected straight through it."""
    threshold = node_radius * block_margin
    edges = set()
    for i, p in enumerate(points):
        order = sorted(range(len(points)), key=lambda j: np.linalg.norm(points[j] - p))
        count = 0
        for j in order:
            if j == i:
                continue
            blocked = any(
                m != i and m != j and _point_blocks_segment(points[m], p, points[j], threshold)
                for m in range(len(points))
            )
            if not blocked and keep_out is not None:
                blocked = _segment_crosses_box(p, points[j], keep_out)
            if blocked:
                continue
            edges.add(tuple(sorted((i, j))))
            count += 1
            if count >= k:
                break
    return edges


def _find_articulation_points(n, adjacency):
    """Tarjan's cut-vertex algorithm: nodes whose removal would split the
    graph into two or more pieces. Stronger than checking for bridges
    (single cut *edges*) -- two branches can each be internally
    cycle-rich, with no bridge anywhere in either one, and still only
    ever meet at one shared node, so every path between them still has
    to pass through it. That shared node is exactly what this finds."""
    disc = [-1] * n
    low = [-1] * n
    points = set()
    timer = [0]

    def dfs(u, parent):
        children = 0
        disc[u] = low[u] = timer[0]
        timer[0] += 1
        for v in adjacency[u]:
            if v == parent:
                continue
            if disc[v] == -1:
                children += 1
                dfs(v, u)
                low[u] = min(low[u], low[v])
                if parent is not None and low[v] >= disc[u]:
                    points.add(u)
            else:
                low[u] = min(low[u], disc[v])
        if parent is None and children > 1:
            points.add(u)

    for start in range(n):
        if disc[start] == -1:
            dfs(start, None)
    return points


def _connected_components(n, adjacency):
    seen = [False] * n
    components = []
    for start in range(n):
        if seen[start]:
            continue
        comp = {start}
        seen[start] = True
        stack = [start]
        while stack:
            u = stack.pop()
            for v in adjacency[u]:
                if not seen[v]:
                    seen[v] = True
                    comp.add(v)
                    stack.append(v)
        components.append(comp)
    return components


def _components_excluding_node(n, adjacency, removed):
    """The pieces the graph splits into with `removed` taken out -- the
    actual fallout of `removed` being a cut vertex, computed directly
    (BFS/DFS over the rest) rather than reasoned about from DFS-tree
    structure, so it's correct regardless of how many separate branches
    hang off `removed` or how deep they go."""
    seen = {removed}
    components = []
    for start in range(n):
        if start in seen:
            continue
        comp = {start}
        seen.add(start)
        stack = [start]
        while stack:
            u = stack.pop()
            for v in adjacency[u]:
                if v != removed and v not in seen:
                    seen.add(v)
                    comp.add(v)
                    stack.append(v)
        components.append(comp)
    return components


def _ensure_two_vertex_connected(points, edges, node_radius, block_margin=1.2, top_k=15, keep_out=None):
    """Adds edges (never removes any) until the net has no cut vertices
    left -- every pair of nodes ends up with two independent paths that
    share no *nodes* in common (not just no shared edge), so there's
    never a single node whose removal -- or, visually, a single point of
    convergence -- is the only thing holding two halves of the net
    together. For each cut vertex found, this reconnects every piece
    the graph would split into without it (see _components_excluding_
    node) via a chain of cross-piece edges that don't touch the cut
    vertex itself -- once those pieces are mutually reachable on their
    own, the cut vertex trivially stops being one. Implies no bridges
    either (for n>=3): a bridge's non-leaf endpoint is always itself a
    cut vertex, so eliminating cut vertices eliminates bridges as a
    side effect, without needing a separate bridge pass. Nearest-valid
    (not already an edge, not visually cutting through some third node
    -- see nearest_neighbor_edges' own blocking check, reused here)
    cross-piece pair wins each time, so added edges stay short and
    locally plausible rather than long diagonal shortcuts. Distances for
    a candidate pairing are computed once via numpy (cheap even for a
    few hundred points); only the top_k nearest candidates then pay for
    the O(n) occlusion check, rather than every pair in the (possibly
    large) cross product -- the nearest candidate passes that check
    often enough that this stays fast even on FINAL_STAGE's 432 nodes.

    keep_out is forwarded to `blocked` unchanged from nearest_neighbor_
    edges' own parameter of the same name -- see its docstring. Without
    this, a cross-piece pair straddling FINAL_STAGE's own arrow (its two
    pieces disconnected, or reconnected only through a cut vertex,
    specifically because nothing is allowed to cross that gap) is
    exactly the scenario this function exists to fix, and it would
    "fix" it by drawing an edge straight through the arrow -- the one
    thing every other candidate edge in this net is already forbidden
    from doing."""
    n = len(points)
    points_arr = np.array(points)
    threshold = node_radius * block_margin
    edges = set(edges)

    def blocked(a, b):
        if keep_out is not None and _segment_crosses_box(points_arr[a], points_arr[b], keep_out):
            return True
        return any(
            m != a and m != b and _point_blocks_segment(points_arr[m], points_arr[a], points_arr[b], threshold)
            for m in range(n)
        )

    def nearest_cross_edge(group_a, group_b):
        a_idx = np.array(sorted(group_a))
        b_idx = np.array(sorted(group_b))
        if len(a_idx) == 0 or len(b_idx) == 0:
            return None
        dists = np.linalg.norm(points_arr[a_idx][:, None, :] - points_arr[b_idx][None, :, :], axis=2)
        for flat in np.argsort(dists, axis=None)[:top_k]:
            ai, bi = np.unravel_index(flat, dists.shape)
            a, b = int(a_idx[ai]), int(b_idx[bi])
            key = tuple(sorted((a, b)))
            if key in edges or blocked(a, b):
                continue
            return key
        return None

    def adjacency_from(edge_set):
        adj = {i: [] for i in range(n)}
        for i, j in edge_set:
            adj[i].append(j)
            adj[j].append(i)
        return adj

    # 1. Merge separate components into one first -- two node-disjoint
    # paths presumes there's at least one path to begin with. Very
    # unlikely to ever fire given how nearest_neighbor_edges builds this
    # net, but cheap to guarantee rather than assume.
    for _ in range(n):
        components = _connected_components(n, adjacency_from(edges))
        if len(components) <= 1:
            break
        rest = set().union(*components[1:])
        new_edge = nearest_cross_edge(components[0], rest)
        if new_edge is None:
            break
        edges.add(new_edge)

    # 2. Eliminate cut vertices so no single node is the only thing
    # connecting two halves of the net. Re-finds cut vertices after
    # every round rather than assuming one pass clears them all -- an
    # added edge can leave a *different* node still a cut vertex (or,
    # rarely, not fully resolve the one it targeted if that side has
    # further internal cut vertices of its own).
    for _ in range(n):
        adjacency = adjacency_from(edges)
        cuts = _find_articulation_points(n, adjacency)
        if not cuts:
            break
        progress = False
        for cut in cuts:
            pieces = _components_excluding_node(n, adjacency, cut)
            if len(pieces) <= 1:
                continue
            for k in range(len(pieces) - 1):
                new_edge = nearest_cross_edge(pieces[k], pieces[k + 1])
                if new_edge is not None:
                    edges.add(new_edge)
                    progress = True
        if not progress:
            # Every remaining cut vertex has no addable alternate (every
            # candidate is either already an edge or visually blocked) --
            # accept it rather than loop forever chasing the impossible.
            break
    return edges


def build_net(
    n_nodes,
    cloud_radius,
    k_neighbors,
    node_radius,
    seed,
    center,
    palette,
    edge_color,
    spacing_mult=3.0,
    radius_y=None,
    shape="squircle",
):
    # min_dist_factor forced to 0 (below sample_cloud's own default) --
    # its radius*min_dist_factor term scales *with* radius, so at the
    # radii net_radius() now produces it would dominate over the
    # spacing_mult*node_radius floor and silently turn back into a
    # scale-invariant packing problem (bigger radius demanding
    # proportionally bigger spacing too, so it never actually buys room
    # for more nodes -- confirmed directly: every STAGES radius past
    # n_nodes=16 plateaued at the same ~25 placed nodes regardless of how
    # large net_radius() made the radius). Forcing it to 0 keeps spacing
    # pinned to the fixed spacing_mult*node_radius floor net_radius() was
    # actually solved against.
    local_points = sample_cloud(
        n_nodes,
        cloud_radius,
        node_radius,
        seed,
        min_dist_factor=0.0,
        spacing_mult=spacing_mult,
        radius_y=radius_y,
        shape=shape,
    )
    points = [center + p for p in local_points]

    nodes_list = [make_node(p, node_radius, palette) for p in points]
    nodes_group = VGroup(*nodes_list)
    # shape="parabola" nets (FINAL_STAGE only) sit right up against their
    # own arrow, with the pinch point at local (0, 0) -- and, unlike every
    # other node an edge might cut through, the arrow itself never appears
    # in local_points to naturally block a segment via the nodes_list loop
    # above, since it isn't a node. keep_out is that arrow's own
    # silhouette in this same local coordinate system (see make_arrow/
    # flow_arrows for GAP, ARROW_LENGTH, ARROW_TIP_HALF_HEIGHT -- the
    # constants that place and size it), padded the same way
    # _point_blocks_segment pads a node, so an edge can't even graze it.
    keep_out = None
    if shape == "parabola":
        margin = node_radius * 1.2
        keep_out = (
            -(ARROW_LENGTH + GAP) - margin,
            -GAP + margin,
            -ARROW_TIP_HALF_HEIGHT - margin,
            ARROW_TIP_HALF_HEIGHT + margin,
        )
    edge_indices = nearest_neighbor_edges(local_points, k_neighbors, node_radius, keep_out=keep_out)
    edge_indices = _ensure_two_vertex_connected(local_points, edge_indices, node_radius, keep_out=keep_out)
    edges_group = VGroup(*[make_edge(nodes_list[i], nodes_list[j], edge_color) for i, j in edge_indices])

    if SIMPLE_STYLE:
        glow = Group()
    else:
        # max(), not cloud_radius alone -- make_glow_blob's own span needs
        # to cover whichever axis the net actually extends further along,
        # not just the nominal (horizontal) cloud_radius, or a vertically
        # stretched net's blob would clip its own top/bottom.
        glow = make_glow_blob(local_points, edge_indices, max(cloud_radius, radius_y or cloud_radius), node_radius)
        glow.move_to(center)

    # Per-node real glow discs (see add_soft_bloom) -- extras, not
    # nodes_group, since ImageMobject can't live inside nodes_group's
    # VGroup. Pulse chains (see add_pulse_chains) aren't started here:
    # that's left to the Scene's grow_in, so a ping can't fire on a net
    # that's still growing in.
    extras = Group()
    if not SIMPLE_STYLE:
        for node in nodes_list:
            add_soft_bloom(node, node_radius, palette[2], extras)

    # Pulse rings (see spawn_pulse_ring) live in their own Group, a total
    # sibling of both extras and net -- never nested inside either one --
    # stashed as an attribute (extras.ring_holder), not a parameter
    # threaded through every call site, purely so add_pulse_chains/
    # stop_effects/resume_effects can reach it without changing their own
    # signatures. A ping can fire at literally any point in this net's
    # life -- mid-grow-in, mid-slide, mid-FadeOut -- and needs to add (or
    # later remove) a ring right then, not whenever it's next safe to.
    # ring_holder never being a family member of anything a Scene
    # animation targets (not extras, not the net Group below) is what
    # makes that safe: FadeIn(extras)/FadeOut(net)/net.animate.shift()
    # all walk a fixed family captured at that animation's own begin() --
    # changing a family's *size* while an animation is mid-interpolation
    # over exactly that family crashes manim outright (confirmed against
    # an actual render: "zip() argument 2 is shorter than argument 1")
    # -- and ring_holder simply never being part of any such family means
    # adding/removing a ring can never be that change, regardless of
    # what's simultaneously playing on the rest of the net. A ring's own
    # position still tracks its node correctly throughout any of that
    # (see spawn_pulse_ring's own tick -- move_to(node.get_center()) runs
    # every frame off the node's own live, currently-animating position,
    # not off any position ring_holder itself would have inherited by
    # being a passenger in the net's own Group).
    ring_holder = Group()
    extras.ring_holder = ring_holder

    # Layering is pinned with explicit z_index rather than left to
    # Group/VGroup insertion order: once individual submobjects here
    # (a single node, a single edge) get animated separately by name in
    # a Scene's self.play() -- exactly what grow_in does -- manim can
    # permanently disturb their relative paint order from then on,
    # confirmed by direct pixel inspection (a static self.add() of this
    # same net always painted correctly; the identical net run through
    # so much as one self.play() targeting its nodes/edges individually
    # did not, regardless of which order they were passed to self.play
    # in). z_index sorting happens after that disturbance and overrides
    # it outright, so it's the only reliable way to guarantee edges
    # never paint over nodes.
    glow.set_z_index(0)
    edges_group.set_z_index(1)
    extras.set_z_index(2)
    ring_holder.set_z_index(2)
    nodes_group.set_z_index(3)

    # Group rather than VGroup: it needs to hold ImageMobjects (glow,
    # extras) alongside the VMobject-based edges/nodes, which VGroup
    # rejects. ring_holder deliberately left out (see its own comment
    # above) -- existing FadeOut/.animate.shift() calls on the whole net
    # move and fade everything else together without any extra code, but
    # rings track their own node's live position directly (see
    # spawn_pulse_ring's own tick) and fade out on their own schedule
    # regardless of whatever the net itself is doing, including once the
    # net's own FadeOut has already finished and removed it.
    net = Group(glow, edges_group, extras, nodes_group)
    return net, nodes_group, edges_group, glow, extras


ARROW_SHAFT_HALF_HEIGHT = 0.11  # shaft thickness, matching the old stroke_width=22 look
ARROW_TIP_HALF_HEIGHT = 0.175  # arrowhead half-width, matching the old tip_length=0.35 look
ARROW_TIP_LENGTH = 0.35


def make_arrow(start_x, end_x):
    """A thick, chunky block arrow -- a single filled polygon (shaft
    rectangle merged with a triangular head into one continuous outline)
    rather than manim's own Arrow, which pairs a separately-rendered Line
    and ArrowTip. Two overlapping shapes of the same color always show a
    seam where they meet at full opacity, and -- worse -- a visibly
    different shade in the overlap while fading out, since two
    stacked semi-transparent layers of the same color never composite to
    look like a single layer at that opacity. A single shape has neither
    problem: there's exactly one fill, so exactly one opacity, everywhere,
    always.
    """
    tip_length = min(ARROW_TIP_LENGTH, abs(end_x - start_x) * 0.6)
    base_x = end_x - tip_length if end_x >= start_x else end_x + tip_length
    points = [
        [start_x, ARROW_SHAFT_HALF_HEIGHT, 0],
        [base_x, ARROW_SHAFT_HALF_HEIGHT, 0],
        [base_x, ARROW_TIP_HALF_HEIGHT, 0],
        [end_x, 0, 0],
        [base_x, -ARROW_TIP_HALF_HEIGHT, 0],
        [base_x, -ARROW_SHAFT_HALF_HEIGHT, 0],
        [start_x, -ARROW_SHAFT_HALF_HEIGHT, 0],
    ]
    arrow = Polygon(*points, color=ARROW_COLOR, fill_opacity=1, stroke_width=0)
    # GrowArrow (used elsewhere for manim's own Arrow) needs an Arrow
    # instance specifically; grow_arrow() below animates any mobject from
    # a point instead, so the tail position is stashed here rather than
    # re-derived from the polygon's bounding box (whose vertical center
    # is skewed toward the wider tip, not the shaft's true centerline).
    arrow.tail_point = np.array([start_x, 0.0, 0.0])
    return arrow


def grow_arrow(arrow, **kwargs):
    """Grow one of this file's own block arrows (see make_arrow) from its
    tail -- the same visual beat as manim's GrowArrow, which only accepts
    its own Arrow class and so can't be used on our plain Polygon."""
    return GrowFromPoint(arrow, arrow.tail_point, **kwargs)


def flow_arrows(left_net_edge, right_net_edge, brace, code):
    """The left (net -> code) and right (code -> net) arrows for one lap.
    Recomputed every time from the actual current geometry (net radii and
    code/brace width both change every cycle), padded by the fixed GAP on
    every one of its four sides (net, arrow, brace/code, arrow, net) --
    rather than a fraction of the available space -- so that gap reads the
    same at every lap regardless of how big the flanking nets are; only
    the arrows' own length absorbs whatever space is left over."""
    brace_left_edge = brace.get_left()[0]
    code_right_edge = code.get_right()[0]
    left_arrow = make_arrow(left_net_edge + GAP, brace_left_edge - GAP)
    right_arrow = make_arrow(code_right_edge + GAP, right_net_edge - GAP)
    return left_arrow, right_arrow


class RecursiveSelfImprovement(ThreeDScene):
    def play(self, *args, **kwargs):
        # INSTANT forces every self.play() call, everywhere in this
        # scene, into a single rendered frame -- overriding whatever
        # run_time each call already asked for, since Scene.compile_
        # animations applies any run_time passed here to every top-level
        # animation via plain setattr, not by rescaling anything. Each
        # animation's own interpolate() still walks its rate_func from
        # alpha=0 to alpha=1 exactly as it always did -- alpha=1 still
        # means "fully settled" -- it now just does that walk across one
        # frame's worth of clock time instead of its scripted duration,
        # so every beat lands in its real final state, just instantly.
        #
        # Except a bare Wait: self.wait() (which hold() is built on)
        # itself goes through self.play(Wait(run_time=duration, ...)) --
        # indistinguishable, at this level, from any other self.play()
        # call -- so without this exclusion INSTANT would silently zero
        # out every hold() too, leaving nothing to actually look at.
        is_bare_wait = len(args) == 1 and isinstance(args[0], Wait)
        if INSTANT and not is_bare_wait:
            kwargs["run_time"] = 1 / config.frame_rate
        if DEBUG_PULSE:
            kinds = ", ".join(type(a).__name__ for a in args)
            print(
                f"ANIM begin kinds=[{kinds}] run_time_kwarg={kwargs.get('run_time')} "
                f"is_bare_wait={is_bare_wait}",
                flush=True,
            )
        super().play(*args, **kwargs)
        if DEBUG_PULSE:
            print("ANIM end", flush=True)
        if DEMO_SECONDS > 0:
            # Every non-wait call in this script passes run_time= explicitly,
            # which Scene.compile_animations applies to every top-level
            # animation uniformly -- so kwargs["run_time"] is exactly this
            # call's own duration. A bare Wait (hold()) never gets a
            # run_time kwarg from Scene.wait, so its duration is read off
            # the Wait animation itself instead.
            duration = kwargs.get("run_time")
            if duration is None and is_bare_wait:
                duration = args[0].run_time
            self._demo_elapsed += duration or 0
            if self._demo_elapsed >= DEMO_SECONDS:
                raise _DemoLimitReached()

    def hold(self, duration):
        """A pause where nothing is being grown, written, or shifted --
        just a beat to let the current frame sit. Clamped to a short
        stand-in length in fast-preview mode (see FULL_TIMING) since these
        contribute render time without any visual change to show for it."""
        self.wait(duration if FULL_TIMING else min(duration, IDLE_HOLD))

    def grow_in(self, nodes_group, edges_group, glow, extras, node_radius, palette, run_time=1.1):
        # Nodes and edges both grow within this single self.play() call --
        # rather than nodes finishing before edges even begin -- so the
        # net reads as growing in all at once. The ambient glow blob and
        # per-node bloom discs (extras) fade in alongside.
        #
        # Each edge's own fade-in is delayed by how late its slower
        # (later-staggered) endpoint node starts growing, rather than
        # fading in on its own independent stagger -- otherwise an edge
        # routinely reaches full opacity while the node it's headed to
        # is still a barely-grown dot, and visibly overshoots past that
        # node's current (tiny) radius out into space it hasn't grown
        # into yet. That reads as the edge rendering on top of the node,
        # even though paint order already puts nodes in front -- z-order
        # can't hide something plainly outside the node's current bounds.
        # Wait/FadeIn durations are computed here in explicit seconds
        # (rather than left as implicit fractions for self.play's own
        # run_time= to rescale) so this doesn't depend on guessing how
        # manim distributes a shared run_time across differently-shaped
        # nested animations -- each Succession's own two parts already
        # sum to exactly run_time.
        node_list = list(nodes_group)
        index_of = {id(n): i for i, n in enumerate(node_list)}
        last_index = max(len(node_list) - 1, 1)
        edge_fade_ins = []
        # CHAIN_TEST skips fading edges in at all -- they still exist in
        # edges_group (add_pulse_chains needs them for its adjacency
        # graph), just never get animated/added to the scene, so they're
        # never drawn.
        if not CHAIN_TEST:
            for edge in edges_group:
                i, j = index_of[id(edge.node_a)], index_of[id(edge.node_b)]
                delay_frac = max(i, j) / last_index
                wait_time = delay_frac * run_time * 0.7
                edge_fade_ins.append(Succession(Wait(wait_time), FadeIn(edge, run_time=run_time - wait_time)))

        # Started before the grow-in's own self.play (rather than after
        # it finishes) so the first pings can fire while nodes are still
        # scaling up from GrowFromCenter, not only once the net is fully
        # formed. Runs under SIMPLE_STYLE too -- see fire()'s own
        # SIMPLE_STYLE check, which keeps the flash but drops the
        # expanding ring.
        #
        # No suspend_mobject_updating=False needed on FadeIn(extras) or the
        # per-node GrowFromCenter below (an earlier version of this used
        # that, to stop Animation.begin()'s default suspend_updating() from
        # freezing the scheduler/flash for the whole grow-in) -- it caused
        # a worse problem than the one it fixed: Animation.update_mobjects
        # always ticks its own starting_mobject (a Mobject.copy(), i.e.
        # copy.deepcopy(extras) or copy.deepcopy(node)) once per frame in
        # addition to the Scene's own per-frame update pass, and since
        # plain functions are atomic under deepcopy (the same function
        # object comes back, not an independent one), that "copy" carries
        # the exact same scheduler/flash closures, over the exact same
        # mutable state, as the real one -- so it fires for real, a second
        # time, on the real net, for as long as that particular animation
        # runs. Confirmed directly: an instrumented run logged 3-4x as many
        # scheduler ticks per rendered frame during grow-in/slide as during
        # a plain hold. add_pulse_chains' own driver mobject sidesteps this
        # instead of fighting it -- see its docstring -- so nothing here
        # needs to touch suspend_mobject_updating at all any more.
        add_pulse_chains(self, nodes_group, edges_group, palette, extras)
        # AnimationGroup() with zero subanimations raises outright (see
        # manim's own Animation.begin()) rather than just being a no-op --
        # CHAIN_TEST leaves edge_fade_ins empty on purpose (see above), so
        # that piece is only included at all when there's something in it.
        top_level = [FadeIn(glow), FadeIn(extras)]
        if edge_fade_ins:
            top_level.append(AnimationGroup(*edge_fade_ins))
        top_level.append(LaggedStart(*[GrowFromCenter(n) for n in node_list], lag_ratio=0.04))
        self.play(*top_level, run_time=run_time)

    def construct(self):
        # Seeds the *global* random module, not just each net's own
        # sample_cloud (which already seeds its own isolated
        # random.Random(seed) instance and so was already deterministic
        # on its own) -- add_pulse_chains' own hop selection
        # (random.randint/random.choice, see its own scheduler) draws
        # from this global stream instead, which is otherwise seeded from
        # OS entropy at interpreter start and so differs every process
        # run. Harmless for a single continuous render (nobody notices
        # one run's ping pattern differing from another's), but it means
        # two SEPARATE invocations of this same scene -- e.g. rendering
        # it in chunks via manim's own -n <from>,<to> flag, one process
        # per chunk -- would each reconstruct the skipped, unrendered
        # portion with a *different* random pulse-chain history than
        # whatever an earlier chunk's process actually rendered, even
        # though every other part of the scene (positions, timing,
        # growth) is already fully deterministic. Fixing this one spot
        # makes the whole scene reproducible frame-for-frame across
        # however many separate processes render however many chunks of
        # it, not just internally consistent within any single run.
        random.seed(20260722)
        self._demo_elapsed = 0.0
        self.camera.background_color = BACKGROUND_COLOR
        self.set_camera_orientation(phi=0 * DEGREES, theta=-90 * DEGREES)

        # One backdrop for the whole video, shared by both mini-movies --
        # not recreated per part -- so neither part's ending has any
        # reason to fade it out. Only the icosahedron finale's own camera
        # move (see construct_main) still needs to fade it, since that's
        # a real technical constraint (fixed-in-frame content misbehaves
        # once the camera moves), not a stylistic choice.
        backdrop = make_backdrop()
        self.add_fixed_in_frame_mobjects(backdrop)

        # Mini-movie 1: a short, standalone taste of the same growth
        # pattern (net -> code -> bigger net) used throughout mini-movie
        # 2, then a beat of plain blank background as a hard cut between
        # the two, then mini-movie 2 -- the full chain -- runs unchanged.
        try:
            if not MAIN_ONLY and not FINAL_ONLY and not LAP_ONLY:
                self.construct_intro()
                if INTRO_ONLY:
                    return
                self.hold(1.0)
            self.construct_main(backdrop)
        except _DemoLimitReached:
            pass

    def construct_intro(self):
        """Mini-movie 1: one simple lap -- a small blue net grows in,
        writes an ordinary training step, and a bigger red net grows in
        response -- then everything clears for the blank cut before
        mini-movie 2. Blue and red rather than the main chain's own
        green-to-magenta progression, since this intro isn't actually
        part of that chain -- it's a self-contained preview of the same
        beat, not its first lap in disguise."""
        m = SPEED_MULTIPLIERS[0]

        net0_radius = STAGES[0]["cloud_radius"]
        net1_radius = INTRO_END_STAGE["cloud_radius"]
        shift = lap_shift(net0_radius, net1_radius, CODE_STAGE_1)

        # Built now (rather than after net0 grows in) purely to read its
        # geometry -- net0's position depends on where this code's brace
        # will sit, not the other way around. It isn't displayed until
        # the Write() below.
        code = make_code_block(CODE_STAGE_1, np.array([MID_X + shift, 0, 0]))
        brace = make_bracket(code, buff=0.25, color=ARROW_COLOR)

        net0, nodes0, edges0, glow0, extras0 = build_net(
            center=np.array([left_net_center(net0_radius, brace.get_left()[0]), 0, 0]),
            palette=BLUE_PALETTE,
            edge_color=BLUE_EDGE,
            **STAGES[0],
        )
        self.grow_in(nodes0, edges0, glow0, extras0, STAGES[0]["node_radius"], BLUE_PALETTE, run_time=1.1 * m)
        self.hold(0.4 * m)

        left_arrow, right_arrow = flow_arrows(
            left_facing_edge(brace.get_left()[0]), right_facing_edge(code.get_right()[0]), brace, code
        )

        self.play(grow_arrow(left_arrow), run_time=0.5 * m)
        self.play(
            write_code(code, m),
            GrowFromCenter(brace, run_time=0.35 * m),
        )
        self.hold(0.4 * m)
        self.play(grow_arrow(right_arrow), run_time=0.5 * m)

        net1, nodes1, edges1, glow1, extras1 = build_net(
            center=np.array([right_net_center(net1_radius, code.get_right()[0]), 0, 0]),
            palette=RED_PALETTE,
            edge_color=RED_EDGE,
            **INTRO_END_STAGE,
        )
        self.grow_in(
            nodes1, edges1, glow1, extras1, INTRO_END_STAGE["node_radius"], RED_PALETTE, run_time=max(1.3 * m, 0.5)
        )
        self.hold(5.0)

        # Each piece fades out on its own rather than as one synchronized
        # block, and the backdrop stays put throughout -- so this reads
        # as the scene's pieces settling away, not the whole picture
        # (background glow included) dimming to black.
        # stop_effects freezes bloom-disc tracking; retire_pings stops
        # each net from starting any *new* chain and freezes whatever
        # rings it still has in flight (see its own docstring) -- both
        # nets are gone for good here, not just repositioning, so both
        # get it, and both their ring_holders are included in the
        # FadeOut below so those frozen rings fade smoothly away with
        # everything else instead of being left behind, fully visible.
        stop_effects(extras0)
        stop_effects(extras1)
        retire_pings(extras0)
        retire_pings(extras1)
        # ring_holder FadeOuts run as plain siblings of the LaggedStart,
        # not nested inside it -- inside, being last in the list, they'd
        # only start staggered near the *end* of the sequence (lag_ratio
        # delays each item's own start relative to the last), fading
        # only across whatever sliver of run_time was left by then. As
        # siblings they get this same self.play's run_time directly
        # (Scene.compile_animations applies a shared run_time to every
        # top-level animation uniformly -- see RecursiveSelfImprovement.
        # play's own comment on INSTANT for the same mechanism), so any
        # ring still around fades across the *entire* span of the
        # sequence instead of being crammed into the tail end of it --
        # confirmed directly against an actual render: nested inside,
        # the rest of a net had already faded to nothing well before its
        # own rings even started fading, reading as leftover rings
        # floating with nothing left to anchor them.
        self.play(
            LaggedStart(
                FadeOut(net0),
                FadeOut(net1),
                FadeOut(code),
                FadeOut(brace),
                FadeOut(left_arrow),
                FadeOut(right_arrow),
                lag_ratio=0.2,
            ),
            FadeOut(extras0.ring_holder),
            FadeOut(extras1.ring_holder),
            run_time=max(1.2 * m, 0.6),
        )

    def construct_main(self, backdrop):
        # Code the *next* lap will show, one entry per lap in order,
        # ending with the final lap's -- known up front since every
        # code block is static content, so a net can be slid straight to
        # where its own next arrow will need it (see the lookahead below)
        # instead of wherever this lap's different code would have put it.
        lap_code_lines = CODE_STAGES + [CODE_STAGE_FINAL]

        # Every net radius in growth order, plus the radius that stands
        # in for the icosahedron when the finale is skipped -- lap_radii[k]
        # and lap_radii[k+1] are exactly the (left, right) pair of nets
        # flanking lap (k+1) (or the final lap, for k=4). Known up front
        # for the same reason as lap_code_lines: it lets lap_shift look
        # ahead to a net's *next* lap, not just its current one.
        lap_radii = [stage["cloud_radius"] for stage in STAGES] + [FINAL_STAGE_RADIUS]

        if FINAL_ONLY:
            # Skip net 0's own grow-in and every earlier stage's lap --
            # jump straight to a net standing in the exact spot the
            # skipped loop's own last iteration would have slid its net
            # into (same position math as new_left_center/next_shift
            # below, with is_final_next always True since this always
            # stands in for the lap right before the final one), grown in
            # near-instantly rather than at its normal pace. The final
            # lap right below this still runs completely unchanged --
            # arrow, code, arrow, the huge final net growing in from
            # off-screen, then the hold and fade-away.
            stage = STAGES[-1]
            if CHAIN_TEST:
                # A copy, not a mutation -- STAGES[-1] is a shared
                # module-level dict, reused verbatim by the normal (non-
                # FINAL_ONLY) loop too.
                stage = dict(stage, node_radius=stage["node_radius"] * CHAIN_TEST_NODE_SCALE)
            stage_radius = stage["cloud_radius"]
            _, brace_left_final = code_edges_for(lap_code_lines[-1])
            current_net, current_nodes, current_edges, current_glow, current_extras = build_net(
                center=np.array([left_net_center(stage_radius, brace_left_final), 0, 0]),
                palette=NET_PALETTES[-1],
                edge_color=NET_EDGE_COLORS[-1],
                **stage,
            )
            self.grow_in(
                current_nodes, current_edges, current_glow, current_extras, stage["node_radius"], NET_PALETTES[-1],
                run_time=0.05,
            )
        elif LAP_ONLY:
            # Net 0 placed near-instantly rather than at its normal pace
            # -- same position math the normal net-0 setup below uses,
            # just without paying to animate through it. The loop right
            # below this still runs completely unchanged going into lap
            # 1 -- its own arrow/code/arrow build-up is what gets skipped
            # there (see the loop's own i==1 check), not anything here.
            shift = lap_shift(lap_radii[0], lap_radii[1], lap_code_lines[0])
            _, brace_left0 = code_edges_for(lap_code_lines[0])
            current_net, current_nodes, current_edges, current_glow, current_extras = build_net(
                center=np.array([left_net_center(STAGES[0]["cloud_radius"], brace_left0 + shift), 0, 0]),
                palette=NET_PALETTES[0],
                edge_color=NET_EDGE_COLORS[0],
                **STAGES[0],
            )
            self.grow_in(
                current_nodes, current_edges, current_glow, current_extras, STAGES[0]["node_radius"], NET_PALETTES[0],
                run_time=0.05,
            )
        else:
            # Net 0 spawns on the left, green for its whole lifetime --
            # grown in at the same unhurried pace as the first loop
            # iteration below, already positioned for lap 1's own code
            # (and shifted to keep lap 1's whole composition centered,
            # since net 0 and lap 1's own right-hand net are almost never
            # the same size) so its arrow is exactly ARROW_LENGTH from the
            # very first frame.
            shift = lap_shift(lap_radii[0], lap_radii[1], lap_code_lines[0])
            _, brace_left0 = code_edges_for(lap_code_lines[0])
            current_net, current_nodes, current_edges, current_glow, current_extras = build_net(
                center=np.array([left_net_center(STAGES[0]["cloud_radius"], brace_left0 + shift), 0, 0]),
                palette=NET_PALETTES[0],
                edge_color=NET_EDGE_COLORS[0],
                **STAGES[0],
            )
            self.grow_in(
                current_nodes,
                current_edges,
                current_glow,
                current_extras,
                STAGES[0]["node_radius"],
                NET_PALETTES[0],
                run_time=1.1 * SPEED_MULTIPLIERS[0],
            )
            self.hold(0.4 * SPEED_MULTIPLIERS[0])

        if not FINAL_ONLY:
            # Left net -> left arrow -> code typing in behind a brace ->
            # right arrow -> a bigger net (its own fixed color) grows on
            # the right -> the left net, code, and arrows vanish while the
            # right net slides into the left slot, becoming "current" for
            # the next lap. Every beat in the lap is scaled by that lap's
            # own multiplier, so early laps linger and later laps snap by
            # increasingly fast. Runs for LAP_ONLY too (only FINAL_ONLY
            # skips this loop outright) -- LAP_ONLY's own i==1/i==2 checks
            # below are what skip/stop the parts it doesn't want, not this
            # guard.
            for i, (stage, code_lines, m) in enumerate(zip(STAGES[1:], CODE_STAGES, SPEED_MULTIPLIERS), start=1):
                stage_radius = stage["cloud_radius"]
                shift = lap_shift(lap_radii[i - 1], lap_radii[i], code_lines)
                code = make_code_block(code_lines, np.array([MID_X + shift, 0, 0]))
                brace = make_bracket(code, buff=0.25, color=ARROW_COLOR)
                left_arrow, right_arrow = flow_arrows(
                    left_facing_edge(brace.get_left()[0]), right_facing_edge(code.get_right()[0]), brace, code
                )

                if LAP_ONLY and i == 1:
                    # The "before" this lap's own build-up leading up to
                    # the net the user actually wants to see -- placed
                    # instantly rather than animated, same reasoning as
                    # net 0 above: the slide right below still needs
                    # something concrete to fade away.
                    self.add(code, brace, left_arrow, right_arrow)
                else:
                    self.play(grow_arrow(left_arrow), run_time=0.5 * m)
                    if LAP_ONLY and i == 2:
                        # The "after" -- this next lap's own left arrow
                        # is the last beat the user wants; stop right
                        # here, before its own code/net get a chance to
                        # start.
                        return
                    self.play(
                        write_code(code, m),
                        GrowFromCenter(brace, run_time=0.35 * m),
                    )
                    self.hold(0.4 * m)

                    self.play(grow_arrow(right_arrow), run_time=0.5 * m)

                next_net, next_nodes, next_edges, next_glow, next_extras = build_net(
                    center=np.array([right_net_center(stage_radius, code.get_right()[0]), 0, 0]),
                    palette=NET_PALETTES[i],
                    edge_color=NET_EDGE_COLORS[i],
                    **stage,
                )
                self.grow_in(
                    next_nodes, next_edges, next_glow, next_extras, stage["node_radius"], NET_PALETTES[i],
                    run_time=max(1.3 * m, 0.5),
                )
                self.hold(0.5 * m)

                # next_net is only repositioning, not disappearing -- no
                # retire_pings for it, so it keeps pinging, new rings
                # included, completely unaffected right through the
                # slide (stop_effects here only freezes its bloom-disc
                # tracking, see stop_effects' own docstring).
                stop_effects(next_extras)
                # current_net, on the other hand, is fading out for
                # good -- retire_pings stops it from starting any *new*
                # chain and freezes whatever rings it still has in
                # flight (its ring_holder is included in the FadeOut
                # below, so those frozen rings fade away with it instead
                # of being left behind, fully visible).
                stop_effects(current_extras)
                retire_pings(current_extras)
                # The lap right after this one is the final, deliberately-
                # overflowing net -- centering the composition around it
                # would try to drag everything sideways to "balance" a net
                # many times bigger than its neighbor, instead of leaving
                # the rest of that lap alone and letting only the final
                # net spill off-screen. So no lap_shift lookahead for that
                # one; plain shift=0 instead, matching the final lap's own
                # below.
                is_final_next = (i + 1 == len(STAGES))
                next_shift = 0 if is_final_next else lap_shift(lap_radii[i], lap_radii[i + 1], lap_code_lines[i])
                _, next_brace_left = code_edges_for(lap_code_lines[i])
                new_left_center = left_net_center(stage_radius, next_brace_left + next_shift)
                slide_shift = new_left_center - right_net_center(stage_radius, code.get_right()[0])
                # Plain next_net.animate.shift() -- no
                # suspend_mobject_updating override needed (see
                # add_pulse_chains' own driver mobject and
                # add_node_flash's docstring for why a plain .animate() no
                # longer risks freezing or double-ticking anything).
                self.play(
                    FadeOut(current_net),
                    FadeOut(code),
                    FadeOut(brace),
                    FadeOut(left_arrow),
                    FadeOut(right_arrow),
                    FadeOut(current_extras.ring_holder),
                    next_net.animate.shift(np.array([slide_shift, 0, 0])),
                    run_time=max(1.0 * m, 0.45),
                )
                resume_effects(next_extras)
                current_net, current_nodes, current_edges, current_glow, current_extras = (
                    next_net, next_nodes, next_edges, next_glow, next_extras,
                )

        self.hold(0.4 * SPEED_MULTIPLIERS[-1])

        # Final lap: the magenta net writes its own code just like the
        # others -- fastest of all, the takeoff now nearly instantaneous --
        # but what grows in its place isn't another flat net -- it's the
        # icosahedron, red, still viewed face-on so it grows in at the
        # same right-hand spot the others did.
        #
        # No lap_shift here (plain shift=0, matching the loop's own
        # lookahead into this same lap) -- FINAL_STAGE_RADIUS is huge on
        # purpose (see FINAL_STAGE's own comment), and centering the
        # composition around it would shove the arrow/code/left-net
        # sideways to "balance" a net many times their size instead of
        # just letting it overflow off-screen on its own.
        shift = 0
        code = make_code_block(CODE_STAGE_FINAL, np.array([MID_X + shift, 0, 0]))
        brace = make_bracket(code, buff=0.25, color=ARROW_COLOR)
        ico_footprint_radius = FINAL_STAGE_RADIUS
        ico_center_x = right_net_center(ico_footprint_radius, code.get_right()[0])
        # The flat stand-in net's own touch point (INCLUDE_FINALE=0
        # below) -- unlike ico_center_x above, this isn't a circle's
        # center; FINAL_STAGE's parabola is pinched to a point at local
        # x=0, so putting that point (not a centroid) at this net's own
        # facing edge is what actually makes it touch the arrow, same as
        # every other net's left edge does.
        final_vertex_x = right_facing_edge(code.get_right()[0])
        left_arrow, right_arrow = flow_arrows(
            left_facing_edge(brace.get_left()[0]), right_facing_edge(code.get_right()[0]), brace, code
        )

        # CHAIN_TEST skips animating (and thereby ever displaying) the
        # arrows/code/brace entirely -- code/brace/arrows above are still
        # built since final_vertex_x/ico_center_x are solved from their
        # geometry, just never played, so they cost nothing and never
        # appear.
        if not CHAIN_TEST:
            self.play(grow_arrow(left_arrow), run_time=0.5 * FINAL_CODE_MULT)
            self.play(
                write_code(code, FINAL_CODE_MULT),
                GrowFromCenter(brace, run_time=0.35 * FINAL_CODE_MULT),
            )
            self.hold(0.4 * FINAL_CODE_MULT)
            self.play(grow_arrow(right_arrow), run_time=0.5 * FINAL_CODE_MULT)

        if INCLUDE_FINALE:
            ico_offset = np.array([ico_center_x, 0, 0])
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
            # current_net is gone for good here -- retire_pings winds its
            # pinging down and freezes whatever rings it still has in
            # flight, ready for the FadeOut below to fade them away
            # along with everything else instead of leaving them behind,
            # fully visible.
            stop_effects(current_extras)
            retire_pings(current_extras)
            ico_group = VGroup(ico_glow, ico_vertices, ico_edges)
            self.play(
                FadeOut(current_net),
                FadeOut(code),
                FadeOut(brace),
                FadeOut(left_arrow),
                FadeOut(right_arrow),
                FadeOut(backdrop),
                FadeOut(current_extras.ring_holder),
                ico_group.animate.shift(np.array([-ico_center_x, 0, 0])),
                run_time=1.2,
            )

            self.move_camera(phi=65 * DEGREES, theta=-50 * DEGREES, run_time=2.2)

            self.begin_ambient_camera_rotation(rate=0.18)
            self.wait(14.0 if FULL_TIMING else SPIN_HOLD)
            self.stop_ambient_camera_rotation()

            self.play(FadeOut(ico_group), run_time=1.5)
            self.hold(0.5)
        else:
            # Finale skipped (INCLUDE_FINALE=0, the default): the right
            # arrow still needs something to point at, so a final flat net
            # grows in the same right-hand spot every earlier net did --
            # red, like the icosahedron it stands in for -- pinned by its
            # own pinch point (see final_vertex_x above) so it touches
            # the arrow the same way every other net's left edge does.
            final_stage = FINAL_STAGE
            if CHAIN_TEST:
                # A copy, not a mutation -- FINAL_STAGE is a shared
                # module-level dict.
                final_stage = dict(FINAL_STAGE, node_radius=FINAL_STAGE["node_radius"] * CHAIN_TEST_NODE_SCALE)
            final_net, final_nodes, final_edges, final_glow, final_extras = build_net(
                center=np.array([final_vertex_x, 0, 0]), palette=RED_PALETTE, edge_color=RED_EDGE, **final_stage
            )
            self.grow_in(
                final_nodes, final_edges, final_glow, final_extras, final_stage["node_radius"], RED_PALETTE,
                run_time=max(1.3 * FINAL_CODE_MULT, 0.5),
            )
            self.hold(FINAL_HOLD_SECONDS)

            # CHAIN_TEST ends the render right here, right after the hold
            # -- skipping the closing fade-away entirely, since the point
            # of this mode is just to see the chains pinging as fast as
            # possible, not the full close-out beat.
            if not CHAIN_TEST:
                # Each piece fades out on its own rather than as one
                # synchronized block, and the backdrop stays put
                # throughout -- so this reads as the scene's pieces
                # settling away, not the whole picture (background glow
                # included) dimming to black. Both nets are gone for
                # good here, so both get retire_pings alongside
                # stop_effects -- pinging winds down and any rings still
                # in flight freeze, ready for their own ring_holders to
                # fade them away along with everything else instead of
                # leaving them behind, fully visible.
                #
                # ring_holder FadeOuts run as plain siblings of the
                # LaggedStart, not nested inside it -- inside, being
                # last in the list, they'd only start staggered near the
                # *end* of the sequence (lag_ratio delays each item's own
                # start relative to the last), fading only across
                # whatever sliver of run_time was left by then. As
                # siblings they get this self.play's run_time directly,
                # so any ring still around fades across the entire span
                # of the sequence instead of the tail end of it --
                # confirmed directly against an actual render: nested
                # inside, the rest of a net had already faded to nothing
                # well before its own rings even started fading, reading
                # as leftover rings floating with nothing left to anchor
                # them.
                stop_effects(current_extras)
                stop_effects(final_extras)
                retire_pings(current_extras)
                retire_pings(final_extras)
                self.play(
                    LaggedStart(
                        FadeOut(current_net),
                        FadeOut(code),
                        FadeOut(brace),
                        FadeOut(left_arrow),
                        FadeOut(right_arrow),
                        FadeOut(final_net),
                        lag_ratio=0.2,
                    ),
                    FadeOut(current_extras.ring_holder),
                    FadeOut(final_extras.ring_holder),
                    run_time=1.6,
                )
