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
# fidelity doesn't matter.
if os.environ.get("LOW_RES", "0") == "1":
    config.pixel_width = 480
    config.pixel_height = 270
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

# Set SIMPLE_STYLE=1 to strip every node down to one flat circle and every
# edge down to one flat line -- no glow discs, no glow blob, no pulse-chain
# animation. Purely a profiling stand-in to isolate how much of the render
# cost Bloom Pulse's per-node glow discs and pulse chains (see module
# docstring) actually account for, not a style meant to ship.
SIMPLE_STYLE = os.environ.get("SIMPLE_STYLE", "0") == "1"

# Set EDGES_ONLY=1 to make node circles fully transparent -- geometry,
# layout, and edge attachment are all unaffected (edges read node
# position off the node mobject same as always), only the node's own
# paint is skipped, for a faster one-off look at edge structure/layout on
# nets with a lot of nodes without also paying to rasterize every node.
EDGES_ONLY = os.environ.get("EDGES_ONLY", "0") == "1"

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
    "import torch",
    "for x_batch, y in loader:",
    "    logits = net(x_batch)",
    "    loss = F.cross_entropy(logits, y)",
    "    loss.backward()",
    "    optimizer.step()",
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


def net_radius(n_nodes, spacing_mult=3.0):
    """The radius net_radius() nodes need at spacing_mult*NODE_RADIUS
    apart, padded by NET_RADIUS_PACKING_K for the rejection sampler's own
    (well short of 100%) packing efficiency -- confirmed reliable
    (placing every node comfortably inside sample_cloud's 4000-try
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
# vertical room instead of staying circular and leaving it empty -- kept
# a bit under frame_height/2 (not all the way to 4.0) so a stretched net
# never quite touches the top/bottom edge. Left at None (circular, same
# as cloud_radius) for the earlier stages, which aren't at the horizontal
# cap yet and don't need it.
MAX_NET_RADIUS_Y = 3.7


def _stage(n_nodes, spacing_mult, k_neighbors, seed, radius_y=None):
    return dict(
        n_nodes=n_nodes,
        cloud_radius=min(net_radius(n_nodes, spacing_mult), MAX_NET_RADIUS),
        k_neighbors=k_neighbors,
        node_radius=NODE_RADIUS,
        seed=seed,
        spacing_mult=spacing_mult,
        radius_y=min(radius_y, MAX_NET_RADIUS_Y) if radius_y is not None else None,
    )


STAGES = [
    _stage(n_nodes=4, spacing_mult=3.00, k_neighbors=3, seed=21),
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
# in the frame meant to contain it. spacing_mult packed noticeably tighter
# than any STAGES entry (down from the default 3.0) -- at 432 nodes and
# the default spacing, the sheer *radius* needed reads as sparse (lots of
# empty space between nodes) rather than dense; this keeps the same 432
# nodes but in much less area, confirmed to still place all of them
# (well inside sample_cloud's 4000-try budget) and still comfortably
# bigger than both MAX_NET_RADIUS and MAX_NET_RADIUS_Y, so it's still
# unmistakably overflowing, just a denser overflow. k_neighbors is kept
# modest despite the huge node count purely for render time -- this many
# nodes already renders slowly; a denser mesh on top of that compounds
# fast.
FINAL_STAGE_SPACING_MULT = 1.8
FINAL_STAGE_RADIUS = net_radius(432, FINAL_STAGE_SPACING_MULT)
FINAL_STAGE = dict(
    n_nodes=432,
    cloud_radius=FINAL_STAGE_RADIUS,
    k_neighbors=4,
    node_radius=NODE_RADIUS,
    seed=9,
    spacing_mult=FINAL_STAGE_SPACING_MULT,
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


def spawn_pulse_ring(node, base_radius, color, extras, max_growth_mult=1.6, duration=2.0, peak_opacity=0.425):
    """A single expanding, fading ring -- a brand-new mobject per ping
    rather than one ring reused per node, so a node pinged again while
    its last ring is still expanding gets a second, independent ring
    instead of the first one being cut short."""
    ring = Circle(radius=base_radius, stroke_color=color, stroke_width=2.4, fill_opacity=0)
    ring.move_to(node.get_center())
    ring.set_z_index(2)  # matches extras -- see build_net's layering comment
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
        mob.set_z_index(2)  # become() replaces the mobject wholesale, z_index included

    ring.add_updater(updater)
    extras.add(ring)


def add_node_flash(node, rest_color, flash_color, attack=0.05, release=0.6):
    """Ramp the node's mid ring (index 0) up to flash_color over
    `attack`, then ease it back down to rest_color over `release` -- a
    quick rise-then-fade, rather than snapping straight to flash_color
    on the first frame and only fading from there (which reads as a
    flat white pop, not a flash)."""
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


def add_pulse_chains(nodes_group, edges_group, base_radius, palette, extras, chain_stagger=0.2):
    """One node pings, then an adjacent node (following an actual edge)
    0.2s later, then another -- 3-5 hops long -- rather than every node
    independently, randomly pinging on its own. A beat after a chain's
    last hop fires, another chain starts from a random node -- longer
    for small nets (fewer than SMALL_NET_THRESHOLD nodes), which have so
    little to hop across that the default gap reads as spammy."""
    core_color, mid_color = palette[0], palette[1]
    node_list = list(nodes_group)
    SMALL_NET_THRESHOLD = 6
    next_chain_cooldown = 0.9 if len(node_list) < SMALL_NET_THRESHOLD else 0.5
    index_of = {id(n): i for i, n in enumerate(node_list)}
    adjacency = {i: [] for i in range(len(node_list))}
    for edge in edges_group:
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
    # Initial cooldown is zero -- this is attached right as a net finishes
    # growing in (or right after it slides into place), so the first
    # chain should fire the instant that happens, not after a pause.
    state = {"cooldown": 0.0, "queue": [], "flash_queue": []}

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
            unvisited = [n for n in adjacency[current] if n not in visited]
            if not unvisited:
                # Dead end -- every neighbor (including wherever this hop
                # just came from) is already visited. The chain just ends
                # here rather than bouncing back the way it came.
                break
            current = random.choice(unvisited)
            path.append(current)
            visited.add(current)

        state["queue"] = [[i * chain_stagger, idx] for i, idx in enumerate(path)]
        state["cooldown"] = next_chain_cooldown

    extras.add_updater(scheduler)


def stop_effects(extras):
    """Freeze this net's bloom-disc tracking and pulse scheduler, and
    drop any pulse ring still mid-flight, rather than merely suspending
    them -- .suspend_updating() turned out not to be enough: even
    suspended, an attached updater somehow still left a following
    net.animate.shift() silently unable to move the mobject (proven by
    an isolated test: identical shift, .suspend_updating() -> no
    movement, .clear_updaters() -> moves correctly). Call before any
    animation that repositions a net. In-flight rings are actually
    removed (not just paused) since a paused one would otherwise sit
    frozen on screen forever with no updater left to finish fading it."""
    extras.clear_updaters()
    for mob in list(extras):
        mob.clear_updaters()
        if isinstance(mob, Circle):
            extras.remove(mob)


def resume_effects(extras, nodes_group, edges_group, node_radius, palette):
    """Re-attach bloom-disc tracking and restart the pulse chains --
    call right after any animation that moves a net (e.g. the slide
    into the left slot)."""
    for disc in extras:
        disc.add_updater(lambda mob: mob.move_to(mob.tracked_node.get_center()))
    if not SIMPLE_STYLE:
        add_pulse_chains(nodes_group, edges_group, node_radius, palette, extras)


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


def sample_cloud(n, radius, node_radius, seed, min_dist_factor=0.32, spacing_mult=3.0, radius_y=None):
    """n points scattered inside a disc of the given radius (or an ellipse,
    if radius_y differs from radius -- see STAGES' vertical stretch for
    later stages), rejecting anything too close to a point already placed
    -- rough blue-noise spacing, so nodes read as an organic cluster
    rather than a grid. Sampled uniformly in a unit disc, then the disc
    itself is stretched to (radius, radius_y) rather than sampled directly
    in ellipse coordinates -- simple, and the min_dist rejection below
    still operates on genuine post-stretch Euclidean distance, so spacing
    stays correct regardless of aspect ratio. Whatever that blue-noise
    spacing works out to, it's never allowed below spacing_mult*node_radius
    -- at the default spacing_mult=3.0, two touching node circles
    (2*node_radius) plus a full extra node_radius of clear margin between
    them, so nodes at this size can never actually overlap, even for the
    sparsest/smallest-radius stages where the proportional spacing alone
    would allow it. Callers packing a fixed on-screen radius tighter to
    fit more nodes (see STAGES) pass a smaller spacing_mult instead."""
    radius_y = radius if radius_y is None else radius_y
    min_dist = max(min(radius, radius_y) * min_dist_factor, spacing_mult * node_radius)
    rng = random.Random(seed)
    points = []
    tries = 0
    while len(points) < n and tries < 4000:
        tries += 1
        x, y = rng.uniform(-1, 1), rng.uniform(-1, 1)
        if x * x + y * y > 1:
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


def nearest_neighbor_edges(points, k, node_radius, block_margin=1.2):
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
    grazing a node's outer rim still counts as blocked."""
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


def _ensure_two_vertex_connected(points, edges, node_radius, block_margin=1.2, top_k=15):
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
    often enough that this stays fast even on FINAL_STAGE's 432 nodes."""
    n = len(points)
    points_arr = np.array(points)
    threshold = node_radius * block_margin
    edges = set(edges)

    def blocked(a, b):
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
    n_nodes, cloud_radius, k_neighbors, node_radius, seed, center, palette, edge_color, spacing_mult=3.0, radius_y=None
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
        n_nodes, cloud_radius, node_radius, seed, min_dist_factor=0.0, spacing_mult=spacing_mult, radius_y=radius_y
    )
    points = [center + p for p in local_points]

    nodes_list = [make_node(p, node_radius, palette) for p in points]
    nodes_group = VGroup(*nodes_list)
    edge_indices = nearest_neighbor_edges(local_points, k_neighbors, node_radius)
    edge_indices = _ensure_two_vertex_connected(local_points, edge_indices, node_radius)
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
    nodes_group.set_z_index(3)

    # Group rather than VGroup: it needs to hold ImageMobjects (glow,
    # extras) alongside the VMobject-based edges/nodes, which VGroup
    # rejects. A member of net itself, not tracked separately, so the
    # existing FadeOut/.animate.shift() calls on the whole net move and
    # fade it right along with everything else without any extra code.
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
        for edge in edges_group:
            i, j = index_of[id(edge.node_a)], index_of[id(edge.node_b)]
            delay_frac = max(i, j) / last_index
            wait_time = delay_frac * run_time * 0.7
            edge_fade_ins.append(Succession(Wait(wait_time), FadeIn(edge, run_time=run_time - wait_time)))

        if os.environ.get("DEBUG_GLOW", "0") == "1":
            state = {"t": 0.0}
            first_edge = edges_group[0] if len(edges_group) else None

            def _dbg(mob, dt):
                state["t"] += dt
                print(
                    f"t={state['t']:.3f} glow_op={glow.get_opacity():.3f} "
                    f"extras_op={extras[0].get_opacity():.3f} "
                    f"node_w={node_list[-1].width:.4f} "
                    f"edge_op={(first_edge.get_stroke_opacity() if first_edge is not None else -1):.3f}"
                )

            glow.add_updater(_dbg)

        self.play(
            FadeIn(glow),
            FadeIn(extras),
            AnimationGroup(*edge_fade_ins),
            LaggedStart(*[GrowFromCenter(n) for n in node_list], lag_ratio=0.04),
            run_time=run_time,
        )
        if os.environ.get("DEBUG_GLOW", "0") == "1":
            glow.clear_updaters()
        # Not started any earlier than this -- a ping firing on a net
        # that's still forming would read as broken, not alive.
        if not SIMPLE_STYLE:
            add_pulse_chains(nodes_group, edges_group, node_radius, palette, extras)

    def construct(self):
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
        self.construct_intro()
        if INTRO_ONLY:
            return
        self.hold(1.0)
        self.construct_main(backdrop)

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
            LaggedStart(
                *[Write(row) for row in code], lag_ratio=0.3, run_time=(0.5 + 0.16 * len(CODE_STAGE_1)) * m
            ),
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
        # Both nets' pulse schedulers are stopped first -- left running,
        # either would keep mutating its extras mid-FadeOut and crash
        # manim's interpolation ("zip() argument 2 is shorter than
        # argument 1", confirmed against an actual full render).
        stop_effects(extras0)
        stop_effects(extras1)
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

        # Net 0 spawns on the left, green for its whole lifetime -- grown
        # in at the same unhurried pace as the first loop iteration below,
        # already positioned for lap 1's own code (and shifted to keep lap
        # 1's whole composition centered, since net 0 and lap 1's own
        # right-hand net are almost never the same size) so its arrow is
        # exactly ARROW_LENGTH from the very first frame.
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

        # Left net -> left arrow -> code typing in behind a brace -> right
        # arrow -> a bigger net (its own fixed color) grows on the right
        # -> the left net, code, and arrows vanish while the right net
        # slides into the left slot, becoming "current" for the next lap.
        # Every beat in the lap is scaled by that lap's own multiplier, so
        # early laps linger and later laps snap by increasingly fast.
        for i, (stage, code_lines, m) in enumerate(zip(STAGES[1:], CODE_STAGES, SPEED_MULTIPLIERS), start=1):
            stage_radius = stage["cloud_radius"]
            shift = lap_shift(lap_radii[i - 1], lap_radii[i], code_lines)
            code = make_code_block(code_lines, np.array([MID_X + shift, 0, 0]))
            brace = make_bracket(code, buff=0.25, color=ARROW_COLOR)
            left_arrow, right_arrow = flow_arrows(
                left_facing_edge(brace.get_left()[0]), right_facing_edge(code.get_right()[0]), brace, code
            )

            self.play(grow_arrow(left_arrow), run_time=0.5 * m)
            self.play(
                LaggedStart(
                    *[Write(row) for row in code], lag_ratio=0.3, run_time=(0.5 + 0.16 * len(code_lines)) * m
                ),
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

            stop_effects(next_extras)
            # current_net's own pulse scheduler is still live too -- left
            # running, it keeps adding/removing pulse-ring mobjects into
            # current_extras for however many frames this FadeOut takes,
            # changing current_net's family size mid-animation and
            # crashing manim's interpolation (confirmed: "zip() argument
            # 2 is shorter than argument 1"). No resume_effects needed
            # afterward -- current_net is being discarded, not reused.
            stop_effects(current_extras)
            # The lap right after this one is the final, deliberately-
            # overflowing net -- centering the composition around it
            # would try to drag everything sideways to "balance" a net
            # many times bigger than its neighbor, instead of leaving
            # the rest of that lap alone and letting only the final net
            # spill off-screen. So no lap_shift lookahead for that one;
            # plain shift=0 instead, matching the final lap's own below.
            is_final_next = (i + 1 == len(STAGES))
            next_shift = 0 if is_final_next else lap_shift(lap_radii[i], lap_radii[i + 1], lap_code_lines[i])
            _, next_brace_left = code_edges_for(lap_code_lines[i])
            new_left_center = left_net_center(stage_radius, next_brace_left + next_shift)
            slide_shift = new_left_center - right_net_center(stage_radius, code.get_right()[0])
            self.play(
                FadeOut(current_net),
                FadeOut(code),
                FadeOut(brace),
                FadeOut(left_arrow),
                FadeOut(right_arrow),
                next_net.animate.shift(np.array([slide_shift, 0, 0])),
                run_time=max(1.0 * m, 0.45),
            )
            resume_effects(next_extras, next_nodes, next_edges, stage["node_radius"], NET_PALETTES[i])
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
        left_arrow, right_arrow = flow_arrows(
            left_facing_edge(brace.get_left()[0]), right_facing_edge(code.get_right()[0]), brace, code
        )

        self.play(grow_arrow(left_arrow), run_time=0.5 * FINAL_CODE_MULT)
        self.play(
            LaggedStart(
                *[Write(row) for row in code],
                lag_ratio=0.3,
                run_time=(0.5 + 0.16 * len(CODE_STAGE_FINAL)) * FINAL_CODE_MULT,
            ),
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
            # current_net's pulse scheduler is stopped first -- left
            # running, it would keep mutating current_extras mid-FadeOut
            # and crash manim's interpolation (see the other stop_effects
            # call in the main loop for the confirmed error).
            stop_effects(current_extras)
            ico_group = VGroup(ico_glow, ico_vertices, ico_edges)
            self.play(
                FadeOut(current_net),
                FadeOut(code),
                FadeOut(brace),
                FadeOut(left_arrow),
                FadeOut(right_arrow),
                FadeOut(backdrop),
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
            # red, like the icosahedron it stands in for -- at the same
            # footprint radius already used to space the arrow above, so
            # the arrow's tip meets its edge.
            final_net, final_nodes, final_edges, final_glow, final_extras = build_net(
                center=np.array([ico_center_x, 0, 0]), palette=RED_PALETTE, edge_color=RED_EDGE, **FINAL_STAGE
            )
            self.grow_in(
                final_nodes, final_edges, final_glow, final_extras, FINAL_STAGE["node_radius"], RED_PALETTE,
                run_time=max(1.3 * FINAL_CODE_MULT, 0.5),
            )
            self.hold(0.6)

            # Each piece fades out on its own rather than as one
            # synchronized block, and the backdrop stays put throughout --
            # so this reads as the scene's pieces settling away, not the
            # whole picture (background glow included) dimming to black.
            # Both nets' pulse schedulers are stopped first -- left
            # running, either would keep mutating its extras mid-FadeOut
            # and crash manim's interpolation (see the other stop_effects
            # calls above for the confirmed error).
            stop_effects(current_extras)
            stop_effects(final_extras)
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
                run_time=1.6,
            )
