# Recursive Self-Improvement Animation

A Manim animation: a glowing neural net grows lap over lap — net → code
→ bigger net — accelerating each time, closing on a huge net that
overflows its own frame. Built for a documentary segment (16:9, no
on-screen narration).

The whole thing lives in one file, `recursive_self_improvement.py`.

## 1. Install prerequisites

- **Python 3.10 or newer** (this was built and tested on 3.14).
- **ffmpeg**, on your system `PATH`. Manim shells out to it to encode
  the final video.
  - Windows: `winget install ffmpeg` (or download from
    [ffmpeg.org](https://ffmpeg.org/download.html) and add it to `PATH`)
  - macOS: `brew install ffmpeg`
  - Linux: `sudo apt install ffmpeg` (or your distro's equivalent)
- A **monospace font** for the on-screen code — the script asks for
  `Consolas` specifically. It's built into Windows; on macOS/Linux,
  install it or swap it for another monospace font you have (search
  `font="Consolas"` in `recursive_self_improvement.py` — every
  occurrence should be changed together).

You do **not** need a LaTeX install — this project only renders plain
text, never `MathTex`/formulas.

## 2. Set up the project

```bash
# from inside the project folder
python -m venv .venv

# activate it:
.venv\Scripts\activate        # Windows
source .venv/bin/activate     # macOS/Linux

pip install -r requirements.txt
```

## 3. Quick test render (do this first)

Before committing to the full render (which takes a long time — see
below), confirm everything's installed correctly with a fast, low-res
preview of just the opening few seconds:

```bash
LOW_RES=1 FAST_PREVIEW=1 DEMO_SECONDS=10 manim -pql recursive_self_improvement.py RecursiveSelfImprovement
```

(On Windows PowerShell, set each env var on its own line first:
`$env:LOW_RES=1; $env:FAST_PREVIEW=1; $env:DEMO_SECONDS=10` — then run
the `manim` command without the leading `VAR=1` prefixes.)

This renders in well under a minute and should pop open a small, rough
preview. If it runs and plays, your setup is good and you're ready for
the real render below. If it errors, it's almost always one of: ffmpeg
not on `PATH`, or the `Consolas` font not found. If the shell says it
can't find the `manim` command at all, use `python -m manim ...`
instead (same arguments) — that always works as long as the `pip
install` above succeeded.

## 4. The full render (4K, 60fps)

```bash
UHD_RES=1 manim -qh --disable_caching -o RecursiveSelfImprovement_Final recursive_self_improvement.py RecursiveSelfImprovement
```

On Windows PowerShell:

```powershell
$env:UHD_RES=1
manim -qh --disable_caching -o RecursiveSelfImprovement_Final recursive_self_improvement.py RecursiveSelfImprovement
```

**This is a long render — plan for it to run for many hours (quite
possibly the better part of a day or more, depending on the machine).**
The scene is a couple of minutes long at full quality with hundreds of
glowing, individually-animated nodes on screen at once in its busiest
moments, at 3840×2160 and 60 frames per second. Some practical notes:

- Run it somewhere it won't get interrupted (leave the machine on,
  don't sleep/suspend it, don't close the terminal).
- Make sure you've got tens of GB of free disk space — manim writes
  every animation as its own file before combining them all at the end.
- Don't pass any of the preview-mode env vars (`LOW_RES`, `FAST_PREVIEW`,
  `DEMO_SECONDS`, etc.) for this run — those exist for fast iteration
  and will shorten holds or shrink the output if left set.

## 5. Where the output ends up

```
media/videos/recursive_self_improvement/2160p60/RecursiveSelfImprovement_Final.mp4
```

`media/` is git-ignored (it's regenerated, not committed), so move or
copy the finished file out of there once it's done.

## Advanced: other render options

The script has a number of environment-variable toggles for fast
iteration on a specific part of the video — none of these are needed
for the final render above, but if you want to inspect or re-render
just one piece:

- `LOW_RES=1` / `MID_RES=1` / `HD_RES=1` / `UHD_RES=1` — resolution,
  from 480×270 up to full 4K (default is 1920×1080).
- `FAST_PREVIEW=1` — clamps every idle hold down to a fraction of a
  second, so you can see every beat without waiting through it.
- `DEMO_SECONDS=N` — stops the render after N seconds of the scene's
  own timeline, instead of rendering the whole thing.
- `INTRO_ONLY=1` / `MAIN_ONLY=1` — render just the short opening lap, or
  just the main growth chain, skipping the other.
- `SIMPLE_STYLE=1` — flat circles and lines only, no glow/rings, for a
  much cheaper preview of layout and timing.

Combine any of these with manim's own quality flag (`-ql`, `-qm`,
`-qh`) the same way as the commands above.
