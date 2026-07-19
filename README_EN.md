# aprende-vslam — Visual SLAM from scratch, in Spanish, verified by numbers

*One-page English summary. The course itself is written in Spanish —
deliberately: there is very little serious, in-depth SLAM material in
Spanish, and this repo exists to fill that gap. Issues and PRs in English
are welcome ([contributing guide](CONTRIBUTING.md)).*

**What it is**: a practical Visual SLAM course in 25 self-contained levels,
from *"an image is a matrix of numbers"* to a complete visual SLAM system
with bundle adjustment and loop closure — then across to real data (TUM
RGB-D, EuRoC), learned features, real-time engineering, a differentiable
3DGS rasterizer, ROS 2, and four bonus theory levels (factor graphs, IMU
preintegration, Kalman filtering from scratch, and the machinery inside
GTSAM: variable elimination, the Bayes tree, and a toy iSAM).

**The pedagogical bet — every claim is a measured number**: each level ends
with an automatic exam (`verificacion.py`) that compares the student's
result against expected numbers (**202 checks across 25 exams**). Examples:
hand-written grayscale == OpenCV within ±1 level; marginalization ==
Schur complement to 4e-16; the full TUM fr2_xyz sequence at **1.4 cm ATE**
with zero lost frames; removing bundle adjustment collapses the system from
5.8 cm to 148 cm — measured, not asserted. Failures are part of the
curriculum: classic bugs are reproduced *on purpose* and quantified (the
±π innovation bug: 39× worse; the one-sided axis-convention bug: 120° of
false attitude; the 3DGS half-pixel bug: 29 dB).

**Where the numbers come from**: the course is the pedagogical
decomposition of a real parent system
([Visual-slam](https://github.com/ariel9874/Visual-slam), `vslam-edu` on
PyPI) — many milestones are that system's actual measurements, including
its failures and the lessons they cost.

**Design rules** ("the constitution"): every level is fully independent
(zero cross-level imports — duplication is deliberate, immediate context
beats DRY); the math lives next to the line of code that uses it; flat
scripts readable top-to-bottom; Windows-proof throughout.

## Quick start

```bash
cd nivel_00_entorno_y_primeros_pixeles
pip install -r requirements.txt   # numpy + opencv + matplotlib
python descarga_datos.py          # only for dataset levels
python 00_hola_pixeles.py         # the lesson: read it top to bottom
python verificacion.py            # the exam: pass it, own the level
```

Prerequisites: basic Python and introductory linear algebra — no computer
vision. Hardware: CPU only through level 16; GPU only for levels 17 and 19;
Docker only for 19, 20 and 24 (optional in 24).

## The 25 levels at a glance

| Block | Levels | Verified milestones (selection) |
|---|---|---|
| A — The camera | 00–04 | shot noise ∝ √signal (slope 0.49); reprojection 0.24 px |
| B — Two views | 05–08 | own Harris == OpenCV (top-50: 100%); VO at **16 cm ATE** |
| C — Odometry → SLAM | 09–15 | PnP+map: 18.6→8.7 cm; full SLAM **5.8 cm** (no BA: 148 cm); TUM fr2_xyz **1.4 cm**; metric RGB-D **2.3 cm** |
| D — Electives | 16–20 | EuRoC stereo drone **9.0 cm**; learned features cross blur **13×** better; real-time: BA 4.4× at machine precision; own differentiable 3DGS rasterizer (58 dB); ROS 2 / REP-105 verified without ROS |
| Bonus — Estimation | 21–24 | four backends, same measurements (8.2 < 9.4 < 25.9 < 29 < 59 cm); IMU preintegration rescues a 5 s visual blackout **12.9×**; KF == linear factor graph to 1e-11; elimination == Schur to 4e-16, toy iSAM **12.9× speedup** with the loop-closure spike measured, GTSAM (Docker) matches to **0.4 mm** |

## For maintainers and reviewers

```bash
python verifica_todos.py          # run every level's exam (accepts --root
                                  # flags to reuse downloaded datasets)
```

The dataset-free exams run in seconds-to-minutes on a clean machine and are
exercised by CI on Ubuntu and Windows (15 levels on Linux, 17 on Windows:
two exams are calibrated against OpenCV's Windows backend — same version,
different trajectory on Linux — and level 19 needs PyTorch, so it runs
locally). Levels 00/05/06 and 14/15/17/18 additionally verify against
TUM/EuRoC sequences.

## License and citation

MIT. To cite, see [CITATION.cff](CITATION.cff).
