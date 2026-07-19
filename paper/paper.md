---
title: 'aprende-vslam: a level-based Visual SLAM course in Spanish where every claim is a verified number'
tags:
  - SLAM
  - computer vision
  - robotics
  - state estimation
  - Python
  - Spanish-language
authors:
  - name: Ariel Eliezer Vazquez Dominguez
    orcid: 0009-0001-4922-9100
    affiliation: 1
affiliations:
  - name: Independent Researcher, Mexico  # TODO(Ariel) - confirmar afiliacion
    index: 1
date: 18 July 2026
bibliography: paper.bib
---

# Summary

`aprende-vslam` is a practical Visual SLAM course in Spanish, organized as
25 self-contained levels that take a student from *"an image is a matrix of
numbers"* to a complete visual SLAM system with bundle adjustment and loop
closure, and beyond: real RGB-D and stereo datasets [@sturm2012; @burri2016],
learned features [@detone2018; @lindenberger2023], real-time engineering, a
differentiable 3D Gaussian Splatting rasterizer built from scratch
[@kerbl2023], a ROS 2 integration, and four bonus levels on estimation
theory — factor graphs, IMU preintegration [@forster2017], Kalman filtering
from first principles, and the machinery inside GTSAM [@dellaert2017;
@kaess2012]: variable elimination, elimination ordering, the Bayes tree,
and a working toy iSAM.

The course's central design decision is that **every claim ends in a
number, and every number has an automatic exam**. Each level ships a
`verificacion.py` script — 202 checks across the 25 levels — that compares
the student's implementation against measured expected values: a
hand-written grayscale conversion must match OpenCV within ±1 intensity
level; marginalization must equal the Schur complement to $4\times10^{-16}$;
the full TUM `fr2_xyz` sequence must track at 1.4 cm ATE with zero lost
frames; removing bundle adjustment must collapse accuracy from 5.8 cm to
148 cm. Passing the exam is the definition of mastering the level.

# Statement of need

Visual SLAM sits at the intersection of geometry, optimization, probability
and software engineering, and the canonical learning resources — textbooks
such as @hartley2004 and @gao2021, and reference systems such as ORB-SLAM2
[@murartal2017] — present either the theory or a finished system, leaving a
gap in between: the student can read the math and can run the system, but
rarely gets to *build the bridge* and check each plank under their own
weight. Course repositories that do bridge it typically verify progress by
inspection ("the trajectory looks right"), which does not scale to
self-study.

`aprende-vslam` addresses both gaps at once:

1. **Verification as assessment.** The 202 automated checks turn
   qualitative milestones into quantitative, reproducible ones, making the
   course usable without an instructor: the exam either passes or names the
   number that failed. Classic failures are part of the curriculum,
   reproduced on purpose and measured — the wrap-around bug in a heading
   innovation (39× worse), a one-sided axis-convention conversion (120° of
   constant false attitude), the half-pixel convention bug in a
   differentiable rasterizer (29 dB).

2. **The Spanish-language gap.** There is very little in-depth, hands-on
   SLAM material in Spanish. The course is written natively in Spanish
   (code, comments, exams and readmes), while remaining runnable and
   reviewable by non-Spanish speakers: the exams print numbers.

The intended audience is advanced undergraduates, graduate students and
self-taught engineers with basic Python and introductory linear algebra;
computer vision is explicitly *not* a prerequisite, since the course builds
it from the sensor up.

# Course design

Six rules act as the course's constitution; three define its character:

- **Total independence.** Zero imports between levels and no shared
  package: any level can be copied to another machine and run. Shared
  infrastructure is *deliberately duplicated* and trimmed to what the
  student knows at that point — immediate context beats DRY in teaching
  code.
- **Numbers with provenance.** The course is the pedagogical decomposition
  of a working parent system (`vslam-edu` on PyPI); many milestones are
  that system's real measurements, including its failures (the sequence
  that defeated feature-based RGB tracking motivates the RGB-D level; the
  13× blur robustness of learned features — and why it still was not
  enough — motivates the sensor lesson).
- **Honest adaptations.** When reality intrudes, the course documents it
  and measures it instead of hiding it: GTSAM publishes no Windows wheel
  for current Python, so the final level runs the real library in a
  container and validates that its answer matches the from-scratch
  implementation to 0.4 mm of ATE.

Levels follow a fixed template — flat, top-to-bottom-readable scripts, a
driver that prints the level's tables and figures, the exam, and exercises
each with its own target number — and close with a bonus arc (levels 21–24)
that builds the estimation trilogy explicitly: batch smoothing, filtering,
and incremental smoothing (iSAM) on identical measurements, ending in a toy
iSAM whose loop-closure cost spike is measured (12× the median step) and
whose 12.9× speedup over per-step batch grows with trajectory length.

# Quality control

Continuous integration runs the dataset-free exams (17 levels) on Ubuntu
and Windows on every push; the dataset levels are verified locally against
TUM and EuRoC sequences, and a maintainer script (`verifica_todos.py`) runs
all 25 exams end to end. The GTSAM container smoke test is exercised with
an optional exam flag (`--docker`).

# Acknowledgements

The course distills lessons from building its parent system, and openly
builds on the ecosystems it teaches: OpenCV, NumPy, Matplotlib, PyTorch,
GTSAM and ROS 2.

# References
