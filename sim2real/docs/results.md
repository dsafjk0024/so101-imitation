# Results

All sim numbers are StackCube success in Isaac Lab. Success means the small cube
is aligned on the large cube and the gripper has retreated by more than 2 cm;
`stacked` counts the stack without the retreat. Each episode is capped at 300
steps (10 s) unless noted otherwise. Real numbers are labelled by the operator.

## Teacher

ResiP sweep from the same DBC base, 2000 iterations each. The table shows sim
success at iteration 2000.

| run | change | success |
|---|---|---|
| step | lr 1e-3, 4 minibatches | 0.796 |
| step_critic | + critic min lr 5e-4, critic 512×2 | 0.780 |
| learn_std | learned residual std | 0.854 |
| **gamma99** | γ = 0.99 (others 0.999) | **0.916** |

gamma99 is the teacher for all synthetic data below.

## First transfer attempt, and what it taught us

`cotrain_v1` was trained on 600 synthetic episodes (one teacher target per tick)
plus 41 real demos. Those demos were recorded under an uncalibrated joint mapping.

- **Sim**: DP 55 % nominal / 70 % DR (40 episodes each).
- **Real**: 0 of 14 attempts.

The failures traced to four problems:

| problem | evidence | fix |
|---|---|---|
| 0.06 rad/tick rate limit | clipped 45 % of teacher ticks; DP closed the gripper before reaching the cube | 0.25 rad/tick arm, 0.5 gripper |
| auto exposure | morning frames at mean 159/130 vs ~115/65 in training | fixed exposure, re-metered at the start pose |
| fingertip ~2 cm off at equal joint readings | open-loop replay of a teacher plan (`fixed_layout_check.py`) | physical scale + 9-point touch + roll alignment (18.2 → 5.5 mm) |
| servo tracking | elbow rest error 3.7° at P = 16 vs 0.9° in sim | P gain 32 (2.1°) |

The fixes were followed by new data:
- 40 real demos recorded under the new mapping.
- Synthetic data regenerated with the fitted wrist mount, the camera mount mesh
  hidden, and the teacher slowed down (`--motion-substeps 2`). The teacher's
  acceptance rate under the stricter success check was 77 %.

## Co-trained student (`cotrain_v3`)

DP, sim 0.93 / real 0.07 per batch, 100k steps.

| evaluation | result |
|---|---|
| sim nominal (20 episodes, 640-step cap) | 100 % |
| sim rendering DR (40 episodes) | 87.5 % |
| real, DP | **11 / 23** |
| real, ACT (same data, 200k steps) | 1 / 6 |

## Data-count ablation

The ablation varies how much synthetic and real data the DP gets:
- Episode subsets are nested: sim 200 ⊂ 400 ⊂ 600 and real 20 ⊂ 30 ⊂ 40.
- The per-batch sim/real mix is fixed at 0.929 / 0.071.
- Each condition trains for 100k steps from scratch.

Sim evaluation uses 40 episodes per condition, all with one fixed seed. Real
evaluation uses 10 rated attempts per condition, run on 2026-10-06 with the same
10 cube layouts in the same order.

| condition | sim nominal | sim DR | real | real 95 % CI |
|---|---|---|---|---|
| sim 200 / real 40 | 52.5 % | 30.0 % | 3 / 10 | 11–60 % |
| sim 400 / real 40 | 92.5 % | 75.0 % | 6 / 10 | 31–83 % |
| sim 600 / real 40 | 95.0 % | 97.5 % | 6 / 10 | 31–83 % |
| sim 600 / real 20 | 90.0 % | 85.0 % | 7 / 10 | 40–89 % |
| sim 600 / real 30 | 97.5 % | 92.5 % | 5 / 10 | 24–76 % |

Reading the table:
- **200 sim episodes are too few**, both in sim and on the real arm.
- **Beyond that, the real differences are within noise.** The largest gap, 3/10
  vs 7/10, has Fisher p = 0.18. Sim ranking does not predict real ranking: the
  best checkpoint in sim came fourth on the real arm.
- **Failures are mostly grasps.** Of 23 real failures, 17 never closed on the
  cube; the jaw went to its empty-closed limit instead of stopping at the cube
  width. The remaining 5 picked the cube but did not leave it on the large cube.
  Of 27 successes, 26 grasped on the first try.
- **Layout matters as much as checkpoint.** Two of the ten layouts failed under
  four of five checkpoints; one succeeded under all five.
- **Caveats**:
  - Conditions ran in a fixed order over 23 minutes at dusk, and front-camera
    brightness fell from 144 to 107. Lighting is confounded with condition.
  - Every action-chunk re-plan overran the 33 ms tick, so the effective control
    rate was about 27 Hz.
