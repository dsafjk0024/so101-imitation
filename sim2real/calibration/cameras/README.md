# Camera calibration records

One YAML file per camera, named after its role: `wrist.yaml`, `front.yaml`.
These files are the single source of truth for camera geometry — the real
capture layer (`so101.real.cameras`) and the Isaac Lab scene
(`so101.scenes.tabletop`) both read them, so the simulated and physical
cameras cannot drift apart.

Produced by `scripts/calibrate_intrinsics.py` (intrinsics),
`scripts/calibrate_handeye.py --via-board --known-board-center-env ...` (front
extrinsic from the fixed table board) and `scripts/known_board_calibration.py
--fit --write` (wrist mount). Prefer regenerating to hand-editing.

`front.yaml` and `wrist.yaml` are the calibration of our rig, kept as a worked
example; recalibrate before using them on another setup.

## Why `K_virtual` exists

Isaac Lab renders an ideal pinhole only:

- `Camera._update_intrinsic_matrices` hardcodes `f_y = f_x`, `c_x = W/2`, `c_y = H/2`
- `spawn_camera` discards aperture offsets entirely (NVIDIA ticket OM-42611)
- `PinholeCameraCfg.from_intrinsic_matrix` silently drops `c_x`/`c_y` and averages `f_x`/`f_y`

So rather than bending the renderer, real frames are remapped onto an ideal
pinhole — `K_virtual` — that Isaac reproduces exactly. Lens distortion is
removed in the same `cv2.remap` call. `K_virtual` is required to have
`fx == fy` and to be centred; the loader rejects anything else, because a
`K_virtual` Isaac cannot render would show up much later as an unexplained
gate failure.

## Fields

| field | meaning |
|---|---|
| `name` | camera role: `wrist` or `front` |
| `device` | V4L2 node, e.g. `/dev/video0` |
| `resolution` | `[width, height]` — must match the capture resolution exactly |
| `fps` | capture rate |
| `K` | measured intrinsic matrix (3x3, row-major) |
| `dist` | measured OpenCV distortion coefficients |
| `K_virtual` | rectification target; `fx == fy`, principal point at the centre |
| `alpha` | `getOptimalNewCameraMatrix` alpha used (0 = crop to all-valid pixels) |
| `extrinsic.parent` | `env` for the front camera, `gripper` for the wrist (the scene always mounts the wrist camera under `Robot/gripper`) |
| `extrinsic.convention` | always `ros` (OpenCV: +Z optical axis, +Y down) |
| `extrinsic.pos` / `quat_wxyz` | camera pose relative to `parent` |
| `exposure` | optional V4L2 overrides; the locked defaults live in `so101.real.cameras.DEFAULT_SPECS` |
| `calib_rms_px` | OpenCV RMS reprojection error from intrinsic calibration (gate: < 0.3) |
| `handeye_rmse_px` | reprojection RMSE of the fit that wrote the extrinsic |
| `hfov_loss_frac` | horizontal FOV lost to `alpha=0` cropping (> 0.10 triggers review) |
| `date` | when the calibration was taken |

## Gotchas

- **Calibrate at the resolution you run at.** USB cameras crop or bin
  differently per mode, so a `K` measured at 1280x720 and halved is quietly
  wrong. The capture layer refuses a calibration whose resolution does not
  match.
- **`convention` is always `ros`.** That is what `cv2.solvePnP` and
  `cv2.calibrateHandEye` return and what Isaac Lab's `OffsetCfg` means by
  `convention="ros"`, so the numbers transfer with no conversion layer to get
  a sign wrong in.
- Exposure is fixed (wrist 83 / gain 0, front 6187) and re-metered at the start
  pose on deployment (`deploy_policy_real.py`, `METER_TARGETS`); auto exposure
  drifts with daylight.
- The wrist mount is fitted under the follower's `wrist_roll` offset; refit it
  whenever that offset changes.

See `example.yaml` for the shape. It is not loaded by anything — the loader
looks for `wrist.yaml` / `front.yaml`.
