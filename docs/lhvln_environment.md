# LH-VLN Local Environment

This note records the local environment used for the time-aware VLN branch.

## Paths

- Code: `/file_system/vepfs/algorithm/intern03/mhw/LH-VLN`
- Data: `/file_system/nas/algorithm/Intern03/data`
- Conda env: `/file_system/vepfs/algorithm/intern03/.conda/envs/lhvln`
- HM3D link in repo: `data/hm3d -> /file_system/nas/algorithm/Intern03/data/versioned_data/hm3d-0.2/hm3d`

## Activate

```bash
source /file_system/vepfs/algorithm/dujun.nie/miniconda3/etc/profile.d/conda.sh
conda activate /file_system/vepfs/algorithm/intern03/.conda/envs/lhvln
```

You can also run the interpreter directly:

```bash
/file_system/vepfs/algorithm/intern03/.conda/envs/lhvln/bin/python
```

## Verified Core Packages

- Python 3.9
- Habitat-Sim 0.3.1
- Torch 2.4.1+cu121
- NumPy 1.23.5
- SciPy 1.10.1
- Numba 0.57.1

`deepspeed==0.6.5` from the upstream requirements is installed but is not compatible
with Torch 2.x because it imports `torch._six`. It is not needed for the
time-aware greedy/oracle baseline.

## Smoke Test

The server currently cannot create a rendering GL context for RGB/depth sensors,
so the time-aware greedy baseline defaults to no-render mode. This still loads
HM3D, navmesh, and semantic objects, and is enough for step-budget search
efficiency baselines.

```bash
cd /file_system/vepfs/algorithm/intern03/mhw/LH-VLN
HABITAT_SIM_LOG=quiet MAGNUM_LOG=quiet EGL_PLATFORM=surfaceless \
  /file_system/vepfs/algorithm/intern03/.conda/envs/lhvln/bin/python \
  tools/run_time_aware_greedy.py \
  --split val \
  --limit 1 \
  --budget-ratio 0.5 \
  --output output/time_aware/greedy_val_smoke.json
```

To try the original rendered simulator path, pass `--render`; this still depends
on a working EGL/OpenGL setup on the machine.
