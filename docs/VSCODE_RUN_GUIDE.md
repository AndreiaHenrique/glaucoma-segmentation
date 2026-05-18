# How to run the code in Visual Studio Code (Windows)

This guide shows you how to (a) set up the project in VS Code, (b) run a fast
**smoke test** (2–3 min) to confirm everything is wired up correctly, and
(c) run the real experiment.

---

## 0. Prerequisites

1. **Python 3.10 or 3.11** — <https://www.python.org/downloads/windows/>.
   During install, tick **"Add python.exe to PATH"**.
2. **Visual Studio Code** — <https://code.visualstudio.com/>.
3. (Optional but recommended) **NVIDIA GPU with current drivers** — training
   on CPU will work but will be ~50× slower.

Check that the Python install is on the PATH:

```powershell
python --version
# should print: Python 3.10.x or 3.11.x
```

## 1. Open the project in VS Code

1. Unzip `glaucoma-segmentation.zip` to, say, `C:\Users\<you>\glaucoma-segmentation`.
2. Open VS Code → **File → Open Folder…** → choose that folder.
3. When VS Code asks *"Do you trust the authors of the files in this folder?"*
   click **Yes**.
4. Install the **Python** extension (left sidebar → Extensions → search "Python"
   by Microsoft → *Install*).

## 2. Create the virtual environment

Open the integrated terminal: **Terminal → New Terminal** (defaults to PowerShell).

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
```

If PowerShell complains about *"running scripts is disabled on this system"*,
run once:

```powershell
Set-ExecutionPolicy -Scope CurrentUser -ExecutionPolicy RemoteSigned
```

…and then re-run `.venv\Scripts\Activate.ps1`. After activation your prompt
should start with `(.venv)`.

Tell VS Code about the interpreter:
**Ctrl+Shift+P** → *"Python: Select Interpreter"* → pick the one inside
`.venv\Scripts\python.exe`.

## 3. Install dependencies

```powershell
python -m pip install --upgrade pip
pip install -r requirements.txt
```

> **GPU users:** the default `pip install torch` installs the CPU-only build.
> For CUDA 12.x, use the official PyTorch selector
> <https://pytorch.org/get-started/locally/> (or):
>
> ```powershell
> pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
> ```

Verify the install worked and your GPU is visible (optional):

```powershell
python -c "import torch; print('CUDA:', torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else '')"
```

## 4. Point the scripts to your data

Easiest way: copy your dataset into `./data/Segmentacao/{train,test}/{image,mask}`
inside the project folder. The scripts use those defaults out of the box, so
you do not need to set anything else.

Otherwise, set environment variables for the session:

```powershell
$env:GLAUCOMA_TRAIN_IMG  = "C:\path\to\train\image"
$env:GLAUCOMA_TRAIN_MASK = "C:\path\to\train\mask"
$env:GLAUCOMA_TEST_IMG   = "C:\path\to\test\image"
$env:GLAUCOMA_TEST_MASK  = "C:\path\to\test\mask"

# For REFUGE later:
$env:REFUGE_IMG_DIR  = "C:\path\to\REFUGE\test\images"
$env:REFUGE_MASK_DIR = "C:\path\to\REFUGE\test\mask"
```

These variables are read inside each script (top of `if __name__ == "__main__"`).

## 5. Quick smoke test (≈ 2–3 minutes)

Before doing the real 6-hour run, do a fast sanity check that the whole
pipeline (data → Optuna → KFold → ensemble → metrics → figures) runs end to
end. Override the paper defaults via env vars:

```powershell
$env:GLAUCOMA_EPOCHS      = "1"
$env:GLAUCOMA_FOLDS       = "2"
$env:GLAUCOMA_TRIALS      = "2"
$env:GLAUCOMA_TUNE_EPOCHS = "1"
$env:GLAUCOMA_PATIENCE    = "1"

cd src
python upernet_optuna.py
```

(The `cd src` is important — the architecture scripts import `pipeline` as a
sibling module.)

If it finishes without errors and prints a metrics table at the end, the
pipeline is correctly wired up. Repeat for the other two architectures:

```powershell
python manet_optuna.py
python dpt_optuna.py
```

> Tip: in VS Code you can also hit **F5** with one of these scripts open to
> launch the debugger. Set `"cwd": "${workspaceFolder}/src"` in
> `.vscode/launch.json` (VS Code can generate it for you).

## 6. Real training (paper-faithful)

Reset the env vars (or open a new PowerShell window) so they default back to
the paper values:

```powershell
Remove-Item Env:GLAUCOMA_EPOCHS, Env:GLAUCOMA_FOLDS, Env:GLAUCOMA_TRIALS, Env:GLAUCOMA_TUNE_EPOCHS, Env:GLAUCOMA_PATIENCE -ErrorAction SilentlyContinue

cd src
python upernet_optuna.py   # ≈ 6 h on an RTX 3060
python manet_optuna.py
python dpt_optuna.py

# After UPerNet finishes (the best model), run external validation:
python test_refuge.py
```

Each script saves its 5-fold ensemble to `../checkpoints/<arch>_models/`
and prints all metrics from the paper (IoU, Dice, Hausdorff, Boundary IoU,
CDR MAE) at the end.

## 7. Annotatio code
cd src
$env:NEW_IMG_DIR  = "C:\path\for\images"
$env:PRED_OUT_DIR = "C:\path\output"
python predict_masks.py

## 8. Troubleshooting

| Symptom | What to do |
|---|---|
| `ModuleNotFoundError: No module named 'pipeline'` | Make sure you run from inside `src/` (`cd src` first) or add the project root to `PYTHONPATH`. |
| `CUDA out of memory` | Lower `batch_size` (edit the Optuna search space in `pipeline.py`) or use a smaller image size. |
| Training is very slow (≥ 1 min / epoch on a small dataset) | You probably have the CPU build of torch — re-install the CUDA build (see §3). |
| `No valid pairs in ...` | The script could not find image / mask pairs with the same stem. Check folder names and file extensions. |
| Numbers do not exactly match the paper | Expected — the original Colab run used slightly different versions of PyTorch / cuDNN. Relative ranking is preserved. |

The **only** code that differs between `upernet_optuna.py`, `manet_optuna.py`
and `dpt_optuna.py` is the `build_model()` function — that is your proof of a
fair comparison.
