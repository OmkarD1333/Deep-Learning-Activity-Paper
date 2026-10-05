# U-Net Biomedical Segmentation — Full Stack

This version connects the supplied notebook workflow to a real web UI.

## Architecture
Browser HTML -> FastAPI -> PyTorch/NumPy/OpenCV/SciPy/Albumentations -> ISBI dataset/checkpoint

## Run on Windows
1. Install Python 3.10/3.11/3.12 recommended for PyTorch compatibility.
2. Open a terminal in `unet_fullstack`.
3. Create/activate a virtual environment.
4. Install dependencies:
   `pip install -r backend/requirements.txt`
5. Start:
   `python backend/main.py`
6. Open:
   `http://127.0.0.1:8000`

Git must be installed because the notebook downloads the ISBI mirror using `git clone`.

## Workflow
- Setup Dataset downloads the same public ISBI 2012 EM dataset mirror used by the notebook.
- Start Training runs the weighted-loss + augmentation + U-Net + SGD training loop in Python.
- The UI polls `/api/status` and displays real epoch metrics.
- A best checkpoint is saved as `backend/unet_best.pt`.
- Visualization loads validation data and the actual trained checkpoint.
- Calculate validation metrics runs IoU and pixel accuracy over the validation loader.
- Upload image runs real PyTorch inference using the checkpoint.

## Important
The default configuration follows the notebook's executable settings:
UNetPadded, 256x256, 2 classes, base channels 64, batch size 2, SGD lr 1e-3, momentum 0.99, 25 epochs, w0=10, sigma=5.

The notebook also defines `UNetOriginal`; it is included in `backend/pipeline.py`.
