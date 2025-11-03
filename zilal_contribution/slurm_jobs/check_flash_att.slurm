# clean any mismatched builds
pip uninstall -y flash-attn flash_attn
pip cache purge

# Install a compatible 2.x release (example pin):
pip install --no-build-isolation --no-cache-dir "flash-attn==2.6.3"

python - <<'PY'
import torch, os, importlib.util as u
print("Torch:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())
print("CUDA runtime version:", torch.version.cuda)
print("Has flash_attn?", u.find_spec("flash_attn") is not None)
PY