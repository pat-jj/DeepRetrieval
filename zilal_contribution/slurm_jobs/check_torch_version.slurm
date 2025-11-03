python - <<'PY'
import torch
print("PyTorch version:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("Device count:", torch.cuda.device_count())
    print("Device name:", torch.cuda.get_device_name(0))
PY


# To check vLLM is installed or not
python -c "import vllm; print('vLLM version:', vllm.__version__)"

#Check where vllm coming from
python - <<'PY'
import vllm, inspect, sys
print("vLLM version:", getattr(vllm, "__version__", "unknown"))
print("Imported from:", vllm.__file__)
print("Python:", sys.version)
print("OK")
PY
pip show -v vllm | sed -n '1,120p'
#Check vllm is working
python - <<'PY'
from vllm import LLM, SamplingParams
llm = LLM(model="TinyLlama/TinyLlama-1.1B-Chat-v1.0", dtype="bfloat16", gpu_memory_utilization=0.3)
out = llm.generate(["Hello"])
print(out[0].outputs[0].text[:60])
PY
