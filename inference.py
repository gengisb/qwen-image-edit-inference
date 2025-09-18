import os, io, json, base64
from PIL import Image
import torch
from diffusers import QwenImageEditPipeline

MODEL_ID = os.environ.get("MODEL_ID", "Qwen/Qwen-Image-Edit")
OFFLOAD_DIR = "/opt/ml/model/offload"  # lives on the endpoint EBS volume

def _ensure_dir(p): os.makedirs(p, exist_ok=True)

def _b64_to_pil(b64: str) -> Image.Image:
    return Image.open(io.BytesIO(base64.b64decode(b64))).convert("RGB")

def _pil_to_b64(img: Image.Image) -> str:
    buf = io.BytesIO(); img.save(buf, format="PNG"); return base64.b64encode(buf.getvalue()).decode()

def model_fn(model_dir):
    _ensure_dir(OFFLOAD_DIR)

    # Expose all GPUs to ONE worker; HF will shard modules across them.
    # IMPORTANT at deploy time: SAGEMAKER_MODEL_SERVER_WORKERS=1
    visible = os.environ.get("CUDA_VISIBLE_DEVICES", "0,1,2,3")
    gpu_ids = [int(x) for x in visible.split(",") if x != ""]

    # Leave headroom on each 24GB card
    max_memory = {i: "22GiB" for i in range(len(gpu_ids))}
    max_memory["cpu"] = "64GiB"

    pipe = QwenImageEditPipeline.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.float16,      # run in fp16
        variant="fp16",                 # ignored if repo lacks fp16 weights
        device_map="auto",              # shard across visible GPUs
        max_memory=max_memory,
        offload_folder=OFFLOAD_DIR,     # CPU/NVMe spill if needed
    )

    pipe.set_progress_bar_config(disable=True)
    pipe.enable_attention_slicing()
    pipe.enable_vae_tiling()
    # If your image has xFormers baked in, uncomment:
    # pipe.enable_xformers_memory_efficient_attention()

    return pipe

def input_fn(request_body, content_type):
    if content_type != "application/json":
        raise ValueError(f"Unsupported content type: {content_type}")
    payload = json.loads(request_body)

    # Accept either base64 image or (rarely) a direct URL-less omission
    if "image_b64" not in payload:
        raise ValueError("Missing 'image_b64' field")
    return payload

def predict_fn(data, model: QwenImageEditPipeline):
    # Map Qwen sample args
    image_b64   = data["image_b64"]
    prompt      = data.get("prompt", "")
    negative    = data.get("negative_prompt", " ")
    strength    = float(data.get("strength", 0.6))              # optional convenience
    guidance    = float(data.get("guidance", 5.0))              # convenience for classic CFG
    true_cfg    = float(data.get("true_cfg_scale", 4.0))        # Qwen-specific
    steps       = int(data.get("num_inference_steps", data.get("steps", 20)))
    seed        = data.get("seed", None)

    # Build generator
    device = "cuda" if torch.cuda.is_available() else "cpu"
    generator = torch.Generator(device=device).manual_seed(int(seed)) if seed is not None else None

    image = _b64_to_pil(image_b64)

    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.float16):
        out = model(
            image=image,
            prompt=prompt,
            negative_prompt=negative,
            num_inference_steps=steps,
            true_cfg_scale=true_cfg,
            # If Qwen pipeline also accepts classic guidance_scale, pass it:
            guidance_scale=guidance,
            generator=generator,
            strength=strength,   # only used if the pipeline honors it (image2image style)
        )

    edited = out.images[0]
    return {"image_b64": _pil_to_b64(edited)}

def output_fn(prediction, accept):
    return json.dumps(prediction), "application/json"
