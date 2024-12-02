import os
import math
import time
import numpy as np
import random
import threading
from PIL import Image, ImageOps
from moviepy.editor import VideoFileClip
from datetime import datetime, timedelta
from huggingface_hub import hf_hub_download, snapshot_download

import insightface
from insightface.app import FaceAnalysis
from facexlib.parsing import init_parsing_model
from facexlib.utils.face_restoration_helper import FaceRestoreHelper

import torch
from diffusers import CogVideoXDPMScheduler
from diffusers.utils import load_image
from diffusers.image_processor import VaeImageProcessor
from diffusers.training_utils import free_memory

from util.utils import *
from util.rife_model import load_rife_model, rife_inference_with_latents
from models.utils import process_face_embeddings
from models.transformer_consisid import ConsisIDTransformer3DModel
from models.pipeline_consisid import ConsisIDPipeline
from models.eva_clip import create_model_and_transforms
from models.eva_clip.constants import OPENAI_DATASET_MEAN, OPENAI_DATASET_STD
from models.eva_clip.utils_qformer import resize_numpy_image_long

import argparse

device = "cuda" if torch.cuda.is_available() else "cpu"

def main():
    parser = argparse.ArgumentParser(description="ConsisID Command Line Interface")
    parser.add_argument("image_path", type=str, help="Path to the input image")
    parser.add_argument("prompt", type=str, help="Prompt text for the generation")
    parser.add_argument("--num_inference_steps", type=int, default=50, help="Number of inference steps")
    parser.add_argument("--guidance_scale", type=float, default=7.0, help="Guidance scale")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for generation")
    parser.add_argument("--output_dir", type=str, default="./output", help="Directory to save the output video")
    args = parser.parse_args()

    # Download models
    hf_hub_download(repo_id="ai-forever/Real-ESRGAN", filename="RealESRGAN_x4.pth", local_dir="model_real_esran")
    snapshot_download(repo_id="AlexWortega/RIFE", local_dir="model_rife")
    snapshot_download(repo_id="BestWishYsh/ConsisID-preview", local_dir="BestWishYsh/ConsisID-preview")

    model_path = "BestWishYsh/ConsisID-preview"
    lora_path = None
    lora_rank = 128
    dtype = torch.bfloat16

    if os.path.exists(os.path.join(model_path, "transformer_ema")):
        subfolder = "transformer_ema"
    else:
        subfolder = "transformer"

    transformer = ConsisIDTransformer3DModel.from_pretrained_cus(model_path, subfolder=subfolder)
    scheduler = CogVideoXDPMScheduler.from_pretrained(model_path, subfolder="scheduler")

    try:
        is_kps = transformer.config.is_kps
    except:
        is_kps = False

    # 1. load face helper models
    face_helper = FaceRestoreHelper(
        upscale_factor=1,
        face_size=512,
        crop_ratio=(1, 1),
        det_model='retinaface_resnet50',
        save_ext='png',
        device=device,
        model_rootpath=os.path.join(model_path, "face_encoder")
    )
    face_helper.face_parse = None
    face_helper.face_parse = init_parsing_model(model_name='bisenet', device=device, model_rootpath=os.path.join(model_path, "face_encoder"))
    face_helper.face_det.eval()
    face_helper.face_parse.eval()

    model, _, _ = create_model_and_transforms('EVA02-CLIP-L-14-336', os.path.join(model_path, "face_encoder", "EVA02_CLIP_L_336_psz14_s6B.pt"), force_custom_clip=True)
    face_clip_model = model.visual
    face_clip_model.eval()

    eva_transform_mean = getattr(face_clip_model, 'image_mean', OPENAI_DATASET_MEAN)
    eva_transform_std = getattr(face_clip_model, 'image_std', OPENAI_DATASET_STD)
    if not isinstance(eva_transform_mean, (list, tuple)):
        eva_transform_mean = (eva_transform_mean,) * 3
    if not isinstance(eva_transform_std, (list, tuple)):
        eva_transform_std = (eva_transform_std,) * 3
    eva_transform_mean = eva_transform_mean
    eva_transform_std = eva_transform_std

    face_main_model = FaceAnalysis(name='antelopev2', root=os.path.join(model_path, "face_encoder"), providers=['CUDAExecutionProvider'])
    handler_ante = insightface.model_zoo.get_model(f'{model_path}/face_encoder/models/antelopev2/glintr100.onnx', providers=['CUDAExecutionProvider'])
    face_main_model.prepare(ctx_id=0, det_size=(640, 640))
    handler_ante.prepare(ctx_id=0)

    face_clip_model.to(device, dtype=dtype)
    face_helper.face_det.to(device)
    face_helper.face_parse.to(device)
    transformer.to(device, dtype=dtype)
    free_memory()

    pipe = ConsisIDPipeline.from_pretrained(model_path, transformer=transformer, scheduler=scheduler, torch_dtype=dtype)
    # If you're using with lora, add this code
    if lora_path:
        pipe.load_lora_weights(lora_path, weight_name="pytorch_lora_weights.safetensors", adapter_name="test_1")
        pipe.fuse_lora(lora_scale=1 / lora_rank)

    scheduler_args = {}
    if "variance_type" in pipe.scheduler.config:
        variance_type = pipe.scheduler.config.variance_type
        if variance_type in ["learned", "learned_range"]:
            variance_type = "fixed_small"
        scheduler_args["variance_type"] = variance_type

    pipe.scheduler = CogVideoXDPMScheduler.from_config(pipe.scheduler.config, **scheduler_args)
    #pipe.to(device)

    pipe.enable_model_cpu_offload()
    pipe.enable_sequential_cpu_offload()
    pipe.vae.enable_slicing()
    pipe.vae.enable_tiling()

    os.makedirs(args.output_dir, exist_ok=True)

    upscale_model = load_sd_upscale("model_real_esran/RealESRGAN_x4.pth", device)
    frame_interpolation_model = load_rife_model("model_rife")

    def infer(
        prompt: str,
        image_input: str,
        num_inference_steps: int,
        guidance_scale: float,
        seed: int = 42,
    ):
        if seed == -1:
            seed = random.randint(0, 2**8 - 1)

        id_image = np.array(ImageOps.exif_transpose(Image.open(image_input)).convert("RGB"))
        id_image = resize_numpy_image_long(id_image, 1024)
        id_cond, id_vit_hidden, align_crop_face_image, face_kps = process_face_embeddings(face_helper, face_clip_model, handler_ante, 
                                                                            eva_transform_mean, eva_transform_std, 
                                                                            face_main_model, device, dtype, id_image, 
                                                                            original_id_image=id_image, is_align_face=True, 
                                                                            cal_uncond=False)

        if is_kps:
            kps_cond = face_kps
        else:
            kps_cond = None

        tensor = align_crop_face_image.cpu().detach()
        tensor = tensor.squeeze()
        tensor = tensor.permute(1, 2, 0)
        tensor = tensor.numpy() * 255
        tensor = tensor.astype(np.uint8)
        image  = ImageOps.exif_transpose(Image.fromarray(tensor))

        prompt = prompt.strip('"')

        generator = torch.Generator(device).manual_seed(seed) if seed else None

        video_pt = pipe(
            prompt=prompt,
            image=image,
            num_videos_per_prompt=1,
            num_inference_steps=num_inference_steps,
            num_frames=49,
            use_dynamic_cfg=False,
            guidance_scale=guidance_scale,
            generator=generator,
            id_vit_hidden=id_vit_hidden,
            id_cond=id_cond,
            kps_cond=kps_cond,
            output_type="pt",
        ).frames

        free_memory()
        return (video_pt, seed)

    def save_video(tensor: Union[List[np.ndarray], List[PIL.Image.Image]], fps: int = 8, output_dir = "output"):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        video_path = f"./{output_dir}/{timestamp}.mp4"
        os.makedirs(os.path.dirname(video_path), exist_ok=True)
        export_to_video(tensor, video_path, fps=fps)
        return video_path
        
    def convert_to_gif(video_path):
        clip = VideoFileClip(video_path)
        gif_path = video_path.replace(".mp4", ".gif")
        clip.write_gif(gif_path, fps=8)
        return gif_path

    def delete_old_files():
        while True:
            now = datetime.now()
            cutoff = now - timedelta(minutes=10)
            directories = [args.output_dir]

            for directory in directories:
                for filename in os.listdir(directory):
                    file_path = os.path.join(directory, filename)
                    if os.path.isfile(file_path):
                        file_mtime = datetime.fromtimestamp(os.path.getmtime(file_path))
                        if file_mtime < cutoff:
                            os.remove(file_path)
            time.sleep(600)

    threading.Thread(target=delete_old_files, daemon=True).start()

    latents, seed = infer(
        args.prompt,
        args.image_path,
        num_inference_steps=args.num_inference_steps,
        guidance_scale=args.guidance_scale,
        seed=args.seed,
    )

    batch_size = latents.shape[0]
    batch_video_frames = []
    for batch_idx in range(batch_size):
        pt_image = latents[batch_idx]
        pt_image = torch.stack([pt_image[i] for i in range(pt_image.shape[0])])

        image_np = VaeImageProcessor.pt_to_numpy(pt_image)
        image_pil = VaeImageProcessor.numpy_to_pil(image_np)
        batch_video_frames.append(image_pil)

    video_path = save_video(batch_video_frames[0], fps=math.ceil((len(batch_video_frames[0]) - 1) / 6), output_dir=args.output_dir)
    gif_path = convert_to_gif(video_path)

    print(f"Video saved to: {video_path}")
    print(f"GIF saved to: {gif_path}")

if __name__ == "__main__":
    main()
