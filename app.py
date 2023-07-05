import argparse
import os
import warnings
from pathlib import Path
from uuid import uuid4
from flask import Flask, request, render_template, redirect, url_for, session
from utils.lora import inject_inferable_lora
import torch
from diffusers import DPMSolverMultistepScheduler, TextToVideoSDPipeline
from models.unet_3d_condition import UNet3DConditionModel
from einops import rearrange
from torch.nn.functional import interpolate
from train import export_to_video, handle_memory_attention, load_primary_models
from utils.lama import inpaint_watermark
import random


app = Flask(__name__, template_folder='/content/Zeroscope_v2_FlaskUI/templates')
app.secret_key = 'ewffwefwe'

def initialize_pipeline(model, device="cuda", xformers=False, sdp=False):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")

        scheduler, tokenizer, text_encoder, vae, _unet = load_primary_models(model)
        del _unet #This is a no op
        unet = UNet3DConditionModel.from_pretrained(model, subfolder='unet')
        # unet.disable_gradient_checkpointing()
        
    pipeline = TextToVideoSDPipeline.from_pretrained(
        pretrained_model_name_or_path=model,
        scheduler=scheduler,
        tokenizer=tokenizer,
        text_encoder=text_encoder.to(device=device, dtype=torch.half),
        vae=vae.to(device=device, dtype=torch.half),
        unet=unet.to(device=device, dtype=torch.half),
    )
    pipeline.scheduler = DPMSolverMultistepScheduler.from_config(pipeline.scheduler.config)
    unet._set_gradient_checkpointing(value=False)
    handle_memory_attention(xformers, sdp, unet)
    vae.enable_slicing()
    return pipeline

@torch.inference_mode()
def inference(
    model,
    prompt,
    negative_prompt=None,
    batch_size=1,
    num_frames=16,
    width=256,
    height=256,
    num_steps=50,
    guidance_scale=9,
    init_video=None,
    init_weight=0.5,
    device="cuda",
    xformers=False,
    sdp=False,
    lora_path='',
    lora_rank=64,
    seed=0,
):
    with torch.autocast(device, dtype=torch.half):
        pipeline = initialize_pipeline(model, device, xformers, sdp)
        inject_inferable_lora(pipeline, lora_path, r=lora_rank)
        prompt = [prompt] * batch_size
        negative_prompt = ([negative_prompt] * batch_size) if negative_prompt is not None else None

        g_cuda = torch.Generator(device='cuda')
        g_cuda.manual_seed(seed)
        g_cpu = torch.Generator()
        g_cpu.manual_seed(seed)

        videos = pipeline(
                  prompt=prompt,
                  negative_prompt=negative_prompt,
                  num_frames=num_frames,
                  height=height,
                  width=width,
                  num_inference_steps=num_steps,
                  generator=g_cuda,
                  guidance_scale=guidance_scale,
                  output_type="pt",
              ).frames

        return videos

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        # Retrieve the form data
        model = request.form.get('model')
        prompt = request.form.get('prompt')
        negative_prompt = request.form.get('negative_prompt')
        batch_size = int(request.form.get('batch_size', 1))
        num_frames = int(request.form.get('num_frames', 30))
        width = int(request.form.get('width'))
        height = int(request.form.get('height'))
        num_steps = int(request.form.get('num_steps', 25))
        guidance_scale = float(request.form.get('guidance_scale', 23))
        init_video = None
        init_weight = float(request.form.get('init_weight', 0.5))
        device = request.form.get('device', 'cuda')
        xformers = request.form.get('xformers', False)
        sdp = request.form.get('sdp', False)
        lora_path = request.form.get('lora_path', '')
        lora_rank = int(request.form.get('lora_rank', 64))
        seed= random.randint(0, ((1<<63)-1))

        output_dir = './output'
        fps = 10
        remove_watermark = False

        args = {
            "model": model,
            "prompt": prompt,
            "negative_prompt": negative_prompt,
            "batch_size": batch_size,
            "num_frames": num_frames,
            "width": int(round(width/8.0)*8.0),
            "height": int(round(height/8.0)*8.0),
            "num_steps": num_steps,
            "guidance_scale": guidance_scale,
            "init_video": init_video,
            "init_weight": init_weight,
            "device": device,
            "xformers": xformers,
            "sdp": sdp,
            "lora_path": lora_path,
            "lora_rank": lora_rank,
            "seed": seed
        }

        import decord

        decord.bridge.set_bridge("torch")

        videos = inference(**args)

        os.makedirs(output_dir, exist_ok=True)
        out_stem = f"{output_dir}/"

        prompt = prompt[:25]
        out_stem += f"{prompt}"
        out_stem += " "
        out_stem += f"{seed}"

        for video in videos:
            if remove_watermark:
                video = rearrange(video, "c f h w -> f c h w").add(1).div(2)
                video = inpaint_watermark(video)
                video = rearrange(video, "f c h w -> f h w c").clamp(0, 1).mul(255)
            else:
                video = rearrange(video, "c f h w -> f h w c").clamp(-1, 1).add(1).mul(127.5)

            video = video.byte().cpu().numpy()

            video_name = f"{out_stem} {str(uuid4())[:8]}.mp4"
            export_to_video(video, video_name, fps)
            session['latest_vid'] = video_name

        return render_template('video.html')

    return render_template('index.html')


@app.route('/video', methods=['POST'])
def display_video():
    video_name = session.get('latest_vid')
    video_url = "/content/Text-To-Video-Finetuning/output/" + video_name  # Replace with the actual URL or path
    return render_template('video.html', video_url=video_url)


if __name__ == "__main__":
    app.run()
