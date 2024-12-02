import torch
from diffusers import CogVideoXImageToVideoPipeline
from diffusers.utils import export_to_video, load_image

# prompt = "BW_STYLE A black and white animated scene unfolds with an anthropomorphic goat surrounded by musical notes and symbols, suggesting a playful environment. Mickey Mouse appears, leaning forward in curiosity as the goat remains still. The goat then engages with Mickey, who bends down to converse or react. The dynamics shift as Mickey grabs the goat, potentially in surprise or playfulness, amidst a minimalistic background. The scene captures the evolving relationship between the two characters in a whimsical, animated setting, emphasizing their interactions and emotions."
# image = load_image(image="/home/robotics/cogvideox-factory/img2.jpg")
# prompt = "a group of people talking to each other and laughing on the street"
# image = load_image(image="/home/robotics/vvla/street.png")
# prompt = "Move the silver pot from in front of the red can, to next to the blue towel at the front edge of the table."
prompt = "put the corn in the sink"
# prompt = "put pumpkin on plate"
image = load_image(image="/home/robotics/cogvideox-factory/img19.jpg")
pipe = CogVideoXImageToVideoPipeline.from_pretrained(
    "THUDM/CogVideoX-5B-I2V",
    torch_dtype=torch.bfloat16
).to("cuda:0")

pipe.load_lora_weights("/home/robotics/cogvideox-factory/models/cogvideox-lora__optimizer_adamw__steps_3000__lr-schedule_cosine_with_restarts__learning-rate_1e-4/checkpoint-1200", adapter_name="cogvideox-lora")
pipe.set_adapters(["cogvideox-lora"], [1.0])

pipe.vae.enable_tiling()
pipe.vae.enable_slicing()

video = pipe(
    prompt=prompt,
    image=image,
    num_videos_per_prompt=1,
    num_inference_steps=10,
    # height=512,
    # width=512,
    num_frames=49,
    guidance_scale=6,
    generator=torch.Generator(device="cuda:0").manual_seed(42),
).frames[0]


export_to_video(video, "output8.mp4", fps=8)
