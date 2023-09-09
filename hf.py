import torch
from huggingface_hub import hf_hub_download
from diffusers import DiffusionPipeline, AutoPipelineForInpainting
from cog_sdxl.dataset_and_utils import TokenEmbeddingsHandler
from PIL import Image
from realesrgan import RealESRGANer
from basicsr.archs.rrdbnet_arch import RRDBNet
import os, cv2

# Parameters
# prompt="hdri view, star wars cantina in the style of <s0><s1>
prompt="tron world in the style of <s0><s1>"
seed=1335

# SDXL panorama LoRA
pipe = DiffusionPipeline.from_pretrained(
        "stabilityai/stable-diffusion-xl-base-1.0",
        torch_dtype=torch.float16,
        variant="fp16",
).to("cuda")
pipe.load_lora_weights("jbilcke-hf/sdxl-panorama", weight_name="lora.safetensors")
text_encoders = [pipe.text_encoder, pipe.text_encoder_2]
tokenizers = [pipe.tokenizer, pipe.tokenizer_2]
embedding_path = hf_hub_download(repo_id="jbilcke-hf/sdxl-panorama", filename="embeddings.pti", repo_type="model")
embhandler = TokenEmbeddingsHandler(text_encoders, tokenizers)
embhandler.load_embeddings(embedding_path)
image = pipe(
    prompt,
    cross_attention_kwargs={"scale": 0.8},
    width=1024,
    height=512,
    generator=torch.manual_seed(seed),
).images[0]
# Output base Panoramic-LoRA image at 1024x512
image.save("1-base.png")

# Calculate the midpoint to split the image in half
width, height = image.size
midpoint = width // 2
left_half = image.crop((0, 0, midpoint, height))
right_half = image.crop((midpoint, 0, width, height))
image.paste(right_half, (0, 0))
image.paste(left_half, (midpoint, 0))
# Save the swapped image at 1024x512
image.save("2-swap.png")

# Upscale image 2x
model_name = 'RealESRGAN_x4plus'
img = cv2.imread(str("2-swap.png"), cv2.IMREAD_UNCHANGED)
model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=4)
model_path = os.path.join('realesrgan', model_name + ".pth")
upsampler = RealESRGANer(
    scale=4,
    model_path=model_path,
    model=model,
    tile=0,
    tile_pad=10,
    pre_pad=0,
    half=True)
output, _ = upsampler.enhance(img, outscale=2)
# Save image at 2048x1024
cv2.imwrite("3-upscaled.png", output)

# Crop square part to inpaint seam
image = Image.open("3-upscaled.png")
width, height = image.size
left = (width - height) // 2
top = 0
right = left + height
bottom = height
# Crop the middle square 1024x1024
middle_square = image.crop((left, top, right, bottom))
middle_square.save("4-square.png")

# Inpaint
pipe = AutoPipelineForInpainting.from_pretrained(
    "diffusers/stable-diffusion-xl-1.0-inpainting-0.1",
    torch_dtype=torch.float16,
    variant="fp16"
).to("cuda")
img_url = "4-square.png"
mask_url = "mask.png"
image = Image.open(img_url)
mask_image = Image.open(mask_url)
generator = torch.Generator(device="cuda").manual_seed(seed)
inpaint = pipe(
  prompt=prompt,
  image=image,
  mask_image=mask_image,
  guidance_scale=8.0,
  num_inference_steps=20,  # steps between 15 and 30 work well for us
  strength=0.99,  # make sure to use `strength` below 1.0
  generator=generator,
).images[0]
inpaint.save("5-inpainted.png")

# Add inpainted image back to original upscaled image
image = Image.open("3-upscaled.png")
width, height = image.size
left = (width - height) // 2
image.paste(inpaint, (left, 0))
# Save the swapped image at 1024x512
image.save("6-final.png")

