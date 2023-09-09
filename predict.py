# Prediction interface for Cog ⚙️
# https://github.com/replicate/cog/blob/main/docs/python.md

from cog import BasePredictor, Input, Path
import torch
from huggingface_hub import hf_hub_download
from diffusers import DiffusionPipeline, AutoPipelineForInpainting
from cog_sdxl.dataset_and_utils import TokenEmbeddingsHandler
from PIL import Image

from realesrgan import RealESRGANer
from basicsr.archs.rrdbnet_arch import RRDBNet
import os, cv2

class Predictor(BasePredictor):
    def setup(self) -> None:
        """Load the model into memory to make running multiple predictions efficient"""
        self.pipe = DiffusionPipeline.from_pretrained(
            "stabilityai/stable-diffusion-xl-base-1.0",
            torch_dtype=torch.float16,
            variant="fp16",
            cache_dir="sdxl-cache"
        ).to("cuda")
        self.pipe.load_lora_weights(
            "jbilcke-hf/sdxl-panorama",
            weight_name="lora.safetensors",
            cache_dir="lora-cache"
        )
        text_encoders = [self.pipe.text_encoder, self.pipe.text_encoder_2]
        tokenizers = [self.pipe.tokenizer, self.pipe.tokenizer_2]
        embedding_path = hf_hub_download(
            repo_id="jbilcke-hf/sdxl-panorama",
            filename="embeddings.pti",
            repo_type="model",
            cache_dir="embedding-cache"
        )
        embhandler = TokenEmbeddingsHandler(text_encoders, tokenizers)
        embhandler.load_embeddings(embedding_path)
        self.pipeInpaint = AutoPipelineForInpainting.from_pretrained(
            "diffusers/stable-diffusion-xl-1.0-inpainting-0.1",
            torch_dtype=torch.float16,
            variant="fp16",
            cache_dir="sdxl-inpaint-cache"
        ).to("cuda")
        

    def predict(
        self,
        prompt: str = Input(description="Prompt", default="tron world"),
        seed: int = Input(description="Leave blank to randomize the seed", default=None),
    ) -> Path:
        """Run a single prediction on the model"""
        if seed is None:
            seed = int.from_bytes(os.urandom(2), "big")
        print(f"Using seed: {seed}")

        prompt_sdxl = prompt + " in the style of <s0><s1>";
        image = self.pipe(
            prompt_sdxl,
            cross_attention_kwargs={"scale": 0.8},
            width=1024,
            height=512,
            generator=torch.manual_seed(seed),
        ).images[0]
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

        #Inpaint
        img_url = "4-square.png"
        mask_url = "mask.png"
        image = Image.open(img_url)
        mask_image = Image.open(mask_url)
        generator = torch.Generator(device="cuda").manual_seed(seed)
        inpaint = self.pipeInpaint(
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
        # Save the swapped image at 2048x1024
        image.save("6-final.png")

        return Path("6-final.png")
