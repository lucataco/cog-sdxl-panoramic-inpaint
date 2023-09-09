# sdxl-panoramic-inpaint Cog model

This is a custom model implementing a workflow to get a seamless 360 sdxl image, as a Cog model. [Cog packages machine learning models as standard containers.](https://github.com/replicate/cog)

First, download the pre-trained weights:

    cog run script/download-weights

Then, you can run predictions:

    cog predict -i prompt="tron world"

## Workflow:

1.SDXL image 1024x512

![alt text](1-base.png)

2. Cut image in half vertically, swap left and right sections

![alt text](2-swap.png)

3. GFPGAN upscale 2x

![alt text](3-upscaled.png)

4. Crop out 1024x1024 middle section to inpaint seam

![alt text](4-square.png)

5. Put flush seam back into upscaled image

![alt text](output.png)
