from PIL import Image, ImageDraw

# Define image size
width = 1024
height = 1024

# Create a black image
image = Image.new('RGB', (width, height), 'black')

# Define the coordinates of the white square
i = 6
left = width/i
top = 0
right = width/i*(i-1)
bottom = height

# Create a drawing context
draw = ImageDraw.Draw(image)
draw.rectangle([left, top, right, bottom], fill='white')

# Save the image to a file (optional)
image.save('mask.png')