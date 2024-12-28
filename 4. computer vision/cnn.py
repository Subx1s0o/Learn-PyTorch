
import torch

images = torch.rand(32, 1, 28, 28)
test_image=images[0]
print(f"Shape of the batch of images: {images.shape}")
print(f"Shape of the single image: {test_image.shape}")
print(f"Single image: {test_image}")

conv_layer = torch.nn.Conv2d(in_channels=3, out_channels=10, kernel_size=3, stride=1, padding=1)
output = conv_layer(test_image)
print(f"Shape of the output: {output.shape}")
print(f"Output: {output}")

