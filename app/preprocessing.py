import io

from PIL import Image
from torchvision import transforms


TRANSFORM = transforms.Compose(
    [
        transforms.Grayscale(num_output_channels=1),
        transforms.Resize((28, 28)),
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)),
    ]
)


def preprocess_image(image_bytes: bytes):
    """Convert an uploaded image into the tensor format used by the CNN."""
    with Image.open(io.BytesIO(image_bytes)) as image:
        image = image.convert("RGB")
        tensor = TRANSFORM(image)

    return tensor.unsqueeze(0)
