from torchvision import transforms
from PIL import Image
import io

def preprocess_image(image_bytes):
    # 1. Open the image from the uploaded bytes
    image = Image.open(io.BytesIO(image_bytes))
    
    # 2. Define the exact same transformations used in training
    transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        transforms.Resize((28, 28)),
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    # 3. Apply transforms
    tensor = transform(image)
    
    # 4. Add a batch dimension. CNN expects [batch_size, channels, height, width]
    # Our single image goes from [1, 28, 28] to [1, 1, 28, 28]
    return tensor.unsqueeze(0)