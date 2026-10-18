
import torch
from PIL import Image
from torchvision import transforms
from model import CNN

model = CNN()
model.load_state_dict(
    torch.load("cnn_mnist.pth", map_location="cpu")
)
model.eval()

# Use the exact 28x28 image used by the Verilog demo.
image = Image.open("processed_input.png").convert("L")

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

tensor = transform(image).unsqueeze(0)

with torch.no_grad():
    scores = model(tensor)
    prediction = scores.argmax(dim=1).item()

print("PyTorch prediction:", prediction)
print("Scores:", scores.squeeze().tolist())
