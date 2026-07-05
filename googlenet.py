import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image

# load pretrained GoogleNet (trained on imageNet)
model = models.googlenet(weights='DEFAULT')
model.eval()  # inference mode

# preprocessing pipeline for GoogleNet
transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

# load and preprocess image
image = Image.open('./images/mangoes.jpg').convert('RGB')
img_tensor = transform(image)

# add batch dimension (C, H, W) tp (1, C, H, W)
input_batch = img_tensor.unsqueeze(0)

#run inference
with torch.no_grad():
    output = model(input_batch)

#output.shape  (1, 1000), 1000 imageNet classes
probabilities = torch.nn.functional.softmax(output[0], dim=0)

#get 5 best predictions
top5_prob, top5_idx = torch.topk(probabilities, 5)
print("Top 5 predictions (class index, probability):")
for prob, idx in zip(top5_prob, top5_idx):
    print(f"Class {idx.item()}: {prob.item():.4f}")