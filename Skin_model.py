from transformers import AutoImageProcessor, AutoModelForImageClassification
from PIL import Image
import torch
import sys


processor = AutoImageProcessor.from_pretrained("Jayanth2002/dinov2-base-finetuned-SkinDisease")
model = AutoModelForImageClassification.from_pretrained("Jayanth2002/dinov2-base-finetuned-SkinDisease")


image_path = "sample_image3.png"  
image = Image.open(image_path).convert("RGB")


inputs = processor(images=image, return_tensors="pt")

with torch.no_grad():
    outputs = model(**inputs)

logits = outputs.logits
predicted_class = logits.argmax(-1).item()
probs = torch.nn.functional.softmax(logits, dim = -1)
top5 = torch.topk(probs, 5)
print("Top Predictions:")
for i in range(5):
    idx = top5.indices[0][i].item()
    confidence = top5.values[0][i].item()
    disease = model.config.id2label[idx]
    print(f"{disease}: {confidence:.2%}")


label = model.config.id2label[predicted_class]

print(f"Based on top probability, predicted skin disease might be {label}")

