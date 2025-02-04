import numpy as np
import torch
from groundingdino.util.inference import load_model, predict

from PIL import Image
from torchvision.transforms.functional import resize, to_tensor, normalize

device = "cpu"

model = load_model(
    "groundingdino/config/GroundingDINO_SwinT_OGC.py",
    "weights/groundingdino_swint_ogc.pth",
    device=device,
)
IMAGE_PATH = "date.png"  # .asset/cat_dog.jpeg"
TEXT_PROMPT = "cat"  # "chair . person . dog ."
BOX_TRESHOLD = 0.35
TEXT_TRESHOLD = 0.25


# _, proc_image = load_image(IMAGE_PATH)
image = Image.open(IMAGE_PATH).convert("RGB")


def new_transform(img: Image.Image):
    ret = resize(img, [800], max_size=1333)
    ret = to_tensor(ret)
    ret = normalize(ret, [0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    return ret


proc_image = new_transform(image)

boxes, logits, phrases = predict(
    model=model,
    image=proc_image,
    caption=TEXT_PROMPT,
    box_threshold=BOX_TRESHOLD,
    text_threshold=TEXT_TRESHOLD,
    device=device,
)

print(f"Boxes: {boxes}")
print(f"Logits: {logits}")
print(f"Phrases: {phrases}")


# expected output:
# Boxes: tensor([[0.5189, 0.4994, 0.4292, 0.9969]])
# Logits: tensor([0.9266])
# Phrases: ['cat']
