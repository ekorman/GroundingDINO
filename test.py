import time
import torch
import torch.nn as nn
from groundingdino.models.GroundingDINO.groundingdino import GroundingDINO
from groundingdino.util.inference import load_model, predict, preprocess_caption

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


def transform(img: Image.Image):
    ret = resize(img, [800], max_size=1333)
    ret = to_tensor(ret)
    ret = normalize(ret, [0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    return ret


proc_image = transform(image)


class Detector(nn.Module):
    def __init__(self, dino: GroundingDINO, text_prompt, device="cpu"):
        super().__init__()
        self.dino = dino
        caption = preprocess_caption(text_prompt)

        text_dict = dino._process_text(caption, device=device)
        self._text_dict = {k: v.detach() for k, v in text_dict.items()}
        # for v in self._text_dict.values():
        #     v.detach()
        #     v.requires_grad = False

    def forward(self, image):
        return self.dino.process_image(image, self._text_dict)


def run_detector():
    det = Detector(model, TEXT_PROMPT)
    out = det(proc_image.unsqueeze(0))
    print("detector output: ", out)


def run_inference():
    caption = preprocess_caption(TEXT_PROMPT)

    with torch.no_grad():
        out = model(proc_image.unsqueeze(0), captions=[caption])

    print("inference output: ", out)


def run_predict():
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


def onnx_export():
    det = Detector(model, TEXT_PROMPT)
    x = torch.rand_like(proc_image.unsqueeze(0))
    torch.onnx.export(det, x, "groundingdino.onnx", verbose=True)


def run_onnx():
    import onnxruntime as ort

    x = proc_image.unsqueeze(0).numpy()

    ort_session = ort.InferenceSession("groundingdino.onnx")
    ort_inputs = {"samples": x}
    N = 10
    start = time.time()
    for _ in range(N):
        ort_outs = ort_session.run(None, ort_inputs)

    print(f"onnx inference time: {(time.time() - start) / N}")

    print("onnx output: ", ort_outs)


# onnx_export()
run_onnx()
