import time
from typing import List
import torch
import torch.nn as nn
from groundingdino.models.GroundingDINO.groundingdino import GroundingDINO
from groundingdino.util.inference import load_model, predict, preprocess_caption
from groundingdino.util.utils import get_phrases_from_posmap

from functional_cat.interfaces import ObjectDetector, ImageInput
from functional_cat.data_types import Detection, BoundingPolygon

from torchvision.ops import box_convert

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


class FixedDINO(ObjectDetector):
    def __init__(self, text_prompt):
        self.text_prompt = text_prompt
        self.detector = Detector(model, text_prompt)

    def __call__(self, imgs: ImageInput, score_thres: float) -> List[List[Detection]]:
        assert len(imgs) == 1
        img_tensor = torch.stack([transform(img) for img in imgs])
        with torch.no_grad():
            out = self.detector(img_tensor)
        prediction_logits = out["pred_logits"].cpu().sigmoid()[0]
        prediction_boxes = out["pred_boxes"].cpu()[0]
        mask = prediction_logits.max(dim=1)[0] > BOX_TRESHOLD
        logits = prediction_logits[mask]
        boxes = prediction_boxes[mask]

        tokenizer = self.detector.dino.tokenizer
        tokenized = tokenizer(self.text_prompt)

        phrases = [
            get_phrases_from_posmap(
                logit > TEXT_TRESHOLD, tokenized, tokenizer
            ).replace(".", "")
            for logit in logits
        ]

        h, w = imgs[0].height, imgs[0].width  # img_tensor.shape[-2:]
        boxes = boxes * torch.Tensor([w, h, w, h])
        xyxy = box_convert(boxes=boxes, in_fmt="cxcywh", out_fmt="xyxy").numpy()
        logits = logits.max(dim=1)[0]

        return [
            [
                Detection(
                    class_label=phrase,
                    boundary=BoundingPolygon.from_bbox(*xyxy[i].tolist()),
                    score=logit.item(),
                )
                for i, (logit, phrase) in enumerate(zip(logits, phrases))
                if logit.item() > score_thres
            ]
        ]

    @property
    def class_labels(self) -> List[str]:
        return [self.text_prompt]


def run_func():
    img = Image.open(IMAGE_PATH).convert("RGB")
    detector = FixedDino(TEXT_PROMPT)
    dets = detector([img], 0.5)[0]
    print("functional output: ", dets)

    for det in dets:
        img = det.draw_on_image(img)
    img.show()


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


run_func()
