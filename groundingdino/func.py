from typing import List
import torch
import torch.nn as nn
from torchvision.ops import box_convert
from PIL import Image
from functional_cat.interfaces import ObjectDetector, ImageInput
from functional_cat.data_types import Detection, BoundingPolygon
from torchvision.transforms.functional import resize, to_tensor, normalize

from groundingdino.util.inference import preprocess_caption
from groundingdino.models.GroundingDINO.groundingdino import GroundingDINO
from groundingdino.util.utils import get_phrases_from_posmap


def transform(img: Image.Image):
    ret = resize(img, [800], max_size=1333)
    ret = to_tensor(ret)
    ret = normalize(ret, [0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    return ret


class Detector(nn.Module):
    def __init__(self, dino: GroundingDINO, text_prompt, device="cpu"):
        super().__init__()
        self.dino = dino
        caption = preprocess_caption(text_prompt)

        text_dict = dino._process_text(caption, device=device)
        self._text_dict = {k: v.detach() for k, v in text_dict.items()}

    def forward(self, image):
        return self.dino.process_image(image, self._text_dict)


class FixedDINO(ObjectDetector):
    def __init__(self, text_prompt, dino, box_threshold=0.35, text_threshold=0.25):
        self.box_threshold = box_threshold
        self.text_threshold = text_threshold
        self.text_prompt = text_prompt
        self.detector = Detector(dino, text_prompt)

    def __call__(self, imgs: ImageInput, score_thres: float) -> List[List[Detection]]:
        assert len(imgs) == 1
        img_tensor = torch.stack([transform(img) for img in imgs])
        with torch.no_grad():
            out = self.detector(img_tensor)
        prediction_logits = out["pred_logits"].cpu().sigmoid()[0]
        prediction_boxes = out["pred_boxes"].cpu()[0]
        mask = prediction_logits.max(dim=1)[0] > self.box_threshold
        logits = prediction_logits[mask]
        boxes = prediction_boxes[mask]

        tokenizer = self.detector.dino.tokenizer
        tokenized = tokenizer(self.text_prompt)

        phrases = [
            get_phrases_from_posmap(
                logit > self.text_threshold, tokenized, tokenizer
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
        return [s for s in self.text_prompt.replace(" ", "").split(".") if s != ""]
