from pathlib import Path
from typing import List
import torch
import torch.nn as nn
from torchvision.ops import box_convert
from PIL import Image
from functional_cat.interfaces import ObjectDetector, ImageInput
from functional_cat.data_types import Detection, BoundingPolygon
from functional_cat.io import download_to_cache, FileFromURL
from torchvision.transforms.functional import resize, to_tensor, normalize


from groundingdino.util.inference import load_model, preprocess_caption
from groundingdino.models.GroundingDINO.groundingdino import GroundingDINO
from groundingdino.util.utils import get_phrases_from_posmap


def transform(img: Image.Image):
    ret = resize(img, [800], max_size=1333)
    ret = to_tensor(ret)
    ret = normalize(ret, [0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    return ret


class Detector(nn.Module):
    def __init__(self, dino: GroundingDINO, text_prompt, device):
        super().__init__()
        self.dino = dino
        caption = preprocess_caption(text_prompt)

        with torch.no_grad():
            text_dict = dino._process_text(caption, device=device)
        self._text_dict = {k: v.detach() for k, v in text_dict.items()}

    def forward(self, image):
        return self.dino.process_image(
            image, {k: v.clone() for k, v in self._text_dict.items()}
        )


class FixedDINO(ObjectDetector):
    weights = FileFromURL(
        url="https://github.com/IDEA-Research/GroundingDINO/releases/download/v0.1.0-alpha/groundingdino_swint_ogc.pth",
        md5="075ebfa7242d913f38cb051fe1a128a2",
    )

    def __init__(
        self,
        class_labels: List[str],
        box_threshold=0.35,
        text_threshold=0.25,
        device: str = None,
    ):
        weights_path = download_to_cache(self.weights)

        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device

        dino = load_model(
            Path(__file__).parent / "config" / "GroundingDINO_SwinT_OGC.py",
            weights_path,
            device=device,
        ).to(
            device
        )  # looks like `load_model` doesn't place it on device??

        self._class_labels = class_labels

        text_prompt = " . ".join(class_labels) + " ."

        self.box_threshold = box_threshold
        self.text_threshold = text_threshold
        self.text_prompt = text_prompt
        self.detector = Detector(dino, text_prompt, device=device)

    def __call__(self, imgs: ImageInput, score_thres: float) -> List[List[Detection]]:
        assert len(imgs) == 1
        img_tensor = torch.stack([transform(img) for img in imgs]).to(self.device)
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
        return self._class_labels
