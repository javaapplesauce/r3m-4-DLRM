import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class ConceptMasker(nn.Module):
    """Generates concept-aware binary masks using Grounding DINO + SAM2.

    Given a natural language task description (e.g. "the red mug"), this module
    localizes the task-relevant object and produces a spatial mask at feature-map
    resolution. The mask is used to filter dense visual features, zeroing out
    background and distractor regions.

    When grounding models are unavailable (CPU-only, no weights downloaded),
    falls back to an all-ones mask (no filtering).
    """

    def __init__(self, threshold=0.5, device="cpu"):
        super().__init__()
        self.threshold = threshold
        self._device = device
        self._grounding_model = None
        self._sam_predictor = None
        self._initialized = False

    def _lazy_init(self):
        if self._initialized:
            return
        self._initialized = True
        try:
            from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection

            self._grounding_processor = AutoProcessor.from_pretrained(
                "IDEA-Research/grounding-dino-tiny"
            )
            self._grounding_model = AutoModelForZeroShotObjectDetection.from_pretrained(
                "IDEA-Research/grounding-dino-tiny"
            ).to(self._device)
            self._grounding_model.eval()
            for p in self._grounding_model.parameters():
                p.requires_grad = False
        except Exception:
            self._grounding_model = None

        try:
            from sam2.build_sam import build_sam2
            from sam2.sam2_image_predictor import SAM2ImagePredictor

            sam = build_sam2(
                "sam2_hiera_l.yaml", "sam2_hiera_large.pt", device=self._device
            )
            self._sam_predictor = SAM2ImagePredictor(sam)
        except Exception:
            self._sam_predictor = None

    @torch.no_grad()
    def _get_bounding_box(self, image_pil, text):
        """Use Grounding DINO to get a bounding box for the text query."""
        inputs = self._grounding_processor(
            images=image_pil, text=text, return_tensors="pt"
        ).to(self._device)
        outputs = self._grounding_model(**inputs)
        results = self._grounding_processor.post_process_grounded_object_detection(
            outputs,
            inputs["input_ids"],
            box_threshold=0.25,
            text_threshold=0.25,
            target_sizes=[image_pil.size[::-1]],
        )[0]
        if len(results["boxes"]) == 0:
            return None
        best = results["scores"].argmax()
        return results["boxes"][best].cpu().numpy()

    @torch.no_grad()
    def forward(self, images, text, feature_h, feature_w):
        """
        Args:
            images: (B, 3, H, W) tensor in [0, 255].
            text: str, natural language description of the target object.
            feature_h: int, height of the feature map to produce mask for.
            feature_w: int, width of the feature map to produce mask for.
        Returns:
            masks: (B, feature_h, feature_w, 1) binary mask tensor.
        """
        self._lazy_init()
        B = images.shape[0]
        device = images.device

        if self._grounding_model is None or self._sam_predictor is None:
            return torch.ones(B, feature_h, feature_w, 1, device=device)

        from PIL import Image as PILImage

        masks = []
        for i in range(B):
            img_np = images[i].permute(1, 2, 0).cpu().numpy().astype(np.uint8)
            img_pil = PILImage.fromarray(img_np)

            box = self._get_bounding_box(img_pil, text)
            if box is None:
                masks.append(torch.ones(feature_h, feature_w, 1, device=device))
                continue

            self._sam_predictor.set_image(img_np)
            sam_masks, _, _ = self._sam_predictor.predict(
                box=box[None], multimask_output=False
            )
            mask = torch.from_numpy(sam_masks[0]).float()
            mask = F.interpolate(
                mask.unsqueeze(0).unsqueeze(0),
                size=(feature_h, feature_w),
                mode="nearest",
            ).squeeze(0).squeeze(0).unsqueeze(-1)
            masks.append(mask.to(device))

        return torch.stack(masks)
