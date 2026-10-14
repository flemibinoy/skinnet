import matplotlib
import numpy as np
import torch
from PIL import Image


class GradCAM:
    def __init__(self, model, layer=None):
        self.model = model
        self.layer = layer if layer is not None else model.features[-1]

    def heatmap(self, tensor, target=None):
        if tensor.ndim != 4 or tensor.shape[0] != 1:
            raise ValueError("Grad-CAM requires one BCHW image")
        activations = []
        handle = self.layer.register_forward_hook(
            lambda module, inputs, output: activations.append(output)
        )
        was_training = self.model.training
        try:
            self.model.eval()
            with torch.enable_grad():
                x = tensor.detach().requires_grad_(True)
                logits = self.model(x)
                target = int(logits.argmax(1).item()) if target is None else int(target)
                if target < 0 or target >= logits.shape[1]:
                    raise ValueError("Invalid target class")
                gradients = torch.autograd.grad(logits[0, target], activations[0])[0]
                cam = (
                    (gradients.mean((2, 3), keepdim=True) * activations[0])
                    .sum(1, keepdim=True)
                    .relu()
                )
                cam = torch.nn.functional.interpolate(
                    cam, size=tensor.shape[2:], mode="bilinear", align_corners=False
                )
                cam = cam[0, 0].detach().cpu().numpy()
                maximum = float(cam.max())
                return cam / maximum if maximum > 0 else np.zeros_like(cam)
        finally:
            handle.remove()
            self.model.train(was_training)

    def overlay(self, image, tensor, target=None):
        cam = self.heatmap(tensor, target)
        resized = np.asarray(Image.fromarray((cam * 255).astype("uint8")).resize(image.size)) / 255
        heat = matplotlib.colormaps["magma"](resized)[..., :3]
        rgb = np.asarray(image.convert("RGB")) / 255
        return Image.fromarray((np.clip(0.6 * rgb + 0.4 * heat, 0, 1) * 255).astype("uint8"))
