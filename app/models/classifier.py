import torch
from torchvision.models import EfficientNet_B0_Weights, efficientnet_b0


def build_model(pretrained=False):
    if pretrained:
        print(
            "Using torchvision ImageNet EfficientNet-B0 weights (~21 MB); downloads if not cached.",
            flush=True,
        )
    try:
        model = efficientnet_b0(weights=EfficientNet_B0_Weights.DEFAULT if pretrained else None)
    except Exception as exc:
        raise RuntimeError(
            "Pretrained weights unavailable. Check connectivity/cache or explicitly set pretrained=false; this trains from scratch, not transfer learning."
        ) from exc
    model.classifier[1] = torch.nn.Linear(model.classifier[1].in_features, 7)
    return model
