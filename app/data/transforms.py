from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms as T


def transform(size, training=False):
    ops = [T.Resize((size, size))]
    if training:
        ops += [T.RandomHorizontalFlip(), T.RandomVerticalFlip(), T.RandomRotation(20)]
    return T.Compose(
        ops + [T.ToTensor(), T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])]
    )


class ManifestDataset(Dataset):
    def __init__(self, frame, size, training=False):
        self.frame = frame.reset_index(drop=True)
        self.transform = transform(size, training)

    def __len__(self):
        return len(self.frame)

    def __getitem__(self, index):
        row = self.frame.iloc[index]
        with Image.open(row.path) as image:
            tensor = self.transform(image.convert("RGB"))
        return tensor, int(row.label)
