import numpy as np
import torch

class Cutout(object):
    """Randomly mask out one or more patches from an image.
    Args:
        n_holes (int): Number of patches to cut out of each image.
        length (int): The length (in pixels) of each square patch.
    """
    def __init__(self, n_holes, length):
        self.n_holes = n_holes
        self.length = length

    def __call__(self, img):
        """
        Args:
            img (Tensor): Tensor image of size (C, H, W).
        Returns:
            Tensor: Image with n_holes of dimension length x length cut out of it.
        """
        h = img.size(1)
        w = img.size(2)

        mask = np.ones((h, w), np.float32)

        for n in range(self.n_holes):
            y = np.random.randint(h)
            x = np.random.randint(w)

            y1 = np.clip(y - self.length // 2, 0, h)
            y2 = np.clip(y + self.length // 2, 0, h)
            x1 = np.clip(x - self.length // 2, 0, w)
            x2 = np.clip(x + self.length // 2, 0, w)

            mask[y1: y2, x1: x2] = 0.

        mask = torch.from_numpy(mask)
        mask = mask.expand_as(img)
        img = img * mask

        return img


class RandomResizedCropSquareNoPad(object):
    """
    This is the no padding version of torchvision's RandomResizedCrop.
    """

    def __init__(self, size, scale=(0.08, 1.0)):
        self.size = size
        self.scale = scale

    def __call__(self, img_pil):
        """
        Args:
            img (PIL image) of size (C, H, W)
        Returns:
            A square cropped version of the original image.

        """
        img = np.array(img_pil)

        height, width, num_channels = img.shape

        ratio = np.random.uniform(self.scale[0], self.scale[1])

        max_width = int(min(height, width) * ratio)

        x_min, x_max = max_width // 2, height - max_width // 2
        y_min, y_max = max_width // 2, width - max_width // 2

        x, y = np.random.randint(x_min, x_max), np.random.randint(y_min, y_max)

        img_rescaled = img[x-  max_width : x + max_width, y - max_width : y + max_width]
        img_resized = cv2.resize(img_rescaled, (self.size, self.size))

        img_pil = Image.fromarray(img_resized)

        return img_pil
