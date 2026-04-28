import cv2
import numpy as np
from PIL import Image

class OpticalNoiseGenerator:
    def __init__(self, base_size=(256, 256)):
        self.base_size = base_size

    def generate_seed_pattern(self, seed: int):
        np.random.seed(seed)
        # Create a base noise pattern representing physical micro-texture
        pattern = np.random.randint(0, 256, self.base_size, dtype=np.uint8)
        # Add optical artifacts (blur, light, warp)
        pattern = cv2.GaussianBlur(pattern, (5, 5), 0)
        
        # Add random light gradient
        x = np.linspace(0.5, 1.5, self.base_size[1])
        y = np.linspace(0.5, 1.5, self.base_size[0])
        X, Y = np.meshgrid(x, y)
        gradient = (X * Y * np.random.rand()).astype(np.float32)
        pattern = np.clip(pattern * gradient, 0, 255).astype(np.uint8)
        
        return Image.fromarray(pattern)

    def generate_batch(self, gtid: str, count: int = 5):
        seed_base = int(hash(gtid) % (10**8))
        return [self.generate_seed_pattern(seed_base + i) for i in range(count)]
