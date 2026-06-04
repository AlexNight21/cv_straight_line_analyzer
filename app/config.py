import os
from pathlib import Path


window_adapt_coef = 1.2

img_name = "img_example.png"
img_path = os.path.join(Path(__file__).parents[1], f"data/{img_name}")
