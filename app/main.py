import cv2
import os
from pathlib import Path
import numpy as np


# set params
img_name = "img_example.png"
img_path = os.path.join(Path(__file__).parents[1], f"data/{img_name}")


def main(img_path:str) -> None:
    
    print(img_path)
    
    # image = cv2.imread(img_path)
    # image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    image = cv2.imdecode(
        np.fromfile(img_path, dtype=np.uint8),
        cv2.IMREAD_COLOR
    )
    
    cv2.imshow("example", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main(
        img_path=img_path,
    )