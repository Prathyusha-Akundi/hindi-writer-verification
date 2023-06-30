import cv2

import numpy as np
from locate_text import TextDetector, LineDetector
import os
import sys
ROOT = sys.argv[1]
OUT = sys.argv[2]
image_dirs = sorted(os.listdir(ROOT))


line_detector = LineDetector()
text_detector = TextDetector()

for idir in image_dirs:
    dir_path = os.path.join(ROOT, idir)
    images = sorted(os.listdir(dir_path))
    # print('Dir path', idir)
    for img_name in images:
        img_path = os.path.join(dir_path, img_name)
        image = cv2.imread(img_path)

        try:
            lines = line_detector.get_lines(image)
            xmin = min(lines[:,:,0])[0]
            xmax = max(lines[:,:,2])[0]
            image = image[:, xmin:xmax, :]
            xmin, ymin, xmax, ymax = text_detector.get_text_box(image)

            #check area
            if (ymax-ymin) * (xmax-xmin) <400:
                continue
            # cv2.rectangle(image, (xmin, ymin), (xmax, ymax), (36,255,12), 3)
            
            #output dims
            x0 = max(0, xmin-15)
            x1 = min(image.shape[1], xmax+15)
            y0 = max(0, ymin-15)
            y1 = min(image.shape[0], ymax+15)
            out_img = image[y0:y1, x0:x1,:]
            os.system(f'mkdir -p {os.path.join(OUT, idir)}')
            cv2.imwrite(f'{os.path.join(OUT, idir, img_name)}', out_img)

        except:
            pass
  

