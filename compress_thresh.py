import numpy as np
import os
from PIL import Image

os.chdir("images")

palette = []
for v in [0, 85, 170, 255]:
    palette += [int(v), int(v), int(v)]
palette += [0,0,0]*(256-4)

for fname in os.listdir():
    if not fname.endswith("threshold_image.png") or fname.startswith("CMP"):
        continue
    n_im = np.array(Image.open(fname))[:,:,0]
    n_im[ n_im == 16 ] = 85
    im = Image.fromarray(n_im, mode="P")
    im.putpalette(palette)
    im.save(f"CMP_{fname}", bits=2)
	
