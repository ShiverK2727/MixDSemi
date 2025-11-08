from PIL import Image
import numpy as np
import glob, os
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from dataloaders import custom_transforms as tr

DATA_DIR = '/app/MixDSemi/data/ProstateSlice/BIDMC/train/image'
out_dir = '/app/MixDSemi/results/clahe_debug'
os.makedirs(out_dir, exist_ok=True)
imgs = sorted(glob.glob(os.path.join(DATA_DIR, '*.png')))
if len(imgs) == 0:
    print('No images found in', DATA_DIR)
    sys.exit(1)

path = imgs[0]
img = Image.open(path)
print('Loaded', path, 'mode', img.mode, 'size', img.size)

# ensure grayscale conversion if single channel
if img.mode == 'RGB' or img.mode == 'RGBA':
    img_rgb = img.convert('RGB')
else:
    img_rgb = img.convert('L')

# Instantiate CLAHE with strong settings
clahe = tr.AdaptiveCLAHE(p=1.0, clipLimit=5.0, tileGridSize=(8,8))

before = np.array(img_rgb)
print('Before: dtype', before.dtype, 'shape', before.shape, 'min/max', before.min(), before.max(), 'mean', before.mean())

out = clahe(img_rgb)
if isinstance(out, dict):
    out_img = out.get('image')
else:
    out_img = out

after = np.array(out_img)
print('After: dtype', after.dtype, 'shape', after.shape, 'min/max', after.min(), after.max(), 'mean', after.mean())

# Save images
before_path = os.path.join(out_dir, 'before.png')
after_path = os.path.join(out_dir, 'after.png')
Image.fromarray(before).save(before_path)
Image.fromarray(after).save(after_path)
print('Saved before/after to', before_path, after_path)
