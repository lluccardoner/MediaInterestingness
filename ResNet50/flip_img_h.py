from PIL import Image
from features import load_and_set as l

path = 'data/balanced_1/train/1/'

d = l.load_directory(path)
for name in d:
    img = Image.open(path + name)
    img_flipped = img.transpose(Image.FLIP_LEFT_RIGHT)
    img_flipped.save(path + 'f_' + name)
