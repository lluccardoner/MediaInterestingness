from PIL import Image

from image.SVM_image_features import load_and_set as l

# script for augmenting the dataset by fliping horizontally all the images
path = 'data/balanced_1/train/1/'

d = l.load_directory(path)
for name in d:
    img = Image.open(path + name)
    img_flipped = img.transpose(Image.FLIP_LEFT_RIGHT)
    img_flipped.save(path + 'f_' + name)
