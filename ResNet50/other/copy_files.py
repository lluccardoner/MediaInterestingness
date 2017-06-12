from shutil import copyfile


# script for copying files to another directory
annotations_path = '/home/lluc/Documents/ME16IN/devset/annotations/devset-image.txt'
with open(annotations_path) as inputfile:
    j = 0
    for line in inputfile:
        l = line.strip().split(',')
        src = '/home/lluc/Documents/ME16IN/devset/videos/' + l[0] + '/images/' + l[1]
        if j < 4044:
            if l[2] == '0':
                dst = '/home/lluc/PycharmProjects/TFG/ResNet50/data/train/0/' + str(j) + '.jpg'
            else:
                dst = '/home/lluc/PycharmProjects/TFG/ResNet50/data/train/1/' + str(j) + '.jpg'
        else:
            if l[2] == '0':
                dst = '/home/lluc/PycharmProjects/TFG/ResNet50/data/validation/0/' + str(j) + '.jpg'
            else:
                dst = '/home/lluc/PycharmProjects/TFG/ResNet50/data/validation/1/' + str(j) + '.jpg'
        copyfile(src, dst)
        j += 1
