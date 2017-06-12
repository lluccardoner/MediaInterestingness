# copy videos with less than 16 frames

import os
from shutil import copyfile

import video.pre_processing.video_prop as vp
from SVM_image_features import load_and_set as l


def copy_videos():
    count = 0
    text = '/home/lluc/Documents/ME16IN/devset/converted_videos.txt'
    # with open(text, 'w') as inputfile:
    for v in range(52):
        dir = '/home/lluc/Documents/ME16IN/devset/videos/video_{}/movies'.format(v)
        mov = l.load_directory(dir)
        for m in mov:
            path = os.path.join(dir, m)
            f = vp.get_num_frames(path)
            if f < 16:
                print(path)
                dir2 = '/home/lluc/Documents/ME16IN/devset/videos/video_{}'.format(v)
                dst = os.path.join(dir2, m)
                # inputfile.write("video_{}, {} \n".format(v, m))
                copyfile(path, dst)
                count += 1
    print("Total {}".format(count))
    # inputfile.close()


def check_frames():
    # out_file = open("/home/lluc/Documents/ME16IN/devset/video_frames.txt", 'w')
    count = 0
    count1 = 0
    count2 = 0
    for i in range(52):
        # dir = '/home/lluc/Documents/ME16IN/devset/videos/video_{}'.format(i)
        dir = '/home/lluc/PycharmProjects/TFG/video/data/devset/video_{}/movies'.format(i)
        mov = l.load_directory(dir)
        for m in mov:
            path = os.path.join(dir, m)
            try:
                print(path)
                f = vp.get_num_frames(path)
                # print(path)
                # out_file.write("video_{},{},{}\n".format(i, m, f))
                if f < 16:
                    count1 += 1
                    # copyfile(path, os.path.join("/home/lluc/PycharmProjects/TFG/video/data/devset", str(i) + "_" + m))
                else:
                    count += 1
            except:
                print(path)
                count2 += 1
                # out_file.write("video_{},{},{}\n".format(i, m, 0))
    print (count, count1, count2)
    # out_file.close()


check_frames()


def tests():
    less_frames = 0
    errors = 0
    moved = 0

    # new_path = '/home/lluc/Documents/ME16IN/devset/251_videos'
    # for i in range(52):
    # path = "/home/lluc/Documents/ME16IN/devset/videos/video_{}/".format(i)
    path2 = "/home/lluc/Documents/ME16IN/devset/251_videos"
    for v in l.load_directory(path2):
        if v.startswith("dup_"):
            # print(v)
            try:
                f = vp.get_num_frames(os.path.join(path2, v))
                if f >= 16:
                    x = v.split("_")
                    path = "/home/lluc/Documents/ME16IN/devset/videos/video_{}".format(x[1])
                    # print (os.path.join(path, x[0]+"_"+x[2]))
                    # move(os.path.join(path2, v), os.path.join(path, os.path.join(path, x[0]+"_"+x[2])))
                    # print (os.path.join(path, x[2]))
                    # move(os.path.join(path2, x[1] + "_" + x[2]), os.path.join(path, v))
                    moved += 1
                elif f < 16:
                    print(f)
                    less_frames += 1
                    # copyfile(os.path.join(path2, v), os.path.join(path, v))
            except:
                # print(os.path.join(path2, v))
                errors += 1
                # copyfile(os.path.join(path, v[4:]), os.path.join(new_path, "{}_{}".format(i, v[4:])))
    print(moved, less_frames, errors, less_frames + errors)


def new_video_database():
    from shutil import move

    copied = 0
    not_copied = 0
    err = 0
    for i in range(52):
        in_dir = "/home/lluc/PycharmProjects/TFG/video/data/devset/video_{}".format(i)
        out_dir = "/home/lluc/PycharmProjects/TFG/video/data/devset/video_{}/movies".format(i)
        d = l.load_directory(in_dir)
        for v in d:
            if v.startswith("dup_"):
                # print(v[4:])
                src = os.path.join(in_dir, v)
                dst = os.path.join(out_dir, v[4:])
                # print(src, dst)
                try:
                    f = vp.get_num_frames(src)
                    if f >= 16:
                        move(src, dst)
                        copied += 1
                    else:
                        not_copied += 1
                except:
                    err += 1

    print (copied, not_copied, err)
