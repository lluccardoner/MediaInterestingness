import json
import os

import video.pre_processing.video_prop as vp
from img_features import load_and_set as l

segments_dir_devset = '/home/lluc/Documents/ME16IN/devset/videos/video_{}/movies'
segments_dir_testset = '/home/lluc/Documents/ME16IN/testset/videos/video_{}/movies'

videos_dir_devset = '/home/lluc/Documents/ME16IN/devset/videos'
videos_dir_testset = '/home/lluc/Documents/ME16IN/testset/videos'

sets_dir = '/home/lluc/Documents/ME16IN/{}/videos'

devset_num_videos = 52
testset_num_videos = 26


def data_to_json():
    # -------------------------------------------------
    devset_dic = {}
    total_dev_segments = 0
    videos_dic = {}
    for v in range(devset_num_videos):
        print (v)
        segments_dic = {}
        segments = l.load_directory(segments_dir_devset.format(v))
        total_segments = 0
        total_frames = 0
        for s in segments:
            # segment info
            segment_path = os.path.join(segments_dir_devset.format(v), s)
            n = s.split('.')
            fps = vp.get_fps(segment_path)
            f = vp.get_num_frames(segment_path)
            d = vp.get_duration(segment_path)
            segment = {
                'fps': fps,
                'frames': f,
                'duration': d
            }
            segments_dic[s] = [segment]
            total_segments += 1
            total_frames += f
        videos_dic['video_{}'.format(v)] = []
        videos_dic['video_{}'.format(v)].append({
            'segments': segments_dic,
            'num_segments': total_segments,
            'num_frames': total_frames
        })
        total_dev_segments += total_segments
    devset_dic['devset'] = []
    devset_dic['devset'].append({
        'num_videos': devset_num_videos,
        'num_segments': total_dev_segments,
        'videos': videos_dic
    })
    # -------------------------------------------------
    testset_dic = {}
    total_train_segments = 0
    videos_dic = {}
    for v in range(devset_num_videos, devset_num_videos + testset_num_videos):
        print (v)
        segments_dic = {}
        segments = l.load_directory(segments_dir_testset.format(v))
        total_segments = 0
        total_frames = 0
        for s in segments:
            # segment info
            segment_path = os.path.join(segments_dir_testset.format(v), s)
            n = s.split('.')
            fps = vp.get_fps(segment_path)
            f = vp.get_num_frames(segment_path)
            d = vp.get_duration(segment_path)
            segment = {
                'fps': fps,
                'frames': f,
                'duration': d
            }
            segments_dic[s] = [segment]
            total_segments += 1
            total_frames += f
        videos_dic['video_{}'.format(v)] = []
        videos_dic['video_{}'.format(v)].append({
            'segments': segments_dic,
            'num_segments': total_segments,
            'num_frames': total_frames
        })
        total_train_segments += total_segments
    devset_dic['testset'] = []
    devset_dic['testset'].append({
        'num_videos': testset_num_videos,
        'num_segments': total_dev_segments,
        'videos': videos_dic
    })
    # -------------------------------------------------
    data = {}
    data['dataset'] = []
    data['dataset'].append(devset_dic)
    data['dataset'].append(testset_dic)

    with open('dataset.json', 'w') as outfile:
        json.dump(data, outfile)


