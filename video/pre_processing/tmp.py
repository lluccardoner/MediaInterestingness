import h5py

in_file = h5py.File('/home/lluc/PycharmProjects/TFG/video/data/features_clips_devset.h5py')
out_file = h5py.File('/home/lluc/PycharmProjects/TFG/video/data/features_clips.h5py')

devset = out_file.create_group('devset')
for v in in_file['devset']:
    print (v)
    vid = devset.create_group(v)
    for c in in_file['devset'][v]:
        print (c)
        data = in_file['devset'][v][c][()]
        vid.create_dataset(c, data=data)

in_file.close()
out_file.close()
