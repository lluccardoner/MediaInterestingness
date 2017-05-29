import cv2

def get_num_frames(video_path):
    """ Return the number of frames of the video track of the video given """

    if cv2.__version__ >= '3.0.0':
        CAP_PROP_FRAME_COUNT = cv2.CAP_PROP_FRAME_COUNT

    else:
        CAP_PROP_FRAME_COUNT = cv2.cv.CV_CAP_PROP_FRAME_COUNT

    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        raise Exception('Could not open the video {}'.format(video_path))
    num_frames = int(cap.get(CAP_PROP_FRAME_COUNT))
    return num_frames


def get_duration(video_path):
    """ Return the duration of the video track of the video given """

    if cv2.__version__ >= '3.0.0':
        CAP_PROP_FRAME_COUNT = cv2.CAP_PROP_FRAME_COUNT
        CAP_PROP_FPS = cv2.CAP_PROP_FPS

    else:
        CAP_PROP_FRAME_COUNT = cv2.cv.CV_CAP_PROP_FRAME_COUNT
        CAP_PROP_FPS = cv2.cv.CV_CAP_PROP_FPS

    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        raise Exception("Could not open the video")
    num_frames = int(cap.get(CAP_PROP_FRAME_COUNT))
    fps = float(cap.get(CAP_PROP_FPS))
    duration = num_frames / fps
    # When everything done, release the capture
    cap.release()
    return duration


def get_fps(video_path):
    """ Return the fps of the video given """

    if cv2.__version__ >= '3.0.0':
        CAP_PROP_FPS = cv2.CAP_PROP_FPS

    else:
        CAP_PROP_FPS = cv2.cv.CV_CAP_PROP_FPS

    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        raise Exception('Could not open the video')
    fps = float(cap.get(CAP_PROP_FPS))
    # When everything done, release the capture
    cap.release()
    return fps



