import cv2
import argparse
def vid_to_frames(num):
    vidcap = cv2.VideoCapture('bad_policy/MontezumaRevenge%d.mp4' % num)
    success,image = vidcap.read()
    count = 0
    while success:
        print('writing', count)
        if count % 4 == 0 and count <= 4 * 385:
            cv2.imwrite(("bad_policy/video%d/frame%d.jpg" % (num ,count)), image)     # save frame as JPEG file      
        success,image = vidcap.read()
        count += 1

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--trial', type=int, default=0)
    args = parser.parse_args()
    vid_to_frames(args.trial)
