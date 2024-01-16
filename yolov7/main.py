from pytube import YouTube
import os
import cv2
from detect_script import detect
import torch
import argparse
def download_video(link, new_filename):
    '''
    To download a YouTube video
    :param link: the YouTube video URL
    :param new_filename: the desired new filename
    :return: None
    '''
    yt = YouTube(link)
    print("Downloading")
    video = yt.streams.get_highest_resolution()
    video.download()
    default_filename = video.default_filename
    os.rename(default_filename, new_filename)


def extract_frames(video_path, output_folder, interval_seconds=60):
    '''
    This code uses OpenCV to open the video, read frames at regular intervals, and save them as individual image files.
    :param video_path: the path to the input video file
    :param output_folder: the path to the folder where frames will be saved
    :param interval_seconds:the interval at which frames should be extracted, with a default value of 60 seconds
    :return: None
    '''

    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_interval = int(interval_seconds * fps)
    os.makedirs(output_folder, exist_ok=True)
    for frame_number in range(0, total_frames, frame_interval):
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
        ret, frame = cap.read()
        if not ret:
            break
        output_path = os.path.join(output_folder, f"frame_{frame_number}.jpg")
        cv2.imwrite(output_path, frame)
    cap.release()

if __name__ == '__main__':
    filename = "output.mp4"

    download_video("https://youtu.be/OzUkvzyBttA?si=eztPrg4AkUFus9LB",filename)

    extract_frames(filename,"frame", 60)

    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', nargs='+', type=str, default='yolov7-e6e.pt', help='model.pt path(s)')
    parser.add_argument('--source', type=str, default='frame/', help='source')  # file/folder, 0 for webcam
    parser.add_argument('--img-size', type=int, default=640, help='inference size (pixels)')
    parser.add_argument('--conf-thres', type=float, default=0.25, help='object confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.45, help='IOU threshold for NMS')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--view-img', action='store_true', help='display results')
    parser.add_argument('--save-txt', action='store_true', help='save results to *.txt')
    parser.add_argument('--save-conf', action='store_true', help='save confidences in --save-txt labels')
    parser.add_argument('--nosave', action='store_true', help='do not save images/videos')
    parser.add_argument('--classes', nargs='+', type=int, help='filter by class: --class 0, or --class 0 2 3')
    parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
    parser.add_argument('--augment', action='store_true', help='augmented inference')
    parser.add_argument('--update', action='store_true', help='update all models')
    parser.add_argument('--project', default='runs/detect', help='save results to project/name')
    parser.add_argument('--name', default='exp', help='save results to project/name')
    parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
    parser.add_argument('--no-trace', action='store_true', help='don`t trace model')
    opt = parser.parse_args()

    with torch.no_grad():
        detect(opt=opt)

