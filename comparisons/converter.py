import os
import ffmpeg

start_dir = os.getcwd()

def convert_to_mp4(mkv_file):
    name, ext = os.path.splitext(mkv_file)
    out_name = name + ".mp4"
    ffmpeg.input(mkv_file).output(out_name).run()
    print("Finished converting {}".format(mkv_file))


files = [
    r"C:\Users\USER\tracking_dataset\gt\mot_challenge\MOT16-all\test\MOT16-05\video\video.mkv",
    r"C:\Users\USER\tracking_dataset\gt\mot_challenge\MOT16-all\test\MOT16-07\video\video.mkv",
    r"C:\Users\USER\tracking_dataset\gt\mot_challenge\MOT16-all\train\MOT16-06\video\video.mkv"
    r"C:\Users\USER\tracking_dataset\gt\mot_challenge\MOT16-all\train\MOT16-08\video\video.mkv"
]

for i in files:
    print(i)
    convert_to_mp4(i)