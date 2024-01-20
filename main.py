import os
from PIL import Image, ImageDraw
from tracker import Tracker


def main():
    source_dir = 'source_frames'
    output_dir = 'output_frames'
    left, top, width, height = 675, 350, 120, 140

    filenames = os.listdir(source_dir)
    frame = Image.open(f'{source_dir}/{filenames[0]}').convert('RGB')
    tracker = Tracker(frame, top, left, height, width, 8, 0.125, 10)
    for i in range(1, len(filenames)):
        frame = Image.open(f'{source_dir}/{filenames[i]}').convert('RGB')
        top, left = tracker.next(frame)
        draw = ImageDraw.Draw(frame)
        draw.rectangle((left, top, left+width, top+height), outline=(255, 0, 0), width=2)
        frame.save(f'{output_dir}/{filenames[i]}')
        print(filenames[i])


if __name__ == '__main__':
    main()
