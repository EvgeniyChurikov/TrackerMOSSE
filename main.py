import os
from PIL import Image, ImageDraw
import torchvision.transforms as transforms
from tracker import Tracker


def main():
    source_dir = 'source_frames'
    output_dir = 'output_frames'
    top, left = 35, 120

    filenames = os.listdir(source_dir)
    frame_pil = Image.open(f'{source_dir}/{filenames[0]}').reduce(2).convert('RGB')
    frame_gray = transforms.Grayscale()(transforms.ToTensor()(frame_pil))[0]
    tracker = Tracker(frame_gray, top, left, 8, 0.8)
    for i in range(1, len(filenames)):
        frame_pil = Image.open(f'{source_dir}/{filenames[i]}').reduce(2).convert('RGB')
        frame_gray = transforms.Grayscale()(transforms.ToTensor()(frame_pil))[0]
        new_top, new_left = tracker.next(frame_gray)
        draw = ImageDraw.Draw(frame_pil)
        draw.rectangle((new_left, new_top, new_left+64, new_top+64), outline=(255, 0, 0), width=2)
        frame_pil.save(f'{output_dir}/{filenames[i]}')
        print(filenames[i])


if __name__ == '__main__':
    main()
