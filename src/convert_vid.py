import PIL
import torchvision.transforms as transforms
import numpy as np
import torch
import time
import cv2
import argparse
import os


if __name__ == '__main__':
    vid_path = '../input/vid/flip.mkv'
    name = vid_path.split('/')[-1]
    ext = name.split('.')[-1]
    new_name = name.split('.'+ext)[0]
    save_vid_path = f'../output/vid/vfi_{new_name}'
    saved_model_path = '../models/weights.pt'
    print_every = 1

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = torch.load(saved_model_path, map_location=torch.device(device))
    model.eval()


    video_capture = cv2.VideoCapture(vid_path)
    fps = video_capture.get(cv2.CAP_PROP_FPS)
    frame_count = int(video_capture.get(cv2.CAP_PROP_FRAME_COUNT))
    success, image = video_capture.read()


    width, height = image.shape[1], image.shape[0]
    video_writer = cv2.VideoWriter('{}.mp4'.format(save_vid_path), cv2.VideoWriter_fourcc(*'MP4V') , round(fps*2.0), (width, height))

    if width % 64 != 0:
        width_pad = int((np.floor(width / 64) + 1) * 64 - width)
    else:
        width_pad = 0
    if height % 64 != 0:
        height_pad = int((np.floor(height / 64) + 1) * 64 - height)
    else:
        height_pad = 0
    transforms = transforms.Compose([
        transforms.Pad((width_pad, height_pad, 0, 0)),
        transforms.ToTensor()
    ])

    frame1 = image
    video_writer.write(frame1)

    cnt = 1

    print("Starting video conversion, printing progress every {} frames...".format(print_every))
    while success:  
        success, image = video_capture.read()
        frame2 = image
        if frame2 is not None:
            frame1_tensor = transforms(PIL.Image.fromarray(frame1))
            frame2_tensor = transforms(PIL.Image.fromarray(frame2))

            with torch.no_grad():
                gen_frame, _, _, _, _ = model(frame1_tensor.unsqueeze(0).to(device), frame2_tensor.unsqueeze(0).to(device))
            gen_frame = gen_frame.squeeze(0).cpu().numpy().transpose((1, 2, 0))
            gen_frame = (gen_frame * 255).astype(np.uint8)

            if width_pad > 0:
                gen_frame = gen_frame[:,width_pad:,:]
            if height_pad > 0:
                gen_frame = gen_frame[height_pad:,:,:]

            frame1 = image

            video_writer.write(gen_frame)
            video_writer.write(frame2)

            cnt += 1

            if cnt % print_every == 0:
                print('{} / {} frames left.'.format(frame_count - cnt, frame_count))

    print("Done!")

    video_writer.release()
    video_capture.release()