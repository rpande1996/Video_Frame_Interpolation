import PIL
from torchvision import transforms
import numpy as np
import torch
import cv2
import os

if __name__ == '__main__':
    folder_name = 'drone/'
    frames_path = f'../input/frames/{folder_name}'
    saved_model_path = '../models/weights.pt'
    t = 0.5
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = torch.load(saved_model_path, map_location=torch.device(device))
    model.eval()

    frames = sorted([i for i in os.listdir(frames_path) if os.path.isfile(os.path.join(frames_path, i))])

    first_path = os.path.join(frames_path, frames[0])
    last_path = os.path.join(frames_path, frames[1])
    first = PIL.Image.open(first_path)
    last = PIL.Image.open(last_path)

    width, height = first.size

    if width % 64 != 0:
        width_pad = int((np.floor(width / 64) + 1) * 64 - width)
    else:
        width_pad = 0
    if height % 64 != 0:
        height_pad = int((np.floor(height / 64) + 1) * 64 - height)
    else:
        height_pad = 0
    transforms1 = transforms.Compose([
        transforms.Pad((width_pad, height_pad, 0, 0)),
        transforms.ToTensor()
    ])
    first = transforms1(first)
    last = transforms1(last)

    save_path = f'../output/frames/{folder_name}' + 'generated/'
    os.makedirs(save_path, exist_ok=True)

    with torch.no_grad():
        img_recon, flow_t_0, flow_t_1, w1, w2 = model(first.unsqueeze(0).to(device), last.unsqueeze(0).to(device), t)

    img_recon = img_recon.squeeze(0).cpu().numpy().transpose((1, 2, 0)) * 255
    if width_pad > 0:
        img_recon = img_recon[:,width_pad:,:]
    if height_pad > 0:
        img_recon = img_recon[height_pad:,:,:]
    img_recon = img_recon.astype(np.uint8)
    PIL.Image.fromarray(img_recon).save("{}/im2.jpg".format(f'../output/frames/{folder_name}'))

    flow_t_0 = flow_t_0.squeeze(0).cpu().numpy().transpose((1, 2, 0))
    if width_pad > 0:
        flow_t_0 = flow_t_0[:,width_pad:,:]
    if height_pad > 0:
        flow_t_0 = flow_t_0[height_pad:,:,:]
    hsv_t_0 = np.zeros((flow_t_0.shape[0], flow_t_0.shape[1], 3), dtype=np.uint8)
    hsv_t_0[..., 1] = 255
    mag, ang = cv2.cartToPolar(flow_t_0[..., 0], flow_t_0[..., 1])
    hsv_t_0[..., 0] = ang * 180 / np.pi / 2
    hsv_t_0[..., 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)
    bgr_t_0 = cv2.cvtColor(hsv_t_0, cv2.COLOR_HSV2BGR)
    PIL.Image.fromarray(bgr_t_0).save("{}/flow_t_0_t={}.jpg".format(save_path, t))

    flow_t_1 = flow_t_1.squeeze(0).cpu().numpy().transpose((1, 2, 0))
    if width_pad > 0:
        flow_t_1 = flow_t_1[:,width_pad:,:]
    if height_pad > 0:
        flow_t_1 = flow_t_1[height_pad:,:,:]
    hsv_t_1 = np.zeros((flow_t_1.shape[0], flow_t_1.shape[1], 3), dtype=np.uint8)
    hsv_t_1[..., 1] = 255
    mag, ang = cv2.cartToPolar(flow_t_1[..., 0], flow_t_1[..., 1])
    hsv_t_1[..., 0] = ang * 180 / np.pi / 2
    hsv_t_1[..., 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)
    bgr_t_1 = cv2.cvtColor(hsv_t_1, cv2.COLOR_HSV2BGR)
    PIL.Image.fromarray(bgr_t_1).save("{}/flow_t_1_t={}.jpg".format(save_path, t))

    w1 = w1.squeeze().cpu().numpy()
    if width_pad > 0:
        w1 = w1[:,width_pad:]
    if height_pad > 0:
        w1 = w1[height_pad:,:]
    PIL.Image.fromarray((w1 * 255).astype(np.uint8), 'L').save("{}/weight_map_t_0_t={}.jpg".format(save_path, t))
    w2 = w2.squeeze().cpu().numpy()
    if width_pad > 0:
        w2 = w2[:,width_pad:]
    if height_pad > 0:
        w2 = w2[height_pad:,:]
    PIL.Image.fromarray((w2 * 255).astype(np.uint8), 'L').save("{}/weight_map_t_1_t={}.jpg".format(save_path, t))
