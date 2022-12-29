import os
import cv2
import numpy as np
import torch
import torch.nn.functional as F


def pinhole2Cassini(img, fov_h, fov_w, ca_h, ca_w):
  # img: pinhole image, numpy array (h,w,c)
  # fov_h: fov veritcal (degree,0-180)
  # fov_w: fov horizontal (degree,0-180)
  # ca_h: cassini height (pixel)
  # ca_w: cassini width (pixel)
  h, w = img.shape[:-1]
  ca_x, ca_y = np.meshgrid(range(ca_w), range(ca_h))
  phi = np.pi / 2 - (ca_x).astype(np.float32) / (ca_w - 1) * np.pi
  theta = np.pi - (ca_y).astype(np.float32) / (ca_h - 1) * 2 * np.pi
  x = np.sin(phi)
  y = np.cos(phi) * np.sin(theta)
  z = np.cos(phi) * np.cos(theta)

  fov_h_max = (fov_h / 180 * np.pi) / 2
  fov_h_min = -(fov_h / 180 * np.pi) / 2
  fov_w_max = (fov_w / 180 * np.pi) / 2
  fov_w_min = -(fov_w / 180 * np.pi) / 2

  print(fov_w, fov_w_max, np.tan(fov_w_max))

  x_1 = (-x / z) / np.tan(fov_w_max)
  y_1 = (-y / z) / np.tan(fov_h_max)

  mask = (-1 <= x_1) & (x_1 <= 1) & (-1 <= y_1) & (y_1 <= 1) & (z > 0)
  grid = np.stack([x_1, y_1], axis=-1)
  grid = torch.from_numpy(grid).unsqueeze(0)

  source_img = torch.from_numpy(img.transpose((2, 0, 1))).unsqueeze(0)
  print(grid.shape, source_img.shape)

  output = torch.nn.functional.grid_sample(source_img, grid, align_corners=True)
  print(output.shape)

  output = output.squeeze(0).numpy().transpose((1, 2, 0))
  print(output.shape)
  output[~mask, :] = 0.0
  cv2.imwrite('pin2ca.png', output.astype(np.uint8))

  cv2.imwrite('test_mask.png', mask * 255)

  print(mask.shape)


class Cassini2pinhole:
  def __init__(self, ca_h, ca_w, p_h, p_w, focal):
    # p_h = np.round(2 * focal * np.tan(fov_h / 2 / 180 * np.pi)).astype(np.int32)
    # p_w = np.round(2 * focal * np.tan(fov_w / 2 / 180 * np.pi)).astype(np.int32)
    xc = (p_w - 1.0) / 2
    yc = (p_h - 1.0) / 2
    face_x, face_y = np.meshgrid(range(p_w), range(p_h))
    x = face_x - xc
    y = face_y - yc
    z = np.ones_like(x) * focal
    x, y, z = x / z, y / z, z / z
    phi = np.arctan2(x, np.sqrt(y * y + z * z)).astype(np.float32)
    theta = np.arctan2(y, z).astype(np.float32)
    phi = phi / (np.pi) * 2.0
    theta = theta / np.pi
    print(phi.shape, phi.max(), phi.min())
    print(theta.shape, theta.max(), theta.min())
    self.grid = np.stack([phi, theta], axis=-1)
    self.grid = torch.from_numpy(self.grid).unsqueeze(0).cuda()

  def trans(self, ca):
    c, h, w = ca.shape
    ca = torch.from_numpy(ca).unsqueeze(0).cuda()
    pinhole = F.grid_sample(ca, self.grid, mode='bilinear', align_corners=True)
    # pinhole = F.interpolate(pinhole, size=(out_h, out_w), mode='bicubic', align_corners=True)
    pinhole = pinhole.squeeze_(0).cpu().numpy()
    return pinhole


if __name__ == '__main__':
  # img = cv2.imread('fov90.png').astype(np.float32)
  fov_h, fov_w = 90, 90
  ca_h, ca_w = 1024, 512
  img = cv2.imread('./imgs/002430_13_rgb1.png').astype(np.float32).transpose((2, 0, 1))
  #pinhole2Cassini(img, fov_h, fov_w, ca_h, ca_w)
  ca_h, ca_w, p_h, p_w, focal = 1024, 512, 384, 512, 220
  fov_h = np.arctan2(p_h / 2, focal) / np.pi * 180 * 2
  fov_w = np.arctan2(p_w / 2, focal) / np.pi * 180 * 2
  print(fov_h, fov_w)
  c2p = Cassini2pinhole(ca_h, ca_w, p_h, p_w, focal)
  pinhole = c2p.trans(img)
  cv2.imwrite('./imgs/c2p.png', pinhole.transpose((1, 2, 0)).astype(np.uint8))
