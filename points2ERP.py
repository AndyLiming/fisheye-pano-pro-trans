import os
import numpy as np
import cv2
import torch
import torch.nn.functional as F


def points2ERP(points, erp_h, erp_w):
  # NOTE:
  # points: np.array (N*3) - (x,y,z)
  # x: right
  # y: down
  # z: front
  total_num = erp_h * erp_w
  out = torch.zeros((total_num, 1)).float()
  x = points[:, 0]
  y = points[:, 1]
  z = points[:, 2]
  rho = np.sqrt(x * x + y * y + z * z)
  theta = np.arcsin(y / rho)
  phi = np.clip(np.arctan2(x, z), -np.pi, np.pi)
  u = (theta + np.pi / 2) / np.pi * (erp_h - 1)
  v = (phi + np.pi) / (np.pi * 2) * (erp_w - 1)
  u1 = np.round(u)
  v1 = np.round(v)
  u1 = np.clip(u1, 0, total_num - 1)
  v1 = np.clip(v1, 0, total_num - 1)
  index = u1 * erp_w + v1
  index = index
  mask = (index >= total_num)

  rho = torch.from_numpy(rho).unsqueeze(-1).float()
  index = torch.from_numpy(index).unsqueeze(-1).long()
  print(index.shape, out.shape, rho.shape)
  out = out.scatter(dim=0, index=index, src=rho)
  out = out.view(erp_h, erp_w, 1)
  mask = (out > 0)
  return out.numpy().squeeze(), mask.numpy().squeeze()


def gen_fake_points(fname, rate=0.5):
  data = np.load(fname)['arr_0'].astype(np.float32)
  # data = np.expand_dims(data, axis=-1)
  h, w = data.shape
  erp_x, erp_y = np.meshgrid(range(w), range(h))
  phi = (erp_x).astype(np.float32) / (w - 1) * 2 * np.pi - np.pi
  #phi = ((erp_x).astype(np.float32) - (erp_w - 1) / 2) / (erp_w / 2) * np.pi
  theta = (erp_y).astype(np.float32) / (h - 1) * np.pi - np.pi / 2
  y = data * np.sin(theta)  # down
  x = data * np.cos(theta) * np.sin(phi)  #right
  z = data * np.cos(theta) * np.cos(phi)  #front
  data = np.dstack((x, y, z))

  data = np.reshape(data, (h * w, 3))

  #np.random.shuffle(data)

  # num = int(h * w)
  # data = data[0:num:10, :]
  return data


def gen_pseudo_map(data, cm_type, mask=None):
  data = cv2.convertScaleAbs(data, alpha=255 / 20, beta=1)
  data = np.clip(data, 0, 255)
  data = (255 - data).astype(np.uint8)
  data = cv2.applyColorMap(data, cm_type)
  if mask is not None:
    print(data.shape, mask.shape)
    data[~mask, :] = 0
  return data


if __name__ == '__main__':
  cm_type = cv2.COLORMAP_JET
  rgb = cv2.imread('./imgs/erp_rgb0_90.jpg')
  erp_h, erp_w = rgb.shape[:2]
  gt_save = np.load('./imgs/erp_depth0_90.npz')['arr_0'].astype(np.float32)
  # gt_save = gen_pseudo_map(gt_save, cm_type)
  # cv2.imwrite('./imgs/points_test_0_90_gt.png', gt_save)
  fake_points = gen_fake_points('./imgs/erp_depth0_90.npz', rate=0.05)
  out, mask = points2ERP(fake_points, erp_h, erp_w)
  print(np.sum(np.abs(out - gt_save)) / (erp_h * erp_w))
  out_save = gen_pseudo_map(out, cm_type, mask)
  cv2.imwrite('./imgs/points_test_0_90.png', out_save)
