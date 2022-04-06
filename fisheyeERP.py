import os
import numpy as np
import cv2
import torch
import torch.nn.functional as F


class ERP2Fisheye:
  def __init__(self, fish_h, fish_w, erp_h, erp_w, FoV, R):
    self.radius = fish_h // 2
    self.FoV = FoV // 2
    self.FovTh = self.FoV / 180 * np.pi
    fish_x, fish_y = np.meshgrid(range(fish_w), range(fish_h))
    fish_x = (fish_x.astype(np.float32) - (fish_w - 1) / 2)
    fish_y = (fish_y.astype(np.float32) - (fish_h - 1) / 2)
    fish_theta = np.sqrt(fish_x * fish_x + fish_y * fish_y) / self.radius * self.FoV  #theta deg
    fish_theta = fish_theta / 180 * np.pi
    fish_phi = np.arctan2(fish_y, fish_x)
    #fish_phi[fish_phi < -np.pi] += 2 * np.pi
    # fish_phi = np.expand_dims(fish_phi, 2)
    # fish_theta = np.expand_dims(fish_theta, 2)
    fish_phi_save = 255 * (fish_phi - np.min(fish_phi)) / (np.max(fish_phi) - np.min(fish_phi))
    cv2.imwrite('fish_phi.png', fish_phi_save.astype(np.uint8))
    fish_theta_save = 255 * (fish_theta - np.min(fish_theta)) / (np.max(fish_theta) - np.min(fish_theta))
    cv2.imwrite('fishe_theta.png', fish_theta_save.astype(np.uint8))

    self.invalidMask = (fish_theta > self.FovTh)
    fish_theta[self.invalidMask] = 0

    y = np.cos(fish_theta)
    x = np.sin(fish_theta) * np.cos(fish_phi)
    z = np.sin(fish_theta) * np.sin(fish_phi)
    coor3d = np.expand_dims(np.dstack((x, -z, y)), axis=-1)

    coor3d_r = np.matmul(R, coor3d)

    theta = np.arccos(coor3d_r[:, :, 1]).astype(np.float32)
    phi = np.arctan2(coor3d_r[:, :, 0], coor3d_r[:, :, 2]).astype(np.float32)

    phi_save = 255 * (phi - np.min(phi)) / (np.max(phi) - np.min(phi))
    cv2.imwrite('phi.png', phi_save.astype(np.uint8))
    theta_save = 255 * (theta - np.min(theta)) / (np.max(theta) - np.min(theta))
    cv2.imwrite('theta.png', theta_save.astype(np.uint8))

    theta = (theta - 0) / (np.pi - 0) * 2 - 1.0
    phi = (phi + np.pi) / (2 * np.pi) * 2 - 1.0
    self.grid = np.concatenate([phi, theta], axis=2)
    print(self.grid.shape)

  def trans(self, erp):
    c, h, w = erp.shape
    g = torch.from_numpy(self.grid).unsqueeze(0)
    erp = torch.from_numpy(erp).unsqueeze(0)
    fisheye = F.grid_sample(erp, g, mode='bilinear', align_corners=True)
    fisheye = fisheye.squeeze_(0).numpy()
    mask = np.repeat(np.expand_dims(self.invalidMask, 0), c, 0)
    fisheye[mask] = 0.0
    return fisheye


class Fisheye2ERP:
  def __init__(self, fish_h, fish_w, erp_h, erp_w, FoV, R):
    self.radius = fish_h // 2
    self.FoV = FoV // 2
    self.FovTh = self.FoV / 180 * np.pi
    erp_x, erp_y = np.meshgrid(range(erp_w), range(erp_h))
    phi = (erp_x).astype(np.float32) / (erp_w - 1) * 2 * np.pi - np.pi
    theta = (erp_y).astype(np.float32) / (erp_h - 1) * np.pi
    y = np.cos(theta)
    x = np.sin(theta) * np.sin(phi)
    z = np.sin(theta) * np.cos(phi)
    coor3d = np.expand_dims(np.dstack((x, y, z)), axis=-1)
    d = np.expand_dims(np.sqrt(x * x + y * y + z * z), -1)
    coor3d_r = np.matmul(R, coor3d)
    fish_theta = np.arccos(coor3d_r[:, :, 2] / d).astype(np.float32)
    fish_phi = np.arctan2(coor3d_r[:, :, 1], coor3d_r[:, :, 0]).astype(np.float32)
    self.invalidMask = (fish_theta > self.FovTh)

    cv2.imwrite('mask.png', self.invalidMask.astype(np.uint8) * 255)
    fish_phi_save = 255 * (fish_phi - np.min(fish_phi)) / (np.max(fish_phi) - np.min(fish_phi))
    cv2.imwrite('fish_phi.png', fish_phi_save.astype(np.uint8))
    fish_theta_save = 255 * (fish_theta - np.min(fish_theta)) / (np.max(fish_theta) - np.min(fish_theta))
    cv2.imwrite('fishe_theta.png', fish_theta_save.astype(np.uint8))
    print(np.max(fish_phi), np.min(fish_phi))

    fish_theta[self.invalidMask] = 0.0
    self.invalidMask = self.invalidMask.squeeze(-1)
    print(np.max(fish_theta), np.min(fish_theta))
    fish_r = fish_theta / (self.FoV / 180 * np.pi) * self.radius
    print(np.max(fish_r), np.min(fish_r))
    fish_x = fish_r * np.cos(fish_phi)
    fish_y = -fish_r * np.sin(fish_phi)
    print(np.max(fish_x), np.min(fish_x))
    print(np.max(fish_y), np.min(fish_y))

    fish_x = (fish_x - np.min(fish_x)) / (np.max(fish_x) - np.min(fish_x)) * 2 - 1.0
    fish_y = (fish_y - np.min(fish_y)) / (np.max(fish_y) - np.min(fish_y)) * 2 - 1.0

    self.grid = np.concatenate([fish_x, fish_y], axis=2)

  def trans(self, fish):
    c, h, w = fish.shape
    print(fish.shape)
    g = torch.from_numpy(self.grid).unsqueeze(0)
    fish = torch.from_numpy(fish).unsqueeze(0)
    erp = F.grid_sample(fish, g, mode='bilinear', align_corners=True)
    erp = erp.squeeze_(0).numpy()
    mask = np.repeat(np.expand_dims(self.invalidMask, 0), c, 0)
    erp[mask] = 0.0
    return erp


def get_rotate_matrix(x_a, y_a, z_a):  # 绕z,y,z轴旋转的角度（弧度制）
  Rx = np.array([[1, 0, 0], [0, np.cos(x_a), -np.sin(x_a)], [0, np.sin(x_a), np.cos(x_a)]])
  Rz = np.array([[np.cos(z_a), -np.sin(z_a), 0], [np.sin(z_a), np.cos(z_a), 0], [0, 0, 1]])
  Ry = np.array([[np.cos(y_a), 0, -np.sin(y_a)], [0, 1, 0], [np.sin(y_a), 0, np.cos(y_a)]])
  R = np.dot(np.dot(Rx, Rz), Ry)
  return R


if __name__ == '__main__':
  rotate = get_rotate_matrix(0, 0, 0)
  rotate_r = np.linalg.inv(rotate)
  FoV = 210
  e2f = ERP2Fisheye(1024, 1024, 512, 1024, FoV, rotate)
  f2e = Fisheye2ERP(1024, 1024, 512, 1024, FoV, rotate_r)
  erp = cv2.imread('../erp.png').transpose((2, 0, 1)).astype(np.float32)
  fisheye = e2f.trans(erp)
  erp_2 = f2e.trans(fisheye)
  fisheye = fisheye.transpose((1, 2, 0))
  fisheye = (fisheye - np.min(fisheye)) / (np.max(fisheye) - np.min(fisheye)) * 255
  cv2.imwrite('fish' + str(FoV) + '.png', fisheye.astype(np.uint8))

  erp_2 = erp_2.transpose((1, 2, 0))
  erp_2 = (erp_2 - np.min(erp_2)) / (np.max(erp_2) - np.min(erp_2)) * 255
  cv2.imwrite('erp_2_' + str(FoV) + '.png', erp_2.astype(np.uint8))
