import os
import numpy as np
import cv2
import torch
import torch.nn.functional as F


class ERP2Fisheye:

  def __init__(self, fish_h, fish_w, erp_h, erp_w, FoV, R=np.identity(3, dtype=np.float32)):
    """
    # fish_h,fish_w: shape of output fish eye images
    # erp_h, erp_w: shape of input erp images
    # FoV: FoV (filed of view) of the fish eye camera
    # R: rotate matrix between fisheye camera and the erp sphere coords. default is identity matrix(no rotate), fisheye lens faces Z axis (front)
    """
    self.radius = fish_h // 2
    self.FoV = FoV // 2
    self.FovTh = self.FoV / 180 * np.pi
    fish_x, fish_y = np.meshgrid(range(fish_w), range(fish_h))
    fish_x = (fish_x.astype(np.float32) - (fish_w - 1) / 2)
    fish_y = (fish_y.astype(np.float32) - (fish_h - 1) / 2)
    fish_theta = np.sqrt(fish_x * fish_x + fish_y * fish_y) / self.radius * self.FoV  #theta deg
    fish_theta = fish_theta / 180 * np.pi
    fish_phi = np.arctan2(fish_y, fish_x)
    #DEBUG
    #fish_phi[fish_phi < -np.pi] += 2 * np.pi
    # fish_phi = np.expand_dims(fish_phi, 2)
    # fish_theta = np.expand_dims(fish_theta, 2)
    # fish_phi_save = 255 * (fish_phi - np.min(fish_phi)) / (np.max(fish_phi) - np.min(fish_phi))
    # cv2.imwrite('fish_phi.png', fish_phi_save.astype(np.uint8))
    # fish_theta_save = 255 * (fish_theta - np.min(fish_theta)) / (np.max(fish_theta) - np.min(fish_theta))
    # cv2.imwrite('fishe_theta.png', fish_theta_save.astype(np.uint8))

    self.invalidMask = (fish_theta > self.FovTh)
    fish_theta[self.invalidMask] = 0

    y = np.cos(fish_theta)
    x = np.sin(fish_theta) * np.cos(fish_phi)
    z = np.sin(fish_theta) * np.sin(fish_phi)
    coor3d = np.expand_dims(np.dstack((x, -z, y)), axis=-1)

    coor3d_r = np.matmul(R, coor3d)

    theta = np.arccos(coor3d_r[:, :, 1]).astype(np.float32)
    phi = np.arctan2(coor3d_r[:, :, 0], coor3d_r[:, :, 2]).astype(np.float32)

    # phi_save = 255 * (phi - np.min(phi)) / (np.max(phi) - np.min(phi))
    # cv2.imwrite('phi.png', phi_save.astype(np.uint8))
    # theta_save = 255 * (theta - np.min(theta)) / (np.max(theta) - np.min(theta))
    # cv2.imwrite('theta.png', theta_save.astype(np.uint8))

    theta = (theta - 0) / (np.pi - 0) * 2 - 1.0
    phi = (phi + np.pi) / (2 * np.pi) * 2 - 1.0
    self.grid = np.concatenate([phi, theta], axis=2)

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

  def __init__(self, fish_h, fish_w, erp_h, erp_w, FoV, R=np.identity(3, dtype=np.float32)):
    self.radius = fish_h // 2
    self.FoV = FoV // 2
    self.FovTh = self.FoV / 180 * np.pi
    erp_x, erp_y = np.meshgrid(range(erp_w), range(erp_h))
    phi = (erp_x).astype(np.float32) / (erp_w - 1) * 2 * np.pi - np.pi
    #phi = ((erp_x).astype(np.float32) - (erp_w - 1) / 2) / (erp_w / 2) * np.pi
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

    fish_theta[self.invalidMask] = 0.0
    self.invalidMask = self.invalidMask.squeeze(-1)

    fish_r = fish_theta / (self.FoV / 180 * np.pi) * self.radius

    fish_x = fish_r * np.cos(fish_phi)
    fish_y = -fish_r * np.sin(fish_phi)

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

  def get_fish_mask(self):
    return self.invalidMask


class KB4fisheye2ERP:

  def __init__(self, fish_h, fish_w, erp_h, erp_w, fish_FoV, fx, fy, cx, cy, kb_theta_coes, R=np.identity(3, dtype=np.float32)):
    self.FoV = fish_FoV // 2
    self.FovTh = self.FoV / 180 * np.pi
    # KB model paras
    self.fx = fx
    self.fy = fy
    self.cx = cx
    self.cy = cy
    self.kb_theta_coes = kb_theta_coes
    # assume the camera model is KB4: d_theta = theta+k1*theta^3+k2*theta^5+k3*theta^7+k4*theta^9
    assert (len(self.kb_theta_coes) == 4)
    erp_x, erp_y = np.meshgrid(range(erp_w), range(erp_h))
    phi = (erp_x).astype(np.float32) / (erp_w - 1) * 2 * np.pi - np.pi
    #phi = ((erp_x).astype(np.float32) - (erp_w - 1) / 2) / (erp_w / 2) * np.pi
    theta = (erp_y).astype(np.float32) / (erp_h - 1) * np.pi - np.pi / 2
    y = np.sin(theta)  # down
    x = np.cos(theta) * np.sin(phi)  #right
    z = np.cos(theta) * np.cos(phi)  #front
    coor3d = np.expand_dims(np.dstack((x, y, z)), axis=-1)
    coor3d_r = np.matmul(R, coor3d)  # rotate and transform used here
    print(coor3d.shape, coor3d_r.shape)
    fisheye_r = np.sqrt(coor3d_r[:, :, 0] * coor3d_r[:, :, 0] + coor3d_r[:, :, 1] * coor3d_r[:, :, 1])
    fish_theta = np.clip(np.arctan2(fisheye_r, coor3d_r[:, :, 2]), 0, np.pi)
    print(fisheye_r.shape, fish_theta.shape)
    fish_theta_sq = fish_theta * fish_theta
    fish_dtheta = fish_theta + self.kb_theta_coes[0] * fish_theta * fish_theta_sq + self.kb_theta_coes[
        1] * fish_theta * fish_theta_sq * fish_theta_sq + self.kb_theta_coes[
            2] * fish_theta * fish_theta_sq * fish_theta_sq * fish_theta_sq + self.kb_theta_coes[
                3] * fish_theta * fish_theta_sq * fish_theta_sq * fish_theta_sq * fish_theta_sq
    # we define y==up and x==left, while in Image Coordinate y==down and x==right
    fish_phi = np.clip(np.arctan2(coor3d_r[:, :, 1], coor3d_r[:, :, 0]), -np.pi, np.pi).astype(np.float32)
    # self.invalidMask = (fish_theta > self.FovTh)
    # fish_theta[self.invalidMask] = 0.0
    fish_x = self.fx * fish_dtheta * np.cos(fish_phi) + self.cx
    fish_y = self.fy * fish_dtheta * np.sin(fish_phi) + self.cy
    self.invalidMask = (fish_x < 0) | (fish_x > fish_w - 1) | (fish_y < 0) | (fish_y > fish_h - 1)

    print(fish_x.min(), fish_x.max(), fish_y.min(), fish_y.max())
    self.invalidMask = self.invalidMask.squeeze(-1)

    fish_x = np.clip((fish_x / (fish_w - 1)) * 2 - 1.0, -1, 1).astype(np.float32)
    fish_y = np.clip((fish_y / (fish_h - 1)) * 2 - 1.0, -1, 1).astype(np.float32)

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

  def get_fish_mask(self):
    return self.invalidMask


def get_rotate_matrix(x_a, y_a, z_a):  # 绕z,y,z轴旋转的角度（弧度制）z轴朝前，y轴朝上，x轴朝左
  Rx = np.array([[1, 0, 0], [0, np.cos(x_a), -np.sin(x_a)], [0, np.sin(x_a), np.cos(x_a)]])
  Rz = np.array([[np.cos(z_a), -np.sin(z_a), 0], [np.sin(z_a), np.cos(z_a), 0], [0, 0, 1]])
  Ry = np.array([[np.cos(y_a), 0, -np.sin(y_a)], [0, 1, 0], [np.sin(y_a), 0, np.cos(y_a)]])
  R = np.dot(np.dot(Rx, Rz), Ry)
  return R


def get_rotate_matrix(x_a, y_a, z_a):  # 绕z,y,z轴旋转的角度（弧度制）z轴朝前，y轴朝上，x轴朝左
  Rx = np.array([[1, 0, 0], [0, np.cos(x_a), -np.sin(x_a)], [0, np.sin(x_a), np.cos(x_a)]])
  Rz = np.array([[np.cos(z_a), -np.sin(z_a), 0], [np.sin(z_a), np.cos(z_a), 0], [0, 0, 1]])
  Ry = np.array([[np.cos(y_a), 0, -np.sin(y_a)], [0, 1, 0], [np.sin(y_a), 0, np.cos(y_a)]])
  R = np.dot(np.dot(Rx, Rz), Ry)
  return R


def save_pseudo_map(data, cm_type, name, max_depth, min_depth=0, log=False, mask=None):
  if mask is None:
    mask = (data > min_depth) & (data < max_depth)
  data[data > max_depth] = 0
  data[data < min_depth] = 0
  if log:
    data = np.log(data - min_depth + 1)
    data = 255 * data / np.log(max_depth + 1)
  else:
    data = (data - data.min()) / (max_depth) * 255
  data = np.clip(data, 0, 255)
  data = (255 - data).astype(np.uint8)
  data = cv2.applyColorMap(data, cm_type)
  data[~mask, :] = 0
  cv2.imwrite(name, data)


if __name__ == '__main__':
  # rotate = get_rotate_matrix(0, np.pi, 0)
  # rotate_r = np.linalg.inv(rotate)
  # FoV = 190
  # e2f = ERP2Fisheye(1024, 1024, 1024, 2048, FoV, rotate)
  # f2e = Fisheye2ERP(1024, 1024, 1024, 2048, FoV, rotate_r)
  # erp = cv2.imread('./imgs/erp_parking.jpg').transpose((2, 0, 1)).astype(np.float32)
  # fisheye = e2f.trans(erp)
  # #erp_2 = f2e.trans(fisheye)
  # fisheye = fisheye.transpose((1, 2, 0))
  # fisheye = (fisheye - np.min(fisheye)) / (np.max(fisheye) - np.min(fisheye)) * 255
  # cv2.imwrite('./imgs/fish_parking-b30' + str(FoV) + '.jpg', fisheye.astype(np.uint8))

  # NOTE: trans pred fish to erp
  # rv = np.array([-1, 0, 0]) * np.pi / 6
  # R, _ = cv2.Rodrigues(rv)
  # FoV = 190
  # max_depth = 20
  # cm_type = cv2.COLORMAP_JET

  # f2ergb = Fisheye2ERP(2560, 2560, 320, 640, FoV, R)
  # fishrgb = cv2.imread('../../datasets/multi_view_huawei_data/huawei_SimpleParking/huawei_parking15/fisheye190/fe_rgb11_205.jpg').transpose((2, 0, 1)).astype(np.float32)
  # fisheye = np.load('../hetero-cameras-depth/outputs/test/newcrf_parking_super-20/huawei_parking15_fe_rgb11_205_pred.npy').astype(np.float32)
  # erp_rgb = f2ergb.trans(fishrgb).transpose((1, 2, 0))
  # #erp_rgb = np.roll(erp_rgb, 320, axis=1)
  # erp_rgb = erp_rgb[80:240, ::]
  # cv2.imwrite('./imgs/parking15_205_mono_rgb.png', erp_rgb.astype(np.uint8))

  # erp_gt11 = np.load('../../datasets/multi_view_huawei_data/huawei_SimpleParking/huawei_parking15/erp/erp_depth11_205.npz')['arr_0'].astype(np.float32)
  # erp_gt11 = cv2.resize(erp_gt11, [640, 320])
  # #erp_gt11 = np.roll(erp_gt11, 320, axis=1)
  # erp_gt11 = erp_gt11[80:240, ::]
  # mask = (erp_gt11 > 0) & (erp_gt11 < max_depth)
  # save_pseudo_map(erp_gt11, cm_type, './imgs/parking15_205_gt11.png', max_depth, min_depth=0)

  # f2e = Fisheye2ERP(1280, 1280, 320, 640, FoV, R)
  # fisheye = np.expand_dims(fisheye, axis=0)
  # erp = f2e.trans(fisheye)
  # erp = erp.squeeze()
  # #erp = np.roll(erp, 320, axis=1)
  # erp = erp[80:240, ::]
  # mask = mask & (erp > 0)
  # save_pseudo_map(erp, cm_type, './imgs/parking15_205_mono_pred.png', max_depth, mask=mask)

  # erp_gt = np.load('../../datasets/multi_view_huawei_data/huawei_SimpleParking/huawei_parking15/erp/erp_depth0_205.npz')['arr_0'].astype(np.float32)
  # erp_gt = cv2.resize(erp_gt, [640, 320])
  # erp_gt = np.roll(erp_gt, 320, axis=1)
  # erp_gt = erp_gt[80:240, ::]
  # mask = (erp_gt > 3) & (erp_gt < max_depth)
  # save_pseudo_map(erp_gt, cm_type, './imgs/parking15_205_gt.png', max_depth, min_depth=3)

  # erp_mv = np.load('./results/parking15_205.npy').astype(np.float32)
  # print(erp_mv.max(), erp_mv.min(), erp_mv.dtype, erp_mv.shape)
  # erp_mv = np.roll(erp_mv, 320, axis=1)
  # save_pseudo_map(erp_mv, cm_type, './imgs/parking15_205_multi_pred.png', max_depth, min_depth=3, mask=mask)

  #------------------------------------------------------------------------------------------------------------#
  rotate = get_rotate_matrix(-np.pi / 12, np.pi, 0)
  rotate_r = np.linalg.inv(rotate)
  FoV = 190
  e2f = ERP2Fisheye(2560, 2560, 2560, 5120, FoV, rotate)
  erp_rgb = cv2.imread(
      '/home/custom_users/Datasets/multi_view_huawei_data/huawei_SimpleParking/huawei_parking15_cmp_ori/erp/erp_rgb0_15.jpg')
  print(erp_rgb.shape)
  erp_rgb = erp_rgb.transpose((2, 0, 1)).astype(np.float32)
  fisheye = e2f.trans(erp_rgb)
  # #erp_2 = f2e.trans(fisheye)
  fisheye = fisheye.transpose((1, 2, 0))
  fisheye = (fisheye - np.min(fisheye)) / (np.max(fisheye) - np.min(fisheye)) * 255
  cv2.imwrite('./imgs/fish_parking15_cmp_ori_15_' + str(FoV) + '_2.jpg', fisheye.astype(np.uint8))

  # erp_2 = erp_2.transpose((1, 2, 0))
  # erp_2 = (erp_2 - np.min(erp_2)) / (np.max(erp_2) - np.min(erp_2)) * 255
  # cv2.imwrite('./imgs/erp_2_' + str(FoV) + '.png', erp_2.astype(np.uint8))

  # fish_mask = ~(f2e.get_fish_mask())
  # cv2.imwrite('./imgs/fish_mask_erp.png', fish_mask * 255)