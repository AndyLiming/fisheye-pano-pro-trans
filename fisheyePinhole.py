import os
import numpy as np
import cv2
import torch
import torch.nn.functional as F
"""
for equi-distant projection fisheye images
"""
# transform fisheye images to cubemap projection
# Note： Fish-eye image
# fisheye : valid imaging area height=width=radius, equi-distant projection,

# Note： Cubemap image
# cube map: each face is a square. 6 faces are arranged into a row with the order:
# back - left - front - right - up - down
# image shape is [h x w] where h=face_h, w=6*face_w
# the width range of the back face is [0:face_w], left face is [face_w:2*face_w], and so on

# Note: coordinate
# x == left, y==up, z==front


class Fisheye2Pinhole:
  def __init__(self, pinhole_h, pinhole_w, fish_h, fish_w, fish_FoV, pinhole_hfov, Rot=np.identity(3, dtype=np.float32)):
    self.FoV = fish_FoV // 2
    self.radius = fish_h // 2
    R_list = []
    self.grid_list = []
    self.invalidMaskList = []
    self.fish_h = fish_h
    self.fish_w = fish_w
    self.cube_face_h = pinhole_h
    self.cube_face_w = pinhole_w

    xc = (pinhole_w - 1.0) / 2
    yc = (pinhole_h - 1.0) / 2
    face_x, face_y = np.meshgrid(range(pinhole_w), range(pinhole_h))
    x = xc - face_x
    y = yc - face_y
    z = np.ones_like(x) * ((pinhole_w / np.tan(pinhole_hfov / 180 * np.pi / 2)) / 2)
    x, y, z = x / z, y / z, z / z
    D = np.sqrt(x * x + y * y + z * z)
    coor3d = np.expand_dims(np.dstack([x, y, z]), -1)
    coor3d_r = np.matmul(Rot, coor3d)

    coor3d_rf = coor3d_r.squeeze()

    fish_theta = np.arccos(coor3d_rf[:, :, 2] / D)
    invalidMask = (fish_theta > (self.FoV / 180 * np.pi))
    fish_theta[invalidMask] = 0
    self.invalidMaskList.append(invalidMask)
    fish_phi = np.arctan2(-coor3d_rf[:, :, 1], coor3d_rf[:, :, 0])

    fish_r = fish_theta / (self.FoV / 180 * np.pi) * self.radius

    fish_x = -fish_r * np.cos(fish_phi)
    fish_y = fish_r * np.sin(fish_phi)

    u = (fish_x) / (fish_w - 1) * 2.0
    v = (fish_y) / (fish_h - 1) * 2.0
    print(coor3d_r.shape, u.shape, v.shape)

    self.grid_list.append(np.dstack([u, v]).astype(np.float32))
    print(self.grid_list[0].shape)

  def trans(self, fish, mode='bilinear'):
    c, h, w = fish.shape
    fish = torch.from_numpy(fish).unsqueeze(0)
    grid = torch.from_numpy(self.grid_list[0]).unsqueeze(0)
    cube = F.grid_sample(fish, grid, mode, align_corners=True)
    cube = cube.squeeze_(0).numpy()
    mask = np.repeat(np.expand_dims(self.invalidMaskList[0], 0), c, 0)
    cube[mask] = 0.0
    return cube


class Pinhole2Fisheye:
  def __init__(self, cube_face_h, cube_face_w, fish_h, fish_w, fish_FoV, pinhole_hfov, Rot=np.identity(3, dtype=np.float32)):
    self.radius = (fish_h) // 2
    self.FoV = fish_FoV // 2
    self.FovTh = self.FoV / 180 * np.pi
    self.fish_h, self.fish_w = fish_h, fish_w
    fish_x, fish_y = np.meshgrid(range(fish_w), range(fish_h))
    fish_x = (fish_x.astype(np.float32) - (fish_w - 1) / 2)
    #fish_y = (fish_y.astype(np.float32) - (fish_h - 1) / 2)
    #fish_x = (fish_x.astype(np.float32)) * fish_w / (fish_w - 1) - self.radius
    fish_y = (fish_y.astype(np.float32)) * fish_h / (fish_h - 1) - self.radius
    fish_theta = np.sqrt(fish_x * fish_x + fish_y * fish_y) / self.radius * self.FoV  #theta deg
    fish_theta = fish_theta / 180 * np.pi
    fish_phi = np.arctan2(-fish_y, -fish_x)

    self.invalidMask = (fish_theta > self.FovTh)

    self.h_th_norm = np.tan((pinhole_hfov / 180 * np.pi) / 2)
    self.v_th_norm = self.h_th_norm * cube_face_h / cube_face_w
    #fish_theta[self.invalidMask] = 0

    z = np.cos(fish_theta)
    x = np.sin(fish_theta) * np.cos(fish_phi)
    y = np.sin(fish_theta) * np.sin(fish_phi)
    coor3d = np.expand_dims(np.dstack((x, y, z)), axis=-1)
    coor3d_r = np.matmul(Rot, coor3d).squeeze(-1)
    x_3d = coor3d_r[:, :, 0:1]
    y_3d = coor3d_r[:, :, 1:2]
    z_3d = coor3d_r[:, :, 2:]
    self.masked_grid_list = []
    self.mask_list = []

    # Compute the front grid
    grid_front_raw = coor3d_r / np.abs(z_3d)
    grid_front_w = -grid_front_raw[:, :, 0] / self.h_th_norm
    grid_front_h = -grid_front_raw[:, :, 1] / self.v_th_norm
    grid_front = np.concatenate([np.expand_dims(grid_front_w, 2), np.expand_dims(grid_front_h, 2)], 2)
    mask_front = ((grid_front_w <= 1) * (grid_front_w >= -1)) * ((grid_front_h <= 1) * (grid_front_h >= -1)) * (grid_front_raw[:, :, 2] > 0)
    masked_grid_front = grid_front * np.float32(np.expand_dims(mask_front, 2))
    self.masked_grid_list.append(masked_grid_front)
    self.mask_list.append(mask_front)

  def get_fish_mask(self):
    return self.invalidMask

  def get_pinhole_mask(self):
    return self.mask_list[0]

  def trans(self, pinhole, mode='bilinear'):  #cube_faces:6 x c x h x w, 6 faces of the cube map. back - left - front - right - up - down
    c, h, w = pinhole.shape
    out = np.zeros([c, self.fish_h, self.fish_w])
    ori = torch.from_numpy(pinhole).unsqueeze(0)
    mask_grid = torch.from_numpy(self.masked_grid_list[0]).unsqueeze(0)
    fish = F.grid_sample(ori, mask_grid, mode, align_corners=True)
    fish = fish.squeeze_(0).numpy()
    mask = ~(np.repeat(np.expand_dims(self.mask_list[0], 0), c, 0))
    fish[mask] = 0.0
    out = out + fish
    return out


class KB4fisheye2Pinhole:
  def __init__(self, pinhole_h, pinhole_w, fish_h, fish_w, fish_FoV, pinhole_hfov, fx, fy, cx, cy, kb_theta_coes, Rot=np.identity(3, dtype=np.float32)):
    self.FoV = fish_FoV // 2
    self.radius = fish_h // 2
    R_list = []
    self.fish_h = fish_h
    self.fish_w = fish_w
    self.cube_face_h = pinhole_h
    self.cube_face_w = pinhole_w

    # KB model paras
    self.fx = fx
    self.fy = fy
    self.cx = cx
    self.cy = cy
    self.kb_theta_coes = kb_theta_coes
    # assume the camera model is KB4: d_theta = theta+k1*theta^3+k2*theta^5+k3*theta^7+k4*theta^9
    assert (len(self.kb_theta_coes) == 4)

    xc = (pinhole_w - 1.0) / 2
    yc = (pinhole_h - 1.0) / 2
    face_x, face_y = np.meshgrid(range(pinhole_w), range(pinhole_h))
    x = xc - face_x
    y = yc - face_y
    z = np.ones_like(x) * ((pinhole_w / np.tan(pinhole_hfov / 180 * np.pi / 2)) / 2)
    x, y, z = x / z, y / z, z / z
    D = np.expand_dims(np.sqrt(x * x + y * y + z * z), -1)
    coor3d = np.expand_dims(np.dstack([z, x, y]), -1)
    coor3d_r = np.matmul(Rot, coor3d)

    fish_theta = np.arccos(coor3d_r[:, :, 0] / D)
    fish_theta_sq = fish_theta * fish_theta
    fish_dtheta = fish_theta + self.kb_theta_coes[0] * fish_theta * fish_theta_sq + self.kb_theta_coes[1] * fish_theta * fish_theta_sq * fish_theta_sq + self.kb_theta_coes[
        2] * fish_theta * fish_theta_sq * fish_theta_sq * fish_theta_sq + self.kb_theta_coes[3] * fish_theta * fish_theta_sq * fish_theta_sq * fish_theta_sq * fish_theta_sq
    fish_phi = np.clip(np.arctan2(-coor3d_r[:, :, 2], -coor3d_r[:, :, 1]), -np.pi, np.pi)

    fish_x = self.fx * fish_dtheta * np.cos(fish_phi) + self.cx
    fish_y = self.fy * fish_dtheta * np.sin(fish_phi) + self.cy
    self.invalidMask = (fish_x < 0) | (fish_x > fish_w - 1) | (fish_y < 0) | (fish_y > fish_h - 1)
    self.invalidMask = self.invalidMask.squeeze(-1)

    fish_x = np.clip((fish_x / (fish_w - 1)) * 2 - 1.0, -1, 1).astype(np.float32)
    fish_y = np.clip((fish_y / (fish_h - 1)) * 2 - 1.0, -1, 1).astype(np.float32)

    self.grid = np.concatenate([fish_x, fish_y], axis=2)

  def trans(self, fish, mode='bilinear'):
    c, h, w = fish.shape
    fish = torch.from_numpy(fish).unsqueeze(0)
    grid = torch.from_numpy(self.grid).unsqueeze(0)
    cube = F.grid_sample(fish, grid, mode, align_corners=True)
    cube = cube.squeeze_(0).numpy()
    mask = np.repeat(np.expand_dims(self.invalidMask, 0), c, 0)
    cube[mask] = 0.0
    return cube


class Pinhole2Pinhole:
  def __init__(self, pinhole_h, pinhole_w, input_h, input_w, input_hfov, pinhole_hfov, fx, fy, cx, cy, pinhole_coes, Rot=np.identity(3, dtype=np.float32)):
    self.fx = fx
    self.fy = fy
    self.cx = cx
    self.cy = cy
    self.pinhole_coes = pinhole_coes
    assert (len(self.pinhole_coes) == 5)
    k1, k2, k3, p1, p2 = pinhole_coes
    xc = (pinhole_w - 1.0) / 2
    yc = (pinhole_h - 1.0) / 2
    face_x, face_y = np.meshgrid(range(pinhole_w), range(pinhole_h))
    x = xc - face_x
    y = yc - face_y
    z = np.ones_like(x) * ((pinhole_w / np.tan(pinhole_hfov / 180 * np.pi / 2)) / 2)
    x, y, z = x / z, y / z, z / z
    D = np.expand_dims(np.sqrt(x * x + y * y + z * z), -1)
    coor3d = np.expand_dims(np.dstack([z, x, y]), -1)
    coor3d_r = np.matmul(Rot, coor3d)  # x-front,y-left,z-up
    u = fx * (-coor3d_r[:, :, 1]) / coor3d_r[:, :, 0] + cx
    v = fy * (-coor3d_r[:, :, 2]) / coor3d_r[:, :, 0] + cy
    self.invalidMask = (u < 0) | (u > input_w - 1) | (v < 0) | (v > input_h - 1)
    self.invalidMask = self.invalidMask.squeeze(-1)

    u = np.clip((u / (input_w - 1)) * 2 - 1.0, -1, 1).astype(np.float32)
    v = np.clip((v / (input_h - 1)) * 2 - 1.0, -1, 1).astype(np.float32)

    self.grid = np.concatenate([u, v], axis=2)

  def trans(self, in_p, mode='bilinear'):
    c, h, w = in_p.shape
    in_p = torch.from_numpy(in_p).unsqueeze(0)
    grid = torch.from_numpy(self.grid).unsqueeze(0)
    out_p = F.grid_sample(in_p, grid, mode, align_corners=True)
    out_p = out_p.squeeze_(0).numpy()
    mask = np.repeat(np.expand_dims(self.invalidMask, 0), c, 0)
    out_p[mask] = 0.0
    return out_p


if __name__ == '__main__':
  r = np.array([0, -1, 0]) * np.pi / 30 * 7
  Rot, _ = cv2.Rodrigues(r)
  r2 = np.array([0, 0, -1]) * np.pi / 6
  Rot2, _ = cv2.Rodrigues(r2)
  Rot = np.matmul(Rot, Rot2)

  fish_FoV = 190
  pinhole_hfov = 100
  pinhole_h, pinhole_w = 1080, 1920
  fish_h, fish_w = 2560, 2560
  fx, fy, cx, cy = 771.9852, 771.9852, 1279.5, 1279.5
  kb_theta_coes = [0, 0, 0, 0]

  f2p = KB4fisheye2Pinhole(pinhole_h, pinhole_w, fish_h, fish_w, fish_FoV, pinhole_hfov, fx, fy, cx, cy, kb_theta_coes, Rot)
  fish = cv2.imread('./imgs/fe_rgb10_6.jpg').transpose((2, 0, 1)).astype(np.float32)
  pinhole = f2p.trans(fish)
  pinhole_save = pinhole.transpose((1, 2, 0)).astype(np.uint8)
  cv2.imwrite('./imgs/kb4f2p_10_6-test.png', pinhole_save)

  input_h, input_w = 1080, 1920
  input_hfov = 100
  pinhole_coes = np.array([0, 0, 0, 0, 0]).astype(np.float64)
  pfx, pfy, pcx, pcy = 805.5356459, 805.5356459, 959.5, 539.5
  K = np.array([[pfx, 0, pcx], [0, pfy, pcy], [0, 0, 1]]).astype(np.float64)
  print(K)
  r_rel = np.array([0, 0, 0]) * np.pi / 6
  R_rel, _ = cv2.Rodrigues(r_rel)
  T_rel = np.array([0.273205, -0.1, -0.073205]).astype(np.float64)
  #T_rel = np.array([0.073205, 0.1, -0.273205]).astype(np.float64)
  #T_rel = np.array([0.2, -0.1, 0.2]).astype(np.float64)
  rect = cv2.stereoRectify(cameraMatrix1=K, distCoeffs1=pinhole_coes, cameraMatrix2=K, distCoeffs2=pinhole_coes, imageSize=(pinhole_w, pinhole_h), R=R_rel, T=T_rel, alpha=-1)
  R1, R2, P1, P2 = rect[:4]
  print(R1, R2)
  R0 = np.array([[0, -1, 0], [0, 0, -1], [1, 0, 0]])
  # R1 = np.matmul(np.matmul(R0.T, R1), R0)
  # #R1 = np.linalg.inv(R1)
  # R2 = np.matmul(np.matmul(R0.T, R2), R0)
  #R2 = np.linalg.inv(R2)
  # pfx2, pfy2, pcx2, pcy2 = P1[0, 0], P1[1, 1], P1[0, 2], P1[1, 2]
  # print(pfx2, pfy2, pcx2, pcy2)
  in_p = cv2.imread('./imgs/ph_rgb5_6.jpg')
  in_p2 = cv2.imread('./imgs/ph_rgb4_6.jpg')
  (map1, map2) = cv2.initUndistortRectifyMap(K, pinhole_coes, R1, P1, size=(pinhole_w, pinhole_h), m1type=cv2.CV_32FC1)
  out_p2 = cv2.remap(pinhole_save, map1, map2, cv2.INTER_CUBIC)
  (map1, map2) = cv2.initUndistortRectifyMap(K, pinhole_coes, R2, P2, size=(pinhole_w, pinhole_h), m1type=cv2.CV_32FC1)
  out_p = cv2.remap(in_p, map1, map2, cv2.INTER_CUBIC)
  # p2p = Pinhole2Pinhole(pinhole_h, pinhole_w, input_h, input_w, input_hfov, pinhole_hfov, pfx, pfy, pcx, pcy, pinhole_coes, R1)
  # in_p = cv2.imread('./imgs/ph_rgb5_6.jpg').transpose((2, 0, 1)).astype(np.float32)
  # out_p = p2p.trans(in_p)
  # out_p = out_p.transpose((1, 2, 0)).astype(np.uint8)
  cv2.imwrite('./imgs/p2p_5_6-test.png', out_p)

  # p2p2 = Pinhole2Pinhole(pinhole_h, pinhole_w, input_h, input_w, input_hfov, pinhole_hfov, pfx, pfy, pcx, pcy, pinhole_coes, R2)

  # out_p2 = p2p2.trans(pinhole)
  # out_p2 = out_p2.transpose((1, 2, 0)).astype(np.uint8)
  cv2.imwrite('./imgs/p2p_10_6-test2.png', out_p2)
  # f2p = Fisheye2Pinhole(1080, 1920, 2560, 2560, FoV)
  # p2f = Pinhole2Fisheye(1080, 1920, 2560, 2560, FoV, 60)
  # fish = cv2.imread('./imgs/fe_rgb13_6.jpg').transpose((2, 0, 1)).astype(np.float32)
  # cube = f2p.trans(fish).astype(np.float32)
  # cubesave = cube.transpose((1, 2, 0)).astype(np.uint8)
  # cubesave = cv2.rotate(cubesave, cv2.ROTATE_90_CLOCKWISE)
  # cv2.imwrite('./imgs/pin_left.png', cubesave)
  # tmp = cv2.imread('./imgs/ph_rgb8_6.jpg')
  # tmp = cv2.rotate(tmp, cv2.ROTATE_90_CLOCKWISE)
  # cv2.imwrite('./imgs/pin_right.png', tmp)
  # fish_back = p2f.trans(cube)
  # fishsave = fish_back.transpose((1, 2, 0)).astype(np.uint8)
  # cv2.imwrite('./imgs/fish_left.png', fishsave)
  # fish2 = c2f.trans(cube)
  # fish2 = fish2.transpose((1, 2, 0))
  # fish2 = (fish2 - np.min(fish2)) / (np.max(fish2) - np.min(fish2)) * 255
  # cv2.imwrite('./imgs/fish2_' + str(FoV) + '.png', fish2.astype(np.uint8))

  # fish_mask = ~(p2f.get_fish_mask())
  # pinhole_mask = p2f.get_pinhole_mask()
  # print(fish_mask.shape, pinhole_mask.shape, fish_mask.dtype, pinhole_mask.dtype)
  # cv2.imwrite('./imgs/fish_mask.png', fish_mask * 255)
  # cv2.imwrite('./imgs/pinhole_mask.png', pinhole_mask * 255)
