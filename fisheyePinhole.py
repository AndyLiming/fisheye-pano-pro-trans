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
  def __init__(self, pinhole_h, pinhole_w, fish_h, fish_w, fish_FoV, Rot=np.identity(3, dtype=np.float32)):
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
    z = np.ones_like(x) * ((pinhole_w * np.sqrt(3)) / 2)
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

  def trans(self, fish):
    c, h, w = fish.shape
    fish = torch.from_numpy(fish).unsqueeze(0)
    grid = torch.from_numpy(self.grid_list[0]).unsqueeze(0)
    cube = F.grid_sample(fish, grid, mode='bilinear', align_corners=True)
    cube = cube.squeeze_(0).numpy()
    mask = np.repeat(np.expand_dims(self.invalidMaskList[0], 0), c, 0)
    cube[mask] = 0.0
    return cube


class Pinhole2Fisheye:
  def __init__(self, cube_face_h, cube_face_w, fish_h, fish_w, fish_FoV, Rot=np.identity(3, dtype=np.float32)):
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
    grid_front_w = -grid_front_raw[:, :, 0]
    grid_front_h = -grid_front_raw[:, :, 1]
    grid_front = np.concatenate([np.expand_dims(grid_front_w, 2), np.expand_dims(grid_front_h, 2)], 2)
    mask_front = ((grid_front_w <= 1) * (grid_front_w >= -1)) * ((grid_front_h <= 1) * (grid_front_h >= -1)) * (grid_front_raw[:, :, 2] > 0)
    masked_grid_front = grid_front * np.float32(np.expand_dims(mask_front, 2))
    self.masked_grid_list.append(masked_grid_front)
    self.mask_list.append(mask_front)

  def trans(self, pinhole):  #cube_faces:6 x c x h x w, 6 faces of the cube map. back - left - front - right - up - down
    c, h, w = pinhole.shape
    out = np.zeros([c, self.fish_h, self.fish_w])
    ori = torch.from_numpy(pinhole).unsqueeze(0)
    mask_grid = torch.from_numpy(self.masked_grid_list[0]).unsqueeze(0)
    fish = F.grid_sample(ori, mask_grid, mode='bilinear', align_corners=True)
    fish = fish.squeeze_(0).numpy()
    mask = ~(np.repeat(np.expand_dims(self.mask_list[0], 0), c, 0))
    fish[mask] = 0.0
    out = out + fish
    return out


if __name__ == '__main__':
  r = np.array([1, 0, 0]) * -np.pi / 6
  Rot, _ = cv2.Rodrigues(r)
  FoV = 190
  f2p = Fisheye2Pinhole(1080, 1920, 2560, 2560, FoV)
  #c2f = Cubemap2Fisheye(512, 512, 1024, 1024, FoV)
  fish = cv2.imread('./imgs/fe_rgb13_6.jpg').transpose((2, 0, 1)).astype(np.float32)
  cube = f2p.trans(fish).astype(np.float32)
  cubesave = cube.transpose((1, 2, 0)).astype(np.uint8)
  cubesave = cv2.rotate(cubesave, cv2.ROTATE_90_CLOCKWISE)
  cv2.imwrite('./imgs/pin_left.png', cubesave)
  tmp = cv2.imread('./imgs/ph_rgb8_6.jpg')
  tmp = cv2.rotate(tmp, cv2.ROTATE_90_CLOCKWISE)
  cv2.imwrite('./imgs/pin_right.png', tmp)
  # fish2 = c2f.trans(cube)
  # fish2 = fish2.transpose((1, 2, 0))
  # fish2 = (fish2 - np.min(fish2)) / (np.max(fish2) - np.min(fish2)) * 255
  # cv2.imwrite('./imgs/fish2_' + str(FoV) + '.png', fish2.astype(np.uint8))
