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


class Fisheye2Cubemap:
  def __init__(self, cube_face_h, cube_face_w, fish_h, fish_w, fish_FoV, Rot=np.identity(3, dtype=np.float32)):
    self.FoV = fish_FoV // 2
    self.radius = fish_h // 2
    R_list = []
    self.grid_list = []
    self.invalidMaskList = []
    phi = np.array([1, 1 / 2, 0, -1 / 2]) * np.pi  # back,left,front,right
    theta = np.array([-1 / 2, 1 / 2]) * np.pi  #up, down
    self.fish_h = fish_h
    self.fish_w = fish_w
    self.cube_face_h = cube_face_h
    self.cube_face_w = cube_face_w
    for ph in phi:
      r = ph * np.array([0, 1, 0])
      R = cv2.Rodrigues(r)[0]
      R_list.append(R)
    for th in theta:
      r = th * np.array([1, 0, 0])
      R = cv2.Rodrigues(r)[0]
      R_list.append(R)

    xc = (cube_face_w - 1.0) / 2
    yc = (cube_face_h - 1.0) / 2
    face_x, face_y = np.meshgrid(range(cube_face_w), range(cube_face_h))
    x = xc - face_x
    y = yc - face_y
    z = np.ones_like(x) * ((cube_face_h - 1.0) / 2)
    x, y, z = x / z, y / z, z / z
    D = np.sqrt(x * x + y * y + z * z)
    coor3d = np.expand_dims(np.dstack([x, y, z]), -1)
    coor3d_r = np.matmul(Rot, coor3d)
    for i in range(6):
      coor3d_rf = np.matmul(R_list[i], coor3d_r).squeeze(-1)

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

      self.grid_list.append(np.dstack([u, v]).astype(np.float32))

  def trans(self, fish):
    c, h, w = fish.shape
    fish = torch.from_numpy(fish).unsqueeze(0)
    out = np.zeros([6, c, self.cube_face_h, self.cube_face_w])
    for i in range(6):
      grid = torch.from_numpy(self.grid_list[i]).unsqueeze(0)
      cube = F.grid_sample(fish, grid, mode='bilinear', align_corners=True)
      cube = cube.squeeze_(0).numpy()
      mask = np.repeat(np.expand_dims(self.invalidMaskList[i], 0), c, 0)
      cube[mask] = 0.0
      # out[:, :, i * self.cube_face_w:(i + 1) * self.cube_face_w] = cube
      out[i, :, :, :] = cube
    return out


class Cubemap2Fisheye:
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
    # Compute the back grid
    grid_back_raw = coor3d_r / np.abs(z_3d)
    grid_back_w = grid_back_raw[:, :, 0]
    grid_back_h = -grid_back_raw[:, :, 1]
    grid_back = np.concatenate([np.expand_dims(grid_back_w, 2), np.expand_dims(grid_back_h, 2)], 2)
    mask_back = ((grid_back_w <= 1) * (grid_back_w >= -1)) * ((grid_back_h <= 1) * (grid_back_h >= -1)) * (grid_back_raw[:, :, 2] < 0)
    masked_grid_back = grid_back * np.float32(np.expand_dims(mask_back, 2))
    self.masked_grid_list.append(masked_grid_back)
    self.mask_list.append(mask_back)
    # Compute the left grid
    grid_left_raw = coor3d_r / np.abs(x_3d)
    grid_left_w = grid_left_raw[:, :, 2]
    grid_left_h = -grid_left_raw[:, :, 1]
    grid_left = np.concatenate([np.expand_dims(grid_left_w, 2), np.expand_dims(grid_left_h, 2)], 2)
    mask_left = ((grid_left_w <= 1) * (grid_left_w >= -1)) * ((grid_left_h <= 1) * (grid_left_h >= -1)) * (grid_left_raw[:, :, 0] > 0)
    masked_grid_left = grid_left * np.float32(np.expand_dims(mask_left, 2))
    self.masked_grid_list.append(masked_grid_left)
    self.mask_list.append(mask_left)
    # Compute the front grid
    grid_front_raw = coor3d_r / np.abs(z_3d)
    grid_front_w = -grid_front_raw[:, :, 0]
    grid_front_h = -grid_front_raw[:, :, 1]
    grid_front = np.concatenate([np.expand_dims(grid_front_w, 2), np.expand_dims(grid_front_h, 2)], 2)
    mask_front = ((grid_front_w <= 1) * (grid_front_w >= -1)) * ((grid_front_h <= 1) * (grid_front_h >= -1)) * (grid_front_raw[:, :, 2] > 0)
    masked_grid_front = grid_front * np.float32(np.expand_dims(mask_front, 2))
    self.masked_grid_list.append(masked_grid_front)
    self.mask_list.append(mask_front)
    # Compute the right grid
    grid_right_raw = coor3d_r / np.abs(x_3d)
    grid_right_w = -grid_right_raw[:, :, 2]
    grid_right_h = -grid_right_raw[:, :, 1]
    grid_right = np.concatenate([np.expand_dims(grid_right_w, 2), np.expand_dims(grid_right_h, 2)], 2)
    mask_right = ((grid_right_w <= 1) * (grid_right_w >= -1)) * ((grid_right_h <= 1) * (grid_right_h >= -1)) * (grid_right_raw[:, :, 0] < 0)
    masked_grid_right = grid_right * np.float32(np.expand_dims(mask_right, 2))
    self.masked_grid_list.append(masked_grid_right)
    self.mask_list.append(mask_right)
    # Compute the up grid
    grid_up_raw = coor3d_r / np.abs(y_3d)
    grid_up_w = -grid_up_raw[:, :, 0]
    grid_up_h = grid_up_raw[:, :, 2]
    grid_up = np.concatenate([np.expand_dims(grid_up_w, 2), np.expand_dims(grid_up_h, 2)], 2)
    mask_up = ((grid_up_w <= 1) * (grid_up_w >= -1)) * ((grid_up_h <= 1) * (grid_up_h >= -1)) * (grid_up_raw[:, :, 1] > 0)
    masked_grid_up = grid_up * np.float32(np.expand_dims(mask_up, 2))
    self.masked_grid_list.append(masked_grid_up)
    self.mask_list.append(mask_up)
    # Compute the down grid
    grid_down_raw = coor3d_r / np.abs(y_3d)
    grid_down_w = -grid_down_raw[:, :, 0]
    grid_down_h = -grid_down_raw[:, :, 2]
    grid_down = np.concatenate([np.expand_dims(grid_down_w, 2), np.expand_dims(grid_down_h, 2)], 2)
    mask_down = ((grid_down_w <= 1) * (grid_down_w >= -1)) * ((grid_down_h <= 1) * (grid_down_h >= -1)) * (grid_down_raw[:, :, 1] < 0)
    masked_grid_down = grid_down * np.float32(np.expand_dims(mask_down, 2))
    self.masked_grid_list.append(masked_grid_down)
    self.mask_list.append(mask_down)

  def trans(self, cube_faces):  #cube_faces:6 x c x h x w, 6 faces of the cube map. back - left - front - right - up - down
    n, c, h, w = cube_faces.shape
    assert n == 6, ("cube map should have 6 faces, but the number of input faces is {}".format(n))
    out = np.zeros([c, self.fish_h, self.fish_w])
    for i in range(0, 6):
      ori = cube_faces[i, :, :, :]
      ori = torch.from_numpy(ori).unsqueeze(0)
      mask_grid = torch.from_numpy(self.masked_grid_list[i]).unsqueeze(0)
      fish = F.grid_sample(ori, mask_grid, mode='bilinear', align_corners=True)
      fish = fish.squeeze_(0).numpy()
      mask = ~(np.repeat(np.expand_dims(self.mask_list[i], 0), c, 0))
      fish[mask] = 0.0
      out = out + fish
    return out


"""
for Kannala_Brandt(KB) projection fisheye images
"""
# transform fisheye images to cubemap projection
# Note： Fish-eye image
# fisheye : valid imaging area height=width=radius, Kannala_Brandt projection,

# Note： Cubemap image
# cube map: each face is a square. 6 faces are arranged into a row with the order:
# back - left - front - right - up - down
# image shape is [h x w] where h=face_h, w=6*face_w
# the width range of the back face is [0:face_w], left face is [face_w:2*face_w], and so on


class FisheyeKB2Cubemap:
  def __init__(self, cube_face_h, cube_face_w, fish_h, fish_w, fish_FoV, fx, fy, cx, cy, kb_theta_coes, Rot=np.identity(3, dtype=np.float32)):
    self.FoV = fish_FoV // 2
    self.radius = fish_h // 2
    R_list = []
    self.grid_list = []
    self.invalidMaskList = []
    phi = np.array([1, 1 / 2, 0, -1 / 2]) * np.pi  # back,left,front,right
    theta = np.array([-1 / 2, 1 / 2]) * np.pi  #up, down

    self.fish_h = fish_h
    self.fish_w = fish_w
    self.cube_face_h = cube_face_h
    self.cube_face_w = cube_face_w

    # KB model paras
    self.fx = fx
    self.fy = fy
    self.cx = cx
    self.cy = cy
    self.kb_theta_coes = kb_theta_coes
    # assume the camera model is KB4: d_theta = theta+k1*theta^3+k2*theta^5+k3*theta^7+k4*theta^9
    assert (len(self.kb_theta_coes) == 4)

    for ph in phi:
      r = ph * np.array([0, 1, 0])
      R = cv2.Rodrigues(r)[0]
      R_list.append(R)
    for th in theta:
      r = th * np.array([1, 0, 0])
      R = cv2.Rodrigues(r)[0]
      R_list.append(R)

    xc = (cube_face_w - 1.0) / 2
    yc = (cube_face_h - 1.0) / 2
    face_x, face_y = np.meshgrid(range(cube_face_w), range(cube_face_h))
    x = xc - face_x
    y = yc - face_y
    z = np.ones_like(x) * ((cube_face_h - 1.0) / 2)
    x, y, z = x / z, y / z, z / z
    D = np.sqrt(x * x + y * y + z * z)
    coor3d = np.expand_dims(np.dstack([x, y, z]), -1)
    coor3d_r = np.matmul(Rot, coor3d)
    for i in range(6):
      coor3d_rf = np.matmul(R_list[i], coor3d_r).squeeze(-1)  #x,y,z = left,up,front
      fisheye_r = np.sqrt(coor3d_rf[:, :, 0] * coor3d_rf[:, :, 0] + coor3d_rf[:, :, 1] * coor3d_rf[:, :, 1])
      fish_theta = np.clip(np.arctan2(fisheye_r, coor3d_rf[:, :, 2]), 0, np.pi)
      invalidMask = (fish_theta > (self.FoV / 180 * np.pi))
      fish_theta[invalidMask] = 0
      fish_theta_sq = fish_theta * fish_theta
      fish_dtheta = fish_theta + self.kb_theta_coes[0] * fish_theta * fish_theta_sq + self.kb_theta_coes[1] * fish_theta * fish_theta_sq * fish_theta_sq + self.kb_theta_coes[
          2] * fish_theta * fish_theta_sq * fish_theta_sq * fish_theta_sq + self.kb_theta_coes[3] * fish_theta * fish_theta_sq * fish_theta_sq * fish_theta_sq * fish_theta_sq
      # we define y==up and x==left, while in Image Coordinate y==down and x==right
      fish_phi = np.clip(np.arctan2(-coor3d_rf[:, :, 1], -coor3d_rf[:, :, 0]), -np.pi, np.pi)
      self.invalidMaskList.append(invalidMask)

      fish_x = self.fx * fish_dtheta * np.cos(fish_phi) + self.cx
      fish_y = self.fy * fish_dtheta * np.sin(fish_phi) + self.cy
      # print(fish_x.max(), fish_x.min())
      # print(fish_y.max(), fish_y.min())

      # normalization
      u = (fish_x) / (fish_w - 1) * 2.0 - 1
      v = (fish_y) / (fish_h - 1) * 2.0 - 1
      u[u > 1] = 1
      u[u < -1] = -1
      v[v > 1] = 1
      v[v < -1] = -1

      self.grid_list.append(np.dstack([u, v]).astype(np.float32))

  def trans(self, fish):
    c, h, w = fish.shape
    fish = torch.from_numpy(fish).unsqueeze(0)
    out = np.zeros([6, c, self.cube_face_h, self.cube_face_w])
    for i in range(6):
      grid = torch.from_numpy(self.grid_list[i]).unsqueeze(0)
      cube = F.grid_sample(fish, grid, mode='bilinear', align_corners=True)
      cube = cube.squeeze_(0).numpy()
      mask = np.repeat(np.expand_dims(self.invalidMaskList[i], 0), c, 0)
      cube[mask] = 0.0
      # out[:, :, i * self.cube_face_w:(i + 1) * self.cube_face_w] = cube
      out[i, :, :, :] = cube
    return out


class Cubemap2FisheyeKB:
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
    #NOTE: need to fixed here
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
    # Compute the back grid
    grid_back_raw = coor3d_r / np.abs(z_3d)
    grid_back_w = grid_back_raw[:, :, 0]
    grid_back_h = -grid_back_raw[:, :, 1]
    grid_back = np.concatenate([np.expand_dims(grid_back_w, 2), np.expand_dims(grid_back_h, 2)], 2)
    mask_back = ((grid_back_w <= 1) * (grid_back_w >= -1)) * ((grid_back_h <= 1) * (grid_back_h >= -1)) * (grid_back_raw[:, :, 2] < 0)
    masked_grid_back = grid_back * np.float32(np.expand_dims(mask_back, 2))
    self.masked_grid_list.append(masked_grid_back)
    self.mask_list.append(mask_back)
    # Compute the left grid
    grid_left_raw = coor3d_r / np.abs(x_3d)
    grid_left_w = grid_left_raw[:, :, 2]
    grid_left_h = -grid_left_raw[:, :, 1]
    grid_left = np.concatenate([np.expand_dims(grid_left_w, 2), np.expand_dims(grid_left_h, 2)], 2)
    mask_left = ((grid_left_w <= 1) * (grid_left_w >= -1)) * ((grid_left_h <= 1) * (grid_left_h >= -1)) * (grid_left_raw[:, :, 0] > 0)
    masked_grid_left = grid_left * np.float32(np.expand_dims(mask_left, 2))
    self.masked_grid_list.append(masked_grid_left)
    self.mask_list.append(mask_left)
    # Compute the front grid
    grid_front_raw = coor3d_r / np.abs(z_3d)
    grid_front_w = -grid_front_raw[:, :, 0]
    grid_front_h = -grid_front_raw[:, :, 1]
    grid_front = np.concatenate([np.expand_dims(grid_front_w, 2), np.expand_dims(grid_front_h, 2)], 2)
    mask_front = ((grid_front_w <= 1) * (grid_front_w >= -1)) * ((grid_front_h <= 1) * (grid_front_h >= -1)) * (grid_front_raw[:, :, 2] > 0)
    masked_grid_front = grid_front * np.float32(np.expand_dims(mask_front, 2))
    self.masked_grid_list.append(masked_grid_front)
    self.mask_list.append(mask_front)
    # Compute the right grid
    grid_right_raw = coor3d_r / np.abs(x_3d)
    grid_right_w = -grid_right_raw[:, :, 2]
    grid_right_h = -grid_right_raw[:, :, 1]
    grid_right = np.concatenate([np.expand_dims(grid_right_w, 2), np.expand_dims(grid_right_h, 2)], 2)
    mask_right = ((grid_right_w <= 1) * (grid_right_w >= -1)) * ((grid_right_h <= 1) * (grid_right_h >= -1)) * (grid_right_raw[:, :, 0] < 0)
    masked_grid_right = grid_right * np.float32(np.expand_dims(mask_right, 2))
    self.masked_grid_list.append(masked_grid_right)
    self.mask_list.append(mask_right)
    # Compute the up grid
    grid_up_raw = coor3d_r / np.abs(y_3d)
    grid_up_w = -grid_up_raw[:, :, 0]
    grid_up_h = grid_up_raw[:, :, 2]
    grid_up = np.concatenate([np.expand_dims(grid_up_w, 2), np.expand_dims(grid_up_h, 2)], 2)
    mask_up = ((grid_up_w <= 1) * (grid_up_w >= -1)) * ((grid_up_h <= 1) * (grid_up_h >= -1)) * (grid_up_raw[:, :, 1] > 0)
    masked_grid_up = grid_up * np.float32(np.expand_dims(mask_up, 2))
    self.masked_grid_list.append(masked_grid_up)
    self.mask_list.append(mask_up)
    # Compute the down grid
    grid_down_raw = coor3d_r / np.abs(y_3d)
    grid_down_w = -grid_down_raw[:, :, 0]
    grid_down_h = -grid_down_raw[:, :, 2]
    grid_down = np.concatenate([np.expand_dims(grid_down_w, 2), np.expand_dims(grid_down_h, 2)], 2)
    mask_down = ((grid_down_w <= 1) * (grid_down_w >= -1)) * ((grid_down_h <= 1) * (grid_down_h >= -1)) * (grid_down_raw[:, :, 1] < 0)
    masked_grid_down = grid_down * np.float32(np.expand_dims(mask_down, 2))
    self.masked_grid_list.append(masked_grid_down)
    self.mask_list.append(mask_down)

  def trans(self, cube_faces):  #cube_faces:6 x c x h x w, 6 faces of the cube map. back - left - front - right - up - down
    n, c, h, w = cube_faces.shape
    assert n == 6, ("cube map should have 6 faces, but the number of input faces is {}".format(n))
    out = np.zeros([c, self.fish_h, self.fish_w])
    for i in range(0, 6):
      ori = cube_faces[i, :, :, :]
      ori = torch.from_numpy(ori).unsqueeze(0)
      mask_grid = torch.from_numpy(self.masked_grid_list[i]).unsqueeze(0)
      fish = F.grid_sample(ori, mask_grid, mode='bilinear', align_corners=True)
      fish = fish.squeeze_(0).numpy()
      mask = ~(np.repeat(np.expand_dims(self.mask_list[i], 0), c, 0))
      fish[mask] = 0.0
      out = out + fish
    return out


if __name__ == '__main__':
  FoV = 210
  fx = fy = (1024) / (FoV / 180 * np.pi)
  cx = cy = (1024 - 1) / 2
  kb_theta_coes = [0, 0, 0, 0]
  #f2c = Fisheye2Cubemap(512, 512, 1024, 1024, FoV)
  f2c = FisheyeKB2Cubemap(512, 512, 1024, 1024, FoV, fx, fy, cx, cy, kb_theta_coes)
  #c2f = Cubemap2Fisheye(512, 512, 1024, 1024, FoV)
  fish = cv2.imread('./imgs/fish' + str(FoV) + '.png').transpose((2, 0, 1)).astype(np.float32)
  cube = f2c.trans(fish).astype(np.float32)
  cubesave = np.zeros([3, 512, 512 * 6])
  for i in range(6):
    cubesave[:, :, i * 512:(i + 1) * 512] = cube[i, :, :, :]
  cubesave = cubesave.transpose((1, 2, 0))
  cv2.imwrite('./imgs/cubesave' + str(FoV) + '_kb.png', cubesave.astype(np.uint8))
  # fish2 = c2f.trans(cube)
  # fish2 = fish2.transpose((1, 2, 0))
  # fish2 = (fish2 - np.min(fish2)) / (np.max(fish2) - np.min(fish2)) * 255
  # cv2.imwrite('./imgs/fish2_' + str(FoV) + '.png', fish2.astype(np.uint8))
