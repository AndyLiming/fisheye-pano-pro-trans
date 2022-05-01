"""
the projection transformation of omnidirectional images
transform between ERP and cube map
class e2c: ERP to CubeMap
class c2e: CUbeMap to ERP
"""

import os
import sys
import cv2
import time
import numpy as np
import torch
import torch.nn.functional as F
from torch.autograd import Variable
import torchvision


class e2c:
  """
  ERP to cube map
  Functions:
  __init__: constructor
  _ToCube: private transformation function
  ToCubeNumpy: 
  ToCubeTensor: callable API
  """
  def __init__(self, batch_size=1, eh=256, ew=512, outW=256, FOV=90, rad=256, CUDA=False):
    """
    Input:
      batch_size: [int] batch size of input tensor, the default is 1 (trans images one by one)
      eh: [int] height of ERP image
      ew: [int] width of ERP image
      outW: [int] width of each face of output cube map
      FOV: [int] field-of-view of the virtual cameras(degree). The default is 90
      rad: [int] radiant of the imaging sphere
      CUDA: [bool] use cuda or not
    Output: None
    """
    R_list = []
    phi = np.array([1, 1 / 2, 0, -1 / 2]) * np.pi  # back,left,front,right
    theta = np.array([-1 / 2, 1 / 2]) * np.pi  #up, down
    self.eh = eh
    self.ew = ew
    self.CUDA = CUDA
    for ph in phi:
      r = ph * np.array([0, 0, 1])
      R = cv2.Rodrigues(r)[0]
      R_list.append(torch.from_numpy(R))
    for th in theta:
      r = th * np.array([1, 0, 0])
      R = cv2.Rodrigues(r)[0]
      R_list.append(torch.from_numpy(R))
    wangle = (180 - FOV) / 2.0
    w_len = 2 * rad * np.sin(np.radians(FOV / 2.0)) / np.sin(np.radians(wangle))
    f = rad / w_len * outW
    cx = (outW - 1) / 2.0
    cy = (outW - 1) / 2.0
    interval = w_len / (outW - 1)

    y_map = np.zeros([outW, outW]) + rad
    x_map = np.tile((np.arange(outW) - cx) * interval, [outW, 1])
    z_map = np.tile((np.arange(outW) - cy) * interval, [outW, 1]).T
    D = np.sqrt(x_map**2 + y_map**2 + z_map**2)
    xyz = np.zeros([outW, outW, 3])
    xyz[:, :, 0] = (rad / D) * x_map[:, :]
    xyz[:, :, 1] = (rad / D) * y_map[:, :]
    xyz[:, :, 2] = (rad / D) * z_map[:, :]
    xyz = torch.from_numpy(xyz)
    reshape_xyz = xyz.view(outW * outW, 3).transpose(0, 1)
    self.batchSize = batch_size
    self.loc = []
    self.grid = []
    for R in R_list:
      result = torch.matmul(R, reshape_xyz).transpose(0, 1)
      tmp_xyz = result.contiguous().view(1, outW, outW, 3)
      self.grid.append(tmp_xyz)
      lon = torch.atan2(result[:, 0], result[:, 1]).view(1, outW, outW, 1) / np.pi
      lat = torch.asin(result[:, 2] / rad).view(1, outW, outW, 1) / (np.pi / 2)

      self.loc.append(torch.cat([lon.repeat(batch_size, 1, 1, 1), lat.repeat(batch_size, 1, 1, 1)], dim=3))
    self.grid_list = []
    for g in self.grid:
      g2 = g.clone()
      scale = f / g2[:, :, :, 2:3]
      g2 *= scale
      self.grid_list.append(g2)

  def _ToCube(self, batch, mode):
    """
    Input:
      batch: [torch tensor] batch data of ERP images/feature maps
      mode: [string] interpolate mode
    Output: Cube map data
    """
    batch_size = batch.size()[0]
    new_lst = [0, 5, 2, 1, 3, 4]
    # 0: back, 1: left, 2: front, 3: right, 4: up, 5: down
    out = []
    for i in new_lst:
      coor = self.loc[i].cuda() if self.CUDA else self.loc[i]
      result = []
      for ii in range(batch_size):
        tmp = F.grid_sample(batch[ii:ii + 1], coor, mode=mode)
        result.append(tmp)
      result = torch.cat(result, dim=0)
      out.append(result)
    return out

  def ToCubeNumpy(self, batch):
    out = self._ToCube(batch)
    result = [x.data.cpu().numpy() for x in out]
    return result

  def ToCubeTensor(self, batch, mode='bilinear'):
    """
    Input:
      batch: [torch tensor] batch data of ERP images/feature maps
      mode: [string] interpolate mode
    Output: Cube map data
    output order: back down front left right up
    """
    assert mode in ['bilinear', 'nearest']
    batch_size = batch.size()[0]
    cube = self._ToCube(batch, mode=mode)
    out_batch = None
    for batch_idx in range(batch_size):
      for cube_idx in range(6):
        patch = torch.unsqueeze(cube[cube_idx][batch_idx, :, :, :], 0)
        if out_batch is None:
          out_batch = patch
        else:
          out_batch = torch.cat([out_batch, patch], dim=0)
    return out_batch


class c2e:
  """
  Cube map to ERP
  Functions:
  __init__: constructor
  _ToEquirec: private transformation function
  getGrid: get grid and mask for 6 faces of cube map
  ToEquirecTensor: callable API
  input faces order: back down front left right up
  """
  def __init__(self, batch_size=1, cubeW=256, outH=256, outW=512, FOV=90, CUDA=False):
    """
    Input:
      batch_size: [int] batch size of input tensor, the default is 6 (6 faces of a cube map)
      cubeW: [int] width of each face of cube map
      outH: [int] height of output ERP image
      outW: [int] width of output ERP image
      FOV: [int] field-of-view of the virtual cameras(degree). The default is 90
      CUDA: [bool] use cuda or not
    Output: None
    """
    self.batch_size = batch_size  # NOTE: not in use at all
    self.cubeW = cubeW
    self.outH = outH
    self.outW = outW
    self.fov = FOV
    self.fov_rad = self.fov * np.pi / 180
    self.CUDA = CUDA

    # Compute the parameters for projection
    self.radius = int(0.5 * cubeW)

    # Map equirectangular pixel to longitude and latitude
    # NOTE: Make end a full length since arange have a right open bound [a, b)
    theta_start = -1 / 2 * np.pi + (np.pi / outW)
    theta_end = 3 / 2 * np.pi
    theta_step = 2 * np.pi / outW
    theta_range = torch.arange(theta_start, theta_end, theta_step)

    phi_start = -0.5 * np.pi + (0.5 * np.pi / outH)
    phi_end = 0.5 * np.pi
    phi_step = np.pi / outH
    phi_range = torch.arange(phi_start, phi_end, phi_step)

    # Stack to get the longitude latitude map
    self.theta_map = theta_range.unsqueeze(0).repeat(outH, 1)
    self.phi_map = phi_range.unsqueeze(-1).repeat(1, outW)
    self.lonlat_map = torch.stack([self.theta_map, self.phi_map], dim=-1)

    # Get mapping relation (h, w, face)
    # [back, down, front, left, right, up] => [0, 1, 2, 3, 4, 5]
    # self.orientation_mask = self.get_orientation_mask()

    # Project each face to 3D cube and convert to pixel coordinates
    self.grid, self.orientation_mask = self.getGrid()

    if self.CUDA:
      self.grid.cuda()
      self.orientation_mask.cuda()

  def getGrid(self):
    """
    Input:
      None
    Output:
      grids and masks of 6 faces 
    """
    # Get the point of equirectangular on 3D ball
    x_3d = (self.radius * torch.cos(self.phi_map) * torch.cos(self.theta_map)).view(self.outH, self.outW, 1)
    y_3d = (self.radius * torch.cos(self.phi_map) * torch.sin(self.theta_map)).view(self.outH, self.outW, 1)
    z_3d = (self.radius * torch.sin(self.phi_map)).view(self.outH, self.outW, 1)

    self.grid_ball = torch.cat([x_3d, y_3d, z_3d], 2).view(self.outH, self.outW, 3)

    # Compute the down grid
    radius_ratio_down = torch.abs(z_3d / self.radius)
    grid_down_raw = self.grid_ball / radius_ratio_down.view(self.outH, self.outW, 1).expand(-1, -1, 3)
    grid_down_w = (-grid_down_raw[:, :, 0].clone() / self.radius).unsqueeze(-1)
    grid_down_h = (-grid_down_raw[:, :, 1].clone() / self.radius).unsqueeze(-1)
    grid_down = torch.cat([grid_down_w, grid_down_h], 2).unsqueeze(0)
    mask_down = (((grid_down_w <= 1) * (grid_down_w >= -1)) * ((grid_down_h <= 1) * (grid_down_h >= -1)) * (grid_down_raw[:, :, 2] == self.radius).unsqueeze(2)).float()

    # Compute the up grid
    radius_ratio_up = torch.abs(z_3d / self.radius)
    grid_up_raw = self.grid_ball / radius_ratio_up.view(self.outH, self.outW, 1).expand(-1, -1, 3)
    grid_up_w = (-grid_up_raw[:, :, 0].clone() / self.radius).unsqueeze(-1)
    grid_up_h = (grid_up_raw[:, :, 1].clone() / self.radius).unsqueeze(-1)
    grid_up = torch.cat([grid_up_w, grid_up_h], 2).unsqueeze(0)
    mask_up = (((grid_up_w <= 1) * (grid_up_w >= -1)) * ((grid_up_h <= 1) * (grid_up_h >= -1)) * (grid_up_raw[:, :, 2] == -self.radius).unsqueeze(2)).float()

    # Compute the front grid
    radius_ratio_front = torch.abs(y_3d / self.radius)
    grid_front_raw = self.grid_ball / radius_ratio_front.view(self.outH, self.outW, 1).expand(-1, -1, 3)
    grid_front_w = (-grid_front_raw[:, :, 0].clone() / self.radius).unsqueeze(-1)
    grid_front_h = (grid_front_raw[:, :, 2].clone() / self.radius).unsqueeze(-1)
    grid_front = torch.cat([grid_front_w, grid_front_h], 2).unsqueeze(0)
    mask_front = (((grid_front_w <= 1) * (grid_front_w >= -1)) * ((grid_front_h <= 1) * (grid_front_h >= -1)) * (torch.round(grid_front_raw[:, :, 1]) == self.radius).unsqueeze(2)).float()

    # Compute the back grid
    radius_ratio_back = torch.abs(y_3d / self.radius)
    grid_back_raw = self.grid_ball / radius_ratio_back.view(self.outH, self.outW, 1).expand(-1, -1, 3)
    grid_back_w = (grid_back_raw[:, :, 0].clone() / self.radius).unsqueeze(-1)
    grid_back_h = (grid_back_raw[:, :, 2].clone() / self.radius).unsqueeze(-1)
    grid_back = torch.cat([grid_back_w, grid_back_h], 2).unsqueeze(0)
    mask_back = (((grid_back_w <= 1) * (grid_back_w >= -1)) * ((grid_back_h <= 1) * (grid_back_h >= -1)) * (torch.round(grid_back_raw[:, :, 1]) == -self.radius).unsqueeze(2)).float()

    # Compute the right grid
    radius_ratio_right = torch.abs(x_3d / self.radius)
    grid_right_raw = self.grid_ball / radius_ratio_right.view(self.outH, self.outW, 1).expand(-1, -1, 3)
    grid_right_w = (-grid_right_raw[:, :, 1].clone() / self.radius).unsqueeze(-1)
    grid_right_h = (grid_right_raw[:, :, 2].clone() / self.radius).unsqueeze(-1)
    grid_right = torch.cat([grid_right_w, grid_right_h], 2).unsqueeze(0)
    mask_right = (((grid_right_w <= 1) * (grid_right_w >= -1)) * ((grid_right_h <= 1) * (grid_right_h >= -1)) * (torch.round(grid_right_raw[:, :, 0]) == -self.radius).unsqueeze(2)).float()

    # Compute the left grid
    radius_ratio_left = torch.abs(x_3d / self.radius)
    grid_left_raw = self.grid_ball / radius_ratio_left.view(self.outH, self.outW, 1).expand(-1, -1, 3)
    grid_left_w = (grid_left_raw[:, :, 1].clone() / self.radius).unsqueeze(-1)
    grid_left_h = (grid_left_raw[:, :, 2].clone() / self.radius).unsqueeze(-1)
    grid_left = torch.cat([grid_left_w, grid_left_h], 2).unsqueeze(0)
    mask_left = (((grid_left_w <= 1) * (grid_left_w >= -1)) * ((grid_left_h <= 1) * (grid_left_h >= -1)) * (torch.round(grid_left_raw[:, :, 0]) == self.radius).unsqueeze(2)).float()

    # Face map contains numbers correspond to that face
    orientation_mask = mask_back * 0 + mask_down * 1 + mask_front * 2 + mask_left * 3 + mask_right * 4 + mask_up * 5

    return torch.cat([grid_back, grid_down, grid_front, grid_left, grid_right, grid_up], 0), orientation_mask

  def _ToEquirec(self, batch, mode):
    """
    Input:
      batch: cube map data [back down front left right up]
      mode: interpolate mode
    Output:
      ERP image
    """
    batch_size, ch, H, W = batch.shape
    if batch_size != 6:
      raise ValueError("Batch size mismatch!!")

    if self.CUDA:
      output = Variable(torch.zeros(1, ch, self.outH, self.outW), requires_grad=False).cuda()
    else:
      output = Variable(torch.zeros(1, ch, self.outH, self.outW), requires_grad=False)

    for ori in range(6):
      grid = self.grid[ori, :, :, :].unsqueeze(0)  # 1, self.output_h, self.output_w, 2
      mask = (self.orientation_mask == ori).unsqueeze(0)  # 1, self.output_h, self.output_w, 1

      if self.CUDA:
        masked_grid = Variable(grid * mask.double().expand(-1, -1, -1, 2)).cuda()  # 1, self.output_h, self.output_w, 2
      else:
        masked_grid = Variable(grid * mask.double().expand(-1, -1, -1, 2))

      source_image = batch[ori].unsqueeze(0)  # 1, ch, H, W
      sampled_image = torch.nn.functional.grid_sample(source_image, masked_grid, mode=mode, align_corners=True)  # 1, ch, self.output_h, self.output_w

      if self.CUDA:
        sampled_image_masked = sampled_image * Variable(mask.float().view(1, 1, self.outH, self.outW).expand(1, ch, -1, -1)).cuda()
      else:
        sampled_image_masked = sampled_image * Variable(mask.float().view(1, 1, self.outH, self.outW).expand(1, ch, -1, -1))
      output = output + sampled_image_masked  # 1, ch, self.output_h, self.output_w

    return output

  def ToEquirecTensor(self, batch, mode='bilinear'):
    """
    Input:
      batch: cube map data [back down front left right up]
      mode: interpolate mode
    Output:
      ERP image
    """
    # Check whether batch size is 6x
    assert mode in ['nearest', 'bilinear']
    batch_size = batch.size()[0]
    if batch_size % 6 != 0:
      raise ValueError("Batch size should be 6x")
    processed = []
    for idx in range(int(batch_size / 6)):
      target = batch[idx * 6:(idx + 1) * 6, :, :, :]
      target_processed = self._ToEquirec(target, mode)
      processed.append(target_processed)

    output = torch.cat(processed, 0)
    return output


# test: erp <--> cube map
"""
a sample of usage
"""
if __name__ == '__main__':
  transe2c = e2c()
  transe2c2 = e2c(eh=512, ew=1024)
  oriSize = 800
  cubeSize = 512
  transc2e = c2e(cubeW=cubeSize, outH=512, outW=1024)
  rootdir = "../datasets/3D60/"
  name = "30_ed1e790a785e4b74b895e41682b2ae881"
  leftImg = cv2.imread(os.path.join(rootdir, "Center_Left_Down/Matterport3D/", name + "_color_0_Left_Down_0.0.png"), cv2.IMREAD_COLOR)
  rightImg = cv2.imread(os.path.join(rootdir, "Right/Matterport3D/", name + "_color_0_Right_0.0.png"), cv2.IMREAD_COLOR)
  depthGT = cv2.imread(os.path.join(rootdir, "Center_Left_Down/Matterport3D/", name + "_depth_0_Left_Down_0.0.exr"), cv2.IMREAD_ANYDEPTH)
  leftImg = leftImg[:, :, (2, 1, 0)]
  leftImg = leftImg.transpose((2, 0, 1)).astype(np.float64)
  rightImg = rightImg.transpose((2, 0, 1)).astype(np.float64)

  leftImg = torch.from_numpy(leftImg)
  rightImg = torch.from_numpy(rightImg)
  depthGT = torch.from_numpy(depthGT)

  leftImg.unsqueeze_(0)
  rightImg.unsqueeze_(0)
  depthGT.unsqueeze_(0)
  depthGT.unsqueeze_(0)

  out_batch = transe2c.ToCubeTensor(leftImg)
  print(out_batch.shape)
  div = torch.zeros([3, 256, 10]).double()
  saveImg = out_batch[0]
  for i in range(1, 6):
    saveImg = torch.cat([saveImg, div, out_batch[i]], dim=2)
  print(saveImg.shape)
  saveImg = (saveImg - torch.min(saveImg)) / (torch.max(saveImg) - torch.min(saveImg))
  torchvision.utils.save_image(saveImg, 'cube.png')
  outErp = transc2e.ToEquirecTensor(out_batch)
  print(outErp.shape)
  saveImg = (outErp - torch.min(outErp)) / (torch.max(outErp) - torch.min(outErp))
  torchvision.utils.save_image(saveImg, 'erp.png')

  # cubeBat = np.ndarray([6, 3, oriSize, oriSize])
  # for i in range(0, 6):
  #   img = cv2.imread(os.path.join("./testImg/", str(i) + ".jpg"), cv2.IMREAD_COLOR)
  #   img = img[:, :, (2, 1, 0)]
  #   img = img.transpose((2, 0, 1)).astype(np.float64)
  #   cubeBat[i, ::] = img
  # cubeBat = torch.from_numpy(cubeBat)
  # outErp = transc2e.ToEquirecTensor(cubeBat)
  # out_batch = transe2c2.ToCubeTensor(outErp)
  # saveImg = (outErp - torch.min(outErp)) / (torch.max(outErp) - torch.min(outErp))
  # torchvision.utils.save_image(saveImg, 'erp.png')
  # div = torch.zeros([3, 256, 10]).double()
  # saveImg = out_batch[0]
  # for i in range(1, 6):
  #   saveImg = torch.cat([saveImg, div, out_batch[i]], dim=2)
  # print(saveImg.shape)
  # saveImg = (saveImg - torch.min(saveImg)) / (torch.max(saveImg) - torch.min(saveImg))
  # torchvision.utils.save_image(saveImg, 'cube.png')