import torch
import torchgeometry as tgm
import cv2
import torchvision.transforms as transforms
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

# input = torch.rand(1, 3, 32, 32)
# homography = torch.eye(3).view(1, 3, 3)
image = cv2.imread('./mingren_01.png')

image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
plt.figure("Image")  # 图像窗口名称
plt.imshow(image)
plt.axis('on')  # 关掉坐标轴为 off
plt.title('warped_img')  # 图像题目

# 必须有这个，要不然无法显示
plt.show()
# image = cv2.imread(r'F:/my-home/1-ai-code/02-paper-code/02-bev_lane_det/junk_task_wave_mlp/checkpoints/mingren_01.png')
# homography = torch.tensor([[[9.97315583e-01, -7.81008847e-19, 2.57702637e+00],
#                             [8.25583487e-16, 9.97315512e-01, 8.56324473e+01],
#                             [1.01449342e-18, -8.12895181e-22, 1.00000000e+00]]],dtype=torch.float)
# homography = torch.tensor([[[1.00007968e+00, 2.67032421e-21, -7.65380859e-02],
#                             [4.78591864e-16, 1.00007971e+00, -3.84232752e+00],
#                             [6.23388628e-19, -0.00000000e+00, 1.00000000e+00]]],dtype=torch.float)
homography = torch.tensor([[[1.00034869e+00, 8.87686342e-20, -3.34777832e-01],
                            [6.79346762e-17, 1.00034869e+00, -1.83143486e+01],
                            [2.10711901e-19, 1.01303868e-22, 1.00000000e+00]]],dtype=torch.float)

print(image.shape)

# image=cv2.warpPerspective(image, homography, (541, 447))
# cv2.imshow('',image)

transf = transforms.ToTensor()
image = transf(image)
# cv2.resize()447
image = image.float().unsqueeze(0)
# image.cuda()
# homography.cuda()
output = tgm.homography_warp(image, homography, (447,541), padding_mode="zeros")  # NxCxHxW
output = output.squeeze(0).cpu().permute(1, 2, 0)
print(output.shape)
output = np.array(output)
# cv2.imshow('',np.array(output))
# key = cv2.waitKey(0)


# output = cv2.cvtColor(output, cv2.COLOR_BGR2RGB)


# 图片路径
# img = Image.open("/home/newj/图片/space.jpeg")

plt.figure("Image")  # 图像窗口名称
plt.imshow(output)
plt.axis('on')  # 关掉坐标轴为 off
plt.title('warped_img')  # 图像题目

# 必须有这个，要不然无法显示
plt.show()
