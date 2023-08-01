import torch
import torchgeometry as tgm
import cv2
import torchvision.transforms as transforms
import numpy as np


image = cv2.imread('./mingren_01.png')
homography = torch.tensor([[[1.00007968e+00, 2.67032421e-21, -7.65380859e-02],
                            [4.78591864e-16, 1.00007971e+00, -3.84232752e+00],
                            [6.23388628e-19, -0.00000000e+00, 1.00000000e+00]]],dtype=torch.float)

print(image.shape)

# image=cv2.warpPerspective(image, homography, (541, 447))
# cv2.imshow('',image)

transf = transforms.ToTensor()
image = transf(image)
# cv2.resize()447
image = image.float().unsqueeze(0)
image.cuda()
homography.cuda()

warper = tgm.HomographyWarper(447, 541)
output = warper(image, homography)  # NxCxHxW
output = output.squeeze(0).cpu().permute(1, 2, 0)
print(output.shape)
output = np.array(output)
# cv2.imshow('',np.array(output))
# key = cv2.waitKey(0)


# output = cv2.cvtColor(output, cv2.COLOR_BGR2RGB)
from PIL import Image
import matplotlib.pyplot as plt

# 图片路径
# img = Image.open("/home/newj/图片/space.jpeg")

plt.figure("Image")  # 图像窗口名称
plt.imshow(output)
plt.axis('on')  # 关掉坐标轴为 off
plt.title('warped_img')  # 图像题目

# 必须有这个，要不然无法显示
plt.show()
