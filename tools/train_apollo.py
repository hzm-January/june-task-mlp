import sys

import cv2
import torchgeometry as tgm
import kornia.geometry.transform as kgt
import kornia.geometry.conversions as kgc
import kornia.enhance as keh
import torch.nn.functional as F

sys.path.append('/home/houzm/houzm/02_code/bev_lane_det-wave-mlp')  # 添加模块搜索路径
import torch
from torch.optim.lr_scheduler import CosineAnnealingLR  # 导入余弦退火学习率调度器
from torch.utils.data import DataLoader  # 导入数据加载器
import torch.nn as nn
from models.util.load_model import load_checkpoint, resume_training  # 导入加载和恢复模型的函数
from models.util.save_model import save_model_dp  # 导入保存模型的函数
from models.loss import IoULoss, NDPushPullLoss  # 导入自定义的损失函数
from utils.config_util import load_config_module  # 导入加载配置文件的函数
from sklearn.metrics import f1_score  # 导入F1分数计算函数
import numpy as np
import os

# os.environ["CUDA_VISIBLE_DEVICES"] = "4,7"
# os.environ["CUDA_VISIBLE_DEVICES"] = "5,6"
# os.environ["CUDA_VISIBLE_DEVICES"] = "4,5"
os.environ["CUDA_VISIBLE_DEVICES"] = "6,7"


# 定义一个继承自nn.Module的类，将模型和损失函数组合在一起
class Combine_Model_and_Loss(torch.nn.Module):
    def __init__(self, model):
        super(Combine_Model_and_Loss, self).__init__()
        self.model = model
        self.bce = torch.nn.BCEWithLogitsLoss(pos_weight=torch.tensor([10.0]))  # 定义二元交叉熵损失函数
        self.iou_loss = IoULoss()  # 定义IoU损失函数
        self.poopoo = NDPushPullLoss(1.0, 1., 1.0, 5.0, 200)  # 定义自定义的NDPushPull损失函数
        self.mse_loss = nn.MSELoss()  # 定义均方误差损失函数
        self.bce_loss = nn.BCELoss()  # 定义二元交叉熵损失函数
        # self.sigmoid = nn.Sigmoid()

    # 正向传播函数
    '''image,image_gt_segment,image_gt_instance,ipm_gt_segment,ipm_gt_instance'''
    # input_data, image_gt, configs, gt_seg_data, gt_emb_data, offset_y_data, z_data, image_gt_segment, image_gt_instance
    def forward(self, inputs, images_gt, configs, gt_seg=None, gt_instance=None, gt_offset_y=None, gt_z=None,
                image_gt_segment=None,
                image_gt_instance=None, train=True):
        # images_gt current camera 图像像素坐标系，尺寸与image相同，图片中只有车道线的信息，不同车道线不同颜色，
        res = self.model(inputs, images_gt.clone(), configs)  # 调用模型进行预测
        image_gt_instance_h = res[0]
        image_gt_segment_h = res[1]
        homograph_matrix = res[2]
        pred, emb, offset_y, z = res[3]
        pred_2d, emb_2d = res[4]
        if train:
            ## 3d pred(8,1,200,48) gt_seg(8,1,200,48) gt_instance(8,1,200,48) emb(8,1,200,48)
            loss_seg = self.bce(pred, gt_seg) + self.iou_loss(torch.sigmoid(pred), gt_seg)  # 计算BEV分割损失和IoU损失
            loss_emb = self.poopoo(emb, gt_instance)  # 计算嵌入向量损失
            loss_offset = self.bce_loss(gt_seg * torch.sigmoid(offset_y), gt_offset_y)  # 计算偏移量损失
            loss_z = self.mse_loss(gt_seg * z, gt_z)  # 计算高度损失
            loss_total = 3 * loss_seg + 0.5 * loss_emb  # 计算总损失
            loss_total = loss_total.unsqueeze(0)  # 将总损失转换成一维张量
            loss_offset = 60 * loss_offset.unsqueeze(0)  # 将偏移量损失转换成一维张量并乘以60
            loss_z = 30 * loss_z.unsqueeze(0)  # 将高度损失转换成一维张量并乘以30
            ## 2d
            # image_gt_segment_h(8,1,144,256) float32 = image_gt current camera 图像像素坐标系，尺寸与image相同，图片中只有车道线的信息，不同车道线不同颜色，(8,1,144,256)
            # image_gt_segment_h(8,1,144,256) 与image_gt_instance唯一的不同是没有不同车道线的标注，只有0 1值，标注像素点是否是车道线。
            # pred_2d (8,1,144,256) float32
            # emb_2d (8,2,144,256)
            loss_seg_2d = self.bce(pred_2d, image_gt_segment_h) + self.iou_loss(torch.sigmoid(pred_2d),
                                                                                image_gt_segment_h)  # 计算2D分割损失和IoU损失
            loss_emb_2d = self.poopoo(emb_2d, image_gt_instance_h)  # 计算2D嵌入向量损失
            loss_total_2d = 3 * loss_seg_2d + 0.5 * loss_emb_2d  # 计算2D总损失
            loss_total_2d = loss_total_2d.unsqueeze(0)  # 将2D总损失转换成一维张量
            # 计算H损失
            # 将Virtual Image上prediction labels用H的逆矩阵变换回Image源图 pred_2d(8,1,144,256)
            # pred_2d_h_inv = cv2.warpPerspective(pred_2d.clone().cpu().numpy(), homograph_matrix.clone().cpu().numpy(), # pred_2d
            #                                     configs.output_2d_shape)  # output_2d_shape(144,256)
            homograph_matrix_inv = torch.inverse(homograph_matrix)
            # homograph_matrix_inv = homograph_matrix_inv.reshape(-1, 9)
            homograph_matrix_inv = F.normalize(homograph_matrix_inv, dim=(1, 2), p=2, eps=1e-6)# H^-1
            # homograph_matrix_inv = homograph_matrix_inv.reshape()
            # homograph_matrix_inv = keh.normalize(homograph_matrix_inv.unsqueeze(0), mean=homograph_matrix_inv.mean(dim=(1, 2)), std=homograph_matrix_inv.var(dim=(1, 2))).squeeze(0)
            # homograph_matrix_inv = kgc.normalize_homography(homograph_matrix_inv,(pred_2d.shape[2], pred_2d.shape[3]), (pred_2d.shape[2], pred_2d.shape[3]))
            homograph_matrix_inv = kgc.denormalize_homography(homograph_matrix_inv, (pred_2d.shape[2], pred_2d.shape[3]), (pred_2d.shape[2], pred_2d.shape[3]))
            pred_2d_h_invs = kgt.warp_perspective(pred_2d, homograph_matrix_inv, configs.output_2d_shape)
            emb_2d_h_invs = kgt.warp_perspective(emb_2d, homograph_matrix_inv, configs.output_2d_shape)
            # pred_2d_h_invs = torch.round(pred_2d_h_invs)
            # emb_2d_h_invs = torch.round(emb_2d_h_invs)
            # pred_2d_h_invs = torch.rand(pred_2d_h_invs)
            # mean = torch.zeros(1, inputs.shape[1])
            # std = 255. * torch.ones(1, inputs.shape[1])
            # pred_2d_h_invs = kgt.homography_warp(pred_2d, keh.denormalize(homograph_matrix_inv, mean, std), configs.output_2d_shape,
            #                                     padding_mode="zeros", normalized_coordinates=False, normalized_homography=False)
            # pred_2d(4,1,144,256)
            # pred_2d_h_invs = torch.zeros_like(pred_2d, dtype=torch.float).cuda()
            # for i in range(pred_2d.shape[0]):
            #     pred_2d_h_inv = cv2.warpPerspective(pred_2d[i].permute(1, 2, 0).detach().cpu().numpy(),
            #                                         homograph_matrix_inv[i].detach().cpu().numpy(),
            #                                         (configs.output_2d_shape[1], configs.output_2d_shape[0]))
            #     pred_2d_h_invs[i] = torch.tensor(pred_2d_h_inv, dtype=torch.float).unsqueeze(
            #         0).cuda()  # images (3,576,1024)

            loss_seg_hg = self.bce(pred_2d_h_invs, image_gt_segment) + self.iou_loss(torch.sigmoid(pred_2d_h_invs),
                                                                                 image_gt_segment)
            loss_emb_hg = self.poopoo(emb_2d_h_invs, image_gt_instance)  # 计算2D嵌入向量损失
            loss_total_hg = 3 * loss_seg_hg + 0.5 * loss_emb_hg  # 计算hg总损失
            return pred, loss_total, loss_total_2d, loss_offset, loss_z, homograph_matrix, homograph_matrix_inv, loss_total_hg, loss_seg_hg, loss_emb_hg # 返回预测结果和损失
        else:
            return pred  # 返回预测结果


# 训练一个epoch的函数
def train_epoch(model, dataset, optimizer, scheduler, configs, epoch):
    # Last iter as mean loss of whole epoch
    model.train()  # 将模型设置为训练模式
    losses_avg = {}
    '''image,image_gt_segment,image_gt_instance,ipm_gt_segment,ipm_gt_instance'''
    # image.float(), image_gt.float(), bev_gt_segment.float(), bev_gt_instance.float(), bev_gt_offset.float(), bev_gt_z.float(), image_gt_segment.float(), image_gt_instance.float()
    for idx, (
            input_data, image_gt, gt_seg_data, gt_emb_data, offset_y_data, z_data, image_gt_segment,
            image_gt_instance) in enumerate(
        dataset):
        # loss_back, loss_iter = forward_on_cuda(gpu, gt_data, input_data, loss, models)
        input_data = input_data.cuda()  # 将输入数据转移到GPU上
        gt_seg_data = gt_seg_data.cuda()  # 将BEV分割标签转移到GPU上
        gt_emb_data = gt_emb_data.cuda()  # 将嵌入向量标签转移到GPU上
        offset_y_data = offset_y_data.cuda()  # 将偏移量标签转移到GPU上
        z_data = z_data.cuda()  # 将高度标签转移到GPU上
        # image_gt_segment = image_gt_segment.cuda() # 将2D分割标签转移到GPU上
        # image_gt_instance = image_gt_instance.cuda() # 将2D嵌入向量标签转移到GPU上

        prediction, loss_total_bev, loss_total_2d, loss_offset, loss_z, hg_matrix, homograph_matrix_inv, loss_total_hg, loss_seg_hg, loss_emb_hg = model(input_data,
                                                                                                   image_gt,
                                                                                                   configs,
                                                                                                   gt_seg_data,
                                                                                                   gt_emb_data,
                                                                                                   offset_y_data,
                                                                                                   z_data,
                                                                                                   image_gt_segment,
                                                                                                   image_gt_instance)  # 正向传播
        loss_back_bev = loss_total_bev.mean()  # 计算BEV总损失的平均值
        loss_back_2d = loss_total_2d.mean()  # 计算2D总损失的平均值
        loss_offset = loss_offset.mean()  # 计算偏移量损失的平均值
        loss_z = loss_z.mean()  # 计算高度损失的平均值
        loss_total_hg = loss_total_hg.mean()
        loss_seg_hg = loss_seg_hg.mean()
        loss_emb_hg = loss_emb_hg.mean()
        loss_back_total = loss_back_bev + 0.5 * loss_back_2d + loss_offset + loss_z + 0.5 * loss_total_hg  # 计算总损失

        ''' caclute loss '''
        optimizer.zero_grad()  # 清空梯度
        loss_back_total.backward()  # 反向传播计算梯度
        optimizer.step()  # 更新模型参数
        if idx % 50 == 0:
            target = gt_seg_data.detach().cpu().numpy().ravel()  # 将BEV分割标签从GPU中取出并展平为一维数组
            pred = torch.sigmoid(prediction).detach().cpu().numpy().ravel()  # 将预测结果从GPU中取出并展平为一维数组
            f1_bev_seg = f1_score((target > 0.5).astype(np.int64), (pred > 0.5).astype(np.int64),
                                  zero_division=1)  # 计算F1分数
            loss_iter = {"【BEV Loss】": loss_back_bev.item(), '【offset loss】': loss_offset.item(),
                         '【z loss】': loss_z.item(),
                         "【F1_BEV_seg】": f1_bev_seg}  # 计算各项损失和F1分数
            # losses_show = loss_iter
            # print('-' * 80)
            # 3d loss(bev loss) = 3 * loss_seg + 0.5 * loss_emb
            # 2d loss = 3 * loss_seg_2d + 0.5 * loss_emb_2d
            # loss_back_total = 3d loss + 0.5 * 2d loss + loss_offset + loss_z + loss_hg
            print('| %3d | Hlr: %.10f | Blr: %.10f | 2d+3d: %f | F1: %f | Offset: %f | Z: %f | 3d: %f | 2d: %f | h: %f | hs: %f | he: %f |' % (
                idx, scheduler.optimizer.param_groups[0]['lr'], scheduler.optimizer.param_groups[1]['lr'], loss_back_total.item(),
                f1_bev_seg, loss_offset.item(), loss_z.item(),
                loss_back_bev.item(), loss_back_2d.item(), loss_total_hg.item(), loss_seg_hg.item(), loss_emb_hg.item()))
            # print('-' * 80)

        if idx != 0 and idx % 350 == 0: #700 350
            # print([i for i in hg_matrix[0].view(1, 9).squeeze(0).detach().cpu().numpy()])
            print('hm__: ', [i for i in hg_matrix[0].view(1, 9).squeeze(0).detach().cpu().numpy()])  # 原始matrix
            hg_mtxs_image = kgc.denormalize_homography(hg_matrix, configs.input_shape, configs.input_shape)
            hg_mtxs_image_gt = kgc.denormalize_homography(hg_matrix, configs.output_2d_shape, configs.output_2d_shape)
            print('hm_m: ',
                  [i for i in hg_mtxs_image[0].view(1, 9).squeeze(0).detach().cpu().numpy()])  # image_denormal
            print('hm_t: ',
                  [i for i in hg_mtxs_image_gt[0].view(1, 9).squeeze(0).detach().cpu().numpy()])  # image_gt_denormal
            print('hm_v: ', [i for i in
                             homograph_matrix_inv[0].view(1, 9).squeeze(0).detach().cpu().numpy()])  # image_gt_denormal



# worker_fuction  加载配置文件
def worker_function(config_file, gpu_id, checkpoint_path=None):
    print('use gpu ids is ' + ','.join([str(i) for i in gpu_id]))
    configs = load_config_module(config_file)  # 加载配置文件

    ''' models and optimizer '''
    model = configs.model()  # 加载模型
    model = Combine_Model_and_Loss(model)  # 将模型和损失函数组合在一起
    if torch.cuda.is_available():
        model = model.cuda()  # 将模型转移到GPU上

    params_hg_ids = list(map(id, model.model.hg.parameters()))
    params_hg = filter(lambda m: (id(m) in params_hg_ids) and m.requires_grad, model.parameters())
    params_not_hg = filter(lambda m: (id(m) not in params_hg_ids) and m.requires_grad, model.parameters())
    # optimizer = configs.optimizer(params=[
    #     {'params': params_hg, 'lr':configs.optimizer_params_hg['lr']},
    #     {'params': params_not_hg, 'lr':configs.optimizer_params['lr']}
    # ])  # 定义优化器
    optimizer = configs.optimizer(params=[
        {'params': params_hg, **configs.optimizer_params_hg},
        {'params': params_not_hg, **configs.optimizer_params}
    ])  # 定义优化器
    model = torch.nn.DataParallel(model)  # 将模型并行化处理

    scheduler = getattr(configs, "scheduler", CosineAnnealingLR)(optimizer, configs.epochs)  # 定义学习率调度器
    if checkpoint_path:
        if getattr(configs, "load_optimizer", True):
            resume_training(checkpoint_path, model.module, optimizer, scheduler, configs.resume_scheduler)  # 恢复模型和优化器
        else:
            load_checkpoint(checkpoint_path, model.module, None)  # 仅恢复模型s

    ''' dataset '''
    Dataset = getattr(configs, "train_dataset", None)  # 获取数据集
    # 用于确认是否载入Dataset
    print(len(Dataset()))
    print("configs:", configs)
    if Dataset is None:
        Dataset = configs.training_dataset  # 如果没有指定数据集，则使用默认数据集
    train_loader = DataLoader(Dataset(), **configs.loader_args, pin_memory=True)  # 加载数据

    ''' get validation '''
    # if configs.with_validation:
    #     val_dataset = Dataset(**configs.val_dataset_args)
    #     val_loader = DataLoader(val_dataset, **configs.val_loader_args, pin_memory=True)
    #     val_loss = getattr(configs, "val_loss", loss)
    #     if eval_only:
    #         loss_mean = val_dp(model, val_loader, val_loss)
    #         print(loss_mean)
    #         return

    for epoch in range(configs.epochs):
        print('*' * 150, epoch)
        train_epoch(model, train_loader, optimizer, scheduler, configs, epoch)

        save_model_dp(model, optimizer, scheduler, configs.model_save_path, 'ep%03d.pth' % epoch)  # 保存模型
        save_model_dp(model, None, None, configs.model_save_path, 'latest.pth')

        scheduler.step()  # 更新学习率


# TODO template config file.
if __name__ == '__main__':
    import warnings

    warnings.filterwarnings("ignore")
    # worker_function('/home/houzm/houzm/02_code/bev_lane_det-cnn/tools/apollo_config.py', gpu_id=[4,5])  # 调用worker_function函数，传入配置文件路径和GPU编号
    worker_function('/home/houzm/houzm/02_code/bev_lane_det-wave-mlp/tools/apollo_config.py',
                    # gpu_id=[4, 7],
                    # gpu_id=[5, 6],
                    # gpu_id=[4, 5],
                    gpu_id=[6,7],
                    # checkpoint_path='/home/houzm/houzm/03_model/bev_lane_det-wave-mlp/apollo/train/0727/ep099.pth'
                    )  # 调用worker_function函数，传入配置文件路径和GPU编号
