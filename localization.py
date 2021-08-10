import torch
from torch import nn
import torchvision.transforms.functional as FV
from sparsepyramids.recursive_pyramidalize import nested_convs, RecursivePyramidalize2D, RecursiveSumDepyramidalize2D, \
    apply_2func_to_nested_tensors, apply_func_to_nested_tensors
from sparsepyramids.edge_pyramid import image_to_edge_pyramid, edge
from sparsepyramids.normal_linear import NormLinear
import math as m
import torch.functional as F
import torch.nn.functional as NNF
from pnums import PInt

class PyrConvLocalizer(torch.nn.Module):
    def __init__(self):
        super().__init__()

        # scene recognition
        self.nc2 = nested_convs(1, 8, 3, padding=1, bias=True)
        self.nc3 = nested_convs(8, 128, 3, padding=1, bias=True)
        #self.nc4 = nested_convs(64, 256, 3, padding=1, bias=True)
        #self.nc5 = nested_convs(256, 128, 3, padding=1, bias=True)

        # scene to pose database
        self.fcn1 = NormLinear(512, 512)
        self.fcn2 = NormLinear(512, 512)
        self.fcn3 = NormLinear(512, 512)
        self.fcn4 = NormLinear(512, 2048)

        self.pyr = RecursivePyramidalize2D(interpolation=FV.InterpolationMode.NEAREST)
        self.de = RecursiveSumDepyramidalize2D(scale_pow=1)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.zero_()
                # m.weight.data.normal_(0, math.sqrt(2. / n))

        self.cuda()

    def forward(self, img):
        if img.shape[1] == 3:
            # assuming bgr from opencv
            b = img[:, 0:1, ...]
            g = img[:, 1:2, ...]
            r = img[:, 2:3, ...]

            img = 0.299 * r + 0.587 * g + 0.114 * b

        eimg = NNF.conv2d(img, edge.to(img.device))

        '''edge = image_to_edge_pyramid(img)
        edge = apply_2func_to_nested_tensors(edge, 1e-12, torch.add)
        norm = apply_func_to_nested_tensors(edge, torch.norm, dim=(2, 3))
        edge = apply_2func_to_nested_tensors(edge, norm, torch.div)
        edgeimg = self.de.forward(edge)'''
        half = list(eimg.shape[-2:])
        half = [int(h // m.sqrt(2)) for h in half]
        edgeimg = FV.resize(eimg, half)
        #edgepyr2 = self.pyr.forward(edgeimg)

        linepyr = self.nc2.forward(edgeimg)
        #lineimg = self.de.forward(linepyr)
        half = list(linepyr.shape[-2:])
        half = [int(h // m.sqrt(2)) for h in half]
        lineimg = FV.resize(linepyr, half)
        #linepyr2 = self.pyr.forward(lineimg)

        shapepyr = self.nc3.forward(lineimg)
        #shapeimg = self.de.forward(shapepyr)
        #half = list(shapepyr.shape[-2:])
        #half = [int(h // m.sqrt(2)) for h in half]
        #shapeimg = FV.resize(shapeimg, half)
        #shapepyr2 = self.pyr.forward(shapeimg)
        sceneimg = FV.resize(shapepyr, [2, 2])

        '''objpyr = self.nc4.forward(shapepyr2)
        objimg = self.de.forward(objpyr)
        half = list(objimg.shape[-2:])
        half = [int(h // m.sqrt(2)) for h in half]
        objimg = FV.resize(objimg, half)
        objpyr2 = self.pyr.forward(objimg)

        scenepyr = self.nc5.forward(objpyr2)
        sceneimg = self.de.forward(scenepyr)
        sceneimg = FV.resize(sceneimg, [2, 2])'''

        fc1 = self.fcn1.forward(sceneimg.ravel())
        fc2 = self.fcn2.forward(fc1)
        fc3 = self.fcn3.forward(fc2)
        fc4 = self.fcn4.forward(fc3)

        return fc4


def simple_loss(inp: torch.Tensor, goal: torch.Tensor):
    global _goal
    _goal = _goal.to(inp.device)
    loss = NNF.smooth_l1_loss(inp, _goal)
    return loss

from IMU import IMUPoseProbabilistic

if __name__ == '__main__':
    from displayarray import display

    model = PyrConvLocalizer()

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-2, betas=(0.9, 0.999))
    model.train()
    optimizer.zero_grad()

    imu_pose_iter = IMUPoseProbabilistic()

    for d in display(0):
        if d:
            img = next(iter(d.values()))[0]
            torch_img = torch.FloatTensor(img).cuda()
            torch_img = torch_img.permute(2, 0, 1)[None, ...]
            sceneimg = FV.resize(torch_img, [256, 256], interpolation=FV.InterpolationMode.NEAREST)

            pose = model.forward(torch_img)
            imu_pose = next(imu_pose_iter)
            pos = pose.asfloat()[-3:]
            pos_confidence = PInt(pose).overall_confidence()
            pos = PInt(pos, confidence=pos_confidence)
            imu_pose_iter.combine_pos(pos)
            loss = simple_loss(pose, imu_pose)
            print(loss)
            print(pose.asfloat())
            loss.backward()
            optimizer.step()
