import os
import torch
import numpy as np
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms as transforms
import torchvision.transforms.functional as ttf


'''
dataset.py 任务
#1 读取数据
#2 按尺寸大小中心裁剪
#3 标签编码
#4 numpy to tensor 
#5 实例化数据类
'''


#在PyTorch中，数据加载需要自定义数据集类，并用此类来实例化数据对象，
#实现自定义的数据集需要继承torch.utils.data包中的Dataset类。
#参考 class VOCSegDataset() https://zhuanlan.zhihu.com/p/32506912

class CamvidDataset(Dataset):
    def __init__(self, file_path=[], crop_size):
        '''
        params:
            files_path(list): 数据&标签路径，列表元素第一个是图片路径，第二个是标签路径
        '''
        
        #1 正确读图片&标签路径
        #判断列表是否有两个元素 -- 图片&标签路径
        if len(file_path) !=2:  
            raise ValueError("同时需要图片和标签文件夹的路径，（图片路径在前）")
            # python的raise用法--python的异常处理机制 详见 http://c.biancheng.net/view/2360.html
        self.img_path = file_path[0]
        self.label_path = file_path[1]
        #2 从路径中取出的图片&标签的文件名保持到两个列表中
        self.imgs = self.read_file(self.img_path)
        self.labels = self.read_file(self.label_path)
        #3 初始化数据处理函数
        self.crop_size=crop_size

    #对原图&标签的处理    
    def __getitem__(self, idx):
        #self.img pytorch的写法,直接调用之前定义的内部变量
        #[idx] 也是pytorch里面写好的，读取列表里面的元素数 
        img = self.imgs[idx]
        label = self.labels[idx]

        #从路径&文件名，读取数据（注意，camvid数据集的图片&标签都是png格式的图像数据哦哦）
        img = Image.open(img)
        label = Image.open(label).convert('RGB')  #统一转一下3通道，免得之后的麻烦

        #中心裁剪 需要输入有 图片、标签、裁剪尺寸
        img, label = self.center_crop(img, label, self.crop_size)
        img, label = self.img_transforms(img, label, self.crop_size)
        return img, label



def read_file(self,path):
    #从文件夹中读取数据
    
    files_list = os.listdir(path) #读取文件夹下所有名称列表
    '''
    print(files_list)
    返回
    ['234.png','123.png','345.png', ...]
    '''
    file_path_list =[os.path.join(path,img) for img in files_list] #路径拼接 把path跟img名拼起来
    '''
    print(file_path_list)
    返回
    ['./camvid/train/234/png','./camvid/train/123.png', ...]
    '''
    file_path_list.sort() #路径排序 sort&sorted比较详见 https://www.jianshu.com/p/7be04a3f30cd
    return file_path_list
    '''
    返回
    ['./camvid/train/123.png','./camvid/train/234/png' ...]
    '''


def center_crop(self, data,label,crop_size):
    #裁剪输入的图片&标签的大小 

    data = ttf.center_crop(data, crop_size)
    label = ttf.center_crop(label, crop_size)
    return data,label

def img_transforms(self, data,label,   )
    #hmmm.. 数值处理就放这




train_imgs = read_file(TRAIN_ROOT)
train_labels = read_file(TRAIN_LABELS)

