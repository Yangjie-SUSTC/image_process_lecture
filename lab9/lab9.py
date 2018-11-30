#!/usr/bin/python3
# -*- coding: utf-8 -*-
'''
 @File  : lab9.py
 @Author: Yangjie
 @license : Copyright(C), SUSTech,Shenzhen,China
 @Contact : yangj3@mail.sustc.edu.cn
 @Date  : 2018/11/20
 @IDE : PyCharm
 @Desc  : this code is lab9 in image and video process. it has different ops to detect edge, and canny, log ,global threshold
 '''
import cv2
import numpy as np
import math
import os


class IMG:
    def __init__(self, name, mark=None):
        self.path = 'D:\graduated\Image_process\lab\PGM_images\\'
        self.savepath = 'D:\graduated\Image_process\lab\lab_report\lab9\imagesave\\'
        self.name = name
        self.prop = '.pgm'
        self.mark = mark
        # self.img=None
        if os.path.exists(self.savepath):
            pass
        else:
            os.mkdir(self.savepath)

    def load(self):
        self.imapath = self.path + self.name + self.prop
        self.img = np.float64(cv2.imread(self.imapath, 0))
        self.save(self.img, 'original')
        return self.img

    def save(self, img, mark=None, flag=0):
        if flag:
            img = cv2.equalizeHist(np.uint8(img))
        self.mark = mark
        savepath = self.savepath + self.name + '_' + self.mark + '.jpg'
        cv2.imwrite(savepath, img)
        return img


class SEG:
    def __init__(self):
        self.name = 'Segmentation'

    def EdgeDtect(self, img, opname, part=None):
        Roberts_x = np.array([-1, 0, 0, 1])
        Roberts_y = np.array([0, -1, 1, 0])
        Prewitt_x = np.array([-1, -1, -1, 0, 0, 0, 1, 1, 1])
        Sobel_x = np.array([-1, -2, -1, 0, 0, 0, 1, 2, 1])

        Roberts_x = Roberts_x.reshape(2, 2)
        Roberts_y = Roberts_y.reshape(2, 2)
        Roberts = [Roberts_x, Roberts_y]

        Prewitt_x = Prewitt_x.reshape(3, 3)
        Prewitt_y = Prewitt_x.T
        Prewitt = [Prewitt_x, Prewitt_y]

        Sobel_x = Sobel_x.reshape(3, 3)
        Sobel_y = Sobel_x.T
        Sobel = [Sobel_x, Sobel_y]

        op_lib = {'Roberts': Roberts, 'Prewitt': Prewitt, 'Sobel': Sobel}
        op_set = op_lib[opname]
        edge = np.zeros((img.shape[0] - op_set[0].shape[0] + 1, img.shape[1] - op_set[0].shape[1] + 1))
        if part is None:
            for k, op in enumerate(op_set):
                temp = np.zeros((img.shape[0] - op.shape[0] + 1, img.shape[1] - op.shape[1] + 1))
                for x in range(img.shape[0] - op.shape[0] + 1):
                    for y in range(img.shape[1] - op.shape[1] + 1):
                        ROI = img[x:x + op.shape[0], y:y + op.shape[1]]
                        temp[x, y] = np.sum(ROI * op)
                edge += temp ** 2

            edge = np.power(edge, 1 / (k + 1))
            return (edge)

        else:
            op = op_set[part]
            for x in range(img.shape[0] - op.shape[0] + 1):
                for y in range(img.shape[1] - op.shape[1] + 1):
                    ROI = img[x:x + op.shape[0], y:y + op.shape[1]]
                    edge[x, y] = np.sum(ROI * op)
            return (edge)


    def Threshold(self, edge, thrshold, percent=1):
        if percent:  # thrshold is percentage
            thrshold = np.max(edge) * thrshold
        aim = edge > thrshold
        edge[aim] = 255
        edge[~aim] = 0
        return (edge)

    def Global_thresholding(self, img, T0):
        T = (np.max(img) + np.min(img)) / 2
        dt = 2 * T0
        while dt > T0:
            G1 = np.array(list(filter(lambda x: x < T, img.reshape(-1))))
            G2 = np.array(list(filter(lambda x: x >= T, img.reshape(-1))))
            u1 = np.mean(G1)
            u2 = np.mean(G2)
            NT = (u1 + u2) / 2
            dt = NT - T
            T = NT
        return (T)

    def Canny(self, img, T_H, T_L):

        blur = cv2.GaussianBlur(img, (3, 3), 2)
        edge_x = self.EdgeDtect(blur, 'Sobel', 0)
        edge_y = self.EdgeDtect(blur, 'Sobel', 1)
        edge = self.EdgeDtect(blur, 'Sobel')
        T_H = np.max(edge) * T_H
        T_L = np.max(edge) * T_L
        a = edge_y / edge_x
        b = np.array(list(map(math.atan, a.reshape(-1))))
        b[np.isnan(b)]=0
        angle = b.reshape(a.shape)
        # step 3 hysteresis thresholding
        g = np.zeros(edge.shape)
        hard = np.zeros(edge.shape)
        lx = []
        ly = []
        tx = int((blur.shape[0] - edge.shape[0]) / 2)
        ty = int((blur.shape[1] - edge.shape[1]) / 2)
        X, Y = np.where(edge > 0)
        temp_edge = cv2.copyMakeBorder(edge, 1, 1, 1, 1, cv2.BORDER_CONSTANT, value=0)
        for x, y in zip(X, Y):
            mag = edge[x, y]
            ang = angle[x, y]
            center_x = x + tx
            center_y = y + ty

            center = blur[center_x, center_y]
            ang_flag = np.ceil((abs(ang) - math.pi / 8) / (math.pi / 4))

            if ang_flag == 1:  # -45
                neighbor = np.array([blur[center_x - 1, center_y - 1], blur[center_x + 1, center_y + 1]])
            elif ang_flag == 2:  # vertical
                neighbor = np.array([blur[center_x, center_y - 1], blur[center_x, center_y + 1]])
            elif ang_flag == 3:  # 45
                neighbor = np.array([blur[center_x + 1, center_y - 1], blur[center_x - 1, center_y + 1]])
            else:  # horizonatl
                neighbor = np.array([blur[center_x - 1, center_y], blur[center_x + 1, center_y]])
            if (mag >= neighbor).all():
                g[x, y] = mag
                if mag >= T_H:
                    hard[x, y] = 255
                elif mag >= T_L:
                    ROI = temp_edge[x:x + 3, y:y + 3]
                    if (ROI >= T_H).any():
                        lx.append(x)
                        ly.append(y)
        temp_g = cv2.copyMakeBorder(g, 1, 1, 1, 1, cv2.BORDER_CONSTANT, value=0)
        for x, y in zip(lx, ly):
            ROI = temp_g[x:x + 3, y:y + 3]
            if (ROI >= T_H).any():
                hard[x, y] = 255
        return (blur, edge, angle, g, hard)

    def Filter(self, img, op):
        xsize = img.shape[0] - op.shape[0] + 1
        ysize = img.shape[1] - op.shape[1] + 1
        src = np.zeros((xsize, ysize))
        for x in range(xsize):
            for y in range(ysize):
                ROI = img[x:x + op.shape[0], y:y + op.shape[1]]
                src[x, y] = np.sum(ROI * op)
        return (src)

    def zeroFind(self, img, x, y, thr):
        dxset = [[-1, 1], [0, 0], [-1, 1], [-1, 1]]
        dyset = [[0, 0], [-1, 1], [-1, 1], [1, -1]]
        value = 0
        for dx, dy in zip(dxset, dyset):
            px1 = img[x + dx[0], y + dy[0]]
            px2 = img[x + dx[1], y + dy[1]]
            '''
            d = np.array([px1, px2]) - thr
            if (d[0] * d[1] < 0) and (abs(d) > thr.all()):
                value = 255
                break
            '''
            d = px1 - px2
            if (px1 * px2 < 0) and (abs(d) > thr):
                value = 255
                break
        return value

    def LOG(self, img, threshold):
        G = np.zeros((5, 5))
        G[0, 2] = 1
        G[1, 1] = 1
        G[1, 2] = 2
        G[1, 3] = 1
        G[2, 0] = 1
        G[2, 1] = 2
        G[2, 2] = -16
        G[2, 3] = 2
        G[2, 4] = 1
        G[3, 1] = 1
        G[3, 2] = 2
        G[3, 3] = 1
        G[4, 2] = 1
        Log_img = self.Filter(img, G)
        thr = np.max(Log_img) * threshold
        edge = np.zeros(Log_img.shape)
        for x in range(1, Log_img.shape[0] - 1):
            for y in range(1, Log_img.shape[1] - 1):
                edge[x, y] = self.zeroFind(Log_img, x, y, thr)
        fedge = edge[1:-1, 1:-1]
        return (G, Log_img, fedge)


if __name__ == '__main__':
    p1 = 1
    p2 = 1
    p3 = 1
    p4 = 1

    if p1:
        print('\n---Problem 1----\n')
        imname = 'large_septagon_gaussian_noise_mean_0_std_50_added'
        Op_name = ['original', 'smoothed']
        I = IMG(imname)
        oimg = I.load()
        blur = cv2.blur(oimg, (5, 5))
        I.save(blur, mark=imname + '_smoothed')
        imgset = [oimg, blur]
        S = SEG()
        for k, imname in enumerate(Op_name):
            print(imname + '....')
            img = imgset[k]

        print('\n==== Problem 1 done ====\n')

    if p2:
        print('\n---Problem 2----\n')
        imnameset = ['septagon_noisy_shaded']
        methodset = ['Partition', 'Otus']
        S = SEG()
        for k, imname in enumerate(imnameset):
            print(imname + '....')
            I = IMG(imname)
            img = I.load()
            for method in methodset:
                print('\t' + method)
                if method == 'Partition':
                    pass
                elif method == 'Otsu':
                    pass

        print('\n==== Problem 2 done ====\n')

    if p3:
        ## -------------- P3----------------
        print('\n---Problem 3----\n')
        imnameset = ['spot_shaded_text_image']
        methodset = ['moving_average_thresholding']
        S = SEG()
        for k, imname in enumerate(imnameset):
            print(imname + '....')
            I = IMG(imname)
            img = I.load()
            for method in methodset:
                print('\t' + method)

        print('\n==== Problem 3 done ====\n')

    if p4:
        ## -------------- P3----------------
        print('\n---Problem 4----\n')
        imnameset = ['defective_weld', 'noisy_region']
        methodset = ['region_growing']
        S = SEG()
        for k, imname in enumerate(imnameset):
            print(imname + '....')
            I = IMG(imname)
            img = I.load()
            for method in methodset:
                print('\t' + method)
        print('\n==== Problem 4 done ====\n')





