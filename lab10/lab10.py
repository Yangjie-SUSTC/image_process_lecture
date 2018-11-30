#!/usr/bin/python3
# -*- coding: utf-8 -*-
'''
 @File  : lab10.py
 @Author: Yangjie
 @license : Copyright(C), SUSTech,Shenzhen,China
 @Contact : yangj3@mail.sustc.edu.cn
 @Date  : 2018/11/30
 @IDE : PyCharm
 @Desc  : this code is lab10 in image and video process. it has
 '''
import cv2
import numpy as np
import math
import os
import matplotlib.pyplot as plt
import cv2
import numpy as np
import math
import os
import matplotlib.pyplot as plt
from skimage import measure
from copy import deepcopy as dcp


class IMG:
    def __init__(self, name, mark=None):
        self.path = 'D:\graduated\Image_process\lab\PGM_images\\'
        self.savepath = 'D:\graduated\Image_process\lab\lab_report\lab10\imagesave\\'
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


    def fsave(self, fig, mark=None):  # save plot fihiure
        # plt.xticks([]), plt.yticks([])
        savepath = self.savepath + self.name + '_' + mark + '.jpg'
        fig.savefig(savepath, dpi=500, bbox_inches='tight')
        plt.close()

    def histsave(self, img, Mark):
        fig = plt.gcf()
        plt.hist(img.ravel(), 256);
        plt.xlabel('Intensity')
        plt.ylabel('Count')
        self.fsave(fig, mark=Mark + '_hist')


class SEG:
    def __init__(self):
        self.name = 'Segmentation'

    def OTSU(self, img):
        hist = cv2.calcHist([img], [0], None, [256], [0, 256]).reshape(-1, )
        TotalNum = sum(hist)
        TotalSum = np.sum(hist * range(256))
        BackCount = 0
        SumBack = 0
        SumFront = 0
        Maxsigma = 0
        for k in range(256):
            Num = hist[k]
            BackCount += Num
            FrontCount = TotalNum - BackCount
            if (FrontCount == 0):
                break
            elif (BackCount * Num == 0):
                continue
            SumBack += k * Num
            MeanBack = SumBack / BackCount
            MeanFront = (TotalSum - SumBack) / FrontCount
            sigma = BackCount * FrontCount * (MeanBack - MeanFront) ** 2
            if (sigma >= Maxsigma):
                Threshold = k
                Maxsigma = sigma
        ret, thimage = cv2.threshold(img, Threshold, 255, cv2.THRESH_BINARY)
        return (Threshold, thimage)

    def Partition(self, img,row,col,I):
        row_index=np.linspace(0,img.shape[0],row+1)
        col_index=np.linspace(0,img.shape[1],col+1)
        Thrimg=np.zeros(img.shape)
        for rowCount in range(row):
            for colCount in range(col):
                lx=int(row_index[rowCount ])
                rx=int(row_index[rowCount +1])
                ly=int(col_index[colCount ])
                ry=int(col_index[colCount +1])
                part=img[lx:rx,ly:ry]
                threshold, Thrpart=S.OTSU(part)
                Thrimg[lx:rx,ly:ry]=Thrpart
                I.histsave(part,Mark='sub_'+str(rowCount)+'_'+str(colCount)+'_Threshold_'+str(threshold))

        for rowline in row_index[1:-1]:
            img[int(rowline)]=255
        for colline in col_index[1:-1]:
            img[:,int(colline)]=255
        return (Thrimg, img)

    def moving_average_thresholding (self,img,n=20,b=0.5):
        ima=img[:]
        for k, row in enumerate(ima):
            if k%2:
                ima[k,:]=row[::-1]
        ima=ima.reshape(-1,)
        m=0
        Threshold=[]
        pada=np.hstack((np.zeros(n),ima))
        for k,p in enumerate(ima):
            m += b*(pada[k+n]-pada[k])/n
            Threshold.append(m)
        ima[ima<Threshold]=0
        ima[~(ima<Threshold)]=255
        ima=ima.reshape(img.shape)
        Threshold=np.array(Threshold )
        Threshold=Threshold.reshape(img.shape)

        for k, row in enumerate(ima):
            if k%2:
                ima[k,:]=row[::-1]
                trow=Threshold[k, :]
                Threshold[k, :] = trow[::-1]
        return (ima,Threshold)

    def EroTonePiont(self, img):
        kernel = np.ones((3, 3), np.uint8)
        onePointImg = np.zeros(img.shape)
        X = []
        Y = []
        label_image, num = measure.label(img, return_num=True)
        for k in range(1, num + 1):
            connet = np.zeros(img.shape)
            x, y = np.where(label_image == k)
            r = int(np.floor(len(x) / 2))
            X.append(x[r])
            Y.append(y[r])
        for x, y in zip(X, Y):
            onePointImg[x, y] = 255
        dilation = cv2.dilate(onePointImg, kernel, iterations=1)
        return (np.uint8(onePointImg), dilation)

    def doubleThr(self, img, T1, T2):
        src = np.zeros(img.shape)
        src[img >= T1] = 100
        src[img >= T2] = 255
        return (src)

    def connect_detec_point(self, A, EM, x, y, visual=0):
        Xk_1 = A * 0
        Xk = dcp(Xk_1)
        Xk[x, y] = 255;
        if visual:
            winName = 'find connect'
            cv2.namedWindow(winName, cv2.WINDOW_NORMAL)
        while (Xk != Xk_1).any():
            if visual:
                cv2.imshow(winName, Xk)
                cv2.waitKey(10)
            # cv2.destroyWindow(winName)
            Xk_1 = Xk
            # temp = self.dilation(Xk_1, EM)
            temp = cv2.dilate(Xk_1, EM, iterations=1)
            Xk = cv2.bitwise_and(dcp(temp), dcp(A))
        if visual:
            cv2.waitKey(100)
            cv2.destroyWindow(winName)
        return Xk

    def connect_detec(self, A, EM, Seed, visual=0):
        Connect_set = A * 0
        result = A * 0
        if visual:
            winName1 = 'Seed'
            cv2.namedWindow(winName1, cv2.WINDOW_NORMAL)
        while Seed.any():

            X, Y = np.where(Seed == 255)
            Conect = self.connect_detec_point(A, EM, X[0], Y[0], visual)
            result += Conect
            if visual:
                cv2.imshow(winName1, result)
            Conect_inv = cv2.bitwise_not(Conect)
            Seed = cv2.bitwise_and(Seed, Conect_inv)

        if visual:
            cv2.waitKey(3000)
            cv2.destroyWindow(winName1)
        return result


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
        oimg = np.uint8(I.load())
        blur = cv2.blur(oimg, (5, 5))
        I.save(blur, mark='smoothed')
        imgset = [oimg, blur]
        S = SEG()
        for k, method in enumerate(Op_name):
            print(method + '....')
            img = imgset[k]
            I.histsave(img, method)
            Threshold, thimage = S.OTSU(img)
            I.save(thimage, mark=method + '_' + str(int(Threshold)))
            ret2, th2 = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            I.save(th2, mark=method + '_opencv_' + str(int(ret2)))

        print('\n==== Problem 1 done ====\n')

    if p2:
        print('\n---Problem 2----\n')
        imnameset = ['septagon_noisy_shaded']
        methodset = ['Otsu','Partition']
        S = SEG()
        for k,imname in enumerate(imnameset):
            print(imname + '....')
            I = IMG(imname)
            img = np.uint8( I.load())
            I.histsave(img,'' )
            for method in methodset:
                print('\t' + method)
                if method == 'Otsu':
                    Threshold, thimage=S.OTSU(img)
                    I.save(thimage, mark=method+'_'+str(int(Threshold)))
                elif method == 'Partition':
                    Thrimg,lineimg=S.Partition( img,2,3,I)
                    I.save(Thrimg,mark=method)
                    I.save(lineimg, mark=method+'_line')
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
                ret2, th2 = cv2.threshold(np.uint8(img), 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
                I.save(th2, mark=method + '_opencv_' + str(int(ret2)))
                thimage,Threshold = S.moving_average_thresholding(img)
                I.save(thimage, mark=method)
                I.save(Threshold , mark=method+'_thrmat')

        print('\n==== Problem 3 done ====\n')

    if p4:
        ## -------------- P4----------------
        print('\n---Problem 4----\n')
        imnameset = ['defective_weld', 'noisy_region']  #
        methodset = ['region_growing']
        S = SEG()
        for k, imname in enumerate(imnameset):
            print(imname + '....')
            I = IMG(imname)
            img = np.uint8(I.load())
            for method in methodset:
                print('\t' + method)
                I.histsave(img, Mark='orignal')
                if imname == 'defective_weld':
                    ret2, thimg = cv2.threshold(img, 254, 255, cv2.THRESH_BINARY)
                    I.save(thimg, mark='Thr_254')
                    seed, dilation = S.EroTonePiont(thimg)
                    I.save(dilation, mark='oneP')
                    diff_img = abs(thimg - img)
                    I.save(diff_img, mark='dif_abs')
                    I.histsave(diff_img, Mark='dif_abs')
                    double_img = S.doubleThr(diff_img, 68, 126)
                    I.save(double_img, mark='double_thr')
                    ret2, thimg2 = cv2.threshold(double_img, 68, 255, cv2.THRESH_BINARY)
                    I.save(thimg2, mark='diff_thr')
                    Q = cv2.bitwise_not(np.uint8(thimg2))

                else:
                    SEED = np.zeros(img.shape)
                    SEED[abs(img - 127) > 10] = 255
                    seed, dilation = S.EroTonePiont(SEED)
                    blur = cv2.blur(img, (5, 5))
                    I.save(blur, mark='blur')
                    Q = np.zeros(img.shape, dtype=np.uint8)
                    Q[abs(blur - 127) > 1] = 255

                kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
                result = S.connect_detec(Q, kernel, seed, visual=0)
                I.save(result, mark='final')
        print('\n==== Problem 4 done ====\n')
