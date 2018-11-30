#!/usr/bin/python3
# -*- coding: utf-8 -*-
'''
 @File  : lab9.py
 @Author: Yangjie
 @license : Copyright(C), SUSTech,Shenzhen,China 
 @Contact : yangj3@mail.sustc.edu.cn
 @Date  : 2018/11/10
 @IDE : PyCharm
 @Desc  : 
 '''
import cv2
import numpy as np
from matplotlib import pyplot as plt
from copy import deepcopy as dcp
import pandas as pd


class IMG:
    def __init__(self, name, mark=None):
        self.path = 'D:\graduated\Image_process\lab\PGM_images\\'
        self.savepath = 'D:\graduated\Image_process\lab\lab_report\lab8\imagesave\\'
        self.name = name
        self.prop = '.pgm'
        self.mark = mark
        # self.img=None

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

    def disp(self, winName, img, sizeflag=cv2.WINDOW_NORMAL):

        img = cv2.equalizeHist(np.uint8(img))
        if sizeflag == 1:
            sizeflag = cv2.WINDOW_AUTOSIZE
        cv2.namedWindow(winName, sizeflag)
        cv2.imshow(winName, img)
        cv2.waitKey(0)
        cv2.destroyWindow(winName)
        return img

    def psave(self, img, mark=None, cb=0):  # shown image in windows and save
        fig = plt.gcf()
        plt.imshow(img, cmap='gray')
        if cb:
            plt.colorbar()
        plt.xticks([]), plt.yticks([])
        savepath = self.savepath + self.name + '_' + mark + '.jpg'
        fig.savefig(savepath, dpi=500, bbox_inches='tight')
        plt.close()

    def fsave(self, fig, mark=None):  # save plot fihiure
        # plt.tick_params(labelsize=20)
        # plt.xticks([]), plt.yticks([])
        savepath = self.savepath + self.name + '_' + mark + '.jpg'
        fig.savefig(savepath, dpi=500, bbox_inches='tight')
        plt.close()

    def plthist(self, img, mark):
        font2 = {'family': 'Times New Roman', 'weight': 'normal', 'size': 25}
        img = np.uint8(img)
        fig = plt.gcf()
        plt.hist(img.ravel(), 256);
        plt.xlabel('Intensity ', font2)
        plt.ylabel('Count ', font2)
        self.fsave(fig, mark)
        plt.close()


def sfft(img):
    f = np.fft.fft2(img)
    fshift = np.fft.fftshift(f)
    return fshift


def isfft(fshift):
    f_ishift = np.fft.ifftshift(fshift)
    img_back = np.fft.ifft2(f_ishift)
    return img_back


def cal_R(x, y, img):
    N = img.shape[0]
    M = img.shape[1]
    u = x - M / 2
    v = N / 2 - y
    R = np.sqrt(u ** 2 + v ** 2)
    return R


class Morphology:
    def __init__(self):
        self.name = 'Morphology'

    def erosion(self, img, EM):  # EM =1
        EM_inv = cv2.bitwise_not(EM)
        up_len = (EM.shape[0] - 1) // 2
        right_len = (EM.shape[1] - 1) // 2
        cmb_img = cv2.copyMakeBorder(img, up_len, up_len, right_len, right_len, cv2.BORDER_CONSTANT, value=1)
        dst = np.uint8(np.ones(cmb_img.shape) * 255)
        X, Y = np.where(img == 0)
        X = X.tolist()
        Y = Y.tolist()
        while X:
            x = X.pop()
            y = Y.pop()
            # dst[x:x + EM.shape[0], y:y + EM.shape[1] ]=cv2.bitwise_and(EM_inv,img[x:x + EM.shape[0], y:y + EM.shape[1] ])
            temp = cv2.bitwise_and(EM_inv, cmb_img[x:x + EM.shape[0], y:y + EM.shape[1]])

            old = dst[x:x + EM.shape[0], y:y + EM.shape[1]]
            dst[x:x + EM.shape[0], y:y + EM.shape[1]] = cv2.bitwise_and(temp, old)
        dst = dst[up_len:-1 * up_len, right_len:-1 * right_len]
        return np.uint8(dst)

    def dilation(self, img, EM):  # EM is 1, foreground is 1
        up_len = (EM.shape[0] - 1) // 2
        right_len = (EM.shape[1] - 1) // 2
        cmb_img = cv2.copyMakeBorder(img, up_len, up_len, right_len, right_len, cv2.BORDER_CONSTANT, value=0)
        # print(cmb_img.shape)
        dst = np.uint8(np.zeros(cmb_img.shape))
        X, Y = np.where(img == 255)
        X = X.tolist()
        Y = Y.tolist()
        while X:
            x = X.pop()
            y = Y.pop()
            # dst[x:x + EM.shape[0], y:y + EM.shape[1] ]=cv2.bitwise_or(EM,cmb_img[x:x + EM.shape[0], y:y + EM.shape[1] ])
            temp = cv2.bitwise_or(EM, cmb_img[x:x + EM.shape[0], y:y + EM.shape[1]])
            old = dst[x:x + EM.shape[0], y:y + EM.shape[1]]
            dst[x:x + EM.shape[0], y:y + EM.shape[1]] = cv2.bitwise_or(temp, old)
        dst = dst[up_len:-1 * up_len, right_len:-1 * right_len]
        return np.uint8(dst)

    def opening(self, img, EM):
        e = self.erosion(img, EM)
        dst = self.dilation(e, EM)
        return dst

    def closing(self, img, EM):
        e = self.dilation(img, EM)
        dst = self.erosion(e, EM)
        return dst

    def boundary(self, img, EM):
        dst = np.uint8(img - self.erosion(img, EM))
        return dst

    def connect_detec_point(self, A, EM, x, y,visual=0):
        Xk_1 = A * 0
        Xk = dcp(Xk_1)
        Xk[x, y] = 255;
        if visual:
            winName = 'find connect'
            cv2.namedWindow(winName, cv2.WINDOW_NORMAL)
        while (Xk != Xk_1).any():
            if visual:
                cv2.imshow(winName, Xk)
                cv2.waitKey(1)
            # |cv2.destroyWindow(winName)
            Xk_1 = Xk
            temp = self.dilation(Xk_1, EM)
            Xk = cv2.bitwise_and(dcp(temp), dcp(A))
        X, Y = np.where(Xk == 255)
        mx = np.mean(X)
        my = np.mean(Y)
        area = len(X)
        data = [X, Y, area, mx, my]
        statics = pd.DataFrame([data], columns=['X', 'Y', 'Area', 'mx', 'my'])
        if visual:
            cv2.waitKey(1)
            cv2.destroyWindow(winName)

        return (statics, Xk)

    def connect_detec(self, A, EM, Seed,visual=0):
        Connect_set = A * 0
        result = pd.DataFrame()
        if visual:
            winName1 = 'Seed'
            cv2.namedWindow(winName1, cv2.WINDOW_NORMAL)
        while Seed.any():
            if visual:
                cv2.imshow(winName1, Seed)
            X, Y = np.where(Seed == 255)
            statics, Conect = self.connect_detec_point(A, EM, X[0], Y[0])
            result = result.append(statics, ignore_index=True)
            Conect_inv = cv2.bitwise_not(Conect)
            Seed = cv2.bitwise_and(Seed, Conect_inv)
        print('Catch %d connected component' % len(result))
        if visual:
            cv2.waitKey(1)
            cv2.destroyWindow(winName1)
        return result


## -------------- P1----------------
print('\n---Problem 1----\n')
imnameset = ['noisy_fingerprint', 'noise_rectangle']
#EM1 = np.uint8(np.ones((3, 3)))*255
EM2 = np.uint8(np.ones((5, 5)))*255
EM1 = np.uint8(np.zeros((5, 5)))*255

EM1[2,:] = 255
EM1[:,2] = 255


EMset =[EM1,EM2]
for k,EM in enumerate(EMset):
    for imname in imnameset:
        I = IMG(imname)
        img = np.uint8(I.load())


        I.save(EM, 'EM_'+str(k))
        M = Morphology()

        print(imname+'\t dilation...')
        temp=M.dilation(img, EM)
        I.save(temp,'dilation'+'_EM_'+str(k))

        print(imname + '\t erosion...')
        temp = M.erosion(img, EM)
        I.save(temp, 'erosion'+'_EM_'+str(k))

        print(imname + '\t opening...')
        temp = M.opening(img, EM)
        I.save(temp, 'opening'+'_EM_'+str(k))

        print(imname + '\t closing...\n')
        temp = M.closing(img, EM)
        I.save(temp, 'closing'+'_EM_'+str(k))
print('\n==== Problem 1 done ====\n')

## -------------- P2----------------
print('\n---Problem 2----\n')
imnameset=['licoln','U']
for imname in imnameset:
        I = IMG(imname)
        img = np.uint8(I.load())

        EM = np.uint8(np.ones((3, 3)))*255
        I.save(EM, 'EM')
        M = Morphology()

        print(imname + '\t boundary...')
        temp = M.boundary(img, EM)
        I.save(temp, 'boundary')
print('\n==== Problem 2 done ====\n')

## -------------- P3----------------
print('\n---Problem 3----\n')
imnameset = ['connected']
for imname in imnameset:
    I = IMG(imname)
    img = np.uint8(I.load())
    EM = np.uint8(np.ones((3, 3)))*255
    I.save(EM, 'EM')
    M = Morphology()

    result = M.connect_detec(img, EM, img)
    result = result.sort_values(by='Area')
    #  save data frame
    result.to_csv(I.savepath + 'result.csv')
    result.index = np.arange(len(result))
    # write latex formate
    latex = result.to_latex(longtable=True, escape=False)
    with open(I.savepath + 'table.txt', 'w') as f:
        f.write(latex)

    data = result[['Area', 'mx', 'my']]
    latex = data.to_latex(longtable=True, escape=False)
    with open(I.savepath + 'data.txt', 'w') as f:
        f.write(latex)

    # get information of each area
    print('\t Generate images   ...  \n ')
    sx = list(set(data['Area'].values))
    sx.sort()
    y = []
    smx = []
    smy = []
    for s in sx:
        select = data[data['Area'] == s]
        mdata = np.mean(select)
        amount = len(select)
        y.append(amount)
        smx.append(mdata[1])
        smy.append(mdata[2])
        #  plot each area set images
        select = result[result['Area'] == s]
        cimg = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR) * 0
        for index, row in select.iterrows():
            xset = row['X']
            yset = row['Y']
            for i, j in zip(list(xset), list(yset)):
                cimg[i, j] = [255, 255, 255]
            if s > 45:
                cv2.circle(cimg, (int(row['my']), int(row['mx'])), 3, (0, 0, 255), -1)
        I.save(cimg, str(s))

    # plot images
    print('\t Plot images ... \n')
    px = list(map(lambda x: np.log(x) / np.log(2), sx))  # range(len(x))
    px = np.array(px) + range(len(sx))
    fig1 = plt.gcf()
    plt.bar(px, y, fc='r', tick_label=list(map(lambda x: str(x), sx)))
    plt.xlabel('Area', fontsize=15)
    plt.ylabel('Count', fontsize=15)
    plt.tick_params(labelsize=10)
    plt.xticks(rotation=45)
    I.fsave(fig1, mark='count')

    fig = plt.gcf()
    x = list(data['Area'].values)
    y = list(data['mx'].values)
    plt.scatter(x, y, c=x, cmap=plt.cm.Accent_r)
    plt.xlabel('Area', {'size': 15})
    plt.ylabel(r'$\bar{x}$', {'size': 15})
    plt.plot(sx, smx)
    plt.tick_params(labelsize=15)
    I.fsave(fig, mark='x')

    fig = plt.gcf()
    x = list(data['Area'].values)
    y = list(data['my'].values)
    plt.scatter(x, y, c=x, cmap=plt.cm.Accent_r)
    plt.plot(sx, smy)
    plt.xlabel('Area', {'size': 15})
    plt.ylabel(r'$\bar{y}$', {'size': 15})
    plt.tick_params(labelsize=15)
    I.fsave(fig, mark='y')

print('\n==== Problem 3 done ====\n')

## -------------- P4----------------
print('\n==== Problem 4  ====\n')
imnameset = ['bubbles_on_black_background']
for imname in imnameset:
    I = IMG(imname)
    img = np.uint8(I.load())
    plt.imshow(img, 'gray')
    particle = img[225:247, 96:118]  # clsoed bouble
    I.save(particle, 'particle')
    p_area = np.sum(particle) / 255
    EM = np.uint8(np.ones((3, 3))) * 255
    I.save(EM, 'full_EM')
    EM[0, 0] = 0
    EM[0, -1] = 0
    EM[-1, 0] = 0
    EM[-1, -1] = 0
    I.save(EM, 'EM')

    M = Morphology()

    # (a)
    print('\n particle  merged with the boundary ... \n')
    add_boundary_img = dcp(img)
    add_boundary_img[-1, :] = 255
    add_boundary_img[0, :] = 255
    add_boundary_img[:, 0] = 255
    add_boundary_img[:, -1] = 255

    statics, add_boundary = M.connect_detec_point(add_boundary_img, EM, 0, 0)
    boundary = cv2.bitwise_and(add_boundary, img)
    I.save(boundary, 'boundary')
    img2 = img - boundary
    I.save(img2, 'seperate+connect')

    print('\n seperate  particles  ... \n')
    result = M.connect_detec(img2, EM, img2)
    select = result[result['Area'] < 1.1 * p_area]
    separate = img * 0
    for index, row in select.iterrows():
        xset = row['X']
        yset = row['Y']
        for i, j in zip(list(xset), list(yset)):
            separate[i, j] = 255
    I.save(separate, 'seperate')

    print('\n connected particles  ... \n')
    connect = img2 - separate
    I.save(connect, 'connect')

    area = result['Area'].values
    plt.close()
    fig = plt.gcf()
    plt.hist(area, 80)
    plt.xlabel('Area', {'size': 15})
    plt.ylabel('Count', {'size': 15})
    plt.tick_params(labelsize=15)
    I.fsave(fig, mark='area_hist')

print('\n==== Problem 4 done ====\n')