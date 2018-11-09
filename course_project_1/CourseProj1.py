#!/usr/bin/python3
# -*- coding: utf-8 -*-
'''
 @File  : CourseProj1.py
 @Author: Yangjie
 @license : Copyright(C), SUSTech,Shenzhen,China 
 @Contact : yangj3@mail.sustc.edu.cn
 @Date  : 2018/11/8
 @IDE : PyCharm
 @Desc  :
 Tis script developed for image and video processing course project 1

 Class:

 It define a IMG class to store several information of image, like image pathway,
 image name, image saved pathway, etc. IMG also has several functions to load, save,
 plot histogram of image.


 Functions:
 sfft(img): fft(img), then shift to center ---> fshift

 isfft(fshift): shift fft img then do ifft ---> img_back

 cal_R(x,y,img): calculate distance from center fot each pixel  ---> R

    logTrans(img): do log transformation on img  --->  cimg

    GammaCorrect(img, gamma): do log transformation on img --->  cimg

 P1(imset): main function of Problem 1, imset is image name set

    GenIma(): generate image --->   img

 P2(imname): main function of Problem 1,imname is image name

    MaskFiltering(img, mask): filter img with mask --->  rimg

    inputmask(): generate mask base on input a  --->  np.array(mask), a

 P3(imname): main function of Problem 3,imname is image name

    add_rand_noise(img): add random noise to img --->  Nimg

    meanf(img, m, n) do mean filtering on img use m* n filter ---> rimg

    P4(imname): main function of Problem 3,imname is image name
P5(imnameset): main function of Problem 1, imnameset is image name set
    it first blur image then minus original image to get mask and
    add mask to original image to unsharpen image

     IHPF(d, img): generate IHPF ---> H, R

     GHPF(d, img): generate GHPF ---> H, R

     BHPF(d, img): generate BHPF ---> H, R

     HP(img, d, n, filter_type): do high pass filter on image base on
     filter type and cut off frequency d, which it ratio of image size,
     and calculate cut off frequency D --->  rimg.real, D

 P6(imset): main function of Problem 6, imset is image name set

 Main functions are excude when  __name__=='__main__'

'''


import cv2
import numpy as np
from matplotlib import pyplot as plt


class IMG:
    def __init__(self, name, mark=None):
        self.path = 'D:\graduated\Image_process\lab\PGM_images\\'
        self.savepath = 'D:\graduated\Image_process\lab\lab_report\course_project_1\imagesave\\'
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
        plt.tick_params(labelsize=20)
        # plt.xticks([]), plt.yticks([])
        savepath = self.savepath + self.name + '_' + mark + '.jpg'
        fig.savefig(savepath, dpi=500, bbox_inches='tight')
        plt.close()
    def plthist(self, img,mark):
        font2 = {'family' :'Times New Roman', 'weight' : 'normal','size'   :25}
        img=np.uint8(img)
        fig = plt . gcf ()
        plt . hist (  img . ravel () ,256 );
        plt . xlabel ( 'Intensity ',font2)
        plt . ylabel ( 'Count ',font2)
        self.fsave (fig , mark)
        plt . close ()


def sfft(img):
    f = np.fft.fft2(img)
    fshift = np.fft.fftshift(f)
    return fshift


def isfft(fshift):
    f_ishift = np.fft.ifftshift(fshift)
    img_back = np.fft.ifft2(f_ishift)
    return img_back




def cal_R(x,y,img):
    N=img.shape[0]
    M=img.shape[1]
    u=x-M/2
    v=N/2-y
    R=np.sqrt(u**2+v**2)
    return R


##---------p1------------------------

def logTrans(img):
    cimg = 255 * np.log(img / 255 + 1) / np.log(2)
    return cimg


def GammaCorrect(img, gamma):
    cimg = 255 * np.power(img / 255, gamma)
    return cimg


def P1(imset):
    for imname in imset:
        I = IMG(imname)  # 'cameraWithNoise' 'LenaWithNoise'
        img = I.load()
        cimg = logTrans(img)
        I.save(cimg, 'logtrans');
        for gamma in [0.25, 0.5, 1, 1.5, 2]:
            cimg = GammaCorrect(img, gamma)
            I.save(cimg, 'gamma_' + str(int(gamma * 100)))


##---------p1------------------------


##-------p2--------------------------
def GenIma():
    img = np.uint8(3 * np.random.randn(256, 256) + 210)
    sqaure = np.random.randint(80, 100, (100, 100))
    img[100:200, 50:150] = sqaure
    return img


def P2(imname):
    I = IMG(imname)
    img = GenIma()
    I.save(img, 'original')
    I.plthist(img, mark='original_hist')

    for x in range(img.shape[0]):
        for y in range(img.shape[1]):
            if img[x, y] > 110:
                img[x, y] = 0

    I.save(img, 'select')
    I.plthist(img, mark='select_hist')


##-------p2--------------------------


## -----------------------------p3---------------------
def MaskFiltering(img, mask):
    m, n = mask.shape[:2]
    rimg = np.zeros((int(img.shape[0] - m + 1), int(img.shape[1] - n + 1)))
    for x in range(rimg.shape[0]):
        for y in range(rimg.shape[1]):
            ROI = img[x:x + m, y:y + n]
            val = ROI * mask
            rimg[x, y] = np.mean(val)
    return rimg


def inputmask():
    a = input('please input mask, \n split with , for items \t\t ; for rows \n')
    row = a.split(';')
    col = list(map(lambda x: x.split(','), row))
    mask = []
    for row in col:
        mask.append(list(map(float, row)))
    print('mask is: \n', mask)
    return np.array(mask), a


def P3(imname):
    I = IMG(imname)  # 'cameraWithNoise' 'LenaWithNoise'
    img = I.load()
    mask, strmask = inputmask()
    rimg = MaskFiltering(img, mask)
    I.save(rimg, mark='mymean')
    cvimg = cv2.blur(np.uint8(img), mask.shape)
    I.save(cvimg, mark='opencvmean');


##-----------------------p3----------------------


##-----------------------------  P4---------------------------------

def add_rand_noise(img):
    noise = np.random.randint(0, 100, (img.shape))
    Nimg = img + noise
    return Nimg


def meanf(img, m, n):
    rimg = np.zeros((int(img.shape[0] - m + 1), int(img.shape[1] - n + 1)))
    for x in range(rimg.shape[0]):
        for y in range(rimg.shape[1]):
            sum = 0;
            for M in range(m):
                for N in range(n):
                    sum += img[x + M, y + N]
            val = sum / (m * n)
            rimg[x, y] = val
    return rimg


def P4(imname):
    I = IMG(imname)  # 'cameraWithNoise' 'LenaWithNoise'
    img = I.load()
    I.plthist(img, 'original_hist')
    Nimg = add_rand_noise(img)
    I.save(Nimg, mark='noised')
    I.plthist(Nimg, mark='noised_hist')
    for mfsize in [3, 5]:
        fimg = meanf(Nimg, mfsize, mfsize)
        I.save(fimg, mark='denoised_' + str(mfsize) + 'x' + str(mfsize))
        I.plthist(fimg, mark='denoised_'+ str(mfsize) + 'x' + str(mfsize)+'_hist')


##-------------------- p4-----------------


##-------------------- p5---------------
def P5(imnameset):
    for imname in imnameset:
        I = IMG(imname)  # 'cameraWithNoise' 'LenaWithNoise'
        img = I.load()
        blur = cv2.GaussianBlur(img, (5, 5), 0)
        I.save(blur, 'blur');
        mask = img - blur
        I.save(mask, 'mask');
        unsharp = mask + img
        I.save(unsharp, 'unsharpe');


##-------------------- p5-------------


##-----------------------------  p6--------------------------

def IHPF(d, img):
    R = np.around(d * img.shape[1] / 2)
    H = np.ones(img.shape)
    for y in range(img.shape[0]):
        for x in range(img.shape[1]):
            r = cal_R(x, y, img)
            if r < R:
                H[y, x] = 0
    return H, R


def GHPF(d, img):
    R = np.around(d * img.shape[1] / 2)
    H = np.zeros(img.shape)
    for y in range(img.shape[0]):
        for x in range(img.shape[1]):
            r = cal_R(x, y, img)
            a = 0.5 * (r / R) ** 2
            H[y, x] = 1 - 1 / np.exp(a)
    return H, R


def BHPF(d, n, img):
    R = np.around(d * img.shape[1] / 2)
    H = np.zeros(img.shape)
    for y in range(img.shape[0]):
        for x in range(img.shape[1]):
            r = cal_R(x, y, img)
            H[y, x] = 1 - 1 / (1 + (r / R) ** (2 * n))

    return H, R


def HP(img, d, n, filter_type):
    fimg = sfft(img)

    if filter_type == 'IHPF':
        H, D = IHPF(d, img)
    if filter_type == 'GHPF':
        H, D = GHPF(d, img)
    if filter_type == 'BHPF':
        H, D = BHPF(d, n, img)
    Fimg = fimg * H
    rimg = isfft(Fimg)
    return rimg.real, D


def P6(imset):
    filter_type_set = ['IHPF', 'GHPF', 'BHPF']
    d = 0.1
    n = 2
    for imname in imset:
        for typef in filter_type_set:
            I = IMG(imname)  # 'cameraWithNoise' 'LenaWithNoise'
            img = I.load()
            HFimg, D = HP(img, d, n, typef)
            I.save(HFimg, mark=typef + '_D0_' + str(int(D)))

# ---------------  p6-----------------------------------


if __name__=='__main__':
    print('-------------problem 1--------------')
    P1( ['lena', 'bridge', 'circles','fingerprint1'])
    

    print('-------------problem 2--------------')
    P2('Genimg')
    

    print('-------------problem 3--------------')
    P3('lena')
   

    print('-------------problem 4--------------')
    P4('lena')
   

    print('-------------problem 5--------------')
    P5( ['lena', 'bridge', 'circles','fingerprint1'])
    

    print('-------------problem 6--------------')
    P6( ['lena', 'bridge', 'circles','fingerprint1'])
	
	
    print('-------------All done--------------')

