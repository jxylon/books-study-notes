import cv2 as cv
import math
import numpy as np
import pandas as pd

img_path = "D:/Documents/images/"


def canny():
    src = cv.imread(img_path + 'orange.jpg')
    cv.imshow('src', src)
    # 原图转灰度图
    gray = cv.cvtColor(src, cv.COLOR_RGB2GRAY)
    cv.imshow('gray', gray)
    # 使用3x3内核降噪
    edge = cv.blur(gray, (3, 3))
    # 运行canny算子
    edge = cv.Canny(edge, 3, 9, None, 3)
    # 填充原图
    dst = src.copy()
    for i in range(src.shape[0]):
        for j in range(src.shape[1]):
            if (edge[i, j] != 0):
                dst[i, j] = (255, 0, 0)
    # 显示
    cv.imshow('canny', dst)
    cv.waitKey(0)
    cv.destroyAllWindows()


def sobel():
    srcImg = cv.imread(img_path + '6.png')
    cv.imshow('src', srcImg)
    # X方向梯度
    grad_x = cv.Sobel(srcImg, cv.CV_16S, 1, 0, None, 3, 1, 1, cv.BORDER_DEFAULT)
    grad_abs_x = cv.convertScaleAbs(grad_x)
    cv.imshow('sobel x', grad_abs_x)
    # Y方向梯度
    grad_y = cv.Sobel(srcImg, cv.CV_16S, 0, 1, None, 3, 1, 1, cv.BORDER_DEFAULT)
    grad_abs_y = cv.convertScaleAbs(grad_y)
    cv.imshow('sobel y', grad_abs_y)
    # 合并梯度
    dst = cv.addWeighted(grad_abs_x, 0.5, grad_abs_y, 0.5, 0)
    cv.imshow('sobel all', dst)
    cv.waitKey(0)


def laplacian():
    # 显示原图
    srcImg = cv.imread(img_path + '6.png')
    cv.imshow('src', srcImg)
    # 高斯滤波降噪
    srcImg = cv.GaussianBlur(srcImg, (3, 3), 0, None, 0, cv.BORDER_DEFAULT)
    # 转换为灰度图
    grayImg = cv.cvtColor(srcImg, cv.COLOR_RGB2GRAY)
    # 使用laplace算子
    dst = cv.Laplacian(grayImg, cv.CV_16S, None, 3, 1, 0, cv.BORDER_DEFAULT)
    # 计算绝对值
    abs_dst = cv.convertScaleAbs(dst)
    # 显示效果图
    cv.imshow('Laplace', abs_dst)
    cv.waitKey(0)


def scharr():
    srcImg = cv.imread(img_path + '2.png')
    cv.imshow('src', srcImg)
    # X方向梯度
    grad_x = cv.Scharr(srcImg, cv.CV_16S, 1, 0, None, 1, 0, cv.BORDER_DEFAULT)
    grad_abs_x = cv.convertScaleAbs(grad_x)
    cv.imshow('scharr x', grad_abs_x)
    # Y方向梯度
    grad_y = cv.Scharr(srcImg, cv.CV_16S, 0, 1, None, 1, 0, cv.BORDER_DEFAULT)
    grad_abs_y = cv.convertScaleAbs(grad_y)
    cv.imshow('scharr y', grad_abs_y)
    # 合并梯度
    dst = cv.addWeighted(grad_abs_x, 0.5, grad_abs_y, 0.5, 0)
    cv.imshow('scharr all', dst)
    cv.waitKey(0)


def houghLine():
    # 原图
    srcImg = cv.imread(img_path + '1.jpg')
    cv.imshow('src', srcImg)
    # 阈值
    threshold = 100
    # 转换为灰度图，并进行边缘检测
    dstImg = cv.cvtColor(srcImg, cv.COLOR_BGR2GRAY)
    cannyImg = cv.Canny(dstImg, 50, 200, None, 3)
    cv.imshow('Cannay', cannyImg)
    # 创建滑动条
    cv.namedWindow('Hough', cv.WINDOW_AUTOSIZE)

    def on_thresChg(iVal):
        tmpImg = cannyImg.copy()
        tmpLines = cv.HoughLinesP(tmpImg, 1, math.pi / 180, iVal + 1, None, 50, 10)
        for line in tmpLines:
            cv.line(dstImg, (line[0, 0], line[0, 1]), (line[0, 2], line[0, 3]), (255, 0, 255), 1, cv.LINE_AA)
        cv.imshow('Hough', dstImg)

    cv.createTrackbar('Value', 'Hough', 100, 200, on_thresChg)

    # 初始化
    on_thresChg(threshold)

    cv.waitKey(0)
    cv.destroyAllWindows()


def houghCircles():
    # 原图
    srcImg = cv.imread(img_path + 'orange.jpg')
    dstImg = srcImg.copy()
    cv.imshow('src', srcImg)

    # 转灰度图，图像平滑
    grayImg = cv.cvtColor(srcImg, cv.COLOR_RGB2GRAY)
    midImg = cv.GaussianBlur(grayImg, (3, 3), 2, None, 2)
    # 霍夫圆变换
    circles = cv.HoughCircles(midImg, cv.HOUGH_GRADIENT, 1.5, 10, None, 200, 100, 0, 0)
    circles = circles.astype(int)
    # 依次绘图
    for circle in circles[0]:
        cv.circle(dstImg, (circle[0], circle[1]), circle[2], (255, 0, 0), 2)
    cv.imshow('hough circle', dstImg)

    cv.waitKey(0)


def remap():
    srcImg = cv.imread(img_path + '6.png')
    cv.imshow('SrcImg', srcImg)

    map_x = np.zeros((srcImg.shape[0], srcImg.shape[1]), np.float32)
    map_y = np.zeros((srcImg.shape[0], srcImg.shape[1]), np.float32)

    cv.namedWindow('DstImg', cv.WINDOW_AUTOSIZE)

    def upmap(k):
        for i in range(srcImg.shape[0]):
            for j in range(srcImg.shape[1]):
                if (k == ord('1')):  # 图片缩小一半
                    if (j > srcImg.shape[1] * 0.25 and j < srcImg.shape[1] * 0.75 and
                            i > srcImg.shape[0] * 0.25 and i < srcImg.shape[0] * 0.75):
                        map_x[i, j] = 2 * (j - srcImg.shape[1] * 0.25) + 0.5
                        map_y[i, j] = 2 * (i - srcImg.shape[0] * 0.25) + 0.5
                    else:
                        map_x[i, j] = 0
                        map_y[i, j] = 0
                elif (k == ord('2')):  # 上下倒转
                    map_x[i, j] = j
                    map_y[i, j] = srcImg.shape[0] - i
                elif (k == ord('3')):  # 左右倒转
                    map_x[i, j] = srcImg.shape[1] - j
                    map_y[i, j] = i
                elif (k == ord('4')):  # 上下左右倒转
                    map_x[i, j] = srcImg.shape[1] - j
                    map_y[i, j] = srcImg.shape[0] - i
                else:
                    pass

    while (True):
        c = cv.waitKey(0)
        if (c == 27 or c == ord('q')):
            break
        upmap(c)
        dstImp = cv.remap(srcImg, map_x, map_y, cv.INTER_LINEAR)
        cv.imshow('DstImg', dstImp)

    cv.destroyAllWindows()


def affline():
    srcImg = cv.imread(img_path + '2.png')
    cv.imshow('SrcImg', srcImg)

    srcTri = np.zeros((3, 2), np.float32)
    dstTri = np.zeros((3, 2), np.float32)

    srcTri[1] = [srcImg.shape[1] - 1, 0]
    srcTri[2] = [0, srcImg.shape[0] - 1]

    dstTri[0] = [0, srcImg.shape[0] * 0.33]
    dstTri[1] = [srcImg.shape[1] * 0.65, srcImg.shape[0] * 0.35]
    dstTri[2] = [srcImg.shape[1] * 0.15, srcImg.shape[0] * 0.6]

    wrapMat = cv.getAffineTransform(srcTri, dstTri)
    dstImg = cv.warpAffine(srcImg, wrapMat, (srcImg.shape[1], srcImg.shape[0]))

    # 缩放，旋转
    center = (dstImg.shape[1] / 2, dstImg.shape[0] / 2)
    angle = -30.0
    scale = 0.8

    rotMat = cv.getRotationMatrix2D(center, angle, scale)
    dstRote = cv.warpAffine(dstImg, rotMat, (dstImg.shape[1], dstImg.shape[0]))

    cv.imshow('Wrap Img', dstImg)
    cv.imshow('RoteImg', dstRote)

    cv.waitKey(0)
    cv.destroyAllWindows()

def hist():
    srcImg = cv.imread(img_path + '3.png')
    cv.imshow('SrcImg', srcImg)

    grayImg = cv.cvtColor(srcImg,cv.COLOR_RGB2GRAY)
    cv.imshow('GrayImg', grayImg)

    dstImg = cv.equalizeHist(grayImg)
    cv.imshow('DstImg', dstImg)

    cv.waitKey(0)
    cv.destroyAllWindows()

if __name__ == '__main__':
    sobel()
