import cv2 as cv
import numpy as np

img_path = "D:/Documents/images/"

def harris():
    # 加载图片
    cv.namedWindow('srcImg', cv.WINDOW_AUTOSIZE)
    srcImg = cv.imread(img_path + 'harris.jpg', cv.IMREAD_COLOR)
    # 转为灰度图
    grayImg = cv.cvtColor(srcImg, cv.COLOR_BGR2GRAY)
    cv.imshow('srcImg', srcImg)

    def on_HarrisChg(v):
        srcTmp = srcImg.copy()
        # 角点检测
        dstImg = cv.cornerHarris(grayImg, 2, 3, 0.04)
        # 归一化
        dstImg = cv.normalize(dstImg, dstImg, 0, 255, cv.NORM_MINMAX)
        # 转为8位无符号整型
        scaleImg = cv.convertScaleAbs(dstImg)
        # 绘制检测图
        for i in range(dstImg.shape[0]):
            for j in range(dstImg.shape[1]):
                if int(dstImg[i][j]) > v + 80:
                    cv.circle(srcTmp, (i, j), 5, (10, 10, 255), 1, 8)
                    cv.circle(scaleImg, (i, j), 5, (0, 10, 255), 1, 8)

        cv.imshow('srcImg', srcTmp)
        cv.imshow('scaleImg', scaleImg)

    cv.createTrackbar('Method:', 'srcImg', 30, 175, on_HarrisChg)
    on_HarrisChg(30)

    cv.waitKey(0)

def shiTomasi():
    cv.namedWindow('srcImg', cv.WINDOW_AUTOSIZE)
    srcImg = cv.imread(img_path + 'shi-tomasi.jpg', cv.IMREAD_COLOR)
    # 转换为灰度图
    grayImg = cv.cvtColor(srcImg, cv.COLOR_BGR2GRAY)
    cv.imshow('srcImg', srcImg)

    def on_ValChg(v):
        if (v < 2):
            v = 2
        qlevel = 0.01  # 最小特征值
        minDis = 10  # 最小距离
        blksz = 3  # 导数的邻域
        k = 0.04  # 系数
        # Shi-Tomasi检测
        conrPtArray = cv.goodFeaturesToTrack(grayImg, v, qlevel, minDis, None, None, blksz, False, k)
        # 绘制检测图
        dstImg = srcImg.copy()
        for i in range(len(conrPtArray)):
            cv.circle(dstImg, (conrPtArray[i][0][0], conrPtArray[i][0][1]), 4, (128, 200, 45), -1, 8)
        cv.imshow('scaleImg', dstImg)

    cv.createTrackbar('Method:', 'srcImg', 30, 500, on_ValChg)
    on_ValChg(30)
    cv.waitKey(0)
    cv.destroyAllWindows()

def subpix():
    cv.namedWindow('srcImg', cv.WINDOW_AUTOSIZE)
    srcImg = cv.imread(img_path + 'cornerSubpix.jpg', cv.IMREAD_COLOR)
    # 转为灰度图
    grayImg = cv.cvtColor(srcImg, cv.COLOR_BGR2GRAY)
    cv.imshow('srcImg', srcImg)

    def on_ValChg(v):
        if (v < 2):
            v = 2

        qlevel = 0.01  # 最小特征值
        minDis = 10  # 最小距离
        blksz = 3  # 导数的邻域
        k = 0.04  # 系数
        # Shi-Tomasi角点检测
        conrPtArray = cv.goodFeaturesToTrack(grayImg, v, qlevel, minDis, None, None, blksz, False, k)
        # 绘制Shi-Tomasi角点检测图
        dstImg = srcImg.copy()
        for i in range(len(conrPtArray)):
            cv.circle(dstImg, (conrPtArray[i][0][0], conrPtArray[i][0][1]), 4, (128, 200, 45), -1, 8)
        cv.imshow('scaleImg', dstImg)
        # 亚像素级角点检测
        winsize = (5, 5) # 搜索窗口尺寸
        zerozone = (-1, -1) # 死区尺寸
        criteria = (cv.TermCriteria_EPS + cv.TermCriteria_MAX_ITER, 40, 0.001) # 中止条件
        corNerArray = cv.cornerSubPix(grayImg, conrPtArray, winsize, zerozone, criteria)
        # 输出角点检测坐标
        for i in range(len(corNerArray)):
            print('%d: [%f,%f]' % (i, corNerArray[i][0][0], corNerArray[i][0][1]))

    cv.createTrackbar('Method:', 'srcImg', 30, 500, on_ValChg)
    on_ValChg(30)
    cv.waitKey(0)


if __name__ == '__main__':
    subpix()