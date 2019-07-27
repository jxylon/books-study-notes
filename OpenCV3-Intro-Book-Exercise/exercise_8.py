import cv2 as cv
import numpy as np
import random
import pandas as pd

img_path = 'D:/Documents/images/'


def contours():
    srcImg = cv.imread(img_path + 'Contour.jpg')
    cv.imshow('srcImg', srcImg)
    # 转成灰度图并均值滤波
    grayImg = cv.cvtColor(srcImg, cv.COLOR_RGB2GRAY)
    grayImg = cv.blur(grayImg, (3, 3))

    def on_ThreshChange(v):
        # 边缘检测
        cannyImg = cv.Canny(grayImg, v, v * 2, 3)
        # 寻找轮廓
        contImg, contlist, hier = cv.findContours(cannyImg, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
        # 绘制轮廓
        for i in range(len(contlist)):
            cv.drawContours(contImg, contlist, i, (255, 0, 0), 2, cv.LINE_AA, hier)
        cv.imshow('contourImg', contImg)

    # 创建窗口
    cv.namedWindow('contourImg', cv.WINDOW_AUTOSIZE)
    cv.createTrackbar('canny value', 'contourImg', 80, 255, on_ThreshChange)
    on_ThreshChange(80)
    cv.waitKey(0)


def hull():
    srcImg = cv.imread(img_path + 'convexhull.jpg')
    grayImg = cv.cvtColor(srcImg, cv.COLOR_RGB2GRAY)
    grayImg = cv.blur(grayImg, (3, 3))

    cv.namedWindow('srcImg', cv.WINDOW_AUTOSIZE)
    cv.imshow('srcImg', srcImg)

    def on_valueChange(v):
        retv, threImg = cv.threshold(grayImg, v, 255, cv.THRESH_BINARY)
        # 寻找轮廓
        threImg, contlist, hier = cv.findContours(threImg, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
        hulllist = []
        # 遍历每个轮廓，寻找凸包
        for i in range(len(contlist)):
            hulllist.append(cv.convexHull(contlist[i]))
        # 绘制轮廓
        drawImg = np.zeros((threImg.shape[0], threImg.shape[1]), np.uint8)
        for i in range(len(contlist)):
            cv.drawContours(drawImg, contlist, i, (255, 0, 0), 1, cv.LINE_AA, hier)
        cv.imshow('display', drawImg)

    cv.createTrackbar('value', 'srcImg', 50, 255, on_valueChange)
    on_valueChange(50)
    cv.waitKey(0)


def drawRect():
    cv.namedWindow('srcImg', cv.WINDOW_AUTOSIZE)
    srcImg = cv.imread(img_path + 'DrawRect.jpg')
    cv.imshow('srcImg', srcImg)

    grayImg = cv.cvtColor(srcImg, cv.COLOR_BGR2GRAY)
    grayImg = cv.blur(grayImg, (3, 3))

    type1 = 0

    def on_ValChg(iVal):
        # 阈值二值化
        retv, dstImg = cv.threshold(grayImg, iVal, 255, cv.THRESH_BINARY)
        cv.imshow('ThresImg', dstImg)
        # 寻找轮廓
        tmpImg, contours, hiery = cv.findContours(dstImg, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)

        # 查找外部矩形、圆
        poly_array = []
        rect_array = []
        circle_array = []
        for tmp in contours:
            poly_array.append(cv.approxPolyDP(tmp, 3, True))
            rect_array.append(cv.boundingRect(tmp))
            circle_array.append(cv.minEnclosingCircle(tmp))
        # 绘制图形
        outimg = np.zeros((dstImg.shape[0], dstImg.shape[1], 3))
        for j in range(len(contours)):
            color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
            if (type1 == 1):
                # 绘制多边形
                cv.drawContours(outimg, poly_array, j, color, 1, 8)
            elif (type1 == 2):
                # 绘制矩形
                cv.rectangle(outimg, (rect_array[j][0], rect_array[j][1]),
                             (rect_array[j][2] + rect_array[j][0], rect_array[j][3] + rect_array[j][1]), color, 2, 8)
            else:
                # 绘制圆 (圆心，半径)
                cv.circle(outimg, ((int)(circle_array[j][0][0]), (int)(circle_array[j][0][1])),
                          (int)(circle_array[j][1]), color, 3, 8)

        cv.imshow('DstImg', outimg)

    cv.createTrackbar('ThresHold Value:', 'srcImg', 50, 255, on_ValChg)

    on_ValChg(50)

    while (True):
        c = cv.waitKey(0)
        if (c == ord('1')):
            type1 = 1
        elif (c == ord('2')):
            type1 = 2
        elif (c == ord('3')):
            type1 = 3
        else:
            break
    cv.destroyAllWindows()


def hu():
    srcImg = cv.imread(img_path + 'hu.jpg')
    grayImg = cv.cvtColor(srcImg, cv.COLOR_RGB2GRAY)
    grayImg = cv.blur(grayImg, (3, 3))

    cv.namedWindow('srcImg', cv.WINDOW_AUTOSIZE)
    cv.imshow('srcImg', srcImg)

    def on_valueChange(v):
        # 边缘检测
        cannyImg = cv.Canny(grayImg, v, v * 2 + 3)
        # 寻找轮廓
        threImg, contlist, hier = cv.findContours(cannyImg, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
        # 查找外部矩形、圆
        mu = []
        mc = []
        for tmp in contlist:
            # 计算矩
            tmphu = cv.moments(tmp)
            mu.append(tmphu)
            # 计算中心矩
            if (tmphu['m00'] != 0.0):
                mc.append((tmphu['m10'] / tmphu['m00'], tmphu['m01'] / tmphu['m00']))
        # 绘制轮廓
        outimg = np.zeros((cannyImg.shape[0], cannyImg.shape[1], 3))
        for j in range(len(contlist)):
            color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
            outimg = cv.drawContours(outimg, contlist, j, color, 2, cv.LINE_AA, hier)
            if (j == len(mc)):
                break
            cv.circle(outimg, (int(mc[j][0]), int(mc[j][1])), 4, color, 1, 8)

        cv.imshow('DstImg', outimg)

    cv.createTrackbar('value', 'srcImg', 50, 255, on_valueChange)
    on_valueChange(50)
    cv.waitKey(0)


def watersheds():
    cv.namedWindow('srcImg', cv.WINDOW_AUTOSIZE)
    srcImg = cv.imread(img_path + 'watershed.jpg')
    # 转为灰度图
    grayImg = cv.cvtColor(srcImg, cv.COLOR_BGR2GRAY)
    maskImg = grayImg.copy()
    maskImg = maskImg * 0
    # 记录最后时刻鼠标的所在坐标
    prevpt = ()

    def on_Mouse(e, x, y, f, p):
        global prevpt
        # 鼠标不在窗口中
        if (x < 0 or x > srcImg.shape[1] or y < 0 or y > srcImg.shape[0]):
            return
        # 鼠标左键相关信息 鼠标左键经过窗口时是抬起状态 或 鼠标抬起
        if (e == cv.EVENT_LBUTTONUP or (not (f and cv.EVENT_FLAG_LBUTTON))):
            prevpt = (-1, -1)
        # 鼠标左键按下
        elif (e == cv.EVENT_LBUTTONDOWN):
            prevpt = (x, y)
        # 鼠标左键按下并移动，绘制出白色线条
        elif (e == cv.EVENT_MOUSEMOVE and (f and cv.EVENT_FLAG_LBUTTON)):
            pt = (x, y)
            if (prevpt[0] < 0):
                prevpt = pt
            cv.line(maskImg, prevpt, pt, 255, 5, 8)
            cv.line(srcImg, prevpt, pt, (255, 255, 255), 5, 8)
            prevpt = pt
            cv.imshow('srcImg', srcImg)

    cv.imshow('srcImg', srcImg)
    src2Img = srcImg.copy()
    cv.setMouseCallback('srcImg', on_Mouse)

    while (True):
        c = cv.waitKey(0)
        if (c == 27 or c == ord('q')):
            break
        elif (c == ord('2')):  # 显示原图
            srcImg = src2Img.copy()
            maskImg = maskImg * 0
            cv.imshow('srcImg', srcImg)
        elif (c == ord('1') or c == ord(' ')):  # 分水岭计算
            # 寻找掩模的轮廓
            ctImg, contList, hireList = cv.findContours(maskImg, cv.RETR_CCOMP, cv.CHAIN_APPROX_SIMPLE)
            if (len(contList) == 0):
                continue
            # 黑底白轮廓图
            tmpImg = np.zeros(maskImg.shape, dtype=np.int32)
            ct = int(0)
            comp = 100
            # 绘制轮廓
            while (True):
                cv.drawContours(tmpImg, contList, ct, comp, -1, 8, hireList, cv.INTER_MAX)
                ct = hireList[0][ct][0]
                comp += 10
                if ct < 0:
                    break
            tmpImg2 = tmpImg.copy().astype(float)
            cv.imshow('tmpImg', tmpImg2)
            # 分水岭算法
            tmpImg = cv.watershed(src2Img, tmpImg)
            waterImg = np.zeros(src2Img.shape, dtype=np.uint8)
            for i in range(tmpImg.shape[0]):
                for j in range(tmpImg.shape[1]):
                    idx = tmpImg[i][j]
                    if idx == -1:
                        waterImg[i][j] = (0, 0, 0)
                    elif (idx > comp):
                        waterImg[i][j] = (255, 255, 255)
                    else:
                        if idx == 110:
                            waterImg[i][j] = (255, 0, 0)  # 蓝色
                        elif idx == 120:
                            waterImg[i][j] = (0, 255, 0)  # 绿色
                        else:
                            waterImg[i][j] = (0, 0, 255)  # 红色

            waterImg = waterImg // 2 + src2Img // 2
            cv.imshow('dstImg', waterImg)

    cv.destroyAllWindows()


def inpaint_demo():
    srcImg = cv.imread(img_path + '1.jpg', -1)
    srcImg1 = srcImg.copy()
    grayImg = cv.cvtColor(srcImg, cv.COLOR_BGR2GRAY)
    inpaintMask = grayImg.copy()
    inpaintMask = inpaintMask * 0
    cv.imshow('srcImg', srcImg1)
    prepoint = ()

    def on_Mouse(e, x, y, f, p):
        global prepoint
        # 鼠标不在窗口中
        if (x < 0 or x > srcImg1.shape[1] or y < 0 or y > srcImg1.shape[0]):
            return
        # 鼠标左键相关信息 鼠标左键经过窗口时是抬起状态 或 鼠标抬起
        if (e == cv.EVENT_LBUTTONUP or (not (f and cv.EVENT_FLAG_LBUTTON))):
            prepoint = (-1, -1)
        # 鼠标左键按下
        elif (e == cv.EVENT_LBUTTONDOWN):
            prepoint = (x, y)
        # 鼠标左键按下并移动，绘制出白色线条
        elif (e == cv.EVENT_MOUSEMOVE and (f and cv.EVENT_FLAG_LBUTTON)):
            pt = (x, y)
            if (prepoint[0] < 0):
                prepoint = pt
            cv.line(inpaintMask, prepoint, pt, 255, 5, 8)
            cv.line(srcImg1, prepoint, pt, (255, 255, 255), 5, 8)
            prepoint = pt
            cv.imshow('srcImg', srcImg1)

    cv.setMouseCallback('srcImg', on_Mouse, 0)
    while (True):
        c = cv.waitKey(0)
        if (c == 27 or c == ord('q')):
            break
        elif (c == ord('2')):
            inpaintMask = inpaintMask * 0
            srcImg1 = srcImg.copy()
            cv.imshow('srcImg', srcImg1)
        elif (c == ord('1') or c == ord(' ')):
            inpaintImg = cv.inpaint(srcImg1, inpaintMask, 3, cv.INPAINT_TELEA)
            cv.imshow('inpainImg', inpaintImg)


if __name__ == '__main__':
    drawRect()
