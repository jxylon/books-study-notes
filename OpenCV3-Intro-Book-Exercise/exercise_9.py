import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt

img_path = "D:/Documents/images/"

def histImg_demo():
    srcImg = cv.imread(img_path + 'hshist.jpg')
    cv.imshow('srcImg', srcImg)
    # 转化为HSV颜色模型
    hsvImg = cv.cvtColor(srcImg, cv.COLOR_RGB2HSV)
    # 色调等级
    hueBinNum = 30
    # 饱和度等级
    saturationBinNum = 32
    histSize = (hueBinNum, saturationBinNum)
    # 色调变化范围
    hueRanges = [0, 180]
    # 饱和度变化范围
    saturationRanges = [0, 256]
    ranges = hueRanges + saturationRanges
    channels = [0, 1]
    # 正式调用calchist
    dstHist = cv.calcHist([hsvImg], channels, None, histSize, ranges)
    minVal, maxVal, minloc, maxloc = cv.minMaxLoc(dstHist)
    scale = 10
    histImg = np.zeros((saturationBinNum * scale, hueBinNum * 10), dtype=np.uint8)

    # 直方图绘制
    intensity_list = []
    for hue in range(hueBinNum):
        for sature in range(saturationBinNum):
            binValue = float(dstHist[hue][sature])
            intensity = int(binValue * 255 / maxVal)
            intensity_list.append(intensity)
            # 正式绘制
            cv.rectangle(histImg, (hue * scale, sature * scale),
                         ((hue + 1) * scale - 1, (sature + 1) * scale - 1), intensity, cv.FILLED)
    # matplotlib 绘制直方图
    plt.hist(intensity_list, color='gray')
    plt.xticks = [0, len(intensity_list)]
    plt.show()

    cv.imshow("h-s his", histImg)
    cv.waitKey(0)


def hist1d():
    srcImg = cv.imread(img_path + 'hisD1.jpg')
    cv.imshow('srcImg', srcImg)
    grayImg = cv.cvtColor(srcImg, cv.COLOR_BGR2GRAY)
    hist = cv.calcHist([grayImg], [0], None, [256], [0, 255])
    minVal, maxVal, minLoc, maxLoc = cv.minMaxLoc(hist)
    dstImg = np.zeros((256, 256), dtype=np.uint8)
    for i in range(256):
        binVal = hist[i]
        itVal = int(binVal * 250 / maxVal)
        cv.rectangle(dstImg, (i, 256 - 1), (i, 256 - itVal), 255)
    cv.imshow('1D hist', dstImg)
    cv.waitKey(0)
    # 使用matplotlib画图
    plt.hist(grayImg.ravel(), 256, [0, 256])
    plt.title('1D hist')
    plt.show()


def histRBG():
    srcImg = cv.imread(img_path + 'hisD1.jpg')
    cv.imshow('srcImg', srcImg)
    # 使用matplotlib画图
    color = ['blue', 'green', 'red']
    for i, c in enumerate(color):
        hist = cv.calcHist([srcImg], [i], None, [256], [0, 256])
        plt.plot(hist, color=c)
        plt.xlim([0, 256])
    plt.show()
    # openCV
    hist = []
    hist.append(cv.calcHist([srcImg], [0], None, [256], [0, 256]))
    hist.append(cv.calcHist([srcImg], [1], None, [256], [0, 256]))
    hist.append(cv.calcHist([srcImg], [2], None, [256], [0, 256]))
    # 求最大值
    mmVal = cv.minMaxLoc(hist[0])
    mmVal2 = cv.minMaxLoc(hist[1])
    mmVal3 = cv.minMaxLoc(hist[2])
    maxVal = max(mmVal[1], mmVal2[1], mmVal3[1])
    # 三色直方图
    dstImg = np.zeros((256, 256 * 3, 3), dtype=np.uint8)
    color = [(255, 0, 0), (0, 255, 0), (0, 0, 255)]
    for i in range(3):
        for j in range(256):
            binVal = hist[i][j]
            ittmp = int(binVal * 255 / maxVal)
            cv.rectangle(dstImg, (i * 256 + j, 256 - 1), (i * 256 + j, 256 - ittmp), color[i])
    cv.imshow('3d hist', dstImg)
    cv.waitKey(0)


def compareHist():
    srcImg = cv.imread(img_path + 'src-93.jpg')

    srcimg2 = cv.imread(img_path + 'test1-93.jpg')
    srcimg3 = cv.imread(img_path + 'test2-93.jpg')

    cv.imshow('src1', srcImg)
    cv.imshow('src2', srcimg2)
    cv.imshow('src3', srcimg3)

    hsvimg1 = cv.cvtColor(srcImg, cv.COLOR_BGR2HSV)
    hsvimg2 = cv.cvtColor(srcimg2, cv.COLOR_BGR2HSV)
    hsvimg3 = cv.cvtColor(srcimg3, cv.COLOR_BGR2HSV)

    # 创建下半部的半身图像
    hsv_halfdown = hsvimg1[hsvimg1.shape[0] // 2:]

    # 直方图
    h_bins = 50
    s_bins = 60
    histSz = [h_bins, s_bins]
    rges = [0, 256, 0, 180]
    channels = [0, 1]

    hist_base = cv.calcHist(hsvimg1, channels, None, histSz, rges)
    hist_base = cv.normalize(hist_base, None, 0, 1, cv.NORM_MINMAX)

    hist_hfdn = cv.calcHist(hsv_halfdown, channels, None, histSz, rges)
    hist_hfdn = cv.normalize(hist_hfdn, None, 0, 1, cv.NORM_MINMAX)
    hist_test1 = cv.calcHist(hsvimg2, channels, None, histSz, rges)
    hist_test1 = cv.normalize(hist_test1, None, 0, 1, cv.NORM_MINMAX)
    hist_test2 = cv.calcHist(hsvimg3, channels, None, histSz, rges)
    hist_test2 = cv.normalize(hist_test2, None, 0, 1, cv.NORM_MINMAX)

    for i in range(6):
        cmp_method = cv.HISTCMP_CORREL + i
        d1 = cv.compareHist(hist_base, hist_base, cmp_method)
        d2 = cv.compareHist(hist_base, hist_hfdn, cmp_method)
        d3 = cv.compareHist(hist_base, hist_test1, cmp_method)
        d4 = cv.compareHist(hist_base, hist_test2, cmp_method)
        print("方法 %d 的匹配结果：\n 【基准图-基准图] %f,【基准-半身】：%f,【基准图-1】: %f,"
              "【基准图-2】: %f" % (cmp_method, d1, d2, d3, d4))

    cv.waitKey(0)
    cv.destroyAllWindows()


def backproject():
    srcImg = cv.imread(img_path + 'hu.jpg')

    hsvImg = cv.cvtColor(srcImg, cv.COLOR_BGR2HSV)
    hueImg = np.zeros(hsvImg.shape, np.uint8)
    hueImg = cv.mixChannels([hsvImg], [hueImg], [0, 0])[0]  # 取数值的第一个

    # 调整值
    def on_BinChg(v):
        # 计算直方图
        if v < 2:
            v = 2
        # sample
        hist = cv.calcHist([hueImg], [0], None, [v], [0, 180])
        hist = cv.normalize(hist, hist, 0, 255, cv.NORM_MINMAX, -1)
        # target
        backImg = cv.calcBackProject([hueImg], [0], hist, [0, 180], 1)
        cv.imshow('BackImg', backImg)
        # 绘制直方图
        histImg = np.zeros((400, 400, 3), np.uint8)
        bin_w = int(400 / v)
        for i in range(v):
            cv.rectangle(histImg, (i * bin_w, 400), ((i + 1) * bin_w, 400 - int(hist[i] * 400 / 255)),
                         (100, 223, 255), -1)

        cv.imshow('histImg', histImg)

    cv.namedWindow('srcImg', cv.WINDOW_AUTOSIZE)
    cv.createTrackbar('HueVal', 'srcImg', 0, 180, on_BinChg)

    cv.imshow('srcImg', srcImg)

    cv.waitKey(0)
    cv.destroyAllWindows()


def matchTemp():
    srcImg = cv.imread(img_path + '1.jpg')
    tmpImg = cv.imread(img_path + '1.1.png')
    cv.imshow('srcImg', srcImg)
    cv.imshow('tempImg', tmpImg)

    # 匹配和标准化
    def on_Match(g_method):
        srcBackImg = srcImg.copy()
        resImg = cv.matchTemplate(srcBackImg, tmpImg, g_method)
        rseImg = cv.normalize(resImg, resImg, 0, 1, cv.NORM_MINMAX)
        minVal, maxVal, minloc, maxloc = cv.minMaxLoc(rseImg)
        # SQDIFF 和 SQDIFF_NORMED，数值越小，匹配结果越高
        if (g_method == cv.TM_SQDIFF or g_method == cv.TM_SQDIFF_NORMED):
            matchLoc = minloc
        else:
            matchLoc = maxloc
        # 画矩形框
        cv.rectangle(srcBackImg, matchLoc, (matchLoc[0] + tmpImg.shape[0] + 1, matchLoc[1] + tmpImg.shape[1] + 1),
                     (0, 0, 255), 2, 8)
        cv.imshow('srcImg', srcBackImg)

    cv.createTrackbar('Method:', 'srcImg', 0, 5, on_Match)
    on_Match(0)
    cv.waitKey(0)


if __name__ == '__main__':
    matchTemp()
