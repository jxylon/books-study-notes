import cv2 as cv
import numpy as np
import pandas as pd

img_path = "D:/Documents/images/"


def showImg():
	"""
	图片的载入与输出
	"""
    scrImg1 = cv.imread(img_path + "1.jpg", 2 | 4)
    scrImg2 = cv.imread(img_path + "1.jpg", 0)
    scrImg3 = cv.imread(img_path + "1.jpg", 199)
    cv.imshow("img1", scrImg1)
    cv.waitKey(0)
    cv.imshow("img2", scrImg2)
    cv.waitKey(0)
    cv.imshow("img3", scrImg3)
    cv.waitKey(0)


def createAlphaMat():
	"""
	自定义图片
	"""
    mat = np.zeros((480, 640, 4), dtype=np.uint8)
    rows = mat.shape[0]
    cols = mat.shape[1]
    for i in range(rows):
        for j in range(cols):
            mat[i][j][0] = 255
            mat[i][j][1] = int(float((cols - j) / cols) * 255)
            mat[i][j][2] = int(float((rows - i) / rows) * 255)
            mat[i][j][3] = int(0.5 * (mat[i][j][1] + mat[i][j][2]))

    cv.imwrite(img_path + "img318.png", mat)
    cv.imshow("alphaImg", mat)
    cv.waitKey(0)


def addAlpha(mat: np):
	"""
	添加通道
	"""
    rows = mat.shape[0]
    cols = mat.shape[1]
    tmplist = []
    for i in range(rows):
        for j in range(cols):
            tmp = np.append(mat[i][j], 100)
            tmplist.append(tmp)
    return np.array(tmplist)


def program319():
	"""
	3.1.9
	"""
    # 1
    girl = cv.imread(img_path + "1.jpg")
    cv.namedWindow("[1] girl")
    cv.imshow("[1] girl", girl)
    # 2
    img_sf = cv.imread(img_path + "sf.png", 199)
    img_dota2 = cv.imread(img_path + "dota2.png")
    cv.namedWindow("[2] sf")
    cv.imshow("[2] sf", img_sf)
    cv.namedWindow("[3] dota2")
    cv.imshow("[3] dota2", img_dota2)
    # ROI 感兴趣区域
    imgROI = img_dota2[200:200 + np.shape(img_sf)[0], 400:400 + np.shape(img_sf)[1]]
    # addWeighted
    cv.addWeighted(imgROI, 0.5, img_sf, 0.3, 0., imgROI)
    cv.namedWindow("[4] sf+dota2")
    cv.imshow("[4] sf+dota2", img_dota2)
    cv.imwrite(img_path + "dota2.png", img_dota2)
    cv.waitKey(0)


def on_TrackBar(g_nAlphaValueSlider):
	"""
	3.2.1
	"""
    srcImg1 = cv.imread(img_path + "4.png")
    srcImg2 = cv.imread(img_path + "5.png")
    g_dAplah = g_nAlphaValueSlider / 100
    g_dBeta = 1 - g_dAplah
    g_dstImg = srcImg1.copy()
    cv.addWeighted(srcImg1, g_dAplah, srcImg2, g_dBeta, 0.0, g_dstImg)
    cv.imshow("trackBar", g_dstImg)


def trackBar():
	"""
	3.2.1
	"""
    g_nAlphaValueSlider = 70
    cv.namedWindow("trackBar", 1)
    # 创建滑动条控件
    cv.createTrackbar("alpha 100", "trackBar", g_nAlphaValueSlider, 100, on_TrackBar)
    on_TrackBar(g_nAlphaValueSlider)
    cv.waitKey(0)


def drawPic():
	"""
	第四章
	"""
    img = np.zeros((1200, 800, 3), dtype=np.uint8)
    # 画椭圆
    cv.ellipse(img,
               (400, 400),  # 圆心 int型
               (120, 280),  # 长短轴 int型
               0,  # 旋转角度 double
               0,  # 开始角度 double
               360,  # 结束角度 double
               (255, 0, 0),  # 线条颜色
               2,  # 线宽
               8)  # 线型 4\8\16
    # 画椭圆
    cv.ellipse(img,
               (400, 400),  # 圆心 int型
               (120, 280),  # 长短轴 int型
               30,
               0,
               360,
               (255, 0, 0),
               2,
               8)
    # 画椭圆
    cv.ellipse(img,
               (400, 400),  # 圆心 int型
               (120, 280),  # 长短轴 int型
               90,
               0,
               360,
               (255, 0, 0),
               2,
               8)
    # 画椭圆
    cv.ellipse(img,
               (400, 400),  # 圆心 int型
               (120, 280),  # 长短轴 int型
               120,
               0,
               360,
               (255, 0, 0),
               2,
               8)
    # 画圆
    cv.circle(img,
              (400, 400),  # 圆心 int型
              20,  # 半径 int型
              (100, 200, 255),  # 线条颜色 （B,G,R）
              2,  # 线宽 int型  当线宽取值为【负】时，画出的圆为【实心圆】
              4)  # 线型 固定4、8、16
    # 画填充多边形
    b = np.array([
        [100, 100],
        [250, 100],
        [300, 220],
        [100, 230]
    ], dtype=np.int)

    cv.fillPoly(img,
                [b],  # 多边形点坐标 必须带[]，表示数组，且必须为int型
                (0, 255, 0),  # 填充颜色  int （B,G,R）
                8)  # 线型 固定4、8、16

    # 画非填充多边形
    d = np.array([
        [400, 100],
        [450, 100],
        [500, 220],
        [300, 230]
    ], dtype=np.int)

    cv.polylines(img,
                 [d],  # 多边形点坐标 必须带[]，表示数组，且必须为int型
                 True,  # 布尔型，True表示的是线段闭合，False表示的是仅保留线段
                 (255, 255, 0),  # 填充颜色  int （B,G,R）
                 2,  # 线宽 int型
                 8)  # 线型 固定4、8、16

    # 画线
    cv.line(img,
            (0, 0),  # 起始点坐标 int
            (1200, 1200),  # 结束点坐标 int
            (255, 255, 255),  # 线条颜色 int
            10,  # 线宽  int
            16)  # 线型 固定4、8、16

    cv.imshow("ellipse", img)
    cv.waitKey(0)


def colorReduce():
	"""
	5.1.5
	"""
    srcImg = cv.imread(img_path + '1.jpg')
    dstImg = srcImg.copy()
    timer = cv.getTickCount()
    for i in range(dstImg.shape[0]):
        for j in range(dstImg.shape[1]):
            dstImg[i, j] = 32 * (dstImg[i, j] // 32) + 32 / 2
    timer = (cv.getTickCount() - timer) / cv.getTickFrequency()
    print(timer)
    # 更推荐这种函数先计算出量化范围，遍历图像时，直接赋值
    desImg2 = srcImg.copy()
    divideWith = 32
    timer = cv.getTickCount()
    table = []
    for i in range(256):
        data = divideWith // 2 + divideWith * (i // divideWith)  # 地板除，只返回整数部分 255//10 = 25
        table.append(data)

    for i in range(desImg2.shape[0]):
        for j in range(desImg2.shape[1]):
            desImg2[i, j][0] = table[desImg2[i, j][0]]
            desImg2[i, j][1] = table[desImg2[i, j][1]]
            desImg2[i, j][2] = table[desImg2[i, j][2]]
    timer = (cv.getTickCount() - timer) / cv.getTickFrequency()
    print(timer)

    # 最推荐使用这种内部查找、替换函数LUT
    desImage3 = srcImg.copy()
    divideWith = 32
    timer = cv.getTickCount()
    # 构造对应矩阵。假设量化减少的倍数是N，则代码实现时就是简单的value/N*N，通常我们会再加上N/2以得到相邻的N的倍数的中间值，
    table = np.array([divideWith // 2 + divideWith * (i // divideWith) for i in range(256)], dtype=np.uint8)
    cv.LUT(srcImg, table, desImage3)  # 将srcImage数据，根据table里的对应关系进行查找、替换，输出到desImage3中。
    timer = (cv.getTickCount() - timer) / cv.getTickFrequency()  # 获取运行时间
    print(timer)


def smColor():
	"""
	分离、合并颜色通道
	"""
    # 分离颜色通道
    srcImg = cv.imread(img_path + '1.jpg')
    (blueImg, greenImg, redImg) = cv.split(srcImg)
    cv.imshow("blue-gray", blueImg)
    cv.imshow("green-gray", greenImg)
    cv.imshow("red-gray", redImg)
    # 合并颜色通道
    zeros = np.zeros(srcImg.shape[0:2], dtype=np.uint8)
    blueNewImg = cv.merge([blueImg, zeros, zeros])  # 颜色通道合并，得到B色图像，将G、R值填充为0。三通道图像，蓝色
    greenNewImg = cv.merge([zeros, greenImg, zeros])  # 颜色通道合并，得到G色图像，将B、R值填充为0。三通道图像，绿色
    redNewImg = cv.merge([zeros, zeros, redImg])  # 颜色通道合并，得到R色图像，将B、G值填充为0。三通道图像，红色
    srcImg = cv.merge((blueImg, greenImg, redImg))
    cv.imshow("blue", blueNewImg)
    cv.imshow("green", greenNewImg)
    cv.imshow("red", redNewImg)
    cv.imshow("merge-3color", srcImg)
    cv.waitKey(0)


srcImg = cv.imread(img_path + "1.jpg")
dstImg = np.zeros((np.shape(srcImg)[0], np.shape(srcImg)[1], 3), dtype=np.uint8)


def on_Contrast(contrast):
    bright = cv.getTrackbarPos("bright", "[Display]")
    cv.namedWindow("[Original]", 1)
    rows = np.shape(srcImg)[0]
    # cols = np.shape(srcImg)[1]
    # # 方法一：for循环遍历
    # for i in range(rows):
    #     for j in range(cols):
    #         for k in range(3):
    #             dstImg[i, j, k] = (contrast * 0.01) * (srcImg[i, j, k] + bright)
    # 方法二：dataframe.applymap遍历
    dstImgList = []
    for i in range(rows):
        tmp_df = pd.DataFrame(srcImg[i])
        tmp_df = tmp_df.applymap(lambda x: (contrast * 0.01) * (x + bright)).astype('uint8')
        dstImgList.append(tmp_df.values)
    dstImg = np.array(dstImgList)
    cv.imshow("[Original]", srcImg)
    cv.imshow("[Display]", dstImg)


def on_Bright(bright):
    contrast = cv.getTrackbarPos("contrast", "[Display]")
    cv.namedWindow("[Original]", 1)
    rows = np.shape(srcImg)[0]
    # cols = np.shape(srcImg)[1]
    # # 方法一：for循环遍历
    # for i in range(rows):
    #     for j in range(cols):
    #         for k in range(3):
    #             dstImg[i, j, k] = (contrast * 0.01) * (srcImg[i, j, k] + bright)
    # 方法二：dataframe.applymap遍历
    dstImgList = []
    for i in range(rows):
        tmp_df = pd.DataFrame(srcImg[i])
        tmp_df = tmp_df.applymap(lambda x: (contrast * 0.01) * (x + bright)).astype('uint8')
        dstImgList.append(tmp_df.values)
    dstImg = np.array(dstImgList)

    cv.imshow("[Original]", srcImg)
    cv.imshow("[Display]", dstImg)


def contrastAndBright():
	# 对比度
    g_nContrastValue = 80
    g_nBrightValue = 80
    cv.namedWindow("[Display]")
    cv.createTrackbar("contrast", "[Display]", g_nContrastValue, 300, on_Contrast)
    cv.createTrackbar("bright", "[Display]", g_nBrightValue, 300, on_Bright)
    on_Contrast(g_nContrastValue)
    on_Bright(g_nBrightValue)
    cv.waitKey(0)


def fourier():
    # 读取图像
    srcImg = cv.imread(img_path + "1.jpg", 0)
    cv.imshow("gray img", srcImg)
    # 得到DFT最优尺寸大小
    rows = np.shape(srcImg)[0]
    cols = np.shape(srcImg)[1]
    m = cv.getOptimalDFTSize(rows)
    n = cv.getOptimalDFTSize(cols)
    # 扩充图像边界，值为0
    dstImg = cv.copyMakeBorder(srcImg, 0, m - rows, 0, n - cols, cv.BORDER_CONSTANT, value=0)
    dstImg = dstImg.astype(np.float32)
    # 为傅里叶变换的结果分配空间
    planes = [dstImg, np.zeros(dstImg.shape[0:], dtype=np.float32)]
    complexI = cv.merge(planes)
    # 离散傅里叶变换
    complexI = cv.dft(complexI)
    # 将复数转换为幅值
    planes0, planes1 = cv.split(complexI)
    magImg = cv.magnitude(planes0, planes1)
    # 对数尺度
    magImg = cv.log(magImg)
    # 剪切和重分布幅度图象限
    mag_rows = np.shape(magImg)[0]
    mag_cols = np.shape(magImg)[1]
    magImg = magImg[0:mag_rows - 2, 0:mag_cols - 2]
    mag_rows -= 2
    mag_cols -= 2
    crow = int(mag_rows / 2)
    ccol = int(mag_cols / 2)
    # 左上和右下交换
    tmp = magImg[0:crow, 0:ccol]
    magImg[0:crow, 0:ccol] = magImg[crow:crow * 2, ccol:ccol * 2]
    magImg[crow:crow * 2, ccol:ccol * 2] = tmp
    # 左下和右上交换
    tmp = magImg[crow:crow * 2, 0:ccol]
    magImg[crow:crow * 2, 0:ccol] = magImg[0:crow, ccol:ccol * 2]
    magImg[0:crow, ccol:ccol * 2] = tmp
    # 归一化
    cv.normalize(magImg, magImg, 0, 1, cv.NORM_MINMAX)
    # 显示效果图
    cv.imshow("magImg", magImg)
    cv.waitKey(0)


g_srcImg = cv.imread(img_path + "1.jpg", 1)
g_dstImg1 = g_srcImg.copy()
g_dstImg2 = g_srcImg.copy()
g_dstImg3 = g_srcImg.copy()

# 空间滤波
def on_BoxBlur(n):
    g_dstImg = cv.boxFilter(g_dstImg1, -1, (n + 1, n + 1))
    cv.imshow("[1] boxBlur", g_dstImg)


def on_MeanBlur(n):
    g_dstImg = cv.blur(g_dstImg2, (n + 1, n + 1))
    cv.imshow("[2] meanBlur", g_dstImg)


def on_GaussianBlur(n):
    g_dstImg = cv.GaussianBlur(g_dstImg3, (2 * n + 1, 2 * n + 1), 0, None, 0)
    cv.imshow("[3] gaussianBlur", g_dstImg)


def on_MedianBlur(n):
    # 中值滤波
    medianImg = cv.medianBlur(srcImg, 2 * n + 1)  # 内核大小必须大于1
    cv.imshow('[4] medianBlur', medianImg)


def on_BilateraBlur(n):
    bilateraImg = cv.bilateralFilter(srcImg, n, n * 2, n / 2)
    cv.imshow("[5] bilateraBlur", bilateraImg)


def blurs():
    # 默认内核值
    g_blurValue = 6
    cv.namedWindow("[0] srcImg", 1)
    cv.imshow("[0] srcImg", g_srcImg)
    # 方框滤波
    cv.namedWindow("[1] boxBlur", 1)
    cv.createTrackbar("core:", "[1] boxBlur", g_blurValue, 40, on_BoxBlur)
    on_BoxBlur(g_blurValue)
    # 均值滤波
    cv.namedWindow("[2] meanBlur", 1)
    cv.createTrackbar("core:", "[2] meanBlur", g_blurValue, 40, on_MeanBlur)
    on_MeanBlur(g_blurValue)
    # 高斯滤波
    cv.namedWindow("[3] gaussianBlur", 1)
    cv.createTrackbar("core:", "[3] gaussianBlur", g_blurValue, 40, on_GaussianBlur)
    on_GaussianBlur(g_blurValue)
    cv.waitKey(0)


def blurs2():
    # 默认内核值
    g_blurValue = 6
    cv.imshow('[0] srcImg', srcImg)
    # 中值滤波
    cv.namedWindow('[4] medianBlur', 1)
    cv.createTrackbar('core', '[4] medianBlur', g_blurValue, 40, on_MedianBlur)
    on_MedianBlur(g_blurValue)
    # 双边滤波
    cv.namedWindow('[5] bilateraBlur', 1)
    cv.createTrackbar('core', '[5] bilateraBlur', g_blurValue, 40, on_BilateraBlur)
    on_BilateraBlur(g_blurValue)

    cv.waitKey(0)


srcImg = cv.imread(img_path + "1.jpg")
g_nTrackBarNum = 0
g_nStructElementSize = 3


def process(ksize):
    global g_nTrackBarNum
    # 自定义核
    element = cv.getStructuringElement(1, (2 * ksize + 1, 2 * ksize + 1))
    if (g_nTrackBarNum == 0):
        dstImg = cv.erode(srcImg, element)
    else:
        dstImg = cv.dilate(srcImg, element)
    cv.imshow('[1] Display', dstImg)


def on_TrackBarNumChange(n):
    global g_nTrackBarNum
    g_nTrackBarNum = n


def dilateAndErode():
    # 原始图
    cv.namedWindow('[0] srcImg')
    cv.imshow('[0] srcImg', srcImg)
    # 初次腐蚀
    cv.namedWindow('[1] Display')
    # 获取自定义核
    element = cv.getStructuringElement(cv.MORPH_RECT, (2 * g_nStructElementSize + 1, 2 * g_nStructElementSize + 1))
    dstImg = cv.erode(srcImg, element)
    cv.imshow('[1] Display', dstImg)
    # 创建滑动条
    cv.createTrackbar('erode|dilate', '[1] Display', g_nTrackBarNum, 1, on_TrackBarNumChange)
    cv.createTrackbar('size', '[1] Display', g_nStructElementSize, 10, process)
    cv.waitKey(0)


def morph_func():
    srcImg = cv.imread(img_path + '1.jpg')
    cv.namedWindow('[srcImg]')
    cv.namedWindow('[displayImg]')
    cv.imshow('[srcImg]', srcImg)
    # 定义核
    element = cv.getStructuringElement(cv.MORPH_RECT, (15, 15))
    # 开运算
    dstImg = cv.morphologyEx(srcImg, cv.MORPH_OPEN, element)
    cv.imshow('[displayImg]', dstImg)
    cv.waitKey(0)
    # 闭运算
    dstImg = cv.morphologyEx(srcImg, cv.MORPH_CLOSE, element)
    cv.imshow('[displayImg]', dstImg)
    cv.waitKey(0)
    # 形态学梯度
    dstImg = cv.morphologyEx(srcImg, cv.MORPH_GRADIENT, element)
    cv.imshow('[displayImg]', dstImg)
    cv.waitKey(0)
    # 顶帽
    dstImg = cv.morphologyEx(srcImg, cv.MORPH_TOPHAT, element)
    cv.imshow('[displayImg]', dstImg)
    cv.waitKey(0)
    # 黑帽
    dstImg = cv.morphologyEx(srcImg, cv.MORPH_BLACKHAT, element)
    cv.imshow('[displayImg]', dstImg)
    cv.waitKey(0)


# 读入原图片
srcImg2 = cv.imread(img_path + '1.jpg')
# 保留一份原图的灰度图
grayImg = cv.cvtColor(srcImg2, cv.COLOR_RGB2GRAY)
# 阈值化类型
g_nThresholdType = 3
# 阈值
g_nThresholdValue = 100


def on_ThresholdValue(g_nThresholdValue):
    global g_nThresholdType
    cv.threshold(srcImg, g_nThresholdValue, 255, g_nThresholdType, dst=dstImg)
    cv.imshow('program', dstImg)


def on_ThresholdType(g_nThresholdType):
    global g_nThresholdValue
    cv.threshold(srcImg, g_nThresholdValue, 255, g_nThresholdType, dst=dstImg)
    cv.imshow('program', dstImg)


def threshold():
    cv.namedWindow('program', cv.WINDOW_AUTOSIZE)
    cv.createTrackbar('mode', 'program', g_nThresholdType, 4, on_ThresholdType)
    cv.createTrackbar('value', 'program', g_nThresholdValue, 255, on_ThresholdValue)
    on_ThresholdType(0)
    on_ThresholdValue(0)
    cv.waitKey(0)


def pyrAndResize():
    srcImg = cv.imread(img_path + '1.jpg')
    cv.imshow('[srcImg]', srcImg)
    srcRow = np.shape(srcImg)[0]
    srcCol = np.shape(srcImg)[1]
    # pyrUp
    pyrUpImg = cv.pyrUp(srcImg, dstsize=(srcCol * 2, srcRow * 2))
    # pyrDown
    pyrDownImg = cv.pyrDown(srcImg, dstsize=(srcCol // 2, srcRow // 2))
    # resize
    resizeImg = cv.resize(srcImg, (srcCol * 2, srcRow * 2))
    cv.imshow('[pyrUpImg]', pyrUpImg)
    cv.imshow('[pyrDownImg]', pyrDownImg)
    cv.imshow('[resizeImg]', resizeImg)
    cv.waitKey(0)

if __name__ == '__main__':
    smColor()
