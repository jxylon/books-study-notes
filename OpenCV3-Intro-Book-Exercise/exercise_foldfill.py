import cv2 as cv
import copy
import random
import numpy
img_path = "D:/Documents/images/"

cv.namedWindow('[srcImg]', cv.WINDOW_AUTOSIZE)
cv.namedWindow('[dstImg]', cv.WINDOW_AUTOSIZE)

srcImg = cv.imread(img_path + 'library.jpg')
dstImg = copy.copy(srcImg)
grayImg = cv.cvtColor(srcImg, cv.COLOR_BGR2GRAY)
maskImg = numpy.zeros((srcImg.shape[0] + 2, srcImg.shape[1] + 2))

cv.imshow('[srcImg]', srcImg)

_negVal = 20  # 负值差
_posiVal = 20  # 正值差
_isClr = True  # 是否显示彩色图
_isMask = False  # 掩模

_iFillMode = 1  # 填充模式
_nConntive = 4  # 连接位数

_newMaskVal = 255


def on_Mouse(e, x, y, t, p):
    if (e != cv.EVENT_LBUTTONDOWN):
        return
    print('x: %d  y:%d' % (x, y))

    lowDiff = 0
    if (_iFillMode != 0):
        lowDiff = _negVal
    updiff = 0
    if (_iFillMode != 0):
        updiff = _posiVal

    flag = _nConntive + _newMaskVal << 8
    if (_iFillMode == 1):
        flag += cv.FLOODFILL_FIXED_RANGE

    # b,g,r
    b = random.randint(0, 255)
    g = random.randint(0, 255)
    r = random.randint(0, 255)

    if (_isClr):
        newCVal = (b, g, r)
        dst = dstImg
        area, dst, maskimg, rect = cv.floodFill(dst, None, (x, y), newCVal, (lowDiff, lowDiff, lowDiff)
                                                 , (updiff, updiff, updiff), flag)
        cv.imshow('[dstImg]', dst)
    else:
        newCVal = (r * 0.299 + g * 0.587 + b * 0.114)
        dst = grayImg
        area, dst, maskimg, rect = cv.floodFill(dst, None, (x, y), newCVal, lowDiff
                                                 , updiff, flag)
        cv.imshow('[dstImg]', dst)


def on_NegChg(iCt):
    _negVal = iCt


def on_PosiChg(iCt):
    _posiVal = iCt


cv.createTrackbar('Negative', '[dstImg]', _negVal, 255, on_NegChg)
cv.createTrackbar('Positive', '[dstImg]', _posiVal, 255, on_PosiChg)
cv.setMouseCallback('[dstImg]', on_Mouse, None)

while (True):
    # 判断是否是彩色模式
    if (_isClr):
        cv.imshow('[dstImg]', dstImg)
    else:
        cv.imshow('[dstImg]', grayImg)

    c = cv.waitKey(0)
    if (c == ord('q') or c == 27):
        print('Exit......')
        break
    elif (c == ord('1')):  # 灰度<->彩色
        if (_isClr):
            grayImg = cv.cvtColor(srcImg, cv.COLOR_BGR2GRAY)
            _isClr = False
        else:
            dstImg = copy.copy(srcImg)
            _isClr = True
    elif (c == ord('2')): # 显示隐藏掩码窗口
        if (_isMask):
            _isMask = False
            cv.destroyWindow('mask')
        else:
            _isMask = True
            cv.imshow('mask', maskImg)
    elif (c == ord('3')): # 恢复原始图像
        dstImg = copy.copy(srcImg)
        grayImg = cv.cvtColor(srcImg, cv.COLOR_BGR2GRAY)
    elif (c == ord('4')): # 空范围漫水填充
        print("空范围浸水填充...")
        _iFillMode = 0
    elif (c == ord('5')): # 渐变、固定范围浸水填充
        print("渐变、固定范围浸水填充...")
        _iFillMode = 1
    elif (c == ord('6')): # 渐变、浮动范围浸水填充
        print("渐变、浮动范围浸水填充...")
        _iFillMode = 2
    elif (c == ord('7')): # 操作标识符的低八位使用4位的连接模式
        _nConntive = 4
    elif (c == ord('8')): # 操作标识符的低八位使用8位的连接模式
        _nConntive = 8
    else:
        pass

cv.destroyAllWindows()
