import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt

img_path = "D:/Documents/images/"


def surf1():
    srcImg = cv.imread(img_path + 'src-93.jpg')
    srcImg2 = cv.imread(img_path + 'test1-93.jpg')
    cv.imshow('srcImg', srcImg2)
    detector = cv.xfeatures2d.SURF_create(400)
    # 检测关键点
    keyPt1 = detector.detect(srcImg, None)
    keyPt2 = detector.detect(srcImg2, None)
    # 绘制关键点
    img_kpt1 = cv.drawKeypoints(srcImg, keyPt1, None, (0, 0, 0), cv.DrawMatchesFlags_DRAW_RICH_KEYPOINTS)
    img_kpt2 = cv.drawKeypoints(srcImg2, keyPt2, None, (0, 0, 0), cv.DrawMatchesFlags_DEFAULT)
    # 展示图片
    cv.imshow('img', img_kpt1)
    cv.imshow('img', img_kpt2)

    cv.waitKey(0)
    cv.destroyAllWindows()


def surf2():
    srcImg = cv.imread(img_path + 'test1-93.jpg')
    srcImg2 = cv.imread(img_path + 'test2-93.jpg')
    cv.imshow('srcImg', srcImg2)

    detector = cv.xfeatures2d.SURF_create(700)
    # 检测关键点
    keypt1 = detector.detect(srcImg)
    keypt2 = detector.detect(srcImg2)
    # 计算描述符
    test = cv.xfeatures2d.SURF_create()
    descpt1 = test.compute(srcImg, keypt1)
    descpt2 = test.compute(srcImg2, keypt2)
    # 匹配两副图的描述子
    matcher = cv.BFMatcher_create()
    matches = matcher.match(descpt1[1], descpt2[1])
    # 绘制出两幅图像中匹配的关键点
    matImg = cv.drawMatches(srcImg, keypt1, srcImg2, keypt2, matches, None)

    cv.imshow('MatchImg', matImg)
    cv.waitKey(0)
    cv.destroyAllWindows()


def flann_surf():
    # 读取图片
    trainImg = cv.imread(img_path + 'book1.jpg')
    # 下采样
    trainImg = cv.pyrDown(trainImg)
    # 转换为灰度图
    grayImg = cv.cvtColor(trainImg, cv.COLOR_BGR2GRAY)
    # surf 提取特征点
    surf_Detector = cv.xfeatures2d.SURF_create(80)  # SURF实例
    trainpt = surf_Detector.detect(grayImg)  # 关键点检测
    surf_desc = cv.xfeatures2d.SURF_create()  # SURF实例
    keypt, descpt = surf_desc.compute(grayImg, trainpt)  # 计算特征向量

    # orb 提取算法
    orb_fdet = cv.ORB_create()  # ORB实例
    orb_pt = orb_fdet.detect(grayImg)  # 关键点检测
    orb_ext = cv.ORB.create()  # ORB实例
    orb_pt, orb_desc = orb_ext.compute(grayImg, orb_pt)  # 计算特征向量
    # flidx
    distParam = {"algorithm": 6, "table_number": 12, "key_size": 20,
                 "multi_probe_level": 2}
    flidx = cv.flann_Index(orb_desc, distParam)

    # flann 进行匹配
    fbMatcher = cv.FlannBasedMatcher_create()  # FLANN实例
    fbMatcher.add([descpt])  # 添加特征
    fbMatcher.train()  # 训练
    # 视频实例
    cap = cv.VideoCapture(0)
    cap.set(cv.CAP_PROP_FRAME_HEIGHT, trainImg.shape[0])

    def detectTmpimg(tmpImg):
        # 转换为灰度图
        tmpImg = cv.cvtColor(tmpImg, cv.COLOR_BGR2GRAY)
        # 特征点检测
        tmppt1 = surf_Detector.detect(tmpImg)
        # 计算特征向量
        tmppt1, tmpdesc = surf_desc.compute(tmpImg, tmppt1)
        # 匹配训练
        matches = fbMatcher.knnMatch(tmpdesc, 2)
        # 得到优秀的匹配点
        goodMatches = []
        for i in range(len(matches)):
            if (matches[i][0].distance < 0.6 * matches[i][1].distance):
                goodMatches.append(matches[i][0])

        dstImg = cv.drawMatches(tmpImg, tmppt1, trainImg, trainpt, goodMatches, None)
        cv.imshow('dstImg', dstImg)

    def orb_detectImg(tmpImg):
        # 转换为灰度图
        tmpImg = cv.cvtColor(tmpImg, cv.COLOR_BGR2GRAY)
        # 特征点检测
        tmppt1 = orb_fdet.detect(tmpImg)
        # 计算特征向量
        tmppt1, tmpdesc = orb_ext.compute(tmpImg, tmppt1)
        # KNN匹配
        indMat, matches = flidx.knnSearch(tmpdesc, 2)

        goodMatches = []
        for i in range(len(matches)):
            if (matches[i][0] < 0.6 * matches[i][1]):
                goodMatches.append(cv.DMatch(i, matches[i][0], matches[i][1]))

        dstImg = cv.drawMatches(tmpImg, tmppt1, trainImg, trainpt, goodMatches, None)
        cv.imshow('dstImg', dstImg)

    while True:
        c = cv.waitKey(1)
        if (c == ord('q')):
            break
        retval, testimg = cap.read()
        if (testimg is None):
            continue
        dtime = cv.getTickCount()
        if (c != ord('1')):
            orb_detectImg(testimg)
        else:
            pass  # detectTmpimg(testimg)

        dtime = cv.getTickCount() - dtime
        print('time : %d' % (cv.getTickFrequency() / dtime))

    cv.destroyAllWindows()


def findStuff():
    srcImg1 = cv.imread(img_path + 'test1-93.jpg')
    srcImg2 = cv.imread(img_path + 'test2-93.jpg')
    cv.imshow('srcImg1', srcImg1)
    cv.imshow('srcImg2', srcImg2)
    # 检测关键点
    detector = cv.xfeatures2d.SURF_create(400)
    kp_object = detector.detect(srcImg1, None)
    kp_scene = detector.detect(srcImg2, None)
    # 计算特征向量
    extractor = cv.xfeatures2d.SURF_create()
    des_object = extractor.compute(srcImg1, kp_object)
    des_scene = extractor.compute(srcImg2, kp_scene)
    # FLANN匹配
    matcher = cv.FlannBasedMatcher_create()
    matches = matcher.match(des_object[1], des_scene[1])
    matches = sorted(matches, key=lambda x: x.distance)
    # 找到最大距离和最小距离
    min_dst = 100
    max_dst = 0
    for i in range(len(matches)):
        dst = matches[i].distance
        min_dst = min(min_dst, dst)
        max_dst = max(max_dst, dst)
    print('最大距离为%.2f,最小距离为%.2f' % (max_dst, min_dst))
    good_matches = []
    for i in range(len(matches)):
        if (matches[i].distance < 3 * min_dst):
            good_matches.append(matches[i])
    # 绘制出匹配到的关键点
    matImg = cv.drawMatches(srcImg1, kp_object, srcImg2, kp_scene, good_matches, None)
    obj = np.float32([kp_object[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
    scene = np.float32([kp_scene[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)
    H, mask = cv.findHomography(obj, scene, cv.RANSAC)
    # 获取角点
    obj_corners = [[[0, 0], [srcImg1.shape[1], 0], [srcImg1.shape[1], srcImg1.shape[0]], [0, srcImg1.shape[0]]]]
    # 透视变换
    sce_corners = cv.perspectiveTransform(np.array(obj_corners, dtype=np.float32).reshape(-1, 1, 2), H)
    sce_corners.dtype = 'int'
    # 绘制直线
    cv.line(matImg, (sce_corners[0][0][0] + srcImg1.shape[1], sce_corners[0][0][1] + 0),
            (sce_corners[1][0][0] + srcImg1.shape[1], sce_corners[1][0][1] + 0), (255, 0, 123), 4)
    cv.line(matImg, (sce_corners[1][0][0] + srcImg1.shape[1], sce_corners[1][0][1] + 0),
            (sce_corners[2][0][0] + srcImg1.shape[1], sce_corners[2][0][1] + 0), (255, 0, 123), 4)
    cv.line(matImg, (sce_corners[2][0][0] + srcImg1.shape[1], sce_corners[2][0][1] + 0),
            (sce_corners[3][0][0] + srcImg1.shape[1], sce_corners[3][0][1] + 0), (255, 0, 123), 4)
    cv.line(matImg, (sce_corners[3][0][0] + srcImg1.shape[1], sce_corners[3][0][1] + 0),
            (sce_corners[0][0][0] + srcImg1.shape[1], sce_corners[0][0][1] + 0), (255, 0, 123), 4)
    cv.imshow('dstImg', matImg)
    cv.waitKey(0)


if __name__ == '__main__':
    findStuff()
