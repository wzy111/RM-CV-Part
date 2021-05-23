import cv2 as cv
import numpy as np


def edge_capture(img):
    # 用拉普拉斯金字塔可去除屏幕上的摩尔纹获得边缘，再用canny检测边缘
    # 此时边缘很细，灰尘会产生很多椒盐类噪声
    pyr = cv.pyrDown(img)
    img_ = cv.pyrUp(pyr)
    sub = cv.subtract(img, img_)
    sub_gray = cv.cvtColor(sub, cv.COLOR_BGR2GRAY)
    dst = cv.Canny(sub_gray, 5, 13)
    return dst


def pre_noise_remove(dst):
    # 形态转换初步除去椒盐类噪声，并加粗边缘，内核大小已用轨迹栏调试法
    kernel1 = np.ones((5, 5))
    kernel2 = np.ones((4, 4))
    dilate = cv.morphologyEx(dst, cv.MORPH_DILATE, kernel1)
    mor_dst = cv.morphologyEx(dilate, cv.MORPH_ERODE, kernel2)
    return mor_dst


def noise_remove_pro(contours):
    # 该函数返回除错后近似曲线的集合，各轮廓边数的集合，除错后检测到的多边形个数
    # 用S/l法过滤较大的椒盐类噪声
    indexes = []    # 正确轮廓在全部轮廓中的索引表
    centroids = []
    lengths = []
    approxes = []
    for i in range(len(contours)):
        l = cv.arcLength(contours[i], True) # 周长
        approx = cv.approxPolyDP(contours[i], 0.017 * l, True)  # 此处近似函数的精度参数还可以继续优化
        S = cv.contourArea(approx)  # 面积
        if S == 0:
            continue
        if S / l >= 5 and S > 10:   # 椒盐类噪声的面积周长比较小，即可理解为直径较小
            indexes.append(i)
            lengths.append(l)
            M = cv.moments(approx)
            Cx = int(M['m10'] / M['m00'])
            Cy = int(M['m01'] / M['m00'])
            centroids.append((Cx, Cy))
            approxes.append(approx)
            cv.drawContours(img, [approx], 0, (0, 255, 0), thickness=2)  # 标记出边缘
    # 原本应是预检测形状，去除错误边缘后再检测一次形状，但后来发现加权重后直接检测就有很高的准确度了
    shape = shape_detect(approxes, indexes)  # 直接检测形状
    # 用质心法去除canny检测的边框效应，质心距太近的会被舍弃
    wrongs = 0
    for i in range(len(lengths)):
        # 始终与后一个边框比较质心距，过近会被标记
        x_dist = centroids[i][0] - centroids[(i + 1) % len(centroids)][0]
        y_dist = centroids[i][1] - centroids[(i + 1) % len(centroids)][1]
        dist = np.sqrt(x_dist ** 2 + y_dist ** 2)
        if dist < (lengths[i] / 8):
            wrongs += 1
        else:
            continue
    number = len(approxes) - wrongs  # 计算数目
    return shape, number


def shape_detect(approxes, indexes):
    # 用投票法选举多边形图形形状
    votes = {'Star': 0,
             'Triangle': 0,
             'Rectangle': 0,
             'Pentagon': 0,
             'Circle': 0,
             'None': 0
             }
    shape = 'None'
    for i in range(len(approxes)):
        approx = approxes[i]
        fig = np.zeros((720, 1280), np.uint8)   # 创建画布
        cv.drawContours(fig, contours, indexes[i], (255, 255, 255), thickness=1)    # 独立画出每个边缘
        circle_or_not = circles_or_not(fig)     # 检测每个边缘是否为圆
        isConvex = cv.isContourConvex(approx)
        if not isConvex and len(approx) == 10:
            votes['Star'] += 1
        elif isConvex and len(approx) == 3:
            votes['Triangle'] += 1
        elif isConvex and len(approx) == 4:
            votes['Rectangle'] += 1
        elif isConvex and len(approx) == 5:
            votes['Pentagon'] += 1
        # Circle和None的权重都是1/3，避免噪声的干扰
        elif circle_or_not:
            votes['Circle'] += 0.34
        else:
            votes['None'] += 0.34
    for key, value in votes.items():
        if value == max(votes.values()):
            shape = key
    return shape


def circles_or_not(gray):
    # 只用来检测是否存在圆
    circles = cv.HoughCircles(gray, cv.HOUGH_GRADIENT, 2, 100, param1=100, param2=60, maxRadius=300)
    try:
        if circles.any():
            return True
    except:
        return False


#main()
'''
# 图片检测的代码
img = cv.imread("C:/Users/10747/Desktop/2.jpg")
img = cv.resize(img, (1280, 720))
dst = edge_capture(img)
mor_dst = pre_noise_remove(dst)
contours, hierarchy = cv.findContours(mor_dst, cv.RETR_TREE, cv.CHAIN_APPROX_NONE)
shape, number = noise_remove_pro(contours)
text = str(shape) + ', ' + str(number)
cv.putText(img, text, (50, 50), cv.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255))
cv.putText(img, text, (51, 51), cv.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0))
cv.imshow("DST", img)
cv.waitKey(0)
'''
# 视频检测
cap = cv.VideoCapture(1)    # 使用外接摄像头
while True:
    ret, img = cap.read()
    img = cv.resize(img, (1280,720))
    dst = edge_capture(img)
    mor_dst = pre_noise_remove(dst)
    contours, hierarchy = cv.findContours(mor_dst, cv.RETR_TREE, cv.CHAIN_APPROX_NONE)
    shape, number = noise_remove_pro(contours)
    text = shape + ', ' + str(number)
    cv.putText(img, text, (50, 50), cv.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255))
    cv.putText(img, text, (51, 51), cv.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0))
    cv.imshow("img", img)
    if cv.waitKey(1) & 0xFF == ord('q'):
        cap.release()
        break

cv.destroyAllWindows()
