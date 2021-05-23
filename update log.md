# 原版阈值法无除噪代码：

4月5日

~~~python
import cv2 as cv
import numpy as np

#detect number and shape of patterns
def shape_and_number(contours, ret):
    number = 0
    if(ret):
        shape = "Circle"
        number = len(contours)
    else:
        for contour in contours:
            l = cv.arcLength(contour, True)
            approx = cv.approxPolyDP(contour, 0.04*l, True)
            if len(approx) == 3:
                shape = "Triangle"
                number += 1
            elif len(approx) == 4:
                shape = "Rectangle"
                number += 1
            elif len(approx) == 5:
                shape = "Pentagon"
                number += 1
            else:
                shape = "Star"
                number += 1
    return shape, number

#detect circles
def circles_or_not(gray):
    circles = cv.HoughCircles(gray, cv.HOUGH_GRADIENT, 1, 50, param1=50, param2=50)
    try:
        if(circles.any()):
            return True
    except:
        return False

cap = cv.VideoCapture(1)
while True:
    #prepare
    retval1, frame = cap.read()
    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    retval2, thresh = cv.threshold(gray, 1, 255, type=cv.THRESH_BINARY)
    #detect
    contours, hierarchy = cv.findContours(thresh, cv.RETR_LIST, cv.CHAIN_APPROX_SIMPLE)
    ret = circles_or_not(gray)
    shape, number = shape_and_number(contours, ret)
    text = shape + ', ' + str(number)
    #同时在偏移的点放置文字以便在黑底和白底都能显示
    cv.putText(frame, text, (50,50), cv.FONT_HERSHEY_SIMPLEX, 1, (255,255,255))
    cv.putText(img, text, (51, 51), cv.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0))
    cv.imshow("cap",frame)
    if cv.waitKey(1):
        break
    
cv.destroyAllWindows()
#print(number)
#print(shape)
~~~



# 加除噪代码：

4月8日

~~~python
import cv2 as cv
import numpy as np


def shape_and_number(contours, circle_or_not):
    number = 0
    shape = 'None'
    if circle_or_not:
        shape = "Circle"
        number = len(contours)
    else:
        for contour in contours:
            l = cv.arcLength(contour, True)
            approx = cv.approxPolyDP(contour, 0.04 * l, True)
            S = cv.contourArea(contour)
            if S <= (0.00001*rows*columns):
                continue
            elif len(approx) == 3:
                shape = "Triangle"
                number += 1
            elif len(approx) == 4:
                shape = "Rectangle"
                number += 1
            elif len(approx) == 5:
                shape = "Pentagon"
                number += 1
            else:
                shape = "Star"
                number += 1
    return shape, number


def circles_or_not(gray):
    circles = cv.HoughCircles(gray, cv.HOUGH_GRADIENT, 1, 100, param1=50, param2=50)
    try:
        if (circles.any()):
            return True
    except:
        return False


cap = cv.VideoCapture(0)
while True:
    ret, img = cap.read()
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    ret, thresh = cv.threshold(gray, 173, 255, cv.THRESH_BINARY)
    #除噪
    kernel = np.ones((3,3))
    dst = cv.morphologyEx(thresh, cv.MORPH_OPEN, kernel)
    rows, columns = dst.shape
    #加边框
    dst = cv.copyMakeBorder(dst, int(0.01*rows), int(0.01*rows), int(0.01*columns), int(0.01*columns)
                            , cv.BORDER_CONSTANT, value=255)
    #
    contours, hierarchy = cv.findContours(dst, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
    contours = contours[2:]

    circle_or_not = circles_or_not(gray)
    shape, number = shape_and_number(contours, circle_or_not)
    text = shape + ', ' + str(number)
    cv.putText(img, text, (50, 50), cv.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255))
    cv.putText(img, text, (51, 51), cv.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0))

    #cv.drawContours(img, contours, 2, (0,255,0), thickness=3)
    cv.imshow("img", img)
    if not cv.waitKey(1):
        break

cv.destroyAllWindows()

~~~



# 修改了检测函数，加强了五角星的检测条件：

4月18日

```python
import cv2 as cv
import numpy as np


def shape_and_number(contours, circle_or_not, rows, columns):
    total = {'Star' : 0,
             'Triangle' : 0,
             'Rectangle' : 0,
             'Pentagon' : 0,
             'Circle' : 0,
             }
    shape = 'None'
    for contour in contours:
        l = cv.arcLength(contour, True)
        approx = cv.approxPolyDP(contour, 0.02 * l, True)
        S = cv.contourArea(contour)
        isConvex = cv.isContourConvex(approx)
        if S <= (0.00001*rows*columns):
            continue
        if not isConvex and len(approx) >= 10:
            total['Star'] += 1
        elif isConvex and len(approx) == 3:
            total['Triangle'] += 1
        elif isConvex and len(approx) == 4:
            total['Rectangle'] += 1
        elif isConvex and len(approx) == 5:
            total['Pentagon'] += 1
        elif isConvex and circle_or_not:
            total['Circle'] += 1
    number = max(total.values())
    for key, value in total.items():
        if value == number:
            shape = key
    return shape, number


def circles_or_not(gray):
    circles = cv.HoughCircles(gray, cv.HOUGH_GRADIENT, 1, 100, param1=50, param2=50)
    try:
        if (circles.any()):
            return True
    except:
        return False


img = cv.imread(r"C:\Users\10747\Desktop\1.PNG")
gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
ret, thresh = cv.threshold(gray, 62, 255, cv.THRESH_BINARY)    #阈值应为62

kernel = np.ones((3,3))
dst = cv.morphologyEx(thresh, cv.MORPH_OPEN, kernel)
rows, columns = dst.shape

dst = cv.copyMakeBorder(dst, int(0.01*rows), int(0.01*rows), int(0.01*columns), int(0.01*columns)
                        , cv.BORDER_CONSTANT, value=255)

contours, hierarchy = cv.findContours(dst, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
contours = contours[2:]

circle_or_not = circles_or_not(gray)
shape, number = shape_and_number(contours, circle_or_not, rows, columns)
text = shape + ', ' + str(number)
cv.putText(img, text, (50, 50), cv.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255))
cv.putText(img, text, (51, 51), cv.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0))

'''
for i in range(100):
    try:
        #cv.drawContours(img, contours, i, (255,0,0), thickness=3)
        l = cv.arcLength(contours[i], True)
        approx = cv.approxPolyDP(contours[i], 0.02 * l, True)
        print(cv.isContourConvex(approx))
        #print(len(approx))
        hull = cv.convexHull(approx, returnPoints=False)
        #print(hull)
        cv.drawContours(img, [np.array([approx[j[0]] for j in hull])], 0, (0, 255, 0), thickness=3)
        #cv.polylines(img, [np.array([approx[j[0]] for j in hull])], True, (0,255,0), 3)#结果与上述函数相同
        #defect = cv.convexityDefects(approx, hull)
        #print(defect)
    except:
        continue
'''

cv.imshow("DST", img)
cv.waitKey(0)
cv.destroyAllWindows()

```

# 修改了边缘提取方法（阈值法改为金字塔+canny法），增加了多个除噪条件：

4月19日

现已发现缺陷为圆形优先级过低，无法检测的缺陷

```python
import cv2 as cv
import numpy as np


def edge_capture(img):
    #用金字塔去除屏幕上的摩尔纹，用canny检测边缘
    pyr = cv.pyrDown(img)
    img_ = cv.pyrUp(pyr)
    sub = cv.subtract(img, img_)
    sub_gray = cv.cvtColor(sub, cv.COLOR_BGR2GRAY)
    dst = cv.Canny(sub_gray, 5, 13)
    return dst


def pre_noise_remove(dst):
    # 形态转换初步除噪
    kernel1 = np.ones((5, 5))
    kernel2 = np.ones((4, 4))
    dilate = cv.morphologyEx(dst, cv.MORPH_DILATE, kernel1)
    mor_dst = cv.morphologyEx(dilate, cv.MORPH_ERODE, kernel2)
    return mor_dst


def noise_remove_pro(contours):
    # 该函数返回除错后近似曲线的集合，各轮廓边数的集合，除错后检测到的多边形个数
    # 用S/l法过滤椒盐类噪声
    indexes = []  # 测试用
    centroids = []
    lengths = []
    approx_lens = []
    approxes = []
    circle = 0
    for i in range(len(contours)):
        l = cv.arcLength(contours[i], True)
        approx = cv.approxPolyDP(contours[i], 0.01 * l, True)
        S = cv.contourArea(approx)
        if S == 0:
            continue
        if S / l >= 5 and S > 10:
            indexes.append(i)
            lengths.append(l)
            M = cv.moments(approx)
            Cx = int(M['m10'] / M['m00'])
            Cy = int(M['m01'] / M['m00'])
            centroids.append((Cx, Cy))
            approx_lens.append(len(approx))
            approxes.append(approx)
            cv.drawContours(img, [approx], 0, (0, 255, 0), thickness=1)
    shape, votes = shape_detect(approxes)  # 圆也会被归类到None中
    # 用质心法去除边框效应，质心距太近的会被舍弃
    wrongs = []
    for i in range(len(lengths)):
        # 始终与后一个边框比较质心距
        x_dist = centroids[i][0] - centroids[(i + 1) % len(centroids)][0]
        y_dist = centroids[i][1] - centroids[(i + 1) % len(centroids)][1]
        dist = np.sqrt(x_dist ** 2 + y_dist ** 2)
        if dist < (lengths[i] / 8):
            # 距离过小，边框相交，与其他边框不同形状的那个即为噪音，若先检测到相同那也直接舍弃后一个
            shape_i = shape_detect([approxes[i]])
            if shape_i == shape and shape != 'None':
                wrongs.append(i + 1)
            elif shape_i != shape and shape != 'None':
                wrongs.append(i)
            else:
                # 若都为None，即可能圆被近近似成多边形，而相交的另一个是噪音，则检测第一个是否为圆形，因为二者必须排除一个
                figure = np.zeros((720, 1280), np.uint8)
                cv.drawContours(figure, contours, indexes[i], (255, 255, 255), thickness=1)
                circle_flag = circles_or_not(figure)
                if circle_flag == True:
                    wrongs.append(i+1)
                else:
                    wrongs.append(i)
            #cv.drawContours(img, contours, indexes[i], (255, 255, 255), thickness=1)
        else:
            continue
    for i in range(len(wrongs)):
        # 除错
        index = wrongs[i] - i
        del approxes[index]
        del approx_lens[index]
    if shape != 'None':
        number = len(approx_lens)   # 多边形个数，不考虑图像相框
    else:
        # 检测每个合格边框是否为圆形
        for i in range(len(indexes)):
            fig = np.zeros((720, 1280), np.uint8)
            cv.drawContours(fig, contours, indexes[i], (255, 255, 255), thickness=1)
            # cv.imshow('fig', fig)
            circle_or_not = circles_or_not(fig)
            if circle_or_not:
                circle += 1
                shape = "Circle"
        number = circle
    return shape, number


def shape_detect(approxes):
    # 用投票法选举多边形图形形状
    votes = {'Star' : 0,
             'Triangle' : 0,
             'Rectangle' : 0,
             'Pentagon' : 0,
             }
    shape = 'None'
    for approx in approxes:
        isConvex = cv.isContourConvex(approx)
        if not isConvex and len(approx) == 10:
            votes['Star'] += 1
        elif isConvex and len(approx) == 3:
            votes['Triangle'] += 1
        elif isConvex and len(approx) == 4:
            votes['Rectangle'] += 1
        elif isConvex and len(approx) == 5:
            votes['Pentagon'] += 1
    for key, value in votes.items():
        if value == max(votes.values()):
            shape = key
    return shape, votes


def circles_or_not(gray):
    # 只用来检测是否存在圆
    circles = cv.HoughCircles(gray, cv.HOUGH_GRADIENT, 2, 100, param1=100, param2=60, maxRadius=300)
    try:
        if circles.any():
            #print(circles, len(circles))
            return True
    except:
        return False


#main()
#img = cv.imread(r"C:\Users\10747\Desktop\wxhq.jpg")
cap = cv.VideoCapture(1)
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
    cv.imshow("DST", img)
    if cv.waitKey(1) & 0xFF == ord('q'):
        cap.release()
        break
cv.destroyAllWindows()

```



# 微调、化简上述代码

5月6日

```python
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

```

