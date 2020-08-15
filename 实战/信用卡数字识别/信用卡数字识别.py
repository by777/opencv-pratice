# encoding: utf-8
# @Time : 20/08/14 22:52
# @Author : Xu Bai
# @File : 信用卡数字识别.py
# @Desc :
# 导入工具包
from imutils import contours
import numpy as np
import argparse
import cv2

import myutils

# 设置参数
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True,
                help="path to input image")
ap.add_argument("-t", "--template", required=True,
                help="path to template OCR-A image")
args = vars(ap.parse_args())

# 指定信用卡类型
FIRST_NUMBER = {
    "3": "American Express",
    "4": "Visa",
    "5": "MasterCard",
    "6": "Discover Card"
}


def cv_show(name, img):
    cv2.imshow(name, img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


# 读取一个模板图像
img = cv2.imread(args["template"])
# 灰度图
ref = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# 二值
ref = cv2.threshold(ref, 10, 255, cv2.THRESH_BINARY_INV)[1]

# 轮廓检测#cv2.findContours()函数接受的参数为二值图，
# cv2.RETR_EXTERNAL只检测外轮廓，cv2.CHAIN_APPROX_SIMPLE只保留终点坐标
# 返回的list中每个元素都是图像中的一个轮廓
refCnts, hierarchy = cv2.findContours(ref.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
# 画所有轮廓
cv2.drawContours(img, refCnts, -1, (0, 0, 255), 3)
print("轮廓数：" + str(np.array(refCnts).shape))

# 排序、从左到右从上到下
refCnts = myutils.sort_contours(refCnts, method="left-to-right")[0]
digits = {}
# 遍历每一个轮廓 i是索引、c是轮廓
for (i, c) in enumerate(refCnts):
    # 计算外接矩形并且resize
    (x, y, w, h) = cv2.boundingRect(c)
    roi = ref[y:y + h, x:x + w]
    roi = cv2.resize(roi, (57, 88))
    # 每一个数字对应一个模板
    digits[i] = roi

# 初始化卷积核
rectKernel = cv2.getStructuringElement(cv2.MORPH_RECT, (9, 3))
sqKernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
# 读入输入图像，预处理
image = cv2.imread(args["image"])
image = myutils.resize(image, width=300)
# cv_show('image', image)
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
# 礼帽操作、突出更明亮的区域
tophat = cv2.morphologyEx(gray, cv2.MORPH_TOPHAT, rectKernel)
cv_show('tophat', tophat)
gradX = cv2.Sobel(tophat, ddepth=cv2.CV_32F, dx=1, dy=0, ksize=-1)  # ksize=-1相当于（3，3）
gradX = np.absolute(gradX)
(minVal, maxVal) = (np.min(gradX), np.max(gradX))
gradX = (255 * ((gray - minVal) / (maxVal - minVal)))
gradX = gradX.astype("uint8")

# 为了让数字区域更像一个区域块，做闭操作（先膨胀再腐蚀）
gradX = cv2.morphologyEx(gradX, cv2.MORPH_CLOSE, rectKernel)
cv_show('gradX', gradX)
# 在做二值化、阈值不清楚、适合双峰、需要把阈值参数设置为0，系统自动判断
thresh = cv2.threshold(gradX, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
cv_show('thresh', thresh)

# 把数字快之间的缝隙再闭操作
thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, sqKernel)
# 计算轮廓在处理后的图像上
threshCnts, hierarchy = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
cnts = threshCnts
cur_img = image.copy()
# 轮廓画在原图像上
cv2.drawContours(cur_img, cnts, -1, (0, 0, 255), 3)
cv_show('img', cur_img)
locs = []  # 保存有价值的区域
# 遍历轮廓、过滤掉不需要的、通过长宽比
for (i, c) in enumerate(cnts):
    # 计算轮廓的外接矩形
    (x, y, w, h) = cv2.boundingRect(c)
    ar = w / float(h)
    if 0.5 < ar < 4.0:
        if (40 < w < 55) and (10 < h < 20):
            locs.append((x, y, w, h))
locs = sorted(locs, key=lambda x: x[0])
output = []

# 遍历每一个轮廓中的数字
for (i, (gX, gY, gW, gH)) in enumerate(locs):
    groupOutput = []
    # 5是为了预留位置
    group = gray[gY - 5: gY + gH + 5, gX - 5:gX + gW + 5]
    cv_show('group', group)
    # 预处理 轮廓检测
    group = cv2.threshold(group, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
    cv_show('group', group)
    # 计算每一组的轮廓
    digitsCnts, hierarchy = cv2.findContours(group.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    digitsCnts = myutils.sort_contours(digitsCnts, method='left-to-right')[0]
    # 计算轮廓中每一组的每一个数值
    for c in digitsCnts:
        (x, y, w, h) = cv2.boundingRect(c)
        roi = group[y:y + h, x: x + w]
        roi = cv2.resize(roi, (57, 88))
        cv_show('roi', roi)
        # 计算匹配得分
        scores = []
        # 在模板中计算每一个得分
        for (digit, digitROI) in digits.items():
            result = cv2.matchTemplate(roi, digitROI, cv2.TM_CCOEFF)
            (_, score, _, _) = cv2.minMaxLoc(result)
            scores.append(score)
        # 得到最合适的数字
        groupOutput.append(str(np.argmax(scores)))
    # 画出来
    cv2.rectangle(image, (gX - 5, gY - 5), (gX + gW + 5, gY + gH + 5), (0, 0, 255), 1)
    cv2.putText(image, ''.join(groupOutput), (gX, gY - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 255), 2)
    # 第一组数计算结果（5，4，1，2）
    output.extend(groupOutput)
    print("Credit Card Type:{}".format(FIRST_NUMBER[output[0]]))
    print("Credit Card Type #：{}".format("".join(output)))
    cv2.imshow("Image", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
