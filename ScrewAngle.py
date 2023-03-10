import cv2
import math

# 得到多边形的点
def getPoint(img_dir):

    img = cv2.imread(img_dir)

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # 二值化处理
    ret, thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)

    # 查找轮廓
    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    points = []

    # 遍历轮廓
    for cnt in contours:
        # 计算多边形逼近
        approx = cv2.approxPolyDP(cnt, 0.01 * cv2.arcLength(cnt, True), True)

        # 如果多边形是六边形，则绘制多边形
        # if len(approx) == 6:
        cv2.drawContours(img, [approx], 0, (0, 0, 255), 3)

        # 获取角坐标
        corners = approx.reshape((-1, 2))
        for corner in corners:
            x, y = corner
            points.append((x, y))
            # cv2.circle(img, (x, y), 5, (0, 255, 0), -1)

    # 显示结果
    # cv2.imshow("Result", img2)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    return points


# 角度计算
def Angle(a, b, c):

    # 计算三个角的余弦值
    cosA = (b ** 2 + c ** 2 - a ** 2) / (2 * b * c)
    cosB = (a ** 2 + c ** 2 - b ** 2) / (2 * a * c)
    cosC = (a ** 2 + b ** 2 - c ** 2) / (2 * a * b)
    # 将余弦值转换为角度值
    angleA = math.degrees(math.acos(cosA))
    angleB = math.degrees(math.acos(cosB))
    angleC = math.degrees(math.acos(cosC))

    # print("角A的度数：", angleA)
    # print("角B的度数：", angleB)
    # print("角C的度数：", angleC)

    return angleC


# 获得六边形的最上面的点与最小面的点，计算中心点，计算角度
# 我的思路：先获得第一个六边形的第一个点，由于角度限制，第二个六边形的对应点的x坐标偏移量不会超过一定数值
def CalAngel(point1, point2):

    y_max = 0
    y_min = 224    # 图像大小为224

    for i in range(len(point1)):
        if point1[i][1] < y_min:
            y_min = point1[i][1]
            x_min = i

        if point1[i][1] > y_max:
            y_max = point1[i][1]
            x_max = i

    # 获得第一个六边形的最上面的点与最小面的点
    # 注意：坐标原点（0,0）在左上角，所以要反着来
    x1_1, y1_1 = point1[x_min]
    x1_2, y1_2 = point1[x_max]

    # 计算第一个六边形的中心点
    cx1 = (x1_1 + x1_2) / 2
    cy1 = (y1_1 + y1_2) / 2
    # 计算三边长度
    a = math.sqrt((cx1 - x1_1) ** 2 + (cy1 - y1_1) ** 2)  # 最长边
    b = abs(cy1 - y1_1)  # 直角边1
    c = abs(cx1 - x1_1)  # 直角边2

    angleC1 = Angle(a, b, c)

    # 如果中心点x坐标大于x1说明三角形在左边
    if cx1 > x1_1:
        angleC1 = -angleC1
    # 如果中心点x坐标小于x1说明三角形在右边
    else:
        angleC1 = angleC1

    # 判断获得第二个六边形的对应点
    x2_1, x2_2, y2_1, y2_2 = 224, 224, 224, 224

    for i in range(len(point2)):
        # 距离小于20时就判断为对应点
        if abs(point2[i][0] - x1_1) < 50 and abs(point2[i][1] - y1_1) < 50:
            if abs(point2[i][0] - x1_1) < abs(x2_1 - x1_1) and abs(point2[i][1] - y1_1) < abs(y2_1 - y1_1):
                x2_1, y2_1 = point2[i]

        if abs(point2[i][0] - x1_2) < 50 and abs(point2[i][1] - y1_2) < 50:
            if abs(point2[i][0] - x1_2) < abs(x2_2 - x1_2) and abs(point2[i][1] - y1_2) < abs(y2_2 - y1_2):
                x2_2, y2_2 = point2[i]

    # 计算第二个六边形的中心点
    cx2 = (x2_1 + x2_2) / 2
    cy2 = (y2_1 + y2_2) / 2
    # 计算三边长度
    a = math.sqrt((cx2 - x2_1) ** 2 + (cy2 - y2_1) ** 2)  # 最长边
    b = abs(cy2 - y2_1)  # 直角边1
    c = abs(cx2 - x2_1)  # 直角边2

    angleC2 = Angle(a, b, c)

    # 如果中心点x坐标大于x1说明三角形在左边
    if cx2 > x2_1:
        angleC2 = -angleC2
    # 如果中心点x坐标小于x1说明三角形在右边
    else:
        angleC2 = angleC2

    return abs(angleC1 - angleC2)


if __name__ == '__main__':

    img1 = r'C:\Users\Administrator\Desktop\Creaw\outputs\000410341_4_0.bmp'
    img2 = r'C:\Users\Administrator\Desktop\Creaw\outputs\001943437_3_3.bmp'

    point1 = getPoint(img1)
    point2 = getPoint(img2)

    angle = CalAngel(point1, point2)

    print('旋转角度为：', angle)
