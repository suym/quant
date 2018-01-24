#coding=utf-8
from random import choice
import numpy as np


def get_step():
    """    获得移动的步长    """
    # 分别代表正半轴和负半轴
    direction = choice([1, -1])
    # 随机选择一个距离
    #distance = choice([0, 1, 2])
    distance = choice(np.random.normal(0,0.1,10000)) 
    step = direction * distance
    return step

class RandomWalk:
    """    一个生成随机漫步数据的类    """
    # 默认漫步5000步
    def __init__(self, num_points=10000):
        self.num_points = num_points
        self.x_values = [0]
        self.y_values = [0]

    def fill_walk(self):
        """        计算随机漫步包含的所有点        """
        while len(self.x_values) < self.num_points:
            x_step = get_step()
            y_step = get_step()
            # 没有位移，跳过不取
            if x_step == 0 and y_step == 0:
                continue

            # 计算下一个点的x和y, 第一次为都0，以前的位置 + 刚才的位移 = 现在的位置
            next_x = self.x_values[-1] + x_step
            next_y = self.y_values[-1] + y_step

            self.x_values.append(next_x)
            self.y_values.append(next_y)
#开始绘制

import matplotlib.pyplot as plt


rw = RandomWalk()
rw.fill_walk()
# figure的调用在plot或者scatter之前
#plt.figure(dpi=300, figsize=(6, 4))
# 这个列表包含了各点的漫步顺序，第一个元素将是漫步的起点，最后一个元素是漫步的终点
point_numbers = list(range(rw.num_points))
# 使用颜色映射绘制颜色深浅不同的点，浅色的是先漫步的，深色是后漫步的，因此可以反应漫步轨迹
plt.scatter(rw.x_values, rw.y_values, c=point_numbers, cmap=plt.cm.Blues, edgecolors='none', s=5)
# 突出起点
plt.scatter(0, 0, c='green', edgecolors='none', s=50)
# 突出终点
plt.scatter(rw.x_values[-1], rw.y_values[-1], c='red', s=50)
plt.annotate('Starting point', xy=(0, 0), xytext=(-1, -1),arrowprops=dict(facecolor='black', shrink=0.05),)
plt.annotate('Ending point', xy=(rw.x_values[-1], rw.y_values[-1]), xytext=(rw.x_values[-1]+1, rw.y_values[-1]+1),arrowprops=dict(facecolor='black', shrink=0.05),)
# 隐藏坐标轴
plt.axes().get_xaxis().set_visible(False)
plt.axes().get_yaxis().set_visible(False)
# 指定分辨率和图像大小，单位是英寸

plt.show()


