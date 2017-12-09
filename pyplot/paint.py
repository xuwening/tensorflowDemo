import numpy as np
import matplotlib.pyplot as plt

## 设置画图大小
plt.figure(figsize=(10, 6), dpi=80)

## 生成-pi到正pi区域256个x值
x = np.linspace(-np.pi, np.pi, 256, endpoint=True)
## 生成y值
c, s = np.cos(x), np.sin(x)

## 画线
# plt.plot(x, c, label="sine")

## 画散点，s控制点的大小
plt.scatter(x, c, s=1, label="sine")

## 手动指定线的颜色，线的宽度，线的样式
plt.plot(x, s, color="gray", linewidth=1.5, linestyle="-", label="cosine")


## 添加图例，将sin、cos的颜色和label在图例中标识出来
plt.legend(loc='upper left')

t = 2 * np.pi / 3
## 画散点
plt.scatter([t, ], [np.cos(t), ], 50, color='blue')
## 画函数上的标签
plt.annotate(r'$cos(\frac{2\pi}{3})=-\frac{1}{2}$',
             xy=(t, np.cos(t)), xycoords='data',
             xytext=(-90, -50), textcoords='offset points', fontsize=16,
             arrowprops=dict(arrowstyle="->", connectionstyle="arc3,rad=.2"))

plt.scatter([t, ],[np.sin(t), ], 50, color='red')
plt.annotate(r'$sin(\frac{2\pi}{3})=\frac{\sqrt{3}}{2}$',
             xy=(t, np.sin(t)), xycoords='data',
             xytext=(+10, +30), textcoords='offset points', fontsize=16,
             arrowprops=dict(arrowstyle="->", connectionstyle="arc3,rad=.2"))

## 同时画两条线，可画多条线，前提是x坐标值一致
# plt.plot(x, c, x, s)

## 设置x，y轴的刻度区域
# plt.xlim(-3.0, 3.0)
# plt.ylim(-4.0, 4.0)

## 更灵活的指定范围
# plt.ylim(s.min()*2, s.max()*2)

## x刻度区域，比xlim更细致(-4坐标到正4坐标，总共9个刻度)
plt.xticks(np.linspace(-4.0, 4.0, 9, endpoint=True))

## 将x轴的刻度标签改成自定义π命名
# plt.xticks([-np.pi, -np.pi/2, 0, np.pi/2, np.pi],
#           [r'$-\pi$', r'$-\pi/2$', r'$0$', r'$+\pi/2$', r'$+\pi$'])

## 手动调整画布边框（默认下为x轴，左为y轴）
## 将上、右边框隐藏，
ax = plt.gca()  # 获取当前轴（相对于subplot）
ax.spines['right'].set_color('None')
ax.spines['top'].set_color('None')
# ax.xaxis.set_ticks_position('bottom')
# ax.yaxis.set_ticks_position('left')

## 将下轴和左轴移动到：数据空间 0 的位置（即x、y轴以0为中心）
ax.spines['bottom'].set_position(('data',0))
ax.spines['left'].set_position(('data',0))

## 整体控制标签的文字大小
for label in ax.get_xticklabels() + ax.get_yticklabels():
    label.set_fontsize(16)
    label.set_bbox(dict(facecolor='white', edgecolor='None', alpha=0.65))

## 画柱状图
X = np.arange(2)
Y1 = (1 - X / float(n)) * np.random.uniform(0.5, 1.0, n)
Y2 = (1 - X / float(n)) * np.random.uniform(0.5, 1.0, n)

plt.bar(X, +Y1, facecolor='#9999ff', edgecolor='white')

## 将绘图保存成文件
plt.savefig('./11.png')

plt.show()