import pickle

import numpy as np
import matplotlib
import matplotlib.pyplot as plt

with open('../rest14_out.pkl', 'rb') as f:
    out = pickle.load(f)
vegetables = ["service", "food"]
farmers = ["great", "service", "but", "the", "food", "was", "dreadful", "!"]

harvest = np.array(out)


def heatmap(data, row_labels, col_labels, ax=None,
            cbar_kw={}, cbarlabel="", **kwargs):
    """
    从一个numpy数组和两个标签列表创建一个热图。

    data
         形状为（N，M）的2D numpy数组。
     row_labels
         长度为N且带有行标签的列表或数组。
     col_labels
         长度为M的列表或数组，带有列的标签。
     ax
         绘制热图的`matplotlib.axes.Axes`实例。 如果
         未提供，请使用当前轴或创建一个新轴。 (可选的。)
     cbar_kw
         带有`matplotlib.Figure.colorbar`参数的字典。 可选的。
     cbarlabel
         颜色条的标签。 可选的。
     **kwargs
         所有其他参数都转发给“imshow”。
    """

    if not ax:
        ax = plt.gca()
    # 如果不在ax，挪动坐标轴

    im = ax.imshow(data, **kwargs)
    # 画图
    # 创造彩条
    cbar = ax.figure.colorbar(im, ax=ax, fraction=0.02, **cbar_kw)
    '''
    im:一个可以映射颜色的对象
    ax=ax:指示im得到的对象在哪里展示（整体）
    '''
    cbar.ax.set_ylabel(cbarlabel, rotation=-90, va="bottom")

    '''
    设置y轴标签
    '''

    ax.set_xticks(np.arange(data.shape[1]))
    ax.set_yticks(np.arange(data.shape[0]))
    # 设置x,y轴刻度间隔
    ax.set_xticklabels(col_labels)
    ax.set_yticklabels(row_labels)
    # 设置横，纵轴

    # 让水平轴标签显示在顶部
    ax.tick_params(top=False, bottom=True,
                   labeltop=False, labelbottom=True)

    # 旋转刻度线标签并设置其对齐方式。
    plt.setp(ax.get_xticklabels(), rotation=-30, ha="left",
             rotation_mode="anchor")

    # 关闭spines并创建白色网格。
    # spines是连接轴刻度标记的线，而且标明了数据区域的边界
    for edge, spine in ax.spines.items():
        spine.set_visible(False)

    ax.set_xticks(np.arange(data.shape[1] + 1) - .5, minor=True)
    ax.set_yticks(np.arange(data.shape[0] + 1) - .5, minor=True)
    ##设置x,y轴刻度间隔
    ax.grid(which="minor", color="w", linestyle='-', linewidth=3)
    # 设置边框主刻度线，颜色为白色，线条格式为'-',线的宽度为3
    ax.tick_params(which="minor", bottom=False, left=False)
    # 设置主刻度线，参数bottom, top, left, right的值为布尔值，分别代表设置绘图区四个边框线上的的刻度线是否显示

    return im, cbar


def annotate_heatmap(im, data=None, valfmt="{x:.2f}",
                     textcolors=["black", "white"],
                     threshold=None, **textkw):
    '''
     im
         要标记的AxesImage。
     data
         用于注释的数据。 如果为None，则使用图像数据。 （可选的。）
     valfmt
         热图内注释的格式。 这应该
         使用字符串格式方法，例如 “ $ {x：.2f}”，或成为
         `matplotlib.ticker.Formatter`。 （可选的。）
     textcolors
         两种颜色规格的列表或数组。 第一个代表值低于阈值，第二个代表高于阈值的值。 （可选的。）
     threshold
         以数据单位表示的值，根据该值，textcolors中的颜色是
         应用。 如果为None（默认），则将颜色图的中间用作
         分离。（ 可选的。）
     **kwargs
         所有其他参数都转发给用于创建的每个`text`调用。
         文字标签。
    '''

    if not isinstance(data, (list, np.ndarray)):
        data = im.get_array()
    # 保证data是一个list类型

    # 将阈值标准化为图像颜色范围。
    if threshold is not None:
        threshold = im.norm(threshold)
    else:
        threshold = im.norm(data.max()) / 2.

    # 将默认对齐方式设置为居中，但允许将其设置为居中
    # 被textkw覆盖。
    kw = dict(horizontalalignment="center",
              verticalalignment="center")
    kw.update(textkw)

    # 获取格式化程序（如果提供了字符串）
    if isinstance(valfmt, str):
        valfmt = matplotlib.ticker.StrMethodFormatter(valfmt)
    # 给热力图标注文本设置格式

    # 遍历数据并为每个“pixel”创建一个“Text”。
    # 根据数据更改文本的颜色。
    texts = []
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            kw.update(color=textcolors[int(im.norm(data[i, j]) > threshold)])
            text = im.axes.text(j, i, valfmt(data[i, j], None), **kw)
            texts.append(text)

    return texts


fig, ax = plt.subplots()


im, cbar = heatmap(harvest, vegetables, farmers, ax=ax,
                   cmap="YlGn", cbarlabel="")
"""
    从一个numpy数组和两个标签列表创建一个热图。

    data
        形状为（N，M）的2D numpy数组。
    row_labels
        长度为N且带有行标签的列表或数组。
    col_labels
        长度为M的列表或数组，带有列的标签。
    ax
        绘制热图的`matplotlib.axes.Axes`实例。 如果
        未提供，请使用当前轴或创建一个新轴。 (可选的。)
    cbar_kw
        带有`matplotlib.Figure.colorbar`参数的字典。 可选的。
    cbarlabel
        颜色条的标签。 可选的。
    **kwargs
        所有其他参数都转发给“imshow”。
"""
# texts = annotate_heatmap(im, None)
'''
    im
        要标记的AxesImage。
    data
        用于注释的数据。 如果为None，则使用图像数据。 （可选的。）
    valfmt
        热图内注释的格式。 这应该
        使用字符串格式方法，例如 “ $ {x：.2f}”，或成为
        `matplotlib.ticker.Formatter`。 （可选的。）
    textcolors
         两种颜色规格的列表或数组。 第一个代表值低于阈值，第二个代表高于阈值的值。 （可选的。）
    threshold
        以数据单位表示的值，根据该值，textcolors中的颜色是
        应用。 如果为None（默认），则将颜色图的中间用作
        分离。（ 可选的。）
    **kwargs
        所有其他参数都转发给用于创建的每个`text`调用。
        文字标签。


'''

fig.tight_layout()  #自动调整子图参数,使之填充整个图像区域。
plt.show()      #图像展示
