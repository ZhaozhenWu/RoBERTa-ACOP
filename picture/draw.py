# -*- coding: utf-8 -*-
import matplotlib
# matplotlib.use('PS')
# mpl.use("Qt5Agg")
# matplotlib.use('Qt5Agg')
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.pyplot import MultipleLocator


# plt.rc('font', family='Times New Roman')
def heatmap(data, row_labels, col_labels, norm, ax=None,
            cbar_kw={}, cbarlabel="", **kwargs):
    """
    Create a heatmap from a numpy array and two lists of labels.

    Parameters
    ----------
    data
        A 2D numpy array of shape (N, M).
    row_labels
        A list or array of length N with the labels for the rows.
    col_labels
        A list or array of length M with the labels for the columns.
    ax
        A `matplotlib.axes.Axes` instance to which the heatmap is plotted.  If
        not provided, use current axes or create a new one.  Optional.
    cbar_kw
        A dictionary with arguments to `matplotlib.Figure.colorbar`.  Optional.
    cbarlabel
        The label for the colorbar.  Optional.
    **kwargs
        All other arguments are forwarded to `imshow`.
    """

    if not ax:
        ax = plt.gca()

    # Plot the heatmap
    #    im = ax.pcolor(data, edgecolors='k', linewidths=4)
    im = ax.imshow(data, **kwargs)

    # Create colorbar
    cbar = None
    #    cbar = ax.figure.colorbar(im, ax=ax, **cbar_kw)
    #    cbar = ax.figure.colorbar(im, ax=ax,vmin=0,vmax=1, **cbar_kw)
    #    cbar.
    #    cbar.set_ticks(np.linspace(0, 1,5))
    #    cbar.ax.set_ylabel(cbarlabel, rotation=-90, va="bottom")

    # We want to show all ticks...
    ax.set_xticks(np.arange(data.shape[1]))
    ax.set_yticks(np.arange(data.shape[0]))
    # ... and label them with the respective list entries.
    ax.set_xticklabels(col_labels, fontsize=23)
    ax.set_yticklabels(row_labels, fontsize=23)

    # Let the horizontal axes labeling appear on top.
    ax.tick_params(top=True, bottom=False,
                   labeltop=True, labelbottom=False)

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=-30, ha="right",
             rotation_mode="anchor")

    # Turn spines off and create white grid.
    for edge, spine in ax.spines.items():
        spine.set_visible(False)

    ax.set_xticks(np.arange(data.shape[1] + 1) - .5, minor=True)
    ax.set_yticks(np.arange(data.shape[0] + 1) - .5, minor=True)
    ax.grid(which="minor", color="w", linestyle='-', linewidth=3)
    ax.tick_params(which="minor", bottom=False, left=False)

    return im, cbar


def annotate_heatmap(im, data=None, valfmt="{x:.2f}",
                     textcolors=["black", "white"],
                     threshold=None, **textkw):
    """
    A function to annotate a heatmap.

    Parameters
    ----------
    im
        The AxesImage to be labeled.
    data
        Data used to annotate.  If None, the image's data is used.  Optional.
    valfmt
        The format of the annotations inside the heatmap.  This should either
        use the string format method, e.g. "$ {x:.2f}", or be a
        `matplotlib.ticker.Formatter`.  Optional.
    textcolors
        A list or array of two color specifications.  The first is used for
        values below a threshold, the second for those above.  Optional.
    threshold
        Value in data units according to which the colors from textcolors are
        applied.  If None (the default) uses the middle of the colormap as
        separation.  Optional.
    **kwargs
        All other arguments are forwarded to each call to `text` used to create
        the text labels.
    """

    if not isinstance(data, (list, np.ndarray)):
        data = im.get_array()

    # Normalize the threshold to the images color range.
    if threshold is not None:
        threshold = im.norm(threshold)
    else:
        threshold = im.norm(data.max()) / 2.

    # Set default alignment to center, but allow it to be
    # overwritten by textkw.
    kw = dict(horizontalalignment="center",
              verticalalignment="center")
    kw.update(textkw)

    # Get the formatter in case a string is supplied
    if isinstance(valfmt, str):
        valfmt = matplotlib.ticker.StrMethodFormatter(valfmt)

    # Loop over the data and create a `Text` for each "pixel".
    # Change the text's color depending on the data.
    texts = []
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            kw.update(color=textcolors[int(im.norm(data[i, j]) > threshold)])
            text = im.axes.text(j, i, valfmt(data[i, j], None), **kw)
            texts.append(text)

    return texts


def draw_attention(words1, words2, att, name='hj'):
    norm = matplotlib.colors.Normalize(vmin=0, vmax=1)
    assert len(words1) == att.shape[0]
    assert len(words2) == att.shape[1]
    fig, ax = plt.subplots(figsize=(20, 20))

    im, cbar = heatmap(att, words1, words2, norm, ax=ax,
                       cmap="YlGn", cbarlabel="Attention Value")
    # texts = annotate_heatmap(im, valfmt="{x:.1f} t")

    fig.tight_layout()
    fig.savefig('./pics/pic' + name + '.eps', dpi=50)


def draw_train_loss(path):
    with open(path, encoding='utf-8') as f:
        lines = f.readlines()

    # 提取trainLoss和validationLoss
    loss_list = []
    acc_list = []
    test_acc_list = []
    test_f1_list = []
    for line in lines:
        loss = np.float64(line.split(",")[0].split("loss: ")[-1])
        loss_list.append(loss)
        acc = np.float64(line.split(",")[1].split("acc: ")[-1])
        acc_list.append(acc)
        test_acc = np.float64(line.split(",")[2].split("test_acc: ")[-1])
        test_acc_list.append(test_acc)
        test_f1 = np.float64(line.split(",")[3].split("test_f1: ")[-1])
        test_f1_list.append(test_f1)
    epoch_num = len(loss_list)

    # draw
    fig = plt.figure()
    xs = np.arange(epoch_num)
    plt.yticks(np.arange(min(loss_list), max(loss_list), 0.1))
    plt.plot(xs, loss_list, color='coral', label="train loss")
    plt.plot(xs, acc_list, color='g', label="train acc")
    plt.plot(xs, test_acc_list, color='r', label="test acc")
    plt.plot(xs, test_f1_list, color='b', label="test f1")
    plt.legend()
    plt.show()
    # plt.savefig("loss.png")


def draw_local_ACOP(file_lap, file_twitter):
    with open(file_lap, encoding='utf-8') as f:
        lines_lap = f.readlines()
    with open(file_twitter, encoding='utf-8') as f:
        lines_twitter = f.readlines()
    acc_lap = []
    acc_twitter = []
    f1_lap = []
    f1_twitter = []
    acc_lap_global = [83.57030015797788]*10
    acc_twitter_global = [77.02312138728323]*10
    f1_lap_global = [80.18676790643422]*10
    f1_twitter_global = [75.55289421157685]*10

    for line in lines_lap:
        acc = np.float64(line.split(",")[0].strip().split("max_test_acc: ")[-1])
        acc_lap.append(acc)
        f1 = np.float64(line.split(",")[1].strip().split("max_test_f1: ")[-1])
        f1_lap.append(f1)

    for line in lines_twitter:
        acc = np.float64(line.split(",")[0].strip().split("max_test_acc: ")[-1])
        acc_twitter.append(acc)
        f1 = np.float64(line.split(",")[1].strip().split("max_test_f1: ")[-1])
        f1_twitter.append(f1)

    acc_lap = [i * 100 for i in acc_lap]
    acc_twitter = [i * 100 for i in acc_twitter]
    f1_lap = [i * 100 for i in f1_lap]
    f1_twitter = [i * 100 for i in f1_twitter]
    # y1 = [10, 13, 5, 40, 30, 60, 70, 12, 55, 25]
    x1 = range(1, 11)
    x2 = range(1, 11)
    x3 = range(1, 11)
    x4 = range(1, 11)
    # y2 = [5, 8, 0, 30, 20, 40, 50, 10, 40, 15]
    x_major_locator = MultipleLocator(1)
    y_major_locator = MultipleLocator(2)
    ax = plt.gca()
    ax.xaxis.set_major_locator(x_major_locator)
    ax.yaxis.set_major_locator(y_major_locator)
    plt.ylim(75, 83)
    plt.plot(x1, f1_lap, label='Laptop', linewidth=1, color='r', marker='o',
             markerfacecolor='red', markersize=4)
    plt.plot(x3, f1_lap_global, label='Laptop-Global', linewidth=1, color='r',
             markerfacecolor='red', markersize=4)
    plt.plot(x2, f1_twitter, label='Twitter', linewidth=1, color='b', marker='^',
             markerfacecolor='blue', markersize=5)
    plt.plot(x4, f1_twitter_global, label='Twitter-Global', linewidth=1, color='b',
             markerfacecolor='blue', markersize=5)
    plt.xlabel('window size')
    plt.ylabel('F1(%)')
    # plt.title('Interesting Graph\nCheck it out')
    plt.legend()
    plt.show()

    # 从pyplot导入MultipleLocator类，这个类用于设置刻度间隔

    # x_values = list(range(11))
    # y_values = [x ** 2 for x in x_values]
    # plt.plot(x_values, y_values, c='green')
    # plt.title('Squares', fontsize=24)
    # plt.tick_params(axis='both', which='major', labelsize=14)
    # plt.xlabel('Numbers', fontsize=14)
    # plt.ylabel('Squares', fontsize=14)
    # x_major_locator = MultipleLocator(1)
    # # 把x轴的刻度间隔设置为1，并存在变量里
    # y_major_locator = MultipleLocator(10)
    # # 把y轴的刻度间隔设置为10，并存在变量里
    # ax = plt.gca()
    # # ax为两条坐标轴的实例
    # ax.xaxis.set_major_locator(x_major_locator)
    # # 把x轴的主刻度设置为1的倍数
    # ax.yaxis.set_major_locator(y_major_locator)
    # # 把y轴的主刻度设置为10的倍数
    # plt.xlim(-0.5, 11)
    # # 把x轴的刻度范围设置为-0.5到11，因为0.5不满一个刻度间隔，所以数字不会显示出来，但是能看到一点空白
    # plt.ylim(-5, 110)
    # # 把y轴的刻度范围设置为-5到110，同理，-5不会标出来，但是能看到一点空白
    # plt.show()


def draw_heap_map():
    import numpy as np
    import matplotlib.pyplot as plt

    vegetables = []
    farmers = ["Farmer Joe", "Upland Bros.", "Smith Gardening",
               "Agrifun", "Organiculture", "BioGoods Ltd.", "Cornylee Corp."]

    harvest = np.array([[0.8, 2.4, 2.5, 3.9, 0.0, 4.0, 0.0],
                        ])

    fig, ax = plt.subplots()
    im = ax.imshow(harvest)

    # Show all ticks and label them with the respective list entries
    ax.set_xticks(np.arange(len(farmers)), labels=farmers)
    ax.set_yticks(np.arange(len(vegetables)), labels=vegetables)

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")

    # Loop over data dimensions and create text annotations.
    for i in range(len(vegetables)):
        for j in range(len(farmers)):
            text = ax.text(j, i, harvest[i, j],
                           ha="center", va="center", color="w")

    ax.set_title("Harvest of local farmers (in tons/year)")
    fig.tight_layout()
    plt.show()


if __name__ == '__main__':
    draw_heap_map()
    # draw_local_ACOP('../log/roberta_lap14_window_val.txt', '../log/roberta_twitter_window_val.txt')
    # draw_train_loss('../log/roberta_rest14_result.txt')

