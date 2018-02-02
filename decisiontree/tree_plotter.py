import matplotlib.pyplot as plt
import operator


plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

decision_node = dict(boxstyle="sawtooth", fc="0.8")
leaf_node = dict(boxstyle="round4", fc="0.8")
arrow_args = dict(arrowstyle="<-")


def plot_mid_text(fig, cntr_pt, parent_pt, txt_string):
    x_mid = (parent_pt[0] - cntr_pt[0]) / 2.0 + cntr_pt[0]
    y_mid = (parent_pt[1] - cntr_pt[1]) / 2.0 + cntr_pt[1]
    fig.text(x_mid, y_mid, txt_string)


def plot_tree(fig, my_tree, total_w, total_d, x_off, y_off, parent_pt, node_txt):
    num_leafs = get_num_leafs(my_tree)
    first_str = list(my_tree.keys())[0]
    cntr_pt = (x_off + (1.0 + float(num_leafs)) / 2.0 / total_w, y_off)

    plot_mid_text(fig, cntr_pt, parent_pt, node_txt)
    plot_node(fig, first_str, cntr_pt, parent_pt, decision_node)
    second_dict = my_tree[first_str]
    y_off = y_off - 1.0 / total_d
    for key in second_dict.keys():
        if type(second_dict[key]) is dict:
            x_off = plot_tree(fig, second_dict[key], total_w, total_d, x_off, y_off, cntr_pt, str(key))
        else:
            x_off = x_off + 1.0 / total_w
            plot_node(fig, second_dict[key], (x_off, y_off), cntr_pt, leaf_node)
            plot_mid_text(fig, (x_off, y_off), cntr_pt, str(key))

    return x_off


def plot_node(fig, node_txt, center_pt, parent_pt, node_type):
    fig.annotate(node_txt, xy=parent_pt,
                 xycoords="axes fraction", xytext=center_pt,
                 textcoords="axes fraction", va="center", ha="center",
                 bbox=node_type, arrowprops=arrow_args)


def create_plot(in_tree):
    fig = plt.figure(1, facecolor='white')
    fig.clf()
    ax_props = dict(xticks=[], yticks=[])
    fig1 = plt.subplot(111, frameon=False, **ax_props)

    total_w = float(get_num_leafs(my_tree))
    total_d = float(get_tree_depth(my_tree))
    x_off = -0.5 / total_w
    y_off = 1.0
    plot_tree(fig1, in_tree, total_w, total_d, x_off, y_off, (0.5, 1.0), '')
    plt.show()


def get_num_leafs(my_tree):
    num_leafs = 0
    first_str = list(my_tree.keys())[0]
    second_dict = my_tree[first_str]
    for key in second_dict.keys():
        if type(second_dict[key]) is dict:
            num_leafs += get_num_leafs(second_dict[key])
        else:
            num_leafs += 1

    return num_leafs


def get_tree_depth(my_tree):
    max_depth = 0
    first_str = list(my_tree.keys())[0]
    second_dict = my_tree[first_str]

    for key in second_dict.keys():
        if type(second_dict[key]) is dict:
            this_depth = 1 + get_tree_depth(second_dict[key])
        else:
            this_depth = 1

        if this_depth > max_depth:
            max_depth = this_depth

    return max_depth


def retrieve_tree(i):
    list_of_trees = [{'no surfacing': {0: 'no', 1: {'flippers': {0: 'no', 1: 'yes'}}}},
                     {'no surfacing': {0: 'no', 1: {'flippers': {0: {'head': {0: 'no', 1: 'yes'}}, 1: 'no'}}}}]
    return list_of_trees[i]


if __name__ == "__main__":
    my_tree = retrieve_tree(0)
    my_tree['no surfacing'][3] = 'maybe'
    create_plot(my_tree)
