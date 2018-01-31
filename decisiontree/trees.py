from math import log


def create_data_set():
    data_set = [[1, 1, 'yes'],
               [1, 1, 'yes'],
               [1, 0, 'no'],
               [0, 1, 'no'],
               [0, 1, 'no']]
    labels = ['no surfacing', 'flippers']
    return data_set, labels


def calc_shannon_ent(data_set):
    num_entries = len(data_set)
    label_counts = {}
    shannon_ent = 0.0

    for feat_vec in data_set:
        current_label = feat_vec[-1]
        if current_label not in label_counts.keys():
            label_counts[current_label] = 0

        label_counts[current_label] += 1

        shannon_ent = 0.0
        for key in label_counts:
            prob = float(label_counts[key]) / num_entries
            shannon_ent -= prob * log(prob, 2)

    return shannon_ent


# Use to remove axis equal to the value
# and return the others axises
def split_data_set(data_set, axis, value):
    ret_data_set = []
    for feat_vec in data_set:
        if feat_vec[axis] == value:
            reduced_feat_vec = feat_vec[:axis]
            reduced_feat_vec.extend(feat_vec[axis+1:])
            ret_data_set.append(reduced_feat_vec)

    return ret_data_set


def choose_best_feature_to_split(data_set):
    num_features = len(data_set[0]) - 1
    base_entropy = calc_shannon_ent(data_set)
    best_info_gain = 0.0
    best_feature = -1

    for i in range(num_features):
        # Get all values of feature i
        feat_list = [example[i] for example in data_set]
        unique_vals = set(feat_list)

        # Calculate the info gain
        new_entropy = 0.0
        for value in unique_vals:
            sub_data_set = split_data_set(data_set, i, value)
            prob = len(sub_data_set) / float(len(data_set))
            new_entropy += prob * calc_shannon_ent(sub_data_set)
        info_gain = base_entropy - new_entropy

        # Choose the best info gain
        if info_gain > best_info_gain:
            best_info_gain = info_gain
            best_feature = i

    return best_feature


def demo1():
    data_set, labels = create_data_set()
    print(calc_shannon_ent(data_set))


def demo2():
    data_set, labels = create_data_set()
    print(split_data_set(data_set, 0, 1))
    print(split_data_set(data_set, 1, 1))


def demo3():
    data_set, labels = create_data_set()
    print(choose_best_feature_to_split(data_set))


if __name__ == '__main__':
    demo3()
