import random


class RNA_Stack:
    """
    栈
    """
    def __init__(self):
        self.data = []  # 储存下标

    def push(self, index):
        """
        入栈
        :return:
        """
        self.data.append(index)

    def pop(self):
        """
        出栈
        :return:
        """
        size = len(self.data) - 1
        return self.data.pop(size)


def reverse(seq, dot, prob):
    """
    数据翻转
    :param seq:
    :param dot:
    :param prob:
    :return:
    """
    seq_rev = seq[::-1]
    dot_rev = dot[::-1]
    for n, d in enumerate(dot_rev):
        if d == 3:
            dot_rev[n] = 4
            continue
        if d == 4:
            dot_rev[n] = 3
            continue
    prob_rev = prob[::-1]
    return seq_rev, dot_rev, prob_rev


def cross_over(seq, dot, prob):
    """
    同源序列互换
    :param seq:
    :param dot:
    :param prob:
    :return:
    """
    rna_stack = RNA_Stack()
    seq_list = list(seq)
    index = 0
    suffix = 0
    # 同源互换
    for s, d, p in zip(seq_list, dot, prob):
        lock = 0
        if d == 3:
            rna_stack.push(index)
        if d == 4:
            suffix = rna_stack.pop()
            lock = 1
        if random.random() >= 0.5 and lock == 1:
            tmp_s = seq_list[index]
            seq_list[index] = seq_list[suffix]
            seq_list[suffix] = tmp_s

            tmp_p = prob[index]
            prob[index] = prob[suffix]
            prob[suffix] = tmp_p
        index += 1
    return seq_list, dot, prob
