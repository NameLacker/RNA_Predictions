import sys
import time
import numpy as np

from utils import vocabulary
from config import RNA_Config

collocations = RNA_Config()


def out(x="\n", end="\n"):
    print(x, end=end)
    sys.stdout.flush()


def format_elapsed(start_time):
    elapsed_time = int(time.time() - start_time)
    minutes, seconds = divmod(elapsed_time, 60)
    hours, minutes = divmod(minutes, 60)
    days, hours = divmod(hours, 24)
    elapsed_string = "{}h{:02}m{:02}s".format(hours, minutes, seconds)
    if days > 0:
        elapsed_string = "{}d{}".format(days, elapsed_string)
    return elapsed_string


def process_vocabulary(data, quiet=False):
    """
    创建并返回词汇表对象
    """
    START = collocations.START
    STOP = collocations.STOP
    UNK = collocations.UNK
    if not quiet:
        print("Initializing vacabularies... ")
    seq_vocab = vocabulary.Vocabulary()
    bracket_vocab = vocabulary.Vocabulary()

    for vocab in [seq_vocab, bracket_vocab]:
        vocab.index(START)
        vocab.index(STOP)
    for x in data[:100]:
        seq = x["sequence"]
        dot = x["structure"]
        for character in seq:
            seq_vocab.index(character)
        for character in dot:
            bracket_vocab.index(character)
    for vocab in [seq_vocab, bracket_vocab]:
        vocab.freeze()
    if not quiet:
        print("Process done!")

    def print_vocabulary(name, vocab):
        special = {START, STOP}
        out("{}({:,}): {}".format(
            name, vocab.size,
            sorted(value for value in vocab.values if value in special) +
            sorted(value for value in vocab.values if value not in special)))

    if not quiet:
        print_vocabulary("Sequence", seq_vocab)
        print_vocabulary("Brackets", bracket_vocab)
    return seq_vocab, bracket_vocab


def operator_rmsd_avg(preds, labels):
    """
    计算预测的RMSD_AVG和RMSD_STD
    :param preds:
    :param labels:
    :return:
    """
    all_rmsd = []
    rmsd_avg = 0
    for pred, label in zip(preds, labels):
        count_pl = 0
        for p, l in zip(pred, label):
            minus = (p - l)**2
            count_pl += minus
        avg = count_pl / len(pred)
        rmsd = np.sqrt(avg)
        rmsd_avg += rmsd
        all_rmsd.append(rmsd)
    rmsd_avg /= len(preds)
    rmsd_std = operator_rmsd_std(all_rmsd, rmsd_avg)
    return rmsd_avg, rmsd_std


def operator_rmsd_std(all_rmsd, rmsd_avg):
    """
    计算RMSD_STD
    :param all_rmsd:
    :param rmsd_avg:
    :return:
    """
    rmsd_std = 0
    for rmsd in all_rmsd:
        minus = (rmsd - rmsd_avg)**2
        rmsd_std += minus
    rmsd_std /= len(all_rmsd)
    rmsd_std = np.sqrt(rmsd_std)
    return rmsd_std