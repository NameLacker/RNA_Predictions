import sys
import time
from utils import vocabulary


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
    Creates and returns vocabulary objects.
    Only iterates through the first 100 sequences, to save computation.
    创建并返回词汇表对象
    只迭代前100个序列，以节省时间
    """
    START = "<START>"
    STOP = "<STOP>"
    UNK = "<UNK>"
    if not quiet:
        print("initializing vacabularies... ")
    seq_vocab = vocabulary.Vocabulary()
    bracket_vocab = vocabulary.Vocabulary()
    # loop_type_vocab = vocabulary.Vocabulary()

    for vocab in [seq_vocab, bracket_vocab]:  # , loop_type_vocab]:
        vocab.index(START)
        vocab.index(STOP)
    for x in data[:100]:
        seq = x["sequence"]
        dot = x["structure"]
        # loop = x["predicted_loop_type"]
        for character in seq:
            seq_vocab.index(character)
        for character in dot:
            bracket_vocab.index(character)
        # for character in loop:
        #    loop_type_vocab.index(character)
    for vocab in [seq_vocab, bracket_vocab]:  # , loop_type_vocab]:
        # vocab.index(UNK)
        vocab.freeze()
    if not quiet:
        print("done.")

    def print_vocabulary(name, vocab):
        # special = {START, STOP, UNK}
        special = {START, STOP}
        out("{}({:,}): {}".format(
            name, vocab.size,
            sorted(value for value in vocab.values if value in special) +
            sorted(value for value in vocab.values if value not in special)))

    if not quiet:
        print_vocabulary("Sequence", seq_vocab)
        print_vocabulary("Brackets", bracket_vocab)
    return seq_vocab, bracket_vocab
