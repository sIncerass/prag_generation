"""
This file contains code which was used to sample 100 instances from the train set data,
to performa manual data inspection and count the number of annotation errors.
"""

import sys
import random

random.seed(1)


def main(src_fn, num_samples=100):
    """
    Sample input data and write to a separate file for further analysis.

    :param src_fn: trainset data
    :param num_samples: number of samples to draw
    :return:
    """

    # calling aux, since this is needed for data analysis purposes
    print("AUX_DATA_ANALYSIS: Sampling input data")
    num_samples = int(num_samples)

    with open(src_fn) as src:
        srcs = [line.strip() for line in src]
        num_instances = len(srcs)
        print("Num instances: ", num_instances)

        sample_indices = random.sample(population=range(num_instances), k=num_samples)
        print("Sample indices: ", sample_indices)

        with open('%s.sample-%d' % (src_fn, num_samples), 'w') as sos:
            for i in sample_indices:
                sos.write('%s\n' % srcs[i])

    print("Done!")


if __name__ == '__main__':
    argvs = sys.argv[1:]

    # train data file and an integer specfying the num of samples to draw
    assert len(argvs) == 2
    print(argvs)
    main(*argvs)
