"""
This file contains code which was used to sample 100 instances from the dev set data,
to performa manual analysis of the predictions by the baseline model and MLP-based model.
"""

import sys
import random

random.seed(1)


def main(base_out_fn, model4_out_fn, template_out_fn, src_fn, num_samples=100):
    """
    Sample predictions by the three models and write them to separate files for further analysis.
    See subsection 5.2 of the paper.

    :param base_out_fn: Baseline prediction file
    :param model4_out_fn:  MLPModel prediction file
    :param template_out_fn: Template-based system prediction file
    :param src_fn: src side of the dev-multi-ref data (devset.csv.multi-ref.src)
    :param num_samples: number of samples to draw
    :return:
    """

    # calling aux, since this is needed for data analysis purposes
    print("AUX_DATA_ANALYSIS: Sampling predictions")
    num_samples = int(num_samples)

    with open(src_fn) as src, \
            open(base_out_fn) as bo, \
            open(model4_out_fn) as mo, \
            open(template_out_fn) as to:
        srcs = [line.strip() for line in src]
        bsnt = [line.strip() for line in bo]
        mosnt = [line.strip() for line in mo]
        tosnt = [line.strip() for line in to]

        num_instances = len(bsnt)
        assert len(srcs) == len(mosnt) == len(tosnt) == num_instances
        print("Num instances: ", num_instances)

        sample_indices = random.sample(population=range(num_instances), k=num_samples)
        print("Sample indices: ", sample_indices)

        with open('%s.sample-%d' % (base_out_fn, num_samples), 'w') as bos, \
                open('%s.sample-%d' % (model4_out_fn, num_samples), 'w') as mos, \
                open('%s.sample-%d' % (template_out_fn, num_samples), 'w') as tos, \
                open('%s.sample-inputs' % (base_out_fn), 'w') as ios:
            for i in sample_indices:
                ios.write('%s\n' % srcs[i])
                bos.write('%s\n' % bsnt[i])
                mos.write('%s\n' % mosnt[i])
                tos.write('%s\n' % tosnt[i])

    print("Done!")


if __name__ == '__main__':
    argvs = sys.argv[1:]

    # three files with predictions, src side of the data and an integer specfying the num of samples to draw
    assert len(argvs) == 5
    print(argvs)
    main(*argvs)
