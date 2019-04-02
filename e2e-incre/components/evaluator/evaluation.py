import subprocess
import re
import os, sys


def eval_output(ref_fn, pred_fn):
    """
    Runs an external evaluation script (COCO/MTeval evaluation, measure_scores.py) and retrieves the scores
    :param pred_fn:
    :param ref_fn:
    :return:
    """

    pat = r"==============\n" \
          r"BLEU: (\d+\.?\d*)\n" \
          r"NIST: (\d+\.?\d*)\n" \
          r"METEOR: (\d+\.?\d*)\n" \
          r"ROUGE_L: (\d+\.?\d*)\n" \
          r"CIDEr: (\d+\.?\d*)\n"

    eval_out = _sh_eval(pred_fn, ref_fn)
    eval_out = eval_out.decode("utf-8")
    scores = re.search(pat, eval_out).group(1, 2, 3, 4, 5)
    # scores should be: bleu, nist, meteor, rouge, cider
    return [float(x) for x in scores]


def _sh_eval(pred_fn, ref_fn):
    """
    Runs measure_scores.py script and processes the output
    :param pred_fn:
    :param ref_fn:
    :return:
    """
    this_dir = os.path.dirname(os.path.abspath(__file__))
    script_fname = os.path.join(this_dir, 'eval_scripts/run_eval.sh')
    out = subprocess.check_output([script_fname, ref_fn, pred_fn])
    return out


if __name__ == '__main__':
    ref_fn = sys.argv[1]
    sys_fn = sys.argv[2]
    bleu, nist, meteor, rouge, cider = eval_output(ref_fn, sys_fn)
