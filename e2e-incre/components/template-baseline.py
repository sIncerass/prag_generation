import sys
import csv
import logging
import re

"""
This file contains code for the template-based E2E NLG Challenge baseline.
It is a simple mixture of two templates and some rules that fix some discrepances
which arise from stitching phrases together.

To make predictions on *filename.txt*, run the following command:

    ```
    python template-baseline.py filename.txt MODE
    ```
    Here, *filename.txt* is either devset or testset CSV file;
    *MODE* can be either 'dev' or 'test'.

    Model-T's predictions are saved in *filename.txt.predicted*.

"""

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)


def process_e2e_mr(s):
    """
    Extract key-value pairs from the input and pack them into a dictionary.
    :param s: src string containing key-value pairs.
    :return: a dictionary w/ key-value pairs corresponding to MR keys and their values on the src side of the input.
    """
    items = s.split(", ")

    k2v = {"name": None,
           "familyFriendly": None,
           "eatType": None,
           "food": None,
           "priceRange": None,
           "near": None,
           "area": None,
           "customer rating": None
           }

    for idx, item in enumerate(items):
        key, raw_val = item.split("[")
        val = raw_val[:-1]
        k2v[key] = val

    return k2v


def _get_price_str(mr_val):
    """
    Handle the price prediction part.
    :param mr_val: value of the 'price' field.
    :return: refined sentence string.
    """
    if not mr_val:
        s = "."
        return s

    if "£" in mr_val:
        s = " in the price range of %s." % mr_val

    else:
        mr_val = 'low' if mr_val == 'cheap' else mr_val
        s = " in the %s price range." % mr_val

    return s


def _get_rating(mr_val, snt):
    """
    Handle the rating part.
    :param mr_val: value of the 'customerRating' field.
    :param snt: sentence string built so far.
    :return: refined sentence string.
    """
    # if the previous sentence ends with a dot
    if snt[-1] != ".":
        beginning = " with"
    else:
        beginning = " It has"

    if mr_val.isalpha():
        s = "%s a %s customer rating" % (beginning, mr_val)

    else:
        s = "%s a customer rating of %s" % (beginning, mr_val)

    return snt + s


def _get_loc(area_val, near_val, snt):
    """
    Handle location string, variant 1.
    :param area_val: value of the 'area' field.
    :param near_val: value of the 'near' field.
    :param snt: incomplete sentence string (string built so far)
    :return:
    """
    tokens = snt.split()

    if "It" in tokens:
        beginning = " and"

    else:
        beginning = ". It"

    if area_val:
        s = "%s is located in the %s area" % (beginning, area_val)
        if near_val:
            s += ", near %s." % near_val

        else:
            s += "."

    elif near_val:
        s = "%s is located near %s." % (beginning, near_val)

    else:
        raise NotImplementedError()

    return snt + s


def _get_loc2(area_val, near_val):
    """
    Handle location string, variant 2.
    :param area_val: value of the 'area' field.
    :param near_val: value of the 'near' field.
    :return:
    """
    if area_val:
        s = " located in the %s area" % area_val
        if near_val:
            s += ", near %s." % near_val

        else:
            s += "."


    elif near_val:
        s = " located near %s." % near_val

    else:
        raise NotImplementedError()

    return s


def postprocess(snt):
    """
    Fix some spelling and punctuation.
    :param snt: sentence string before post-processing
    :return: sentence string after post-processing
    """
    tokens = snt.split()
    for idx, t in enumerate(tokens):

        # Fix article usage
        if t.lower() == "a":
            if tokens[idx + 1][0] in ["a", "A"]:
                tokens[idx] = "%sn" % t

        elif t.lower() == "an":
            if tokens[idx + 1][0] not in ["a", "A"]:
                tokens[idx] = "%s" % t[0]

    # Add a trailing dot
    last_token = tokens[-1]
    if last_token[-1] != ".":
        tokens[-1] = last_token + "."

    return " ".join(tokens)


def make_prediction(xd):
    """
    Main function to make a prediction.

    Our template has a generic part and a field-specific part which we called SUBTEMPLATE:

        [SUBTEMPLATE-1] which serves [food] in the [price] price range.
        It has a [customerRating] customer rating.
        It is located in [area] area, near [near].
        [SUBTEMPLATE-2]

    If the familyFriendly field's value is "yes", the SUBTEMPLATES are:

        - SUBTEMPLATE-1: [name] is a family-friendly [eatType]
        - SUBTEMPLATE-2: None

    Otherwise:

        - SUBTEMPLATE-1: [name] is a [eatType]
        - SUBTEMPLATE-2: It is not family friendly.

    There are some variations of the ordering which we resolve through if-else statements.
    Finally, there is also a post-processing step which handles punctuation mistakes and article choice (a/an).

    :param xd: a dictionary containing input data.
    The dictionary has the following keys:
    ["name", "familyFriendly", "eatType", "food", "priceRange", "near", "area", "customer rating"]
    :return: a string denoting the sentence which describes input xd.
    """

    name = xd["name"]
    assert name is not None
    if xd["familyFriendly"] == "yes":
        friendly = True

    elif xd["familyFriendly"] == "no":
        friendly = False

    else:
        friendly = None

    restaurant_type = xd["eatType"] or 'dining place'

    food_type = xd["food"]
    price_range = xd["priceRange"]
    rating = xd["customer rating"]
    near = xd["near"]
    area = xd["area"]

    if friendly is True:
        snt = "%s is a family-friendly %s" % (name, restaurant_type)
    else:
        snt = "%s is a %s" % (name, restaurant_type)

    if food_type or price_range:

        if food_type == "Fast food":
            food_type = "fast"

        elif food_type is None:
            food_type = ""

        food = "%s food" % food_type
        snt += " which serves %s" % food
        price_range = _get_price_str(price_range)
        snt += price_range
        # ends with a dot

        if rating:
            snt = _get_rating(rating, snt)
            if near or area:
                snt = _get_loc(area, near, snt)
                # ends with a dot
            else:
                snt += "."
                # ends with a dot

        elif near or area:
            snt = snt[:-1]  # strip the trailing dot
            snt = _get_loc(area, near, snt)
            # ends with a dot

        if friendly is False:
            snt += " It is not family friendly."

        snt = postprocess(snt)
        return snt

    # no dot?
    if near or area:
        snt += _get_loc2(area, near)
        # ends with a dot

    if rating:
        snt = _get_rating(rating, snt)

    if friendly is False:
        tokens_so_far = snt.split()
        if tokens_so_far[-1][-1] != ".":
            snt += ". It is not family friendly."

        else:
            snt += " It is not family friendly."

    snt = postprocess(snt)

    return snt


def run(fname, mode='dev'):
    """
    Main function.
    :param fname: filename with the input.
    :param mode: operation mode, either dev (input has both MR and TEXT) or test (only MR given).
    :return:
    """

    input_data = []
    predictions = []
    with open(fname, 'r') as csv_file:
        reader = csv.reader(csv_file, delimiter=',', quotechar='"')
        header = next(reader)

        # Files should have headers
        if header == ['mr', 'ref']:
            assert mode == 'dev'
            logger.info('Predicting on DEV data')

        elif header == ['MR']:
            assert mode == 'test'
            logger.info('Predicting on TEST data')

        else:
            logger.error('The file does not contain a header!')

        first_row = next(reader)
        curr_x = first_row[0]
        input_data.append(curr_x)
        xd = process_e2e_mr(curr_x)
        p = make_prediction(xd)
        predictions.append(p)

        for row in list(reader):
            x = row[0]
            input_data.append(x)
            this_xd = process_e2e_mr(x)

            # if same MR, skip (multi-ref dev data)
            if x == curr_x:
                continue

            else:
                p = make_prediction(this_xd)
                predictions.append(p)
                curr_x = x

    # Saving predictions
    predictions_fname = "%s.predicted" % fname
    with open(predictions_fname, "w") as out:
        logger.info("Saving predictions to --> %s" % predictions_fname)

        if mode == 'test':
            # a TSV file with paired MR and predictions
            for idx, p in enumerate(predictions):
                out.write('"%s"\t"%s"\n' % (input_data[idx], p))

        else:
            # Just predictions, for evaluation with automatic scripts
            for p in predictions:
                out.write("%s\n" % p)

    logger.info("Done")


def test():
    logger.info("Running a test: prediction by a template baseline")
    x1 = "name[The Vaults]"
    x2 = "name[The Vaults], eatType[pub]"
    x3 = "name[The Vaults], eatType[pub], priceRange[more than £30]"
    x4 = "name[The Vaults], eatType[pub], priceRange[more than £30], customer rating[5 out of 5]"
    x5 = "name[The Vaults], eatType[pub], priceRange[more than £30], customer rating[5 out of 5], near[Café Adriatic]"
    x6 = "name[The Vaults], eatType[pub], priceRange[more than £30], customer rating[5 out of 5], " \
         "near[Café Adriatic], food[English]"
    x7 = "name[The Vaults], eatType[pub], priceRange[more than £30], customer rating[5 out of 5], " \
         "near[Café Adriatic], food[English], familyFriendly[yes]"

    for x in [x1, x2, x3, x4, x5, x6, x7]:
        data = process_e2e_mr(x)
        p = make_prediction(data)

        print("%s\n%s\n\n" % (x, p))


if __name__ == "__main__":
    args = sys.argv

    if len(args) == 3:
        dev_data_fn = sys.argv[1]
        mode = sys.argv[2]
        assert mode in ['test', 'dev']
        run(dev_data_fn, mode=mode)

    elif len(args) == 2:
        dev_data_fn = sys.argv[1]
        run(dev_data_fn, mode='dev')

    elif len(args) == 1:
        test()

    else:
        raise NotImplementedError()
