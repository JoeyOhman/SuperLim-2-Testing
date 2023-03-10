#! /usr/bin/env python
# -*- coding: utf-8
'''Python implementation of Krippendorff's alpha -- inter-rater reliability

Gerlof Bouma
Språkbanken Text / Dept Swedish, Multilingualism, Language Technology
University of Gothenburg
v0.2.1 20230303
v0.2   20230203
v0.1   20230131

Code originally based upon:

  krippendorff_alpha.py
  https://github.com/grrrr/krippendorff-alpha
  GPLv3.0 to Thomas Grill, 2017

'''


def nominal_metric(a, b):
    return a != b


def interval_metric(a, b):
    return (a - b) ** 2


def ratio_metric(a, b):
    return ((a - b) / (a + b)) ** 2


def krippendorff_alpha(data,
                       *,
                       summarized=False,
                       metric=interval_metric,
                       convert_items=float,
                       missing_items=[]):
    '''Calculate Krippendorff's alpha (inter-rater reliability):

    1. When summarized is False (default), data is in the format

    [
        [score_item1, score_item2, score_item3, ...]  # coder 1
        [score_item1, score_item2, score_item3, ...]  # coder 2
        ...                                           # etc
    ]

    or it is a list of dictionaries with explicit identifiers for the items
    (to easily allow for missing data)

    [
        {item1: score,              item3:score, ...}  # coder 1
        {item1: score, item2:score, item3:score, ...}  # coder 2
        ...                                            # etc
    ]

    (Mixed strategies are technically allowed but likely error prone.)
    In these formats, missing items are filtered according to
    missing_items. Scores will be passed through the function supplied
    by convert_items, so they can be of any type convert_items will
    accept.

    2. If summarized is True, data should be a dictionary mapping a
    combination of scores (across coders) to the number of times that
    combination was observed in the data:

    {
        (score_a, score_b, score_c): nr_obs_1,
        (score_d, score_e): nr_obs_2,
        (score_f, score_g, score_h): nr_obs_3,
        ...
    }

    Because of the possibility of having missing values, the keys may
    be of differing length, but they should all be of length >=
    2. Scores are passed to the metric as is, and not converted nor
    checked for special missing items values.

    A particular use case for this way of specifying the data is to
    directly supply a confusion matrix in evaluation of a system
    (coder 2) against a gold standard (coder 1). For instance, for a
    binary task, we get:

    data =
    {
        (True,True):  nr_true_positives,
        (True,False): nr_false_negatives,
        (False,True): nr_false_positives,
        (True,True):  nr_true_negatives,
    }

    for which we can calculate Krippendorff's alpha by

    krippendorff_alpha(data,metric=nominal_metric)

    ---

    Args:
        data: data

        summarized: boolean indicating whether the data is in
            summarized form or not (default: False)

        metric: function calculating the pairwise distance (default:
            interval_metric)

        convert_items: function for the type conversion of items, used
            when summarized is False (default: float). Value None will
            bypass item conversion.

        missing_items: "score(s)" indicating a missing item, used when
            summarized is False; a list/tuple value can be used to
            specify several markers for missing items, anything else
            is treated as a single missing item marker (default: [])

    Returns:
        Krippendorff's alpha as float

    Raises:

    '''

    if summarized:
        values = {combination: nr_obs
                  for combination, nr_obs in data.items()
                  if len(combination) > 1 and nr_obs > 0}
    else:
        if convert_items is None:
            convert_items = lambda x: x  # noqa

        # set of constants identifying missing values
        if isinstance(missing_items, (list, tuple)):
            maskitems = list(missing_items)
        else:
            maskitems = [missing_items]

        # convert input data to a dict of items
        units = {}
        for d in data:
            try:
                # try if d behaves as a dict
                diter = d.items()
            except AttributeError:
                # sequence assumed for d
                diter = enumerate(d)

            for it, g in diter:
                if g not in maskitems:
                    units.setdefault(it, []).append(convert_items(g))

        values = {}
        for v in units.values():
            v = tuple(v)

            if len(v) < 2:
                pass
            elif v in values:
                values[v] += 1
            else:
                values[v] = 1

    n = sum(len(v) * m for v, m in values.items())  # number of pairable values
    if n == 0:
        raise ValueError("No items to compare.")

    Do = sum(m * sum(metric(gi, gj)
                     for gi in grades
                     for gj in grades) / (len(grades) - 1)
             for grades, m in values.items()) / n

    if Do == 0:
        return 1.0

    De = sum(m1 * m2 * metric(gi, gj)
             for grades1, m1 in values.items()
             for grades2, m2 in values.items()
             for gi in grades1
             for gj in grades2) / (n * (n - 1))

    return 1.0 - Do / De if (Do and De) else 1.0


if __name__ == '__main__':
    print("Example from http://en.wikipedia.org/wiki/Krippendorff's_Alpha")

    data = (
        "NA    NA    NA    NA    *    3    4     1     2     1     1     3     3    NA     3",  # coder A
        " 1    NA     2     1    3    3    4     3    NA    NA    NA    NA    NA    NA    NA",  # coder B
        "NA    NA     2     1    3    4    4    NA     2     1     1     3     3    NA     4",  # coder C
    )

    missing = ['NA', '*']  # indicator for missing values
    array = [d.split() for d in data]  # convert to 2D list of string items

    print("nominal metric: %.3f" % krippendorff_alpha(array, metric=nominal_metric, missing_items=missing))
    print("interval metric: %.3f" % krippendorff_alpha(array, metric=interval_metric, missing_items=missing))