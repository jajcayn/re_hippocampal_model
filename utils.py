"""
Helper functions
"""

import logging
from functools import partial
from multiprocessing import Pool, cpu_count

from tqdm import tqdm


def run_in_parallel(
    partial_function,
    iterable,
    workers=cpu_count(),
    length=None,
    assert_ordered=False,
):
    """
    Wrapper for running functions in parallel with tqdm bar.

    :param partial_function: partial function to be evaluated
    :type partial_function: :class:`_functools.partial`
    :param iterable: iterable comprised of arguments to be fed to partial
        function
    :type iterable: iterable
    :param workers: number of workers to be used
    :type workers: int
    :param length: Length of the iterable / generator.
    :type length: int|None
    :param assert_ordered: whether to assert order of results same as the
        iterable (imap vs imap_unordered)
    :type assert_ordered: bool
    :return: list of values returned by partial function
    :rtype: list
    """
    total = length
    if total is None:
        try:
            total = len(iterable)
        except (TypeError, AttributeError):
            pass

    # wrap method in order to get original exception from a worker process
    partial_function = partial(_worker_fn, fn=partial_function)

    pool = Pool(workers)
    imap_func = pool.imap_unordered if not assert_ordered else pool.imap
    results = []
    for result in tqdm(imap_func(partial_function, iterable), total=total):
        results.append(result)
    pool.close()
    pool.join()

    return results


def _worker_fn(item, fn):
    """
    Wrapper for worker method in order to get original exception from
    a worker process and to log correct exception stacktrace.

    :param item: item from iterable
    :param fn: partial function to be evaluated
    :type fn: :class:`_functools.partial`
    """
    try:
        return fn(item)
    except Exception as e:
        logging.exception(e)
        raise
