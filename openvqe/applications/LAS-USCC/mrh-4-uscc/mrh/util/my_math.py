from mrh.util import params
import numpy as np

def round_thresh (the_float, the_thresh):
    round_diff = round (the_float) - the_float
    round_sign = -1 if round_diff<0 else 1
    round_diff = round_sign * (min (abs (round_diff), abs (the_thresh)))
    return the_float + round_diff
    
def is_close_to_integer (the_float, atol=params.num_zero_atol, rtol=params.num_zero_rtol):
    return np.isclose (the_float, round (the_float), atol=atol, rtol=rtol)

