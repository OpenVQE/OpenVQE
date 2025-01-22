import sys
import numpy as np
import warnings
import traceback

'''
def warn_with_traceback(message, category, filename, lineno, file=None, line=None):

    log = file if hasattr(file,'write') else sys.stderr
    traceback.print_stack(file=log)
    log.write(warnings.formatwarning(message, category, filename, lineno, line))

warnings.showwarning = warn_with_traceback
'''

def prettyprint_ndarray (mat, fmt='{:9.2e}'):
    mat = np.asarray (mat)
    fmt_str = ' '.join (fmt for col in range (mat.shape[1]))
    return '\n'.join (fmt_str.format (*row) for row in mat)


mcpdft_removal_warning = FutureWarning((
    "Most MC-PDFT and MC-DCFT modules have been moved to pyscf-forge "
    "(github.com/pyscf/pyscf-forge) and will be removed from mrh soon."
    ))
warnings.filterwarnings ("once", message=str(mcpdft_removal_warning),
                         category=FutureWarning)
def mcpdft_removal_warn (): warnings.warn (mcpdft_removal_warning,
                                           stacklevel=3)

lassi_dir_warning = FutureWarning((
    "LASSI code is being moved from my_pyscf.mcscf.lassi_* module files "
    "to my_pyscf.lassi.* module files; the former will be removed soon."
    ))
warnings.filterwarnings ("once", message=str(lassi_dir_warning),
                         category=FutureWarning)
def lassi_dir_warn (): warnings.warn (lassi_dir_warning, stacklevel=3)
