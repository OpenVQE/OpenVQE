from sympy import Symbol, IndexedBase, symbols
from pyspark import RDD
from drudge.term import Range, Term, Vec, try_resolve_range
from drudge import Drudge, FockDrudge, PartHoleDrudge, prod_
from drudge.fock import FERMI, CR, AN, UP, DOWN, parse_field_op, _compare_field_ops
from drudge.drudge import Tensor, TensorDef
import functools

class PartActvHoleDrudge (PartHoleDrudge):
    ''' A Drudge for active-space problems. Child of PartHoleDrudge.

    Extra attributes:

        actv_orb : tuple(drudge.term.Range, tuple(sympy.Symbol)) instance
            symbolic range of active orbitals.
            Default range symbols are 'A', 0, na
            Default dummies are x, y, z, w, x1, x2, x3, ..., x50

        rdm : IndexedBase instance
            symbol used for reduced density matrices.

        set_rdm : callable; args: (rdm : IndexedBase, rdm_max_order : integer); returns: None 
            assigns rdm attribute and sets up antisymmetry up to order rdm_max_order. If dbbar == False then rdm_max_order = 0.

        eval_wev : callable; args: (tensor : Tensor); returns: Tensor
            evaluates wave-function expectation value of Tensor

    '''

    DEFAULT_PART_DUMMS = tuple(Symbol(i) for i in 'abcd') + tuple(
        Symbol('a{}'.format(i)) for i in range(50)
    )
    DEFAULT_ACTV_DUMMS = tuple(Symbol(i) for i in 'xyzw') + tuple(
        Symbol('x{}'.format(i)) for i in range(50)
    )
    DEFAULT_HOLE_DUMMS = tuple(Symbol(i) for i in 'ijkl') + tuple(
        Symbol('i{}'.format(i)) for i in range(50)
    )
    DEFAULT_ORB_DUMMS = tuple(Symbol(i) for i in 'pqrs') + tuple(
        Symbol('p{}'.format(i)) for i in range(50)
    )

    def __init__(self, *args, op_label='c',
                 part_orb=(Range('V', 0, Symbol('nv')), DEFAULT_PART_DUMMS),
                 actv_orb=(Range('A', 0, Symbol('na')), DEFAULT_ACTV_DUMMS),
                 hole_orb=(Range('O', 0, Symbol('no')), DEFAULT_HOLE_DUMMS),
                 all_orb_dumms=DEFAULT_ORB_DUMMS, spin=(),
                 one_body=IndexedBase('h'), two_body=IndexedBase('v'),
                 fock=IndexedBase('f'), rdm=IndexedBase('g'), rdm_max_order=4,
                 dbbar=True, **kwargs):
        ''' drudge.PartHoleDrudge.__init__ is too complicated for me to call and modify it straightforwardly. '''

        self.part_range = part_orb[0]
        self.actv_range = actv_orb[0]
        self.hole_range = hole_orb[0]

        ''' I hope this calls drudge.GenMBDrudge.__init__ '''
        super(PartHoleDrudge, self).__init__(*args, exch=FERMI, op_label=op_label,
            orb=(part_orb, actv_orb, hole_orb), spin=spin,
            one_body=one_body, two_body=two_body, dbbar=dbbar, **kwargs)

        self.all_orb_dumms = tuple(all_orb_dumms)
        self.set_name(*self.all_orb_dumms)
        self.add_resolver({
            i: (self.part_range, self.actv_range, self.hole_range) for i in all_orb_dumms
        })
        self.set_rdm (rdm, rdm_max_order, dbbar)
        self.set_name(no=self.hole_range.size)
        self.set_name(na=self.actv_range.size)
        self.set_name(nv=self.part_range.size)
        self.set_tensor_method('eval_wev', self.eval_wev)
        self.set_tensor_method('eval_awev', self.eval_awev)
        self.fock = fock

    def set_rdm (self, rdm, rdm_max_order, dbbar):
        ''' Set up the IndexedBase for rdms, including antisymmetry if requested. '''
        self.rdm = rdm
        for idx in range (0, rdm_max_order):
            if dbbar and idx>0:
                self.set_dbbar_base (rdm, idx+1)
            else:
                self.set_n_body_base (rdm, idx+1)

    def eval_wev (self, tensor: Tensor, actv_dumms=None):
        ''' Evaluates wave-function expectation value of Tensor. The vecs in the active range are normal-ordered and
        replaced with rdm elements. The vecs in the particle and hole ranges are treated as in PartHoleDrudge.'''

        return super ().eval_fermi_vev (self.eval_awev (tensor))

    def eval_awev (self, tensor: Tensor):
        terms = self._eval_awev (tensor)
        return Tensor (self, terms)

    def _eval_awev (self, tensor: Tensor):
        ''' Evaluates the active-orbital part of the expectation value of the wave function without operating on
        the vecs in the particle and hole ranges '''

        count_actv_ops = self.count_actv_ops
        is_conserved = self.is_conserved
        op2rdm = self.op2rdm

        # Filter out the non-number-conserving terms
        terms_to_keep = tensor.terms.filter (lambda x: count_actv_ops (x) == 0)
        nonzero_terms = tensor.terms.filter (is_conserved)

        # Identify the terms with some actv vecs to mess with
        terms_to_proc = nonzero_terms.filter (lambda x: count_actv_ops (x) > 0)

        # Pull the actv vecs to the left and normal-order them
        ordered_terms = self.normal_order (terms_to_proc, comparator=self._actvonly_comparator)

        # Replace actv vecs with rdm elements
        procced_terms = ordered_terms.map (op2rdm)
        return terms_to_keep.union (procced_terms)

    @property
    def _actvonly_comparator (self):
        ''' Get the comparator for the partial normal ordering operation in which particles and holes are ignored and active ops are pushed to the left.'''

        op_parser = self.op_parser
        resolvers = self.resolvers
        inac_ranges = tuple ([self.part_range, self.hole_range])

        return functools.partial(_actvonly_compare_field_ops, op_parser=op_parser, resolvers=resolvers, inac_ranges=inac_ranges, ignore_inac=True)

    @property
    def comparator (self):
        ''' Since neither the creation nor the annihilation operators of the active range(s) annihilate either to the left or to the right, I define
        normal-ordering as pushing all active creation > active annihilation > inactive creation > inactive annihilation '''

        op_parser = self.op_parser
        resolvers = self.resolvers
        inac_ranges = tuple ([self.part_range, self.hole_range])

        return functools.partial(_actvonly_compare_field_ops, op_parser=op_parser, resolvers=resolvers, inac_ranges=inac_ranges, ignore_inac=False)

    @property
    def is_conserved (self):
        ''' A function that checks whether active operators conserve particle number.  Change this in subclasses for other conservation rules (i.e., spin)'''

        _a = self.actv_range
        _o = self.op_parser
        _r = self.resolvers

        def is_conserved (term, actv_range, op_parser, resolvers):
            n_actv = _count_actv_ops (term, actv_range, op_parser, resolvers)
            return n_actv[0] == n_actv[1]

        return functools.partial (is_conserved, actv_range=_a, op_parser=_o, resolvers=_r)

    @property
    def count_actv_ops (self):
        ''' Counts the total number of cr/an operators in the active space'''

        _a = self.actv_range
        _o = self.op_parser
        _r = self.resolvers

        def _count (term: Term, actv_range, op_parser, resolvers):
            n_actv = _count_actv_ops (term, actv_range, op_parser, resolvers)
            return sum (n_actv)

        return functools.partial (_count, actv_range=_a, op_parser=_o, resolvers=_r)

    @property
    def op2rdm (self):
        ''' Exchanges normal-ordered active-space cr/an operators for rdm elements, as a callable that maps a Term to another Term. 
            Change this for spin-summed rdm's, etc., in child classes. '''

        _g = self.rdm
        _a = self.actv_range
        _o = self.op_parser
        _r = self.resolvers

        return functools.partial (_subst_actv_vec_rdm, rdm=_g, actv_range=_a, op_parser=_o, resolvers=_r)

    def set_antisymm_n_body_base (self, base: IndexedBase, n_body: int, n_body2=None):
        ''' Get a base which is symmetric over between-column permutations and antisymmetric over inside-column permutations'''

        n_body = int(n_body)
        n_body2 = n_body if n_body2 is None else int (n_body2)
        n_slots = n_body + n_body2

        begin1 = 0
        end1 = n_body
        begin2 = end1
        end2 = 2 * n_body

        cycl = Perm(
            self._form_cycl(begin1, end1) + self._form_cycl(begin2, end2)
        )
        transp = Perm(
            self._form_transp(begin1, end1) + self._form_transp(begin2, end2)
        )
        transp_acc = NEG if self._exch == FERMI else IDENT
        colum = Perm (self._form_columnar_transp (0, n_slots), transp_acc)
        gens = [cycl, transp, colum]

        self.set_symm(base, gens, valence=n_slots)

        return

    def set_antisymm_dbbar_base (self, base: IndexedBase, n_body: int, n_body2=None):
        ''' Get a base which is antisymmetric over vertical particle-hole interchanges as well as among particles and holes'''
        if n_body==1 and (n_body2 is None or n_body2==1):
            self.set_dbbar_base (base, 2, 0)
            return

        # First part: the same as set_dbbar_base
        n_body = int (n_body)
        n_body2 = n_body if n_body2 is None else int (n_body2)
        n_slots = n_body + n_body2
    
        transp_acc = NEG if self._exch == FERMI else IDENT
        cycl_accs = [
            NEG if self._exch == FERMI and i % 2 == 0 else IDENT
            for i in [n_body, n_body2]
        ]  # When either body is zero, this value is kinda wrong but not used.

        gens = []

        if n_body > 1:
            second_half = list(range(n_body, n_slots))
            gens.append(Perm(
                self._form_cycl(0, n_body) + second_half, cycl_accs[0]
            ))
            gens.append(Perm(
                self._form_transp(0, n_body) + second_half, transp_acc
            ))

        if n_body2 > 1:
            first_half = list(range(0, n_body))
            gens.append(Perm(
                first_half + self._form_cycl(n_body, n_slots), cycl_accs[1]
            ))
            gens.append(Perm(
                first_half + self._form_transp(n_body, n_slots), transp_acc
            ))

        # Second part: vertical columnar permutation.  I ~think~ I only need to add the first one
        if n_body > 0 and n_body2 > 0:
            gens.append (Perm (self._form_columnar_transp(0, n_slots), transp_acc))

        self.set_symm(base, gens, valence=n_slots)

        return        
        
        @staticmethod
        def _form_columnar_transp (begin, end): 
            '''Form a pre-image array with the first particle and the first hole (i.e., first and last idxes) transposed'''
            res = list(range(begin,end))
            before_end = end-1
            res[0] = before_end
            res[before_end] = 0
            return res


def _subst_actv_vec_rdm (term: Term, rdm: IndexedBase, actv_range, op_parser, resolvers):
    parsed = [op_parser (vec, term) for vec in term.vecs]
    ranges = [try_resolve_range (p[2][0], dict(term.sums), resolvers.value) for p in parsed]
    inac_vecs = [vec for r, vec in zip (ranges, term.vecs) if not r==actv_range]
    actv_parsed = [p for r, p in zip (ranges, parsed) if r==actv_range]
    amp = term.amp
    if len (actv_parsed) > 0:
        actv_cr_idxs = sum([[p[2][0]] for p in actv_parsed if p[1]==CR], [])
        actv_an_idxs = sum([[p[2][0]] for p in reversed (actv_parsed) if p[1]==AN], [])
        actv_idxs = actv_cr_idxs + actv_an_idxs
        amp = amp * rdm[actv_idxs]
    return Term (term.sums, amp, inac_vecs)

def _count_actv_ops (term: Term, actv_range, op_parser, resolvers):
    parsed = [op_parser (op, term) for op in term.vecs]
    actv_chars = [p[1] for p in parsed if try_resolve_range (p[2][0], dict(term.sums), resolvers.value) == actv_range]
    n_cr = sum(char == CR for char in actv_chars)
    n_an = sum(char == AN for char in actv_chars)
    return tuple([n_cr, n_an])


# Remember: False means DO TRANSPOSE THE OPERATORS
def _actvonly_compare_field_ops(
        op1: Vec, op2: Vec, term: Term,
        op_parser: FockDrudge.OP_PARSER,
        resolvers, inac_ranges, ignore_inac: bool
):
    '''Wrapper of FockDrudge._compare_field_ops that returns True (do not transpose) when both vecs are in the inactive ranges,
    and pushes vecs in the active ranges to the left (return False if op2 is active and op1 is not, and True if the opposite)
    before wrapping.'''

    label1, char1, indices1 = op_parser(op1, term)
    label2, char2, indices2 = op_parser(op2, term)

    op1inac = try_resolve_range (indices1[0], dict(term.sums), resolvers.value) in inac_ranges
    op2inac = try_resolve_range (indices2[0], dict(term.sums), resolvers.value) in inac_ranges

    if op1inac and op2inac and ignore_inac:
        return True
    elif op1inac == (not op2inac):
        return op2inac
    else:
        return _compare_field_ops (op1, op2, term, op_parser)

class SpinOneHalfPartActvHoleDrudge(PartActvHoleDrudge):

    def __init__(
            self, *args,
            part_orb=(
                    Range('V', 0, Symbol('nv')),
                    PartActvHoleDrudge.DEFAULT_PART_DUMMS + symbols('beta gamma')
            ),
            actv_orb=(
                    Range('A', 0, Symbol('na')),
                    PartActvHoleDrudge.DEFAULT_ACTV_DUMMS + symbols('mu nu')
            ),
            hole_orb=(
                    Range('O', 0, Symbol('no')),
                    PartActvHoleDrudge.DEFAULT_HOLE_DUMMS + symbols('delta epsilon')
            ), spin=(UP, DOWN),
            **kwargs
    ):
        """Initialize the particle-hole drudge."""

        super().__init__(
            *args, spin=spin, dbbar=False,
            part_orb=part_orb, actv_orb=actv_orb, hole_orb=hole_orb, **kwargs
        )


class RestrictedPartActvHoleDrudge(SpinOneHalfPartActvHoleDrudge):
    """Drudge for the particle-hole problems on restricted reference.

    Similar to :py:class:`SpinOneHalfPartHoldDrudge`, this drudge deals with
    particle-hole problems with explicit one-half spin.  However, here the spin
    quantum number is summed symbolically.  This gives **much** faster
    derivations for theories based on restricted reference, but offers less
    flexibility.

    .. attribute:: spin_range

        The symbolic range for spin values.

    .. attribute:: spin_dumms

        The dummies for the spin quantum number.

    .. attribute:: e_"""

    def __init__(
            self, *args,
            spin_range=Range(r'\uparrow\downarrow', 0, 2),
            spin_dumms=tuple(Symbol('sigma{}'.format(i)) for i in range(50)),
            **kwargs
    ):
        """Initialize the restricted particle-hole drudge."""

        super().__init__(
            *args, spin=(spin_range, spin_dumms), **kwargs
        )
        self.add_resolver({
            UP: spin_range,
            DOWN: spin_range
        })

        self.spin_range = spin_range
        self.spin_dumms = self.dumms.value[spin_range]

        sigma = self.dumms.value[spin_range][0]
        p = Symbol('p')
        q = Symbol('q')
        self.e_ = TensorDef(Vec('E'), (p, q), self.sum(
            (sigma, spin_range), self.cr[p, sigma] * self.an[q, sigma]
        ))
        self.set_name(e_=self.e_)


