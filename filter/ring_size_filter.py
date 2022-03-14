from filter.filter import Filter


class RingSizeFilter(Filter):
    def check(mol, conf):
        ri = mol.GetRingInfo()
        max_ring_size = max((len(r) for r in ri.AtomRings()), default=0)
        return max_ring_size <= conf['ring_size_filter']['threshold']
