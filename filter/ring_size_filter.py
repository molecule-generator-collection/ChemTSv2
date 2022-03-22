from filter.filter import Filter


class RingSizeFilter(Filter):
    def check(mol, conf):
        ri = mol.GetRingInfo()
        max_ring_size = max((len(r) for r in ri.AtomRings()), default=0)
        if 'threshold' in conf['ring_size_filter'].keys():
            return max_ring_size <= conf['ring_size_filter']['threshold']

        elif 'min_threshold' in conf['ring_size_filter'].keys() and 'max_threshold' in conf['ring_size_filter'].keys():
            min_ring_size = min((len(r) for r in ri.AtomRings()), default=100)
            return max_ring_size <= conf['ring_size_filter']['max_threshold'] and min_ring_size >= conf['ring_size_filter']['min_threshold']
