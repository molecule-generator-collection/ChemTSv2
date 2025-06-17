from chemtsv2.abc import Filter
from chemtsv2.utils import transform_linker_to_mol, attach_fragment_to_all_sites


class RingSizeFilter(Filter):
    def check(mol, conf):
        ri = mol.GetRingInfo()
        max_ring_size = max((len(r) for r in ri.AtomRings()), default=0)
        if "threshold" in conf["ring_size_filter"].keys():
            return max_ring_size <= conf["ring_size_filter"]["threshold"]

        elif (
            "min_threshold" in conf["ring_size_filter"].keys()
            and "max_threshold" in conf["ring_size_filter"].keys()
        ):
            min_ring_size = min((len(r) for r in ri.AtomRings()), default=100)
            return (
                max_ring_size <= conf["ring_size_filter"]["max_threshold"]
                and min_ring_size >= conf["ring_size_filter"]["min_threshold"]
            )


class RingSizeFilterForXMol(Filter):
    def check(mol, conf):
        @transform_linker_to_mol(conf)
        def _check(mol, conf):
            return RingSizeFilter.check(mol, conf)

        return _check(mol, conf)


class RingSizeFilterForDecoration(Filter):
    def check(mol, conf):
        @attach_fragment_to_all_sites(conf)
        def _check(mol, conf):
            return RingSizeFilter.check(mol, conf)

        return _check(mol, conf)
