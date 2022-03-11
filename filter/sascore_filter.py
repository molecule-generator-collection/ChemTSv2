import sys
sys.path.append("./data/")
import sascorer

from filter.filter import Filter


class SascoreFilter(Filter):
    def check(mol, conf):
        return conf['sascore_filter']['threshold'] > sascorer.calculateScore(mol)