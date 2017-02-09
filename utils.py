import numpy as np

SECTION_DICT = {}
section_stops = [6, 15, 16, 25, 28, 39, 41, 44, 47, 50, 64, 68, 71, 72, 84, 86, 90, 93, 94, 97, 98, 100]
section = 1
for i in xrange(1,100):
    if section_stops[section-1] == i:
        section += 1
    SECTION_DICT[i] = section

def get_section_codes(codes):
    coarse_codes = coarsen_codes(codes)
    return np.array([SECTION_DICT[coarse_code] for coarse_code in coarse_codes])

def coarsen_codes(codes, level=1):
    return codes/np.power(10, 2*(5-level))

def get_coarse_code_index_dict(codes, level=1):
    """Returns dictionary pairing unique, level 'level' codes contained in the official codes with their index in the output embeddings"""
    unique_coarse_category_codes = np.sort(list(set(coarsen_codes(codes))))
    nunique_coarse_category_codes = unique_coarse_category_codes.shape[0]
    unique_code_dict = {code:index for code, index in zip(unique_coarse_category_codes, np.arange(nunique_coarse_category_codes))}
    return unique_code_dict

def get_coarse_index_code_dict(codes, level=1):
    return {index:code for code, index in get_coarse_code_index_dict(codes, level=level).iteritems()}
