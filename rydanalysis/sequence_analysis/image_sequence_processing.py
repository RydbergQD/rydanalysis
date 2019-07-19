import numpy as np
import pandas as pd

def calc_mean_seq(images):
    return np.nanmean(images,axis=0)

def calc_std_seq(images):
    return np.nanstd(images,axis=0)

def remove_fringes(image, base):
    raise  NotImplementedError("Fringe removal is not yet implemented")

def stderr_weighted_average(g):
    rel_err = g.amp.stderr/g.amp.value
    weights = 1/rel_err
    return (g.image_od * weights).sum()/weights.sum()

def fittoSeries(fit):
    p = fit.params
    dict_p = {key : par2dict(p[key]) for key in p}
    flat_dict = flatten_dict(dict_p)
    df = pd.DataFrame(flat_dict,index=[0])
    #ps = pd.Series(flatten_dict(dict_p))
    return df.squeeze()

def iteritems_nested(d):
    def fetch (suffixes, v0) :
        if isinstance(v0, dict):
            for k, v in v0.items() :
                for i in fetch(suffixes + [k], v):  # "yield from" in python3.3
                    yield i
        else:
            yield (suffixes, v0)
    
    return fetch([], d)

def flatten_dict(d) :
    return { tuple(ks): v for ks, v in iteritems_nested(d)}

def par2dict(p):
    return dict(
        value = p.value,
        min = p.min,
        max = p.max,
        init_value = p.init_value,
        stderr = p.stderr,
        #correl = p.correl,
        vary = p.vary,
    )
