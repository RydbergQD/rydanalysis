from sys import path, argv
import os
import xarray as xr
from prefect import task
path.insert(0, '/home/david/Institut/data-analysis/rydanalysis/')
from rydanalysis import *
from rydanalysis.single_shot.image_processing import *


@task
def create_analysis_dir(seq_path, analysis_root):
    analysis_dir = Directory(os.path.join(analysis_root,seq_path))
    return analysis_dir.path

@task
def create_new_data_dir(data_root, seq_path):
    data_dir = Directory(os.path.join(data_root, seq_path))
    return data_dir.path


def old_to_new_datastructure(in_dir, out_dir):
    old_scan = OldStructure(in_dir)
    old_scan.create_new(out_dir)
    
    return out_dir


def calc_batch_transmission(seq_dir, out_dir,
                            remove_fringes=True,
                            replace_invalid=True,
                            ref_basis=10,
                            mask=None,
                            mask_kwargs=None,
                            xslice=slice(None),
                            yslice=slice(None)):
    
    """Calculate the transmission single_image for a sequence of raw experimental data

    Args:

    :param seq_dir: directory of the raw data in the new format
    :param out_dir: directory to store the calculated transmission single_image
    :param remove_fringes: 
    :param replace_invalid: 
    :param ref_basis: 
    :param mask: 
    :param mask_kwargs: 
    :param xslice: 
    :param yslice: 

    """
     
    seq = ExpSequence(seq_dir)
    im = seq.raw_data.get_images()
    im = im.isel({'x':xslice,'y':yslice})
    bg = im['image_05']
    ref = im['image_03']-bg
    atoms = im['image_01']-bg
    
    if remove_fringes:
        if mask=='elliptical':
            mask = elliptical_mask(ref[0].shape,**mask_kwargs)
        refimages = agnostic_select(ref,ref_basis).values
        B_inv, R = prepare_ref_basis(refimages,mask=mask)
        ref = xr.apply_ufunc(calc_ref_image,
                         atoms,
                         kwargs=dict(mask=mask,B_inv=B_inv,R=R),
                         vectorize=True,
                         input_core_dims=[['x','y']],
                         output_core_dims=[['x','y']])
        
    if replace_invalid:
        ref = xr.apply_ufunc(nn_replace_invalid,
                                   ref,
                                   kwargs=dict(invalid=0),
                                   vectorize=True,
                                   input_core_dims=[['x','y']],
                                   output_core_dims=[['x','y']])
        
    trans = atoms.astype(float)/ref
    trans.name='transmission'
    trans = trans.reset_index('concat_dim')
    trans.to_netcdf(os.path.join(out_dir,'transmission.h5'))
    return trans
