import numpy as np

# def calc_absorbtion_image(images, bg = None):
#     if bg is None:
#         bg = images[2]
#     return (images[0]-bg)/(images[1]-bg)

def absorbtion_to_OD(image):
    return -np.log(1-image)

def crop_image(image,xslice=slice(0,-1),yslice=slice(0,-1)):
    return image[(xslice,yslice)]


def calc_absorbtion_image(image, bg = None):
    if bg is None:
        bg = image[5]
    light = image[3]-bg
    atoms = image[1]-bg
    trans = atoms/light
    #trans =interpolate_invalid(trans)
    np.place(trans,trans>=1,1)
    np.place(trans,light==0,1)
    np.place(trans,trans<=0,0.0001)
    absorb = 1-trans
    #np.place(absorb,(light<=0)&(atoms>0),0.00001)
    #np.place(absorb,(light<0)&(atoms==0),1.)
    #np.place(absorb,transmission>1,1.)
    #print(absorb.min(),absorb.max())
    return absorb
