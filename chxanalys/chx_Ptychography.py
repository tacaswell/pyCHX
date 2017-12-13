"""
Temporary Function developed during Aug2-Aug5
"""


def create_crop_images(imgs, w, cen, mask):
    cx,cy = np.int_(cen)
    return pims.pipeline(lambda img:  (mask*img)[ cx-w//2:cx+ w//2, cy-w//2:cy+ w//2, ])(imgs) 

def get_reconstruction_pixel_size( wavelength, num_pixel,  dis_det_sam,pixel_size=75.0 ):
    '''Get reconstruction_pixel_size
    Input:
        wavelength: A (float)
        num_pixel: pixel number (int)
        pixel_size: 75 um (Eiger detector)
        dis_det_sam: meter (float)
    Output:
        rec pixel size in nm
    
    '''
    NA = num_pixel*pixel_size * 10**(-6) /(2*dis_det_sam)  
    return wavelength/ (2*NA) /10.



def do_reconstruction(  uid, data_dir0, mask, probe = None, num_iterations =5  ):
    #load data
    data_dir = os.path.join(data_dir0, '%s/'%uid)
    os.makedirs(data_dir, exist_ok=True)
    print('Results from this analysis will be stashed in the directory %s' % data_dir)
    uidstr = 'uid=%s'%uid    
    #md = get_meta_data( uid )
    imgs = load_data( uid, 'eiger4m_single_image', reverse= True  )
    md = imgs.md
    #md.update( imgs.md );Nimg = len(imgs);
    #imgsa = apply_mask( imgs, mask )
    inc_x0 =  imgs[0].shape[0] - md['beam_center_y'] 
    inc_y0=   md['beam_center_x']
    pixel_mask =  1- np.int_( np.array( imgs.md['pixel_mask'], dtype= bool)  )
    mask  *= pixel_mask
    #create diff_patterns
    imgsc = create_crop_images(imgs, w=effective_det_size, cen=[inc_x0, inc_y0], mask=mask)
    diff_patterns = imgsc # np.array(  )
    
    # Scan pattern and get positions
    tab = get_table(db[uid])
    posx = tab['diff_xh']
    posy = tab['diff_yh']
    positions0 = np.vstack( [posx, posy ]).T
    
    positions_ = positions0 - np.array( [positions0[:,0].min(),positions0[:,1].min() ]  )
    positions = np.int_( positions_ *10**6/ rec_pix_size  )
    positions -= positions[positions.shape[0]//2]
    
    # Get Probe
    if probe is None:
        probe = makeCircle(radius = probe_radius, img_size =  effective_det_size, 
                   cenx = effective_det_size//2 , ceny = effective_det_size//2)
        
    ## Create ptychography reconstruction object
    ptyc_rec_obj = PtychographyReconstruction(
        diffraction_pattern_stack = diff_patterns,
        aperture_guess = probe,
        initial_object = None,
        num_iterations = num_iterations,
        aperture_positions = positions,
        reconstructed_pixel_size = 1,
        show_progress = True,
        #diffraction_pattern_type='images'
        )

    # reconstruct ptychography
    reconstruction = ptyc_rec_obj.reconstruct()
    np.save( data_dir +  'uid=%s_probe'%uid, ptyc_rec_obj.aperture)
    np.save( data_dir +  'uid=%s_reconstruction'%uid, reconstruction)     
    return reconstruction, ptyc_rec_obj.aperture
