"""
Phase Retrieval Codes for NSLS-II Coherent Hard X-ray beamline

Adapted from: 

M180G Phase Retrieval
Author: AJ Pryor
Jianwei (John) Miao Coherent Imaging Group
University of California, Los Angeles
Copyright (c) 2016. All Rights Reserved.

Modified by: Yuan Hung Lo (2017)

"""

    
    
    
from PIL import Image
import numpy as np
import time
import matplotlib.pyplot as plt
from IPython import display
#from mask_maker import *



# This defines a decorator to time other functions
def timeThisFunction(function):
    from time import time
    def inner(*args, **kwargs):
        t0 = time()
        value = function(*args,**kwargs)
        print("Function call to \"{0:s}\" completed in {1:.3f} seconds".format(function.__name__,time()-t0))
        return value
    return inner

class Initializer(object):
    """File loader class for initializing data for reconstruction at CHX."""

    import h5py

    def __init__(self, fdir, px_size, probe_radius, arr_size):
        self.fdir = fdir
        self.px_size = px_size
        self.probe_radius = probe_radius
        self.length = arr_size
        self.cen = int(np.floor(self.length/2))

    # def loadPatterns(self):
    #     """ import data """
    #     self.data =
    #     self.length = self.data.shape[0]

    # def loadPositions(self):

    def makeProbeGuess(self):
        return makeCircle(self.probe_radius, self.length, self.cen, self.cen)


class DiffractionPattern(object):
    "Base class for containing a diffraction pattern"

    def __init__(self, data):
        "Build a diffraction pattern object from an existing numpy array representing the diffraction data"

        self.data = data
        dims = np.shape(self.data) # get initial dimensions of array


    @staticmethod # static method means the following is just a regular function inside of the class definition
    def fromTIFF(filename):
        "Factory method for constructing a DiffractionPattern object from a tiff image"
        from PIL import Image
        data = np.array(Image.open(filename), dtype=float) # load in data
        # return DiffractionPattern(np.array(data,dtype=float)) #return a new DiffractionPattern object
        return data  # return a new DiffractionPattern object

    @staticmethod
    def fromTIFFstack(directory_name, filestart):
        import os
        import glob
        from PIL import Image
        # data = np.array([np.array(Image.open(filename)) for filename in os.listdir(directory_name)], dtype=float)
        data = np.array([np.array(Image.open(filename)) for filename in glob.glob(directory_name + filestart + '*.tiff')])
        return data

    @timeThisFunction # adds a timer to the function
    def maskBrightValues(self, threshold_fraction = 0.98):
        # flag pixels at or near saturation limit as unmeasured
        self.data [self.data > (threshold_fraction * np.amax(self.data))] = -1

    @timeThisFunction # adds a timer to the function
    def hermitianSymmetrize(self):
        """Enforce Hermitian symmetry (centrosymmetry) on the diffraction pattern"""
        print ("Using the slower version of hermitianSymmetrize , this may take a while....")

        # define center, "//" performs floor division. This idiom works regardless
        # of whether the dimension is odd or even.
        dimX, dimY = np.shape[ self.data]
        centerX = dimX//2
        centerY = dimY//2

        # Now we want to loop over the array and combine each pixel with its symmetry mate. If
        # the dimension of our array is even, then the pixels at position "0" do not have a symmetry mate,
        # so we must check for that. Otherwise we will get an error for trying to index out of bounds
        if dimX % 2 == 0: # the "%" performs modular division, which gets the remainder
            startX = 1
        else:
            startX = 0

        if dimY % 2 == 0:
            startY = 1
        else:
            startY = 0

        # Now that we have the housekeeping stuff out of the way, we can actually do the loop
        # We have to keep up with two sets of coordinates -> (X,Y) refers to the
        # position of where the value is located in the array and counts 0, 1, 2, etc.
        # On the other hand,(centeredX, centeredY) gives the coordinates relative to the origin
        # so that we can find the Hermitian symmetry mate, which is at (-centeredX, -centeredY)
        for X in range(startX, dimX):
            for Y in range(startY, dimY):

                # for each pixel X, Y, get the centered coordinate
                centeredX = X - centerX
                centeredY = Y - centerY

                # get the array coordinate of the symmetry mate and shift back by the center
                symmetry_X = (-1 * centeredX) + centerX
                symmetry_Y = (-1 * centeredY) + centerY

                # get the values from the array
                val1 = self.data[X, Y]
                val2 = self.data[symmetry_X, symmetry_Y]

                # if both values exist, take their average. If only one exists, use it for both. If
                # neither exists, then the final value is unknown (so we do nothing)

                if (val1 != -1) and (val2 != -1): #if both exist, take the average
                    self.data[X, Y] = (val1 + val2) / 2
                    self.data[symmetry_X, symmetry_Y] = (val1 + val2) / 2
                elif (val1 == -1): #if val1 does not exist, use val2 for both
                    self.data[X, Y] = val2
                    self.data[symmetry_X, symmetry_Y] = val2
                else: #then val2 must not exist
                    self.data[X, Y] = val1
                    self.data[symmetry_X, symmetry_Y] = val1
                self.data[self.data == 0] = -1

    @timeThisFunction # adds a timer to the function
    def hermitianSymmetrize_express(self):
        """
        Functions the same as hermitianSymmetrize, except is ~10,000 times faster, but more cryptic

        Applies Hermitian symmetry (centrosymmetry) to the diffraction pattern. If one symmetry mate is not equal to the complex conjugate of the other
        their average is taken. If only one of them exists (is nonzero), then the one value is used. If neither exists
        the value remains 0. In terms of implementation, this function produces Hermitian symmetry by adding the object
        to its complex conjugate with the indices reversed. This requires the array to be odd, so there is also a check
        to make the array odd and then take back the original size at the end, if necessary.

        """
        startDims = np.shape(self.data) # initial dimensions

        # remember the initial dimensions for the end
        dimx = startDims[0]
        dimy = startDims[1]
        flag = False # flag to trigger copying to new odd dimensioned array

        #check if any dimension is odd
        if dimx % 2 == 0:
            dimx += 1
            flag = True

        if dimy % 2 == 0:
            dimy += 1
            flag = True

        if flag: # if any dimensions are even, create a new with all odd dimensions and copy array
            newInput = np.zeros((dimx,dimy), dtype=float) #new array
            newInput[:startDims[0], :startDims[1]] = self.data # copy values
            newInput[newInput == -1] = 0
            numberOfValues = (newInput != 0).astype(float) #track number of values for averaging
            newInput = newInput + newInput[::-1, ::-1] # combine Hermitian symmetry mates
            numberOfValues = numberOfValues + numberOfValues[::-1, ::-1] # track number of points included in each sum
            newInput[numberOfValues != 0] =  newInput[numberOfValues != 0] / numberOfValues[numberOfValues != 0] # take average where two values existed
            self.data = newInput[:startDims[0], :startDims[1]] # return original dimensions
        else: # otherwise, save yourself the trouble of copying the matrix over. See previous comments for line-by-line
            self.data[self.data == -1] = 0 #temporarily remove flags
            numberOfValues = (self.data != 0).astype(int)
            self.data = self.data + self.data[::-1, ::-1]
            numberOfValues = numberOfValues + numberOfValues[::-1, ::-1]
            self.data[numberOfValues != 0] = self.data[numberOfValues != 0] / numberOfValues[numberOfValues != 0]
        self.data[self.data == 0] = -1 # reflag

    @timeThisFunction # adds a timer to the function
    def correctCenter(self,search_box_half_size = 5):
        "This method optimizes the location of the diffraction pattern's center and shifts it accordingly \
         It does so by searching a range of centers determined by search_box_half_size. For each center, the \
         error between centrosymmetric partners is checked. The optimized center is the position which    \
         minimizes this error"
        import matplotlib.pyplot as plt

        plt.figure()
        plt.imshow(self.data)
        plt.title('Double-click the center')
        # plot.show()
        center_guess_y, center_guess_x = (plt.ginput(1)[0])
        center_guess_x = int(center_guess_x)
        center_guess_y = int(center_guess_y)
        plt.close()

        dimX, dimY = np.shape(self.data)
        # If guesses for the center aren't provided, use the center of the array as a guess

        originalDimx = dimX
        originalDimy = dimY

        if center_guess_x is None:
            center_guess_x = dimX // 2

        if center_guess_y is None:
            center_guess_y = dimY // 2

        bigDimX = max(center_guess_x,originalDimx-center_guess_x-1)
        bigDimY = max(center_guess_y,originalDimy-center_guess_y-1)

        padding_1_x = abs(center_guess_x-bigDimX) + search_box_half_size
        padding_2_x = abs( (originalDimx - center_guess_x - 1) - bigDimX)+ search_box_half_size

        padding_1_y = abs(center_guess_y-bigDimY)+ search_box_half_size
        padding_2_y = abs( (originalDimy - center_guess_y - 1) - bigDimY)+ search_box_half_size


        self.data = np.pad(self.data,((padding_1_x, padding_2_x),(padding_1_y, padding_2_y)),mode='constant')
        dimx, dimy = np.shape(self.data) # initial dimensions
        startDims = (dimx, dimy)
        center_guess_x = dimx//2
        center_guess_y = dimy//2

        flag = False # flag to trigger copying to new odd dimensioned array

        #check if any dimension is odd
        if dimx % 2 == 0:
            dimx += 1
            flag = True

        if dimy % 2 == 0:
            dimy += 1
            flag = True

        if flag: # if any dimensions are even, create a new with all odd dimensions and copy array
            temp_data = np.zeros((dimx,dimy), dtype=float) #new array
            temp_data[:startDims[0], :startDims[1]] = self.data # copy values
            input = temp_data

        else:
            temp_data = self.data

        temp_data[temp_data == -1 ] = 0 # remove flags

        #initialize minimum error to a large value
        best_error = 1e30

        #initialize the best shifts to be 0
        bestShiftX = 0
        bestShiftY = 0

        #loop over the various center positions
        for xShift in range(-search_box_half_size,search_box_half_size+1):
            for yShift in range(-search_box_half_size,search_box_half_size+1):

                #shift the data
                temp_array = np.roll(temp_data,xShift,axis=0)
                temp_array = np.roll(temp_array,yShift,axis=1)
                temp_array_reversed = temp_array[::-1, ::-1]

                numberOfValues = (temp_array != 0).astype(float)
                numberOfValues =  numberOfValues + numberOfValues[::-1, ::-1]
                difference_map = np.abs(temp_array - temp_array_reversed)

                normalization_term = np.sum(abs(temp_array[numberOfValues == 2]))
                error_between_symmetry_mates = np.sum(difference_map[numberOfValues == 2]) / normalization_term
                if error_between_symmetry_mates < best_error:
                    best_error = error_between_symmetry_mates
                    bestShiftX = xShift
                    bestShiftY = yShift
        self.data = np.roll(self.data, bestShiftX, axis=0)
        self.data = np.roll(self.data, bestShiftY, axis=1)
        self.data = self.data[ search_box_half_size : -search_box_half_size, search_box_half_size:-search_box_half_size ]

    @timeThisFunction
    def makeArraySquare(self):
        """ Pad image to square array size that is the nearest even number greater than or equal to the current dimensions"""

        dimx, dimy = np.shape(self.data)
        new_dim = max(dimx,dimy) + (max(dimx,dimy)%2) # Take the ceiling even value above the larger dimension
        padding_x = ((new_dim - dimx)//2, (new_dim - dimx)//2 + (new_dim - dimx)%2 )
        padding_y = ((new_dim - dimy)//2, (new_dim - dimy)//2 + (new_dim - dimy)%2 )
        self.data = np.pad(self.data,(padding_x, padding_y), mode='constant')
        self.data [ self.data ==0] = -1

    @timeThisFunction # adds a timer to the function
    def subtractBackground(self, bg_filename):
        from PIL import Image
        BG = Image.open(bg_filename)
        BG = np.array(BG,dtype=float)
        self.data -= BG # shorthand for self.data = self.data - BG
        self.data[(self.data <= 0 )] = -1 # any negative values are to be flagged as "missing" with a -1

    @timeThisFunction # adds a timer to the function
    def convertToFourierModulus(self):
        self.data[self.data != -1] = np.sqrt(self.data[self.data != -1])

    @timeThisFunction
    def binImage(self, bin_factor_x=1, bin_factor_y=1, fraction_required_to_keep = 0.5):
        # bin an image by bin_factor_x in X and bin_factor_y in Y by averaging all pixels in an bin_factor_x by bin_factor_y rectangle
        # This is accomplished using convolution followed by downsampling, with the downsampling chosen so that the center
        # of the binned image coincides with the center of the original unbinned one.

        from scipy.signal import convolve2d
        self.data [self.data <0 ] = 0 # temporarily remove flags
        numberOfValues = (self.data != 0).astype(int) # record positions that have a value
        binning_kernel = np.ones((bin_factor_x, bin_factor_y), dtype=float) # create binning kernel (all values within this get averaged)
        self.data = convolve2d(self.data, binning_kernel, mode='same') # perform 2D convolution
        numberOfValues = convolve2d(numberOfValues, binning_kernel, mode='same') # do the same with the number of values
        self.data[ numberOfValues > 1 ] = self.data[ numberOfValues > 1 ] / numberOfValues[ numberOfValues > 1 ] # take average, accounting for how many datapoints went into each point
        self.data[ numberOfValues < (bin_factor_x * bin_factor_y * fraction_required_to_keep)] = -1 # if too few values existed for averaging because too many of the pixels were unknown, make the resulting pixel unknown
        dimx, dimy = np.shape(self.data) # get dimensions
        centerX = dimx//2 # get center in X direction
        centerY = dimy//2 # get center in Y direction

        # Now take the smaller array from the smoothed large one to obtain the final binned image. The phrase "centerX % bin_factor_x"
        # is to ensure that the subarray we take includes the exact center of the big array. For example if our original image is
        # 1000x1000 then the central pixel is at position 500 (starting from 0). If we are binning this by 5 we want a 200x200 array
        # where the new central pixel at x=100 corresponds to the old array at x=500, so "centerX % bin_factor_x" ->
        # 500 % 5 = 0, so we would be indexing 0::5 = [0, 5, 10, 15..., 500, 505...] which is what we want. The same scenario with a
        # 1004x1004 image needs the center of the 200x200 array to be at x=502, and 502 % 5 = 2 and we index
        # 2::5 = [2,7,12..., 502, 507 ...]
        self.data = self.data[ centerX % bin_factor_x :: bin_factor_x, centerY % bin_factor_y :: bin_factor_y ]


class PtychographyReconstruction(object):
    ''' ptychography reconstruction object for reconstruction using ePIE '''

    def __init__(self, diffraction_pattern_stack, aperture_positions, reconstructed_pixel_size = 1, num_iterations = 100, aperture_guess = None, initial_object = None, show_progress = True, diffraction_pattern_type='generator' ):
        '''diffraction_pattern_type='generator': diffraction_pattern_stack is a generator, shape will be N, Dimx, Dimy
           diffraction_pattern_type='images': diffraction_pattern_stack is a numpy array, shape will be  Dimx, Dimy, N
        '''
        
        self.diffraction_pattern_type  = diffraction_pattern_type
        self.diffraction_pattern_stack = diffraction_pattern_stack # NumPy array of dimension N x N x number_of_patterns
        self.num_iterations = num_iterations
        self.aperture_guess = aperture_guess # initial guess of the aperture
        self.show_progress = show_progress
        if self.diffraction_pattern_type == 'images':
            dp_dimX, dp_dimY, number_of_patterns = np.shape(self.diffraction_pattern_stack)             
        else:
            img = self.diffraction_pattern_stack[0]
            dp_dimX, dp_dimY, number_of_patterns = img.shape[1], img.shape[0], len( self.diffraction_pattern_stack) 
        self.dp_dimX,self.dp_dimY,self.number_of_patterns = dp_dimX, dp_dimY, number_of_patterns
        # Adjust the aperture positions. Convert into pixels, center the origin at 0 and
        # add an offset of size (dp_dimX, dp_dimY) as a small buffer
        aperture_pos_X, aperture_pos_Y = zip(*aperture_positions)
        min_x_pos = min(aperture_pos_X) / reconstructed_pixel_size
        min_y_pos = min(aperture_pos_Y) / reconstructed_pixel_size
        aperture_pos_X = [int(pos/reconstructed_pixel_size - min_x_pos) + dp_dimX for pos in aperture_pos_X]
        aperture_pos_Y = [int(pos/reconstructed_pixel_size - min_y_pos) + dp_dimY for pos in aperture_pos_Y]
        self.aperture_positions = [pair for pair in zip(aperture_pos_X, aperture_pos_Y)]
        self.number_of_apertures = len(self.aperture_positions)
        # determine size of the macro reconstruction
        big_dim_X, big_dim_Y = max(aperture_pos_X) + dp_dimX, max(aperture_pos_Y)+ dp_dimY

        # Initialize array to hold reconstruction
        self.reconstruction = np.zeros((big_dim_X, big_dim_Y), dtype=complex)
        self.display_results_during_reconstruction = False

        if aperture_guess is None:
            self.aperture = np.ones((dp_dimX, dp_dimX), dtype=complex)
        else:
            self.aperture = aperture_guess
        # If no initial object was provided, default to zeros
        if initial_object is None:
            self.initial_object = np.zeros((big_dim_X, big_dim_Y), dtype=complex)
        else:
            self.initial_object = initial_object

    def reconstruct(self):
        aperture_update_start = 5 # start updating aperture on iteration 5
        beta_object = 0.9
        beta_aperture = 0.9
        dp_dimX, dp_dimY, number_of_patterns   = self.dp_dimX,self.dp_dimY,self.number_of_patterns  
        x_crop_vector = np.arange(dp_dimX) - dp_dimX//2
        minX,maxX = np.min(x_crop_vector), np.max(x_crop_vector) + 1
        y_crop_vector = np.arange(dp_dimY) - dp_dimY//2
        minY,maxY = np.min(y_crop_vector), np.max(y_crop_vector) + 1


        for iteration in range(self.num_iterations):
            # randomly loop over the apertures each iteration
            for cur_apert_num in np.random.permutation(range(self.number_of_apertures)):
                # crop out the relevant sub-region of the reconstruction
                x_center, y_center = self.aperture_positions[cur_apert_num][0] , self.aperture_positions[cur_apert_num][1]
                r_space = self.reconstruction[ minX+x_center:maxX+x_center, minY+y_center:maxY+y_center ]
                buffer_r_space = np.copy(r_space)

                buffer_exit_wave = r_space * self.aperture
                update_exit_wave = my_fft(np.copy(buffer_exit_wave))
                if self.diffraction_pattern_type == 'images':                                         
                    current_dp =  self.diffraction_pattern_stack[:, :, cur_apert_num]  
                else:
                    current_dp = self.diffraction_pattern_stack[ cur_apert_num ]   
                    
                current_dp = np.sqrt(    current_dp  )   #from intensity to amplitude
                
                update_exit_wave[ current_dp != -1 ] = abs(current_dp[current_dp != -1])\
                                                     * np.exp(1j*np.angle(update_exit_wave[current_dp != -1]))

                update_exit_wave = my_ifft(update_exit_wave)
                
                # max_ap = np.max(np.abs(self.aperture))
                # norm_factor = beta_object / max_ap**2
                diff_wave = (update_exit_wave - buffer_exit_wave)
                new_r_space = buffer_r_space + diff_wave * \
                                    np.conjugate(self.aperture) * beta_object / np.max(np.abs(self.aperture))**2
                self.reconstruction[ minX+x_center:maxX+x_center, minY+y_center:maxY+y_center ] = new_r_space

            if iteration > aperture_update_start:
                # norm_factor_apert = beta_aperture / np.max(np.abs(r_space))**2
                print('Update probe here')
                self.aperture = self.aperture + beta_aperture / np.max(np.abs(r_space))**2 * \
                    np.conjugate(buffer_r_space)*diff_wave

            if iteration % 5 == 0 and iteration != self.num_iterations - 1:
                print("Iteration {}".format(iteration))
                self.displayResult()

        return self.reconstruction

    def displayResult(self):
        fig = plt.figure(101)
        plt.subplot(221)
        plt.imshow(np.abs(self.reconstruction),origin='lower')
        plt.draw()
        plt.title('Reconstruction Magnitude')
        plt.subplot(222)
        plt.imshow(np.angle(self.reconstruction),origin='lower')
        plt.draw()
        plt.title('Reconstruction Phase')
        plt.subplot(223)
        plt.imshow(np.abs(self.aperture),origin='lower')
        plt.title("Aperture Magnitude")
        plt.draw()
        plt.subplot(224)
        plt.imshow(np.angle(self.aperture),origin='lower')
        plt.title("Aperture Phase")
        plt.draw()
        fig.canvas.draw()
        
        
        #fig.tight_layout()
        # display.display(fig)
        # display.clear_output(wait=True)
        # time.sleep(.00001)

    def displayResult2(self):
        
        fig = plt.figure() 
        ax = fig.add_subplot(2,2,1 ) 
        ax.imshow(np.abs(self.reconstruction))         
        ax.set_title('Reconstruction Magnitude')

        ax = fig.add_subplot(2,2,2 ) 
        ax.imshow(np.angle(self.reconstruction))        
        ax.set_title('Reconstruction Phase')

        ax = fig.add_subplot(2,2,3 ) 
        ax.imshow(np.abs(self.aperture))
        ax.set_title("Aperture Magnitude")
        
        ax = fig.add_subplot(2,2,4 ) 
        ax.imshow(np.angle(self.aperture))
        ax.set_title("Aperture Phase")
        fig.tight_layout()
        
        
class PostProcesser(object):
    ''' class for visualizing and selecting area of interest for in situ CDI measurements after obtaining the 
    ptychography results '''

    def __init__(self, fov, positions, probe_radius, arr_size):
        self.fov = fov
        self.positions = positions
        self.probe_radius = probe_radius
        self.arr_size = arr_size

    def fovPicker(self):
        fig = plt.figure()
        ax = plt.gca()
        plt.imshow(abs(self.fov))
        for n in np.arange(self.positions.shape[0]):
            circle = plt.Circle(self.positions[n, :], self.probe_radius, color='r', fill=False)
            ax.add_artist(circle)
        plt.title("After closing figure, enter desired position for in situ CDI measurement")
        fig.canvas.draw()

        index = input("Enter desired position for in situ CDI measurement: ")
        #print(index)
        #index = index.astype(int);
        index = int(index)

        self.fov_roi = crop_roi(image = self.fov,
                           crop_size = self.arr_size,
                           cenx = self.positions[index,1],
                           ceny = self.positions[index,0])

        return self.fov_roi

    def defineVerts(self):
        coords = []
        fig = plt.figure()
        ax = fig.add_subplot(111)
        plt.imshow(self.fov_roi)
        ax.set_title('click to build line segments')
        line, = ax.plot([], [])  # empty line
        LineBuilder(line)
        plt.show()

    def makeStatSupp(self, coords):
        h, w = (self.arr_size, self.arr_size)
        y, x = np.mgrid[:h, :w]
        points = np.transpose((x.ravel(), y.ravel()))
        p = path.Path(coords)
        mask = p.contains_points(points)
        return mask.reshape(h, w)

class InSituCDIReconstruction(object):
    """ in situ CDI reconstruction class """
    def __init__(self,diffraction_patterns, num_iterations = 200, stat_support = None, probe = None, initial_object = None):
        self.diffraction_patterns = diffraction_patterns
        self.num_iterations = num_iterations
        self.probe = probe
        self.stat_support = stat_support

        print(np.shape(self.diffraction_patterns))
        dimZ, dimY, dimX = np.shape(self.diffraction_patterns.data)

        self.reconstructions = np.zeros((dimZ,dimY,dimX), dtype=float)
        self.display_results_during_reconstruction = True

        if initial_object is None:
            self.initial_object = np.array(np.random.rand(dimZ,dimY,dimX), dtype=complex)
            # self.initial_object = np.array(np.zeros((dimZ,dimY,dimX)), dtype=complex)
        else:
            self.initial_object = initial_object

    def reconstruct(self):
        # np.seterr(divide='ignore', invalid='ignore')
        print("Reconstructing dynamic information with in situ CDI...")

        # Get dimensions of diffraction patterns
        dimZ, dimY, dimX = np.shape(self.diffraction_patterns)

        # set measured data points mask
        measured_data_mask = self.diffraction_patterns != -1
        ref_data_mask = self.stat_support == 1
        probe = np.copy(self.probe)

        # initialize object
        R = self.initial_object * 1e-10

        # initialize reference area
        ref_area = np.ones((dimY,dimX)) * 1e-6
        kerr_top = np.zeros(dimZ)
        kerr_bottom = np.zeros(dimZ)

        self.errK = np.zeros(self.num_iterations, dtype=float)
        best_error = 1e30
        beta = 0.9
        beta_ref = 0.8
        alpha = 1e-10

        for itr in range(self.num_iterations):
            if itr % 10 == 1:
                print("Iteration{0} \t\t Minimum Error = {1:.2f}".format(itr, best_error))

            for frame in range(10):
                current_frame = np.copy(R[frame, :, :])
                current_data_mask = measured_data_mask[frame, :, :]

                current_frame[ref_data_mask == 1] = beta_ref * ref_area[ref_data_mask == 1] + (1-beta_ref) * current_frame[ref_data_mask == 1]

                current_exit_wave = current_frame * probe

                updated_k = my_fft(current_exit_wave)
                check_k = abs(updated_k)

                measured_k = np.copy(self.diffraction_patterns[frame, :, :])
                updated_k[current_data_mask] = measured_k[current_data_mask] * np.exp(1j * np.angle(updated_k[current_data_mask]))
                # updated_k[current_data_mask] = measured_k[current_data_mask] * updated_k[current_data_mask] / np.abs(updated_k[current_data_mask])

                updated_exit_wave = my_ifft(updated_k)

                diff_exit_wave = updated_exit_wave - current_exit_wave
                update_fn = (np.abs(probe) / np.amax(abs(probe))) * (np.conj(probe) / (alpha + np.power(abs(probe),2)))
                updated_object = current_frame + update_fn * diff_exit_wave

                ref_area = np.copy(updated_object)
                R[frame, :, :] = np.copy(updated_object)

                kerr_top[frame] = np.sum(abs(measured_k[current_data_mask] - check_k[current_data_mask]))
                kerr_bottom[frame] = np.sum(abs(measured_k[current_data_mask]))

            # check R-factor
            errK = np.sum(kerr_top)/np.sum(kerr_bottom)
            self.errK[itr] = errK
            if errK < best_error:
                best_error = errK
                self.reconstructions = np.copy(R)

            if self.display_results_during_reconstruction & (itr % 10 == 0):
                self.displayResults()

    def displayResults(self):
        from matplotlib import pyplot as plt
        plt.figure(2)
        plt.subplot(121)
        plt.imshow(abs(self.reconstructions[5, :, :]), cmap='gray')
        plt.subplot(122)
        plt.plot(range(self.num_iterations), self.errK, 'ko')
        plt.draw()
        plt.pause(1e-12)


def my_fft(arr):
    #computes forward FFT of arr and shifts origin to center of array
    return np.fft.fftshift(np.fft.fftn(np.fft.ifftshift(arr)))

def my_ifft(arr):
    #computes inverse FFT of arr and shifts origin to center of array
    return np.fft.fftshift(np.fft.ifftn(np.fft.ifftshift(arr)))

def combine_images(directory_name, output_filename = None):
    """loop over all tiff images in directory_name and average them"""
    import os
    from PIL import Image
    image_count = 0
    average_image = None
    for filename in os.listdir(directory_name):
        file,ext = os.path.splitext(filename)
        if ext==".tif" or ext==".tiff":
            # print(directory_name + filename)
            if image_count == 0:
                average_image = np.array(Image.open(directory_name + filename),dtype=float)
            else:
                average_image += np.array(Image.open(directory_name + filename),dtype=float)

            image_count+=1
    try:
        average_image/=image_count # take average
    except TypeError:
        print ("\n\n\n\nNO VALID TIFF IMAGES IN DIRECTORY!\n\n\n")
        raise
    if output_filename is not None:
        np.save(output_filename,average_image)
    return average_image

def makeCircle(radius, img_size, cenx, ceny):
    """Make binary circle with specified radius and image size (img_size) centered at
    (cenx,ceny)"""

    cen = np.floor((img_size / 2)).astype(int)
    xx, yy = np.mgrid[-cen:cen, -cen:cen]
    out = np.sqrt(xx ** 2 + yy ** 2) <= radius
    out = np.roll(out, cenx - cen, axis=1)
    out = np.roll(out, ceny - cen, axis=0)
    return out

def crop_roi(image, crop_size, cenx, ceny):
    
    crop_size_x = crop_size
    crop_size_y = crop_size

    # cenx = int(np.floor(image.shape[0]/2))
    # ceny = int(np.floor(image.shape[1]/2))
    
    cenx = int(cenx)
    ceny = int(ceny)
    
    half_crop_size_x = np.floor(crop_size_x//2)
    half_crop_size_y = np.floor(crop_size_y//2)

    if crop_size % 2 ==0:        
        cropped_im = image[ int(ceny - half_crop_size_y): int(ceny + half_crop_size_y),
                   int(cenx - half_crop_size_x): int(cenx + half_crop_size_x)]
    else:
        cropped_im = image[ int(ceny - half_crop_size_y): int(ceny + half_crop_size_y + 1),
                   int(cenx - half_crop_size_x): int(cenx + half_crop_size_x + 1)]

    return cropped_im

def createSupp(arr_size, coords):
    h, w = arr_size
    y, x = np.mgrid[:h, :w]
    points = np.transpose((x.ravel(), y.ravel()))
    p = path.Path(coords)
    mask = p.contains_points(points)
    return mask.reshape(h, w)

def ptychography_demo():
    # create simulated ptychography data
    obj = Image.open('images/lena.png')
    obj = np.pad(obj, pad_width=50, mode='constant')

    ptychography_obj = simulate.make_ptychography_data(big_obj=obj, ccd_size=300, scan_dim=6, offset=0.15)  # pixels
    positions = ptychography_obj.gridscan()
    probe = ptychography_obj.make_probe(probe_radius=70, dx=4, dy=4, z=-1000, wavelength=.635)  # units in microns
    obj_stack = ptychography_obj.make_obj_stack(positions)
    diff_patterns = ptychography_obj.make_dps(obj_stack, probe)

    # display simulation data
    fig = plt.figure(1)
    plt.subplot(221)
    plt.imshow(obj)
    plt.title('Object full field of view')
    plt.subplot(222)
    plt.imshow(abs(probe))
    plt.title('Probe (magnitude)')
    for n in range(max(positions.shape)):
        plt.subplot(223)
        plt.imshow(abs(obj_stack[:, :, n] * probe))
        plt.title('Exit wave at position %d' % (n))
        plt.subplot(224)
        plt.imshow(np.log(diff_patterns[:, :, n]))
        plt.title('Diffraction intensity')
        plt.pause(.1)
    plt.draw()
    plt.pause(1)
    input("<Hit Enter Twice To Close Figure and Continue>")
    plt.close(fig)

    # create ptychography reconstruction object
    ptyc_reconstruction = PtychographyReconstruction(
        diffraction_pattern_stack=diff_patterns,
        aperture_guess=probe,
        initial_object=None,
        num_iterations=100,
        aperture_positions=positions,
        reconstructed_pixel_size=1,
        show_progress=True)

    # reconstruct ptychography
    ptyc_reconstruction.reconstruct()
    ptyc_reconstruction.displayResults()

if __name__ == '__main__':
    ptychography_demo()
