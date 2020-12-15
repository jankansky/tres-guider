#!/usr/bin/python3

import sys
import socket
import os
from configobj import ConfigObj
import logging
import threading
import math
import datetime

import numpy as np
from matplotlib import pyplot as plt
import select
import time
from scipy.ndimage.filters import gaussian_filter

import utils

plt.ion()
plt.switch_backend('agg')

################################################################################
class DynamicPlot():
    #Suppose we know the x range
    min_x = 0
    max_x = 2000
    min_y = 0
    max_y = 255

#-------------------------------------------------------------------------------
    def on_launch(self):
        #Set up plot
        self.figure, self.ax = plt.subplots()
        self.lines, = self.ax.plot([],[], marker=None,linestyle="-")
        #Autoscale on unknown axis and known lims on the other
        self.ax.set_xlim(self.min_x, self.max_x)
        self.ax.set_ylim(self.min_y, self.max_y)
        #Other stuff
        self.ax.grid()
        plt.xlabel("Pixel")
        plt.ylabel("Counts")
        plt.title("Simulated Cam Horizontal Graph")
        ...

#-------------------------------------------------------------------------------
    def on_running(self, xdata, ydata):
        #Update data (with the new _and_ the old points)
        self.min_x = np.min(xdata)
        self.max_x = np.max(xdata)
        self.max_y = np.max(ydata)
        self.ax.set_xlim(self.min_x, self.max_x)
        self.ax.set_ylim(self.min_y, self.max_y+10)
        self.lines.set_xdata(xdata)
        self.lines.set_ydata(ydata)
        #Need both of these in order to rescale
        self.ax.relim()
        self.ax.autoscale_view()
        #We need to draw *and* flush
        self.figure.canvas.draw()
        self.figure.canvas.flush_events()

################################################################################
class SimulatedGuiderCam(threading.Thread):
    def __init__(self,base_directory,config_file,new_image_callback,
                 logger=None):
        """Initialize the Simulated camera with default configuration"""

        threading.Thread.__init__(self)
        # set up the log file
        if logger == None:
            self.logger = utils.setup_logger(base_directory + '/log/',
                                             'simulated_cam')
        else:
            self.logger = logger

        config_file = base_directory + '/config/' + config_file
        if os.path.exists(config_file):
            config = ConfigObj(config_file)
        else:
            logging.error('Config file not found: ('
                          + config_file + ')')
            sys.exit()

            
        self.camera_xdim_pix = int(config['SIMULATED_CAM_XDIM'])
        self.camera_ydim_pix = int(config['SIMULATED_CAM_YDIM'])

        self.camera_xcenter_pix = (float(self.camera_xdim_pix)/2) - 0.5
        self.camera_ycenter_pix = (float(self.camera_ydim_pix)/2) - 0.5
        
        self.pixel_size_um = float(config['SIMULATED_CAM_PIXEL_SIZE_UM'])
        
        self.model = "Simulated camera"
        self.sn = "%ix%i" % (self.camera_xdim_pix,self.camera_ydim_pix)

        self.new_image_callback = new_image_callback

        self.x1 = 0
        self.x2 = self.camera_xdim_pix
        self.y1 = 0
        self.y2 = self.camera_ydim_pix
        self.xbin = 1
        self.ybin = 1

        self.set_roi(self.x1,self.y1,self.x2,self.y2)
        self.logger.info("Setting frame rate to 10Hz")
        self.set_frame_period(1.0)
        self.set_gain(1.0)
        self.logger.info("Setting exposure time to max for 10Hz framerate")
        self.set_exposure_time(0.1)
        self.horiz_graph = DynamicPlot()
        self.horiz_graph.on_launch()
        self.xdata = []
        self.ydata = []

        # Hardcoded here.  Should match values in tres_guider.ini
        self.arcsec_per_um_at_ccd = 0.02477
        self.hole_width_arcsec = 2.44  # arcseconds
        
        # Used for creating a simulated image
        self.arcsec_per_pix = self.arcsec_per_um_at_ccd * self.pixel_size_um
        


        self.star_xpositions = [1000,500,1000]
        self.star_ypositions = [1000,500,500]
        self.star_fluxes = [1e6,0.5e6,1.5e6]
        self.star_fwhm = 1.0
        self.noise = 2.0
        self.background = 5
        self.jitter_amplitude = 0.0 # Amplitude to inject into integrator "
        self.jitter_gain = 0.0      # Gain term of integrator
        self.x_jitter_offset = 0.0  # arcsec initial offsets
        self.y_jitter_offset = 0.0  # Null out initial offsets

        self.simulated_fsm_x_correction = 0. # arcsec
        self.simulated_fsm_y_correction = 0.

        self.hole_width_pix = self.um_to_pix(self.hole_width_arcsec /
                                             self.arcsec_per_um_at_ccd)
        
        xwidth = self.x2 - self.x1
        ywidth = self.y2 - self.y1
        
        hole_mask = np.zeros((ywidth,xwidth),dtype=np.float64) + 1.0
        hole_X1, hole_X2 = np.mgrid[0:ywidth,0:xwidth]
        xdist_from_hole = (hole_X2 - (xwidth/2) + 0.5)
        ydist_from_hole = (hole_X1 - (ywidth/2) + 0.5)

        dist_from_hole = ((xdist_from_hole*xdist_from_hole) + (ydist_from_hole * ydist_from_hole))**0.5
        inside_hole = np.where(dist_from_hole < self.hole_width_pix/2)
        hole_mask[inside_hole] = 0
        self.hole_mask = gaussian_filter(hole_mask,sigma=1)

        
#-------------------------------------------------------------------------------
    def __del__(self):
        """Close camera and free resources"""
        pass


#-------------------------------------------------------------------------------
    def set_gain(self,gain):
        self.gain = gain
        self.logger.info("Setting camera gain to %i" % (self.gain,))

#-------------------------------------------------------------------------------
    def set_roi(self,x1,y1,x2,y2):
        """Set the region of interest readout

        arguments:
        x1: (int) top left corner of image x coordinate
        y1: (int) top left corner of image y coordinate
        x1: (int) bottom right corner of image x coordinate
        y1: (int) bottom right corner of image y coordinate        
        """
        
        self.x1 = x1
        self.x2 = x2
        self.y1 = y1
        self.y2 = y2
        self.logger.info("Setting camera ROI to %i,%i,%i,%i" %
                         (self.x1,self.y1,self.x2,self.y2))

#-------------------------------------------------------------------------------
    def get_roi(self):
        """Get the current region of interest
        """
        
        return(self.x1,self.y1,self.x2,self.y2)

#-------------------------------------------------------------------------------
    def set_binning(self,xbin,ybin):
        """Set the camera binning
        xbin: Binning factor for x dimension
        ybin: Binning factor for y dimension
        """
        
        self.xbin = xbin
        self.ybin = ybin

#-------------------------------------------------------------------------------
    def get_binning(self,xbin,ybin):
        """Get the camera binning
        """
        
        return(self.xbin,self.ybin)
        
#-------------------------------------------------------------------------------
    def get_camsize(self):
        return(self.camera_xdim_pix,self.camera_ydim_pix)

#-------------------------------------------------------------------------------
    def pix_to_um_from_center(self,x,y):
        '''
        Takes an x and y pixel coordinate and returns the offset from the
        center of the camera in microns
        '''
        return((float(x) - self.camera_xcenter_pix)*self.pixel_size_um,
               (float(y) - self.camera_ycenter_pix)*self.pixel_size_um)    

#-------------------------------------------------------------------------------
    def um_from_center_to_pix(self,x,y):
        '''
        Takes a distance in microns from the center of the camera and returns
        corresponding pixel coordinates
        '''
        return((float(x)/self.pixel_size_um)+self.camera_xcenter_pix,
               (float(y)/self.pixel_size_um)+self.camera_ycenter_pix)

#-------------------------------------------------------------------------------
    def um_to_pix(self,um):
        return(float(um)/self.pixel_size_um)

#-------------------------------------------------------------------------------
    def pix_to_um(self,pix):
        return(float(pix)*self.pixel_size_um)    
    
#-------------------------------------------------------------------------------
    def set_frame_period(self,seconds):
        """Set the frame period of camera readouts

        arguments:
        seconds: (float) time between successive readouts of the camera
        """

        self.frame_period = seconds

#-------------------------------------------------------------------------------
    def get_frame_period(self):
        """Set the frame period of camera readouts

        arguments:
        seconds: (float) time between successive readouts of the camera
        """
        self.logger.info("Frame persiod is set to %f" % (self.frame_period,))
    
#-------------------------------------------------------------------------------
    def set_exposure_time(self,seconds):
        """Set the exposure time of camera

        arguments:
        seconds: (float) duration of exposure time. Must be less than the
        frame period as dicated by the camera timing
        """
        
        self.exposure_time = seconds        

#-------------------------------------------------------------------------------
    def get_exposure_time(self):
        """Get the exposure time of camera frames

        arguments:
        seconds: (float) time of camera exposure
        """
        return(self.exposure_time)

#-------------------------------------------------------------------------------
    def get_pixel_size(self):
        return(self.pixel_size_um)
    
#-------------------------------------------------------------------------------
    def populate_header(self,hdr):
        ''' Fill in the camera related FITS header fields'''
        
        hdr['EXPTIME'] = (self.exposure_time,'Exposure time (s)')
        datasec = '[' + str(self.x1) + ':' + str(self.x2) + ',' + \
            str(self.y1) + ':' + str(self.y2) + ']'        
        hdr['DATASEC'] = (datasec,"Region of CCD read")
        
        hdr['CCDSUM'] = (str(self.xbin) + ' ' + str(self.ybin),
                         'CCD on-chip summing')

        
        hdr['GAIN'] = (self.gain,'electrons/ADU')
        hdr['CAMMOD'] = (self.model,'Model of the acquisition camera')
        hdr['CAMERASN'] = (self.sn,'Serial number of the acquisition camera')
        hdr['CCD-TEMP'] = (-999,'CCD Temperature (C)') # do we have this?
    
#-------------------------------------------------------------------------------
    def set_callback(self,new_image_callback):
        """Set a callback function for incoming images"""
        self.new_image_callback = new_image_callback        
        pass

#-------------------------------------------------------------------------------
    def clear_callback(self):
        '''Remove callback function for incoming images'''
        self.new_image_callback = None
        pass

#-------------------------------------------------------------------------------
    def get_hole_mask(self):
        '''
        Simulate the loss of light from the camera due to the fiber-coupling
        hole drilled into the turning mirror.
        '''

        return(self.hole_mask[self.y1:self.y2,self.x1:self.x2])

#-------------------------------------------------------------------------------
    def set_simulated_fsm_correction(self,x,y):
        '''
        When simulating the camera image, use this function to set the simulated FSM correction
        tip/tilt so that the simulated stars shift according to the FSM action
        '''
        self.simulated_fsm_x_correction = x
        self.simulated_fsm_y_correction = y
    
#-------------------------------------------------------------------------------
    def simulate_star_image(self,x,y,flux,fwhm,background=300.0,noise=0.0,
                            jitter=0.0):
        ''' 
        This creates a simple simulated image of a star field
        The idea is to be able to test guide performance without being on sky
        x -- an array of X centroids of the stars (only integers tested!)
        y -- an array of Y centroids of the stars (only integers tested!)
        flux -- an array of fluxes of the stars (electrons)
        fwhm -- the fwhm of the stars (arcsec)
        background -- the sky background of the image
        noise -- readnoise of the image
        jitter -- amplitude of jitter injected into jitter integrator
        '''
        
        xwidth = self.x2 - self.x1
        ywidth = self.y2 - self.y1
        self.image = np.zeros((ywidth,xwidth),dtype=np.float64) + \
            background + np.random.normal(scale=noise,size=(ywidth,xwidth))
        
        # add a guide star?
        sigma = fwhm / self.arcsec_per_pix
        mu = 0.0
        
        boxsize = math.ceil(sigma * 10.0)
        
        # make sure it's even to make the indices/centroids come out right
        if (boxsize % 2 == 1):
            boxsize += 1 
        
        xgrid,ygrid = np.meshgrid(np.linspace(-boxsize,boxsize,2*boxsize+1),
                                  np.linspace(-boxsize,boxsize,2*boxsize+1))
        d = np.sqrt(xgrid*xgrid+ygrid*ygrid)
        g = np.exp(-((d-mu)**2 / (2.0 * sigma**2)))
        g = g/np.sum(g) # normalize the gaussian

        self.x_jitter_offset += (self.jitter_gain *
                                 (jitter / self.arcsec_per_pix) *
                                 np.random.randn())
        self.y_jitter_offset += (self.jitter_gain *
                                 (jitter / self.arcsec_per_pix) *
                                 np.random.randn())

        
        # add each of the stars
        for ii in range(len(x)):

            xii = x[ii] - self.x1 + 1 + \
                int(np.round((self.x_jitter_offset +
                              self.simulated_fsm_x_correction) /
                             self.arcsec_per_pix))
            yii = y[ii] - self.y1 + 1 + \
                int(np.round((self.y_jitter_offset +
                              self.simulated_fsm_y_correction) /
                             self.arcsec_per_pix))
            
            # make sure the stamp fits on the image (if not, truncate)
            if (xii >= boxsize):
                x1 = xii - boxsize
                x1stamp = 0
            else:
                x1 = 0
                x1stamp = boxsize - xii
                
            if (xii <= (xwidth - boxsize)):
                x2 = xii + boxsize + 1
                x2stamp = 2 * boxsize + 1
            else:
                x2 = xwidth
                x2stamp = xwidth - xii + boxsize
                
            if (yii >= boxsize):
                y1 = yii-boxsize
                y1stamp = 0
            else:
                y1 = 0
                y1stamp = boxsize-yii
                
            if (yii <= (ywidth-boxsize)):
                y2 = yii + boxsize + 1
                y2stamp = 2 * boxsize + 1
            else:
                y2 = ywidth
                y2stamp = ywidth - yii + boxsize
            
            if ((y2-y1) > 0 and (x2-x1) > 0):
                # normalize the star to desired flux
                star = g[y1stamp:y2stamp,x1stamp:x2stamp]*flux[ii]

                # add Poisson noise; convert to ADU
                noise = np.random.normal(size=(y2stamp-y1stamp,
                                               x2stamp-x1stamp))
                noisystar = (star + np.sqrt(star) * noise) / self.gain

                # add the star to the image
                self.image[y1:y2,x1:x2] += noisystar
            else:
                pass
#                self.logger.warning("star off image (" + str(xii) + "," +
#                                      str(yii) + "); ignoring")
                
        # now convert to 16 bit int
        self.image = self.image.astype(np.int16)

        current_hole_mask = self.get_hole_mask()
        self.image = self.image * current_hole_mask
        neg_pix = np.where(self.image < 0)
        if len(neg_pix) > 0:
            self.image[neg_pix] = 0
        h, w = self.image.shape
        return(self.image)

#-------------------------------------------------------------------------------
    def tick(self):
        ''' Deliver a simualted camera frame to the callback function'''
        roi = self.simulate_star_image(self.star_xpositions,
                                       self.star_ypositions,
                                       self.star_fluxes,
                                       self.star_fwhm,
                                       noise=self.noise,
                                       background=self.background,
                                       jitter=self.jitter_amplitude)
        self.dateobs = datetime.datetime.utcnow()
        if (len(self.xdata) !=  self.x2-self.x1) or (self.x1!= self.xdata[0]):
            self.xdata = np.linspace(self.x1,self.x2-1,self.x2-self.x1)
            
        if (self.new_image_callback != None):
            self.new_image_callback(self.dateobs,self.camera_xdim_pix,
                                    self.camera_ydim_pix,self.x1,self.y1,
                                    self.x2,self.y2,roi)
                
        self.ydata = np.sum(roi,axis=0)/(self.y2-self.y1)

        self.horiz_graph.on_running(self.xdata,self.ydata)

        self.frame_counter += 1;
        
            
#-------------------------------------------------------------------------------
    def run(self):
        '''Start simulated camera readout of frames cv, and deliver to 
        guider callback function
        '''
        print("Staring camera framing")
        self.frame_counter = 0

        self.ticker = threading.Event()
        self.framing = True
        while not self.ticker.wait(self.frame_period):
            if self.framing:
                self.tick()
            else:
                return

#-------------------------------------------------------------------------------
    def stop_framing(self):
        '''Cleanup operations'''
        self.framing = False

################################################################################
if __name__ == "__main__":
    if socket.gethostname() == 'tres-guider':
        base_directory = '/home/tres/tres-guider'
    elif socket.gethostname() == 'core2duo':
        base_directory = '/home/jkansky/tres-guider-jan'        
    elif socket.gethostname() == 'Jason-THINK':
        base_directory = 'C:/tres-guider/'
    else:
        base_directory = './'
        
    config_file = 'simulated_cam.ini'
    
    cam = SimulatedGuiderCam(base_directory,config_file,new_image_callback=None)
    
    cam.get_frame_period()
    cam.get_exposure_time()
    cam.start_framing()

    cam.stop_framing()
