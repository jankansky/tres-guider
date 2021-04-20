#!/usr/bin/python3

import sys
import socket
import os
from configobj import ConfigObj
import logging
import threading
import math
import datetime
import utils
import numpy as np
import select
import time
from scipy.ndimage.filters import gaussian_filter

from pyAndorSDK3 import AndorSDK3

################################################################################
class AndorGuiderCam(threading.Thread):
    def __init__(self,base_directory,config_file,new_image_callback,
                 logger=None):
        """Initialize the Andor camera with default configuration"""

        threading.Thread.__init__(self)
        # set up the log file
        if logger == None:
            self.logger = utils.setup_logger(base_directory + '/log/',
                                             'zyla')
        else:
            self.logger = logger

        config_file = base_directory + '/config/' + config_file
        if os.path.exists(config_file):
            config = ConfigObj(config_file)
        else:
            logging.error('Config file not found: ('
                          + config_file + ')')
            sys.exit()

        self.sdk3 = AndorSDK3()
        self.imager = self.sdk3.cameras[0].camera()
        
        self.terminate = False  # Should we exit this thread
        self.camera_xdim_pix = int(config['CAM_XDIM'])
        self.camera_ydim_pix = int(config['CAM_YDIM'])
        if self.imager.SensorWidth != self.camera_xdim_pix:
            print("Sensor X size mismatch %i %i" % (self.camera_xdim_pix,
                                                    self.imager.SensorWidth))
            exit()
            
        if self.imager.SensorWidth != self.camera_ydim_pix:
            print("Sensor Y size mismatch %i %i" % (self.camera_ydim_pix,
                                                    self.imager.SensorHeight))
            exit()

            
        self.camera_xcenter_pix = (float(self.camera_xdim_pix)/2) - 0.5
        self.camera_ycenter_pix = (float(self.camera_ydim_pix)/2) - 0.5
        
        self.pixel_size_um = float(config['CAM_PIXEL_SIZE_UM'])
        if self.imager.PixelHeight != self.pixel_size_um:
            print("Pixel size mismatch %f %f" % (self.pixel_size_um,
                                                 self.imager.PixelHeight))
            exit()
        if self.imager.PixelWidth != self.pixel_size_um:
            print("Pixel size mismatch %f %f" % (self.pixel_size_um,
                                                 self.imager.PixelWidth))
            exit()
            
            
        
        self.model = config['MODEL']
        self.sn = config['SN']
        
        #self.imager.SensorCooling = True
        print("Current sensor temperature is %f" %
              (self.imager.SensorTemperature,))
        print(self.imager.TemperatureStatus)
        
        self.new_image_callback = new_image_callback

        self.x1 = 0
        self.x2 = self.camera_xdim_pix
        self.y1 = 0
        self.y2 = self.camera_ydim_pix


        self.xbin = 1
        self.ybin = 1

        self.set_roi(self.x1,self.y1,self.x2,self.y2)
        self.logger.info("Setting frame rate to 0.5 Hz")
        self.set_frame_period(2.0)
        self.set_gain(1.0)
        self.logger.info("Setting exposure time to max for 0.5 Hz framerate")
        self.set_exposure_time(2.0)

        # Hardcoded here.  Should match values in tres_guider.ini
        # These don't belong here since they are optics and system related but
        # we temporarily need them here to create simulated stars injected
        # into the image
        self.arcsec_per_um_at_ccd = 0.02477
        self.hole_width_arcsec = 2.44  # arcseconds
        
        # Used for creating a simulated image
        self.arcsec_per_pix = self.arcsec_per_um_at_ccd * self.pixel_size_um
        

        # Create three simulated stars
        self.star_xpositions = [0.0,-30.0,0.0]   # arcsec from center
        self.star_ypositions = [0.0,-30.0,-30.0] # arcsec from center
        self.star_fluxes = [1e6,0.5e6,1.5e6]
        self.star_fwhm = 1.0
        self.noise = 2.0
        self.background = 0
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

        dist_from_hole = ((xdist_from_hole*xdist_from_hole) +
                          (ydist_from_hole * ydist_from_hole))**0.5
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
        self.logger.info("Not supported by the Zyla")

#-------------------------------------------------------------------------------
    def set_roi(self,x1,y1,x2,y2):
        """Set the region of interest readout

        arguments:
        x1: (int) top left corner of image x coordinate
        y1: (int) top left corner of image y coordinate
        x2: (int) bottom right corner of image x coordinate
        y2: (int) bottom right corner of image y coordinate        
        """
        
        self.x1 = int(x1)
        self.x2 = int(x2)
        self.y1 = int(y1)
        self.y2 = int(y2)
        self.imager.AOIWidth = self.x2 - self.x1
        self.imager.AOILeft = self.x1 + 1
        self.imager.AOIHeight = self.y2 - self.y1
        self.imager.AOITop = self.y1 + 1 
        
        self.logger.info("Setting camera ROI to %i,%i,%i,%i" %
                         (self.x1,self.y1,self.x2,self.y2))

#-------------------------------------------------------------------------------
    def get_roi(self):
        """Get the current region of interest
        """
        self.x1 = self.imager.AOILeft - 1
        self.y1 = self.imager.AOITop - 1
        self.x2 = self.imager.AOIWidth + self.x1
        self.y2 = self.imager.AOIHeight + self.y1
        
        return(self.x1,self.y1,self.x2,self.y2)

#-------------------------------------------------------------------------------
    def set_binning(self,xbin,ybin):
        """Set the camera binning
        xbin: Binning factor for x dimension
        ybin: Binning factor for y dimension
        """

        self.imager.AOIHBin = xbin
        self.imager.AOIVBin = ybin
        
        self.xbin = xbin
        self.ybin = ybin

#-------------------------------------------------------------------------------
    def get_binning(self,xbin,ybin):
        """Get the camera binning
        """
        
        return(self.imager.AOIHBin,self.imager.AOIVBin)
        
#-------------------------------------------------------------------------------
    def get_camsize(self):
        return(self.imager.PixelWidth,self.imager.PixelHeight)

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

        self.imager.FrameRate = 1. / seconds
        self.frame_period = seconds

#-------------------------------------------------------------------------------
    def get_frame_period(self):
        """Set the frame period of camera readouts

        arguments:
        seconds: (float) time between successive readouts of the camera
        """
        self.logger.info("Frame persiod is set to %f" %
                         (1./self.imager.FrameRate,))
        return(1./self.imager.FrameRate)
    
#-------------------------------------------------------------------------------
    def set_exposure_time(self,seconds):
        """Set the exposure time of camera

        arguments:
        seconds: (float) duration of exposure time. Must be less than the
        frame period as dicated by the camera timing
        """
        
        self.exposure_time = seconds        
        self.imager.ExposureTime = self.exposure_time
        
#-------------------------------------------------------------------------------
    def get_exposure_time(self):
        """Get the exposure time of camera frames

        arguments:
        seconds: (float) time of camera exposure
        """
        return(self.imager.ExposureTime)

#-------------------------------------------------------------------------------
    def get_pixel_size(self):
        return(self.imager.PixelWidth)
    
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
    def simulate_star_image(self,xpos,ypos,flux,fwhm,background=300.0,noise=0.0,
                            jitter=0.0):
        ''' 
        This creates a simple simulated image of a star field
        The idea is to be able to test guide performance without being on sky
        xpos -- an array of X centroids of the stars (arcsec from cam center)
        ypos -- an array of Y centroids of the stars (arcsec from cam center)
        flux -- an array of fluxes of the stars (electrons)
        fwhm -- the fwhm of the stars (arcsec)
        background -- the sky background of the image
        noise -- readnoise of the image
        jitter -- amplitude of jitter injected into jitter integrator
        '''
        
        xwidth = self.camera_xdim_pix 
        ywidth = self.camera_ydim_pix
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
        for ii in range(len(xpos)):

            xii = int(np.round(xpos[ii]/self.arcsec_per_pix +
                               self.camera_xcenter_pix + \
                               (self.x_jitter_offset +
                                self.simulated_fsm_x_correction) /
                               self.arcsec_per_pix))
            yii = int(np.round(ypos[ii]/self.arcsec_per_pix +
                               self.camera_ycenter_pix + \
                               (self.y_jitter_offset +
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

        self.image = self.image * self.hole_mask
        neg_pix = np.where(self.image < 0)
        if len(neg_pix) > 0:
            self.image[neg_pix] = 0
        h, w = self.image.shape
        return(self.image)

#-------------------------------------------------------------------------------
    def tick(self):
        ''' Deliver a simualted camera frame to the callback function'''
        cam_image = self.imager.acquire(timeout=(self.exposure_time*1000) +
                                        1000).image.astype(np.int16)

        sim_star_img = self.simulate_star_image(self.star_xpositions,
                                                self.star_ypositions,
                                                self.star_fluxes,
                                                self.star_fwhm,
                                                noise=self.noise,
                                                background=self.background,
                                                jitter=self.jitter_amplitude)
        sim_star_roi = sim_star_img[self.y1:self.y2,self.x1:self.x2] 
        if (cam_image.shape != sim_star_roi.shape):
                    print("Mismatch between camera frame and current ROI")
                    print(roi.shape)
                    print(star_roi.shape)
                    continue
        
        roi = cam_image * self.hole_mask[self.y1:self.y2,self.x1:self.x2] \
              + sim_star_roi
        self.dateobs = datetime.datetime.utcnow()
            
        if (self.new_image_callback != None):
            self.new_image_callback(self.dateobs,self.camera_xdim_pix,
                                    self.camera_ydim_pix,self.x1,self.y1,
                                    self.x2,self.y2,roi)
                
        self.frame_counter += 1;
            
#-------------------------------------------------------------------------------
    def run(self):
        '''Start simulated camera readout of frames cv, and deliver to 
        guider callback function
        '''
        self.logger.info("Staring andor camera thread and framing")
        self.frame_counter = 0

        self.ticker = threading.Event()
        self.framing = True
        while not self.ticker.wait(self.frame_period):
            if self.framing:
                self.tick()
            if self.terminate:
                return

#-------------------------------------------------------------------------------
    def start_framing(self):
        '''Enable camera framing'''
        self.logger.info("Starting andor camera framing")
        self.framing = True

            
#-------------------------------------------------------------------------------
    def stop_framing(self):
        '''Cleanup operations'''
        self.logger.info("Stopping andor camera framing")
        self.framing = False

#-------------------------------------------------------------------------------
    def exit(self):
        self.logger.info("Andor camera thread terminating")
        self.framing = False
        self.terminate = True

################################################################################
if __name__ == "__main__":
    if socket.gethostname() == 'tres-guider':
        base_directory = '/home/tres/jan/tres-guider'
    elif socket.gethostname() == 'core2duo':
        base_directory = '/home/jkansky/tres-guider-jan'        
    elif socket.gethostname() == 'Jason-THINK':
        base_directory = 'C:/tres-guider/'
    else:
        base_directory = './'
        
    config_file = 'zyla.ini'
    
    cam = AndorGuiderCam(base_directory,config_file,new_image_callback=None)

    print(cam.get_roi())
    print(cam.get_frame_period())
    print(cam.get_exposure_time())
#    cam.get_frame_period()
#    cam.get_exposure_time()
#    cam.start_framing()

#    cam.stop_framing()
