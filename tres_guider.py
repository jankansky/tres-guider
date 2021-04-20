#!/usr/bin/python3

import time
import socket
import os
import math
import logging
import redis
import asyncio
import aioredis
import json
import struct
from configobj import ConfigObj
from astropy.io import fits
import numpy as np
import datetime
import cv2

import dfm_telescope as telescope

import simulated_cam
import asi_cam
#import andor_cam
#import adimec_q4a180

import utils
import tiptilt
import pid
import calstage

import cv2_image_processor as image_processor
#import centroid_image_processor as image_processor

################################################################################
class TelemetrySender():
    '''Send guider images and derived telemetry to redis for GUI usage'''
    
#-------------------------------------------------------------------------------
    def __init__(self,config):
        self.redis = redis.StrictRedis(host=config['REDIS_SERVER'],
                                       port=config['REDIS_PORT'])
#        self.pub_tracking_data = self.redis.pubsub()

#-------------------------------------------------------------------------------
    def send(self,key,data):
        self.redis.publish(key,data)
        
################################################################################
class EventSender():
    '''Send guider events to Redis for GUI usage'''
    
#-------------------------------------------------------------------------------
    def __init__(self,config):
        self.server_host = config['REDIS_SERVER']
        self.server_port = config['REDIS_PORT']
        self.redis = redis.Redis(host=self.server_host,
                                 port=self.server_port)
        self.command_counter = 0
        
#-------------------------------------------------------------------------------
    def send(self,command_dict):
        '''
        Pass in a dict of key value pairs to set in underlying code
        '''
        command_json_struct = json.dumps(command_dict)
        res = self.redis.publish('guider_state',command_json_struct)
        print("Sent event %i" % (self.command_counter,))
        self.command_counter += 1
        return(0)
        
################################################################################
class EventReceiver():
    '''Receive gui events from Redis to modify guider operation'''    
    def __init__(self,config,callback):
        self.config = config
        self.redis = redis.StrictRedis(host=config['REDIS_SERVER'],
                                       port=config['REDIS_PORT'])
        self.pub_tracking_data = self.redis.pubsub()
        self.command_callback = callback
        
        self.receive_task = asyncio.ensure_future(
            self.receive_telem(self.command_callback,
                               self.config['REDIS_SERVER'],
                               self.config['REDIS_PORT'],
                               'guider_commands'))
        
#-------------------------------------------------------------------------------
    @staticmethod
    async def receive_telem(callback,server,port,data_name):
        redis_sub = await aioredis.create_redis('redis://'+server+':'+port)
        res = await redis_sub.subscribe(data_name)
        sub_channel = res[0]        
        
        receive_counter = 0        
        while (await sub_channel.wait_message()):
            msg = await sub_channel.get_json()
            callback(msg)
            receive_counter += 1

        

################################################################################
class SimulatedTelescope():
    '''Manage telescope interface'''
    
    def __init__(self,config):
        pass

#-------------------------------------------------------------------------------
    def get_objname(self):
        return('test')
    
#-------------------------------------------------------------------------------
    def populate_header(self,hdr):
        '''Retrieve and set telescope information FITS headers 
        (requires communication with TCS) '''
        
        #hdr['SITELAT'] = (latitude,"Site Latitude (deg)")
        #hdr['SITELONG'] = (longitude,"Site East Longitude (deg)")
        #hdr['SITEALT'] = (elevation,"Site Altitude (m)")
        #hdr['RA'] = (ra, "Solved RA (J2000 deg)")
        #hdr['DEC'] = (dec,"Solved Dec (J2000 deg)")
        #hdr['ALT'] = (alt,'Telescope altitude (deg)')
        #hdr['AZ'] = (az,'Telescope azimuth (deg E of N)')
        #hdr['AIRMASS'] = (airmass,"airmass (plane approximation)")
        #hdr['HOURANG'] = (hourang,"Hour angle")
        #hdr['PMODEL'] = ('',"Pointing Model File")
        #hdr['FOCPOS'] = (focus,"Focus Position (microns)")
        #hdr['ROTPOS'] = (rotpos,"Mechanical rotator position (degrees)")
        #hdr['PARANG'] = (parang,"Parallactic Angle (degrees)")
        #hdr['SKYPA' ] = (skypa,"Position angle on the sky (degrees E of N)")

        #hdr['MOONRA'] = (moonra, "Moon RA (J2000 deg)")
        #hdr['MOONDEC'] =  (moondec, "Moon Dec (J2000 deg)")
        #hdr['MOONPHAS'] = (moonphase, "Moon Phase (Fraction)")
        #hdr['MOONDIST'] = (moonsep, "Distance between pointing and moon (deg)")
        
        pass

#-------------------------------------------------------------------------------
    def populate_header(self,hdr):
        '''Retrieve and set telescope information FITS headers 
        (requires communication with TCS) '''
        
        #hdr['SITELAT'] = (latitude,"Site Latitude (deg)")
        #hdr['SITELONG'] = (longitude,"Site East Longitude (deg)")
        #hdr['SITEALT'] = (elevation,"Site Altitude (m)")
        #hdr['RA'] = (ra, "Solved RA (J2000 deg)")
        #hdr['DEC'] = (dec,"Solved Dec (J2000 deg)")
        #hdr['ALT'] = (alt,'Telescope altitude (deg)')
        #hdr['AZ'] = (az,'Telescope azimuth (deg E of N)')
        #hdr['AIRMASS'] = (airmass,"airmass (plane approximation)")
        #hdr['HOURANG'] = (hourang,"Hour angle")
        #hdr['PMODEL'] = ('',"Pointing Model File")
        #hdr['FOCPOS'] = (focus,"Focus Position (microns)")
        #hdr['ROTPOS'] = (rotpos,"Mechanical rotator position (degrees)")
        #hdr['PARANG'] = (parang,"Parallactic Angle (degrees)")
        #hdr['SKYPA' ] = (skypa,"Position angle on the sky (degrees E of N)")

        #hdr['MOONRA'] = (moonra, "Moon RA (J2000 deg)")
        #hdr['MOONDEC'] =  (moondec, "Moon Dec (J2000 deg)")
        #hdr['MOONPHAS'] = (moonphase, "Moon Phase (Fraction)")
        #hdr['MOONDIST'] = (moonsep, "Distance between pointing and moon (deg)")
        
        pass
    
    
################################################################################
class WeatherStation():
    '''Manage weather data interface'''
    
    def __init__(self,config):
        pass

#-------------------------------------------------------------------------------
    def populate_header(self,hdr):
        '''Retrieve and set weather information FITS headers'''
        
        pass
    
################################################################################
class GuidingSystem():
    def __init__(self,base_directory,cam_type='simulated',
                 tiptilt_type='simulated',
                 calstage_type='simulated',
                 telescope_type='simulated'):
        '''
        This code is the central actor in performing the TRES guiding 
        operations. Here we instantiate the required software components, 
        listen for and process events, receive and process images, send
        corrections, generate and send telemetry, and log data.
 
        base_directory is where the code resides
        cam_type: 'simulated', 'adimec', 'andor'
        tiptilt_type = 'simulated', 'pi'
        calstage_type: 'simulated', 'pi'
        telescope_type: 'simulated', 'dfm'
        '''
        
        self.base_directory = base_directory
        self.config_file = 'tres_guider.ini'
        self.config_file = self.base_directory + '/config/' + self.config_file

        self.logger = utils.setup_logger(base_directory + '/log/', 'tres')
        
        if os.path.exists(self.config_file):
            config = ConfigObj(self.config_file)
        else:
            logging.error('Config file not found: ('
                          + self.config_file + ')')
            exit()

        self.cam_type = cam_type
        self.tiptilt_type = tiptilt_type
        self.calstage_type = calstage_type
        self.telescope_type = telescope_type

        # Telescope focal plane magnification
        self.arcsec_per_um_hole_mirror = \
            float(config['ARCSEC_PER_UM_HOLE_MIRROR'])
        self.arcsec_per_um_guider_ccd = \
            float(config['ARCSEC_PER_UM_GUIDER_CCD'])
        self.guider_magnification = float(config['GUIDER_MAGNIFICATION'])
        
        self.init_ipc(config)
        self.init_camera(config)
        self.init_calstage(config)
        self.init_image_processing(config)
        self.init_pid(config)
        self.init_tiptilt(config)
        self.init_telescope(config)
        self.weather_station = WeatherStation(config)
        
        # Position of science fiber with respect to on-axis in arcsec
        self.sci_fiber_x = float(config['SCI_FIBER_X'])
        self.sci_fiber_y = float(config['SCI_FIBER_Y'])

        # Diameter of science fiber bucket in arcsec        
        self.sci_fiber_xwidth = float(config['SCI_FIBER_XWIDTH'])
        self.sci_fiber_ywidth = float(config['SCI_FIBER_YWIDTH'])

        self.sci_hole_xwidth = float(config['SCI_HOLE_XWIDTH'])
        self.sci_hole_ywidth = float(config['SCI_HOLE_YWIDTH'])

        
        # Position of sky reference fiber with respect to on-axis in arcsec
        self.sky_fiber_x = float(config['SKY_FIBER_X'])
        self.sky_fiber_y = float(config['SKY_FIBER_Y'])
        
        # Diameter of sky refernce fiber bucket in arcsec                
        self.sky_fiber_xwidth = float(config['SKY_FIBER_XWIDTH'])
        self.sky_fiber_ywidth = float(config['SKY_FIBER_YWIDTH'])

        self.sky_hole_xwidth = float(config['SKY_HOLE_XWIDTH'])
        self.sky_hole_ywidth = float(config['SKY_HOLE_YWIDTH'])

        
        # Where we store our 
        self.datapath = config['DATAPATH']
        
        #Set default tracking setpoint in arcsec with respect to on-axis
        self.pid.setPoint((self.sci_fiber_x,self.sci_fiber_y))

        # Default to no detected stars, and no offsets
        self.stars = []
        self.offset = (0.0,0.0)

        self.command_tree = {'loop_state':self.set_loop_state,
                             'framing':self.set_framing_state,
                             'toggle_framing':self.toggle_framing_state,
                             'roi':self.set_roi}
        
        # Start in open loop
        self.framing = True
        self.guide_status = 'Open'
        self.save_images = False

        # Send data via redis telem channel?
        self.telemetry = True

        self.counter = 0
        
#-------------------------------------------------------------------------------
    def init_ipc(self,config):
        '''Establish inter-process communicaiton mechanisms for telemetry and
        GUI interactions'''
        
        self.telem_out = TelemetrySender(config)
        self.event_sender = EventSender(config)
        self.event_receiver = EventReceiver(config,self.command_callback)

#-------------------------------------------------------------------------------
    def init_camera(self,config):
        '''
        Open camera specified by self.cam_type and register new image callback
        to process data
        '''        

        # Position of camera center pixel with respect to on-axis in arcsec
        self.cam_center_x = float(config['CAM_CENTER_X'])
        self.cam_center_y = float(config['CAM_CENTER_Y'])

        # Rotation of camera with respect to telescope axes
        self.cam_rotation = float(config['CAM_ROTATION_DEGREES'])
        self.cam_flip_x = float(config['CAM_FLIP_X'])
        self.cam_flip_y = float(config['CAM_FLIP_Y'])        

        if self.cam_type == 'simulated':
            self.cam = simulated_cam.SimulatedGuiderCam(base_directory,
                                                        'simulated_cam.ini',
                                                        new_image_callback=\
                                                        self.new_image_callback)
        elif self.cam_type == 'adimec':
            self.cam = adimec_q4a180.AdimecGuiderCam(base_directory,
                                                     'adimec_cam.ini',
                                                     new_image_callback=\
                                                     self.new_image_callback)
        elif self.cam_type == 'zyla':
            self.cam = andor_cam.AndorGuiderCam(base_directory,'zyla.ini',
                                                new_image_callback=\
                                                self.new_image_callback)
        elif self.cam_type == 'asi':
            self.cam = asi_cam.ASIGuiderCam(base_directory,'asi_cam.ini',
                                            new_image_callback=\
                                            self.new_image_callback)
        else:
            self.logger.error("Unknown cam type")
            exit()
        
        self.arcsec_per_pixel = self.arcsec_per_um_guider_ccd * \
            self.cam.get_pixel_size()

        self.pixel_size_um = self.cam.get_pixel_size();

        # Assume square ROI when tracking
        self.roi_size_pixels = (int(np.round(float(config['ROI_SIZE_ARCSEC']) /
                                             self.arcsec_per_pixel)))


#-------------------------------------------------------------------------------
    def init_calstage(self,config):
        '''Instantiate calstage object and initialize it'''
        
        if (self.calstage_type == 'simulated'):
            self.calstage = calstage.CalStage(self.base_directory,
                                              'calstage.ini',
                                              logger = self.logger,
                                              simulate = True)
        elif (self.calstage_type == 'pi'):
            self.calstage = calstage.CalStage(self.base_directory,
                                              'calstage.ini',
                                              logger = self.logger,
                                              simulate = False)
            
#-------------------------------------------------------------------------------
    def init_image_processing(self,config):
        '''
        Instantiate object that ingests raw images and returns centroids
        '''
        
        # Only pull in stars that are within this distance from the trackpoint
        self.guiding_tolerance = float(config['GUIDING_TOLERANCE']) # arcsec

        self.image_processor = image_processor.ImageProcessor(config)
        
#-------------------------------------------------------------------------------
    def init_pid(self,config):
        '''
        Instantiate PID object that takes centroids and computes steering
        mirror commands. Initialize it with default servo parameters
        '''
        self.KPx = float(config['KPx'])
        self.KIx = float(config['KIx'])
        self.KDx = float(config['KDx'])
        self.KPy = float(config['KPy'])
        self.KIy = float(config['KIy'])
        self.KDy = float(config['KDy'])
        self.Imax = float(config['Imax'])
        self.Dband = float(config['Dband'])
        self.Corr_max = float(config['Corr_max'])
        
        self.pid = pid.PID(P=np.array([self.KPx, self.KPy]),
                           I=np.array([self.KIx, self.KIy]),
                           D=np.array([self.KDx, self.KDy]),
                           Integrator_max = self.Imax,
                           Deadband = self.Dband,
                           Correction_max = self.Corr_max,
                           logger = self.logger)
        
#-------------------------------------------------------------------------------
    def init_tiptilt(self,config):
        '''Instantiate tiptilt object and initialize tiptilt stage'''
        if (self.tiptilt_type == 'simulated'):
            self.tiptilt = tiptilt.TipTilt(self.base_directory,'tip_tilt.ini',
                                           logger = self.logger,
                                           simulate = True)
        elif (self.tiptilt_type == 'pi'):
            self.tiptilt = tiptilt.TipTilt(self.base_directory,'tip_tilt.ini',
                                           logger = self.logger,
                                           simulate = False)
        # Move to middle of range to start
#        self.tiptilt.move_tip_tilt(1.0,1.0)

#-------------------------------------------------------------------------------
    def init_telescope(self,config):
        '''Instantiate telescope object and initialize telescope connection'''
        if (self.telescope_type == 'simulated'):
            self.telescope = SimulatedTelescope(config)
        elif (self.telescope_type == 'dfm'):
            self.telescope = DFMTelescope(config)

#-------------------------------------------------------------------------------
    def get_header(self):
        '''Latch current values to create header for FITS file'''
        hdr = fits.Header()

        hdr['INSTRUME'] = ('TRES','Name of the instrument')
        hdr['OBSERVER'] = ('',"Observer") # do we have this info?
        hdr['DATE-OBS'] = (self.dateobs.strftime('%Y-%m-%dT%H:%M:%S.%f'),
                           'YYYY-MM-DDThh:mm:ss.ssssss (UTC)')
        hdr['PIXSCALE'] = (self.arcsec_per_pixel,'arcsec/pixel')
        hdr['GUIDSTAT'] = (self.guide_status,'Status of the guiding loop')

        # Fiber info
        hdr['XSCIFIB'] = (self.sci_fiber_x,
                          'X position of sci fiber centroid in arcsec')
        hdr['YSCIFIB'] = (self.sci_fiber_y,
                          'Y position of sci fiber centroid in arcsec')
        hdr['XSKYFIB'] = (self.sky_fiber_x,
                          'X position of sky fiber centroid in arcsec')
        hdr['YSKYFIB'] = (self.sky_fiber_y,
                          'Y position of sky fiber centroid in arcsec')
        
        self.cam.populate_header(hdr)
        self.calstage.populate_header(hdr)
        self.tiptilt.populate_header(hdr)
        self.telescope.populate_header(hdr)
        self.image_processor.populate_header(hdr)
        self.weather_station.populate_header(hdr)


        # target information (do we have this?)
        #hdr['TARGRA'] = (ra, "Target RA (J2000 deg)")
        #hdr['TARGDEC'] = (dec,"Target Dec (J2000 deg)")
        #hdr['PMRA'] = (pmra, "Target Proper Motion in RA (mas/yr)")
        #hdr['PMDEC'] = (pmdec, "Target Proper Motion in DEC (mas/yr)")
        #hdr['PARLAX'] = (parallax, "Target Parallax (mas)")
        #hdr['RV'] = (rv, "Target RV (km/s)")

        
        return(hdr)

#-------------------------------------------------------------------------------
    def save_image(self,image,overwrite=False):
        '''
        Populate a FITS header by probing objects and attach it to the image.
        Write the image to disk
        '''
        
        self.logger.info("Saving " + filename)
        hdr = self.get_header()
        hdu = fits.PrimaryHDU(self.image, header=hdr)
        hdulist = fits.HDUList([hdu])
        hdulist.writeto(filename, overwrite=overwrite)

#-------------------------------------------------------------------------------
    def save_image_in_thread(self,image):
        '''
        Construct a filename and start the thread that writes the image to disk
        Return thread descriptor so that code can join on the thread.
        '''
        
        objname = self.telescope.get_objname()
        files = glob.glob(self.guider.datapath + "*.fits")
        index = str(len(files)+1).zfill(4)
        datestr = datetime.datetime.utcnow().strftime('%m%d%y') 
        filename = self.guider.datapath + objname + '.' + datestr + \
            '.guider.' + index + '.fits'

        # saving can go on in the background
        save_image_thread = threading.Thread(target=self.save_image,
                                             args=(filename,))
        save_image_thread.name = 'save_image_thread'
        save_image_thread.start()
        return(save_image_thread)

#-------------------------------------------------------------------------------
    def new_image_callback(self,dateobs,camera_xdim_pix,camera_ydim_pix,camera_x1,
                           camera_y1,camera_x2,camera_y2,image):
        '''
        Process newimage and display if requested.  Pass in datetime of image
        acquisition, camera full x and y dimension in pixels, plus the current
        ROI in use, followed by the image
        '''

        print(self.counter)
        self.counter += 1
        self.dateobs = dateobs 
        self.image = image
        self.camera_xdim_pix = camera_xdim_pix
        self.camera_ydim_pix = camera_ydim_pix
        self.camera_xcenter_pix = (float(self.camera_xdim_pix)/2) - 0.5
        self.camera_ycenter_pix = (float(self.camera_ydim_pix)/2) - 0.5
        self.x1 = camera_x1
        self.y1 = camera_y1
        self.x2 = camera_x2
        self.y2 = camera_y2

        # Sanity check
        if ((self.image.shape[0] != self.y2-self.y1) or
            (self.image.shape[1] != self.x2-self.x1)):
            self.logger.error("Mismatch in image dimensions")
            return

        # Analyze and act on the new image
        self.perform_guiding(self.image,self.camera_xdim_pix,self.camera_ydim_pix,
                             self.x1,self.x2,self.y1,self.y2)

        if self.telemetry:
            self.send_telemetry()
        
        if self.save_images:
            save_thread = self.save_image_in_thread()

#-------------------------------------------------------------------------------
    def set_loop_state(self,param):
        if param == 0:
            self.guide_status = 'Open'
            res = self.event_sender.send({'loop_state':0})
            print('Opened loops')            
        elif param == 1:
            self.guide_status = 'Closed'
            res = self.event_sender.send({'loop_state':1})
            print('Closed loops')
        return(0)
            
#-------------------------------------------------------------------------------
    def set_framing_state(self,param):
        if param == 0:
            self.framing = False
            self.cam.stop_framing()
            res = self.event_sender.send({'framing':0})
            print('Stopping camera framing')            
        elif param == 1:
            self.framing = True
            self.cam.start_framing()
            res = self.event_sender.send({'framing':1})
            print('Starting camera framing')
        return(0)

#-------------------------------------------------------------------------------
    def toggle_framing_state(self,param):
        if param > 0:
            if self.framing:
                self.cam.stop_framing()
                self.framing = False
                res = self.event_sender.send({'framing':0})
                print('Stopping camera framing')
            elif param == 0:
                self.framing = True
                self.cam.start_framing()
                res = self.event_sender.send({'framing':1})
                print('Starting camera framing')
        return(0)
    
#-------------------------------------------------------------------------------
    def validate_roi(self,x1,y1,x2,y2):
        if x1 >= x2:
            return(False)
        if y1 >= y2:
            return(False)
        if (x2-x1 > self.camera_xdim):
            return(False)
        if (y2-y1 > self.camera_ydim):
            return(False)
        if x1 < 0:
            return(False)
        if y1 < 0:
            return(False)
        if x1 > self.camera_xdim-1:
            return(False)
        if y1 > self.camera_ydim-1:
            return(False)
        if x2 > self.camera_xdim-1:
            return(False)
        if y2 > self.camera_ydim-1:
            return(False)
        return(True)
        
#-------------------------------------------------------------------------------
    def set_roi(self,param):
        print(param)
        print(len(param))
        if len(param) != 4:
            return(-1)
        
#        res = self.validate_roi(param[0],param[1],param[2],param[3])
#        if not res:
#            print("Error validating ROI")
#            return(False)
        
        self.cam.set_roi(param[0],param[1],param[2],param[3])
        
        return(0)

    
#-------------------------------------------------------------------------------
    def command_callback(self,command_dict):
        '''
        This is where events from the GUI arrive, and we act on them.
        '''
        for command in command_dict:
            res = self.command_tree[command](command_dict[command])
        print("Done with command processing")

#-------------------------------------------------------------------------------
    def perform_guiding(self,image,camera_xdim_pix,camera_ydim_pix,x1,x2,y1,y2):
        '''
        This is where the action happens.
        Find stars in image and compute centroids of starlike objects.
        Use star closest to desired position and use for feedback
        '''

        # Find stars and compute centroids in pixels
        (self.stars,img) = self.image_processor.get_stars(image)

        if (len(self.stars) == 0):
            self.logger.info("No stars passed to perform_guiding.")
            self.ndx = 0
            self.x_mispointings_arcsec = [0.0]
            self.y_mispointings_arcsec = [0.0]
            self.ra_mispointings_arcsec = [0.0]
            self.dec_mispointings_arcsec = [0.0]
            self.x_correction_arcsec = 0.0
            self.y_correction_arcsec = 0.0
            self.ra_correction_arcsec = 0.0
            self.dec_correction_arcsec = 0.0
            self.flux_counts = 0.0
            self.star_fwhm = 0.0
            self.stars = np.array([])
            return

        # Add top left corner of ROI in order to have centroids be with
        # respect to full camera frame and then offset to place 0,0 in the
        # center of the camera
        self.stars[:,0] += x1 - camera_xdim_pix/2
        self.stars[:,1] += y1 - camera_ydim_pix/2
        
        # find the closest star to the desired science fiber position, keeping
        # offsets in mind. Fiber positions and offsets are in arcseconds, so
        # convert them to pixels
        x_errors_pixels = self.stars[:,0] - \
            ((self.sci_fiber_x + self.offset[0]) / self.arcsec_per_pixel)
        y_errors_pixels = self.stars[:,1] - \
            ((self.sci_fiber_y + self.offset[1]) / self.arcsec_per_pixel)

        self.x_mispointings_arcsec = x_errors_pixels * self.arcsec_per_pixel
        self.y_mispointings_arcsec = y_errors_pixels * self.arcsec_per_pixel
        
        # Compute star distance from trackpoint in arcseconds
        dists_arcsec = np.sqrt(x_errors_pixels**2 + y_errors_pixels**2) * \
            self.arcsec_per_pixel
        
        self.ndx = np.argmin(dists_arcsec)

        self.logger.info("Using guide star "+ str(self.ndx) +" (" +
                         str(self.stars[self.ndx,0] * self.arcsec_per_pixel) + \
                         ',' + str(self.stars[self.ndx,1] * \
                                   self.arcsec_per_pixel) +
                         ') dist: ' + str(dists_arcsec[self.ndx]) +
                         ' arcsec from the requested position Trackpoint:(' +
                         str(self.sci_fiber_x - self.offset[0]) + ',' +
                         str(self.sci_fiber_y - self.offset[1]) + ')')
        
        # if the star disappears, don't correct to a different star
        # magnitude tolerance, too? (probably not -- clouds could cause trouble)
        if (dists_arcsec[self.ndx] < self.guiding_tolerance):
            # Set desired position of star to be on the fiber, plus whatever
            # offset we want on top of that.
            self.pid.setPoint((self.sci_fiber_x + self.offset[0],
                               self.sci_fiber_y + self.offset[1]))

            # calculate the X & Y corrections needed in this step of the PID
            # loop in arcsec
            self.x_correction_arcsec,self.y_correction_arcsec = \
                self.pid.update(np.array([self.stars[self.ndx,0] *
                                          self.arcsec_per_pixel,
                                          self.stars[self.ndx,1] *
                                          self.arcsec_per_pixel]))
            self.logger.info('X Correction": '+ str(self.x_correction_arcsec) +\
                             ' Y Correction": ' + str(self.y_correction_arcsec))
                         
            if ((self.cam_type == 'simulated' or self.cam_type == 'zyla'
                 or self.cam_type == 'asi') and self.guide_status == 'Closed'):
                self.cam.set_simulated_fsm_correction(self.x_correction_arcsec,
                                                      self.y_correction_arcsec)
            
            # convert X & Y pixel to North & East arcsec offset
            # don't need cos(dec) term unless we send via mount
            PA = 0.0 # get from Telescope? Config file? user?

            self.dec_mispointings_arcsec = \
                (self.x_mispointings_arcsec * math.cos(PA) -
                 self.y_mispointings_arcsec * math.sin(PA))
            self.ra_mispointings_arcsec  = \
                (self.x_mispointings_arcsec * math.sin(PA) +
                 self.y_mispointings_arcsec * math.cos(PA))
                
            self.dec_correction_arcsec = \
                (self.x_correction_arcsec * math.cos(PA) - \
                 self.y_correction_arcsec * math.sin(PA))
            self.ra_correction_arcsec  = \
                (self.x_correction_arcsec * math.sin(PA) + \
                 self.y_correction_arcsec * math.cos(PA))

            self.flux_counts = self.stars[self.ndx,2]
            self.star_fwhm = 1.0+np.random.uniform(low=-0.1,high=0.1)
        
            # TODO: make sure the move is within range
            # move_in_range = True
                
            # # send correction to tip/tilt
            # if move_in_range:
            #     self.logger.info("Moving tip/tilt " + str(north) +
            #                      '" North, ' + str(east) + '" East')
            #     self.tiptilt.move_north_east(north,east)
            # else:
            #     # TODO: move telescope, recenter tip/tilt
            #     self.logger.error("Tip/tilt out of range. Must manually recenter")
        else:
            self.x_correction_arcsec = [0.0]
            self.y_correction_arcsec = [0.0]
            self.ra_mispointings_arcsec = [0.0]
            self.dec_mispointings_arcsec = [0.0]
            self.ra_correction_arcsec = 0.0
            self.dec_correction_arcsec = 0.0
            self.flux_counts = 0.0
            self.star_fwhm = 0.0

            
#-------------------------------------------------------------------------------
    def send_telemetry(self):

#        self.telem_image = self.image.astype(np.int16)
        h, w = self.image.shape
#        shape = struct.pack('>II',h,w)
#        encoded_img = self.telem_image.tobytes()

        telem_struct = \
            json.dumps({'timestamp':\
                        self.dateobs.strftime('%Y-%m-%dT%H:%M:%S.%f'),
                        'x_mispointing':self.x_mispointings_arcsec[self.ndx],
                        'y_mispointing':self.y_mispointings_arcsec[self.ndx],
                        'ra_mispointing':self.ra_mispointings_arcsec[self.ndx],
                        'dec_mispointing':self.dec_mispointings_arcsec[self.ndx],
                        'x_correction':self.x_correction_arcsec,
                        'y_correction':self.y_correction_arcsec,
                        'ra_correction':self.ra_correction_arcsec,
                        'dec_correction':self.dec_correction_arcsec,
                        'counts':self.flux_counts,
                        'fwhm':self.star_fwhm,
                        'platescale':self.arcsec_per_pixel,
                        'camera_xdim_pix':self.camera_xdim_pix,
                        'camera_ydim_pix':self.camera_ydim_pix,
                        'roi_x1':self.x1,
                        'roi_x2':self.x2,
                        'roi_y1':self.y1,
                        'roi_y2':self.y2,
                        'height':h,
                        'width':w,
                        'sci_fiber_x':self.sci_fiber_x,
                        'sci_fiber_xwidth':self.sci_fiber_xwidth,
                        'sci_hole_xwidth':self.sci_hole_xwidth,
                        'sky_fiber_x':self.sky_fiber_x,
                        'sky_fiber_xwidth':self.sky_fiber_xwidth,
                        'sky_hole_xwidth':self.sky_hole_xwidth,
                        'guider_magnification':self.guider_magnification,
                        'arcsec_per_um_guider_ccd':self.arcsec_per_um_guider_ccd,
                        'sci_fiber_y':self.sci_fiber_y,
                        'sci_fiber_ywidth':self.sci_fiber_ywidth,
                        'sci_hole_ywidth':self.sci_hole_ywidth,
                        'sky_fiber_y':self.sky_fiber_y,
                        'sky_fiber_ywidth':self.sky_fiber_ywidth,
                        'sky_hole_ywidth':self.sky_hole_ywidth,
                        'guider_magnification':self.guider_magnification,
                        'camera_xcenter_pix':self.camera_xcenter_pix,
                        'camera_ycenter_pix':self.camera_ycenter_pix,
                        'pixel_size_um':self.pixel_size_um,
                        'arcsec_per_um_guider_ccd':self.arcsec_per_um_guider_ccd,
                        'stars':self.stars.tolist(),
                        'image':self.image.tolist()})

        self.telem_out.send('guider_data',telem_struct)

            
#-------------------------------------------------------------------------------
    def start(self):
        self.cam.start()
        self.framing = True
        print("Camera framing started")

#-------------------------------------------------------------------------------
    def stop(self):
        self.cam.stop_framing()
        self.framing = False


################################################################################
if __name__ == "__main__":
    base_directory = "./"
    
    guider = GuidingSystem(base_directory,
                           cam_type='asi',
                           tiptilt_type='simulated',
                           calstage_type='simulated')
    guider.start()
    loop = asyncio.get_event_loop()
    try:
        loop.run_forever()
    finally:
        loop.close()
    print("TRES Guider stopping")

    
