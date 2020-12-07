#!/usr/bin/python3

import sys
import serial
import socket
import os
from configobj import ConfigObj
import logging

import numpy as np
from matplotlib import pyplot as plt
import select
import v4l2capture
import time

import utils

plt.ion()

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
        plt.title("Adimec Camera Horizontal Graph")

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
class AdimecGuiderCam():
    def __init__(self,base_directory,config_file,new_image_callback,
                 logger=None):
        """Initialize the Adimec camera with default configuration"""

        # set up the log file
        if logger == None:
            self.logger = utils.setup_logger(base_directory + '/log/',
                                             'adimec_cam')
        else:
            self.logger = logger

        config_file = base_directory + '/config/' + config_file
        if os.path.exists(config_file):
            config = ConfigObj(config_file)
        else:
            logging.error('Config file not found: ('
                          + config_file + ')')
            sys.exit()

            
        
        self.camera_xdim_pix = int(config['ADIMEC_XDIM'])
        self.camera_ydim_pix = int(config['ADIMEC_YDIM'])
        
        self.camera_xcenter_pix = (float(self.camera_xdim_pix)/2) - 0.5
        self.camera_ycenter_pix = (float(self.camera_ydim_pix)/2) - 0.5
        
        self.pixel_size_um = float(config['ADIMEC_PIXEL_SIZE_UM'])        

        self.model = "Adimec"

        self.video = v4l2capture.Video_device(config['ADIMEC_VIDEO_DEV'])
        size = self.video.set_format(self.camera_xdim_pix,
                                     self.camera_ydim_pix,fourcc='GREY')
        self.video.create_buffers(5)
        self.video.queue_all_buffers()
    
        self.new_image_callback = new_image_callback
        self.ser = serial.Serial(config['ADIMEC_SERIAL'],baudrate=57600,
                                 timeout=4)
        self.ser.write(b'@ID?\r')
        cam_id = self.ser.read_until(b'\r')
        self.sn = cam_id[3:].decode("utf-8")
        
        self.x1 = 0
        self.x2 = self.camera_xdim_pix
        self.y1 = 0
        self.y2 = self.camera_ydim_pix
        self.xbin = 1
        self.ybin = 1
        self.set_roi(self.x1,self.y1,self.x2,self.y2)
        print("Setting frame rate to 10Hz")
        self.set_frame_period(0.1)
        self.analog_gains = [1.0, 1.2, 1.4, 1.6]
        self.set_gain(0)
        print("Setting exposure time to max for 10Hz framerate")
        self.set_exposure_time(0.1)
        self.spectrum = DynamicPlot()
        self.spectrum.on_launch()
        self.xdata = []
        self.ydata = []
         
#-------------------------------------------------------------------------------
    def __del__(self):
        """Close camera and free resources"""
        self.ser.close()


#-------------------------------------------------------------------------------
    def set_gain(self,gain_index):
        self.gain_index = gain_index
        self.gain = self.analog_gains[gain_index]
        print("Setting camera gain to %i" % (self.analog_gains[gain_index],))
        self.ser.write(b"@PGA%i\r" % (self.gain_index,))
        response = self.ser.read(1)        

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
        print("Setting camera ROI to %i,%i,%i,%i" % (self.x1,self.y1,
                                                     self.x2,self.y2))
        self.ser.write(b"@ROI%i;%i;%i;%i\r" % (x1,y1,x2,y2))
        response = self.ser.read(1)

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
        self.ser.write(b'@FP%i\r' % (int(seconds*1E6),))
        response = self.ser.read(1)        

#-------------------------------------------------------------------------------
    def get_frame_period(self):
        """Set the frame period of camera readouts

        arguments:
        seconds: (float) time between successive readouts of the camera
        """
        try:
            self.ser.write(b'@FP?\r')
            response = self.ser.read_until(b'\r')
            frame_period = float(response[2:].decode("utf-8"))/1E6
            self.frame_period = frame_period
        except:
            print("error in get_frame_period")
            exit
        print(frame_period)
    
#-------------------------------------------------------------------------------
    def set_exposure_time(self,seconds):
        """Set the exposure time of camera

        arguments:
        seconds: (float) duration of exposure time. Must be less than the
        frame period as dicated by the camera timing
        """
        self.exposure_time = seconds
        self.ser.write(b'@IT%i\r' % (int(seconds*1E6),))
        response = self.ser.read(1)        

#-------------------------------------------------------------------------------
    def get_exposure_time(self):
        """Get the exposure time of camera frames

        arguments:
        seconds: (float) time of camera exposure
        """
        try:
            self.ser.write(b'@IT?\r')
            response = self.ser.read_until(b'\r')
            exposure_time = float(response[2:].decode("utf-8"))/1E6
            self.exposure_time = exposure_time
        except:
            print("error in get_exposure_time")
            exit
        print(exposure_time)
        
        return(exposure_time)

#-------------------------------------------------------------------------------
    def get_temp(self):
        try:
            self.ser.write(b'@TM?\r')
            response = self.ser.read_until(b'\r')
            temp = float(response[2:].decode("utf-8"))
            self.temp = temp
        except:
            print("error in get_temp")
            
        return(temp)

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

        temp = self.get_temp
        hdr['CCD-TEMP'] = (temp,'CCD Temperature (C)')


    
#-------------------------------------------------------------------------------
    def set_callback(self,new_image_callback):
        """Set a callback function for incoming images"""
        self.new_image_callback = new_image_callback

#-------------------------------------------------------------------------------
    def clear_callback(self):
        """Remove callback function for incoming images"""
        self.new_image_callback = None
        pass

#-------------------------------------------------------------------------------
    def start_framing(self):
        """Start camera readout of frames and display using opencv, plot
        processed data using matplotlib
        """

        frame_counter = 0
        
        self.video.start()

        self.dark_ref = np.fromfile("dark_ref.dat",dtype=float)
        
        while True:
            select.select((self.video,),(), ())
            image_data = self.video.read_and_queue()
            frame = np.frombuffer(image_data,dtype=np.uint8).reshape(2000,2000)

            if (len(self.xdata) !=  self.x2-self.x1):
                self.xdata = np.linspace(0,self.x2-self.x1-1,self.x2-self.x1)
            
            roi = frame[self.y1:self.y2,self.x1:self.x2]

            if (self.new_image_callback != None):
                self.new_image_callback(self.dateobs,self.camera_xdim_pix,
                                        self.camera_ydim_pix,self.x1,self.y1,
                                        self.x2,self.y2,roi)

            self.ydata = np.sum(roi,axis=0)/(self.y2-self.y1)

            self.spectrum.on_running(self.xdata,self.ydata)
#            print("Frame %i" % (frame_counter,))
            frame_counter += 1;

#-------------------------------------------------------------------------------
    def stop_framing(self):
        """Stop camera readouts
        """
        self.video.close()
        pass


################################################################################
if __name__ == "__main__":
    base_directory = './'
        
    config_file = 'adimec_cam.ini'
    
    cam = AdimecGuiderCam(base_directory,config_file,new_image_callback=None)
    
    cam.get_frame_period()
    cam.get_exposure_time()
    cam.start_framing()
  
    cam.stop_framing()
