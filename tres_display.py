#!/usr/bin/python3
import sys
import os
import utils
from configobj import ConfigObj
import asyncio
import aioredis
import redis
import datetime
import json
import cv2
import numpy as np
from matplotlib import dates

################################################################################
class CommandSender():
    '''Send guider commands to Redis for be passed down the guider via pub/sub'''
    
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
        res = self.redis.publish('guider_commands',command_json_struct)
        print("Sent command %i" % (self.command_counter,))
        self.command_counter += 1
        return(0)


################################################################################
class TresDisplay():
    def __init__(self,base_directory,config_file):
        super(TresDisplay,self).__init__()
        self.base_directory = base_directory
        self.config_file = config_file
        self.config_file = self.base_directory + '/config/' + self.config_file

        self.logger = utils.setup_logger(base_directory + '/log/', 'tres')
        
        if os.path.exists(self.config_file):
            self.config = ConfigObj(self.config_file)
        else:
            logging.error('Config file not found: ('
                          + self.config_file + ')')
            exit()

        
        self.display_name = "TRES Guider Image"
        self.screen_res = 500 
        self.rebin_factor = 1.0

        # Is the user holding mouse button to adjust color map?
        self.adjusting_image = False 

        # Current colormap tweaks
        self.contrast = 0.0
        self.brightness = 0.0

        self.roi_size_pixels = 64
        self.command_sender = CommandSender(self.config)        
        
        self.receive_task = asyncio.ensure_future(
            self.receive_telem(self.on_guider_data,
                               self.config['REDIS_SERVER'],
                               self.config['REDIS_PORT'],
                               'guider_data'))
        
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
            
#-------------------------------------------------------------------------------
    def mouse_event(self,event,x,y,flags,param):
        '''
        Temorarily including display functionality here via CV2 library.
        When user clicks on the screen, the left button is used to set
        an ROI around the star that was clicked.  The right button is used to
        revert back to a full frame readout
        '''
        if (event == cv2.EVENT_MBUTTONDOWN):
            self.logger.info(("User clicked at pixel %i %i" %
                              (x*self.rebin_factor,y*self.rebin_factor)))

            click_x = int(np.round(self.roi_x1 + float(x) * self.rebin_factor))
            click_y = int(np.round(self.roi_y1 + float(y) * self.rebin_factor))
            
            if (click_x > self.camera_xdim_pix-self.roi_size_pixels/2):
                click_x = self.camera_xdim_pix-self.roi_size_pixels/2

            if (click_x < self.roi_size_pixels/2):
                click_x = self.roi_size_pixels/2
                
                
            if (click_y > self.camera_ydim_pix-self.roi_size_pixels/2):
                click_y = self.camera_ydim_pix-self.roi_size_pixels/2
                
            if (click_y < self.roi_size_pixels/2):
                click_y = self.roi_size_pixels/2

                
            click_x = int(np.round(click_x))
            click_y = int(np.round(click_y))
            
            self.roi_x1 = click_x - int(self.roi_size_pixels/2)
            self.roi_x2 = click_x + int(self.roi_size_pixels/2)
            self.roi_y1 = click_y - int(self.roi_size_pixels/2)
            self.roi_y2 = click_y + int(self.roi_size_pixels/2)
            print(self.roi_x1,self.roi_x2,self.roi_y1,self.roi_y2)
            self.command_sender.send({'roi':[self.roi_x1,self.roi_y1,
                                           self.roi_x2,self.roi_y2]})

        if event == cv2.EVENT_RBUTTONDOWN:
            self.roi_x1 = 0
            self.roi_x2 = self.camera_xdim_pix
            self.roi_y1 = 0
            self.roi_y2 = self.camera_ydim_pix
            self.command_sender.send({'roi':[self.roi_x1,self.roi_y1,
                                           self.roi_x2,self.roi_y2]})
            
        if event == cv2.EVENT_LBUTTONDOWN:
            self.adjusting_image = True
            
        if event == cv2.EVENT_LBUTTONUP:
            self.adjusting_image = False

        if self.adjusting_image:
            if event == cv2.EVENT_MOUSEMOVE:
                self.brightness = (128 * (x - self.screen_res/2) /
                                   (self.screen_res/2))
                self.contrast = (128 * (y - self.screen_res/2) /
                                 (self.screen_res/2))
                frame_uint8_scaled = self.adjust_brightness_contrast(
                    self.frame_uint8,contrast=self.contrast,
                    brightness=self.brightness)
                frame_uint8_scaled = cv2.applyColorMap(frame_uint8_scaled,
                                                       cv2.COLORMAP_JET)
                self.annotate_image(frame_uint8_scaled,self.stars)
                cv2.imshow(self.display_name,frame_uint8_scaled)                

#-------------------------------------------------------------------------------
    def cam_pix_to_display_pix(self,x,y):
        '''
        Temorarily including display functionality here via CV2 library.

        The screen display is rescaled from the camera pixels.  To draw
        items on the display, this converts from raw camera pixel coordinates
        to display window coordinates.  The camera pixel coordinates are always
        in terms of the the full frame indices, even if an ROI is active.'''

        return((x - self.roi_x1) / self.rebin_factor,
               (y - self.roi_y1) / self.rebin_factor)

#-------------------------------------------------------------------------------
    def cam_pix_to_um_from_center(self,x,y):
        '''
        Takes an x and y pixel coordinate and returns the offset from the
        center of the camera in microns
        '''
        return((float(x) - self.camera_xcenter_pix)*self.pixel_size_um,
               (float(y) - self.camera_ycenter_pix)*self.pixel_size_um)    

#-------------------------------------------------------------------------------
    def cam_um_from_center_to_pix(self,x,y):
        '''
        Takes a distance in microns from the center of the camera and returns
        corresponding pixel coordinates
        '''
        return((float(x)/self.pixel_size_um)+self.camera_xcenter_pix,
               (float(y)/self.pixel_size_um)+self.camera_ycenter_pix)

#-------------------------------------------------------------------------------
    def cam_um_to_pix(self,um):
        return(float(um)/self.pixel_size_um)

#-------------------------------------------------------------------------------
    def cam_pix_to_um(self,pix):
        return(float(pix)*self.pixel_size_um)    
    
    
#-------------------------------------------------------------------------------
    def draw_sci_fiber_bucket(self,img):
        '''
        Temorarily including display functionality here via CV2 library.

        This draws a circle into the img passed in to denotes the science
        fiber bucket.
        '''

        # Figure out how many camera pixels from the camera center to the
        # fiber location
        (sci_fiber_xcenter_pix,
         sci_fiber_ycenter_pix) =  \
             self.cam_um_from_center_to_pix(self.sci_fiber_x *
                                        self.guider_magnification /
                                        self.arcsec_per_um_guider_ccd,
                                        self.sci_fiber_y  *
                                        self.guider_magnification /
                                        self.arcsec_per_um_guider_ccd)

        # Convert from camera pixels to screen pixels. They are different due
        # to display rebin factor
        (sci_fiber_display_xcen_pix,
         sci_fiber_display_ycen_pix) =  \
             self.cam_pix_to_display_pix(sci_fiber_xcenter_pix,
                                         sci_fiber_ycenter_pix)

        sci_fiber_xwidth_cam_pix = \
            self.cam_um_to_pix(self.sci_fiber_xwidth  *
                           self.guider_magnification /
                           self.arcsec_per_um_guider_ccd)        
        sci_fiber_ywidth_cam_pix = \
            self.cam_um_to_pix(self.sci_fiber_ywidth *
                               self.guider_magnification/
                               self.arcsec_per_um_guider_ccd)

        sci_fiber_xwidth_display_pix = (sci_fiber_xwidth_cam_pix /
                                        self.rebin_factor)
        
        sci_fiber_ywidth_display_pix = (sci_fiber_ywidth_cam_pix /
                                        self.rebin_factor)

        cv2.ellipse(img,(int(np.round(sci_fiber_display_xcen_pix)),
                        int(np.round(sci_fiber_display_ycen_pix))),
                    (int(np.round(sci_fiber_xwidth_display_pix)),
                     int(np.round(sci_fiber_ywidth_display_pix))),
                    0.0,0,360,
                    (255,255,255),1)

        sci_hole_xwidth_cam_pix = \
            self.cam_um_to_pix(self.sci_hole_xwidth  *
                               self.guider_magnification /
                               self.arcsec_per_um_guider_ccd)
        sci_hole_ywidth_cam_pix = \
            self.cam_um_to_pix(self.sci_hole_ywidth  *
                               self.guider_magnification /
                               self.arcsec_per_um_guider_ccd)

        sci_hole_xwidth_display_pix = (sci_hole_xwidth_cam_pix /
                                        self.rebin_factor)
        
        sci_hole_ywidth_display_pix = (sci_hole_ywidth_cam_pix /
                                        self.rebin_factor)

        cv2.ellipse(img,(int(np.round(sci_fiber_display_xcen_pix)),
                         int(np.round(sci_fiber_display_ycen_pix))),
                    (int(np.round(sci_hole_xwidth_display_pix)),
                     int(np.round(sci_hole_ywidth_display_pix))),
                    0.0,0,360,
                    (255,255,255),1)
        
#-------------------------------------------------------------------------------
    def draw_sky_fiber_bucket(self,img):
        '''
        Temorarily including display functionality here via CV2 library.

        This draws a circle into the img passed in to denotes the sky fiber
        bucket.
        '''

        # Figure out how many camera pixels from the camera center to the
        # fiber location
        (sky_fiber_xcenter_pix,
         sky_fiber_ycenter_pix) =  \
             self.cam_um_from_center_to_pix(self.sky_fiber_x *
                                            self.guider_magnification /
                                            self.arcsec_per_um_guider_ccd,
                                            self.sky_fiber_y *
                                            self.guider_magnification /
                                            self.arcsec_per_um_guider_ccd)

        # Convert from camera pixels to screen pixels. They are different due
        # to display rebin factor
        (sky_fiber_display_xcen_pix,
         sky_fiber_display_ycen_pix) =  \
             self.cam_pix_to_display_pix(sky_fiber_xcenter_pix,
                                         sky_fiber_ycenter_pix)

        sky_fiber_xwidth_cam_pix = \
            self.cam_um_to_pix(self.sky_fiber_xwidth  *
                               self.guider_magnification/
                               self.arcsec_per_um_guider_ccd)
        sky_fiber_ywidth_cam_pix = \
            self.cam_um_to_pix(self.sky_fiber_ywidth  *
                               self.guider_magnification/
                               self.arcsec_per_um_guider_ccd)

        sky_fiber_xwidth_display_pix = (sky_fiber_xwidth_cam_pix /
                                        self.rebin_factor)
        
        sky_fiber_ywidth_display_pix = (sky_fiber_ywidth_cam_pix /
                                        self.rebin_factor)

        cv2.ellipse(img,(int(np.round(sky_fiber_display_xcen_pix)),
                        int(np.round(sky_fiber_display_ycen_pix))),
                    (int(np.round(sky_fiber_xwidth_display_pix)),
                     int(np.round(sky_fiber_ywidth_display_pix))),
                    0.0,0,360,
                    (255,0,255),1)

        sky_hole_xwidth_cam_pix = \
            self.cam_um_to_pix(self.sky_hole_xwidth  *
                               self.guider_magnification /
                               self.arcsec_per_um_guider_ccd)
        sky_hole_ywidth_cam_pix = \
            self.cam_um_to_pix(self.sky_hole_ywidth  *
                               self.guider_magnification/
                               self.arcsec_per_um_guider_ccd)

        sky_hole_xwidth_display_pix = (sky_hole_xwidth_cam_pix /
                                        self.rebin_factor)
        
        sky_hole_ywidth_display_pix = (sky_hole_ywidth_cam_pix /
                                        self.rebin_factor)

        cv2.ellipse(img,(int(np.round(sky_fiber_display_xcen_pix)),
                        int(np.round(sky_fiber_display_ycen_pix))),
                    (int(np.round(sky_hole_xwidth_display_pix)),
                     int(np.round(sky_hole_ywidth_display_pix))),
                    0.0,0,360,
                    (200,0,200),1)

        
#-------------------------------------------------------------------------------
    def draw_centroids(self,img,stars):
        '''
        Temorarily including display functionality here via CV2 library.

        Draw a marker into the image passed in at each star location.
        Currently star centroids are in camera pixels.  Change this to arcsec
        eventually.
        '''

        if (len(stars) == 0):
            return
        
        num_stars = stars.shape[0]
        
        for loop in range(num_stars):
            (star_xcenter_pix,
             star_ycenter_pix) = \
                 self.cam_pix_to_display_pix(stars[loop,0] + self.camera_xdim_pix/2,
                                             stars[loop,1] + self.camera_ydim_pix/2)
            cv2.drawMarker(img,(int(np.round(star_xcenter_pix)),
                                int(np.round(star_ycenter_pix))),(0,255,255),
                           markerType=cv2.MARKER_CROSS, thickness=1)

#-------------------------------------------------------------------------------
    def annotate_image(self,img,stars):
        '''
        Temorarily including display functionality here via CV2 library.

        Draw sci and sky fiber locations in frame
        Draw centroid locations in frame
        Draw platesolving locations in frame eventually
        '''
        
        self.draw_sci_fiber_bucket(img)
        self.draw_sky_fiber_bucket(img)
        self.draw_centroids(img,stars)
        pass

#-------------------------------------------------------------------------------
    def adjust_brightness_contrast(self,input_img,brightness=0,contrast=0):
        '''
        Temorarily including display functionality here via CV2 library.

        Adjust an image brightness and contrast prior for display purposes
        inputs:
        input_img: The image we are manipulating
        brighness: 0 means unchanged.  
        '''
        
        if (brightness != 0):
            if (brightness > 0):
                shadow = brightness
                highlight = 255
            else:
                shadow = 0
                highlight = 255 + brightness
            alpha_b = (highlight - shadow) / 255
            gamma_b = shadow
        
            buf = cv2.addWeighted(input_img,alpha_b,input_img,0,gamma_b)
        else:
            buf = input_img.copy()
    
        if (contrast != 0):
            f = 131 * (contrast + 127) / (127 * (131 - contrast))
            alpha_c = f
            gamma_c = 127 * (1 - f)
        
            buf = cv2.addWeighted(buf,alpha_c,buf,0,gamma_c)

        return(buf)

#-------------------------------------------------------------------------------
    def refresh_image(self):
        self.rebin_factor = max(self.roi_x2-self.roi_x1,
                                self.roi_y2-self.roi_y1) / self.screen_res
        self.frame_scaled = cv2.resize(self.image,
                                       (int(np.round((self.roi_y2-self.roi_y1)/
                                                     self.rebin_factor)),
                                        int(np.round((self.roi_x2-self.roi_x1)/
                                                     self.rebin_factor))),
                                       interpolation=cv2.INTER_NEAREST)

        frame_scaled_min = np.min(self.frame_scaled)
        frame_scaled_max = np.max(self.frame_scaled)
        if frame_scaled_max < 20:
            frame_scaled_max = 20
        self.frame_uint8 = ((self.frame_scaled / frame_scaled_max) *
                            255).astype(np.uint8)
        frame_uint8_scaled = self.adjust_brightness_contrast(
            self.frame_uint8,contrast=self.contrast,
            brightness=self.brightness)
        
        frame_uint8_scaled = cv2.applyColorMap(frame_uint8_scaled,
                                               cv2.COLORMAP_JET)
        self.annotate_image(frame_uint8_scaled,self.stars)            
        cv2.imshow(self.display_name,frame_uint8_scaled)

        # q to quit, spacebar to stop/start framing, r to reset colormap
        k = cv2.waitKey(1)
        if(k & 0xFF == ord('q')):
            cv2.destroyAllWindows()
            exit()
        elif (k & 0xFF == ord(' ')):
            self.command_sender.send({'toggle_framing':1})
        elif (k & 0xFF == ord('r')):
            self.brightness = 0.0
            self.contrast = 0.0
                
            frame_uint8_scaled = self.adjust_brightness_contrast(
                self.frame_uint8,contrast=self.contrast,
                brightness=self.brightness)
            frame_uint8_scaled = cv2.applyColorMap(frame_uint8_scaled,
                                                   cv2.COLORMAP_JET)
            self.annotate_image(frame_uint8_scaled,self.stars)
            cv2.imshow(self.display_name,frame_uint8_scaled)

    
#------------------------------------------------------------------------------
    def on_guider_data(self,new_data):
        print("Received guider data");
        ts = new_data['timestamp']
        tstamp  = datetime.datetime.strptime(ts,'%Y-%m-%dT%H:%M:%S.%f')
        newtime = dates.date2num(tstamp)
        self.camera_xdim_pix = new_data['camera_xdim_pix']
        self.camera_ydim_pix = new_data['camera_ydim_pix']
        self.stars = np.asarray(new_data['stars'])
        self.platescale = new_data['platescale']
        self.roi_x1 = new_data['roi_x1']
        self.roi_x2 = new_data['roi_x2']
        self.roi_y1 = new_data['roi_y1']
        self.roi_y2 = new_data['roi_y2']
        self.roi_xdim = self.roi_x2 - self.roi_x1 + 1
        self.roi_ydim = self.roi_y2 - self.roi_y1 + 1
        self.roi_xcenter = self.roi_xdim/2
        self.roi_ycenter = self.roi_ydim/2

        self.region_xcen = (self.roi_xcenter -
                            new_data['x_mispointing']/self.platescale)
        self.region_ycen = (self.roi_ycenter -
                            new_data['y_mispointing']/self.platescale)
        self.region_radius = new_data['fwhm']/self.platescale
        self.sci_fiber_x = new_data['sci_fiber_x']
        self.sci_fiber_xwidth = new_data['sci_fiber_xwidth']
        self.sci_hole_xwidth = new_data['sci_hole_xwidth']
        self.sky_fiber_x = new_data['sky_fiber_x']
        self.sky_fiber_xwidth = new_data['sky_fiber_xwidth']
        self.sky_hole_xwidth = new_data['sky_hole_xwidth']
        self.guider_magnification = new_data['guider_magnification']
        self.arcsec_per_um_guider_ccd= new_data['arcsec_per_um_guider_ccd']
        self.sci_fiber_y = new_data['sci_fiber_y']
        self.sci_fiber_ywidth= new_data['sci_fiber_ywidth']
        self.sci_hole_ywidth = new_data['sci_hole_ywidth']
        self.sky_fiber_y = new_data['sky_fiber_y']
        self.sky_fiber_ywidth = new_data['sky_fiber_ywidth']
        self.sky_hole_ywidth = new_data['sky_hole_ywidth']
        self.guider_magnification = new_data['guider_magnification']
        self.arcsec_per_um_guider_ccd = new_data['arcsec_per_um_guider_ccd']
        self.camera_xcenter_pix = new_data['camera_xcenter_pix']
        self.camera_ycenter_pix = new_data['camera_ycenter_pix']
        self.pixel_size_um = new_data['pixel_size_um']
        self.image = np.asarray(new_data['image'])
        self.refresh_image()
        print("Done ingesting guider data")


#------------------------------------------------------------------------------ 
    def start(self):
        flags = (cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO |
                 cv2.WINDOW_GUI_NORMAL)
        cv2.namedWindow(self.display_name,flags)
        cv2.resizeWindow(self.display_name,self.screen_res,
                         self.screen_res)
            
        cv2.setMouseCallback(self.display_name,self.mouse_event,self)
   
################################################################################
if __name__ == "__main__":
    base_directory = './'

    config_file = 'tres_display.ini'

    tresdisplay = TresDisplay(base_directory,config_file)
    tresdisplay.start()
    loop = asyncio.get_event_loop()
    try:
        loop.run_forever()
    finally:
        loop.close()
    
    print("TRES Display stopping")

