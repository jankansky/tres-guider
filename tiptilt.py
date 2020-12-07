from configobj import ConfigObj
from pipython import GCSDevice, pitools
import socket
import math
import ipdb
import sys, os
import logging
import utils
import numpy as np
import argparse
import time
import redis

class TipTilt:

    def __init__(self, base_directory, config_file, logger=None,
                 simulate=False):
        self.base_directory=base_directory
        self.config_file = self.base_directory + '/config/' + config_file

        self.simulate=simulate
        
        # set up the log file
        if logger == None:
            self.logger = utils.setup_logger(base_directory + '/log/', 'tiptilt')
        else: self.logger = logger
        
        # read the config file
        if os.path.exists(self.config_file):
            config = ConfigObj(self.config_file)
        else:
            self.logger.error('Config file not found: (' + self.config_file + ')')
            sys.exit()

        self.redis = redis.Redis(host=config['REDIS_SERVER'],
                                 port=config['REDIS_PORT'])
            
        # serial number of the TIP/TILT stage and controller
        self.sn = config['SN_TIPTILT']
        self.model = config['MODEL_TIPTILT']
        self.sn_controller = config['SN_CONTROLLER']
        self.model_controller = config['MODEL_CONTROLLER']

        # stage steps per arcsecond on the sky 
        self.steps_per_arcsec = float(config['STEPS_PER_ARCSEC'])
        
        # angle between North and axis A, in radians
        self.theta = float(config['THETA'])*math.pi/180.0

        # if there's a flip in the mapping, apply it
        if config['FLIP']: self.sign = -1.0 
        else: self.sign = 1.0

        # use the PI python library to initialize the device
        if not self.simulate: self.tiptilt = GCSDevice()

        self.get_allowed_ranges()
        
        self.position = {'A':0.0,'B':0.0}
        
    def get_allowed_ranges(self):

        if self.simulate:
            self.mintip = -100.0
            self.maxtip = 100.0
            self.mintilt = -100.0
            self.maxtilt = 100.0
        else:
            self.mintip = self.tiptilt.qTMN()['A']
            self.maxtip = self.tiptilt.qTMX()['A']
            self.mintilt = self.tiptilt.qTMN()['B']
            self.maxtilt = self.tiptilt.qTMX()['B']

    def connect(self):

        if self.simulate:
            self.get_allowed_ranges()
            return

        if self.tiptilt.IsConnected():
            self.logger.info('Already connected')
        
        usbdevices = self.tiptilt.EnumerateUSB()

        #"*** GCSError: There is no interface or DLL handle with the given ID (-9)" That error requires a power cycle

        found = False
        if len(usbdevices) == 0:
            self.logger.error("No PI devices found")
            sys.exit()
            
        for device in usbdevices:
            if self.sn_controller in device:
                found = True
        if not found:
            self.logger.error('Serial number in ' + self.config_file + ' (' + self.sn_controller + ') does not match any of the connected USB devices; check power and USB')
            for device in usbdevices:
                self.logger.info(str(device) + ' is connected')
            sys.exit()

        self.tiptilt.ConnectUSB(serialnum=self.sn_controller)
        #self.tiptilt.ConnectUSB(serialnum=self.sntiptilt)
        if not self.tiptilt.IsConnected():
            self.logger.error('Error connecting to device')

        pitools.startup(self.tiptilt)#, stages=['S-340'], servostates=[True])#, refmodes=('FNL'))

        # get the allowed ranges
        self.get_allowed_ranges()
        
        # move to the middle of the range
        self.move_tip_tilt(1.0,1.0)
        
    '''
    Move the tip/tilt stage directly in stage coordinates. Check for
    limits, connectivity, etc
    '''
    def move_tip_tilt(self, tip, tilt):
        # make sure we're connected
        if not self.simulate:
            if not self.tiptilt.IsConnected():
                self.connect()

        # check bounds of requested move
        if tip < self.mintip or tip > self.maxtip or tilt < self.mintilt or tilt > self.maxtilt:
            self.logger.error("Requested move out of range")
            return

        self.logger.info("moving to tip = " + str(tip) + ', tilt=' + str(tilt))
        
        # move
        if not self.simulate:
            self.tiptilt.MOV('A',tip)
            self.tiptilt.MOV('B',tilt)

        self.position['A'] = tip
        self.position['B'] = tilt

        self.redis.set('tip',tip)
        self.redis.set('tilt',tilt)
        
        # wait for move?
        
        # make sure it moved where we wanted
        #if position != self.get_position():
        #    pass

        return

    '''
    Move the tip/tilt stage North and East a specified number of
    arcseconds on the sky 
    '''
    def move_north_east(self, north, east):

        # get current position
        ttpos = self.get_position()
                
        # translate north & east (arcsec on sky) to 
        # tip and tilt (stage steps)
        # requires angle, magnitude, and flip
        tip  = (ttpos['A'] + self.sign * self.steps_per_arcsec * (north * math.cos(self.theta) -
                                                                  east * math.sin(self.theta)))
        tilt = (ttpos['B'] + self.steps_per_arcsec*(north * math.sin(self.theta) +
                                                    east * math.cos(self.theta)))

        # move the tip/tilt
        return self.move_tip_tilt(tip,tilt)


    # qPOS doesn't seem to work, but it seems to move
    # keep track of the position ourselves
    def get_position(self):

        return self.position
        
        # make sure we're connected
        if not self.tiptilt.IsConnected():
            self.connect()

        return self.tiptilt.qPOS()
        return (self.tiptilt.qPOS('A'),self.tiptilt.qPOS('B'))

    def populate_header(self,hdr):
        ttpos = self.get_position()
        hdr['TTTIPPOS'] = (ttpos['A'],
                           "Tip Position of the tip/tilt stage (urad)")
        hdr['TTTILPOS'] = (ttpos['B'],
                           "Tilt Position of the tip/tilt stage (urad)")
        hdr['TTMODEL'] = (self.model,
                          "Model number of the tip/tilt stage")
        hdr['TTSN'] = (self.sn,
                       "Serial number of the tip/tilt stage")
        hdr['TTCMODEL'] = (self.model_controller,
                           "Model number of the tip/tilt stage controller")
        hdr['TTCSN'] = (self.sn_controller,
                        "Serial number of the tip/tilt stage controller")

    

if __name__ == '__main__':


    parser = argparse.ArgumentParser(description='Control the tip/tilt stage for the TRES front end')
    parser.add_argument('--home'  , dest='home'  , action='store_true', default=False, help='Move the tip/tilt stage to the center of motion')
    parser.add_argument('--move1'  , dest='move1'  , action='store_true', default=False, help='Move the tip/tilt stage to the (0,0) extreme')
    parser.add_argument('--move2'  , dest='move2'  , action='store_true', default=False, help='Move the tip/tilt stage to the (0,2) extreme')
    parser.add_argument('--move3'  , dest='move3'  , action='store_true', default=False, help='Move the tip/tilt stage to the (2,2) extreme')
    parser.add_argument('--move4'  , dest='move4'  , action='store_true', default=False, help='Move the tip/tilt stage to the (2,0) extreme')
    opt = parser.parse_args()
    
    if socket.gethostname() == 'tres-guider':
        base_directory = '/home/tres/tres-guider'
    else:
        print('unsupported system')
        sys.exit()

    config_file = 'tiptilt.ini'
    tiptilt = TipTilt(base_directory, config_file)

    tiptilt.connect()

    if opt.home:
        tiptilt.move_tip_tilt(1.0,1.0)
        time.sleep(5)
        tiptilt.move_tip_tilt(0.0,0.0)
        time.sleep(5)
        tiptilt.move_tip_tilt(2.0,2.0)
        time.sleep(5)
        tiptilt.move_tip_tilt(1.0,1.0)
    elif opt.move1:
        tiptilt.move_tip_tilt(0.0,0.0)
    elif opt.move2:
        tiptilt.move_tip_tilt(0.0,2.0)
    elif opt.move3:
        tiptilt.move_tip_tilt(2.0,2.0)
    elif opt.move4:
        tiptilt.move_tip_tilt(2.0,0.0)

#    print(tiptilt.get_position())

#    tiptilt.move_tip_tilt(0.0,0.0)
#    print(tiptilt.get_position())

    ipdb.set_trace()

