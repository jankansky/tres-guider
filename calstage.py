from configobj import ConfigObj
from pipython import GCSDevice, pitools
import socket
import math
import ipdb
import sys, os
import utils
import logging
import time

################################################################################
class CalStage:
    def __init__(self, base_directory, config_file, logger=None,
                 simulate=False):
        self.base_directory=base_directory
        self.config_file = self.base_directory + '/config/' + config_file

        # set up the log file
        if logger == None:
            self.logger = utils.setup_logger(base_directory + '/log/',
                                             'calstage')
        else: self.logger = logger

        # read the config file
        config = ConfigObj(self.config_file)

        # read the config file
        if os.path.exists(self.config_file):
            config = ConfigObj(self.config_file)
        else:
            self.logger.error('Config file not found: (' +
                              self.config_file + ')')
            sys.exit()

        self.science_position = float(config['SCIENCEPOS'])
        self.sky_position = float(config['SKYPOS'])
        self.out_position = float(config['OUTPOS'])
        self.sn = config['SNCALSTAGE']
        self.model = config['MODEL']
        self.sn_controller = config['SNCONTROLLER']
        self.model_controller = config['MODELCONTROLLER']
        self.minpos = None
        self.maxpos = None
        self.simulate = simulate
        self.simulated_position = 0.0
        
        # use the PI python library to initialize the device
        if not self.simulate: self.calstage = GCSDevice()

#-------------------------------------------------------------------------------
    def connect(self):
        '''
        Search for USB PI devices on the system. Connect, enable servo, and
        home to negative limit.
        '''
        # if simulating, just wait a second and return
        if self.simulate:
            time.sleep(1.0)
            return
        
        usbdevices = self.calstage.EnumerateUSB()

        found = False
        if len(usbdevices) == 0:
            self.logger.error("No PI devices found")
            sys.exit()
            
        for device in usbdevices:
            if self.sn_controller in device:
                found = True
        if not found:
            self.logger.error('Serial number in ' + self.config_file +
                              ' (' + self.sn_controller +
                              ') does not match any of the connected \
                              USB devices; check power and USB')
            for device in usbdevices:
                self.logger.info(str(device) + ' is connected')
            sys.exit()		

        self.calstage.ConnectUSB(serialnum=self.sn_controller)
        if not self.calstage.IsConnected():
            self.logger.error('Error connecting to device')

        # enable servo and home to negative limit if necessary
        pitools.startup(self.calstage, refmodes=('FNL'))
        
        # enable servo home to center if necessary
        #pitools.startup(self.calstage, refmodes=('FRF')) 

#-------------------------------------------------------------------------------
    def allowedMove(self, position):
        '''
        Check if position is an allowable place to go. Return boolean.
        '''
        if self.minpos == None:
            if self.simulate:
                self.minpos = 0.0
            else:
                self.minpos = self.calstage.qTMN()['1']
        if self.maxpos == None:
            if self.simulate:
                self.maxpos = 26.0
            else:
                self.maxpos = self.calstage.qTMX()['1']
            
        if position > self.maxpos:
            return(False)
        if position < self.minpos:
            return(False)
        return(True)
    
#-------------------------------------------------------------------------------
    def move(self, position):
        '''
        Issue the command to move to specified position without waiting, and
        return True if commanding was successful. Return false if stage is
        disconnected, or position demand is out of range.
        '''
        if self.simulate:
            time.sleep(1.0)
            return(True)
        else:
            # make sure we're connected
            if not self.calstage.IsConnected():
                self.connect()
                if not self.calstage.IsConnected():
                    self.logger.error("Error connecting to stage")
                    return(False)

        # make sure the move is in range
        if not self.allowedMove(position):
            self.logger.error("Requested move out of bounds")
            return(False)
           
        # move the stage
        self.logger.info("moving the stage to " + str(position))
        if self.simulate:
            time.sleep(1.0)
            self.simulated_position = position
        else:
            self.calstage.MOV('1',position)
        
        # success!
        return(True)

#-------------------------------------------------------------------------------
    def move_and_wait(self, position, tol=0.001):
        '''
        Issue the command to move to specified position and wait for move to
        complete. Return True if commanding was successful. Return False if
        stage is disconnected, or position demand is out of range.
        '''
        # move the stage
        if not self.move(position):
            return(False)

        # wait for it to stop
        if not self.simulate:
            pitools.waitontarget(self.calstage, axes=['1'])

        # make sure it moved where we wanted (within tolerance)
        if abs(position - self.get_position()) > tol:
            self.logger.error("Error moving to requested position")
            return(False)
        
        # success!
        return(True)

#-------------------------------------------------------------------------------
    def get_position(self):
        '''
        Return current cal stage position
        '''
        if self.simulate:
            return(self.simulated_position)
        
        return (self.calstage.qPOS()['1'])

#-------------------------------------------------------------------------------
    def get_position_string(self, tolerance=0.01):
        '''
        Translate current stage position to a string indicating the current mode
        '''
        position = self.get_position()
        
        if abs(position - self.science_position) < tolerance:
            return('Science')
        if abs(position - self.sky_position) < tolerance:
            return('Sky')
        if abs(position - self.out_position) < tolerance:
            return('Out')
        return 'Unknown'
    
#-------------------------------------------------------------------------------
    def move_to_science(self):
        '''
        Move stage to science position
        '''
        return self.move_and_wait(self.science_position)

#-------------------------------------------------------------------------------
    def move_to_sky(self):
        '''
        Move stage to sky position
        '''        
        return self.move_and_wait(self.sky_position)

#-------------------------------------------------------------------------------
    def move_to_out(self):
        '''
        Move stage to out position
        '''
        return self.move_and_wait(self.out_position)

#-------------------------------------------------------------------------------
    def populate_header(self,hdr):
        '''
        Fill in the FITS header passed in with the current state
        '''
        hdr['CSPOS'] = (self.get_position_string(),
                        "Position of the cal stage (string)")
        hdr['CSPOSN'] = (self.get_position(),
                         "Position of the cal stage (mm)")
        hdr['CSMODEL'] = (self.model,"Model Number of the cal stage")
        hdr['CSSN'] = (self.sn,
                       "Serial number of the calibration stage")
        hdr['CSCMODEL'] = (self.model_controller,
                           "Model of the cal stage controller")
        hdr['CSCSN'] = (self.sn_controller,
                        "Serial number of the cal stage controller")

################################################################################
if __name__ == '__main__':

    if socket.gethostname() == 'tres-guider':
        base_directory = '/home/tres/tres-guider'
    elif socket.gethostname() == 'core2duo':
        base_directory = '/home/jkansky/tres-guider-jan'                
    elif socket.gethostname() == 'Jason-THINK':
        base_directory = 'C:/tres-guider/'
    else:
        base_directory = "./"
        print('unsupported system')

    config_file = 'calstage.ini'
    calstage = CalStage(base_directory, config_file)

    ipdb.set_trace()
    calstage.connect()
    calstage.move_to_science()
    calstage.move_to_sky()
    calstage.move_to_out()

    ipdb.set_trace()
