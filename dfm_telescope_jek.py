# Implementation of TCS from DFM here (pgs 17-32):
# https://arm.dfmsupport.com/jobs/857/TCSGO%20Manual%20v1.3.pdf
# local copy here in TCSGO_Manual_v1.3.pdf

from configobj import ConfigObj
from pipython import GCSDevice, pitools
import socket
import math
import ipdb
import sys, os
import utils
import logging
import time
import socket
import datetime
import select


################################################################################
class Telescope:
    ''' 
    this is a telescope class that implements DFM Galil TCS
    '''
    def __init__(self, base_directory, config_file, logger=None,
                 simulate=False):
        self.base_directory=base_directory
        self.config_file = self.base_directory + '/config/' + config_file

        # set up the log file
        if logger == None:
            self.logger = utils.setup_logger(base_directory + '/log/',
                                             'telescope')
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

        self.server = config['SERVER']
        self.port = int(config['PORT'])
        self.epoch = config['EPOCH']
        self.max_jog = float(config['MAX_JOG'])

#-------------------------------------------------------------------------------
    def send(self,cmd, timeout=5.0, readback=True):
        '''
        send a command to the Galil TCS
        '''
        s = socket.socket(socket.AF_INET,socket.SOCK_STREAM)

        try:
            s.connect((self.server,self.port))
            self.logger.info("Connected")
        except:
            self.logger.error("No response from the Galil TCS Server (" +
                              self.server +
                              ':' + str(self.port) +
                              ". Start TCSGalil and try again")
            ipdb.set_trace()

            sys.exit() # too harsh?

            return None
            
        self.logger.info("Sending command: " + cmd)
        s.send(cmd +'\r')

        if not readback: return None       
        
        data = ''
        t0 = datetime.datetime.utcnow()
        time_elapsed = 0.0
        while True and time_elapsed < timeout:
            time_elapsed = (datetime.datetime.utcnow()-t0).total_seconds()
            data += s.recv(1024)
            if data[-1] == ';': break

        self.logger.info("Closing connection")
        s.close()
            
        if time_elapsed > timeout:
            self.logger.error("Timeout sending command to TCS Galil")

        self.logger.info("Message received: " + data)
            
        return data

#-------------------------------------------------------------------------------
    def slew(self, raj2000, decj2000):

        if self.simulate:
            time.sleep(1.0)
            return True
        else:
            # make sure we're connected
            if not self.IsConnected():
                self.connect()
                if not self.IsConnected():
                    self.logger.error("Error connecting to telescope")
                    return False

        # make sure the move is in range
        if not self.allowedMove(raj2000, decj2000):
            self.logger.error("Requested move out of bounds")
            return False
           
        # slew the telescope
        self.logger.info("moving the telescope to RA (J2000)" + str(raj2000) +
                         ', DEC (J2000):' + str(decj2000))
        if self.simulate:
            time.sleep(1.0)
        else: self.send("#6," + str(raj2000) + "," + str(decj2000) + "," +
                        str(epoch) + ";")
        return True

#-------------------------------------------------------------------------------
    def offset_target_object(self,east,north):
        '''This jogs the telescope. Note if it is not in CHASE mode, 
        set_mount_mode must be set to 1(?)
        if requested jog is out of range, status['target_out_of_range'] will
        be True
        '''
        
        # this is a limit imposed by me here. No such limit exists in the
        # telescope
        if abs(east) > self.max_jog or abs(north) > self.max_jog:
            self.logger.error('Cannot jog more than ' + str(self.max_jog) +
                              '" in any direction')
            
        return self.send("#7," + str(east) + "," + str(north) + ";",
                         readback=False)

    
#-------------------------------------------------------------------------------
    def set_mount_mode(self, mode):
        if ((mode != 0) and (mode != 1) and (mode != 2) and (mode != 3)):
            self.logger.error("Supplied mount mode (" + str(mode) +
                              ") not allowed; must be 0 (Hold Position), 1 \
                              (Follow Target), 2 (Slew to Target), or \
                              3 (Chase Target)")
            return False
            
        return self.send("#12," + str(mode) + ";", readback=False)

#-------------------------------------------------------------------------------
    def read_mount_coordinates(self):
        response = self.send("#25;")

        # trim leading "#" and trailing ";", then split by ','
        arr = response[1:-1].split(',')
        coords = {}
        if len(arr) == 10:
            # convert date and time to a datetime object
            # what is a 'decimal year'? It doesn't change often enough to be
            # precise, nor is it an integer number of days...
            year = int(float(arr[9]))
            seconds = (float(arr[9])-year)*365.25*86400.0
            hours = float(arr[8])
            date = datetime.datetime(year,1,1,0) + \
                datetime.timedelta(seconds=seconds)
            
            coords['HA'] = arr[0]
            coords['RA'] = arr[1]
            coords['Dec'] = arr[2]
            coords['equinox'] = arr[3]
            coords['airmass'] = arr[4]
            coords['zenith_distance'] = arr[5]
            coords['azimuth'] = arr[6]
            coords['sidereal_time'] = arr[7]
            coords['date'] = date
        return coords
    
#-------------------------------------------------------------------------------
    def read_tcs_status(self):
        response = self.send("#26;")

        # trim leading "#" and trailing ";", then split by ','
        arr = response[1:-1].split(',')

        status = {}
        if len(arr) == 8:
            byte0 = format(int(arr[0]),'b').zfill(8)
            status['drive_switch_up']      = byte0[-1]=='1'
            status['track_switch_up']      = byte0[-2]=='1'
            status['aux_track_switch_up']  = byte0[-3]=='1'
            status['dome_track_switch_up'] = byte0[-4]=='1'
            status['auto_dome_switch_up']  = byte0[-5]=='1'
            status['amplifiers_switch_up'] = byte0[-6]=='1'

            byte1 = format(int(arr[1]),'b').zfill(8)
            status['north_button_pressed'] = byte1[-1]=='1'
            status['south_button_pressed'] = byte1[-2]=='1'
            status['east_button_pressed']  = byte1[-3]=='1'
            status['west_button_pressed']  = byte1[-4]=='1'
            status['set_button_pressed']   = byte1[-5]=='1'
            status['slew_button_pressed']  = byte1[-6]=='1'
            status['ascom_guiding']        = byte1[-7]=='1'

            byte2 = format(int(arr[2]),'b').zfill(8)
            status['horizon_limit_reached']  = byte2[-1]=='1'
            status['east_travel_limit']      = byte2[-2]=='1'
            status['west_travel_limit']      = byte2[-3]=='1'
            status['north_travel_limit']     = byte2[-4]=='1'
            status['south_travel_limit']     = byte2[-5]=='1'
            status['target_out_of_range']    = byte2[-6]=='1'
            status['approaching_soft_limit'] = byte2[-7]=='1'
            status['reached_soft_limit']     = byte2[-8]=='1'

            byte3 = format(int(arr[3]),'b').zfill(8)
            status['find_home_mode'] = byte3[-1]=='1'
            status['goto_az_mode']   = byte3[-2]=='1'
            status['track_mode']     = byte3[-3]=='1'
            status['rotating_left']  = byte3[-4]=='1'
            status['rotating_right'] = byte3[-5]=='1'
            status['at_home']        = byte3[-6]=='1'
            status['manual_mode']    = byte3[-7]=='1'

            byte4 = format(int(arr[4]),'b').zfill(8)
            status['focus_moving_in']        = byte4[-1]=='1'
            status['focus_moving_out']       = byte4[-2]=='1'
            status['focus_at_outer_limit']   = byte4[-3]=='1'
            status['focus_at_inner_limit']   = byte4[-4]=='1'
            status['instrument_rotator_ccw'] = byte4[-5]=='1'
            status['instrument_rotator_cw']  = byte4[-6]=='1'

            byte5 = format(int(arr[5]),'b').zfill(8)
            status['serial_comm_ready']        = byte5[-1]=='1'
            status['tcpip_connected']          = byte5[-2]=='1'
            status['use_cos_dec']              = byte5[-3]=='1'
            status['use_rate_corrections']     = byte5[-4]=='1'
            status['east_side_pier_operation'] = byte5[-5]=='1'
            status['west_side_pier_operation'] = byte5[-6]=='1'
            status['galil_data_is_stale']      = byte5[-7]=='1'

            byte6 = format(int(arr[6]),'b').zfill(8)
            status['mount_in_chase_target_mode']   = byte6[-1]=='1'
            status['mount_in_slew_to_target_mode'] = byte6[-2]=='1'
            status['mount_in_follow_target_mode']  = byte6[-3]=='1'
            status['mount_in_hold_position_mode']  = byte6[-4]=='1'
            status['target_captured']              = byte6[-5]=='1'
            
            byte7 = format(int(arr[7]),'b').zfill(8)
            status['dome_upper_shutter_open']   = byte7[-1]=='1'
            status['dome_upper_shutter_closed'] = byte7[-2]=='1'
            status['dome_lower_shutter_open']   = byte7[-3]=='1'
            status['dome_lower_shutter_closed'] = byte7[-4]=='1'
            status['mirror_doors_open']         = byte7[-5]=='1'
            status['mirror_doors_closed']       = byte7[-6]=='1'

        return status
            
#-------------------------------------------------------------------------------
    def apply_mount_corrections(self):
        '''
        turns on the pointing model corrections
        '''        
        self.logger.info("Turning on mount corrections")
        self.send('#66,1;')

#-------------------------------------------------------------------------------
    def unapply_mount_corrections(self):
        '''
        turns off the pointing model corrections
        '''
        self.logger.info("Turning off mount corrections")
        self.send('#66,0;')

#-------------------------------------------------------------------------------
    def populate_header(self,hdr):
        '''
        Retrieve and set telescope information FITS headers 
        (requires communication with TCS)
        '''
        
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
if __name__ == '__main__':
    base_directory = './'
    
    config_file = 'telescope.ini'
    telescope = telescope(base_directory, config_file)

    t0 = datetime.datetime.utcnow()
    status = telescope.read_tcs_status()
    print((datetime.datetime.utcnow() - t0).total_seconds())
    if status['target_out_of_range']:
        print('target out of range')


    # set the target to zenith
#    telescope.send('#10;',readback=False)
#    ipdb.set_trace()
    
    # change to chase target mode
    telescope.set_mount_mode(3)

    for i in range(100):
        time.sleep(1.5)
        status = telescope.read_tcs_status()
        print(status['mount_in_chase_target_mode'])
        print(status['mount_in_hold_position_mode'])
        print()

    ipdb.set_trace()

        
    time.sleep(0.75)
    status = telescope.read_tcs_status()
    print(status['mount_in_chase_target_mode'])
    print(status['mount_in_hold_position_mode'])
    print()

    status = telescope.read_tcs_status()
    print(status['mount_in_chase_target_mode'])
    print(status['mount_in_hold_position_mode'])
    print()

    status = telescope.read_tcs_status()
    print(status['mount_in_chase_target_mode'])
    print(status['mount_in_hold_position_mode'])
    print()
    
    coords1 = telescope.read_mount_coordinates()
    print("coords before slew")
    print(coords1)

    east = 0.0
    north = 100.0
    response = telescope.offset_target_object(east,north)

    coords2 = telescope.read_mount_coordinates()
    print()
    print("coords after slew")
    print(coords2)

    # if status queries fail, reboot TCSGalil, then reinitialize telescope
    # sending rapid fire commands seems to tank server (no more than 1 per
    # second)
    
    ipdb.set_trace()

