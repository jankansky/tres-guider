#!/usr/bin/python3
import sys
import os
import socket
import utils
from configobj import ConfigObj
import asyncio
import aioredis
from collections import deque
import datetime
import struct
import redis
import json

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import dates

#import pyds9

from PyQt5 import QtCore, QtWidgets
from PyQt5.QtWidgets import QApplication, QWidget
from PyQt5.QtCore import pyqtSignal, pyqtSlot, QSettings, Qt

from quamash import QEventLoop, QThreadExecutor


from tres_guider_control_ui import Ui_Dialog

################################################################################
class EventSender():
    '''Send guider events to Redis for GUI usage'''
    
    def __init__(self,config):
        self.server_host = config['REDIS_SERVER']
        self.server_port = config['REDIS_PORT']
        self.redis = redis.Redis(host=self.server_host,
                                 port=self.server_port)
        self.command_counter = 0

    def send(self,command_dict):
        '''
        Pass in a dict of key value pairs to set in underlying code
        '''
        command_json_struct = json.dumps(command_dict)
        res = self.redis.publish('guider_commands',command_json_struct)
        self.command_counter += 1
        print("Sent command %i" % (self.command_counter,))
        return(res)


################################################################################
class TresGUI(QWidget):
    def __init__(self, base_directory,config_file,event_loop,
                 logger=None,parent = None):
        '''
        Constructor
        '''
        super(TresGUI,self).__init__()
        
        self.base_directory=base_directory
        self.config_file = self.base_directory + '/config/' + config_file

        # set up the log file
        if logger == None:
            self.logger = utils.setup_logger(base_directory + '/log/', 'tres')
        else: self.logger = logger


        # read the config file
        if os.path.exists(self.config_file):
            self.config = ConfigObj(self.config_file)
        else:
            self.logger.error('Config file not found: (' + \
                              self.config_file + ')')
            sys.exit()        
        
        QWidget.__init__(self,parent)
        self.ui = Ui_Dialog()
        self.ui.setupUi(self)

#        self.ui.loopButton.setOutbg('binoRed')
#        self.ui.loopButton.setInbg('binoGreen')
#        self.ui.loopButton.setOuttext('Open')
#        self.ui.loopButton.setIntext('Closed')
#        self.ui.loopButton.on_toggled(False)

        self.dequelen  = 20
        self.axrect = [0.1, 0.15, 0.8, 0.7]
        
        self.gdr_focus_axis_lim = 0.25
        self.gdr_focus_data  = {'timestamp'  : deque([], self.dequelen), 
                          'fwhm' : deque([], self.dequelen)}
        self.gdr_focus_fig   = self.ui.focusMplwidget.figure   
        self.plot_focus_cnt  = -1     
#        self.replotf()

        self.gdr_counts_axis_lim = 0.25
        self.gdr_counts_data  = {'timestamp'  : deque([], self.dequelen), 
                          'counts' : deque([], self.dequelen)}
        self.gdr_counts_fig   = self.ui.bcountsMplwidget.figure   
        self.plot_counts_cnt  = -1     


        self.gdr_tilt_axis_lim = 0.25
        self.gdr_tilt_data_x = {'timestamp' : deque([], self.dequelen), 
                                'dx' : deque([], self.dequelen)}
        self.gdr_tilt_data_y = {'timestamp' : deque([], self.dequelen), 
                                'dy' : deque([], self.dequelen)}
        self.gdr_tilt_fig = self.ui.azelMplwidget.figure        
        #         self.replott()

        self.plot_tilt_cnt = -1

#        self.ds9 = pyds9.DS9('TRES Guider')
#        self.ds9.set('zoom to fit')
#        self.ds9.set('scale zscale')

        self.guider_xdim = 2048  # Defaults for Zyla 4.2 sCMOS camera
        self.guider_ydim = 2048        
        self.roi_x1 = 996
        self.roi_x2 = 1052
        self.roi_y1 = 996
        self.roi_y2 = 1052
        self.roi_xdim = self.roi_x2 - self.roi_x1 + 1
        self.roi_ydim = self.roi_y2 - self.roi_y1 +1
        
        self.region_xcen = self.roi_xdim/2
        self.region_ycen = self.roi_ydim/2
        self.region_radius = 20

        self.event_tree = {'loop_state':self.on_loop_changed,
                             'framing':self.on_framing_changed}
        
        self.receive_task = asyncio.ensure_future(
            self.receive_telem(self.on_guider_data,
                               self.config['REDIS_SERVER'],
                               self.config['REDIS_PORT'],
                               'guider_data'))

        self.receive_event_task = asyncio.ensure_future(
            self.receive_events(self.on_events,
                               self.config['REDIS_SERVER'],
                               self.config['REDIS_PORT'],
                               'guider_state'))
        
        self.command_sender = EventSender(self.config)

#-------------------------------------------------------------------------------
#    async def master(self):
#        with QThreadExecutor(1) as exec:
#            await loop.run_in_executor(exec,self.receive_telem)
            
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


    @staticmethod
    async def receive_image(callback,server,port,data_name):
        redis_sub = await aioredis.create_redis('redis://'+server+':'+port)
        res = await redis_sub.subscribe(data_name)
        sub_channel = res[0]        
        
        receive_counter = 0        
        while (await sub_channel.wait_message()):
            msg = await sub_channel.get()
            callback(msg)
            receive_counter += 1
            
    @staticmethod
    async def receive_events(callback,server,port,data_name):
        redis_sub = await aioredis.create_redis('redis://'+server+':'+port)
        res = await redis_sub.subscribe(data_name)
        sub_channel = res[0]        
        
        receive_counter = 0        
        while (await sub_channel.wait_message()):
            msg = await sub_channel.get_json()
            callback(msg)
            receive_counter += 1

            

#-------------------------------------------------------------------------------
    def replot_counts(self):
        # bucket counts plot
#        print('entering replot_counts')
        
        self.plot_counts_cnt += 1
        initplots = False        
        if self.plot_counts_cnt % 10 == 0:
            initplots = True
            
        if initplots:
            self.gdr_counts_fig.clear()
            self.gdrbax = self.gdr_counts_fig.add_axes(self.axrect)
        else:
            self.gdrbax.clear()

#         self.gdr_counts_fig.clear()
#         self.gdrbax = self.gdr_counts_fig.add_axes(self.axrect)
        self.gdrbax.set_title('Bucket counts / 1000')
        hfmt = dates.DateFormatter('%H:%M:%S')
        if len(self.gdr_counts_data['timestamp']) > 0:
            self.gdrbax.xaxis.set_major_locator(dates.MinuteLocator())
            self.gdrbax.xaxis.set_major_formatter(hfmt)
        ylim = self.gdr_counts_axis_lim
        tmp = list(self.gdr_counts_data['counts'])
        if len(tmp) >= 1:
            maxx = abs(max(tmp))
            minx = abs(min(tmp))
            if minx > maxx:
                maxx = minx
            if maxx > ylim:
                ylim = maxx
        self.gdrbax.set_ylim([0, ylim * 1.1])
        self.gdrbax.grid(True)
        self.gdrbax.plot(self.gdr_counts_data['timestamp'], self.gdr_counts_data['counts'], '-go')
#         self.gdrbax.axhline(1.5, 0, 1, color = 'g')
#         self.gdrbax.axhline(3.0, 0, 1, color = 'y')
#         self.gdrbax.axhline(-1.5, 0, 1, color = 'g')
#         self.gdrbax.axhline(-3.0, 0, 1, color = 'y')
        self.ui.bcountsMplwidget.draw()
#        print('leaving replot_counts')

#-------------------------------------------------------------------------------
    def replot_focus(self):
        # focus plot
        self.plot_focus_cnt += 1
        initplots = False        
        if self.plot_focus_cnt % 10 == 0:
            initplots = True
            
        if initplots:
            self.gdr_focus_fig.clear()
            self.gdrfax = self.gdr_focus_fig.add_axes(self.axrect)  # aspect ratio
        else:
            self.gdrfax.clear()

#         self.gdr_focus_fig.clear()
#         self.gdrfax = self.gdr_focus_fig.add_axes(self.axrect)
        self.gdrfax.set_title('Average FWHM ["]')
        hfmt = dates.DateFormatter('%H:%M:%S')
        if len(self.gdr_focus_data['timestamp']) > 0:
            self.gdrfax.xaxis.set_major_locator(dates.MinuteLocator())
            self.gdrfax.xaxis.set_major_formatter(hfmt)
        ylim = self.gdr_focus_axis_lim
        tmp = list(self.gdr_focus_data['fwhm'])
        if len(tmp) >= 1:
            maxx = abs(max(tmp))
            minx = abs(min(tmp))
            if minx > maxx:
                maxx = minx
            if maxx > ylim:
                ylim = maxx
        self.gdrfax.set_ylim([0.0, ylim * 1.1])
        self.gdrfax.grid(True)
        self.gdrfax.plot(self.gdr_focus_data['timestamp'], self.gdr_focus_data['fwhm'], '-go')
#         self.gdrfax.legend(['Box x', 'Box y', 'Star x', 'Star y'])
#         self.gdrfax.axhline(1.5, 0, 1, color = 'g')
#         self.gdrfax.axhline(3.0, 0, 1, color = 'y')
#         self.gdrfax.axhline(-1.5, 0, 1, color = 'g')
#         self.gdrfax.axhline(-3.0, 0, 1, color = 'y')
        self.ui.focusMplwidget.draw()
            
#-------------------------------------------------------------------------------
    def replot_tilts(self):
#        print('entering replot_tilts')
        
        self.plot_tilt_cnt += 1
        initplots = False        
        if self.plot_tilt_cnt % 10 == 0:
            initplots = True
            
        if initplots:
            self.gdr_tilt_fig.clear()
            self.gdrtax = self.gdr_tilt_fig.add_axes(self.axrect)  # aspect ratio
        else:
            self.gdrtax.clear()
       
        self.gdrtax.set_title('Pointing error [Instrument "]')
        hfmt = dates.DateFormatter('%H:%M:%S')
        if len(self.gdr_tilt_data_x['timestamp']) > 0:
            self.gdrtax.xaxis.set_major_locator(dates.MinuteLocator())
            self.gdrtax.xaxis.set_major_formatter(hfmt)
        ylim = self.gdr_tilt_axis_lim
        tmp = list(self.gdr_tilt_data_x['dx']) + list(self.gdr_tilt_data_y['dy'])
        if len(tmp) >= 1:
            maxx = abs(max(tmp))
            minx = abs(min(tmp))
            if minx > maxx:
                maxx = minx
            if maxx > ylim:
                ylim = maxx
        self.gdrtax.set_ylim([-ylim * 1.1, ylim * 1.1])
        self.gdrtax.grid(True)
        self.gdrtax.plot(self.gdr_tilt_data_x['timestamp'], self.gdr_tilt_data_x['dx'], '-go')
        self.gdrtax.plot(self.gdr_tilt_data_y['timestamp'], self.gdr_tilt_data_y['dy'], '-ro')
        self.gdrtax.legend(['dx', 'dy'],loc='upper left',framealpha=0.4)
#         self.gdrtax.axhline(1.5, 0, 1, color = 'g')
#         self.gdrtax.axhline(3.0, 0, 1, color = 'y')
#         self.gdrtax.axhline(-1.5, 0, 1, color = 'g')
#         self.gdrtax.axhline(-3.0, 0, 1, color = 'y')
        self.ui.azelMplwidget.draw()
#        print('leaving replot_tilts')
            
 #------------------------------------------------------------------------------
    def on_guider_data(self,new_data):
#        print("Received guider data");
        ts = new_data['timestamp']
        tstamp  = datetime.datetime.strptime(ts,'%Y-%m-%dT%H:%M:%S.%f')
        newtime = dates.date2num(tstamp)
        self.gdr_tilt_data_x['timestamp'].append(newtime)
        self.gdr_tilt_data_x['dx'].append(new_data['x_mispointing'])
        self.gdr_tilt_data_y['timestamp'].append(newtime)
        self.gdr_tilt_data_y['dy'].append(new_data['y_mispointing'])
        self.gdr_focus_data['timestamp'].append(newtime)
        self.gdr_focus_data['fwhm'].append(new_data['fwhm'])        
        self.gdr_counts_data['timestamp'].append(newtime)
        self.gdr_counts_data['counts'].append(new_data['counts'])
        self.platescale = new_data['platescale']
        self.roi_x1 = new_data['roi_x1']
        self.roi_x2 = new_data['roi_x2']
        self.roi_y1 = new_data['roi_y1']
        self.roi_y2 = new_data['roi_y2']
        self.roi_xdim = self.roi_x2 - self.roi_x1 + 1
        self.roi_ydim = self.roi_y2 - self.roi_y1 + 1
        self.roi_xcenter = self.roi_xdim/2
        self.roi_ycenter = self.roi_ydim/2

        self.region_xcen = self.roi_xcenter - new_data['x_mispointing']/self.platescale
        self.region_ycen = self.roi_ycenter - new_data['y_mispointing']/self.platescale
        self.region_radius = new_data['fwhm']/self.platescale

#        self.ds9.set('regions', 'image; circle(%f,%f,%f)' %
#                     (self.region_xcen,self.region_ycen,self.region_radius))
#        print(new_data)
        self.replot_tilts()
        self.replot_focus()
        self.replot_counts()

#        print("Received image data");
#        h,w = struct.unpack('>II',new_data[:8])
#        image = np.frombuffer(new_data[8:],dtype=np.int16).reshape(h,w)
#        ts = new_data['timestamp']
#        tstamp  = datetime.datetime.strptime(ts,'%Y-%m-%dT%H:%M:%S.%f')
#        newtime = dates.date2num(tstamp)
#        print("Got new image")
#        self.ds9.set_np2arr(image)

#------------------------------------------------------------------------------
    def on_events(self,new_events_dict):
        print("Got events")
        print(new_events_dict)              
        for new_event in new_events_dict:
            print(new_event)
            print(new_events_dict[new_event])
            res = self.event_tree[new_event](new_events_dict[new_event])
        print("Done processing events");


#-------------------------------------------------------------------------------
    def on_dx(self,message):
        print(message)

#-------------------------------------------------------------------------------
    def on_dy(self,message):
        print(message)

#-------------------------------------------------------------------------------
    def on_gdr_onoff(self):
        pass
    
#-------------------------------------------------------------------------------
    def on_set_offsets(self):
        pass

#-------------------------------------------------------------------------------
    def on_enable(self):
        pass

#-------------------------------------------------------------------------------
    def on_standby(self):
        pass

#-------------------------------------------------------------------------------
    def on_details(self):
        pass

#-------------------------------------------------------------------------------
    def on_config(self):
        pass

#-------------------------------------------------------------------------------
    def on_azelxy(self):
        pass

#-------------------------------------------------------------------------------
    def on_change_loop(self,checked):
        print("In on_change_loop")
        if self.ui.loopButton.text() == 'Open':
            res = self.command_sender.send({'loop_state':1})
            print("Closing loops")
        elif self.ui.loopButton.text() == 'Closed':
            res = self.command_sender.send({'loop_state':0})
            print("Opening loops")

#-------------------------------------------------------------------------------
    def on_bucketsize(self):
        pass
    
#-------------------------------------------------------------------------------
    def on_loop_changed(self,sval):
        if sval == True:
            self.ui.loopButton.setText('Closed')
#            self.loopButton.toggled.emit(True)
        elif sval == False:
            self.ui.loopButton.setText('Open')            
#            self.loopButton.toggled.emit(False)

#-------------------------------------------------------------------------------
    def on_framing_changed(self,sval):
        if sval == True:
            pass
#            self.loopButton.setText('Closed')
#            self.loopButton.toggled.emit(True)
        elif sval == False:
            pass
#            self.loopButton.setText('Open')            
#            self.loopButton.toggled.emit(False)


    
################################################################################
if __name__ == "__main__":
    base_directory = './'

    config_file = 'tres_gui.ini'
    
    app = QApplication(sys.argv)
    loop = QEventLoop(app)
    loop.set_debug(True)
    asyncio.set_event_loop(loop)

    tresgui = TresGUI(base_directory,config_file,loop)
    tresgui.show()
    
    if sys.version_info[0] == 3 and sys.version_info[1] >= 8:
        asyncio.events._set_running_loop(loop)
        
    with loop:
        loop.run_forever()
        
        
    
