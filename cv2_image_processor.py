#!/usr/bin/python3
#from __future__ import division

import numpy as np
import cv2

################################################################################
class ImageProcessor():
    '''Manage image processing to compute guiding feedback'''
    def __init__(self,config):
        pass

#-------------------------------------------------------------------------------
    def populate_header(self,hdr):
        '''
        Populate FITS header passed in to set current Image processing state
        '''
        # WCS solution
        #hdr['EPOCH'] = (2000,'Epoch of coordinates')
        #hdr['SECPIX'] = (self.platescale,'arcsec/pixel')
        #hdr['CTYPE1'] = ("RA---TAN","TAN projection")
        #hdr['CTYPE2'] = ("DEC--TAN","TAN projection")
        #hdr['CUNIT1'] = ("deg","X pixel scale units")
        #hdr['CUNIT2'] = ("deg","Y pixel scale units")
        #hdr['CRVAL1'] = (ra,"RA of reference point")
        #hdr['CRVAL2'] = (dec,"DEC of reference point")
        #hdr['CRPIX1'] = (self.x_sci_fiber,"X reference pixel")
        #hdr['CRPIX2'] = (self.y_sci_fiber,"Y reference pixel")
        #hdr['CD1_1'] = (-self.platescale*math.cos(skypa),"DL/DX")
        #hdr['CD1_2'] = (self.platescale*math.sin(skypa),"DL/DY")
        #hdr['CD2_1'] = (self.platescale*math.sin(skypa),"DM/DX")
        #hdr['CD2_2'] = (self.platescale*math.cos(skypa),"DM/DY")
        pass

#-------------------------------------------------------------------------------
    def threshold_tozero(self,frame, otsu = False):
        '''
        '''
        gray = np.round(frame*255.0/65535).astype('uint8')
        if otsu == False:
            half = np.max(gray)/2
            ret, th2 = cv2.threshold(gray, half, 255, cv2.THRESH_TOZERO)
        if otsu == True:
            ret, th2 = cv2.threshold(gray, 0, 255,
                                     cv2.THRESH_TOZERO+cv2.THRESH_OTSU )
        return th2

#-------------------------------------------------------------------------------
    def threshold_binary(self,frame, otsu = False):
        gray = np.round(frame*255.0/65535).astype('uint8')
        if otsu == False:
            half = np.max(gray)/2
            ret, th2 = cv2.threshold(gray, half, 255, cv2.THRESH_BINARY)
        if otsu == True:
            ret, th2 = cv2.threshold(gray, 0, 255,
                                     cv2.THRESH_BINARY+cv2.THRESH_OTSU )
        return th2

#-------------------------------------------------------------------------------
    def robust_std(self,x):
        y = x.flatten()
        n = len(y)
        y.sort()
        ind_qt1 = int(round((n+1)/4.))
        ind_qt3 = int(round((n+1)*3/4.))
        IQR = y[ind_qt3]- y[ind_qt1]
        lowFense = y[ind_qt1] - 1.5*IQR
        highFense = y[ind_qt3] + 1.5*IQR
        ok = (y>lowFense)*(y<highFense)
        yy=y[ok]
        return yy.std(dtype='double')

#-------------------------------------------------------------------------------
    def threshold_robust(self,frame, otsu = False):
        gray = np.round(frame*255.0/65535).astype('uint8')
        if otsu == False:
            val = np.max(gray)/10
            ret, th2 = cv2.threshold(gray, val, 0, cv2.THRESH_TOZERO)
        if otsu == True:
            ret, th2 = cv2.threshold(gray, 0, 255,
                                     cv2.THRESH_BINARY+cv2.THRESH_OTSU )
        return th2
    
#-------------------------------------------------------------------------------
    def threshold_pyguide(self,image,level =3):
        stddev = self.robust_std(image)
        median = np.median(image)
        goodpix = image>median+stddev*level
        return goodpix

#-------------------------------------------------------------------------------
    def centroid_brightest_blob(self,thresholded_image):
    
        contours,hierarchy = cv2.findContours(thresholded_image,
                                              cv2.RETR_LIST,
                                              cv2.CHAIN_APPROX_SIMPLE)
        if len(contours)==0:
            return (-1, -1) 
    
        else:
            max_area = 0
            for cnt in contours:
                area = cv2.contourArea(cnt)
                if area > max_area:
                    max_area = area
                    best_cnt = cnt

            # finding centroids of best_cnt and draw a circle there
            M = cv2.moments(best_cnt)
            cx,cy = (M['m10']/M['m00']), (M['m01']/M['m00'])
            return (cx, cy)

#-------------------------------------------------------------------------------
    def centroid_all_blobs(self,thresholded_image, areacutoff=30):
        thresholded_copy = thresholded_image.copy()
        contours,hierarchy = cv2.findContours(thresholded_copy,
                                              cv2.RETR_LIST,
                                              cv2.CHAIN_APPROX_SIMPLE)
        if len(contours)==0:
            return []#np.zeros((1, 3)) 
    
        else:
            outarray = np.zeros((1, 3))
            counter = 0
            for cnt in contours:
                M = cv2.moments(cnt)
                if M['m00']<areacutoff:
                    counter = counter+1
                    continue
                cx,cy,ssum = (M['m10']/M['m00']), (M['m01']/M['m00']), M['m00']
                outarray=np.vstack((outarray, np.array((cx, cy, ssum))))
        return outarray[1:,:]

    
#-------------------------------------------------------------------------------
    def get_stars(self,image):
        '''
        Find the centroids of all star-like objects above threshold and return
        as list
        '''
        d = np.array(image, dtype='float')
        th = self.threshold_pyguide(d, level = 4)
        imtofeed = np.array(np.round((d*th)/np.max(d*th)*255), dtype='uint8')
        if np.max(th) == False:
            return ([],imtofeed) # nothing above the threshhold
        stars = self.centroid_all_blobs(imtofeed,areacutoff = 160)
        
        return(stars,imtofeed)


################################################################################
