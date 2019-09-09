# import the necessary packages
import numpy as np
import cv2
import sys, os, time, traceback
from datetime import datetime
from time import gmtime, strftime
from scipy.spatial import distance as dist



class IPCamVideoAlarm_CaptureMotion():
    def __init__(self, frame_height, frame_width, crop_height, crop_width):
        
        self.font = cv2.FONT_HERSHEY_SIMPLEX

        # Inizializzazioni variabili per i processi
        self.movimentframe = None
        
        # MOTION
        """self.mog2 = mog2  # false uso la procedura semplificata ARm, True uso il rilevamento MOG2
        if self.mog2:
            self.initCheckMovimentMOG2()"""
        
        self.asMotion = False  #ho un movimento    
        self.asThresh = False #ho un frame di movimento
        self.thresh = None #frame del movimento o MOG o standard
        self.frame1gray = self.frame2gray = None #frame del movimento in grigio utilizzati per la differenza

        self.max_cnts = 50 #numero massimo di countours per frame
        self.min_area_moviment = 100 #area minima degli oggetti
        self.max_area_moviment = 15000 #area massima degli oggetti        
        self.sensitivity_value = 47 #sensibilita' notte 23-25 giorno 37-43
        self.kernelOp = np.ones((1,1),np.uint8)
        self.kernelCl = np.ones((17,17),np.uint8)
        #FRAME
        self.height = frame_height
        self.width = frame_width

        #CROP FRAME
        self.crop_height = crop_height
        self.crop_width = crop_width
        
        self.centro_pesato = (int(self.width/2), int(self.height/2))
        
            
    """def initCheckMovimentMOG2(self):
        self.fgbg = cv2.createBackgroundSubtractorMOG2(detectShadows = True) #Create the background substractor
        self.kernelOp = np.ones((3,3),np.uint8)
        self.kernelCl = np.ones((11,11),np.uint8)
        print ("Inizializzo MOG2") """     

           
    def cropBox(self, cx, cy):
        if cx - self.crop_width/2 < 0:
            cx = int(self.crop_width/2)
            
        if cx + self.crop_width/2 > self.width:
            cx = int(self.width - self.crop_width/2)
            
        if cy - self.crop_height/2 < 0:
            cy = int(self.crop_height/2)
            
        if cy + self.crop_height/2 > self.height:
            cy = int(self.height - self.crop_height/2)
            
        return (cx - int(self.crop_width/2), cy - int(self.crop_height/2)),(cx + int(self.crop_width/2), cy + int(self.crop_height/2))
        
    def detectHasMoved(self, frame):
        self.asMotion = False
        self.movimentframe = frame
        try:
            _, cnts, _ = cv2.findContours(self.thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if len(cnts) > 0 and len(cnts) < self.max_cnts:
                points_center = []
                area_weights = []
                cmax = max(cnts, key = cv2.contourArea)
                cmin = min(cnts, key = cv2.contourArea)
                max_contour_area = cv2.contourArea(cmax)
                min_contour_area = cv2.contourArea(cmin)

                # stampo l'area rilevata                    
                print ('Contour Area %s Min :%.2f  Max :%.2f')%(len(cnts), min_contour_area, max_contour_area)

                for c in cnts:
                    area = cv2.contourArea(c)                
                    if area >= self.min_area_moviment and area <= self.max_area_moviment:
                        
                        # disegno il contorno
                        cv2.drawContours(self.movimentframe, c, -1, (0,0,200), 3, 8)

                        M = cv2.moments(c)
                       
                        center = (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))
                        points_center.append(center)
                        area_weights.append(area)
                        # il cerchio viene generato sul frame corrente per essere visto
                        cv2.circle(self.movimentframe, center, 5, (0,0,255), -1)
                        cv2.putText(self.movimentframe, "Area: {0:.2f}".format(max_contour_area),(center[0]+10,center[1]),self.font,0.5,(0,0,255), 2)
                #calcolo il centro di massa di tutti i countors che sono dentro al range
                if len(points_center) > 0:
                    print ('Contour Area %s Min :%.2f  Max :%.2f')%(len(cnts), min_contour_area, max_contour_area)
                    p = np.array(points_center)                    
                    # media
                    #c = tuple(np.mean(p, axis=0, dtype=np.int))
                    #media pesata con l'area
                    a = np.average(p, axis=0, weights=area_weights)
                    self.centro_pesato = (int(a[0]), int(a[1]))
                                       


                    # marco il momento del movimento
                    self.trigger_time = time.time()                    
                    self.asMotion = True
                    
            cv2.circle(self.movimentframe, self.centro_pesato, 10, (0,255, 0), -1)                    
            ll, ur = self.cropBox(self.centro_pesato[0], self.centro_pesato[1])            
            cv2.rectangle(self.movimentframe, ll, ur, (0, 255, 0), 2)
                           
            return self.asMotion    
        except:
            print ("[ERROR] somethingHasMoved error")
            tb = sys.exc_info()[2]
            tbinfo = traceback.format_tb(tb)[0]
            pymsg = "PYTHON ERRORS:\nTraceback Info:\n" + tbinfo + "\nError Info:\n    " + \
                str(sys.exc_type)+ ": " + str(sys.exc_value) + "\n"
            print(pymsg)
            return self.asMotion
       
    def movimentARM(self, frame):
        try:                    
            self.frame2gray  = cv2.cvtColor (frame, cv2.COLOR_BGR2GRAY) #current gray frame
            if self.frame1gray == None:
                self.frame1gray = self.frame2gray
                
            #Absdiff to get the difference between to the frames
            frameDiff = cv2.absdiff(self.frame1gray, self.frame2gray)
            imBin = cv2.threshold(frameDiff, self.sensitivity_value, 255, cv2.THRESH_BINARY)[1]
            mask = cv2.morphologyEx(imBin, cv2.MORPH_OPEN, self.kernelOp)
            self.thresh =  cv2.morphologyEx(mask , cv2.MORPH_CLOSE, self.kernelCl)
            
            #mask = cv2.dilate(imBin, None, iterations=2)
            self.frame1gray = self.frame2gray
            return True
        except:
            self.asThresh = False
            print ("[ERROR] movimentARM error")
            tb = sys.exc_info()[2]
            tbinfo = traceback.format_tb(tb)[0]
            pymsg = "PYTHON ERRORS:\nTraceback Info:\n" + tbinfo + "\nError Info:\n    " + \
                str(sys.exc_type)+ ": " + str(sys.exc_value) + "\n"
            print(pymsg)
            return False

    """def movimentMOG2(self, frame):
        self.asThresh = True
        fgmask = self.fgbg.apply(frame) #Use the substractor
        try:
            ret,imBin= cv2.threshold(fgmask,127,255,cv2.THRESH_BINARY)
            #Opening (erode->dilate) para quitar ruido.
            mask = cv2.morphologyEx(imBin, cv2.MORPH_OPEN, self.kernelOp)
            #Closing (dilate -> erode) para juntar regiones blancas.
            mask =  cv2.morphologyEx(mask , cv2.MORPH_CLOSE, self.kernelCl)
            return mask
        except:
            self.asThresh = False
            print('Errore movimentMOG2')"""
 
    def eventMotion(self, framein):
        try:
            if self.movimentARM(framein):            
                # i valori di anali posso essere commentati per permetterne al software di determinarli esternamente alla procedura
                #self.max_cnts = 50
                #self.min_area_moviment = 200
                #self.max_area_moviment = 15000            
                if self.detectHasMoved(framein):
                    return True
            return False
        except:
            print ("[ERROR] Thresh error")
            tb = sys.exc_info()[2]
            tbinfo = traceback.format_tb(tb)[0]
            pymsg = "PYTHON ERRORS:\nTraceback Info:\n" + tbinfo + "\nError Info:\n    " + \
                str(sys.exc_type)+ ": " + str(sys.exc_value) + "\n"
            print(pymsg)
            return False

