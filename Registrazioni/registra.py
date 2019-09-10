# import the necessary packages
import numpy as np
import cv2
import sys, os, time, traceback
from datetime import datetime
from time import gmtime, strftime
from scipy.spatial import distance as dist
from ClassIPCamCaptureMotion import IPCamVideoAlarm_CaptureMotion
from ClassIPCamRec import IPCamVideoAlarm_Recording

class IPCamVideoAlarm():
    def __init__(self, sorgente, cartella, nome_cam, ruota=0.0, scala=1, mog2=False):

        self.font = cv2.FONT_HERSHEY_SIMPLEX
        self.scale_percent = scala

        # Inizializzazione variabili
        self.nome_cam = nome_cam # nome della IPCam o del file
        self.ruota = ruota

        # Inizializzazioni variabili per i processi
        self.curframe = None
        
        # MASK
        self.asMask = False
        self.maskframe = None

        # ROI
        self.asROI = False
        self.ROI = None

        # FINESTRE
        self.initWindows()

        #PROCESSO DI CAPTURE 
        self.fvs = cv2.VideoCapture(sorgente)
        time.sleep(1.0)

        width = int(self.fvs.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(self.fvs.get(cv2.CAP_PROP_FRAME_HEIGHT ))
        self.fps = self.fvs.get(cv2.CAP_PROP_FPS)
        print(("Video STREAMING FPS:%s (w:%s h:%s)")%(self.fps, width, height))

        self.processImage()
        self.height, self.width = self.curframe.shape[:2]
        print(("Frame format(w:%s h:%s)")%(self.height, self.width))

        # MOTION                
        #self.crop_height = 160
        #self.crop_width = 160 
        self.motion = IPCamVideoAlarm_CaptureMotion(self.height, self.width, 160, 160)

        # RECORDING
        self.recording = IPCamVideoAlarm_Recording(cartella, nome_cam, self.fps, self.width, self.height)

    def defineMask(self, maskfile):
        try:
            self.asMask = True
            self.is_mask_window = True
            mask = cv2.imread (maskfile)
            mask = cv2.cvtColor (mask, cv2.COLOR_BGR2GRAY)
            retm, self.mask = cv2.threshold (mask, 10, 255, cv2.THRESH_BINARY)
        except:
            self.asMask = False
            self.is_mask_window = False
            tb = sys.exc_info()[2]
            tbinfo = traceback.format_tb(tb)[0]
            pymsg = "PYTHON ERRORS:\nTraceback Info:\n" + tbinfo + "\nError Info:\n    " + \
                str(sys.exc_type)+ ": " + str(sys.exc_value) + "\n"
            print(pymsg)
        
    def defineRoi(self, alphafile):
        try:
            self.asROI = True
            self.ROI = cv2.imread (alphafile)
        except:
            self.asROI = False
            tb = sys.exc_info()[2]
            tbinfo = traceback.format_tb(tb)[0]
            pymsg = "PYTHON ERRORS:\nTraceback Info:\n" + tbinfo + "\nError Info:\n    " + \
                str(sys.exc_type)+ ": " + str(sys.exc_value) + "\n"
            print(pymsg)
  
    def addMask(self, frame):
        try:
            if self.asMask:
                self.maskframe = cv2.bitwise_and (frame, frame, mask = self.mask)
            else:
                self.maskframe = frame
            return True
        except:
            print ("[ERROR ] addMask")
            return False

    def addRoi(self, frame):
        k = cv2.addWeighted(frame, 0.8, self.ROI, 0.2, -1)
        return k

    def resizeImage_percent(self, image):#parameter angel in degrees
        #t0 = time.time()
        width = int(image.shape[1] * self.scale_percent)
        height = int(image.shape[0] * self.scale_percent)
        dim = (width, height)
        # resize image
        resized = cv2.resize(image, dim, interpolation = cv2.INTER_AREA)
        #print ("    Tempo resize %s", time.time()-t0)
        return resized
           
    def rotateImage(self, image):#parameter angel in degrees
        #t0 = time.time()
        t = cv2.transpose(image)
        k = cv2.flip(t, 0)
        #print ("    Tempo rotazione %s", time.time()-t0)
        return k
            
    def processImage(self):
        try:
            ret, self.curframe = self.fvs.read()
            if ret:
                t0 = time.time()
                if self.scale_percent < 1:
                    self.curframe = self.resizeImage_percent(self.curframe)
                if self.ruota == 90.0:
                        self.curframe = self.rotateImage(self.curframe)
                self.addMask(self.curframe)
                print ("    processImage: %.4f (sec)"%(time.time()-t0))
                return True
            return False

        except:
            print ("[ERROR ] Image process")
            tb = sys.exc_info()[2]
            tbinfo = traceback.format_tb(tb)[0]
            pymsg = "PYTHON ERRORS:\nTraceback Info:\n" + tbinfo + "\nError Info:\n    " + \
                str(sys.exc_type)+ ": " + str(sys.exc_value) + "\n"
            print(pymsg)
            return False

    def alarmMotionImage(self):
        try:
            self.motion.max_cnts = 50
            self.motion.min_area_moviment = 500
            self.motion.max_area_moviment = 15000
            alarm = False
            if self.processImage():
                alarm = self.motion.eventMotion(self.maskframe)
                self.maskframe = self.motion.movimentframe
            return alarm
        except:
            print ("[ERROR] Alarm Image")
            tb = sys.exc_info()[2]
            tbinfo = traceback.format_tb(tb)[0]
            pymsg = "PYTHON ERRORS:\nTraceback Info:\n" + tbinfo + "\nError Info:\n    " + \
                str(sys.exc_type)+ ": " + str(sys.exc_value) + "\n"
            print(pymsg)
            return False


    # INIZIO FINESTRE
    def initWindows(self):
        # inizializzo finestra Live
        self.is_live_window = True
        self.live_window = self.nome_cam + " Live"
        
        # inizializzo finestra Mask
        self.is_mask_window = False
        self.mask_window = self.nome_cam + " Mask"
        
        # inizializzo finestra Roi
        self.is_roi_window = False
        self.roi_window = self.nome_cam + " ROI"

        # inizializzo finestra thresh
        self.is_thresh_window = False
        self.thresh_window = self.nome_cam + " Thresh"

        # inizializzo finestra motion
        self.is_motion_window = False
        self.motion_window = self.nome_cam + " Motion"

        #nome finistra LIve
        cv2.namedWindow(self.live_window)

    def initWindowMask(self):
        # print self.is_mask_window, self.asMask
        if not self.is_mask_window and self.asMask:
            cv2.namedWindow(self.mask_window)
            self.is_mask_window = True
            
            
    def closeWindowMask(self):
        if self.is_mask_window:
            cv2.destroyWindow(self.mask_window)
            self.is_mask_window = False      

    def initWindowROI(self):
        if not self.is_roi_window and self.asROI:
            cv2.namedWindow(self.roi_window)
            self.is_roi_window = True
            
    def closeWindowROI(self):
        if self.is_roi_window:
            cv2.destroyWindow(self.roi_window)
            self.is_roi_window = False 

    def initWindowThresh(self):
        if not self.is_thresh_window:
            cv2.namedWindow(self.thresh_window)
            self.is_thresh_window = True
            
    def closeWindowThresh(self):
        if self.is_thresh_window:
            cv2.destroyWindow(self.thresh_window)
            self.is_thresh_window = False 

    def initWindowMotion(self):
        if not self.is_motion_window:
            cv2.namedWindow(self.motion_window)
            self.is_motion_window = False
            
    def closeWindowMotion(self):
        if self.is_motion_window:
            cv2.destroyWindow(self.motion_window)
            self.is_motion_window = False 

   
    def showLiveImage(self): #mostra tutte le finestre attive
        try:            
            self.showMaskImage()
            self.showRoiImage()
            self.showThreshImageARM()
            self.showMotionImage()
            #self.showThreshImageMOG2()
            cv2.imshow(self.live_window, self.curframe) #viene mostrato per ultimo per ottenere le eventuali visualizzazioni
        except:
            print ("[ERROR] imshow error")

    def showMaskImage(self):
        try:
            if self.is_mask_window:                
                cv2.imshow(self.mask_window, self.maskframe)
        except:
            print ("[ERROR] MASK imshow error")

    def showRoiImage(self):
        try:
            if self.is_roi_window:
                m = self.addRoi(self.curframe)
                cv2.imshow(self.roi_window, m)
        except:
            print ("[ERROR] ROI imshow error")

    def showThreshImageARM(self):
        try:
            if self.is_thresh_window:
                cv2.imshow(self.thresh_window, self.thresh)
        except:
            print ("[ERROR] Thresh error")
            tb = sys.exc_info()[2]
            tbinfo = traceback.format_tb(tb)[0]
            pymsg = "PYTHON ERRORS:\nTraceback Info:\n" + tbinfo + "\nError Info:\n    " + \
                str(sys.exc_type)+ ": " + str(sys.exc_value) + "\n"
            print(pymsg)
            
    def showThreshImageMOG2(self):
        try:
            if self.is_thresh_window:
                m = self.movimentMOG2(self.curframe)
                n = self.somethingHasMoved(m)
                cv2.imshow(self.thresh_window, n)
        except:
            print ("[ERROR] Thresh error")
            tb = sys.exc_info()[2]
            tbinfo = traceback.format_tb(tb)[0]
            pymsg = "PYTHON ERRORS:\nTraceback Info:\n" + tbinfo + "\nError Info:\n    " + \
                str(sys.exc_type)+ ": " + str(sys.exc_value) + "\n"
            print(pymsg)
            
    def showMotionImage(self):
        try:
            if self.is_motion_window:
                cv2.imshow(self.motion_window, self.motion)
        except:
            print ("[ERROR] Thresh error")
            tb = sys.exc_info()[2]
            tbinfo = traceback.format_tb(tb)[0]
            pymsg = "PYTHON ERRORS:\nTraceback Info:\n" + tbinfo + "\nError Info:\n    " + \
                str(sys.exc_type)+ ": " + str(sys.exc_value) + "\n"
            print(pymsg)
    #FINE FINESTRE

    def alarm_masqueradeImage(self):
        num_colored_pixel = cv2.countNonZero(self.frame2gray)
        if num_colored_pixel == 0:
            print ("Image is black %s"%num_colored_pixel)
            return True
        return False

    def alarm_defocus(self):
        self.num_defocus = cv2.Laplacian(self.curframe, cv2.CV_64F).var()
        if self.num_defocus > 2000:            
            self.sensitivity_value = 37
            #print ("Defocus %s %s"%(num_defocus, self.sensitivity_value))
        elif self.num_defocus > 1000 and num_defocus <= 2000:
            self.sensitivity_value = 33
            #print ("Defocus %s %s"%(num_defocus, self.sensitivity_value))
        elif self.num_defocus > 800 and num_defocus <= 1000:
            self.sensitivity_value = 29
            #print ("Defocus %s %s"%(num_defocus, self.sensitivity_value))
        elif self.num_defocus > 400 and num_defocus <= 800:
            self.sensitivity_value = 25
            #print ("Defocus %s %s"%(num_defocus, self.sensitivity_value))          
        elif self.num_defocus > 100 and num_defocus <= 400:
            self.sensitivity_value = 21
            #print ("Defocus %s %s"%(num_defocus, self.sensitivity_value))
        else:
            print ("Defocus %s"%self.num_defocus)
            return True
        
        return False

    def close(self):
        self.recording.close()
           
        # distruggo le viste
        cv2.destroyAllWindows()
  

if __name__=="__main__":
    os.environ['TZ'] = 'Europe/Rome'

    # load our serialized model from disk
    print("[INFO] Start Video Recording...")

 
    video_path = 'rtsp://admin:luca2006@vignale.duckdns.org/Streaming/Channels/102'
    video_store = "/home/odroid/cv4/Registrazioni/"

    cam1 = IPCamVideoAlarm(video_path, video_store, "Cam1", 90.0, 1, False)
    # cam1.defineMask("C:\\opencv\\RaspyPerson\\MascheraCam1.jpg")
    # cam1.defineRoi("C:\\opencv\\RaspyPerson\\RoiCam1.jpg")
   
    t0 = time.time()
    ora = strftime("%d-%b-%Y-%H-%M-%S", time.localtime())
    print("[INFO] Starting video alarm process...%s"%ora) 
    while True:
            startt0 = time.time()
            if cam1.alarmMotionImage():
                cam1.recording.time_recording = 5 #default 2 sec
                cam1.recording.trigger_time = time.time()
                cam1.recording.isRecording = True                
                print ("Allarme!!!!!!!!!!!!!!!")
            cam1.recording.Video(cam1.curframe)
            cam1.showLiveImage()
            # print ("        Elaborazione in %s ",   time.time()- startt0)

            
               
            key = cv2.waitKey(3) & 0xFF
            if key == 27 or key == ord('q'):
                cam1.close()
                exit(0)
            elif key == ord('r'):
                if not cam1.recording.isRecording:
                    print ("Start recording")
                    cam1.recording.asRecording = True
                    cam1.recording.isRecording = True
                    cam1.recording.time_recording = 10 #default 2 sec
                    cam1.recording.trigger_time = time.time()
                else:
                    cam1.recording.time_recording = 2 #default 2 sec
                    cam1.recording.isRecording = False
                    print ("Stop recording")             
            elif key == ord('s'):
                if not sleep:
                    sleep = True
                else:
                    sleep = False
            elif key == ord('m'): # gestione finestre con maschera
                if not cam1.is_mask_window:
                     cam1.initWindowMask()
                else:
                    cam1.closeWindowMask()                  
            elif key == ord('o'): # gestione finestre con ROI
                if not cam1.is_roi_window:
                    cam1.initWindowROI()
                else:
                    cam1.closeWindowROI()
            elif key == ord('a'): # gestione finestre con MOTION semplificato ARM
                if not cam1.is_thresh_window:
                    cam1.initWindowThresh()
                else:
                    cam1.closeWindowThresh()
            elif key == ord('f'): # salvo foto del frame corrente
                cam1.recording.Foto(cam1.maskframe)
                print ("Frame salvato")
                
            elif key == ord('z'): # salvo foto del frame corrente
                cam1.sensitivity_value += 5
                # print cam1.sensitivity_value

            elif key == ord('x'): # salvo foto del frame corrente
                cam1.sensitivity_value -= 5
                # print cam1.sensitivity_value
            elif key == ord('i'): # salvo foto del frame corrente
                # print cam1.sensitivity_value, cam1.num_defocus
                pass
