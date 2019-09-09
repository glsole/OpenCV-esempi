#!/usr/bin/python
# -*- coding: utf-8 -*-


# import the necessary packages
import numpy as np
import cv2
import sys, os, time
from datetime import datetime
from time import gmtime, strftime
from multiprocessing import Process
from multiprocessing import Queue



classNames = { 0: 'background',
   1: 'person', 2: 'bicycle', 3: 'car', 4: 'motorcycle', 5: 'airplane', 6: 'bus',
   7: 'train', 8: 'truck', 9: 'boat', 10: 'traffic light', 11: 'fire hydrant',
   13: 'stop sign', 14: 'parking meter', 15: 'bench', 16: 'bird', 17: 'cat',
   18: 'dog', 19: 'horse', 20: 'sheep', 21: 'cow', 22: 'elephant', 23: 'bear',
   24: 'zebra', 25: 'giraffe', 27: 'backpack', 28: 'umbrella', 31: 'handbag',
   32: 'tie', 33: 'suitcase', 34: 'frisbee', 35: 'skis', 36: 'snowboard',
   37: 'sports ball', 38: 'kite', 39: 'baseball bat', 40: 'baseball glove',
   41: 'skateboard', 42: 'surfboard', 43: 'tennis racket', 44: 'bottle',
   46: 'wine glass', 47: 'cup', 48: 'fork', 49: 'knife', 50: 'spoon',
   51: 'bowl', 52: 'banana', 53: 'apple', 54: 'sandwich', 55: 'orange',
   56: 'broccoli', 57: 'carrot', 58: 'hot dog', 59: 'pizza', 60: 'donut',
   61: 'cake', 62: 'chair', 63: 'couch', 64: 'potted plant', 65: 'bed',
   67: 'dining table', 70: 'toilet', 72: 'tv', 73: 'laptop', 74: 'mouse',
   75: 'remote', 76: 'keyboard', 77: 'cell phone', 78: 'microwave', 79: 'oven',
   80: 'toaster', 81: 'sink', 82: 'refrigerator', 84: 'book', 85: 'clock',
86: 'vase', 87: 'scissors', 88: 'teddy bear', 89: 'hair drier', 90: 'toothbrush' }
COLORS = np.random.uniform(0, 255, size=(len(classNames), 3))




class IPCamVideoAlarm():

    def __init__(self, nome, sorgente, maschera):
        self.nome = nome
        self.source = sorgente
        self.writer = None
        self.doRecord=True
        self.isRecording = False
        
        self.recordfile = ''
        self.recordw = 160
        self.recordh = 160
        maschera = cv2.imread (maschera)
        self.mascheragray = cv2.cvtColor (maschera, cv2.COLOR_BGR2GRAY)
        retm, self.mask = cv2.threshold (self.mascheragray, 10, 255, cv2.THRESH_BINARY)

        # Initialize frame rate calculation
        self.frame_rate_calc = 1
        self.freq = cv2.getTickFrequency()
        self.font = cv2.FONT_HERSHEY_SIMPLEX
        
        self.saveframe = False

        self.showWindows = True
        self.font = None
        
        self.blur = 3
        self.sensitivity_value = 35
        self.min_area = 1200
        self.kernelOp = np.ones((2,2),np.uint8)
        self.kernelCl = np.ones((21,21),np.uint8)

        self.curframe = None   
        self.frame1gray = None
        self.frame1gray = None

        self.thresh = None
        self.cropframe = None
        self.min_confidence = 0.75
        
        
        print("[INFO] starting video file thread...")
        self.fvs = cv2.VideoCapture(self.source)
        time.sleep(1.0)

        self.time_last_frame_read = time.time()
        self.last_frame_read = self.fvs.get(cv2.CAP_PROP_POS_FRAMES )
        fps = self.fvs.get(cv2.CAP_PROP_FPS)
        #print "Frames per second using video.get(cv2.CAP_PROP_FPS) : {0}".format(fps)

        self.curframe = self.readImageFrame()
        self.frame1gray  = cv2.cvtColor (self.curframe, cv2.COLOR_BGR2GRAY)
        self.frame1gray = cv2.bitwise_and (self.frame1gray, self.frame1gray, mask = self.mask)
        self.frame2gray = self.frame1gray
        
        self.height, self.width = self.frame1gray.shape[:2]
        self.boxcrop_p1 = (int(self.width/2 - self.recordw/2), int(self.height/2 - self.recordh/2))
        self.boxcrop_p2 = (int(self.width/2 + self.recordw/2), int(self.height/2 + self.recordh/2))

        # REGISTRAZIONE
        #inizializzo la dimensione del video
        self.record_height, self.record_width = self.curframe.shape[:2]
        
        #inizializzo le variabili, le cartelle (video e foto) e il file di registrazione video
        cartella = "/home/pi/Public/"
        self.initRecorder(cartella)
            
        if self.showWindows:
            cv2.namedWindow("Image")
            
    def readImageFrame(self):
        # loop over frames from the video file stream
        self.tc1 = cv2.getTickCount()
        delta_ultima_lettura = time.time()-self.time_last_frame_read
        

        nframe_da_recuperare = int(delta_ultima_lettura/0.1 -2.0)
        for i in range(0, nframe_da_recuperare):
            ret, frame = self.fvs.read()
            self.last_frame_read = self.fvs.get(cv2.CAP_PROP_POS_FRAMES )
            #print (('tempo trascorso dall ultima lettura %s ultimo letto %s  recupero %s')%(delta_ultima_lettura, self.last_frame_read, nframe_da_recuperare))
            
        
        ret, frame = self.fvs.read()
        self.time_last_frame_read = time.time()
        self.last_frame_read = self.fvs.get(cv2.CAP_PROP_POS_FRAMES )
        return frame
    
    def processImage(self):
        self.curframe = self.readImageFrame()
        self.frame2gray  = cv2.cvtColor (self.curframe, cv2.COLOR_BGR2GRAY)
        self.frame2gray = cv2.bitwise_and (self.frame2gray, self.frame2gray, mask = self.mask)

        
        #Absdiff to get the difference between to the frames
        frameDiff = cv2.absdiff(self.frame1gray, self.frame2gray)
        self.thresh = cv2.threshold(frameDiff, self.sensitivity_value, 255, cv2.THRESH_BINARY)[1]
        self.thresh = cv2.dilate(self.thresh, None, iterations=2)
        self.frame1gray = self.frame2gray
        

    def somethingHasMoved(self):
        motion = False
        _, cnts, _ = cv2.findContours(self.thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)        
        if len(cnts) > 0:            
            x1=0            
            y1=0            
            x2=0
            y2=0
            #print ("Oggetti in movimento: %s"%len(cnts))
            for c in cnts:
                (x, y, w, h) = cv2.boundingRect(c)
                # calcolo il rettangolo di massimo ingombro
                aarea = w*h
                if aarea > self.min_area:
                    #print ("    rettangolo motion: %s %s"%(w, h))
                    if x < x1 or x1 == 0:
                        x1 = x
                    if x+w > x2 or x2 == 0:
                        x2 = x+w
                    if y < y1 or y1 == 0:
                        y1 = y
                    if y+h > y2 or y2 == 0:
                        y2 = y+h
            #ricalcolo area        
            aarea = (x2-x1)*(y2-y1)
            center = (int((x2+x1)/2),  int((y2+y1)/2))           
            if aarea > self.min_area:
                #print (aarea)
                self.cropBox(center)
                motion = True
                self.trigger_time = time.time()                   
                    
        cv2.rectangle(self.curframe, self.boxcrop_p1, self.boxcrop_p2, (0, 255, 0), 2)
        return motion
    
    def cropBox(self, center):
        cx = center[0]
        cy = center[1]

        if cx - self.recordw/2 < 0:
            cx = int(self.recordw/2)
            
        if cx + self.recordw/2 > self.width:
            cx = int(self.width - self.recordw/2)
            
        if cy - self.recordh/2 < 0:
            cy = int(self.recordh/2)
            
        if cy + self.recordh/2 > self.height:
            cy = int(self.height - self.recordh/2)

        self.boxcrop_p1 = (cx - int(self.recordw/2), cy - int(self.recordh/2))
        self.boxcrop_p2 = (cx + int(self.recordw/2), cy + int(self.recordh/2))

    def cropFrameMotion(self):
        cropframe = self.curframe[self.boxcrop_p1[1]:self.boxcrop_p2[1], self.boxcrop_p1[0]:self.boxcrop_p2[0]]
        M = cv2.getRotationMatrix2D((self.recordh/2,self.recordw/2),90,1)
        self.cropframe = cv2.warpAffine(cropframe, M, (self.recordw,self.recordh))
        return self.cropframe

    # INIZIO REGISTRAZIONE
    def initRecorder(self, cartella): #Create the recorder
        try:
            # inizializzo a False il flag della registrazione video
            # sara' inizializzato a True quabdo voglio registrare
            self.isRecording = False
            
            # inizializzo asRecording a False
            # sara marcato a True se faccio registrazione su questo video se rimane False al termine il file verra' cancellato
            self.asRecording = False
            
            # inizializzo savedframe a False
            # sara marcato a True quando verra' salvato un frame in una foto
            # serve per non salvare costantemente le foto
            self.savedframe = False

            #cartella di archiviazione
            self.recording_dir = cartella

            # verifico se esite la cartella principale di registazione esiste, altrimento la creo
            if not os.path.exists(self.recording_dir):
                os.mkdir(self.recording_dir)
                
            # verifico se esite la cartella di registazione video, altrimento la creo
            self.recording_video_dir = os.path.join(self.recording_dir, "video")
            if not os.path.exists(self.recording_video_dir):
                os.mkdir(self.recording_video_dir)
                
            # verifico se esite la cartella di registazione foto, altrimento la creo    
            self.recording_foto_dir = os.path.join(self.recording_dir, "foto")
            if not os.path.exists(self.recording_foto_dir):
                os.mkdir(self.recording_foto_dir) 

            #inizializzo la registrazione video        
            nome_recorded_file = self.nome_cam + '_%s.avi'%strftime("%d-%b-%Y-%H-%M-%S", time.localtime())
            self.recording_video_file = os.path.join(self.recording_video_dir, nome_recorded_file)           

            # Define the codec and create VideoWriter object
            fourcc = cv2.VideoWriter_fourcc(*'XVID')
            self.record_fps = 10.0
            self.writer = cv2.VideoWriter(self.recording_video_file, fourcc, self.record_fps, (self.record_width,self.record_height))
            print(("File %s per la registrazione video FPS:%s (w:%s h:%s)")%(self.recording_video_file, self.record_fps, self.record_width,self.record_height))
        except:
            tb = sys.exc_info()[2]
            tbinfo = traceback.format_tb(tb)[0]
            pymsg = "PYTHON ERRORS:\nTraceback Info:\n" + tbinfo + "\nError Info:\n    " + \
                str(sys.exc_type)+ ": " + str(sys.exc_value) + "\n"
            print(pymsg)
            print(("Errore inizializzazione file %s per la registrazione video")%(self.recording_video_file))
            return False

    def recording(self):
        try:
            if self.isRecording:
                self.writer.write(self.curframe) #Write the frame
                self.asRecording = True
        except:
            print(("Errore registrazione frame %s")%(self.recording_video_file))
            
    def recordingframe(self):
        try:
            if not self.savedframe:
                nome_recorded_file = self.nome_cam + '_%s.jpg'%strftime("%d-%b-%Y-%H-%M-%S", time.localtime())
                nameframe = os.path.join(self.recording_foto_dir, nome_recorded_file)
                cv2.imwrite(nameframe, self.curframe)     # save frame as JPEG file
                print(("Salvato frame in %s")%(nameframe))
                self.savedframe = True
        except:
            print(("Errore registrazione frame %s")%(self.recording_video_file))
    # FINE REGISTRAZIONE

      
    def alarm_masqueradeImage(self):
        num_colored_pixel = cv2.countNonZero(self.frame2gray)
        if num_colored_pixel == 0:
            print ("Image is black %s"%num_colored_pixel)
            return True
        return False

    def alarm_defocus(self):
        num_defocus = cv2.Laplacian(self.curframe, cv2.CV_64F).var()        
        if num_defocus < 100:
            print ("Defocus %s"%num_defocus)
            return True
        return False

    def showImage(self):
        try:    
            self.tc2 = cv2.getTickCount()
            time1 = (self.tc2-self.tc1)/self.freq
            self.frame_rate_calc = 1/time1
            # show the output frame
            #cv2.putText(self.curframe,"FPS: {0:.2f}".format(self.frame_rate_calc),(10,350),self.font,1,(255,255,0),2,cv2.LINE_AA)
            cv2.imshow("Image", self.curframe)
        except:
            print ("[ERROR] imshow error")

    def showImageThresh(self):
        try:
            # display the size of the queue on the frame            
            cv2.imshow("Image", self.frame2gray)
        except:
            print ("[ERROR] imshow error")
  

    def close(self):
        # chiudo la registrazione
        self.writer.release()
        
        # se non ho registrato cancello il file
        if self.asRecording == False:
            print(("Cancello file %s per la registrazione video, non ci sono frame registrati")%(self.recording_video_file))
            os.remove(self.recording_video_file)
            
        # distruggo le viste
        cv2.destroyAllWindows()




def detectionTensor(frame):
    persona = False

    # initialize the list of class labels MobileNet SSD was trained to
    # detect, then generate a set of bounding box colors for each claSS
    start_time = time.time()
    inWidth = 224
    inHeight = 224
    WHRatio = inWidth / float(inHeight)
    inScaleFactor =  1 / ((103.94 + 116.78 + 123.68)/3) #0.00784
    meanVal = 127.5
    swapRB = True

    blob = cv2.dnn.blobFromImage(frame, inScaleFactor, (inWidth, inHeight), (103.94, 116.78, 123.68), swapRB)
    net.setInput(blob)
    detections = net.forward()


    if detections is not None:
                # loop over the detections
                #print("# loop over the detections")
                for i in np.arange(0, detections.shape[2]):                    
                    # extract the confidence (i.e., probability) associated with
                    # the prediction
                    confidence = detections[0, 0, i, 2]
                    idx = int(detections[0, 0, i, 1])
                    #time_detection = time.time()
                    #label = "   [INFO] {}: {:.2f} tempo di rilevamento{:.2f}".format(classNames[idx],confidence * 100, time_detection-start_time)
                    #print (label)
                    #print('# confidence %s'%confidence)
                    # filter out weak detections by ensuring the `confidence` is
                    # greater than the minimum confidence
                    if confidence > 0.75:                        
                        # extract the index of the class label from the
                        # `detections`, then compute the (x, y)-coordinates of
                        # the bounding box for the object                
                        
                        #box = detections[0, 0, i, 3:7] * np.array([width, height, width, height])
                        #(x1, y1, x2, y2) = box.astype("int")
                        #cv2.rectangle(frame, (x1, y1), (x2, y2), (0,255,0), 2)
                        ora = strftime("%d-%b-%Y-%H-%M-%S", time.localtime())
                        label = "[ALARM] {}: {:.2f} alle{}:%".format(classNames[idx],confidence * 100, ora)
                        print (label)
                        if idx == 1:
                            persona =  True
    label = "[INFO] DNN tempo di rilevamento: {:.2f}".format(time.time()-start_time)
    #print (label)
    return persona

           

if __name__=="__main__":
    os.environ['TZ'] = 'Europe/Rome'
    t = time.time() # tempo apertura programma

    # load our serialized model from disk
    print("[INFO] loading model...")
    pb_path = "C:/opencv/RaspyPerson/models/frozen_inference_graph.pb"
    pbtxt_path = "C:/opencv/RaspyPerson/models/ssdlite_mobilenet_v2_coco.pbtxt"
    #video_path = "/home/pi/Public/Cam2_20-Jun-2018-13-16-47.avi"
    video_path = "rtsp://admin:luca2006@87.16.45.252/Streaming/Channels/102"
    
    # inizializzo il lettore Net DNN di tensorFlow
    net = cv2.dnn.readNetFromTensorflow(pb_path, pbtxt_path)


    stato = 'ON'
    while True:
        if stato == 'ON':
            t0 = time.time()
            ora = strftime("%d-%b-%Y-%H-%M-%S", time.localtime())
            cam1 = IPCamVideoAlarm("Cam1", video_path, "C:/opencv/RaspyPerson/MascheraCam1.jpg")
            print("[INFO] Starting video alarm process...%s"%ora) 
            last_alarm = 0
            while True:
                startt0 = time.time()
                cam1.processImage()
                #cam1.alarm_masqueradeImage()
                #cam1.alarm_defocus()
                
                if cam1.somethingHasMoved():                    
                    framemotion = cam1.cropFrameMotion()
                    #cam1.isRecording = True #flag per la registrazione                   
                     
                       
                cam1.showImage()
                #cam1.recording() #registro se c'e un movimento per 10 sec

        

                key = cv2.waitKey(3) & 0xFF
                if key == 27 or key == ord('q'):
                    cam1.close()
                    exit(0)

        else:
            time.sleep(5)
            started = False            
            stato = oh.get_status("itm_alarm_arm_perimetrale")






