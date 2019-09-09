# import the necessary packages
import numpy as np
import cv2
import sys, os, time
from datetime import datetime
from time import gmtime, strftime
from collections import deque
#from classOpenhab2 import *


classNames = { 0: 'background',
   1: 'persona', 2: 'bicycle', 3: 'auto', 4: 'motorcycle', 5: 'airplane', 6: 'bus',
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


        self.linelength = 20
        self.pts = deque(maxlen=self.linelength)


        self.nome = nome
        self.source = sorgente
        self.writer = None
        self.doRecord=True
        self.isRecording = False
        self.recorddir = "C:\\opencv\\Video\\"
        self.recordfile = ''
        self.recordw = 150
        self.recordh = 150
        
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
        self.min_area = 400
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

        self.time_last_frame_read = self.time_last_frame_process = time.time()
        self.last_frame_read = self.fvs.get(cv2.CAP_PROP_POS_FRAMES )
        
        
        fps = self.fvs.get(cv2.CAP_PROP_FPS)
        print "Frames per second using video.get(cv2.CAP_PROP_FPS) : {0}".format(fps)

        self.curframe = self.readImageFrame()
        self.frame1gray  = cv2.cvtColor (self.curframe, cv2.COLOR_BGR2GRAY)
        #self.frame1gray = cv2.bitwise_and (self.frame1gray, self.frame1gray, mask = self.mask)
        self.frame2gray = self.frame1gray
        
        self.height, self.width = self.frame1gray.shape[:2]
        self.boxcrop_p1 = (int(self.width/2 - self.recordw/2), int(self.height/2 - self.recordh/2))
        self.boxcrop_p2 = (int(self.width/2 + self.recordw/2), int(self.height/2 + self.recordh/2))

        if self.doRecord:
            self.initRecorder()
            
        if self.showWindows:
            cv2.namedWindow("Image")
            
    def rotateImage(self, image):#parameter angel in degrees
        t = cv2.transpose(image)
        k = cv2.flip(t, 0)
        return k
    
    def info_logs(self, s, n=None, i = 'PROC'):
        i=0
        if i:
            ora = strftime("%d-%b-%Y-%H-%M-%S", time.localtime())
            questo_momento = time.time()        
            if i == 'START':
                self.time_last_frame_process = questo_momento
                self.time_last_frame_read = questo_momento
                print ('[START-INFO] %s : %s   delle: %s')%(s, n, ora)
            elif i == 'END':
                print ('[END-INFO]   %s : %s')%(s, (questo_momento - self.time_last_frame_read))
                print('')
                print('')
            else:
                print ('    - %s in %s')%(s, (questo_momento - self.time_last_frame_process))
            self.time_last_frame_process = questo_momento
                
    def readImageFrame(self):
        # loop over frames from the video file stream
        self.tc1 = cv2.getTickCount()
        ora = strftime("%d-%b-%Y-%H-%M-%S", time.localtime())
        self.last_frame_read = self.fvs.get(cv2.CAP_PROP_POS_FRAMES )
        self.info_logs(' Inizio lettura Frame', self.last_frame_read - 1, 'START')      
        ret, frame = self.fvs.read()
        self.last_frame_read = self.fvs.get(cv2.CAP_PROP_POS_FRAMES )
        self.info_logs('readImageFrame: ')
        #frame = self.rotateImage(frame)
        #frame = cv2.resize(frame, (224, 224))
        return frame
    
    def processImage(self):
        self.curframe = self.readImageFrame()
        self.frame2gray  = cv2.cvtColor (self.curframe, cv2.COLOR_BGR2GRAY)
        #self.frame2gray = cv2.bitwise_and (self.frame2gray, self.frame2gray, mask = self.mask)

        
        #Absdiff to get the difference between to the frames
        frameDiff = cv2.absdiff(self.frame1gray, self.frame2gray)
        self.thresh = cv2.threshold(frameDiff, self.sensitivity_value, 255, cv2.THRESH_BINARY)[1]
        self.thresh = cv2.dilate(self.thresh, None, iterations=2)
        self.frame1gray = self.frame2gray
        self.info_logs ('processImage :')


    def somethingHasMoved(self):
        motion = False
        _, cnts, _ = cv2.findContours(self.thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)        
        if len(cnts) > 0:            
            c = max(cnts, key = cv2.contourArea)       
            aarea = cv2.contourArea(c)
                       
            if aarea > self.min_area:
                #print ('Max countours :%s')%(area)
                #cv2.drawContours(frame, c, -1, (0,255,0), 3, 8)
                #################
                #   TRACKING    #
                #################            
                M = cv2.moments(c)
                center = (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))
                self.pts.appendleft(center)
                self.cropBox(center)
                motion = True
                self.trigger_time = time.time()

        """# loop over the set of tracked points
        for i in np.arange(1, len(self.pts)):
            thickness = int(np.sqrt(self.linelength / float(i + 1)) * 2.5)
            cv2.circle(self.curframe,self.pts[0], 5, (0,0,255), -1)
            #cv2.line(self.curframe, self.pts[i - 1], self.pts[i], (0, 0, 255), thickness) 
        """        
                    
        #cv2.rectangle(self.curframe, self.boxcrop_p1, self.boxcrop_p2, (0, 255, 0), 2)
        self.info_logs ('somethingHasMoved :')
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
        self.info_logs ('    Processato cropBox:')

    def cropFrameMotion(self):
        self.cropframe = self.curframe[self.boxcrop_p1[1]:self.boxcrop_p2[1], self.boxcrop_p1[0]:self.boxcrop_p2[0]]
        self.cropframe = detectionTensor(self.cropframe)
        return self.cropframe

    def initRecorder(self): #Create the recorder
        self.recordfile = self.recorddir + self.nome + '_%s.avi'%strftime("%d-%b-%Y-%H-%M-%S", time.localtime())
        # Checks and deletes the output file
        # You cant have a existing file or it will through an error
        if os.path.isfile(self.recordfile):
            os.remove(self.recordfile)

        # Define the codec and create VideoWriter object
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        #self.writer = cv2.VideoWriter(self.recordfile, fourcc, 10.0, (self.recordw,self.recordh))
        self.writer = cv2.VideoWriter(self.recordfile, fourcc, 10.0, (224,224))

        #FPS set at 15 because it seems to be the fps of my cam but should be ajusted to your needs
        self.font = cv2.FONT_HERSHEY_SIMPLEX #Creates a font

    def recording(self):
        if self.isRecording:
            if time.time() >= self.trigger_time + 10: #Record during 10 seconds
                print ("Stop recording")
                self.isRecording = False
            else:
                #cv2.PutText(self.curframe,datetime.now().strftime("%b %d, %H:%M:%S"), (25,30),self.font, 0) #Put date on the frame                
                self.writer.write(self.cropframe) #Write the frame
                
    def recordingframe(self):
        if not self.saveframe:
            nameframe = self.recorddir + self.nome + '.jpg'
            cv2.imwrite(nameframe, self.cropframe)     # save frame as JPEG file
            self.saveframe = True

      
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

    def showImageCrop(self):
        try:
            # display the size of the queue on the frame            
            cv2.imshow("Image Crop", self.cropframe)
        except:
            print ("[ERROR] imshow error")
  

    def close(self):
        #print("[INFO] elasped time: {:.2f}".format(self.fps.elapsed()))
        #print("[INFO] approx. FPS: {:.2f}".format(self.fps.fps()))
        self.writer.release()
        # do a bit of cleanup
        cv2.destroyAllWindows()




def detectionTensor( frame):

    # initialize the list of class labels MobileNet SSD was trained to
    # detect, then generate a set of bounding box colors for each class
    inWidth = 224
    inHeight = 224
    frame = cv2.resize(frame, (inWidth, inHeight))
    WHRatio = inWidth / float(inHeight)
    inScaleFactor =  1 / ((103.94 + 116.78 + 123.68)/3) #0.00784
    meanVal = 127.5
    swapRB = True
    (h, w) = frame.shape[:2] #dimensione dell'immagine
    

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
                    #print('# confidence %s'%confidence)
                    # filter out weak detections by ensuring the `confidence` is
                    # greater than the minimum confidence
                    if confidence > 0.6:                        
                        # extract the index of the class label from the
                        # `detections`, then compute the (x, y)-coordinates of
                        # the bounding box for the object                
                        idx = int(detections[0, 0, i, 1])
                        i=1
                        if i == 1:
                            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])

                            (x1, y1, x2, y2) = box.astype("int")
                            base_center = (x1+(x2-x1)/2, y2)
                            cv2.rectangle(frame, (x1, y1), (x2, y2), COLORS[idx], 2)
                            y = y1 - 15 if y1 - 15 > 15 else y1 + 15
                            label = "{}: {:.2f}%".format(classNames[idx],confidence * 100)
                            cv2.circle(frame,base_center, 5, (0,0,255), -1)
                            cv2.putText(frame, label, (x1, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLORS[idx], 2)                            
                            print (label)
    return frame
           

if __name__=="__main__":
    os.environ['TZ'] = 'Europe/Rome'
    t = time.time() # tempo apertura programma

    print("[INFO] check qopenHab switch perimetrale ...")
    #oh = OpenHab()

    # load our serialized model from disk
    print("[INFO] loading model...")
    pb_path = "C:\\opencv\\RaspyPerson\\models\\frozen_inference_graph.pb"
    pbtxt_path = "C:\\opencv\\RaspyPerson\\models\\ssdlite_mobilenet_v2_coco.pbtxt"
    video_path = "C:\\opencv\\Video\\Sorgenti\\Cam1_25-lug-2018-21-45-52.avi"
    #video_path = 'rtsp://admin:luca2006@79.46.47.94/Streaming/Channels/102'
    #video_path = "rtsp://admin:luca2006@192.168.1.64/Streaming/Channels/102"
    
    # inizializzo il lettore Net DNN di tensorFlow
    net = cv2.dnn.readNetFromTensorflow(pb_path, pbtxt_path)

    # leggo lo stato dello switch dell'allarme
    #stato = oh.get_status("itm_alarm_arm_perimetrale")
    stato = 'ON'

    while True:
        if stato == 'ON':
            t0 = time.time()
            ora = strftime("%d-%b-%Y-%H-%M-%S", time.localtime())
            cam1 = IPCamVideoAlarm("Cam1", video_path, "C:\\opencv\\RaspyPerson\\MascheraCam1.jpg")
            print("[INFO] Starting video alarm process...%s"%ora) 
            
            while True:
                
                cam1.processImage()
                #cam1.alarm_masqueradeImage()
                #cam1.alarm_defocus()
                startt0 = time.time()
                if cam1.somethingHasMoved():
                    cam1.isRecording = True #flag per la registrazione
                    cam1.cropFrameMotion()                     
                    cam1.recording() #registro se c'e un movimento per 10 sec
                    cam1.showImageCrop()
                cam1.showImage()
                


                cam1.info_logs ("TempInvioo globale di rilevamento: ", 0, 'END')

        
                # ogni 20 secondi controllo lo stato dello switch dell'allarme
                t1 = time.time()
                if t1 >= t0+20:
                    #stato = oh.get_status("itm_alarm_arm_perimetrale")
                    t0 = time.time()
                    if stato == 'OFF':
                        cam1.close()
                        break
               
                key = cv2.waitKey(3) & 0xFF
                if key == 27 or key == ord('q'):
                    cam1.close()
                    exit(0)              
        else:
            time.sleep(5)
            started = False            
            stato = oh.get_status("itm_alarm_arm_perimetrale")






