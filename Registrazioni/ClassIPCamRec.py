# import the necessary packages
import numpy as np
import cv2
import sys, os, time, traceback
from datetime import datetime
from time import gmtime, strftime




class IPCamVideoAlarm_Recording():
    def __init__(self, cartella, nome_cam, fps, width, height):

        # Inizializzazione variabili
        self.nome_cam = nome_cam # nome della IPCam o del file

        # Inizializzazioni variabili per i processi
        self.recframe = None
        
        # REGISTRAZIONE
        #inizializzo la dimensione del video
        self.record_fps = fps
        self.record_height = height
        self.record_width =  width
        


        self.time_recording = 2 #tempo per la registrazione
            
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
        
        #inizializzo le variabili, le cartelle (video e foto) e il file di registrazione video
        self.initRecorder(cartella)

    # INIZIO REGISTRAZIONE
    def initRecorder(self, cartella): #Create the recorder
        try:


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

            self.writer = cv2.VideoWriter(self.recording_video_file, fourcc, self.record_fps, (self.record_width, self.record_height))
            print(("File %s per la registrazione video FPS:%s (w:%s h:%s)")%(self.recording_video_file, self.record_fps, self.record_width,self.record_height))
        except:
            tb = sys.exc_info()[2]
            tbinfo = traceback.format_tb(tb)[0]
            pymsg = "PYTHON ERRORS:\nTraceback Info:\n" + tbinfo + "\nError Info:\n    " + \
                str(sys.exc_type)+ ": " + str(sys.exc_value) + "\n"
            print(pymsg)
            print(("Errore inizializzazione file %s per la registrazione video")%(self.recording_video_file))
            return False

    def Video(self, frame):
        try:
            if self.time_recording > 0 and self.isRecording:
                if time.time() >= self.trigger_time + self.time_recording: #Record during n seconds
                    print ("[INFO] Stop recording")
                    self.isRecording = False
            if self.isRecording:
                    self.writer.write(frame) #Write the frame
                    self.asRecording = True
        except:
            tb = sys.exc_info()[2]
            tbinfo = traceback.format_tb(tb)[0]
            pymsg = "PYTHON ERRORS:\nTraceback Info:\n" + tbinfo + "\nError Info:\n    " + \
                str(sys.exc_type)+ ": " + str(sys.exc_value) + "\n"
            print(pymsg)
            
            return False
            print(("Errore registrazione frame %s")%(self.recording_video_file))
            
    def Foto(self, frame):
        try:
            if not self.savedframe:
                nome_recorded_file = self.nome_cam + '_%s.jpg'%strftime("%d-%b-%Y-%H-%M-%S", time.localtime())
                nameframe = os.path.join(self.recording_foto_dir, nome_recorded_file)
                cv2.imwrite(nameframe, frame)     # save frame as JPEG file
                print(("Salvato frame in %s")%(nameframe))
                self.savedframe = True
        except:
            print(("Errore registrazione frame %s")%(self.recording_video_file))
    # FINE REGISTRAZIONE

    def close(self):
        # chiudo la registrazione
        self.writer.release()
        
        # se non ho registrato cancello il file
        if self.asRecording == False:
            print(("Cancello file %s per la registrazione video, non ci sono frame registrati")%(self.recording_video_file))
            os.remove(self.recording_video_file)


