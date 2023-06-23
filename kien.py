import sys
from PIL import Image
import datetime
from PyQt5.QtWidgets import QApplication, QMainWindow
from PyQt5.QtGui import QPixmap,QImage
from PyQt5.QtCore import Qt, QTimer, QTime
from demo import Ui_MainWindow
from sklearn.decomposition import PCA
from detect_recognize_face import *
from csv import writer
import cv2
import numpy as np
import pandas as pd





class MainWindow(QMainWindow, Ui_MainWindow):
    def __init__(self):
        super().__init__()
        self.setupUi(self)

        self.face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
        self.log = []
        self.all_student = None

        self.student_now, self.time_now = None,None
        
        self.start_time = '07:00:00'
        self.end_time   = '23:59:59'
        self.current_time = None

        self.data=np.load('savedfaces.npz')
        self.faces,self.target=self.data['faces'],self.data['target']
        self.pca,self.eigenfaces=get_eigenfaces(self.faces,100)
        self.svm=train_model(self.faces,self.target,self.pca)

        self.pushButton.clicked.connect(self.start_program)
        # self.recognize.clicked.connect(self.check_student)
        self.update_time()
        self.get_student()

########################################################################################################### GET ALL STUDENT IN CLASS
    def get_student(self):
        df_stu = pd.read_csv('student_in_class.csv')
        list_student = df_stu['Ho va Ten'].to_list()
        self.all_student = list_student

########################################################################################################### SHOW LOG HISTORY
    def put_text_lcd_log(self):
        res = ''
        for ele in self.log:
            res += f'   {ele[0]} - {ele[1]} - {ele[2]}\n'
        self.textBrowser.setPlainText(res)

########################################################################################################### START PROGRAM
    def start_program(self):
        # create a timer to update the label
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_frame)
        self.timer.start(30)
        # open the default webcam
        self.cap = cv2.VideoCapture(0)

########################################################################################################### CHECK POINT
    def check_point_time(self):
        oc_current_time = QTime.currentTime()
        yeahh = oc_current_time.toString('hh:mm:ss')
        format_time = yeahh.split(':')
        hour,minute,second = format_time[0],format_time[1],format_time[2]
        return (int(hour)*3600+int(minute)*60+int(second))

########################################################################################################### UPDATE FRAME FROM CV2
    def update_frame(self):
        # read a frame from the webcam
        ret, frame = self.cap.read()
        
        try:
            frame,name = recogize_face(frame,self.pca,self.svm)
            student_name = name[0]

            if student_name != self.student_now:
                self.student_now = student_name
                self.time_now = self.check_point_time()
            else:
                duration = self.check_point_time() - self.time_now
                if duration >= 5:
                    self.check_student(student_name) # Check status student
        except:
            frame = cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
            

        # convert the OpenCV frame to a QPixmap
        qimage = QImage(frame.data, frame.shape[1], frame.shape[0], QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(qimage)

        # update the label with the new frame
        self.screen.setPixmap(pixmap)

########################################################################################################### UPDATE CLOCK TIME & ABSENT ALL STUDENT ELSE
    def update_lcd_time(self):
        # UPDATE LCD CLOCK TIME
        time = QTime.currentTime()
        self.current_time = time.toString('hh:mm:ss')
        self.clock.display(self.current_time)

        # ABSENT ALL STUDENT ELSE
        time_current,time_end = self.convert_time(self.current_time),self.convert_time(self.end_time)
        if time_current> time_end:
            self.end_time = '99:99:99'
            for ele in self.all_student:
                self.save_data(ele,'absent')
        
    def update_time(self):
        self.timerrrr = QTimer()
        self.timerrrr.timeout.connect(self.update_lcd_time) # update every 1000ms (1 second)
        self.timerrrr.start(1000)

########################################################################################################### CHECK STUDENT ATTENT TO CLASS THEN SAVE DATA
    def convert_time(self,a):
        format_time = a.split(':')
        hour,minute,second = format_time[0],format_time[1],format_time[2]
        return (int(hour)*3600+int(minute)*60+int(second))
    
    def save_data(self,name,status_ss):
        #push data in csv file
        df = pd.read_csv('student_in_class.csv')
        temp_name = df['Ho va Ten'].tolist()
        index_name = temp_name.index(name)
        currr_time = str(datetime.datetime.now().strftime('%d/%m'))
        df[currr_time][index_name] = status_ss
        df.to_csv('student_in_class.csv',index=False)

        #push data in lcd log
        self.log.append([self.current_time,name,status_ss])
        self.put_text_lcd_log()
    
    def check_student(self,currrrrrr_name):
        if currrrrrr_name in self.all_student:
            nahhh = self.all_student.pop(self.all_student.index(currrrrrr_name))

            start = self.convert_time(self.start_time)
            end   = self.convert_time(self.end_time)
            curr = self.convert_time(self.current_time)
            status = None
            
            # check time student enter classroom
            if curr < start:
                status = 'attend'
                
            elif (start<curr) and (curr<end):
                absent_time = curr - start
                hours = absent_time // 3600
                minutes = (absent_time % 3600) // 60
                seconds = (absent_time % 60) // 1
                status = f'late ({hours}h{minutes}p{seconds}s)'

            elif end < curr:
                status = 'absent' 

            self.save_data(currrrrrr_name,status)
    

########################################################################################################### ABSENT ALL STUDENT ELSE IN LIST  
    
            


app = QApplication(sys.argv)
window = MainWindow()
window.show()
sys.exit(app.exec_())
