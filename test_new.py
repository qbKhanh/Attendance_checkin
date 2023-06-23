import sys
from PIL import Image
import datetime
from PyQt5.QtWidgets import QApplication, QMainWindow
from PyQt5.QtWidgets import QDialog, QTableWidget, QTableWidgetItem, QVBoxLayout
from PyQt5 import QtGui
from PyQt5.QtGui import *
from PyQt5.QtCore import Qt, QTimer, QTime
from GUI.demo import Ui_MainWindow
from GUI.StudentList import StudentListDialog
from sklearn.decomposition import PCA
from csv import writer
import cv2
import numpy as np
import pandas as pd

from main import *

class MainWindow(QMainWindow, Ui_MainWindow):
    def __init__(self):
        super().__init__()
        self.setupUi(self)

        self.input_name_current = None
        self.input_mssv_current = None
        self.student_now = None
        self.time_now = None
        self.df = pd.read_csv('data_format.csv')
        self.student_in_class = self.df['Name'].tolist()

        self.log = ''
        self.recog = Recognition()

        self.student_list.clicked.connect(lambda: self.open_file())
        self.add.clicked.connect(self.add_student)
        self.start_program()
        
    # add student to data
    def add_student(self):
        self.log += 'log> Adding new student\n'
        self.input_name_current = self.name_input.toPlainText()
        self.input_mssv_current = self.mssv_input.toPlainText()

        if (len(self.input_name_current) == 0) or (len(self.input_mssv_current) == 0):
            if (len(self.input_name_current) == 0):
                self.log += '   Name empty\n' 
            if (len(self.input_mssv_current) == 0):
                self.log += '   MSSV empty\n' 
        else:
            self.log += f'  Name: {self.input_name_current}\n  MSSV: {self.input_mssv_current}\n'
            data_create_csv = datetime.today().date().strftime('%d-%m-%y')
            try:
                df = pd.read_csv(f'{data_create_csv}.csv')
            except:
                df = pd.read_csv('data_format.csv')
            self.student_in_class.append(self.input_name_current)
            new_row = {'roll_no': self.input_mssv_current, 'Name': self.input_name_current, 'Status': 'Absent', 'Time': '-', 'Note': '-'}
            df = df.append(new_row, ignore_index=True)
            df.to_csv(f'{data_create_csv}.csv',index=False)

            self.name_input.clear()
            self.mssv_input.clear()

            self.log += 'log> Training.....\n'
            self.print_log()

            self.recog.add_new(self.input_name_current)
            
            
            self.recog.svm_Classify()
            self.log += '   Complete!\n' 
            self.print_log()



        self.print_log()
        self.cap = cv2.VideoCapture(0)

    # print Log
    def print_log(self):
        self.check_log.setText(str(self.log))
        self.check_log.verticalScrollBar().setValue(self.check_log.verticalScrollBar().maximum())

    @staticmethod
    def open_file():
        data_create_csv = datetime.today().date().strftime('%d-%m-%y')
        try:
            df = f'{data_create_csv}.csv'
            student_list = StudentListDialog(df)
            student_list.exec_()
        except:
            df = 'data_format.csv'
            student_list = StudentListDialog(df)
            student_list.exec_()
        



    # create a timer to update the label
    def start_program(self):
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_frame)
        self.timer.start(30)
        # open the default webcam
        self.cap = cv2.VideoCapture(0)

    def check_point_time(self):
        oc_current_time = QTime.currentTime()
        yeahh = oc_current_time.toString('hh:mm:ss')
        format_time = yeahh.split(':')
        hour,minute,second = format_time[0],format_time[1],format_time[2]
        return (int(hour)*3600+int(minute)*60+int(second))

    def update_frame(self):
        # read a frame from the webcam
        ret, frame = self.cap.read()
        if ret:
            try:
                frame, name, face = self.recog.face_reg(frame)
                if name != self.student_now:
                    self.student_now = name
                    self.time_now = self.check_point_time()
                else:
                    duration = self.check_point_time() - self.time_now
                    if duration >= 5:
                        if name in self.student_in_class:
                            df_index = self.student_in_class.index(f"{name}")
                            self.student_in_class.pop(df_index)
                            oc_current_time = QTime.currentTime()
                            time = oc_current_time.toString('hh:mm:ss')
                            self.log += 'log> Student Check-in\n'
                            self.log += f'   {time} - {self.student_now} - Attend\n' 
                            self.print_log()
                            self.status_csv(name,time)


                face = np.array(face)
                face = cv2.cvtColor(face,cv2.COLOR_RGB2BGR)
                #face = cv2.resize(face,(,10))
            # convert the OpenCV frame to a QPixmap
                qimage_2 = QImage(face.data, face.shape[1], face.shape[0], QImage.Format_RGB888)
                pixmap_2 = QPixmap.fromImage(qimage_2)
                self.face.setPixmap(pixmap_2)
        
            except:
                print('No Face')
                frame = self.recog.face_reg(frame)


        try:
            frame = cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
            # convert the OpenCV frame to a QPixmap
            qimage = QImage(frame.data, frame.shape[1], frame.shape[0], QImage.Format_RGB888)
            pixmap = QPixmap.fromImage(qimage)

            # update the label with the new frame
            self.frame.setPixmap(pixmap)
        except:pass

    def status_csv(self,name,time):
        data_create_csv = datetime.today().date().strftime('%d-%m-%y')
        try:
            df = pd.read_csv(f'{data_create_csv}.csv')
        except:
            df = pd.read_csv('data_format.csv')
        idx = df[df['Name'] == name].index.values[0]
        df.iloc[idx][2] = 'present'
        df.iloc[idx][3] = time
        
        # data_create_csv = data_create_csv.strftime("%m-%d")
        df.to_csv(f'{data_create_csv}.csv',index=False)



app = QApplication(sys.argv)
window = MainWindow()
window.show()
sys.exit(app.exec_())
