from PyQt5.QtCore import QTimer, Qt
from PyQt5.QtWidgets import QDialog, QApplication
from PyQt5.QtGui import QImage, QPixmap, QKeyEvent
from PyQt5.uic import loadUi
import cv2
import pandas as pd
from datetime import datetime


# text = output_scan
# self.student_information.setText()

now = datetime.now()
formatted_date = now.strftime("%d-%m-%Y")


class Scan(QDialog):
    def __init__(self):
        super().__init__()

        loadUi('GUI\Scan.ui', self)
        self.capture = None
        self.video.setScaledContents(True)
        # result func: sau khi điểm danh, present đánh 1 else đánh 0 dùng if {học sinh} in {danh sách hs}

        # Connect button clicked signal to function
        self.result.clicked.connect(self.attendance)  # ko dung lambda de ko p goi parameter # output cua scan cho vao day
        self.end.clicked.connect(lambda: self.quit())
        self.add.clicked.connect(lambda: self.saveToCSV())

        # absolutely vital, else __init__ will run in an infinite loop
        # use the camera without the use of a button
        QTimer.singleShot(1, lambda: self.start_camera())

    def displayCameraFeed(self, frame):
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        # Convert the frame to a QImage
        qImg = QImage(frame.data, frame.shape[1], frame.shape[0], QImage.Format_RGB888)

        # Convert the QImage to a QPixmap
        pixmap = QPixmap.fromImage(qImg)

        # Set the pixmap on the QLabel widget
        self.video.setPixmap(pixmap)

    def start_camera(self):
        # nên dùng nút end để out màn hình nếu ko sẽ xảy ra lỗi nếu start lại màn hình scan
        self.capture = cv2.VideoCapture(0)

        self.capture.set(cv2.CAP_PROP_FRAME_WIDTH, 971)
        self.capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 541)

        while self.capture.isOpened():
            ret, frame = self.capture.read()

            if ret:
                self.displayCameraFeed(frame)
            else:
                break

            cv2.waitKey(30)

    def saveToCSV(self):  # demo. dùng pandas để add tên mới mssv mới vào file csv
        global formatted_date
        text = self.line_edit.text()
        lst = [text.split()[0], int(text.split()[1]), '']
        df = pd.read_csv('GUI\demo.csv')
        df.loc[len(df.index)] = lst
        print('New student was added')
        df.to_csv(f'{formatted_date}.csv', index=False)

    def attendance(self, student_id):  # input is a list of student ids after scanning
        global formatted_date
        df = pd.read_csv('GUI\demo.csv')
        # gọi file gốc, lưu vào file ngày hôm đó
        for id in df['ID']:
            index = df.index[df['ID'] == id].tolist()[0]
            if id in student_id:
                df.loc[index, 'Attendance'] = 1
            else:
                df.loc[index, 'Attendance'] = 0
        df.to_csv(f'{formatted_date}.csv', index=False)

    def capture_image(self):
        # Initialize the camera
        capture = cv2.VideoCapture(0)

        # Capture a frame from the camera
        ret, frame = self.capture.read()
        if ret:
            # Convert the image to RGB format and display it in the label widget
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            h, w, ch = frame.shape
            image = QImage(frame.data, w, h, QImage.Format_RGB888)
            pixmap = QPixmap.fromImage(image)
            self.student_image.setPixmap(pixmap)

    def keyPressEvent(self, event: QKeyEvent):
        if event.key() == Qt.Key_C:
            print("The 'c' key was pressed")
            self.capture_image()

    def quit(self):
        if self.capture:
            self.capture.release()
        self.close()


# Create the application and window
if __name__ == '__main__':
    import sys
    app = QApplication(sys.argv)
    window = Scan()
    window.show()

    # Run the event loop
    sys.exit(app.exec_())
