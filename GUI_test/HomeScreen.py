from PyQt5.QtWidgets import QApplication, QMainWindow
from PyQt5.uic import loadUi
from StudentList import StudentListDialog
from ScanScreen import Scan

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()

        # Load the UI file for the main window
        loadUi('GUI\HomeScreen.ui', self)

        self.start.clicked.connect(lambda: self.open_dialog_window())
        self.student_list.clicked.connect(lambda: self.open_file())
        self.URL.clicked.connect(self.showLineEdit)

        # Create a line edit
        self.line_edit.returnPressed.connect(self.hideLineEdit)

    @staticmethod
    def open_dialog_window():
        # Create the dialog window and show it
        dialog = Scan()
        dialog.exec_()

    @staticmethod
    def open_file():
        student_list = StudentListDialog('GUI\demo.csv')
        student_list.exec_()

    def showLineEdit(self):
        # Hide the button and show the line edit
        self.URL.hide()
        self.line_edit.show()
        self.line_edit.setFocus()

    def hideLineEdit(self):
        # Get the text from the line edit and show the button again
        text = self.line_edit.text()
        self.line_edit.hide()
        self.URL.show()
        self.URL.setFocus()

    def keyPressEvent(self, event):
        if event.key() == 16777216:  # 16777216 is the keycode for the Esc key
            self.hideLineEdit()
        elif event.key() == 16777220:  # 16777220 is the keycode for the Enter key
            text = self.line_edit.text()
            self.line_edit.clear()
            self.hideLineEdit()
            return text


if __name__ == '__main__':
    import sys
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())
