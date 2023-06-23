from PyQt5.QtWidgets import QDialog, QTableWidget, QTableWidgetItem, QVBoxLayout
from PyQt5.uic import loadUi
import pandas as pd


class StudentListDialog(QDialog):
    def __init__(self, filename):
        super().__init__()

        # Load the UI file for the dialog window
        loadUi('GUI\StudentList.ui', self)

        # Load the student list from CSV file
        student_df = pd.read_csv(filename)

        # Create a table widget to display the student list
        self.tableWidget = QTableWidget()
        self.tableWidget.setColumnCount(len(student_df.columns))
        self.tableWidget.setRowCount(len(student_df.index))
        self.tableWidget.setHorizontalHeaderLabels(student_df.columns)

        # Populate the table with the data
        for row in range(len(student_df.index)):
            for col in range(len(student_df.columns)):
                self.tableWidget.setItem(row, col, QTableWidgetItem(str(student_df.iloc[row, col])))

        # Add the table widget to the dialog layout
        layout = QVBoxLayout()
        layout.addWidget(self.tableWidget)
        self.setLayout(layout)
