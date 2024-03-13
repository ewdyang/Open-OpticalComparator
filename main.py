import sys
from PySide6.QtWidgets import QApplication
from imageViewer import MainWindow

def main():
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    app.exec()
    window.raise_()
    
if(__name__ == "__main__"):
    main()