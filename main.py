import sys

from PyQt5.QtWidgets import QApplication

from main_window import HeightMainWindow


APP_STYLESHEET = """
QWidget {
    background-color: #F5F7FA;
    color: #1F2937;
    font-family: "Microsoft YaHei", "Segoe UI";
}
QGroupBox {
    background-color: #FFFFFF;
    border: 1px solid #E4E7EC;
    border-radius: 8px;
    margin-top: 12px;
    padding-top: 4px;
}
QGroupBox::title {
    subcontrol-origin: margin;
    subcontrol-position: top left;
    left: 12px;
    color: #0066CC;
    font-weight: bold;
}
QPushButton {
    background: #0066CC;
    color: #FFFFFF;
    border: none;
    border-radius: 5px;
    padding: 8px 12px;
    font-weight: 600;
}
QPushButton:hover {
    background: #0050A0;
}
"""


def main():
    app = QApplication(sys.argv)
    app.setStyleSheet(APP_STYLESHEET)
    window = HeightMainWindow()
    window.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
