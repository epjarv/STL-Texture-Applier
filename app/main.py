import os
import sys
import traceback
from pathlib import Path

os.environ.setdefault("QT_API", "pyside6")


def main() -> None:
    log_path = Path(__file__).resolve().parent.parent / "finish-error.log"
    try:
        from PySide6.QtGui import QFont
        from PySide6.QtWidgets import QApplication

        from app.window import MainWindow

        app = QApplication(sys.argv)
        app.setApplicationName("STL Texture Applier")
        app.setOrganizationName("STL Texture Applier")
        app.setFont(QFont("Segoe UI", 10))
        window = MainWindow()
        window.show()
        sys.exit(app.exec())
    except Exception:
        log_path.write_text(traceback.format_exc(), encoding="utf-8")
        raise


if __name__ == "__main__":
    main()
