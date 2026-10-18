"""Dark, quiet stylesheet for the Finish desktop window."""

STYLESHEET = """
QMainWindow, QWidget#root {
    background: #121314;
}
QWidget {
    color: #eceae4;
    font-family: "Segoe UI";
    font-size: 13px;
}
QLabel, QCheckBox, QMenuBar {
    background: transparent;
}
QFrame#panel {
    background: #191b1e;
    border: 1px solid #2a2d32;
    border-radius: 14px;
}
QFrame#viewportFrame {
    background: #101113;
    border: 1px solid #2a2d32;
    border-radius: 14px;
}
QLabel#title {
    font-size: 18px;
    font-weight: 600;
    color: #f4f2ec;
    background: transparent;
}
QLabel#subtitle, QLabel#muted, QLabel#hint {
    color: #9b9d98;
    background: transparent;
}
QLabel#section {
    color: #8f938c;
    font-size: 11px;
    font-weight: 600;
    background: transparent;
}
QLabel#value {
    color: #f4f2ec;
    background: transparent;
}
QLabel#status {
    color: #b9bbb6;
    background: transparent;
}
QScrollArea, QScrollArea > QWidget > QWidget {
    background: transparent;
    border: none;
}
QScrollBar:vertical {
    background: transparent;
    width: 10px;
    margin: 4px 0;
}
QScrollBar::handle:vertical {
    background: #3a3e44;
    border-radius: 4px;
    min-height: 24px;
}
QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {
    height: 0;
}
QPushButton {
    background: #2a2e34;
    color: #f0eee8;
    border: 1px solid #3c4047;
    border-radius: 8px;
    padding: 7px 12px;
}
QPushButton:hover { background: #343840; }
QPushButton:pressed { background: #24272c; }
QPushButton:disabled { color: #7d807c; background: #23262a; }
QPushButton#primary {
    background: #efece4;
    color: #1c1d1f;
    border: none;
    border-radius: 9px;
    padding: 11px 14px;
    font-weight: 600;
}
QPushButton#primary:hover { background: #f7f4ee; }
QPushButton#primary:pressed { background: #e2dfd6; }
QPushButton#primary:disabled { background: #3a3d42; color: #8d8f8c; }
QPushButton#ghost {
    background: transparent;
    border: 1px solid #3a3e44;
    border-radius: 8px;
    padding: 5px 10px;
    color: #d9d6cf;
}
QPushButton#ghost:hover { background: #26292e; }
QPushButton#ghost:checked {
    background: #efece4;
    color: #1c1d1f;
    border-color: #efece4;
}
QToolButton {
    background: #141618;
    color: #d7d4cd;
    border: 1px solid #31353b;
    border-radius: 10px;
    padding: 6px 4px 5px 4px;
    font-size: 11px;
}
QToolButton:hover { background: #22252a; }
QToolButton:checked {
    background: #26282c;
    border: 2px solid #efece4;
    color: #f4f2ec;
}
QComboBox, QDoubleSpinBox {
    background: #141618;
    color: #f0eee8;
    border: 1px solid #34383f;
    border-radius: 8px;
    padding: 5px 8px;
}
QComboBox:hover, QDoubleSpinBox:hover { border-color: #4a4e55; }
QComboBox::drop-down { border: none; width: 18px; }
QComboBox QAbstractItemView {
    background: #1c1e22;
    color: #f0eee8;
    border: 1px solid #34383f;
    selection-background-color: #343840;
    outline: none;
}
QSlider::groove:horizontal {
    height: 4px;
    background: #33373d;
    border-radius: 2px;
}
QSlider::sub-page:horizontal {
    background: #b7b2a6;
    border-radius: 2px;
}
QSlider::handle:horizontal {
    background: #f4f1ea;
    width: 14px;
    height: 14px;
    margin: -6px 0;
    border-radius: 7px;
}
QCheckBox { background: transparent; spacing: 8px; color: #e4e2dc; }
QCheckBox::indicator {
    width: 15px;
    height: 15px;
    border-radius: 4px;
    border: 1px solid #5a5e66;
    background: #141618;
}
QCheckBox::indicator:checked {
    background: #efece4;
    border-color: #efece4;
}
QProgressBar {
    background: transparent;
    border: none;
    max-height: 3px;
    min-height: 3px;
}
QProgressBar::chunk { background: #efece4; border-radius: 1px; }
QMenuBar { background: transparent; color: #d9d6cf; }
QMenuBar::item { background: transparent; padding: 4px 8px; }
QMenuBar::item:selected { background: #2a2d32; border-radius: 4px; }
QMenu {
    background: #1c1e22;
    color: #f0eee8;
    border: 1px solid #34383f;
    padding: 4px;
}
QMenu::item { padding: 6px 22px; border-radius: 4px; }
QMenu::item:selected { background: #343840; }
QToolTip {
    background: #1c1e22;
    color: #f0eee8;
    border: 1px solid #34383f;
    padding: 4px 6px;
}
"""
