"""
Audio Repeater - listen to an input device on an output device (WASAPI shared mode).

Two dark UI modes: Neumorphism and Glass. Switch with the toggle in the title bar.

Install:   pip install PySide6 sounddevice numpy
Run:       python audio_repeater.py
Build exe: pyinstaller --onefile --windowed --name AudioRepeater audio_repeater.py
"""
import json
import math
import os
import sys
from pathlib import Path

import numpy as np
import sounddevice as sd
from PySide6.QtCore import QPointF, QRectF, Qt, QTimer, QVariantAnimation, Signal
from PySide6.QtGui import (QBrush, QColor, QFont, QLinearGradient, QPainter,
                           QPainterPath, QPen, QPolygonF, QRadialGradient)
from PySide6.QtWidgets import (QAbstractButton, QApplication, QCheckBox,
                               QComboBox, QHBoxLayout, QLabel, QSizePolicy,
                               QSlider, QToolButton, QVBoxLayout, QWidget)

# --------------------------------------------------------------------------
# Settings
# --------------------------------------------------------------------------
SETTINGS_FILE = Path(os.getenv("APPDATA", str(Path.home()))) / "AudioRepeater" / "settings.json"
DEFAULTS = {"theme": "neu", "input": "", "output": "", "volume": 100,
            "latency": 0, "autostart": False}


def load_settings():
    data = dict(DEFAULTS)
    try:
        data.update(json.loads(SETTINGS_FILE.read_text(encoding="utf-8")))
    except Exception:
        pass
    return data


def save_settings(data):
    try:
        SETTINGS_FILE.parent.mkdir(parents=True, exist_ok=True)
        SETTINGS_FILE.write_text(json.dumps(data, indent=2), encoding="utf-8")
    except Exception:
        pass


# --------------------------------------------------------------------------
# Audio engine
# --------------------------------------------------------------------------
def wasapi_index():
    for i, h in enumerate(sd.query_hostapis()):
        if "WASAPI" in h["name"]:
            return i
    return None


def list_devices():
    """Returns (inputs, outputs, default_in_name, default_out_name) for WASAPI."""
    ins, outs, din, dout = [], [], "", ""
    idx = wasapi_index()
    if idx is None:
        return ins, outs, din, dout
    devices = sd.query_devices()
    for i, d in enumerate(devices):
        if d["hostapi"] != idx:
            continue
        if d["max_input_channels"] > 0:
            ins.append((i, d["name"]))
        if d["max_output_channels"] > 0:
            outs.append((i, d["name"]))
    h = sd.query_hostapis(idx)
    try:
        din = devices[h["default_input_device"]]["name"]
        dout = devices[h["default_output_device"]]["name"]
    except Exception:
        pass
    return ins, outs, din, dout


def reinit_portaudio():
    """Re-scan devices (only call while no stream is open)."""
    try:
        sd._terminate()
        sd._initialize()
    except Exception:
        pass


def wasapi_settings():
    try:
        return sd.WasapiSettings(exclusive=False, auto_convert=True)
    except TypeError:  # older sounddevice
        return sd.WasapiSettings(exclusive=False)


class Repeater:
    def __init__(self):
        self.stream = None
        self.volume = 1.0
        self.muted = False
        self.ch_in = self.ch_out = 2
        self.samplerate = 0
        self._pl = 0.0
        self._pr = 0.0

    @property
    def active(self):
        return self.stream is not None and self.stream.active

    def start(self, in_idx, out_idx, latency):
        self.stop()
        din, dout = sd.query_devices(in_idx), sd.query_devices(out_idx)
        self.ch_in = max(1, min(2, din["max_input_channels"]))
        self.ch_out = max(1, min(2, dout["max_output_channels"]))
        rates = []
        for r in (din["default_samplerate"], dout["default_samplerate"], 48000, 44100):
            if r and int(r) not in rates:
                rates.append(int(r))
        last = None
        for sr in rates:
            try:
                st = sd.Stream(device=(in_idx, out_idx), samplerate=sr,
                               channels=(self.ch_in, self.ch_out), dtype="float32",
                               latency=latency, callback=self._callback,
                               extra_settings=wasapi_settings())
                st.start()
                self.stream, self.samplerate = st, sr
                return
            except Exception as e:  # try next sample rate
                last = e
        raise last

    def stop(self):
        if self.stream is not None:
            try:
                self.stream.stop()
                self.stream.close()
            except Exception:
                pass
            self.stream = None

    def _callback(self, indata, outdata, frames, time, status):
        pk = np.abs(indata).max(axis=0)
        l = float(pk[0])
        r = float(pk[1]) if len(pk) > 1 else l
        if l > self._pl:
            self._pl = l
        if r > self._pr:
            self._pr = r
        gain = 0.0 if self.muted else self.volume
        if self.ch_in == 2 and self.ch_out == 1:
            out = indata.mean(axis=1, keepdims=True) * gain
        else:  # same channel count, or mono -> stereo (broadcast)
            out = indata * gain
        np.clip(out, -1.0, 1.0, out=out)
        outdata[:] = out

    def read_peaks(self):
        l, r = self._pl, self._pr
        self._pl = self._pr = 0.0
        return l, r


# --------------------------------------------------------------------------
# Theme
# --------------------------------------------------------------------------
def C(hex_or_color, a=255):
    c = QColor(hex_or_color)
    c.setAlpha(a)
    return c


def css(c):
    return f"rgba({c.red()},{c.green()},{c.blue()},{c.alpha() / 255:.3f})"


PAL = {
    "neu": dict(bg=C("#262a33"), hi=C("#3a4151"), lo=C("#16181d"),
                text=C("#d9dde7"), sub=C("#8b94a9"),
                accent=C("#4fd1c5"), accent2=C("#38bdf8"), danger=C("#fb7185")),
    "glass": dict(bg=C("#14102e"), hi=C("#ffffff", 40), lo=C("#000000", 90),
                  text=C("#f6f5ff"), sub=C("#ffffff", 165),
                  accent=C("#a78bfa"), accent2=C("#22d3ee"), danger=C("#fb7185")),
}
MODE = "neu"


def P():
    return PAL[MODE]


def soft_shadow(p, rect, radius, offset, color, blur, exclude=False):
    """Fake a blurred shadow with stacked translucent rounded rects."""
    p.save()
    if exclude:  # keep the shadow out from under translucent bodies
        path = QPainterPath()
        path.addRect(rect.adjusted(-300, -300, 300, 300))
        path.addRoundedRect(rect, radius, radius)
        path.setFillRule(Qt.OddEvenFill)
        p.setClipPath(path)
    p.setPen(Qt.NoPen)
    n = max(4, blur)
    center = color.alphaF()
    per_layer = 1 - (1 - center) ** (1.0 / n)
    c = QColor(color)
    c.setAlphaF(per_layer)
    p.setBrush(c)
    for i in range(n):
        g = blur * (1 - i / n)
        p.drawRoundedRect(rect.translated(offset).adjusted(-g, -g, g, g), radius + g, radius + g)
    p.restore()


def draw_raised(p, rect, r, depth=6, blur=14):
    pal = P()
    soft_shadow(p, rect, r, QPointF(depth, depth), C(pal["lo"], 215), blur)
    soft_shadow(p, rect, r, QPointF(-depth, -depth), C(pal["hi"], 170), blur)
    g = QLinearGradient(rect.topLeft(), rect.bottomRight())
    g.setColorAt(0, pal["bg"].lighter(106))
    g.setColorAt(1, pal["bg"].darker(108))
    p.setPen(Qt.NoPen)
    p.setBrush(QBrush(g))
    p.drawRoundedRect(rect, r, r)


def draw_inset(p, rect, r):
    pal = P()
    g = QLinearGradient(rect.topLeft(), rect.bottomRight())
    g.setColorAt(0, pal["bg"].darker(135))
    g.setColorAt(1, pal["bg"].darker(108))
    p.setPen(Qt.NoPen)
    p.setBrush(QBrush(g))
    p.drawRoundedRect(rect, r, r)
    edge = QLinearGradient(rect.topLeft(), rect.bottomRight())
    edge.setColorAt(0, C(pal["lo"], 230))
    edge.setColorAt(0.5, C(pal["lo"], 0))
    edge.setColorAt(1, C(pal["hi"], 200))
    p.setBrush(Qt.NoBrush)
    p.setPen(QPen(QBrush(edge), 2))
    p.drawRoundedRect(rect.adjusted(1, 1, -1, -1), r - 1, r - 1)


def draw_glass(p, rect, r, shadow=True, tint=None):
    if shadow:
        soft_shadow(p, rect, r, QPointF(0, 8), C("#000000", 95), 18, exclude=True)
    g = QLinearGradient(rect.topLeft(), rect.bottomRight())
    g.setColorAt(0, tint[0] if tint else C("#ffffff", 46))
    g.setColorAt(1, tint[1] if tint else C("#ffffff", 14))
    p.setPen(Qt.NoPen)
    p.setBrush(QBrush(g))
    p.drawRoundedRect(rect, r, r)
    edge = QLinearGradient(rect.topLeft(), rect.bottomRight())
    edge.setColorAt(0, C("#ffffff", 130))
    edge.setColorAt(0.5, C("#ffffff", 28))
    edge.setColorAt(1, C("#ffffff", 70))
    p.setBrush(Qt.NoBrush)
    p.setPen(QPen(QBrush(edge), 1.2))
    p.drawRoundedRect(rect, r, r)


def draw_well(p, rect, r):
    """A recessed area (meter background, switch track)."""
    if MODE == "neu":
        draw_inset(p, rect, r)
    else:
        p.setPen(Qt.NoPen)
        p.setBrush(C("#000000", 70))
        p.drawRoundedRect(rect, r, r)
        p.setBrush(Qt.NoBrush)
        p.setPen(QPen(C("#ffffff", 45), 1))
        p.drawRoundedRect(rect, r, r)


def build_qss():
    pal = P()
    neu = MODE == "neu"
    text, sub = css(pal["text"]), css(pal["sub"])
    acc, acc2, danger = css(pal["accent"]), css(pal["accent2"]), css(pal["danger"])
    if neu:
        field_bg, field_border = "#1f232a", "rgba(0,0,0,0.45)"
        popup, groove = "#20242b", "#191c21"
        handle = "qlineargradient(x1:0,y1:0,x2:1,y2:1, stop:0 #434b5c, stop:1 #2d333f)"
        handle_border = "rgba(255,255,255,0.08)"
    else:
        field_bg, field_border = "rgba(255,255,255,0.09)", "rgba(255,255,255,0.26)"
        popup, groove = "#241d4a", "rgba(0,0,0,0.32)"
        handle = "qlineargradient(x1:0,y1:0,x2:1,y2:1, stop:0 rgba(255,255,255,0.96), stop:1 rgba(255,255,255,0.72))"
        handle_border = "rgba(255,255,255,0.5)"
    return f"""
* {{ font-family: "Segoe UI Variable", "Segoe UI", sans-serif; }}
QLabel {{ color: {text}; font-size: 13px; background: transparent; }}
QLabel#cap {{ color: {sub}; font-size: 12px; font-weight: 600; }}
QLabel#value {{ color: {acc}; font-size: 13px; font-weight: 600; }}
QLabel#title {{ color: {text}; font-size: 14px; font-weight: 700; }}
QLabel#status {{ color: {sub}; font-size: 12px; }}
QLabel#status[err="true"] {{ color: {danger}; }}
QComboBox {{ background: {field_bg}; border: 1px solid {field_border}; border-radius: 13px;
            padding: 9px 34px 9px 14px; color: {text}; font-size: 13px; min-height: 22px; }}
QComboBox:hover {{ border-color: {acc}; }}
QComboBox::drop-down {{ border: none; width: 0px; }}
QComboBox QAbstractItemView {{ background: {popup}; color: {text}; border: 1px solid {field_border};
            selection-background-color: {acc}; selection-color: #0b0d12; outline: 0; padding: 4px; }}
QSlider::groove:horizontal {{ height: 8px; background: {groove}; border-radius: 4px; }}
QSlider::sub-page:horizontal {{ background: qlineargradient(x1:0,y1:0,x2:1,y2:0, stop:0 {acc2}, stop:1 {acc}); border-radius: 4px; }}
QSlider::handle:horizontal {{ width: 22px; height: 22px; margin: -8px 0; border-radius: 11px;
            background: {handle}; border: 1px solid {handle_border}; }}
QCheckBox {{ color: {sub}; font-size: 12px; spacing: 8px; background: transparent; }}
QCheckBox::indicator {{ width: 18px; height: 18px; border-radius: 6px; background: {field_bg}; border: 1px solid {field_border}; }}
QCheckBox::indicator:checked {{ background: {acc}; border-color: {acc}; }}
QToolButton {{ background: transparent; border: none; color: {sub}; font-size: 14px; border-radius: 10px; padding: 4px 10px; }}
QToolButton:hover {{ background: rgba(255,255,255,0.09); color: {text}; }}
QToolButton#link {{ color: {acc}; font-size: 12px; font-weight: 600; }}
QToolButton#close:hover {{ background: {danger}; color: white; }}
"""


# --------------------------------------------------------------------------
# Custom widgets
# --------------------------------------------------------------------------
def small_label(text):
    l = QLabel(text)
    l.setObjectName("cap")
    return l


class Panel(QWidget):
    """Raised neumorphic card / glass pane. Margins leave room for shadows."""
    PAD = 14

    def __init__(self, inner=16):
        super().__init__()
        self.lay = QVBoxLayout(self)
        m = self.PAD + inner
        self.lay.setContentsMargins(m, m - 2, m, m - 2)
        self.lay.setSpacing(8)

    def paintEvent(self, e):
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)
        rect = QRectF(self.rect()).adjusted(self.PAD, self.PAD, -self.PAD, -self.PAD)
        if MODE == "neu":
            draw_raised(p, rect, 24)
        else:
            draw_glass(p, rect, 24)


class Combo(QComboBox):
    def __init__(self):
        super().__init__()
        self.setSizeAdjustPolicy(QComboBox.AdjustToMinimumContentsLengthWithIcon)
        self.setMinimumContentsLength(8)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        self.setMaxVisibleItems(10)

    def paintEvent(self, e):
        super().paintEvent(e)
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)
        p.setPen(QPen(P()["sub"], 2, Qt.SolidLine, Qt.RoundCap, Qt.RoundJoin))
        x, y = self.width() - 22, self.height() / 2
        p.drawPolyline(QPolygonF([QPointF(x - 5, y - 2), QPointF(x, y + 3), QPointF(x + 5, y - 2)]))


class ModeSwitch(QWidget):
    changed = Signal(str)

    def __init__(self):
        super().__init__()
        self.setFixedSize(156, 34)
        self.setCursor(Qt.PointingHandCursor)
        self._t = 0.0
        self.anim = QVariantAnimation(self)
        self.anim.setDuration(180)
        self.anim.valueChanged.connect(self._tick)

    def _tick(self, v):
        self._t = float(v)
        self.update()

    def set_mode(self, mode, animate=True):
        target = 0.0 if mode == "neu" else 1.0
        if animate:
            self.anim.stop()
            self.anim.setStartValue(self._t)
            self.anim.setEndValue(target)
            self.anim.start()
        else:
            self._t = target
            self.update()

    def mousePressEvent(self, e):
        mode = "neu" if e.position().x() < self.width() / 2 else "glass"
        if mode != MODE:
            self.set_mode(mode)
            self.changed.emit(mode)

    def paintEvent(self, e):
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)
        r = QRectF(self.rect()).adjusted(2, 2, -2, -2)
        draw_well(p, r, r.height() / 2)
        kw = r.width() / 2
        knob = QRectF(r.left() + self._t * kw, r.top(), kw, r.height()).adjusted(3, 3, -3, -3)
        if MODE == "neu":
            draw_raised(p, knob, knob.height() / 2, depth=2, blur=4)
        else:
            draw_glass(p, knob, knob.height() / 2, shadow=False,
                       tint=(C(P()["accent"], 150), C(P()["accent2"], 110)))
        f = QFont("Segoe UI", 9)
        f.setWeight(QFont.DemiBold)
        p.setFont(f)
        for i, label in enumerate(("Neumorphic", "Glass")):
            active = (self._t < 0.5) == (i == 0)
            p.setPen(P()["accent"] if (active and MODE == "neu") else (P()["text"] if active else P()["sub"]))
            p.drawText(QRectF(r.left() + i * kw, r.top(), kw, r.height()), Qt.AlignCenter, label)


class PowerButton(QAbstractButton):
    def __init__(self):
        super().__init__()
        self.setCheckable(True)
        self.setFixedSize(132, 132)
        self.setCursor(Qt.PointingHandCursor)

    def paintEvent(self, e):
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)
        pal = P()
        cx, cy, R = self.width() / 2, self.height() / 2, 46
        rect = QRectF(cx - R, cy - R, 2 * R, 2 * R)
        on = self.isChecked()
        pressed = self.isDown()
        acc = pal["accent"]
        if MODE == "neu":
            if on or pressed:
                draw_inset(p, rect, R)
            else:
                draw_raised(p, rect, R, depth=7, blur=14)
        else:
            soft_shadow(p, rect, R, QPointF(0, 8), C("#000000", 95), 16, exclude=True)
            if on:
                soft_shadow(p, rect, R, QPointF(0, 0), C(acc, 170), 22)
                draw_glass(p, rect, R, shadow=False,
                           tint=(C(pal["accent"], 210), C(pal["accent2"], 150)))
            else:
                draw_glass(p, rect, R, shadow=False)
        # power icon
        arc = QRectF(cx - 17, cy - 16, 34, 34)

        def icon(pen):
            p.setPen(pen)
            p.setBrush(Qt.NoBrush)
            p.drawArc(arc, 120 * 16, 300 * 16)
            p.drawLine(QPointF(cx, cy - 21), QPointF(cx, cy - 3))

        if on and MODE == "neu":
            for w, a in ((17, 26), (11, 52)):
                icon(QPen(C(acc, a), w, Qt.SolidLine, Qt.RoundCap))
        if MODE == "glass":
            col = QColor("#ffffff") if on else C(pal["text"], 200)
        else:
            col = acc if on else pal["sub"]
        icon(QPen(col, 5, Qt.SolidLine, Qt.RoundCap))


class Meter(QWidget):
    N = 40

    def __init__(self):
        super().__init__()
        self.setFixedHeight(66)
        self.lvl = [0.0, 0.0]
        self.hold = [0.0, 0.0]

    @staticmethod
    def norm(v):
        db = 20 * math.log10(max(v, 1e-5))
        return min(1.0, max(0.0, (db + 60) / 60))

    def feed(self, l, r):
        for i, v in enumerate((self.norm(l), self.norm(r))):
            self.lvl[i] = max(v, self.lvl[i] - 0.045)
            self.hold[i] = max(v, self.hold[i] - 0.007)
        self.update()

    def paintEvent(self, e):
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)
        pal = P()
        r = QRectF(self.rect()).adjusted(1, 1, -1, -1)
        draw_well(p, r, 16)
        inner = r.adjusted(34, 11, -14, -11)
        gap, sg, n = 7, 3, self.N
        rh = (inner.height() - gap) / 2
        sw = (inner.width() - (n - 1) * sg) / n
        f = QFont("Segoe UI", 8)
        f.setWeight(QFont.DemiBold)
        p.setFont(f)
        for row in range(2):
            y = inner.top() + row * (rh + gap)
            p.setPen(pal["sub"])
            p.drawText(QRectF(r.left() + 10, y, 18, rh), Qt.AlignCenter, "LR"[row])
            p.setPen(Qt.NoPen)
            hold_i = int(self.hold[row] * (n - 1))
            for i in range(n):
                t = i / (n - 1)
                x = inner.left() + i * (sw + sg)
                if t <= self.lvl[row] and self.lvl[row] > 0:
                    col = pal["accent"] if t < 0.66 else (C("#fbbf24") if t < 0.86 else pal["danger"])
                elif i == hold_i and self.hold[row] > 0.02:
                    col = C(pal["text"], 200)
                else:
                    col = C("#ffffff", 22)
                p.setBrush(col)
                p.drawRoundedRect(QRectF(x, y, sw, rh), 1.8, 1.8)


class TitleBar(QWidget):
    def mousePressEvent(self, e):
        if e.button() == Qt.LeftButton and self.window().windowHandle():
            self.window().windowHandle().startSystemMove()


# --------------------------------------------------------------------------
# Main window
# --------------------------------------------------------------------------
LATENCIES = [("Low", "low"), ("Balanced", 0.05), ("Safe", 0.15)]


class MainWindow(QWidget):
    def __init__(self):
        super().__init__()
        global MODE
        self.cfg = load_settings()
        MODE = self.cfg["theme"] if self.cfg["theme"] in PAL else "neu"
        self.engine = Repeater()
        self.want = False
        self._tick_n = 0

        self.setWindowTitle("Audio Repeater")
        self.setWindowFlags(Qt.FramelessWindowHint | Qt.Window)
        self.setAttribute(Qt.WA_TranslucentBackground)
        self.setFixedSize(500, 660)

        root = QVBoxLayout(self)
        root.setContentsMargins(20, 12, 20, 16)
        root.setSpacing(-8)

        # title bar
        self.titlebar = TitleBar()
        tb = QHBoxLayout(self.titlebar)
        tb.setContentsMargins(8, 0, 0, 0)
        title = QLabel("Audio Repeater")
        title.setObjectName("title")
        self.switch = ModeSwitch()
        self.switch.set_mode(MODE, animate=False)
        self.switch.changed.connect(self.set_mode)
        btn_min, btn_close = QToolButton(), QToolButton()
        btn_min.setText("\u2013")
        btn_close.setText("\u2715")
        btn_close.setObjectName("close")
        btn_min.clicked.connect(self.showMinimized)
        btn_close.clicked.connect(self.close)
        tb.addWidget(title)
        tb.addStretch()
        tb.addWidget(self.switch)
        tb.addSpacing(6)
        tb.addWidget(btn_min)
        tb.addWidget(btn_close)
        self.titlebar.setFixedHeight(46)
        root.addWidget(self.titlebar)

        # routing
        self.cb_in, self.cb_out = Combo(), Combo()
        rp = Panel()
        hdr = QHBoxLayout()
        hdr.addWidget(small_label("Input device"))
        hdr.addStretch()
        btn_refresh = QToolButton()
        btn_refresh.setObjectName("link")
        btn_refresh.setText("\u21bb Refresh devices")
        btn_refresh.clicked.connect(lambda: self.refresh_devices())
        hdr.addWidget(btn_refresh)
        rp.lay.addLayout(hdr)
        rp.lay.addWidget(self.cb_in)
        rp.lay.addSpacing(4)
        rp.lay.addWidget(small_label("Output device"))
        rp.lay.addWidget(self.cb_out)
        root.addWidget(rp)

        # meter
        mp = Panel()
        mh = QHBoxLayout()
        mh.addWidget(small_label("Level"))
        mh.addStretch()
        self.lbl_db = QLabel("\u2212\u221e dB")
        self.lbl_db.setObjectName("value")
        mh.addWidget(self.lbl_db)
        mp.lay.addLayout(mh)
        self.meter = Meter()
        mp.lay.addWidget(self.meter)
        root.addWidget(mp)

        # volume + power
        bottom = QHBoxLayout()
        bottom.setSpacing(-4)
        vp = Panel()
        vh = QHBoxLayout()
        vh.addWidget(small_label("Volume"))
        vh.addStretch()
        self.lbl_vol = QLabel("100%")
        self.lbl_vol.setObjectName("value")
        vh.addWidget(self.lbl_vol)
        vp.lay.addLayout(vh)
        self.slider = QSlider(Qt.Horizontal)
        self.slider.setRange(0, 200)
        self.slider.setFixedHeight(30)
        vp.lay.addWidget(self.slider)
        row = QHBoxLayout()
        self.chk_mute = QCheckBox("Mute")
        self.cb_lat = Combo()
        self.cb_lat.setMinimumContentsLength(5)
        self.cb_lat.setFixedWidth(122)
        for name, _ in LATENCIES:
            self.cb_lat.addItem(name)
        row.addWidget(self.chk_mute)
        row.addStretch()
        row.addWidget(self.cb_lat)
        vp.lay.addLayout(row)
        bottom.addWidget(vp, 1)

        pcol = QVBoxLayout()
        pcol.setSpacing(0)
        self.power = PowerButton()
        self.lbl_state = small_label("Off")
        self.lbl_state.setAlignment(Qt.AlignHCenter)
        pcol.addStretch()
        pcol.addWidget(self.power, 0, Qt.AlignHCenter)
        pcol.addWidget(self.lbl_state)
        pcol.addStretch()
        bottom.addLayout(pcol)
        root.addLayout(bottom)

        # footer
        root.addStretch()
        foot = QHBoxLayout()
        foot.setContentsMargins(16, 0, 16, 0)
        self.status = QLabel("Ready")
        self.status.setObjectName("status")
        self.chk_auto = QCheckBox("Start on launch")
        self.chk_auto.setChecked(bool(self.cfg["autostart"]))
        foot.addWidget(self.status, 1)
        foot.addWidget(self.chk_auto)
        root.addLayout(foot)

        # initial state
        self.slider.setValue(int(self.cfg["volume"]))
        self.engine.volume = self.slider.value() / 100
        self.lbl_vol.setText(f"{self.slider.value()}%")
        self.cb_lat.setCurrentIndex(min(int(self.cfg["latency"]), len(LATENCIES) - 1))
        self.setStyleSheet(build_qss())
        self.refresh_devices(restart=False)

        # signals
        self.slider.valueChanged.connect(self.on_volume)
        self.slider.sliderReleased.connect(self.save)
        self.chk_mute.toggled.connect(lambda v: setattr(self.engine, "muted", v))
        self.chk_auto.toggled.connect(lambda _: self.save())
        self.cb_in.currentIndexChanged.connect(self.on_route_changed)
        self.cb_out.currentIndexChanged.connect(self.on_route_changed)
        self.cb_lat.currentIndexChanged.connect(self.on_route_changed)
        self.power.toggled.connect(self.on_power)

        self.ui_timer = QTimer(self, interval=33, timeout=self.on_ui_tick)
        self.ui_timer.start()
        self.watchdog = QTimer(self, interval=2500, timeout=self.on_watchdog)
        self.watchdog.start()

        if self.chk_auto.isChecked():
            QTimer.singleShot(500, lambda: self.power.setChecked(True))

    # ---- painting
    def paintEvent(self, e):
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)
        r = QRectF(self.rect()).adjusted(0.5, 0.5, -0.5, -0.5)
        path = QPainterPath()
        path.addRoundedRect(r, 30, 30)
        if MODE == "neu":
            g = QLinearGradient(r.topLeft(), r.bottomRight())
            g.setColorAt(0, C("#2c313c"))
            g.setColorAt(1, C("#22252d"))
            p.fillPath(path, QBrush(g))
            p.setPen(QPen(C("#ffffff", 12), 1))
        else:
            g = QLinearGradient(r.topLeft(), r.bottomLeft())
            g.setColorAt(0, C("#0e0b26"))
            g.setColorAt(0.5, C("#1a1342"))
            g.setColorAt(1, C("#0d1630"))
            p.fillPath(path, QBrush(g))
            p.save()
            p.setClipPath(path)
            w, h = r.width(), r.height()
            for fx, fy, rad, col in ((0.10, 0.08, 330, "#7c3aed"), (1.0, 0.33, 300, "#06b6d4"),
                                      (0.18, 0.96, 320, "#ec4899"), (0.85, 0.92, 220, "#6366f1")):
                rg = QRadialGradient(QPointF(fx * w, fy * h), rad)
                rg.setColorAt(0, C(col, 165))
                rg.setColorAt(1, C(col, 0))
                p.fillRect(r, QBrush(rg))
            p.restore()
            p.setPen(QPen(C("#ffffff", 40), 1))
        p.setBrush(Qt.NoBrush)
        p.drawPath(path)

    # ---- theme
    def set_mode(self, mode):
        global MODE
        MODE = mode
        self.setStyleSheet(build_qss())
        for w in self.findChildren(QWidget):
            w.update()
        self.update()
        self.save()

    # ---- devices
    def refresh_devices(self, restart=True):
        was_running = self.want
        self.engine.stop()
        reinit_portaudio()
        ins, outs, din, dout = list_devices()
        for cb, items, saved, default in ((self.cb_in, ins, self.cfg["input"], din),
                                          (self.cb_out, outs, self.cfg["output"], dout)):
            cb.blockSignals(True)
            cb.clear()
            for _, name in items:
                cb.addItem(name)
            names = [n for _, n in items]
            pick = saved if saved in names else (default if default in names else None)
            if pick:
                cb.setCurrentIndex(names.index(pick))
            cb.blockSignals(False)
        if not ins or not outs:
            self.set_status("No WASAPI devices found (Windows only)", err=True)
        if was_running and restart:
            self.try_start()

    def find_index(self, name, kind):
        ins, outs, _, _ = list_devices()
        for i, n in (ins if kind == "in" else outs):
            if n == name:
                return i
        return None

    # ---- actions
    def on_volume(self, v):
        self.engine.volume = v / 100
        self.lbl_vol.setText(f"{v}%")

    def on_route_changed(self, *_):
        self.save()
        if self.want:
            self.try_start()

    def on_power(self, on):
        self.want = on
        self.lbl_state.setText("Listening" if on else "Off")
        if on:
            self.try_start()
        else:
            self.engine.stop()
            self.set_status("Stopped")

    def try_start(self):
        try:
            self.engine.stop()
            reinit_portaudio()
            i = self.find_index(self.cb_in.currentText(), "in")
            o = self.find_index(self.cb_out.currentText(), "out")
            if i is None or o is None:
                raise RuntimeError("device not available")
            self.engine.start(i, o, LATENCIES[self.cb_lat.currentIndex()][1])
            e = self.engine
            self.set_status(f"Repeating \u00b7 {e.samplerate} Hz \u00b7 {e.ch_in}\u2192{e.ch_out} ch \u00b7 WASAPI shared")
            return True
        except Exception as ex:
            msg = str(ex).replace("\n", " ")
            self.set_status(f"Waiting for device: {msg[:70]}", err=True)
            return False

    def on_watchdog(self):
        """Auto-reconnect if the device was unplugged or the stream died."""
        if self.want and not self.engine.active:
            self.try_start()

    def on_ui_tick(self):
        l, r = self.engine.read_peaks()
        self.meter.feed(l, r)
        self._tick_n += 1
        if self._tick_n % 4 == 0:
            pk = max(l, r)
            self.lbl_db.setText("\u2212\u221e dB" if pk < 1e-5 else f"{20 * math.log10(pk):.1f} dB")

    def set_status(self, text, err=False):
        self.status.setText(text)
        self.status.setProperty("err", "true" if err else "false")
        self.status.style().unpolish(self.status)
        self.status.style().polish(self.status)

    def save(self):
        self.cfg.update(theme=MODE, input=self.cb_in.currentText(), output=self.cb_out.currentText(),
                        volume=self.slider.value(), latency=self.cb_lat.currentIndex(),
                        autostart=self.chk_auto.isChecked())
        save_settings(self.cfg)

    def closeEvent(self, e):
        self.want = False
        self.engine.stop()
        self.save()
        super().closeEvent(e)


if __name__ == "__main__":
    app = QApplication(sys.argv)
    app.setApplicationName("Audio Repeater")
    win = MainWindow()
    win.show()
    sys.exit(app.exec())
