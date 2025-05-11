import cv2
import sys
from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtGui import QImage, QPixmap, QFont
from PyQt5.QtWidgets import QApplication, QLabel, QMainWindow, QVBoxLayout, QWidget
from keras import models
import numpy as np
import pyttsx3
import threading

MODEL_NAME = "sign-language.h5"

class IsaretDiliUygulamasi(QMainWindow):
    def __init__(self):
        super().__init__()

        # Modeli yükle ve metin okuma motorunu başlat
        self.model = models.load_model(MODEL_NAME)
        self.sesli_motor = pyttsx3.init()
        self.sesli_motor.setProperty('rate', 150)  # Konuşma hızı
        self.sesli_motor.setProperty('voice', 'turkish')  # Türkçe ses
        self.initUI()

    def initUI(self):
        self.central_widget = QWidget(self)
        self.setCentralWidget(self.central_widget)

        self.layout = QVBoxLayout(self.central_widget)

        # Kamera görüntüsünü gösteren QLabel
        self.kamera_etiketi = QLabel(self)
        self.layout.addWidget(self.kamera_etiketi)

        # Tahmin edilen harfi gösteren QLabel
        self.tahmin_harf_etiketi = QLabel(self)
        self.tahmin_harf_etiketi.setFont(QFont('Arial', 20))  # QLabel üzerindeki metni büyütme
        self.layout.addWidget(self.tahmin_harf_etiketi)

        # Kamera görüntüsünü düzenli olarak güncelleyen QTimer
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.kareyi_guncelle)
        self.timer.start(30)  # Her 30 milisaniyede bir güncelleme

        # Kamera için VideoCapture başlat
        self.video_capture = cv2.VideoCapture(0)

        self.setGeometry(100, 100, 600, 600)
        self.setWindowTitle('İşaret Dili Tanıma')
        self.show()

    def kareyi_guncelle(self):
        ret, kare = self.video_capture.read()
        kare = cv2.flip(kare, 1)

        # Kamera görüntüsünü sol QLabel'e ekle
        kare_rgb = cv2.cvtColor(kare, cv2.COLOR_BGR2RGB)
        image = QImage(kare_rgb.data, kare_rgb.shape[1], kare_rgb.shape[0], QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(image)
        self.kamera_etiketi.setPixmap(pixmap)

        # Sağdaki QLabel'e tahmin edilen harfi ekle
        kesilmis = kare[50:800, 50:800]
        yeniden_boyutlandirilmis = (cv2.cvtColor(cv2.resize(kesilmis, (28, 28)), cv2.COLOR_RGB2GRAY)) / 255.0
        veri = yeniden_boyutlandirilmis.reshape(-1, 28, 28, 1)
        model_cikisi = self.model.predict([veri])[0]
        tahmin_edilen_etiket = np.argmax(model_cikisi)

        if max(model_cikisi) > 0.9:
            harf = self.get_harf(tahmin_edilen_etiket)
            self.tahmin_harf_etiketi.setText(f"Tahmin Edilen Harf: {harf}")
            threading.Thread(target=self.harf_oku, args=(harf,)).start()

    def harf_oku(self, harf):
        # Sesli olarak harfi oku
        self.sesli_motor.say(f"{harf}")
        self.sesli_motor.runAndWait()

    def get_harf(self, etiket):
        harfler = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
        return harfler[etiket] if 0 <= etiket < len(harfler) else ""

def main():
    app = QApplication(sys.argv)
    uygulama = IsaretDiliUygulamasi()
    sys.exit(app.exec_())

if __name__ == '__main__':
    main()
