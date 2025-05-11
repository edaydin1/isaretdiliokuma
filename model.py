import pandas as pd
import numpy as np
import os
from random import shuffle
from tqdm import tqdm
from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPool2D
from keras.optimizers import Adam

MODEL_NAME = "sign-language.h5"

def veri_isleme():
    # Eğitim ve test veri setlerini oku
    egitim_df = pd.read_csv('sign_mnist_train.csv')
    test_df = pd.read_csv('sign_mnist_test.csv')

    # Gerekli sütunları seç
    egitim_veri = egitim_df.iloc[:, 1:].values
    test_veri = test_df.iloc[:, 1:].values

    # Etiketleri seç
    egitim_etiket = egitim_df['label'].values
    test_etiket = test_df['label'].values

    # Veriyi düzenle: Reshape ve normalizasyon
    egitim_veri = np.array(egitim_veri).reshape((-1, 1, 28, 28)).astype(np.uint8) / 255.0
    test_veri = np.array(test_veri).reshape((-1, 1, 28, 28)).astype(np.uint8) / 255.0

    # Etiketleri kategorik formata dönüştür
    egitim_etiket = to_categorical(egitim_etiket, 25).astype(np.uint8)

    # Eğitim veri setini karıştır
    egitim_veri_listesi = list(zip(egitim_veri, egitim_etiket))
    shuffle(egitim_veri_listesi)

    # Test veri setini düzenle
    test_veri_listesi = list(zip(test_veri, range(1, len(test_veri) + 1)))

    return egitim_veri_listesi, test_veri_listesi, test_etiket

def model_olustur():
    model = Sequential()
    model.add(Conv2D(32, (2, 2), activation='relu', input_shape=(28, 28, 1)))
    model.add(Conv2D(64, (2, 2), activation='relu'))
    model.add(MaxPool2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Conv2D(32, (2, 2), activation='relu'))
    model.add(Conv2D(64, (2, 2), activation='relu'))
    model.add(MaxPool2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Conv2D(32, (2, 2), activation='relu'))
    model.add(Conv2D(64, (2, 2), activation='relu'))
    model.add(MaxPool2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Flatten())
    model.add(Dense(1024, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(25, activation='softmax'))
    optimizer = Adam(lr=0.001)
    model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])

    return model

def egitim():
    # Veriyi işle ve modeli oluştur
    egitim_verisi, test_verisi, test_etiket = veri_isleme()

    model = model_olustur()

    # Eğer model dosyası varsa, ağırlıkları yükle
    if os.path.exists(MODEL_NAME):
        model.load_weights(MODEL_NAME)
        print('Model var, ağırlıklar yüklendi.')

    # Eğitim ve doğrulama setlerini oluştur
    egitim_seti = egitim_verisi[:-500]
    test_seti = egitim_verisi[-500:]
    X, y = zip(*egitim_seti)
    test_x, test_y = zip(*test_seti)

    X = np.array(X).reshape([-1, 28, 28, 1])
    y = np.array(y)

    test_x = np.array(test_x).reshape([-1, 28, 28, 1])
    test_y = np.array(test_y)

    # Modeli eğit
    model.fit(X, y, epochs=5, verbose=1, validation_data=(test_x, test_y))

    # Eğitilmiş modeli kaydet
    model.save(MODEL_NAME)

    return model, test_verisi, test_etiket

def test_et(model, test_verisi, test_etiket):
    dogru = 0

    for veri, true_label in zip(test_verisi, test_etiket):
        img_data, img_num = veri[0], veri[1]

        # Veriyi yeniden şekillendir
        data = img_data.reshape(-1, 28, 28, 1)
        model_out = model.predict([data])[0]

        # Tahmin edilen etiketi bul
        tahmin_edilen_etiket = np.argmax(model_out)

        # Doğru tahminleri say
        if true_label == tahmin_edilen_etiket:
            dogru += 1

    # Doğruluk oranını yazdır
    print("Test Doğruluğu: {:.2f}%".format(float(dogru) / len(test_verisi) * 100))


if __name__ == "__main__":
    model, test_verisi, test_etiket = egitim()
    test_et(model, test_verisi, test_etiket)
