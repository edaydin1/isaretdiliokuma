Projemin amacı kamera açıldığına parmak işaretleriyle gösterilen harfin tanımlanması. Projemde opencv kütüphanesi ile görüntüyü işleyip keras kütüphanesini içerisinden cnn modeli ile modelimi oluşturdum. Oluşturduğum modeli qtdesigner kullanarak masaüstü uygulaması haline getirdim.

# 🤟 İşaret Dili Tanıma Projesi

Bu proje, 📸 görüntüler üzerinden Amerikan İşaret Dili (ASL) harflerini tanıyan bir yapay zeka modeli sunar. `Keras` ve `TensorFlow` kullanılarak eğitilmiş bu model, MNIST benzeri bir işaret dili veri seti olan `sign_mnist` ile geliştirilmiştir. Hedefimiz, 💬 sözlü iletişimin mümkün olmadığı durumlarda, işaret dilini dijital ortama taşımaktır.

Model, hazır bir şekilde `sign-language.h5` dosyasında yer alır. Ayrıca eğitim ve test verileri `.csv` formatında proje klasöründe mevcuttur. Eğitim sürecini tekrar başlatmak veya farklı bir model denemek istersen, `model.py` dosyasını kullanabilirsin. Tahminleri test etmek ve sonucu gözlemlemek için ise `main.py` seni bekliyor. 🧠➡️🔤

Klasörde ayrıca 📷 `Alfabe.png` adında işaret dili harflerinin yer aldığı görsel bir referans da bulunmakta. Bu görsel hem kullanıcı hem de geliştirici açısından oldukça yardımcı olacaktır.

🧰 Bu projeyi çalıştırmak için şu kütüphanelere ihtiyacın olacak:
- `TensorFlow`
- `NumPy`
- `Pandas`
- `Matplotlib`
- `Scikit-learn`

Kurulum için terminalde aşağıdaki komutu kullanabilirsin:

```bash
pip install tensorflow numpy pandas matplotlib scikit-learn

🚀 Projeyi çalıştırmak çok kolay:

Eğitim için: python model.py

Tahmin/test için: python main.py
