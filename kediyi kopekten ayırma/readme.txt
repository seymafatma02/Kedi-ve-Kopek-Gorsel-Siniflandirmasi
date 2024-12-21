Kedi ve Köpek Sınıflandırıcı

Bu proje, görsellerdeki kedileri ve köpekleri ayırt etmek için bir yapay sinir ağı (CNN) modelinin kullanıldığı ikili sınıflandırma modelidir. Model, TensorFlow ve Keras kullanılarak oluşturulmuştur.
Proje Özeti

Bu projenin amacı, kedi ve köpek görsellerini sınıflandırabilen bir makine öğrenmesi modeli oluşturmaktır. Görseller ön işleme tabi tutulur, yeniden boyutlandırılır, normalize edilir ve ardından bir CNN ile eğitilir. Eğitilen model, ilerleyen zamanlarda kullanılmak üzere cat_dog_classifier.h5 olarak kaydedilir.
Gereksinimler

Bu projeyi çalıştırabilmek için aşağıdaki Python kütüphanelerinin yüklü olması gerekmektedir:

    numpy
    opencv-python
    matplotlib
    tensorflow
    tqdm

Gerekli kütüphaneleri yüklemek için aşağıdaki komutu kullanabilirsiniz:

pip install numpy opencv-python matplotlib tensorflow tqdm

Veri Seti

Bu projede kullanılan veri seti, "PetImages" klasöründe bulunan ve "Kedi" ve "Köpek" olmak üzere iki kategoriye ayrılmış görsellerden oluşmaktadır. Görseller önce okunur, yeniden boyutlandırılır ve gri tonlama (grayscale) formatına dönüştürülür. Her görsel, kedi için 0 ve köpek için 1 etiketleriyle etiketlenir.
Proje Nasıl Çalışır?

    Veri Ön İşleme: Görseller okunur, yeniden boyutlandırılır ve gri tonlamaya dönüştürülür. Veri karıştırılır ve eğitim için X (görseller) ve Y (etiketler) olarak ayrılır.
    Model Mimarisi: İki konvolüsyonel katman, ardından batch normalization, flatten ve dense katmanları eklenerek bir Convolutional Neural Network (CNN) oluşturulur.
    Eğitim: Model, Adam optimizasyon algoritması ve binary cross-entropy kayıp fonksiyonu ile eğitilir. Eğitim süreci, 10 epoch boyunca devam eder ve doğrulama verisi için %20'lik bir oran kullanılır.
    Modelin Kaydedilmesi: Eğitim tamamlandıktan sonra model, cat_dog_classifier.h5 olarak kaydedilir.

Kullanım Talimatları

    Bu repository'yi klonlayın:

git clone https://github.com/kullaniciadi/kedi-kopek-siniflandirici.git

Proje klasörüne gidin:

cd kedi-kopek-siniflandirici

Modeli eğitmek için aşağıdaki komutu çalıştırın:

python train_model.py

Model eğitildikten sonra, görsel sınıflandırmak için aşağıdaki komutu kullanabilirsiniz:

    python classify_image.py --image_path "gorsel_yolu"

Sonuçlar

Eğitim süreci, hem eğitim hem de doğrulama setleri için doğruluk (accuracy) ve kayıp (loss) değerlerini gösteren grafiklerle sonuçlanır.
Lisans

Bu proje MIT Lisansı altında lisanslanmıştır.