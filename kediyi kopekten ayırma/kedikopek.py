import os
import numpy 
#numpy matris işlemleri uyapmak için kullandığımız kutuphane
import matplotlib.pyplot as plt 
import cv2#opencv nin resim yükleme fonksiyonu ,yeniden boyutlandırma içinde fonksiyonu var
import pickle 
from tqdm import tqdm
from matplotlib import pyplot
import random
from tensorflow.keras import Sequential
#sequantial sıralı birimeler yapay sinir agı ma eger cnn olsaydı matris uzerinde 2*2 seklinde matrisde gezicekti
from tensorflow.keras.layers import Conv2D,Activation,BatchNormalization,Flatten,Dense
#lyer katmanlar her bir katmının olabiliceği
#aktivasyon fonksiyonları grafşkler olan sin taj olandı
# karsılastıgımız  sayılar cok buyuk yada cok kucuk olabilir bunların uzerinden aktivasyon fonksiyonları dediğimiz 
#bu degerleri biraz duzgun tutmayı saglayan 
# her seferinde belli bir sayıda resmi sisteme sokmuş ollucaz
#agırlıkları guncellemeden once mesela 10 tane resim aldım bunları sistemden geçirdim agırlıkları bu 10 unu agırlıkların ortalamasına gore alıcam
#batchnormalization:birden fazla resim aldık bunları normalize ettik 1 tane garip bir ornek sistemin garip ogrenmesinii etkilemesin grup grup bakarken bu grup uzerinden normalize işlemide uygulalayalım
#daha detaylı bir sekilde ogrenmeye devam etsin diye kullandıgımız layer 

#flatten duz duvara dizme layeri








data_dir='C:\\kediyi kopekten ayırma\\PetImages'
KATEGORILER=["Cat","Dog"]#kategorı array 
DIR="PetImages"#ana klasor 
BOYUT=64# 2 inin kuvvetleri olursa daha guzel olur

veri=[]
#dosya okudukça içerisine eklemiş olucam 
#dikkat ediceğim bir unsurda dosyalarımın her birinini boyutu farklı 
#cnn işleminde 2 boyutlu resimlerde bir filtre gezdirerek aralardan bilgi cıkarma metotu gibi 
#kare gorsel boyutu olursa guzel 
#sistem girdi olarak sabit bir bout bekler 
#cnn genel kare geziyor

for kategori in KATEGORILER:
    klasor_adresi=os.path.join(DIR,kategori)#KLASORLERİ BİRLEŞTİR DERKEN KLASOR NASIL BİRLEŞİYORSA O ŞEKİLDE BİRLEŞTİR
    deger=KATEGORILER.index(kategori)
    #resim okucaz
    for resim_adi in tqdm(os.listdir(klasor_adresi)):
        resim_adresi=os.path.join (klasor_adresi,resim_adresi)#resim_adi ni klasor adresine ekledik ,resim adresinden git bu resmi oku diyoruz okurkende 
        resim=cv2.imread(resim_adresi,cv2.IMREAD_GRAYSCALE)
        #bu resmi nerden yuklicez 
        #IMREAD_GRAYSCALE RENKSİZ BİR GORSEL OLARAK OKUR
        #YEDİDEN BOYUTLANDIRMAK İÇİN RESİZE YENİDEN BOYUTLANDIRMAYI SAĞLIYOR
        #resim bos degilse boyutlandır ve ekle 
        if(resim is None ):
            print("hata")
        else :
            resim=cv2.resize(resim,(BOYUT,BOYUT ))#2 parametre olarak nasıl boyut istiyorsak onu yazıcaz
        #resmimiz hazır verimize eklicez
            veri.append([resim,deger])#resme ait olan kedi mi kopek mi sınıfı 
        
pyplot.imshow(veri[0][0],cmap='gray')


#bu verilerle bir traning yani eğitim yapıcaksak karıstırmamızda fayda var olusturdugum tum kedileri gonderip ondan sonrada kopekleri gonderirsek
#ilk gelen kedi olur sonradan gelenler kopek olur olur 
#veriiçeirisindeki sıramalardan dolayı bazı yanlış seyler ogrenmemesi için biz genelde veriyi krıstırak harmalaarak gonderirizi
#veriyi ki sistem dosyalar arasındaki ilişkiyi ogrensin bunların sıralmasını ogrenmesini istemiyoruz 
#  elimizdeki veriyi rastegele karıstırmak isticem bununiçinde 
#randomı kullanıcm
random.shuffle(veri)
#verinin kendi içeriği ve labellları yani etiketleri ayırmak isticem bunun içinde dongu kullanıcam
 #biz neye bakıcaz ve ne bulmaya calısıcaz onları ayrı ayrı tutmamız gerekiyor eğiticeğimiz sisteme gonderirken
 #genel de bunları soyle yaparlar biz x=[] bakıyoruz elimizde x dediğimiz sey resim ve bir de bulmaya calıstıgımız y var y=[] y dediğimiz sey de 
#0 veya 1 kedi mi kopek dediğimiz sey xlere bakarak y ninin ne oldugunuu tahmin etmeye çalıştığımız sistem olucak 

X=[]
Y=[]
for x,y in veri:# x ve y verinin içerisindeki herbir x ve y demiş oluyrum
 X.append(x)
 Y.append(y)


 del veri # diyerek ram doldurmamaya calısıyorum
 pyplot.imshow(X[0],cmap='gray')
 # 1 saatde kaldım

 #eğiticeğimiz sisteme numpy arrayleri gondermiş olucaz
#numpuy arraylerine cevirdikten sonra bununu uzerinden baska fonksiyon cagırıcaz
#reshape() yeniden boyutlandırma fonksiyonunu cagır
X=numpy.array(X).reshape(-1,BOYUT,BOYUT,1)
#NEDEN 1 RENKSİZ KAnalda calıstıgım için 1 eger renklı bir kanalda calıssaydım 3 yazıaktım
#basına -1 yazmamızın sebei herhangi bir sayı gelebirlir manasında 
Y=numpy.array(X).reshape(-1,1)#tek boyutlu bilgi içeriyor
#numppy arry gonderdiğimiz zamna gorsellestirebilir miyiz 
pyplot.imshow(numpy.column_stack(X[0]),cmao='gray')
#normalizasion normalize etmek0 lie 1 arasına indirgeme
#bir veriyi normalize etmek istiyorum
#bir resmi normalize ediyorsak her bir piksel 0 ile 255 deger arasında alır
X=X/255.0#float olarak yadım yoksa tam sayı bolmesi yapabilri
#normalize ederken dikkkat edilecek unsur bir defa normalize ediyoruz
# kerasın içinde hazır bazı modeller var ama biz modeli bastan yazıyormus gibi olucsz,
model=Sequential()
#model dediğimiz sey farklı farklı katmanlardan olusuyor 
model.add(Conv2D(filters=64,kernel_size=(3,3),input_shape=X[0].shape,activation='relu'))
#girdilerin ne tür girdiler olucak onu soylememeiz lazım input_shape
#kare gorselin uzerinde gezdirdiğimiz bir kernel_size
model.add(Conv2D(filters=32,kernel_size=(3,3),activation='relu'))
#input_shape ilk layer ilk katmanda bahsediyoruz cunku modelimize gelecek olan ınput ne olacagı belli değil
#modelimize ardısık katmanlar ekledikçe modelimiz bilitor j-kendi içindeki katmanları onu takip eden katmanların baglantısnı nasıl olucak kaclık bir input beklediği
#için bizim bunları yazmamız gerekmiyor
model.add(BatchNormalization())#bir grup verimiz var pat diye hepsini gondermiyoruz
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dense(1, activation='sigmoid'))  # Output layer for binary classification

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

history = model.fit(X, Y, batch_size=32, epochs=10, validation_split=0.2)

# Modeli kaydet
model.save("cat_dog_classifier.h5")

# Eğitim sonuçlarını görselleştir
plt.plot(history.history['accuracy'], label='Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.show()

plt.plot(history.history['loss'], label='Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.show()

