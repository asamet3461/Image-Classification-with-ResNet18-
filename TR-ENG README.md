#  Image Classification with ResNet18  
#  ResNet18 ile Görüntü Sınıflandırması

## Description | Açıklama

This project uses transfer learning with a ResNet18 backbone for image classification. It includes training on ship images and fine-tuning on an automobile dataset.

Bu proje, ResNet18 tabanlı transfer öğrenme yöntemiyle görüntü sınıflandırması yapar. Gemi resimleri üzerinde eğitim ve ardından otomobil veri kümesi üzerinde ince ayar içerir.

---

##  Steps | Adımlar

1. **Model Setup | Model Kurulumu**
   - Load pretrained ResNet18  
   - Replace final fully connected layer based on number of classes  

   - Hazır eğitilmiş ResNet18 modeli yüklenir  
   - Son tam bağlantılı katman sınıf sayısına göre değiştirilir

2. **Training | Eğitim**
   - Train on Ship Dataset using Adam optimizer and CrossEntropyLoss  
   - Track training/validation accuracy and loss per epoch  

   - Adam optimizasyonu ve CrossEntropyLoss ile gemi veri kümesi üzerinde eğitilir  
   - Her epoch’ta doğruluk ve kayıp takip edilir

3. **Fine-Tuning | İnce Ayar**
   - Replace dataset with automobile images  
   - Modify final layer and retrain using same function  

   - Veri kümesi otomobil görselleriyle değiştirilir  
   - Son katman uyarlanır ve aynı eğitim fonksiyonu ile yeniden eğitilir

---

##  Output | Çıktı

- Ship classification accuracy: **~87%**  
- Automobile classification performance: **High/Comparable**  

- Gemi sınıflandırma doğruluğu: **~%87**  
- Otomobil sınıflandırması: **Yüksek/Benzer performans**

---

##  Requirements | Gereksinimler

```bash
pip install torch torchvision matplotlib
