Study Case 3 — AI Engineering: Evaluation + Export + Optimization

Bu çalışmada, hazır bir görsel modelin yalnızca çalıştırılması değil; inference çalıştırma,
değerlendirme, export etme, performans ölçme ve konfigürasyonlar arasında kıyaslama.
Sahada çalışan sistemlerde (özellikle edge senaryolarında) tek başına doğruluk yeterli
olmaz. Doğruluk–gecikme–model boyutu dengesi, format dönüşümlerinin güvenilirliği ve
ölçümlerin tekrarlanabilirliği kritik hale gelir. Bu case, bu mühendislik yaklaşımını ölçer.
Önemli: Ağır model eğitimi beklemiyoruz. Pretrained bir model seçip onun etrafında
doğru bir değerlendirme/benchmark altyapısı kurmanı istiyoruz.

Şu datasetlerden birini seçebilirsin:
    ● COCO 2017 val (öneri: 200–500 görsel gibi küçük bir alt küme)
    ● Pascal VOC 2007 test (daha küçük bir seçenek)
Kendi alt küme seçimini yapabilirsin; ancak seçimini repo içinde net şekilde belirtmelisin.

Pretrained model kullanmanı bekliyoruz, model training yapmanı beklemiyoruz.
Aşağıdaki seçeneklerden yalnızca birini seç:
    ● YOLOv5n / YOLOv8n
    ● SSD-MobileNet benzeri bir PyTorch detektörü
    ● ONNX’e sorunsuz export edilebilen başka hafif bir detektör

Amaç: seçtiğin modelle değerlendirme/export/benchmark süreçlerini sağlam kurmak.
Bu case sonunda, “mini bir benchmark harness” ortaya çıkmalı. Yani:
    ● Dataset üzerinde inference çalıştırıp sayısal sonuçlar üreten,
    ● Modeli ONNX formatına çıkarıp,
    ● PyTorch ve ONNX çıktılarını tutarlılık açısından kontrol edip,
    ● En az bir optimizasyon/konfigürasyon farkı ile tradeoff gösteren,
    ● Sonuçları tekrar üretilebilir dosyalar ve görsellerle raporlayan bir yapı.

Burada görev tarif etmiyoruz. Ancak çözüme giden yolda, üretime benzer refleksler (ölçüm
adaleti, parity check, düzgün raporlama) bekliyoruz.

İpuçları
Bu case’in güçlü görünmesi için genelde şu tür mühendislik refleksleri işe yarar:
    ● Ölçümlerde warmup ve tekrar koşum (N run) gibi adil benchmark yaklaşımı
    ● Export sonrası “aynı input → benzer output” mantığıyla parity check
    ● Sonuçları sadece tek sayı değil; CSV + plot gibi anlaşılır formatlarla sunma
    ● Konfigürasyon değişince (çözünürlük/batch/optimizasyon) ortaya çıkan tradeoff’ları netleştirme

Teslim Formatı
1) Git Repository
Çalışmanı bir Git (Github/Gitlab vs.) reposu olarak paylaşmalısın. Repo temiz, çalıştırılabilir
ve tekrar üretilebilir olmalı.

2) Repo içinde zorunlu dosyalar
● README.md
○ Projenin kısa özeti
○ Kurulum ve çalıştırma talimatı
○ Seçtiğin dataset ve kullandığın alt küme tanımı
○ Seçtiğin model ve versiyonu
○ Üretilen çıktıların nerede oluştuğu (klasör yapısı)
● requirements.txt veya environment.yml
● Kod yapısı (tercihen src/ altında)
○ Notebook kullanabilirsin
● report.pdf (öneri: 2–4 sayfa)
○ Export sürecinde sorun yaşadıysan neydi, nasıl çözdün?
○ Parity check yaklaşımın (nasıl karşılaştırdın, tolerans mantığın neydi?)
○ Sonuçların özeti + final önerin

3) Çalıştırma standardı (beklenen örnek kullanım)
Repo, aşağıdaki gibi tek bir komutla çalıştırılabilir olmalı (isimler farklı olabilir, mantık aynı
olmalı):
● python benchmark.py --backend pytorch|onnx --images 300 --out
results/
Burada amaç: PyTorch backend ve ONNX backend için aynı akışın koşup sonuç üretmesi.

4) Çıktı
Tek bir results/ klasörü altında (veya benzer şekilde) şu tür çıktılar bekliyoruz:
● metrics.csv
● latency.csv
● Plot’lar (örn. accuracy–latency scatter, bar chart’lar)
● (Varsa) predictions.json gibi inference çıktıları