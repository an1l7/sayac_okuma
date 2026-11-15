
## Kurulum (Windows 11)

Python 3.10 veya 3.11 önerilir. NVIDIA GPU ve PyTorch için [resmî PyTorch kurulum yönergesini](https://pytorch.org/get-started/locally/) izleyin. İlk kurulumda internet gerekir.

```powershell
py -3.11 -m venv .venv
.venv\Scripts\python.exe -m pip install --upgrade pip
.venv\Scripts\python.exe -m pip install -r requirements.txt
```

**Model daha önce indirilmişse (çevrimdışı):** `BASLAT_OFFLINE.bat` dosyasına çift tıklayın ya da:

```powershell
$env:MODEL_LOCAL_ONLY="1"
.venv\Scripts\python.exe -u -m uvicorn main:app --host 127.0.0.1 --port 8001
```

**Model hiç indirilmemişse (ilk indirme):** İnternet açıkken `BASLAT_ILK_KURULUM.bat` dosyasını çalıştırın ya da:

```powershell
$env:MODEL_LOCAL_ONLY="0"
.venv\Scripts\python.exe -u -m uvicorn main:app --host 127.0.0.1 --port 8001
```

Tarayıcı: **http://127.0.0.1:8001/**. İlk analizde model yüklenir. Model daha önce başka bir Hugging Face önbelleğine veya farklı Windows kullanıcısının klasörüne indirilmişse internet olmadan model bulunamayabilir. Varsayılan Qwen2-VL model önbelleği sistemde zaten varsa yeniden indirilmez. Model klasörünü doğrudan kullanmak için `MODEL_ID=C:\model_klasoru` gibi yerel dizin verebilirsiniz.

**Önemli:** `MODEL_LOCAL_ONLY=0` indirmeye izin verir, indirmenin başarılı olacağını garanti etmez. VPN/SSL/sertifika sorunu ayrı çözülmelidir. Model ilk kez indirildikten ve önbellek tamamlandıktan sonra `BASLAT_OFFLINE.bat` ile bağlantısız çalıştırabilirsiniz.

## Özellikler

- Koyu/açık tema; yan menülü yeni kontrol paneli; duyarlı (responsive) yerleşim
- Çoklu dosya, sürükle-bırak ve panodan görsel yapıştırma
- Sıralı analiz kuyruğu, duraklatma, başarısız dosyayı yeniden deneme
- Anlık işlem sayıları, başarı oranı, ilerleme çubuğu
- Önizleme; parlaklık, kontrast, sağa/sola döndürme, aynalama (analizden önce)
- Sonuç düzenleme/doğrulama, arama, filtreleme, sıralama
- CSV ve JSON dışa aktarma; isteğe bağlı yerel oturum kaydı / oturum geçmişi
- Varsayılan gizlilik: anonim dosya adları, maskeli seri numaraları; kayıt/dışa aktarmada da uygulanır
- Çevrimdışı model önbelleği; ilk analizde isteğe bağlı model yükleme

## Gizlilik

**Fotoğraflar ve orijinal örnek veri pakete alınmaz ve sunucuya kalıcı olarak kaydedilmez.** Tarayıcı önbelleği yerine yalnızca kullanıcının tema/gizlilik tercihleri localStorage'a yazılır. Analiz modeli bilgisayarınızda çalışır, yükleme yalnızca `127.0.0.1` üzerinden kendi sunucunuza yapılır. İlk kurulumda Hugging Face ile model indirme bağlantısı kurulabilir.

Gizlilik modu açıkken seri numarasının sadece son dört karakteri görünür; **kaydedilen/export edilen veri de maskelidir** (orijinal seri sonradan geri getirilemez). Gizlilik modunu kapatırsanız dışa aktarım ham kimlikleri içerir; oturum kaydetmeden önce onay istenir. Sayaç fotoğraflarındaki kişisel etiketleri otomatik bulan/gizleyen bir görüntü anonimleştirme algoritması **yoktur**; fotoğrafları paylaşmadan önce ayrıca kontrol edin.

`results/` klasöründeki geçmiş oturumlar yalnızca 'Oturumu kaydet' tıklanırsa oluşur ve ZIP'e alınmaz. Eski repo fotoğrafları, isimler, dosya yolları, seri no listeleri, e-posta ve API anahtarları bu dağıtımda yer almaz.

## Uyumluluk ve sınırlamalar

- Orijinal uç nokta yanıtları değişmedi: `POST /api/analyze` → `{"status":"success","m3_degeri":"...","seri_no":"..."}`.
- `POST /api/save-results` eski `{"results":[...]}` gövdesini kabul eder. Yeni `anonymize` alanı varsayılan `true` olur. **Varsayılan kayıt davranışı anonimleştirildiği için eski entegrasyonlar ham değer bekliyorsa `anonymize:false` geçmelidir.**
- Eski uygulamanın `"KASKI "` seri ön eki yerine yeni uygulama sadece bulunan sayısal seri değerini döndürür. Sonuç alanının adı korunur.
- Model çıktısının doğruluğu fotoğraf kalitesine bağlıdır; kullanıcı sonuçları kontrol etmelidir.
- Python ortamı, model ağırlıkları (birkaç GB) ve görsel örnekler pakete konmadı. Önce bağımlılıkları kurmanız gerekir.
- Resmî donanımda ve gerçek model ağırlıklarıyla uçtan uca GPU OCR testi bu ortamda yapılamadı; sunucu API / statik arayüz testleri eklenmiştir.

## API

| Yöntem | Adres | Açıklama |
|---|---|---|
| `GET` | `/api/status` | Sunucu ve model durumu |
| `POST` | `/api/analyze` | `multipart/form-data` içindeki `file` görselini analiz eder |
| `POST` | `/api/save-results` | Sonuç listesini JSON olarak kaydeder |
| `GET` | `/api/history` | Yerel kayıt özetleri |
| `GET` | `/api/history/{id}` | Kaydedilmiş metin sonuçları |

## Test

```powershell
.venv\Scripts\python.exe -m pip install pytest httpx
.venv\Scripts\python.exe -m pytest -q
```

## Güvenlik

Sunucuyu mümkün olduğunca `127.0.0.1` üzerinde başlatın; uygulamada kullanıcı girişi bulunmadığından ağa açılan sistemleri güvenilir ağ dışında paylaşmayın. Dışa aktarılan CSV değerleri formül enjeksiyonuna karşı metin olarak kaçışlanır. Büyük dosyalara boyut sınırı uygulanır.
