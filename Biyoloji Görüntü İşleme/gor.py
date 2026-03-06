# koloni_watershed_demo.py
import cv2
import numpy as np

def process_and_count(path, output_path="result_watershed.png",
                      clahe_clip=2.0, clahe_grid=(8,8),
                      tophat_kernel_size=35,
                      morph_kernel_size=3,
                      min_area=30):
    # 1) oku
    img = cv2.imread(path)
    if img is None:
        raise FileNotFoundError(path)
    orig = img.copy()

    # 2) gri + CLAHE (kontrast artırma)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=clahe_clip, tileGridSize=clahe_grid)
    gray_clahe = clahe.apply(gray)

    # 3) Top-hat -> parlak küçük nesneleri vurgula
    kernel_tophat = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (tophat_kernel_size, tophat_kernel_size))
    tophat = cv2.morphologyEx(gray_clahe, cv2.MORPH_TOPHAT, kernel_tophat)

    # 4) Blur + Otsu threshold (otomatik eşik)
    blur = cv2.GaussianBlur(tophat, (5,5), 0)
    _, thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # 5) Morfolojik temizlik (noise azaltma / küçük açıklıkları kapatma)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (morph_kernel_size, morph_kernel_size))
    clean = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=1)
    clean = cv2.morphologyEx(clean, cv2.MORPH_CLOSE, kernel, iterations=2)

    # 6) Surekli bölgeler için uzaklık dönüşümü ve watershed hazırlığı
    # surekli alanların iç noktalarını bul
    dist = cv2.distanceTransform(clean, distanceType=cv2.DIST_L2, maskSize=5)
    # normalize etmek görünür olsun
    dist_norm = cv2.normalize(dist, None, 0, 1.0, cv2.NORM_MINMAX)

    # 7) surekli maksimumları (peak) tespit et
    _, sure_fg = cv2.threshold(dist, 0.4*dist.max(), 255, 0)
    sure_fg = np.uint8(sure_fg)
    sure_bg = cv2.dilate(clean, kernel, iterations=3)
    unknown = cv2.subtract(sure_bg, sure_fg)

    # 8) marker labelling
    _, markers = cv2.connectedComponents(sure_fg)
    # markers için 1'den başlat
    markers = markers + 1
    markers[unknown==255] = 0

    # 9) watershed (renkli sınırlar elde et)
    img_watershed = orig.copy()
    markers = cv2.watershed(img_watershed, markers)
    # sınırları kırmızı ile çiz
    img_watershed[markers == -1] = [0,0,255]

    # 10) Her marker için kontur ve sayma (sadece yeter büyük olanları al)
    unique_markers = np.unique(markers)
    count = 0
    for m in unique_markers:
        if m <= 1:  # 0 = unknown, 1 = background base.
            continue
        mask = np.uint8(markers == m)
        cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not cnts:
            continue
        area = cv2.contourArea(cnts[0])
        if area >= min_area:
            count += 1
            x,y,w,h = cv2.boundingRect(cnts[0])
            cv2.rectangle(img_watershed, (x,y), (x+w,y+h), (0,255,0), 2)
            cv2.putText(img_watershed, str(count), (x, y-5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 1, cv2.LINE_AA)

    # 11) Kaydet ve raporla
    cv2.imwrite(output_path, img_watershed)
    print(f"İşlenen görsel: {path}")
    print(f"Tahmini koloni sayısı: {count}")
    print(f"Sonuç görsel kaydedildi: {output_path}")

    # isteğe bağlı ek çıktı dosyaları
    cv2.imwrite("debug_thresh.png", thresh)
    cv2.imwrite("debug_tophat.png", tophat)
    cv2.imwrite("debug_dist.png", (dist_norm*255).astype(np.uint8))

    return count

if __name__ == "__main__":
    # örnek: kendi dosya adını buraya yaz
    count = process_and_count("koloni3.png", output_path="result_watershed.png",
                              clahe_clip=2.0, clahe_grid=(8,8),
                              tophat_kernel_size=35,
                              morph_kernel_size=3,
                              min_area=30)
