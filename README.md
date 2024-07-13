# PROJEK-UAS-PCD-C
Ujian Akhir Semester Pengolahan Citra Digital

Nama : Cynthia Maharani

Nim : 202231046

# Pemaparan Source

# 1. Import Lybrary

```bash
import cv2
import numpy as np
import matplotlib.pyplot as plt
%matplotlib inline
import skimage
```

Kode yang diberikan mengimpor berbagai perpustakaan penting yang digunakan dalam pengolahan citra dan visualisasi data dengan Python. Pertama, modul `cv2` dari OpenCV diimpor, yang merupakan pustaka utama untuk pemrosesan citra dan video, menawarkan berbagai fungsi untuk operasi visi komputer. Kemudian, `numpy` diimpor dengan alias `np`, yang merupakan pustaka fundamental untuk komputasi ilmiah di Python, menyediakan array multidimensi besar dan matriks beserta berbagai fungsi matematika tingkat tinggi. Selanjutnya, `pyplot` dari Matplotlib diimpor dengan alias `plt`, yang digunakan untuk membuat visualisasi data seperti grafik dan menampilkan gambar. Perintah khusus `%matplotlib inline` digunakan untuk memastikan bahwa grafik Matplotlib akan ditampilkan langsung dalam Jupyter Notebook. Terakhir, modul `skimage` dari scikit-image diimpor, yang merupakan pustaka open-source untuk pemrosesan gambar, menyediakan algoritma efisien untuk berbagai operasi pemrosesan gambar. Kombinasi dari pustaka-pustaka ini memungkinkan pengguna untuk membaca, memanipulasi, dan menampilkan gambar, serta menerapkan berbagai transformasi dan analisis pada gambar tersebut.

# 2. Membaca dan menampilkan Gambar

```bash
image = cv2.imread('Media.jpg')

cv2.imshow("cyn", image)
cv2.waitKey(0)
cv2.destroyAllWindows()
```

Kode tersebut membaca dan menampilkan gambar menggunakan OpenCV. Pertama, gambar dengan nama file 'Media.jpg' dibaca dan dimuat ke dalam variabel image menggunakan fungsi cv2.imread. Fungsi ini mengambil nama file gambar sebagai argumen dan mengembalikan gambar dalam bentuk array NumPy. Selanjutnya, gambar yang dimuat ditampilkan dalam jendela baru dengan judul "cyn" menggunakan fungsi cv2.imshow. Fungsi ini menampilkan gambar yang diberikan dalam jendela dengan nama yang ditentukan. Fungsi cv2.waitKey(0) digunakan untuk menunggu input dari keyboard; jendela akan tetap terbuka sampai ada tombol yang ditekan. Setelah tombol ditekan, cv2.destroyAllWindows dipanggil untuk menutup semua jendela yang dibuat oleh OpenCV. Kombinasi fungsi-fungsi ini memungkinkan pengguna untuk memuat dan melihat gambar di jendela yang terpisah hingga mereka memutuskan untuk menutupnya dengan menekan tombol apa pun.

# 3. Konversi gambar RGB ke grayscale

```bash
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
edges = cv2.Canny(image, 130, 90)
```
Kode tersebut melakukan konversi gambar berwarna ke dalam gambar grayscale dan mendeteksi tepi pada gambar menggunakan OpenCV. Pertama, gambar berwarna yang telah dimuat dalam variabel image dikonversi ke dalam format grayscale menggunakan fungsi cv2.cvtColor. Fungsi ini mengubah ruang warna gambar dari BGR (Blue, Green, Red) ke grayscale, dan hasilnya disimpan dalam variabel gray. Konversi ini bermanfaat untuk mengurangi kompleksitas pengolahan citra karena gambar grayscale hanya memiliki satu kanal warna dibandingkan dengan tiga kanal pada gambar berwarna. Selanjutnya, deteksi tepi pada gambar dilakukan menggunakan algoritma Canny dengan fungsi cv2.Canny. Fungsi ini mendeteksi tepi dengan menetapkan dua threshold, yaitu 130 dan 90, yang mengontrol sensitivitas deteksi tepi. Hasil dari deteksi tepi ini disimpan dalam variabel edges. Dengan dua langkah ini, kode tersebut menghasilkan dua versi gambar: satu dalam grayscale dan satu yang menyoroti tepi-tepi objek dalam gambar asli.

# 4. Menampilkan hasil deteksi gambar

```bash
cv2.imshow("Foto cyn", edges)
cv2.waitKey(0)
cv2.destroyAllWindows()
```
Kode tersebut menampilkan hasil deteksi tepi dari gambar menggunakan OpenCV. Pertama, gambar yang telah diproses untuk mendeteksi tepi dan disimpan dalam variabel edges ditampilkan dalam jendela baru dengan judul "Foto cyn" menggunakan fungsi cv2.imshow. Fungsi ini menampilkan gambar yang diberikan dalam jendela dengan nama yang ditentukan, dalam hal ini menunjukkan hasil deteksi tepi yang telah dilakukan sebelumnya. Fungsi cv2.waitKey(0) digunakan untuk menunggu input dari keyboard, sehingga jendela akan tetap terbuka sampai ada tombol yang ditekan oleh pengguna. Setelah tombol ditekan, fungsi cv2.destroyAllWindows dipanggil untuk menutup semua jendela yang dibuat oleh OpenCV. Dengan langkah-langkah ini, pengguna dapat melihat hasil deteksi tepi pada gambar dalam jendela terpisah hingga mereka menutupnya dengan menekan tombol apa pun.

# 5. Memvisualisasi gambar grayscale dan hasil deteksi tepi

```bash
fig, axs = plt.subplots(1,2, figsize =(10,10))
ax = axs.ravel()

ax[0].imshow(gray, cmap = "gray")
ax[0].set_title("Original Image")

ax[1].imshow(edges, cmap = "gray")
ax[1].set_title("Canny Edge Detection")
```

Kode tersebut menggunakan Matplotlib untuk membuat visualisasi gambar dalam bentuk subplot, menampilkan gambar grayscale asli dan hasil deteksi tepi Canny secara berdampingan. Pertama, fig, axs = plt.subplots(1, 2, figsize=(10, 10)) membuat sebuah figure dan dua subplot yang disusun dalam satu baris dan dua kolom, dengan ukuran keseluruhan 10x10 inci. Variabel axs adalah array dari objek subplot, dan ax = axs.ravel() meratakan array tersebut untuk memudahkan akses ke setiap subplot secara individual. Pada subplot pertama (ax[0]), gambar grayscale (gray) ditampilkan menggunakan ax[0].imshow(gray, cmap="gray"), dengan colormap "gray" untuk memastikan gambar ditampilkan dalam skala abu-abu. Judul subplot pertama diatur menjadi "Original Image" menggunakan ax[0].set_title("Original Image"). Pada subplot kedua (ax[1]), hasil deteksi tepi Canny (edges) ditampilkan juga dalam skala abu-abu menggunakan ax[1].imshow(edges, cmap="gray"), dan diberi judul "Canny Edge Detection" menggunakan ax[1].set_title("Canny Edge Detection"). Dengan kode ini, kedua gambar ditampilkan berdampingan dalam satu figure untuk memudahkan perbandingan antara gambar asli dan hasil deteksi tepinya.

# 6. Mendeteksi Garis tepi menggunakan algoritma hough

```bash
lines = cv2.HoughLinesP(edges, 1, np.pi/255, 0, maxLineGap = 2)
image_line = image.copy()
```

Kode tersebut menggunakan transformasi Hough untuk mendeteksi garis-garis pada gambar yang telah diproses menggunakan deteksi tepi Canny, dan kemudian membuat salinan dari gambar asli untuk menampilkan hasil deteksi garis. Pertama, lines = cv2.HoughLinesP(edges, 1, np.pi/255, 0, maxLineGap=2) menggunakan fungsi cv2.HoughLinesP untuk mendeteksi garis pada gambar tepi (edges). Fungsi ini menggunakan parameter berikut:

- edges: gambar tepi yang dihasilkan dari deteksi tepi Canny.
- 1: jarak resolusi dalam piksel dari parameter rho.
- np.pi/255: resolusi sudut dalam radian dari parameter theta.
- 0: nilai ambang minimum yang diperlukan untuk mendeteksi garis (threshold).
- maxLineGap=2: jarak maksimum antara segmen garis yang diizinkan untuk dipertimbangkan sebagai satu garis tunggal.
- Fungsi ini mengembalikan sebuah array yang berisi koordinat garis yang terdeteksi.

Selanjutnya, image_line = image.copy() membuat salinan dari gambar asli (image) dan menyimpannya dalam variabel image_line. Salinan ini digunakan untuk menggambar garis-garis yang terdeteksi pada gambar asli tanpa mengubah gambar asli itu sendiri. Dengan langkah-langkah ini, kode tersebut menyiapkan data deteksi garis dan membuat salinan gambar asli untuk menampilkan hasil deteksi garis nantinya.

# 7. Mendeklarasikan setiap garis pada gambar

```bash
for line in lines:
    x1, y1, x2, y2 = line[0]
    cv2.line(image_line, (x1, y1), (x2, y2), (0, 255, 0), 3)
```

Kode tersebut menggambar garis-garis yang terdeteksi pada gambar salinan menggunakan OpenCV. Loop for line in lines: iterasi melalui setiap garis yang terdeteksi dan disimpan dalam array lines, yang dihasilkan oleh fungsi cv2.HoughLinesP sebelumnya. Setiap line dalam array lines berisi empat nilai yang mewakili koordinat titik awal (x1, y1) dan titik akhir (x2, y2) dari garis tersebut. Dalam loop, x1, y1, x2, y2 = line[0] mengekstrak koordinat-koordinat ini dari setiap garis.
Kemudian, sebuah loop for line in lines: digunakan untuk mengakses setiap garis yang terdeteksi dalam lines. Setiap garis direpresentasikan sebagai empat nilai yang menyatakan koordinat titik awal dan akhir garis. Dalam loop ini, koordinat ini diekstraksi menggunakan x1, y1, x2, y2 = line[0]. Selanjutnya, fungsi cv2.line(image_line, (x1, y1), (x2, y2), (0, 255, 0), 3) digunakan untuk menggambar garis pada gambar salinan (image_line). Parameter-parameter yang diberikan termasuk koordinat titik awal dan akhir garis, warna garis (dalam format BGR), dan ketebalan garis dalam piksel.

Hasil dari proses ini adalah gambar asli yang telah diperbarui dengan garis-garis yang terdeteksi ditambahkan. Ini memungkinkan visualisasi yang jelas tentang letak dan orientasi garis-garis yang diidentifikasi oleh transformasi HoughLinesP. Dengan cara ini, pengguna dapat memahami dan menganalisis distribusi garis-garis dalam gambar, yang dapat digunakan untuk berbagai aplikasi dalam pemrosesan citra dan visi komputer.

# 8. Menampilkan semua gambar yang sudah di olah dalam satu visualisasi

```bash
fig, axs = plt.subplots(1, 3, figsize=(15, 5))  
ax = axs.ravel()

ax[0].imshow(gray, cmap="gray")
ax[0].set_title("Original Image")

ax[1].imshow(edges, cmap="gray")
ax[1].set_title("Canny Edges Image")

ax[2].imshow(image_line, cmap="gray")
ax[2].set_title("Contours Image")

plt.show()
```

Codingan tersebut menggunakan Matplotlib untuk membuat tiga subplot dalam satu baris dengan ukuran total 15x5 inci. Pertama, fig, axs = plt.subplots(1, 3, figsize=(15, 5)) membuat sebuah figure dengan tiga subplot yang tersusun dalam satu baris (1 baris dan 3 kolom), dengan ukuran keseluruhan 15x5 inci. Variabel axs adalah array dari objek subplot, dan ax = axs.ravel() meratakan array tersebut untuk memudahkan akses ke setiap subplot secara individual.

Pada subplot pertama (ax[0]), gambar grayscale asli (gray) ditampilkan menggunakan ax[0].imshow(gray, cmap="gray"), dengan menggunakan colormap "gray" untuk memastikan gambar ditampilkan dalam skala abu-abu. Judul subplot pertama diatur menjadi "Original Image" dengan ax[0].set_title("Original Image").

Pada subplot kedua (ax[1]), gambar hasil deteksi tepi Canny (edges) ditampilkan juga dalam skala abu-abu menggunakan ax[1].imshow(edges, cmap="gray"), dan diberi judul "Canny Edges Image" dengan ax[1].set_title("Canny Edges Image").

Pada subplot ketiga (ax[2]), gambar asli yang telah diperbarui dengan garis-garis yang terdeteksi (image_line) ditampilkan dalam skala abu-abu menggunakan ax[2].imshow(image_line, cmap="gray"), dan diberi judul "Contours Image" dengan ax[2].set_title("Contours Image").

Terakhir, plt.show() digunakan untuk menampilkan figure dengan semua subplot yang telah disiapkan. Dengan cara ini, pengguna dapat dengan mudah membandingkan gambar asli, gambar hasil deteksi tepi, dan gambar yang telah diperbarui dengan garis-garis yang terdeteksi dalam satu tampilan visual yang terorganisir. Ini sangat berguna untuk analisis dan evaluasi hasil dari proses pemrosesan gambar yang telah dilakukan sebelumnya.

# 9. Menampilkan format Grayscale

```bash
edges
```
Adalah variabel yang menyimpan gambar hasil dari proses deteksi tepi menggunakan algoritma Canny. Gambar ini biasanya dalam format grayscale di mana tepi-tepi objek dalam gambar lebih menonjol sebagai daerah dengan nilai intensitas yang tinggi.


# 10. Menyimpan hasil transformasi Hough

```bash
lines
```
Adalah variabel yang menyimpan hasil dari transformasi Hough yang digunakan untuk mendeteksi garis-garis dalam gambar tepi (edges). lines umumnya berupa array atau matriks yang berisi koordinat-koordinat dari setiap garis yang terdeteksi dalam gambar. Setiap baris dari matriks ini mungkin berisi empat nilai yang mewakili koordinat titik awal dan akhir dari garis yang terdeteksi.


# 11.Variable yang menujukan dimensi dari gambar

```bash
lines.shape
```
Adalah atribut dari variabel line (bukan lines) yang menunjukkan dimensi dari line. Jika line adalah array NumPy, line.shape akan memberikan tuple yang menunjukkan jumlah elemen di setiap dimensi dari line. Misalnya, jika line adalah matriks, line.shape akan memberikan (jumlah baris, jumlah kolom) dari matriks tersebut.
