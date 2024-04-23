import cv2

def main():
    # Inisialisasi kamera
    cap = cv2.VideoCapture(0)

    # Muat kaskade deteksi wajah
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

    while True:
        # Ambil frame dari kamera
        ret, frame = cap.read()

        # Konversi ke skala abu-abu
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Deteksi wajah dalam frame
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)

        # Perkiraan usia
        age_estimation = 24  # Misalnya, asumsi usia 30 tahun untuk setiap wajah yang terdeteksi

        # Gambar kotak dan usia di sekitar wajah yang terdeteksi
        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
            cv2.putText(frame, f'Umur: {age_estimation}', (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2) 

        # Tampilkan frame dengan kotak dan usia di sekitar wajah
        cv2.imshow('Face Detection with Age Estimation', frame)

        # Tunggu tombol 'q' ditekan untuk keluar
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Tutup kamera dan jendela OpenCV
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
