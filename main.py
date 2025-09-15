import numpy as np
import cv2
import time
from collections import deque

prev_time = time.time()

def nothing(x):
    pass

webcam = cv2.VideoCapture(0)
cv2.namedWindow("Trackbars Red")
cv2.namedWindow("Trackbars Green")

# Trackbars rojo
cv2.createTrackbar("LH", "Trackbars Red", 136, 179, nothing)
cv2.createTrackbar("LS", "Trackbars Red", 50, 255, nothing)
cv2.createTrackbar("LV", "Trackbars Red", 26, 255, nothing)
cv2.createTrackbar("UH", "Trackbars Red", 180, 179, nothing)
cv2.createTrackbar("US", "Trackbars Red", 255, 255, nothing)
cv2.createTrackbar("UV", "Trackbars Red", 255, 255, nothing)

# Trackbars verde
cv2.createTrackbar("LH", "Trackbars Green", 58, 179, nothing)
cv2.createTrackbar("LS", "Trackbars Green", 54, 255, nothing)
cv2.createTrackbar("LV", "Trackbars Green", 40, 255, nothing)
cv2.createTrackbar("UH", "Trackbars Green", 83, 179, nothing)
cv2.createTrackbar("US", "Trackbars Green", 255, 255, nothing)
cv2.createTrackbar("UV", "Trackbars Green", 227, 255, nothing)

kernal = np.ones((3, 3), "uint8")

# Historial de objetos más cercanos (buffer de 10 frames)
historial = deque(maxlen=10)

while True:
    _, imageFrame = webcam.read()
    imageFrame = cv2.resize(imageFrame, (320, 240))
    hsvFrame = cv2.cvtColor(imageFrame, cv2.COLOR_BGR2HSV)

    # Trackbars rojo
    l_h_red = cv2.getTrackbarPos("LH", "Trackbars Red")
    l_s_red = cv2.getTrackbarPos("LS", "Trackbars Red")
    l_v_red = cv2.getTrackbarPos("LV", "Trackbars Red")
    u_h_red = cv2.getTrackbarPos("UH", "Trackbars Red")
    u_s_red = cv2.getTrackbarPos("US", "Trackbars Red")
    u_v_red = cv2.getTrackbarPos("UV", "Trackbars Red")

    red_lower = np.array([l_h_red, l_s_red, l_v_red], np.uint8)
    red_upper = np.array([u_h_red, u_s_red, u_v_red], np.uint8)
    red_mask = cv2.inRange(hsvFrame, red_lower, red_upper)

    # Trackbars verde
    l_h_green = cv2.getTrackbarPos("LH", "Trackbars Green")
    l_s_green = cv2.getTrackbarPos("LS", "Trackbars Green")
    l_v_green = cv2.getTrackbarPos("LV", "Trackbars Green")
    u_h_green = cv2.getTrackbarPos("UH", "Trackbars Green")
    u_s_green = cv2.getTrackbarPos("US", "Trackbars Green")
    u_v_green = cv2.getTrackbarPos("UV", "Trackbars Green")

    green_lower = np.array([l_h_green, l_s_green, l_v_green], np.uint8)
    green_upper = np.array([u_h_green, u_s_green, u_v_green], np.uint8)
    green_mask = cv2.inRange(hsvFrame, green_lower, green_upper)

    # Morfología para limpiar ruido
    red_mask = cv2.morphologyEx(red_mask, cv2.MORPH_OPEN, kernal)
    red_mask = cv2.morphologyEx(red_mask, cv2.MORPH_CLOSE, kernal)

    green_mask = cv2.morphologyEx(green_mask, cv2.MORPH_OPEN, kernal)
    green_mask = cv2.morphologyEx(green_mask, cv2.MORPH_CLOSE, kernal)

    # Contornos
    contours_red, _ = cv2.findContours(red_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    contours_green, _ = cv2.findContours(green_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    objetos_detectados = []

    # Contornos rojos
    for contour in contours_red:
        area = cv2.contourArea(contour)
        if area > 300:
            x, y, w, h = cv2.boundingRect(contour)
            objetos_detectados.append((area, (x, y, w, h), "Red", (0, 0, 255)))

    # Contornos verdes
    for contour in contours_green:
        area = cv2.contourArea(contour)
        if area > 300:
            x, y, w, h = cv2.boundingRect(contour)
            objetos_detectados.append((area, (x, y, w, h), "Green", (0, 255, 0)))

    mas_cercano = None
    if objetos_detectados:
        candidato = max(objetos_detectados, key=lambda x: x[0])
        historial.append(candidato)

        # Promedio: contamos cuál color aparece más en el historial
        colores = [obj[2] for obj in historial]
        color_dominante = max(set(colores), key=colores.count)

        # Elegimos el más grande de ese color dentro del frame actual
        candidatos_color = [obj for obj in objetos_detectados if obj[2] == color_dominante]
        if candidatos_color:
            mas_cercano = max(candidatos_color, key=lambda x: x[0])

    # Dibujar objetos
    for area, (x, y, w, h), color_name, color_bgr in objetos_detectados:
        label = f"{color_name} Colour"
        if mas_cercano and (area == mas_cercano[0] and color_name == mas_cercano[2]):
            label += " (mas cercano)"
            cv2.rectangle(imageFrame, (x, y), (x + w, y + h), (0, 255, 255), 3)  # resaltar más
        else:
            cv2.rectangle(imageFrame, (x, y), (x + w, y + h), color_bgr, 2)
        cv2.putText(imageFrame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color_bgr, 2)

    # FPS
    fps = 1 / (time.time() - prev_time)
    prev_time = time.time()
    cv2.putText(imageFrame, f"FPS: {fps:.1f}", (10, 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    cv2.imshow("Color Detection", imageFrame)

    if cv2.waitKey(10) & 0xFF == ord('q'):
        webcam.release()
        cv2.destroyAllWindows()
        break
