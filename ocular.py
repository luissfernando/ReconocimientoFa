# $$$$$$$$$   noterminado de implementar falta $$$$$$$$$$$$$$ 
import cv2
import dlib
import pyautogui
import numpy as np

# Inicializar el detector de rostros y el predictor de puntos faciales
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

# Inicializar captura de video
cap = cv2.VideoCapture(0)

# Obtener el tamaño de la pantalla
SCREEN_WIDTH, SCREEN_HEIGHT = pyautogui.size()

def midpoint(p1, p2):
    return int((p1.x + p2.x) / 2), int((p1.y + p2.y) / 2)

def calculate_gaze_ratio(eye_points, facial_landmarks):
    left_point = (facial_landmarks.part(eye_points[0]).x, facial_landmarks.part(eye_points[0]).y)
    right_point = (facial_landmarks.part(eye_points[3]).x, facial_landmarks.part(eye_points[3]).y)
    center_top = midpoint(facial_landmarks.part(eye_points[1]), facial_landmarks.part(eye_points[2]))
    center_bottom = midpoint(facial_landmarks.part(eye_points[5]), facial_landmarks.part(eye_points[4]))

    hor_line_length = np.linalg.norm(np.array(left_point) - np.array(right_point))
    ver_line_length = np.linalg.norm(np.array(center_top) - np.array(center_bottom))

    return hor_line_length / ver_line_length

while True:
    ret, frame = cap.read()
    if not ret:
        print("No se pudo acceder a la cámara")
        break

    # Convertir el frame a RGB
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Verificar el tipo y la forma de la imagen
    print("Tipo de la imagen:", rgb_frame.dtype)
    print("Forma de la imagen:", rgb_frame.shape)

    # Intentar detectar rostros en la imagen RGB
    try:
        faces = detector(rgb_frame)
        print("Número de rostros detectados:", len(faces))
    except Exception as e:
        print("Error al detectar rostros:", e)
        continue

    for face in faces:
        landmarks = predictor(rgb_frame, face)

        # Calcular las proporciones de los ojos
        left_eye_ratio = calculate_gaze_ratio([36, 37, 38, 39, 40, 41], landmarks)
        right_eye_ratio = calculate_gaze_ratio([42, 43, 44, 45, 46, 47], landmarks)

        gaze_ratio = (left_eye_ratio + right_eye_ratio) / 2

        # Calcular la posición del ratón
        x_mouse = np.interp(gaze_ratio, [0.8, 1.2], [0, SCREEN_WIDTH])
        y_mouse = np.interp(face.top(), [0, frame.shape[0]], [0, SCREEN_HEIGHT])

        pyautogui.moveTo(int(x_mouse), int(y_mouse))

        # Detección de parpadeo para el clic
        left_eye_blink = (landmarks.part(37).y + landmarks.part(38).y + landmarks.part(40).y + landmarks.part(41).y) / 4
        right_eye_blink = (landmarks.part(43).y + landmarks.part(44).y + landmarks.part(46).y + landmarks.part(47).y) / 4

        blink_ratio = (left_eye_blink + right_eye_blink) / 2

        if blink_ratio < (face.bottom() - face.top()) / 2 - 10:
            pyautogui.click()

    # Mostrar el frame en la ventana de OpenCV
    cv2.imshow('Frame', frame)

    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
