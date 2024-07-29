import cv2
import numpy as np
import pyautogui
import time

# Configurar el tamaño de la región de captura
region = (150, 160, 780, 450)

# Inicializar el bucle de captura de pantalla
try:
    while True:
        # Captura de pantalla en la región especificada
        screenshot = pyautogui.screenshot(region=region)
        
        # Convertir la captura a un formato compatible con OpenCV
        screenshot_np = np.array(screenshot)
        screenshot_bgr = cv2.cvtColor(screenshot_np, cv2.COLOR_RGB2BGR)
        
        # Mostrar la imagen en una ventana
        cv2.imshow("Screenshot", screenshot_bgr)
        
        # Salir del bucle si se presiona la tecla 'ESC'
        if cv2.waitKey(1) & 0xFF == 27:
            break
        
        # Esperar 5 segundo antes de la próxima captura de pantalla
        time.sleep(5)

except Exception as e:
    print(f"Se produjo un error: {e}")

finally:
    # Liberar los recursos y cerrar las ventanas de OpenCV
    cv2.destroyAllWindows()
