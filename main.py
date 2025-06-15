from typing import List, Literal
import numpy as np
import math
import cv2


def track_parking_lot(video_path: str):


    def qtd_pixels_diferentes(frame, empty_frame, mask):
        """
        Calcula a quantidade de pixels diferentes entre o frame atual e a imagem de referência (vaga vazia) de acordo com a máscara fornecida.
        Args:
            frame: O frame atual do vídeo.
            empty_frame: A imagem de referência da vaga vazia.
            mask: A máscara que define a área de interesse (vaga).
        Returns:
            int: A quantidade de pixels diferentes na área definida pela máscara.
        """

        current_frame = cv2.bitwise_and(frame, frame, mask=mask)
        baseline_image = cv2.bitwise_and(empty_frame, empty_frame, mask=mask)


        cv2.medianBlur(baseline_image, 7)
        cv2.medianBlur(current_frame, 7)
        difference = cv2.absdiff(baseline_image, current_frame)
        
        gray_difference = cv2.cvtColor(difference, cv2.COLOR_BGR2GRAY)
        
        _, difference_mask = cv2.threshold(gray_difference, 200, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
        
        non_zero_count = cv2.countNonZero(difference_mask)
        return non_zero_count


    def avg_color_difference(frame, empty_frame, mask):
        """
        Calcula a distância média de cor entre o frame atual e a imagem de referência (vaga vazia) de acordo com a máscara fornecida.
        Args:
            frame: O frame atual do vídeo.
            empty_frame: A imagem de referência da vaga vazia.
            mask: A máscara que define a área de interesse (vaga).
        Returns:
            float: A distância média de cor entre o frame atual e a imagem de referência na área definida pela máscara.
        """
        cor_media_vazio = cv2.mean(empty_frame, mask=mask)
        cor_media_frame = cv2.mean(frame, mask=mask)
        return np.linalg.norm(np.array(cor_media_vazio[:3]) - np.array(cor_media_frame[:3]))


    def std_deviation_of_color(frame, mask) -> float:
        """
        Calcula o desvio padrão das cores dentro da vaga.
        Args:
            frame: O frame atual do vídeo.
            mask: A máscara que define a área da vaga.
        Returns:
            float: O desvio padrão das cores na área definida pela máscara.
        """
        return cv2.meanStdDev(frame, mask=mask)[1][0][0]


    # Carrega o vídeo
    cap = cv2.VideoCapture(video_path)
    img_vazio = cv2.imread('media\\background.png')

    # Loop através de cada frame no vídeo
    while cap.isOpened():

        # Lê o próximo frame
        ret, frame = cap.read()
        if not ret:
            break

        spots = [
            [(60, 150), (230, 150), (140, 430), (0, 430), (0, 290)]
            #[(240, 140), (390, 140), (390, 420), (240, 420)],
            #[(430, 135), (580, 135), (580, 415), (430, 415)],
            #[(610, 135), (760, 135), (760, 415), (610, 415)],
        ]


        # Itera sobre cada vaga
        for idx, vaga in enumerate(spots):

            # Cria uma máscara para a vaga atual (polígono)
            vaga_mask = np.zeros(frame.shape[:2], dtype=np.uint8)
            cv2.fillPoly(vaga_mask, [np.array(vaga, np.int32)], 255)

            # Calcula o desvio padrão de cor dentro da vaga
            std_dev = std_deviation_of_color(frame, vaga_mask)
            # print(f"Desvio padrão de cor para a vaga {idx+1}: {std_dev}")

            # Calcula a distância média de cor entre a imagem da vaga atual e a imagem de referência (vaga vazia)
            # distancia_euclidiana = avg_color_difference(frame, img_vazio, vaga_mask)
            # print(f"Distância média de cor para a vaga {idx+1}: {distancia_euclidiana}")

            # Calcula a quantidade de pixels diferentes entre o frame atual e a imagem de referência (vaga vazia) usando a máscara
            # qtd_pixel_dif = qtd_pixels_diferentes(frame, img_vazio, vaga_mask)
            # print(f"Quantidade de pixels diferentes para a vaga {idx+1}: {qtd_pixel_dif}")

            # Exibe na tela o status da vaga
            occupied = True if std_dev > 20 else False
            color = (0, 0, 255) if occupied else (0, 255, 0)
            cv2.putText(frame, f"Vaga {idx+1}: {'Ocupada' if occupied else 'Livre'}. {int(std_dev)}", (60, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
            
            # Desenha o polígono com o contorno da vaga no frame
            cv2.polylines(frame, [np.array(vaga, np.int32)], isClosed=True, color=(255, 255, 255), thickness=2)
        
        cv2.imshow('frame', frame)

        # Encerra o loop se a tecla 'q' for pressionada ou se a janela principal for fechada
        if cv2.waitKey(30) & 0xFF == ord('q') or cv2.getWindowProperty("frame", cv2.WND_PROP_VISIBLE) < 1:
            break
            
    # Libera os recursos
    cap.release()
    cv2.destroyAllWindows()

    return

track_parking_lot(video_path='media\\Estacionamento.mp4')
