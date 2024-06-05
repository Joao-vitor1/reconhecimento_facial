from cv2 import (imread, cvtColor, COLOR_BGR2RGB, VideoCapture, resize, rectangle, putText, FONT_HERSHEY_SIMPLEX,
                 imshow, waitKey, destroyAllWindows, namedWindow, imwrite)
from face_recognition import (face_encodings, face_locations, compare_faces, face_distance)
from os import (listdir, path, makedirs, remove) #adicionado o remove
from multiprocessing import Pool
from numpy import argmin
from string import ascii_lowercase
from random import choice
import pandas as pd
from datetime import datetime, timedelta

csv_directory = "registro_iteracoes.csv"
data_hora = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

class FaceRecognitionSystem:
    def __init__(self):
        """
            Inicializa o sistema de reconhecimento facial.
        """
        self.unknown_faces_seen_at = {}
        print("Nome que sera usado para salvar um rosto novo")
        print("Caso seja desconhecido deixe em branco")
        self.newUser = input("insira um nome: ")
        self.cap = VideoCapture(0)  # Inicializa a câmera
        self.cap.set(3, 640)  # Define a largura para 640 pixels (VGA)
        self.cap.set(4, 480)  # Define a altura para 480 pixels (VGA)
        self.ids_criados = [] #salvando ids criados para evitar repetição
        namedWindow('Webcam')

    
    """ gerando um id unico para cada imagem"""
    def generate_unique_id(self,length=8):
        while True:
            unique_id = ''.join(choice(ascii_lowercase) for _ in range(length))

            if unique_id not in self.ids_criados:
                self.ids_criados.append(unique_id)
                return  unique_idv

    @staticmethod
    def find_faces(img):
        images = cvtColor(resize(img, (0, 0), None, 0.25, 0.25), COLOR_BGR2RGB)
        faces_cur_frame = face_locations(images)
        encode_cur_frame = face_encodings(images, faces_cur_frame)
        return list(zip(encode_cur_frame, faces_cur_frame))
    
    #@staticmethod
    def save_img(self, directory, img, archive_name="sem_nome"):
        makedirs(directory, exist_ok=True)
        imwrite(f"{directory}/{self.newUser + '-' + archive_name}.jpg", img)

    @staticmethod
    def remove_invalid(directory):
        
        for cl in listdir(directory):
            image = imread(f'{directory}/{cl}')
            encoding = face_encodings(cvtColor(image, COLOR_BGR2RGB))
            if encoding:
                print("OK")
            else:
                remove(f'{directory}/{cl}')
           

    def process_frame(self, img):
       
        current_time = datetime.now()
        time_delay = 3 #tempo abaixado em 2 segundos 
        salvo = False
        name = ""
        encodings_and_locations = self.find_faces(img)

        #!!!verificar o que acontece para duas pessoas tambem
        for encodeFace, faceLoc in encodings_and_locations:
            data = pd.read_csv(csv_directory)

            top, right, bottom, left = faceLoc
            top *= 4
            right *= 4
            bottom *= 4
            left *= 4

            last_key = list(self.unknown_faces_seen_at.keys())[-1] if self.unknown_faces_seen_at else None
            last_seen = self.unknown_faces_seen_at.get(last_key, None)
            
            unknown_face = img[top:bottom, left:right]
            rectangle(img, (left, top), (right, bottom), (0, 0, 255), 2)
            
            if last_seen is None or (current_time - last_seen).total_seconds() >= time_delay:
                unique_id = self.generate_unique_id()
                print("Rosto desconhecido encontrado")
                
                data_hora = current_time.strftime("%Y-%m-%d %H:%M:%S")
                #salvando a captura em uma tabela de histórico
                
                data.loc[len(data)] = [data_hora
                , "Não reconhecido", f"RD/{self.newUser + '-' +unique_id}"]
                #salvando imagem do rosto
                self.save_img("RD", unknown_face, unique_id)
                self.unknown_faces_seen_at[unique_id] = current_time
                data.to_csv(csv_directory, index=False)
                #avisando que imagem do usuario foi salvo
                print("usuario",f'{self.newUser}', " salvo!")


    def run(self):
        timer = 5  # Defina o tempo de atraso (em segundos) conforme necessário
        last_access_time = datetime.now() - timedelta(seconds=timer)  # Inicialize com um tempo que permitirá a primeira ação

        while True: 
            success, img = self.cap.read()

            current_time = datetime.now()
            time_elapsed = (current_time - last_access_time).total_seconds()
            self.process_frame(img)

            imshow('Webcam', img)

            if waitKey(1) & 0xFF == ord('q'):
                break

        self.remove_invalid('RD')
        self.cap.release()
        destroyAllWindows()

if __name__ == '__main__':
    myFaceRecognitionSystem = FaceRecognitionSystem()
    myFaceRecognitionSystem.run()
