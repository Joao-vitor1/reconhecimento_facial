from cv2 import (imread, cvtColor, COLOR_BGR2RGB, VideoCapture, resize, rectangle, putText, FONT_HERSHEY_SIMPLEX,
                 imshow, waitKey, destroyAllWindows, namedWindow, imwrite, VideoWriter_fourcc, CAP_PROP_FRAME_WIDTH,
                 CAP_PROP_FRAME_HEIGHT,CAP_PROP_FPS, VideoWriter)
from os import (listdir, path, makedirs)
from multiprocessing import Pool
from numpy import argmin, argmax
from string import ascii_lowercase
from random import choice
import pandas as pd
from datetime import datetime, timedelta
from ultralytics import YOLO 
from deepface import DeepFace
import numpy as np 
from torch import tensor
from sklearn.metrics.pairwise import cosine_similarity
from pickle import load
import time
csv_directory = "registro_iteracoes.csv"


class DB:
    def __init__(self, path_arquivo = 'encoders_rostos.txt') -> None:
        
        self.encode_dict =  {}
        self.read_encodings('embeddings/'+path_arquivo)

    def read_encodings(self, path_arquivo):
        with open(path_arquivo, 'rb') as arquivo:  #rb : leitura binaria
            self.encode_dict = load(arquivo)
            print("Dicionario carregado com sucesso.")



class FaceRecognitionSystem:
    def __init__(self, database, threshold = 0.7):
        """
            Inicializa o sistema de reconhecimento facial.
        """
        self.dataBase = database
        self.threshold = threshold
        self.unknown_faces_seen_at = {}
        self.cap = VideoCapture(0)  # Inicializa a câmera
        self.cap.set(3, 640)  # Define a largura para 640 pixels (VGA)
        self.cap.set(4, 480)  # Define a altura para 480 pixels (VGA)
        self.model = YOLO('modelos/yolov8n-face.pt')
        namedWindow('Webcam')
   
    @staticmethod
    def compare_embeddings(prov_emb, list_embeddings):
        list_similarity = [] 
        for embedding in list_embeddings:
            similarity = cosine_similarity([prov_emb], [embedding])
            list_similarity.append(similarity[0][0])
        return list_similarity[argmax(list_similarity)]

    @staticmethod
    def save_img(directory, img, archive_name="sem_nome"):
        makedirs(directory, exist_ok=True)
        imwrite(f"{directory}/{archive_name}.jpg", img)

    @staticmethod
    def generate_unique_id(length=8):
        return ''.join(choice(ascii_lowercase) for _ in range(length))
        
       
    def find_faces(self, img_face):
        #...............................
        inicio = time.process_time()

        prov_emb = DeepFace.represent(img_face, model_name='Facenet', enforce_detection = False)[0]['embedding'] #melhorar velocidade
        prov_emb = tensor(prov_emb)
        
        fim = time.process_time()
        print('time DEEPFACE: ',fim - inicio)
        #...............................
        
        names = []
        S = []
        #...............................
        inicio = time.process_time()
        for name, list_embeddings in self.dataBase.encode_dict.items(): 
            similaridade = self.compare_embeddings(prov_emb, list_embeddings)
            print(name,': ', similaridade)
            
            if similaridade > self.threshold: 
                names.append(name)
                S.append(similaridade)

        fim = time.process_time()
        print('time SIMILALITY:',fim - inicio) 
        #...............................
            
        print('N:',names)
        print('S:',S)    
        return names, S

    def process_frame(self, img):
        access_granted = False
        nome = ""
        
        current_time = datetime.now()
        time_delay = 5

        img_reduzida = cvtColor(resize(img, (0, 0), None, 0.25, 0.25), COLOR_BGR2RGB)
        #...............................
        inicio = time.process_time()
        
        result = self.model(img_reduzida)
        
        fim = time.process_time()
        print('time YOLO:',fim - inicio)
        #...............................

        dados_faces = result[0].boxes
        for face in dados_faces:
            
            left,top,right,bottom = map(int, face.xyxy[0])
            img_face = img_reduzida[top:bottom, left:right]

            names,similaritys = self.find_faces(img_face)

            top *= 4
            right *= 4
            bottom *= 4
            left *= 4
   
            if len(names) > 0:
                match = argmax(similaritys)
                nome = names[match].upper()
                access_granted = True  # Acesso liberado se houver uma correspondência

                rectangle(img, (left, top), (right, bottom), (0, 255, 0), 2)
                putText(img, nome, (left, top - 10), FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
            
            else:
                # data = pd.read_csv(csv_directory)
                unique_id = self.generate_unique_id()

                last_key = list(self.unknown_faces_seen_at.keys())[-1] if self.unknown_faces_seen_at else None
                last_seen = self.unknown_faces_seen_at.get(last_key, None)
                
                unknown_face = img[top:bottom, left:right]
                rectangle(img, (left, top), (right, bottom), (0, 0, 255), 2)

                # if last_seen is None or (current_time - last_seen).total_seconds() >= time_delay:
                #     print("Rosto desconhecido encontrado")
                    
                #     data_hora = current_time.strftime("%Y-%m-%d %H:%M:%S")
                #     data.loc[len(data)] = [data_hora, "Não reconhecido", f'RD/{unique_id}']
                    
                #     self.save_img("RD", unknown_face, unique_id)
                #     self.unknown_faces_seen_at[unique_id] = current_time
                #     data.to_csv(csv_directory, index=False)
                #     print("Acesso registrado!")

        return access_granted, nome



    def run(self):
        timer = 5  # Defina o tempo de atraso (em segundos) conforme necessário
        last_access_time = datetime.now() - timedelta(seconds=timer)  # Inicialize com um tempo que permitirá a primeira ação

        while True:
            success, img = self.cap.read()

            current_time = datetime.now()
            time_elapsed = (current_time - last_access_time).total_seconds()
            access_granted, nome = self.process_frame(img)

            if access_granted:
                if time_elapsed >= timer:
                    print(f"Seja bem-vindo {nome}, acesso liberado!")
                    # data_hora = current_time.strftime("%Y-%m-%d %H:%M:%S")
                    # data = pd.read_csv(csv_directory)
                    # data.loc[len(data)] = [data_hora, "Reconhecido", f'DB/{nome.lower()}.jpg']
                    # data.to_csv(csv_directory, index=False)
                    # last_access_time = current_time  
                    print("Acesso registrado!")

            imshow('Webcam', img)

            if waitKey(1) == 27:
                break

        self.cap.release()
        destroyAllWindows()


    def read_video(self, video):

        self.cap.release()
        self.cap = VideoCapture('barbado_loiro.mp4')

        if not self.cap.isOpened():
          print("erro na leitura do video")
          

        fourcc = VideoWriter_fourcc(*'XVID') #codecs do video (mecanismo de codificacao)
        w = int(self.cap.get(CAP_PROP_FRAME_WIDTH)) #largura do video em pixels
        h = int(self.cap.get(CAP_PROP_FRAME_HEIGHT)) #argura do video em pixels
        fps = self.cap.get(CAP_PROP_FPS) #fps do videos em frames/seg


        #avi eh um dos formatos de videos 
        out = VideoWriter("video_saida.avi", fourcc, fps, (w, h)) #iniciando video de saida
        count = 0
        while self.cap.isOpened():
            success, img = self.cap.read()
            if not success:
                break

            __, __ = self.process_frame(img)

            out.write(img)
            count += 1
            print('frame:', count)

        self.cap.release()
        out.release()
        destroyAllWindows()


if __name__ == '__main__':
    limite_similaridade = 0.7
    myDatabase = DB('joao_emb.txt')
    myFaceRecognitionSystem = FaceRecognitionSystem(myDatabase, limite_similaridade)
    myFaceRecognitionSystem.run()