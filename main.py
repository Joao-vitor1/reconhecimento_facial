from cv2 import (imread, cvtColor, COLOR_BGR2RGB, VideoCapture, resize, rectangle, putText, FONT_HERSHEY_SIMPLEX,
                 imshow, waitKey, destroyAllWindows, namedWindow, imwrite)
from face_recognition import (face_encodings, face_locations, compare_faces, face_distance)
from os import (listdir, path, makedirs)
from multiprocessing import Pool
from numpy import argmin
from string import ascii_lowercase
from random import choice
import pandas as pd
from datetime import datetime, timedelta

csv_directory = "registro_iteracoes.csv"
data_hora = datetime.now().strftime("%Y-%m-%d %H:%M:%S")


class DB:
    """Classe responsável por cuidar do acesso ao banco de dados com as imagens"""
    directory = 'DB'

    def __init__(self) -> None:
        print("Iniciando o sistema... \nCarregando o Banco de Dados... ")
        self.images, self.names= [], []
        self.get_img_and_name_general()
        print("Banco carregado com sucesso... \nIniciando o encoding das imagens...")
        self.encode_list =  []
        self.find_encodings()
        print("Encoding terminado com sucesso... \nSistema iniciado com sucesso")
        print("Encodings:")

        """filtro de imagens que não conseguiram gerar encodes"""
        None_locations = []
        for position, value  in enumerate(self.encode_list):
            #obrigatorio uso do 'is'
            if value is None:
                print("tem valores None:", self.names[position])
                None_locations.insert(0,position)

        # for i in None_locations:
        #     self.names.pop(i)
        #     self.images.pop(i)
        #     self.encode_list.pop(i)

        
    def get_img_and_name_general(self) -> None:
        for cl in listdir(DB.directory):
            #transformando a imagem em um tensor e adiciona em self.images
            self.images.append(imread(f'{DB.directory}/{cl}'))
            self.names.append(cl.split('-')[0]) #pegando apenas o primeiro nome 

    def find_encodings(self) -> None:
        with Pool(processes=None) as pool:
             #intera sobre cada imagem a função encode_face
            self.encode_list = pool.map(self.encode_face, self.images)

    @staticmethod
    def encode_face(image) -> None:
        #obtendo codificações do rosto da pessoa
        encoding = face_encodings(cvtColor(image, COLOR_BGR2RGB))
        if encoding:
            return encoding[0]
        else:
            print(encoding)


class FaceRecognitionSystem:
    def __init__(self, database, distance_limit):
        """
            Inicializa o sistema de reconhecimento facial.
        """
        self.dataBase = database
        self.limite_distancia = distance_limit
        self.unknown_faces_seen_at = {}
        self.cap = VideoCapture(0)  # Inicializa a câmera
        self.cap.set(3, 640)  # Define a largura para 640 pixels (VGA)
        self.cap.set(4, 480)  # Define a altura para 480 pixels (VGA)

        namedWindow('Webcam')

    @staticmethod
    def generate_unique_id(length=8):
        #gera um id de 8 letras aleatorias
        return ''.join(choice(ascii_lowercase) for _ in range(length))

    @staticmethod
    def find_faces(img):
        #redimesiona e converte para RGB
        images = cvtColor(resize(img, (0, 0), None, 0.25, 0.25), COLOR_BGR2RGB)
        #localização do rosto
        faces_cur_frame = face_locations(images)
        #encoding do rosto
        encode_cur_frame = face_encodings(images, faces_cur_frame)
        return list(zip(encode_cur_frame, faces_cur_frame))
    
    #@staticmethod
    def save_img(self, directory, img, archive_name="sem_nome"):
        #cria diretorio caso não exista
        makedirs(directory, exist_ok=True)
        #salva imagem como jpg
        imwrite(f"{directory}/{archive_name}.jpg", img)

    def process_frame(self, img):
        access_granted = False
        nome = ""
        
        current_time = datetime.now()
        time_delay = 5
        #localização do rosto e enconding do frame
        encodings_and_locations = self.find_faces(img)

        for encodeFace, faceLoc in encodings_and_locations:
            #compara a face no frame (se houver) com todas do BD
            #matches é uma lista de booleanos com tamanho = número de rostos no BD
            matches = compare_faces(self.dataBase.encode_list, encodeFace, self.limite_distancia)
            #medida de semelhança entre os rostos
            distancia = face_distance(self.dataBase.encode_list, encodeFace)
            #obtendo a pessoa com maior semelhança
            match = argmin(distancia)
            top, right, bottom, left = faceLoc
            top *= 4
            right *= 4
            bottom *= 4
            left *= 4

            if matches[match]:
                nome = self.dataBase.names[match].upper()
                access_granted = True  # Acesso liberado se houver uma correspondência

            if distancia[match] <= self.limite_distancia:
                #desenha retangulo e nome da pessoa
                rectangle(img, (left, top), (right, bottom), (0, 255, 0), 2) 
                putText(img, nome, (left, top - 10), FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
            else:
                data = pd.read_csv(csv_directory)
                unique_id = self.generate_unique_id()
                #pegando o ultimo rosto (desconhecido) salvo
                last_key = list(self.unknown_faces_seen_at.keys())[-1] if self.unknown_faces_seen_at else None
                #pegando o horario do ultimo rosto salvo
                last_seen = self.unknown_faces_seen_at.get(last_key, None)
                #pegando imagem da pessoa desconhecida no momento (agora)
                unknown_face = img[top:bottom, left:right]
                rectangle(img, (left, top), (right, bottom), (0, 0, 255), 2)

                if last_seen is None or (current_time - last_seen).total_seconds() >= time_delay:
                    print("Rosto desconhecido encontrado")
                    
                    data_hora = current_time.strftime("%Y-%m-%d %H:%M:%S")
                    #adicionando no historico de interações
                    data.loc[len(data)] = [data_hora, "Não reconhecido", f'RD/{unique_id}']
                    
                    self.save_img("RD", unknown_face, unique_id)
                    #rosto salvo vira o ultimo rosto visto para proxima checagem
                    self.unknown_faces_seen_at[unique_id] = current_time
                    data.to_csv(csv_directory, index=False)
                    print("Acesso registrado!")


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
                #reconhece um rosto novo a cada 5 segundos
                if time_elapsed >= timer:
                    print(f"Seja bem-vindo {nome}, acesso liberado!")
                    #adicionando no historico de interações
                    data_hora = current_time.strftime("%Y-%m-%d %H:%M:%S")
                    data = pd.read_csv(csv_directory)
                    data.loc[len(data)] = [data_hora, "Reconhecido", f'DB/{nome.lower()}.jpg']
                    data.to_csv(csv_directory, index=False)
                    last_access_time = current_time  
                    print("Acesso registrado!")

            imshow('Webcam', img)

            if waitKey(1) & 0xFF == ord('q'):
                break

        self.cap.release()
        destroyAllWindows()


if __name__ == '__main__':
    limite_distancia = 0.4
    myDatabase = DB()
    myFaceRecognitionSystem = FaceRecognitionSystem(myDatabase, limite_distancia)
    myFaceRecognitionSystem.run()
