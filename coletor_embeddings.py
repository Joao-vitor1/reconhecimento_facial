from cv2 import (imread, cvtColor, COLOR_BGR2RGB, VideoCapture, resize, rectangle, putText, FONT_HERSHEY_SIMPLEX,
                 imshow, waitKey, destroyAllWindows, namedWindow, imwrite, CAP_PROP_FPS)
from os import (listdir, path, makedirs)
from random import choice
from ultralytics import YOLO 
from deepface import DeepFace
import cv2
import numpy as np 
import torch
import pickle
from datetime import datetime, timedelta
from string import ascii_lowercase


def read_encodings(path_arquivo = 'encoders_rostos.txt'):
'''
	Lê um arquivo de embeddings (path_arquivo) e retorna ele no formato de dicionário.
'''
    with open(path_arquivo, 'rb') as arquivo:  
        encode_dict = pickle.load(arquivo)
        print("Dicionario carregado com sucesso.")

    return encode_dict

def write_encodings(dict_emb, path_arquivo = 'teste_encoders.txt'):
'''
	Armazena uma estrutura do tipo dicionário em um arquivo .txt.
'''
    with open(path_arquivo, 'wb') as arquivo:  
        pickle.dump(dict_emb, arquivo)
        print("Dicionario salvo com sucesso.")

def criar_nome(dict_emb):
'''
	Função auxiliar de 'make_embeddings'.
	Verifica e cadastra um nome para as novas embeddings.
'''
	name = input('digite um nome: ')

	if name in dict_emb.keys():
		if 'Y' != input('nome ja existe no arquivo, deseja adicionar mais embeddings a ele? [Y/N]').upper():
			name =  input('digite  um novo nome ')
	
	dict_emb[name] = []
	return dict_emb, name


def coletor(directory ='DBE'):
'''
	Coleta a cada 2 segundos fotos do rosto de uma pessoa a partir web can.
	A coleta só será realizada caso seja detectado apenas uma pessoa.
	As fotos serão armazenas no diretório passado no atributo 'directory'.
'''
	model = YOLO('modelos/yolov8n-face.pt')
	cap = cv2.VideoCapture(0)

	if not cap.isOpened():
		print("erro na leitura do video") 

	fps = cap.get(cv2.CAP_PROP_FPS)
	
	count = 0
	last_access_time = datetime.now()
	while cap.isOpened():
		success, img = cap.read()
		if not success:
		    break

		faces = model(img)[0].boxes
		num_faces = len(faces.cls)

		for face in faces:
			x1,y1,x2,y2 = map(int, face.xyxy[0])

			if num_faces > 1:
				print('mais de uma pessoa detectada, permitido apenas uma pessoa por vez')
				rectangle(img, (x1, y1), (x2, y2), (0, 0, 255), 2)
			else:
				rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)

		current_time = datetime.now()
		time_elapsed = (current_time - last_access_time).total_seconds()
		if time_elapsed > 2 and num_faces == 1:
			last_access_time = current_time
			count +=1
			print('numero de imagens captadas:',(count))

			img_face = img[y1:y2,x1:x2]
			name = generate_unique_id()
			save_img(directory, img_face, name)

		imshow('Webcam', img)

		if cv2.waitKey(1) == 27:
			break

	cap.release()
	cv2.destroyAllWindows()


def generate_embedding(img_face):
'''
	Função auxiliar de make_embeddings.
	Recebe uma imagem e retorna um tensor de embeddings.
'''
	prov_emb = DeepFace.represent(img_face, model_name='Facenet', enforce_detection = False)[0]['embedding'] 
	return torch.tensor(prov_emb)



def generate_unique_id(length=8):
'''
	Função auxiliar de 'coletor' e 'coletor_videos'.
	Gera um id único para a imagem á ser armazenada.

'''
    return ''.join(choice(ascii_lowercase) for _ in range(length))




def save_img(directory, img, archive_name="sem_nome"):
'''
	Função auxiliar de 'coletor' e 'coletor_videos'.
	Salva a imagem de rostos no diretório especificado por 'directory'.

'''
    makedirs(directory, exist_ok=True)
    imwrite(f"{directory}/{archive_name}.jpg", img)



def coletor_videos(directory ='DBE', video):
'''
	A cada 1 segundo coleta a imagem de todas as faces detectadas no video 
	especificado no atributo 'video'.
	As imagens coletadas são salvas no diretório especificado em 'directory'.
'''
	model = YOLO('yolov8n-face.pt')
	cap = VideoCapture(video)
	fps = cap.get(CAP_PROP_FPS) 

	if not cap.isOpened():
		print("erro na leitura do video")

	count_frame = 0
	while cap.isOpened():
		success, img = cap.read()
		
		if not success:
		    break

		if count_frame % fps == 0:
			faces = model(img)[0].boxes
			num_faces = len(faces.cls)
			
			for face in faces:
				x1,y1,x2,y2 = map(int, face.xyxy[0])
				img_face = img[y1:y2,x1:x2]
				name = generate_unique_id()
				save_img(directory, img_face, name)
		
		count_frame += 1

	

def make_embeddings(directory, dict_emb = {}):
'''
	Gera embeddings de uma pessoa a partir de um diretório.

	atributos
		diretory: Diretório das imagens que gerarão os embeddings de uma única pessoa
		dict_emb: Dicionário com as embeddings já cadastradas. Esse dicionário pode ser 
				  tanto um vazio como um geradado a partir de um arquivo já existente através
				  da função read_encodings.
		
	A função 'make_embeddings' concatena as novas embeddings geradas com as presentes
	na variável dict_emb.
'''	
	dict_emb, name = criar_nome(dict_emb)
	for name_img in listdir(directory):
		if name_img.endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff')):
			img = imread(path.join(directory, name_img))
			dict_emb[name].append(generate_embedding(img))
	return dict_emb
	