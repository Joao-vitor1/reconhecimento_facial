# Reconhecimento Facial
Um código simples e funcional para uma futura aplicação utilizada na InPacta (Incubadora) da  Universidade Federal do Rio Grande do Norte (UFRN).
Desenvolvido em Python com a biblioteca `OpenCV` ('cv2') para processamento de imagens e o módulo `face_recognition` para reconhecimento facial. 
O sistema pode monitorar uma webcam em tempo real, reconhecer rostos e salvar dados de acesso em um banco de dados.

Atualmente, a versão mais recente do código utiliza a biblioteca `Yolo` para a detecção de rostos, fazendo uso de um modelo pré-treinado no dataset
[celeba-dataset](https://www.kaggle.com/datasets/jessicali9530/celeba-dataset) e a biblioteca `DeepFace` para a geração de embeddings.

**Possiveis atualizações**
  
  Adicionar um codigo de alteração de imagens (data augumentation) para tornar os dados mais diversificados e fazer a rede lidar com situações que normalmente os dados     de testes normais não apresentariam.
  
# :pushpin: Tabela de conteúdos

- [Tecnologias](#computer-tecnologias)
- [Setup de Pacotes](#gear-setup-de-pacotes)
- [Como rodar](#tv-como-rodar)

# :computer: Tecnologias

- [x] Python
- [x] Face Recognition
- [x] OpenCV
- [x] Dlib
- [x] Cmake
- [x] Pandas
- [x] Yolo
- [x] DeepFace

# :gear: Setup de Pacotes

```shell
## Com Python 3 e Python PiP instalados, siga os passos abaixos para configurar o ambiente para rodar a aplicação.
## Tenha certeza que, no Windows, tenha o pacote do Visual Studio de desenvolvimento em C++ desktop instalados.
## OBS: Tenha certeza de estar na raiz do projeto antes de qualquer um dos passos a seguir!

### Ambiente e Pacotes

# Crie um ambiente virtual com o virtualenv para a aplicação e ative o ambiente:
$ python3 -m virtualenv venv
$ source venv/bin/activate

## OBS: Dependendo do shell que você estiver usando, a extensão do arquivo activate pode ser necessária de alteração.
## Ex: Terminal com Fish Shell:
$ source venv/bin/activate.fish

# Use o PiP para instalar os pacotes necessários localizados no requirements.txt:
$ pip install -r requirements.txt
```

# :tv: Como Rodar

```shell
## Para rodar a aplicação, apenas execute no diretório raiz da aplicação:
$ python3 main.py
```
# :bulb: Cadastrando rosto
Para que seu rosto seja reconhecido, é necessário que as embeddings que representam seu rosto estejam cadastradas no arquivo que será lido pela `main.py`. Uma pequena biblioteca `coletor_embeddings.py` foi criada para auxiliar o usuário nessa tarefa.

- Exemplo de código
```python
import coletor_embeddings as colet_emb
#ADICIONANDO EMBEEDINGS PROPRIAS EM UM ARQUIVO DE EMBEDDINGS

dict_emb = colet_emb.read_encodings('embeddings/joao_emb.txt')  #ler dicionário de embeding
coletor('DBE') #coleta imagens do rosto. (para ao precionar ESC)

print('selecione as melhores fotos antes de progedir') 			# Opicional
while 'Y' != input('se fotos já forma selecionadas digite [Y]'):	# Opicional
	pass

new_emb = colet_emb.make_embeddings('DBE', dict_emb)  #gera e salva novas embeddings no dicionário
colet_emb.write_encodings(new_emb, 'novo_emb.txt')  #cria novo arquivo com as embeddings

``` 

