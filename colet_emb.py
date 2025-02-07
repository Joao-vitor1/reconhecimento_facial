import coletor_embeddings as colet_emb


#ADICIONANDO EMBEEDINGS PROPRIAS EM UM ARQUIVO DE EMBEDDINGS

dict_emb = colet_emb.read_encodings('embeddings/joao_emb.txt')  #ler dicionário de embeding
coletor('DBE') #coleta imagens do rosto. (para ao precionar ESC)

print('selecione as melhores fotos antes de progedir') 
while 'Y' != input('se fotos já forma selecionadas digite [Y]'):
	pass
new_emb = colet_emb.make_embeddings('DBE', dict_emb)  #gera e salva novas embeddings no dicionário
colet_emb.write_encodings(new_emb, 'novo_emb.txt')  #cria novo arquivo com as embeddings
