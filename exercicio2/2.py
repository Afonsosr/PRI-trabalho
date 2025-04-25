import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
import ujson
import re

# Baixar as bibliotecas necessárias
nltk.download('stopwords')
nltk.download('punkt')

lista_aux = []

stemmer = PorterStemmer()
stop_words = set(stopwords.words('english'))

f = open('saudeTitulos.json', 'r', encoding = "utf-8")
titulos = ujson.load(f)
titulo_limpo = ""
caracteres_especiais = '''!()-—[]{};:0123456789'"\,#$%^&<>./?@*_~+=’‘'''

for titulo in titulos:
    titulo_limpo = ''
    for letra in titulo:
        if letra in caracteres_especiais:
            titulo_limpo += ' '
        else:
            titulo_limpo += letra   

    palavras = word_tokenize(titulo_limpo)

    stemmed_words = []
    for palavra in palavras:
        if palavra.lower() not in stop_words:
            stemmed_words.append(stemmer.stem(palavra))

    titulo_limpo = ' '.join(stemmed_words)

    lista_aux.append(''.join(titulo_limpo))

    # Salvar a lista de palavras com stemming
with open('2_resultados.json', 'w') as f:
    f.write('[')
    for i in range(0,len(lista_aux)-2):
        f.write(ujson.dumps(lista_aux[i]) + ',\n') 
    f.write(ujson.dumps(lista_aux[i+1]) +']')
