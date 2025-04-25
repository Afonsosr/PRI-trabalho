import ujson
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import nltk

# Baixar os pacotes necessários
nltk.download('punkt')
nltk.download('stopwords')

# Inicializar componentes
stemmer = PorterStemmer()
stop_words = set(stopwords.words('english'))

# Ler os títulos já tratados
with open('2_resultados.json', 'r') as f:
    group_titles = ujson.load(f)

# Criar o índice
grupo_index = {}

for idx, titulo in enumerate(group_titles):
    palavras = word_tokenize(titulo)
    for palavra in palavras:
        palavra = palavra.lower()
        if palavra not in stop_words:
            palavra_stem = stemmer.stem(palavra)
            if palavra_stem not in grupo_index:
                grupo_index[palavra_stem] = []
            grupo_index[palavra_stem].append(idx)

# Guardar o índice
with open('grupo_indexed_dictionary.json', 'w') as f:
    ujson.dump(grupo_index, f)

print("Índice do grupo especializado criado com sucesso!")
