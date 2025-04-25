import streamlit as st
from PIL import Image
import ujson
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity as sklearn_cosine_similarity
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer, WordNetLemmatizer
import numpy as np


import nltk
nltk.download('stopwords') #palavras que comuns que não têm grande significado por isso não passam pela indexação
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('omw-1.4')

# Set up the NLTK components
stemmer = PorterStemmer()
lemmatizer = WordNetLemmatizer()
stop_words = stopwords.words('english')
tfidf = TfidfVectorizer()

# Load the data
f1 = 'publication_list_stemmed.json'
with open(f1, 'r') as f:
    pub_list_first_stem = ujson.load(f)
file_path = "publication_indexed_dictionary.json"
with open(file_path, 'r') as f:
    pub_index = ujson.load(f)
with open('author_list_stemmed.json', 'r') as f:
    author_list_first_stem = ujson.load(f)
with open('author_indexed_dictionary.json', 'r') as f:
    author_index = ujson.load(f)
with open('author_names.json', 'r') as f:
    author_name = ujson.load(f)
with open('pub_name.json', 'r') as f:
    pub_name = ujson.load(f)
with open('pub_url.json', 'r') as f:
    pub_url = ujson.load(f)
with open('pub_cu_author.json', 'r') as f:
    pub_cu_author = ujson.load(f)
with open('pub_date.json', 'r') as f:
    pub_date = ujson.load(f)
with open('abstracts.json', 'r') as f:
    abstract = ujson.load(f)
with open('grupo_indexed_dictionary.json', 'r') as f:
    grupo_index = ujson.load(f)
with open('2_resultados.json', 'r') as f:
    group_titles = ujson.load(f)


# Exercício 3 – Estratégias de Indexação: Lemas vs Stems


def indexação(token, indexing_strategy):
    #Inicializa variáveis para guardar os resultados
    stem_temp = ""
    lemmatize_temp = ""
    stem_word_file = []
    lemmatize_word_file = []
    word_list = word_tokenize(token) #divide a frase em palavras individuais

    #aplicar o stemmer e o lemmatizer
    for x in word_list:
        if x not in stop_words:
            stem_temp += stemmer.stem(x) + " "
            lemmatize_temp += lemmatizer.lemmatize(x) + " "

    # Remove espaços extra e guarda o resultado final nas listas criadas
    stem_word_file.append(stem_temp.strip())
    lemmatize_word_file.append(lemmatize_temp.strip())

    if indexing_strategy == "Lemming":
        return lemmatize_word_file
    else:
        return stem_word_file


# Exercício 4 – Operadores Lógicos AND, OR, NOT

def search_data(input_text, operator_val, search_type, indexing_strategy, rank_escolha_botao):
    output_data = {}
    input_text = input_text.lower().split()

    # ---------------------- OPERADOR OR ----------------------
    if operator_val == 'OR':
        pointer = set()
        for token in input_text:
            if len(input_text) < 2:
                st.warning("Please enter at least 2 words to apply the operator.")
                break

            final_word_file = indexação(token, indexing_strategy)

            if search_type == "publication" and pub_index.get(final_word_file[0]):
                pointer.update(pub_index.get(final_word_file[0]))
            elif search_type == "author" and author_index.get(final_word_file[0]):
                pointer.update(author_index.get(final_word_file[0]))
            elif search_type == "grupo" and grupo_index.get(final_word_file[0]):
                pointer.update(grupo_index.get(final_word_file[0]))

        if not pointer:
            output_data = {}
        else:
            temp_file = [
                pub_list_first_stem[j] if search_type == "publication"
                else author_list_first_stem[j] if search_type == "author"
                else group_titles[j]
                for j in pointer
            ]

            query = " ".join(input_text)
            if rank_escolha_botao.lower() == 'sklearn':
                temp_file = tfidf.fit_transform(temp_file)
                cosine_output = sklearn_cosine_similarity(temp_file, tfidf.transform([query])).flatten()
            else:
                query_tfidf_vector, document_tfidf_vectors = vetor_tfidf(query, temp_file)
                cosine_output = [similaridade_cosseno(query_tfidf_vector, doc_vector) for doc_vector in document_tfidf_vectors]

            for i, j in enumerate(pointer):
                output_data[j] = cosine_output[i]

    # ---------------------- OPERADOR AND ----------------------
    elif operator_val == 'AND':
        pointer = []
        match_word = []

        for token in input_text:
            if len(input_text) < 2:
                st.warning("Please enter at least 2 words to apply the operator.")
                break

            final_word_file = indexação(token, indexing_strategy)

            if search_type == "publication" and pub_index.get(final_word_file[0]):
                set1 = set(pub_index.get(final_word_file[0]))
            elif search_type == "author" and author_index.get(final_word_file[0]):
                set1 = set(author_index.get(final_word_file[0]))
            elif search_type == "grupo" and grupo_index.get(final_word_file[0]):
                set1 = set(grupo_index.get(final_word_file[0]))
            else:
                set1 = set()

            pointer.extend(list(set1))

            if not match_word:
                match_word = list(set1)
            else:
                match_word = list(set(match_word) & set1)

        if len(input_text) > 1:
            if not match_word:
                output_data = {}
            else:
                temp_file = [
                    pub_list_first_stem[j] if search_type == "publication"
                    else author_list_first_stem[j] if search_type == "author"
                    else group_titles[j]
                    for j in match_word
                ]

                query = " ".join(input_text)
                if rank_escolha_botao.lower() == 'sklearn':
                    temp_file = tfidf.fit_transform(temp_file)
                    cosine_output = sklearn_cosine_similarity(temp_file, tfidf.transform([query])).flatten()
                else:
                    query_tfidf_vector, document_tfidf_vectors = vetor_tfidf(query, temp_file)
                    cosine_output = [similaridade_cosseno(query_tfidf_vector, doc_vector) for doc_vector in document_tfidf_vectors]

                for i, j in enumerate(match_word):
                    output_data[j] = cosine_output[i]

    # ---------------------- OPERADOR NOT ----------------------
    elif operator_val == 'NOT':
        pointer = []
        for token in input_text:
            if len(input_text) < 2:
                st.warning("Please enter at least 2 words to apply the operator.")
                break

            final_word_file = indexação(token, indexing_strategy)

            if search_type == "publication" and pub_index.get(final_word_file[0]):
                pointer.extend(pub_index.get(final_word_file[0]))
            elif search_type == "author" and author_index.get(final_word_file[0]):
                pointer.extend(author_index.get(final_word_file[0]))
            elif search_type == "grupo" and grupo_index.get(final_word_file[0]):
                pointer.extend(grupo_index.get(final_word_file[0]))

        full_range = range(
            len(pub_list_first_stem) if search_type == "publication"
            else len(author_list_first_stem) if search_type == "author"
            else len(group_titles)
        )
        not_pointer = [i for i in full_range if i not in pointer]

        if not_pointer:
            temp_file = [
                pub_list_first_stem[j] if search_type == "publication"
                else author_list_first_stem[j] if search_type == "author"
                else group_titles[j]
                for j in not_pointer
            ]

            query = " ".join(input_text)
            if rank_escolha_botao.lower() == 'sklearn':
                temp_file = tfidf.fit_transform(temp_file)
                cosine_output = sklearn_cosine_similarity(temp_file, tfidf.transform([query])).flatten()
            else:
                query_tfidf_vector, document_tfidf_vectors = vetor_tfidf(query, temp_file)
                cosine_output = [similaridade_cosseno(query_tfidf_vector, doc_vector) for doc_vector in document_tfidf_vectors]

            for i, j in enumerate(not_pointer):
                output_data[j] = cosine_output[i]

    # ---------------------- OPERADOR INVÁLIDO ----------------------
    else:
        st.warning("Invalid operator value.")
        output_data = {}

    return output_data



# Exercicio 5

def tf(termo,documento):
    c = 0
    for elem in documento:
        if termo == elem:
            c += 1 
    return c

def idf(palavra,conj):
    num_docs = len(conj) + 1
    
    num_docs_c_palavra = 1 # em vez de, no fim, se somar 1, começa-se logo a contagem com 1
    for d in conj:
        if palavra in d:
            num_docs_c_palavra += 1
    
    idf = np.log10(num_docs/num_docs_c_palavra) + 1
    return idf

def tf_idf(termo,documento,conj):
    resultado = tf(termo,documento) * idf(termo,conj)
    return resultado


def similaridade_cosseno(query,doc):
    #print("query_tfidf shape:", query.shape)
    #print("document_tfidf shape:", document_tfidf.shape)
    produto_escalar = np.dot(query, doc) 
    norma_query = np.linalg.norm(query)
    norma_doc = np.linalg.norm(doc)
    res = produto_escalar / (norma_query * norma_doc)
    return res

def vetor_tfidf(query,docs):
    conj = []
    for doc in docs:
        doc = doc.split()
        conj.append(doc)
    palavras_unicas = list(set(query.split() + [p for doc in conj for p in doc]))
    tfidf_vetor_query = [tf_idf(pal,query.split(),conj) for pal in palavras_unicas]
    vetores_tfidf_docs = []
    for doc in conj:
        doc_vetor_tfidf = np.array([tf_idf(palav,doc,conj) for palav in palavras_unicas])
        vetores_tfidf_docs.append(doc_vetor_tfidf)
    return np.array(tfidf_vetor_query), np.array(vetores_tfidf_docs)

def app():

    image = Image.open('cire.png')
    st.image(image)

    st.markdown("<p style='text-align: center;'> Uncover the brilliance: Explore profiles, groundbreaking work, and cutting-edge research by the exceptional minds of Coventry University.</p>", unsafe_allow_html=True)

    input_text = st.text_input("Search research:", key="query_input")
    operator_val = st.radio(
        "Search Filters",
        ['AND', 'OR', 'NOT'],
        index=1,
        key="operator_input",
        horizontal=True,
    )

    search_type = st.radio(
        "Search in:",
        ['Publications', 'Authors', 'Group Specialized'],
        index=0,
        key="search_type_input",
        horizontal=True,
    )

    indexing_strategy = st.radio(
        "Indexing Strategies:",
        ['Lemming', 'Stemming'],
        index=0,
        key="indexing_strategies_input",
        horizontal=True,
    )

    rank_escolha_botao = st.radio(
        "Weighting Factor:",
        ['Sklearn', 'Manual'],
        index=0,
        key="rank_escolha_botao",
        horizontal=True,
    )

    if st.button("SEARCH"):
        print("Indexing strategy:", indexing_strategy)
        if search_type == "Publications":
            if operator_val == 'AND':
                output_data = search_data(input_text, 'AND', "publication", indexing_strategy, rank_escolha_botao)
            elif operator_val == 'OR':
                output_data = search_data(input_text, 'OR', "publication", indexing_strategy, rank_escolha_botao)
            elif operator_val == 'NOT':
                output_data = search_data(input_text, 'NOT', "publication", indexing_strategy, rank_escolha_botao)
            else:
                st.warning("Invalid operator value.")
                output_data = {}
        elif search_type == "Authors":
            if operator_val == 'AND':
                output_data = search_data(input_text, 'AND', "author", indexing_strategy, rank_escolha_botao)
            elif operator_val == 'OR':
                output_data = search_data(input_text, 'OR', "author", indexing_strategy, rank_escolha_botao)
            elif operator_val == 'NOT':
                output_data = search_data(input_text, 'NOT', "author", indexing_strategy, rank_escolha_botao)
            else:
                st.warning("Invalid operator value.")
                output_data = {}
        elif search_type == "Group Specialized":
            if operator_val == 'AND':
                output_data = search_data(input_text, 'AND', "grupo", indexing_strategy, rank_escolha_botao)
            elif operator_val == 'OR':
                output_data = search_data(input_text, 'OR', "grupo", indexing_strategy, rank_escolha_botao)
            elif operator_val == 'NOT':
                output_data = search_data(input_text, 'NOT', "grupo", indexing_strategy, rank_escolha_botao)
            else:
                st.warning("Invalid operator value.")
                output_data = {}
    else:
        output_data = {}

    show_results(output_data, search_type)



    st.markdown("<p style='text-align: center;'> Brought to you with ❤ by <a href='https://github.com/maladeep'>Mala Deep</a> | Data © Coventry University </p>", unsafe_allow_html=True)


def show_results(output_data, search_type):
    aa = 0
    rank_sorting = sorted(output_data.items(), key=lambda z: z[1], reverse=True)

    # Show the total number of research results
    st.info(f"Showing results for: {len(rank_sorting)}")

    # Show the cards
    N_cards_per_row = 3
    for n_row, (id_val, ranking) in enumerate(rank_sorting):
        i = n_row % N_cards_per_row
        if i == 0:
            st.write("---")
            cols = st.columns(N_cards_per_row, gap="large")
        # Draw the card
        with cols[n_row % N_cards_per_row]:
            if search_type == "Publications":
                url = pub_url[id_val]
                abstract_text = findAbstractByUrl(url, abstract)
                abstract_text = abstract_text[:200] + '...' if len(abstract_text) > 200 else abstract_text
                st.caption(f"{pub_date[id_val].strip()}")
                st.markdown(f"**{pub_cu_author[id_val].strip()}**")
                st.markdown(f"*{pub_name[id_val].strip()}*")
                st.markdown(f"**Abstract:** {abstract_text}")
                st.markdown(f"**{pub_url[id_val]}**")
                st.markdown(f"Ranking: {ranking:.2f}")

            elif search_type == "Authors":
                st.caption(f"{pub_date[id_val].strip()}")
                st.markdown(f"**{author_name[id_val].strip()}**")
                st.markdown(f"*{pub_name[id_val].strip()}*")
                st.markdown(f"**{pub_url[id_val]}**")
                st.markdown(f"Ranking: {ranking:.2f}")

            elif search_type == "Group Specialized":
                st.markdown(f"**Title:** {group_titles[id_val]}")  # <- Aqui mostras os títulos
                st.markdown(f"Ranking: {ranking:.2f}")

        aa += 1

    if aa == 0:
        st.info("No results found. Please try again.")
    else:
        st.info(f"Results shown for: {aa}")

def findAbstractByUrl(url, abstracts):
    for entry in abstracts:
        if entry["link"] == url:
            return entry["abstract"]
    return "Abstract not found"


if __name__ == '__main__':
    app()