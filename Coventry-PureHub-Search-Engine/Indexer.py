import ujson
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
nltk.download('punkt_tab')
# Preprocessing data before indexing
with open('scraper_results.json', 'r', encoding='utf-8') as doc:
    scraper_results = ujson.load(doc)  # Já carrega como objeto Python

# Initialize empty lists to store publication name, URL, author, and date
pubName = []
pubURL = []
pubCUAuthor = []
pubDate = []
pubabstract = []

# Get the length of the scraper_results (number of publications)
array_length = len(scraper_results)
print(array_length)

# Separate name, url, date, author and abstract into different files
for item in scraper_results:
    pubName.append(item["name"])
    pubURL.append(item["pub_url"])
    pubCUAuthor.append(item["cu_author"])
    pubDate.append(item["date"])
    pubabstract.append(item["abstract"])

# Save the separated data into JSON files
with open('pub_name.json', 'w', encoding='utf-8') as f:
    ujson.dump(pubName, f, ensure_ascii=False, indent=4)

with open('pub_url.json', 'w', encoding='utf-8') as f:
    ujson.dump(pubURL, f, ensure_ascii=False, indent=4)

with open('pub_cu_author.json', 'w', encoding='utf-8') as f:
    ujson.dump(pubCUAuthor, f, ensure_ascii=False, indent=4)

with open('pub_date.json', 'w', encoding='utf-8') as f:
    ujson.dump(pubDate, f, ensure_ascii=False, indent=4)

with open('pub_abstract.json', 'w', encoding='utf-8') as f:
    ujson.dump(pubabstract, f, ensure_ascii=False, indent=4)


with open('pub_name.json', 'r', encoding='utf-8') as f:
    publication = f.read()

#Load JSON File


#Downloading libraries to use its methods
nltk.download('stopwords')
nltk.download('punkt')

#Predefined stopwords in nltk are used
stop_words = stopwords.words('english')
stemmer = PorterStemmer()
pub_list_first_stem = []
pub_list = []
pub_list_wo_sc = []
print(len(pubName))

for file in pubName:
    #Splitting strings to tokens(words)
    words = word_tokenize(file)
    stem_word = ""
    for i in words:
        if i.lower() not in stop_words:
            stem_word += stemmer.stem(i) + " "
    pub_list_first_stem.append(stem_word)
    pub_list.append(file)

#Removing all below characters
special_characters = '''!()-—[]{};:'"\, <>./?@#$%^&*_~0123456789+=’‘'''
for file in pub_list:
    word_wo_sc = ""
    if len(file.split()) ==1 : pub_list_wo_sc.append(file)
    else:
        for a in file:
            if a in special_characters:
                word_wo_sc += ' '
            else:
                word_wo_sc += a
        #print(word_wo_sc)
        pub_list_wo_sc.append(word_wo_sc)

#Stemming Process
pub_list_stem_wo_sw = []
for name in pub_list_wo_sc:
    words = word_tokenize(name)
    stem_word = ""
    for a in words:
        if a.lower() not in stop_words:
            stem_word += stemmer.stem(a) + ' '
    pub_list_stem_wo_sw.append(stem_word.lower())

data_dict = {} #Inverted Index holder

# Indexing process
for a in range(len(pub_list_stem_wo_sw)):
    for b in pub_list_stem_wo_sw[a].split():
        if b not in data_dict:
             data_dict[b] = [a]
        else:
            data_dict[b].append(a)

# printing the lenght
print(len(pub_list_wo_sc))
print(len(pub_list_stem_wo_sw))
print(len(pub_list_first_stem))
print(len(pub_list))

# with open('publication_list.json', 'w') as f:
#     ujson.dump(pub_list, f)

with open('publication_list_stemmed.json', 'w') as f:
    ujson.dump(pub_list_first_stem, f)

with open('publication_indexed_dictionary.json', 'w') as f:
    ujson.dump(data_dict, f)
