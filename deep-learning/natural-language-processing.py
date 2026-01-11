import nltk
import os
nltk.download('rslp')
nltk.download('stopwords')
nltk.download('punkt')
import re
import pandas as pd 
from nltk.tokenize import word_tokenize 
from sklearn.feature_extraction.text import CountVectorizer
from nltk.tokenize import RegexpTokenizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn import metrics

# Importando dataset
# Obtém o diretório do script e constrói o caminho do arquivo
script_dir = os.path.dirname(os.path.abspath(__file__))
dataset_path = os.path.join(script_dir, "../dataset-test/dataset-nlp.csv")

if not os.path.exists(dataset_path):
    print(f"Erro: Arquivo 'dataset-nlp.csv' não encontrado em {script_dir}")
    print("Por favor, certifique-se de que o arquivo dataset.csv está no mesmo diretório do script.")
    print("\nO script requer um arquivo CSV com as seguintes colunas:")
    print("  - id (opcional)")
    print("  - text_pt (texto em português)")
    print("  - text_en (opcional)")
    print("  - sentiment (valores: 'pos' ou 'neg')")
    raise FileNotFoundError(f"Arquivo 'dataset.csv' não encontrado em {script_dir}")

df = pd.read_csv(dataset_path, sep="," , encoding="utf8")

df.head()

df.groupby('sentiment').count()

# Remove columns e create column
df.drop(columns=['id', 'text_en'], axis=1, inplace=True)
df['classification'] = df["sentiment"].replace(["neg", "pos"],[0, 1])

# Texto para minusculo
text_lower = [t.lower() for t in df['text_pt']]
df['text_pt'] = text_lower

df.head(5)

# funcao para remover brackets
def remove_brackets(column):
    for x in range(1,len(column)):
        return(re.sub('[\[\]]','',repr(column)))

# # %%time


stop_words = nltk.corpus.stopwords.words('portuguese')
stemmer = nltk.stem.RSLPStemmer()

# Trabalhar com stemmer e stopwords da base de treinamento/teste
# stemmer = andando, andei -> andar (simplificação)
# stop_words = remover a, o, de, da, do, etc. (palavras irrelevantes)

for x in range(0,len(df['text_pt'])):

    # Remover as stop words do texto
    word_tokens = word_tokenize(df['text_pt'][x]) 
    filtered_sentence = [w for w in word_tokens if not w in stop_words] 
    
    # Remover sufixos 
    line=[]
    text_tokenized = word_tokenize((remove_brackets(filtered_sentence)))
    line =  [stemmer.stem(word) for word in text_tokenized]
    df['text_pt'][x] = (remove_brackets(line))


# Regex para remover alguns valores do dataset  (simbolos, numeros...)
token = RegexpTokenizer(r'[a-zA-Z0-9]+')

# Cria o 'vetorizador' de acordo com os parametros abaixo
cv = CountVectorizer(lowercase=True,stop_words=None,ngram_range = (1,2),
                     tokenizer = token.tokenize)

# Matrixsparse da representação da coluna  text_pt
text_counts= cv.fit_transform(df['text_pt'])

# Vocabulario
# cv.vocabulary_

######### TESTANDO O MODELO #########

# Divindo no dataset em treino e teste
X_train, X_test, y_train, y_test = train_test_split(text_counts, df['classification'], 
                                                    test_size=0.34, random_state=1, 
                                                    shuffle=True)
# Criar modelo e treinar
clf = MultinomialNB().fit(X_train, y_train)

# Fazendo  predict do valor de X para teste de acuracidade
y_predicted= clf.predict(X_test)
print("MultinomialNB Accuracy:",metrics.accuracy_score(y_test, y_predicted).round(3))

##### FIXME TENTAR MELHORAR ESSA ACURACIA COM DEEP LEARNING

# Separa por paragrafos 
with open('texto_teste.txt', 'r') as file_teste:
    paragraph = file_teste.read().split('\n\n')

# Separa por frases
with open('./texto_teste.txt', 'r') as file_teste:
    phrase = file_teste.read().split('.')

    #Importar stemmer novamente
stemmer = nltk.stem.RSLPStemmer()

# Criar dataframe
df_result = pd.DataFrame()


# Fazer a tokanização, remocao de stop words e 
# transformar os dados para predict
neg,pos=0,0
for x in range(0,len(phrase)-1):

    # Texto tokenizado
    text_tokenized = word_tokenize(phrase[x])

    # Remove stop words do texto
    filtered_sentence = [w for w in text_tokenized if not w in stop_words] 

    # Cria stemmer do texto input
    line =  [stemmer.stem(word) for word in filtered_sentence]
    line = (remove_brackets(line))

    # Criar prediction para cada frase
    value_trans = cv.transform([line])
    predict_phrase = clf.predict(value_trans)

    # Contar por tipo de prediction (positivo e negativo)
    if predict_phrase==0:pos+=1
    else:neg+=1

# Salvar valores no dataframe
df_result['positive'] = [pos]
df_result['negative'] = [neg]


def generate_piechart(df_result):
    
    import matplotlib.pyplot as plt
    labels = df_result.columns.tolist()
    sizes = df_result.values.tolist()[0]
    color = ['lightskyblue', 'lightcoral']
    explode = (0.15, 0)

    fig1, ax1 = plt.subplots(figsize=(5,5))
    ax1.pie(sizes, labels=labels,  explode=explode,
            shadow=True, autopct='%1.1f%%',  startangle=140, colors=color)

    ax1.set_title('Sentiment Analysis by phrases - NLTK', fontsize=15)

    ax1.axis('equal')
    plt.show()
    print("Quantity by paragraph: {}".format(len(paragraph)))
    print("Quantity by phrases: {}".format(len(phrase)-1))
    print("Quantity by positives phrases: {}".format(df_result['positive']
                                                     .values.tolist()[0]))
    
    print("Quantity by negatives phrases: {}".format(df_result['negative']
                                                     .values.tolist()[0]))
    
# Gerar gráfico    
generate_piechart(df_result)