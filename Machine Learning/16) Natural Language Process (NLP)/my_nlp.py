"""
kütüphanesi -nltk
doğaldil işleme gibi birşey
"""
import pandas as pd
#%% data 
data = pd.read_csv(r"gender-classifier.csv",encoding="latin1" )
data = pd.concat([data.gender,data.description],axis=1)
data.dropna(axis=0,inplace=True)
data.gender = [1 if each == "female" else 0  for each in data.gender]  # kadın=1 erkek=0

#%% cleaning data
# regular expression RE   "[^a-zA-Z]" = a dan z ye kadar olanları bul
import re

first_description = data.description[4]
description = re.sub("[^a-zA-Z]"," ",first_description)  # a dan z ye ve A dan Z ye kadar olan harfleri bulma geri kalanları " " (space) ile degistir
description =description.lower() # hepsini kuçuk har yapar

#%% stopwords (irrelavent words) gereksiz kelimeler
import nltk #natural language tool kit
nltk.download("wordnet") #corpus diye bir klasöre infiriyor
from nltk.corpus import stopwords #sonra corpus klasöründen import ediyoruz

#description =description.split()     # sadece boşlukları ayırır 
#split yerine tokenizer kullanabiliriz
description =nltk.word_tokenize(description)
# split kullanırsak "shouldn't " gibi kelimeler "should" ve "not" diye ikiye ayrılmaz ama word_tokenize() kullanirsak ayrilir

#%% gereksiz kelimeleri çıkar
description = [word for word in description if not word in set(stopwords.words("english"))]

#%%lemmatazation (kelimelerin köklerini bulmak)
import nltk as nlp
lemma = nlp.WordNetLemmatizer()
description = [lemma.lemmatize(word) for word in description]

description = " ".join(description)

#%% bütün dataya üsttekileri uygulamak
description_list =[]
for description in data.description:
    description = re.sub("[^a-zA-Z]"," ",description)  # a dan z ye ve A dan Z ye kadar olan harfleri bulma geri kalanları " " (space) ile degistir
    description =description.lower() # hepsini kuçuk har yapar
    description =nltk.word_tokenize(description)
    description = [word for word in description if not word in set(stopwords.words("english"))]
    lemma = nlp.WordNetLemmatizer()
    description = [lemma.lemmatize(word) for word in description]
    description = " ".join(description)
    description_list.append(description) 
    
#%% bag of words 
from sklearn.feature_extraction.text import CountVectorizer #bag of words yaratmak için kullandığım method
max_features = 5000 #kaç kelime tutacak

count_vectorizer = CountVectorizer(max_features=max_features,stop_words="english")

sparce_matrix = count_vectorizer.fit_transform(description_list).toarray()
print("en sik kullanilan {} kelimeler: {}".format(max_features,count_vectorizer.get_feature_names_out()))

#%% train test split
y= data.iloc[:,0].values # male or female classes  
x= sparce_matrix
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.1,random_state=42)

#%%naive bayes 
from sklearn.naive_bayes import GaussianNB
nb = GaussianNB()
nb.fit(x_train, y_train)
y_pred = nb.predict(x_test)

print("score: ",nb.score(x_test,y_test))






