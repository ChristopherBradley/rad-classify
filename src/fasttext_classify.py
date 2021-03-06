import time
import string
import pickle

import pandas as pd
import fasttext
import nltk
import yake

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score, f1_score

from path_variables import WOS5736_X, WOS5736_Y, fasttext_train, pickle_keywords, pretrained_vectors, results_file, test_classifications
from os.path import exists

SEED = 0

# 1. Load the data into a dataframe
print("Loading the data")
with open(WOS5736_X) as file:
    inputs = file.readlines()
with open(WOS5736_Y) as file:
    outputs = file.readlines()
inputs = [i.strip() for i in inputs] 
outputs = [o.strip() for o in outputs]
df = pd.DataFrame({"Abstract":inputs, "Keywords": inputs, "Label_ID":outputs})
start = time.time()

# 1.1 Preprocess the inputs. This is good for some quick results before using Yake. 
# But if using Yake then you get better results without this preprocessing.
preprocessing = False
if preprocessing:
    print(f"Started preprocessing at {time.ctime()}")

    nltk.download('stopwords')
    stop_words = set(nltk.corpus.stopwords.words('english'))

    df["Keywords"] = [a.lower() for a in df["Keywords"]]    # Lowercase
    df["Keywords"] = [a.translate(str.maketrans('','',string.punctuation)) for a in df["Keywords"]] # Remove punctuation
    df["Keywords"] = [" ".join([word for word in abstract.split() if word not in stop_words]) for abstract in df["Keywords"]]  # Remove stopwords
    
    # Experiments indicate no improvements from stemming or lemmatizing
    # nltk.download('omw-1.4')
    # tokens = [nltk.WordPunctTokenizer().tokenize(a) for a in df["Keywords"]]
    # df["Keywords"] = [" ".join(nltk.WordNetLemmatizer().lemmatize(word) for word in abstract) for abstract in tokens]  # Lemmatizing
    # df["Keywords"] = [" ".join(nltk.PorterStemmer().stem(word) for word in abstract) for abstract in tokens]  # Stemming

# 1.2 Extract the keywords and save with pickle so I don't have to worry about datatype conversions
# This takes a long time (~30 mins for WOS5736) and gives about a 2% accuracy boost (90% up from 88%) compared to just preprocessing
yake_keywords = True
def extract_keywords(text):
    """picking good parameters for yake"""
    keyword_extractor = yake.KeywordExtractor(n=1, top=len(str(text)))
    keywords = keyword_extractor.extract_keywords(text)
    return keywords

if yake_keywords:
    if not exists(pickle_keywords):
        keywords = []
        for i, abstract in enumerate(df["Keywords"]):
            if i%10 == 0:
                print(f"Extracted {i}/{len(df['Abstract'])} keywords, {time.ctime()}")
            abstract_keywords = extract_keywords(abstract)
            keywords.append(abstract_keywords)
        with open(pickle_keywords, 'wb') as file:
            pickle.dump(keywords, file)
        print(f"Took {time.time() - start} seconds to extract the keywords")

    with open(pickle_keywords, 'rb') as file:
        keywords = pickle.load(file)
    cutoff = 1
    df["Keywords"] = [" ".join(keyword[0] for keyword in abstract if keyword[1] < cutoff) for abstract in keywords]
    df["Keywords"] = [a.lower() for a in df["Keywords"]]    # Lowercasing after extracting keywords gives the best results

# 2. Format for fasttext
df_train, df_test = train_test_split(df, random_state=SEED)
fasttext_format = []
for index, row in df_train.iterrows():
    fasttext_row = "__label__" + row["Label_ID"] + " " + row["Keywords"] + "\n"
    fasttext_format.append(fasttext_row)
with open(fasttext_train, 'w') as file:
	file.writelines(fasttext_format)

# 3. Train the Model
print("Training the model. If using pretrained vectors, it may take a few mins before anything starts to happen.")
model = fasttext.train_supervised(
    input=fasttext_train,
    seed=SEED,
    epoch=100,
    lr=0.1,
    dim=100,    # dim needs to be 300 if using pretrained vectors
    wordNgrams=1,
    # pretrainedVectors=pretrained_vectors
    )
duration = time.time() - start

# 4. Evaluate the Results
predictions = [model.predict(a) for a in df_test["Keywords"]]
df_test["Prediction"] = [str(p[0][0][len("__label__"):]) for p in predictions]
df_test["Confidence"] = [p[1][0] for p in predictions]

df_confusion = pd.DataFrame(confusion_matrix(df_test["Prediction"], df_test['Label_ID']))

df_results = pd.Series({
    "Accuracy": accuracy_score(df_test["Prediction"], df_test['Label_ID']),
    "f1_micro": f1_score(df_test["Prediction"], df_test['Label_ID'], average='micro'),
    "f1_macro": f1_score(df_test["Prediction"], df_test['Label_ID'], average='macro'),
    "precision_micro": precision_score(df_test["Prediction"], df_test['Label_ID'], average='micro'),
    "precision_macro": precision_score(df_test["Prediction"], df_test['Label_ID'], average='macro'),
    "recall_micro": recall_score(df_test["Prediction"], df_test['Label_ID'], average='micro'),
    "recall_macro": recall_score(df_test["Prediction"], df_test['Label_ID'], average='macro')
})
print(df_results)
df_results.to_csv(results_file, header=False)
df_test.to_csv(test_classifications)