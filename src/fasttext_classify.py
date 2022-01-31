import time
import string

import pandas as pd
import fasttext
import nltk

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix

from path_variables import WOS5736_X, WOS5736_Y, fasttext_train

SEED = 0

# 1. Load the data into a dataframe
print("Loading the data")
with open(WOS5736_X) as file:
    inputs = file.readlines()
with open(WOS5736_Y) as file:
    outputs = file.readlines()

 # remove the newline characters
inputs = [i.strip() for i in inputs] 
outputs = [o.strip() for o in outputs]

df = pd.DataFrame({"Abstract":inputs, "Label_ID":outputs})
df_train, df_test = train_test_split(df, random_state=SEED)
start = time.time()

# 1.5 Preprocess the inputs
preprocess = False   # Experiments indicate no significant improvements from preprocessing
if preprocess:
    print(f"Started preprocessing at {time.ctime()}")

    nltk.download('stopwords')
    nltk.download('omw-1.4')
    stop_words = set(nltk.corpus.stopwords.words('english'))
    tokens = [nltk.WordPunctTokenizer().tokenize(a) for a in df["Abstract"]]

    df["Abstract_original"] = df["Abstract"]
    df["Abstract"] = [a.lower() for a in df["Abstract"]]    # Lowercase
    df["Abstract"] = [a.translate(str.maketrans('','',string.punctuation)) for a in df["Abstract"]] # Remove punctuation
    df["Abstract"] = [" ".join([word for word in abstract.split() if word not in stop_words]) for abstract in df["Abstract"]]  # Remove stopwords
    df["Abstract"] = [" ".join(nltk.WordNetLemmatizer().lemmatize(word) for word in abstract) for abstract in tokens]  # Lemmatizing
    df["Abstract"] = [" ".join(nltk.PorterStemmer().stem(word) for word in abstract) for abstract in tokens]  # Stemming

# 2. Format for fasttext
fasttext_format = []
for index, row in df_train.iterrows():
    fasttext_row = "__label__" + row["Label_ID"] + " " + row["Abstract"] + "\n"
    fasttext_format.append(fasttext_row)
with open(fasttext_train, 'w') as file:
	file.writelines(fasttext_format)

# 3. Train the Model
print("Training the model")
model = fasttext.train_supervised(
    input=fasttext_train,
    seed=SEED,
    epoch=100,
    lr=0.1,
    dim=100,
    wordNgrams=1
    )
duration = time.time() - start

# 4. Evaluate the Results
predictions = [model.predict(a) for a in df_test["Abstract"]]
predicted_labels = [str(p[0][0][len("__label__"):]) for p in predictions]
predicted_confidence = [p[1] for p in predictions]

df_confusion = pd.DataFrame(confusion_matrix(predicted_labels, df_test['Label_ID']))
accuracy = accuracy_score(predicted_labels, df_test['Label_ID'])
print(f"Accuracy: {accuracy}")
print(f"Training time: {duration}")
