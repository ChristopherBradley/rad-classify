import numpy as np
import pandas as pd
import fasttext
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix

from path_variables import WOS5736_X, WOS5736_Y, fasttext_train

seed = 0
# np.random.seed(seed)

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
df_train, df_test = train_test_split(df, random_state=seed)

# 2. Format for fasttext
fasttext_format = []
for index, row in df_train.iterrows():
    fasttext_row = "__label__" + row["Label_ID"] + " " + row["Abstract"] + "\n"
    fasttext_format.append(fasttext_row)
with open(fasttext_train, 'w') as file:
	file.writelines(fasttext_format)

# 3. Train the Model
print("Training the model")
model = fasttext.train_supervised(input=fasttext_train)

# 4. Evaluate the Results
predictions = [model.predict(a) for a in df_test["Abstract"]]
predicted_labels = [str(p[0][0][len("__label__"):]) for p in predictions]
predicted_confidence = [p[1] for p in predictions]

df_confusion = pd.DataFrame(confusion_matrix(predicted_labels, df_test['Label_ID']))
accuracy = accuracy_score(predicted_labels, df_test['Label_ID'])
print(f"Accuracy: {accuracy}")
