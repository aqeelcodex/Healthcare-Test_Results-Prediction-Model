import pandas as pd
import numpy as np

from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Embedding, Flatten, Dense, Concatenate
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping

# Importing the dataset and analysing
health = pd.read_csv("healthcare_dataset.csv")
health = health.iloc[:, ~health.columns.duplicated()]
print(health.head())
print(health.isnull().sum())
print(health.dtypes)

# Data preprocessing
health.columns = health.columns.str.replace(" ", "_")
# 1.Droping un_important columns
health.drop(columns= ["Name", "Date_of_Admission", "Discharge_Date"], inplace=True)

print("Numeric Columns")
num_columns = health.select_dtypes(exclude="object").columns
for col in num_columns:
    print(f"{col}: {health[col].nunique()}")

print("Categorial Columns")
cat_columns = health.select_dtypes(include="object").columns
for col in cat_columns:
    print(f"{col}: {health[col].nunique()}")
    
target = "Test_Results"
label_target = LabelEncoder()
health[target] = label_target.fit_transform(health[target])

cat_columns_label = ["Gender", "Blood_Type", "Medical_Condition", "Doctor", "Hospital", "Insurance_Provider", "Admission_Type", "Medication"]
encoders = {}
for col in cat_columns_label:
    label = LabelEncoder()
    health[col] = label.fit_transform(health[col])
    encoders[col] = label

final_num_columns = ["Age", "Gender", "Blood_Type", "Medical_Condition", "Insurance_Provider", "Billing_Amount", "Room_Number", "Admission_Type", "Medication"]
scale = StandardScaler()
health[final_num_columns] = scale.fit_transform(health[final_num_columns])
print("scaling is applied on numeric columns")

# Applying Embedding technique over the categorial columns with big unique values.

emb_columns = ["Doctor", "Hospital"]
vocab_size = {col: health[col].nunique() for col in emb_columns}
print(f"{health[col]}: {vocab_size}")

emb_inputs = []
emb_layer = []
for col in emb_columns:
    final_vocab_size = health[col].nunique() + 1
    emb_dim = min(50, final_vocab_size // 2)  ## Best Rule to find the emb_dim

    inputs = Input(shape= (1,), name= f"{col}_input")
    emb = Embedding(input_dim = final_vocab_size, output_dim = emb_dim, name= f"{col}_emb")(inputs)
    emb = Flatten()(emb)

    emb_inputs.append(inputs)
    emb_layer.append(emb)

x = Concatenate()(emb_layer)
num_columns = ["Age", "Gender", "Blood_Type", "Medical_Condition", "Insurance_Provider", "Billing_Amount", "Room_Number", "Admission_Type", "Medication"]
num_inputs = Input(shape= (len(num_columns), ), name= "numeric_columns")
x = Concatenate()([x, num_inputs])

x = Dense(128, activation= "relu")(x)
x = Dense(64, activation= "relu")(x)
output = Dense(len(label_target.classes_), activation= "softmax")(x)

all_inputs = emb_inputs + [num_inputs]

model = Model(inputs= all_inputs, outputs = output)
model.compile(optimizer= Adam(learning_rate= 0.001), loss= "sparse_categorical_crossentropy", metrics= ["accuracy"])
print("Congratulation! Model is designed via Artificial Neural Neurons")

X = health.drop(columns= target)
y = health[target]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
X_train_inputs = [X_train[col].values for col in emb_columns] + [X_train[num_columns].values]
X_test_inputs = [X_test[col].values for col in emb_columns] + [X_test[num_columns].values]
early_stopping = EarlyStopping(
    monitor="val_loss",     
    patience=5,               
    restore_best_weights=True   
)
model.fit(X_train_inputs, y_train, epochs= 50, batch_size= 32, validation_split= 0.2, callbacks= [early_stopping])
y_pred_train = model.predict(X_train_inputs)
y_pred_test = model.predict(X_test_inputs)

y_pred_train = np.argmax(y_pred_train, axis=1)
y_pred_test = np.argmax(y_pred_test, axis=1)

print("Train Accuracy:", accuracy_score(y_train, y_pred_train))
print("Test Accuracy:", accuracy_score(y_test, y_pred_test))