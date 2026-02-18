from preprocessing import preprocess_data,clean_text
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score,classification_report,confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
import pickle 

train_data = pd.read_csv(r"C:\Users\ASUS\OneDrive\Documents\Project\train.csv")
test_data = pd.read_csv(r"C:\Users\ASUS\OneDrive\Documents\Project\test.csv")

y_train = train_data['sentiment']
y_test = test_data['sentiment']

tf,X_train,X_test = preprocess_data(train_data,test_data)

model = LogisticRegression(max_iter=10000,C=3,solver='liblinear',class_weight='balanced')
model.fit(X_train,y_train)

pred = model.predict(X_test)

cm = confusion_matrix(y_test,pred)

sns.heatmap(cm,annot=True)
plt.show()

print("Accuracy:",accuracy_score(y_test,pred))

X_new = ["The movie was released last Friday and has a runtime of two hours."]

X_new_cleaned = [clean_text(X_new[0])]

X_new_final = tf.transform(X_new_cleaned)

probabilities = model.predict_proba(X_new_final)

negative_prob = probabilities[0][0]
positive_prob = probabilities[0][1]

print(f"Negative: {negative_prob * 100:.2f}%")
print(f"Positive: {positive_prob * 100:.2f}%")

pickle.dump(model, open("model.pkl", "wb"))
pickle.dump(tf, open("vectorizer.pkl", "wb"))

print("Saved successfully!")