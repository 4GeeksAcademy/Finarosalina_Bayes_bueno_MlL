from sklearn.naive_bayes import MultinomialNB, GaussianNB, BernoulliNB
from sklearn.metrics import accuracy_score, classification_report
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
import pandas as pd

url = "https://raw.githubusercontent.com/4GeeksAcademy/naive-bayes-project-tutorial/main/playstore_reviews.csv"
df = pd.read_csv(url)

df.drop(columns=["package_name"], inplace=True)

df["review"] = df["review"].str.strip().str.lower()

X = df["review"]
y = df["polarity"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

vectorizer = CountVectorizer(stop_words='english')
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# convertir a array denso para Gaussiana
X_train_dense = X_train_vec.toarray()
X_test_dense = X_test_vec.toarray()

results = {}

mnb = MultinomialNB()
mnb.fit(X_train_vec, y_train)

y_pred_train_mnb = mnb.predict(X_train_vec)
results['MultinomialNB_train'] = {
    'Accuracy': accuracy_score(y_train, y_pred_train_mnb),
    'Classification Report': classification_report(y_train, y_pred_train_mnb)
}

y_pred_test_mnb = mnb.predict(X_test_vec)
results['MultinomialNB_test'] = {
    'Accuracy': accuracy_score(y_test, y_pred_test_mnb),
    'Classification Report': classification_report(y_test, y_pred_test_mnb)
}

gnb = GaussianNB()
gnb.fit(X_train_dense, y_train)

y_pred_train_gnb = gnb.predict(X_train_dense)
results['GaussianNB_train'] = {
    'Accuracy': accuracy_score(y_train, y_pred_train_gnb),
    'Classification Report': classification_report(y_train, y_pred_train_gnb)
}

y_pred_test_gnb = gnb.predict(X_test_dense)
results['GaussianNB_test'] = {
    'Accuracy': accuracy_score(y_test, y_pred_test_gnb),
    'Classification Report': classification_report(y_test, y_pred_test_gnb)
}

bernoulli = BernoulliNB()
bernoulli.fit(X_train_vec, y_train)

y_pred_train_bernoulli = bernoulli.predict(X_train_vec)
results['BernoulliNB_train'] = {
    'Accuracy': accuracy_score(y_train, y_pred_train_bernoulli),
    'Classification Report': classification_report(y_train, y_pred_train_bernoulli)
}


y_pred_test_bernoulli = bernoulli.predict(X_test_vec)
results['BernoulliNB_test'] = {
    'Accuracy': accuracy_score(y_test, y_pred_test_bernoulli),
    'Classification Report': classification_report(y_test, y_pred_test_bernoulli)
}

for model in results:
    print(f"\nResultados para {model}:")
    print(f"Accuracy: {results[model]['Accuracy']}")
    print(f"Clasificación:\n{results[model]['Classification Report']}")


from sklearn.ensemble import VotingClassifier, RandomForestClassifier

rf = RandomForestClassifier(random_state=42)

#  Crear el VotingClassifier con ambos modelos
voting_clf = VotingClassifier(estimators=[('mnb', mnb), ('rf', rf)], voting='soft')

#  Entrenar el VotingClassifier
voting_clf.fit(X_train_vec, y_train)

# Evaluar el modelo en el conjunto de prueba
y_pred_test_voting = voting_clf.predict(X_test_vec)
print("\nResultados para VotingClassifier - Test:")
print(f"Accuracy: {accuracy_score(y_test, y_pred_test_voting)}")
print(f"Clasificación:\n{classification_report(y_test, y_pred_test_voting)}")

# Evaluación en el conjunto de entrenamiento
y_pred_train_voting = voting_clf.predict(X_train_vec)
print("\nResultados para VotingClassifier - Train:")
print(f"Accuracy: {accuracy_score(y_train, y_pred_train_voting)}")
print(f"Clasificación:\n{classification_report(y_train, y_pred_train_voting)}")


from sklearn.model_selection import RandomizedSearchCV

# 7. Optimización de MultinomialNB con RandomizedSearchCV
param_dist = {
    'alpha': [0.1, 0.5, 1.0, 2.0, 5.0], 
}

random_search_mnb = RandomizedSearchCV(mnb, param_distributions=param_dist, n_iter=10, cv=3, verbose=2, random_state=42, n_jobs=-1)

random_search_mnb.fit(X_train_vec, y_train)

print("Mejores parámetros encontrados para MultinomialNB:", random_search_mnb.best_params_)

best_mnb = random_search_mnb.best_estimator_

y_pred_mnb_test = best_mnb.predict(X_test_vec)
print("\nResultados para MultinomialNB optimizado - Test:")
print(f"Accuracy: {accuracy_score(y_test, y_pred_mnb_test)}")
print(f"Clasificación:\n{classification_report(y_test, y_pred_mnb_test)}")

y_pred_mnb_train = best_mnb.predict(X_train_vec)
print("\nResultados para MultinomialNB optimizado - Train:")
print(f"Accuracy: {accuracy_score(y_train, y_pred_mnb_train)}")
print(f"Clasificación:\n{classification_report(y_train, y_pred_mnb_train)}")
import pickle  

voting_clf_opt = VotingClassifier(estimators=[('mnb', best_mnb), ('rf', rf)], voting='soft')
voting_clf_opt.fit(X_train_vec, y_train)

with open("/workspaces/Finarosalina_Bayes_bueno_MlL/models/modelo_voting_classifier_opt.pkl", "wb") as f:
    pickle.dump(voting_clf_opt, f)

import xgboost as xgb
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
import pandas as pd

url = "https://raw.githubusercontent.com/4GeeksAcademy/naive-bayes-project-tutorial/main/playstore_reviews.csv"
df = pd.read_csv(url)


df = df.drop(columns=['package_name'])

df["review"] = df["review"].str.strip().str.lower()

X = df["review"]
y = df["polarity"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

vectorizer = CountVectorizer(stop_words='english')
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)


X_train_dense = X_train_vec.toarray()
X_test_dense = X_test_vec.toarray()


xgb_model = xgb.XGBClassifier(random_state=42)
xgb_model.fit(X_train_dense, y_train)

y_pred_test_xgb = xgb_model.predict(X_test_dense)
print("\nResultados para XGBoost - Test:")
print(f"Accuracy: {accuracy_score(y_test, y_pred_test_xgb)}")
print(f"Clasificación:\n{classification_report(y_test, y_pred_test_xgb)}")


y_pred_train_xgb = xgb_model.predict(X_train_dense)
print("\nResultados para XGBoost - Train:")
print(f"Accuracy: {accuracy_score(y_train, y_pred_train_xgb)}")
print(f"Clasificación:\n{classification_report(y_train, y_pred_train_xgb)}")

from sklearn.model_selection import GridSearchCV
import xgboost as xgb

param_grid = {
    'n_estimators': [100, 200, 300],
    'learning_rate': [0.01, 0.1, 0.2],
    'max_depth': [3, 5, 7],
    'subsample': [0.8, 1.0],
    'colsample_bytree': [0.8, 1.0]
}


xgb_model = xgb.XGBClassifier(random_state=42)

# Ajuste de hiperparámetros con GridSearchCV
grid_search = GridSearchCV(estimator=xgb_model, param_grid=param_grid, cv=3, n_jobs=-1, verbose=2)
grid_search.fit(X_train_dense, y_train)


print("Mejores parámetros encontrados para XGBoost:", grid_search.best_params_)


best_xgb = grid_search.best_estimator_

y_pred_test_xgb = best_xgb.predict(X_test_dense)
print("\nResultados para XGBoost optimizado - Test:")
print(f"Accuracy: {accuracy_score(y_test, y_pred_test_xgb)}")
print(f"Clasificación:\n{classification_report(y_test, y_pred_test_xgb)}")


y_pred_train_xgb = best_xgb.predict(X_train_dense)
print("\nResultados para XGBoost optimizado - Train:")
print(f"Accuracy: {accuracy_score(y_train, y_pred_train_xgb)}")
print(f"Clasificación:\n{classification_report(y_train, y_pred_train_xgb)}")

import numpy as np

# Guardar X_train_dense y X_test_dense como archivos CSV
np.savetxt('/workspaces/Finarosalina_Bayes_bueno_MlL/data/processed/X_train.csv', X_train_dense, delimiter=',')
np.savetxt('/workspaces/Finarosalina_Bayes_bueno_MlL/data/processed/X_test.csv', X_test_dense, delimiter=',')

np.savetxt('/workspaces/Finarosalina_Bayes_bueno_MlL/data/processed/y_train.csv', y_train, delimiter=',')
np.savetxt('/workspaces/Finarosalina_Bayes_bueno_MlL/data/processed/y_test.csv', y_test, delimiter=',')

import nbformat

# Cargar el archivo .ipynb
notebook_path = '/workspaces/Finarosalina_Bayes_bueno_MlL/src/explore.ipynb'
with open(notebook_path, 'r') as f:
    notebook_content = nbformat.read(f, as_version=4)

# Extraer las celdas de código
code_cells = [cell['source'] for cell in notebook_content.cells if cell.cell_type == 'code']

# Guardar el código en un archivo .py
output_path = '/workspaces/Finarosalina_Bayes_bueno_MlL/src/app.py'
with open(output_path, 'w') as f:
    for code in code_cells:
        f.write(code + '\n')

