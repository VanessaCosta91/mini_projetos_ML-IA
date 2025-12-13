from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import pandas as pd
import joblib

# Carrega dados
df = pd.read_csv('data/outputs/tw_limpo.csv')
print(df.head())

# Preparação dos dados
TEXTO_COL = 'Texto_limpo'
ROTULO_COL = 'Classificacao'

# Seleciona e define X e Y
X_raw = df[TEXTO_COL].astype(str)
y_raw = df[ROTULO_COL].astype(str).str.strip()

# Conta quantidade de texpo por rótulo
print('\nDistibuição de rótulos:\n', y_raw.value_counts())

# Separa o que é teste e o que é treino
X_train_raw, X_test_raw, y_train, y_test = train_test_split(X_raw, y_raw, test_size=0.2, random_state=42, stratify=y_raw)

# Vetorização
vetorizador = CountVectorizer(max_features=10000, ngram_range=(1,2))

# Treino
X_train = vetorizador.fit_transform(X_train_raw)
X_test = vetorizador.transform(X_test_raw)

modelo = MultinomialNB()
modelo.fit(X_train, y_train)

# Avaliação do modelo
y_pred = modelo.predict(X_test)

print('\nAcurácia:', accuracy_score(y_test, y_pred))
print('\nRelatório de classificação:\n', classification_report(y_test, y_pred))
print('\nMatriz de confusão:\n', confusion_matrix(y_test, y_pred))

# testando com novos textos

novos_tw = [
    'A policia conseguir dimininuir a taxa de roubos',
    'Estou triste com essa calamidade.',
    'Previsão de chuva para hoje'
]

X_novos = vetorizador.transform(novos_tw)
previsoes = modelo.predict(X_novos)

print('\nPredições para novos tw:')
print('\nPredições para novos tw:')
for tw, pred in zip(novos_tw, previsoes):
    print(f"- \"{tw}\": {pred}")


# salvar modelo
joblib.dump(vetorizador, 'modelos/vetorizador_count.pkl')
joblib.dump(modelo, 'modelos/modelo_multinomialnb.pkl')

print("Arquivos salvos em: modelos/")