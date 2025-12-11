from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB

frases = [
    'Edorei.',
    'Não recomendo.',
    'Pessimo atendimento',
    'Muito bom, pode confiar',
    'Horrível, odiei',
    'Estou muito satisfeito com a compra',
    'Vou voltar com certeza',
    'Não prestou, veio com defeito',
    'Nunca mais vou comprar',
    'Vou comprar novamente',
    'Amei o produto'
]

rotulos = [1, 0, 0, 1, 0, 1, 1, 0, 0, 1,1]

vetorizador = CountVectorizer()
X = vetorizador.fit_transform(frases)

# cria e treina o modelo
modelo = MultinomialNB()
modelo.fit(X, rotulos)

# Teste
novas_frases = [
    'Amei',
    'Odiei',
    'Pode comprar sem medo',
    'Não vá'
]

X_teste = vetorizador.transform(novas_frases)
previsoes = modelo.predict(X_teste)

for i, frase in enumerate(novas_frases):
    if previsoes[i] == 1:
        print(frase, "é uma frase positiva")
    else:
        print(frase, "é uma frase negativa")