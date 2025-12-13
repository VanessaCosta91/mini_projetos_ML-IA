import pandas as pd
import re

df = pd.read_csv('data/inputs/tw_pt.csv')

df = df[['Text', 'Classificacao']]

def limpar_texto(texto):

    texto = str(texto).lower() # converse todos caracteres para minusculo
    texto = re.sub(r'http\S+|www\S+|https\S+','', texto) # remove url
    texto = re.sub(r'@\w+', '', texto) #remove o @
    texto = re.sub(r'[^0-9a-zA-Záéíóúãõâêîôûçàèìòù ÁÉÍÓÚÃÕÂÊÎÔÛÇÀÈÌÒÙ ]','', texto) # remove caracteres especiais
    texto = re.sub(r'\s+', ' ', texto).strip() # padroniza espaços

    return texto


df['Texto_limpo'] = df['Text'].apply(limpar_texto)

df = df[['Texto_limpo', 'Classificacao']]
print(df.head())

df.to_csv('data/outputs/tw_limpo.csv', index=False)  