# Classificador de Emoções em Texto

Este projeto aplica técnicas fundamentais de Processamento de Linguagem Natural (NLP) para construir um modelo capaz de identificar emoções expressas em frases curtas escritas em português.  
O foco está em desenvolver um fluxo simples e funcional que permita entender como modelos probabilísticos lidam com dados textuais reais.

---

## Propósito do Projeto

O objetivo principal é treinar um algoritmo de Machine Learning que consiga reconhecer emoções básicas (como positivo, negativo ou neutro) a partir de textos previamente limpos e rotulados.  
Com isso, o projeto solidifica conhecimentos essenciais:

- preparação e limpeza de texto para modelos de NLP;
- transformação de palavras em vetores numéricos;
- uso de algoritmos clássicos de classificação;
- avaliação do desempenho em dados reais;
- aplicação prática do modelo em novas frases.

Este classificador foi desenvolvido para ser simples, rápido e interpretável, permitindo observar de maneira clara como escolhas de pré-processamento influenciam o resultado final.

---

## Tecnologias Utilizadas

- **Python** para lógica e manipulação dos dados  
- **scikit-learn** como biblioteca principal de Machine Learning  
  - `CountVectorizer` para converter o texto em matriz de frequências  
  - `MultinomialNB` como modelo de classificação probabilística  
- **pandas** para leitura e organização do dataset  
- **joblib** para salvar o modelo treinado e o vetorizador  

---

## Como Executar o Projeto

### 1) Ativar o ambiente virtual

Windows (PowerShell):
```
.\venv\Scripts\Activate
```

### 2) Executar o script principal

```
python src/classificador.py

```

A execução exibirá:

- métricas de desempenho do modelo nos dados de teste;
- matriz de confusão para análise das classificações corretas e incorretas;
- predições das emoções para novas frases fornecidas ao final do script.

## Resultados

O modelo gerado é capaz de classificar emoções de maneira rápida e eficiente, apresentando um fluxo completo de Machine Learning:

- leitura do dataset limpo;
- vetorização das frases;
- treinamento do classificador;
- avaliação das métricas;
- predição em frases inéditas.