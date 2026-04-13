**Qualidade de Software**<br />
<br />
*Pontifícia Universidade Católica do Rio de Janeiro*<br />
<br />
*Curso de Engenharia de Software*<br />
<br />
*Aluno: Vinicius do Carmo Brito*<br />
<br />
---


# Descubra a posição do jogador

Este projeto utiliza técnicas de Machine Learning para prever a posição de um jogador de futebol (Zagueiro, Meio-campista ou Atacante) com base em atributos técnicos.

A solução inclui treinamento de modelos, avaliação de desempenho, testes automatizados e uma aplicação web construída com Flask para realizar predições.


---

## Funcionalidades

* Classificação de jogadores por posição
* Treinamento automático com seleção do melhor modelo
* Avaliação com métricas de desempenho
* Interface web para predição
* Testes automatizados para validação do modelo

---

## Tecnologias Utilizadas

* Python
* Scikit-learn
* Pandas
* NumPy
* Flask
* Joblib
* Pytest

---

## Estrutura do Projeto

```id="f9j2la"
.
├── artifacts/
│   └── model.joblib
├── src/
│   ├── train_model.py
│   └── test_model.py
├── app/
│   ├── app.py
├── requirements.txt
```

---

## Como o Modelo Funciona

O modelo utiliza os seguintes atributos:

* SprintSpeed
* Finishing
* ShortPassing
* Vision
* Marking
* StandingTackle

Essas variáveis são utilizadas para prever uma das seguintes classes:

* Zagueiro
* Meio-campista
* Atacante

O dataset é carregado de uma fonte pública 

---

## Instalação

```bash id="p3k1z7"
git clone https://github.com/vibrito/qualidade-de-software
cd qualidade-de-software-main

python -m venv venv
source venv/bin/activate   # Linux/Mac
venv\Scripts\activate      # Windows

pip install -r requirements.txt
```

---

## Treinar o Modelo

```bash id="b0k3p2"
python src/train_model.py
```

O script:

* Treina múltiplos modelos (KNN, Decision Tree, Naive Bayes, SVM)
* Realiza busca de hiperparâmetros quando aplicável
* Seleciona automaticamente o melhor modelo
* Salva o modelo em `artifacts/model.joblib`

---

## Executar a Aplicação

```bash id="z8k2n1"
python app/app.py
```

A aplicação será iniciada em:

http://localhost:5001

Permite inserir valores manualmente e obter a predição.

---

## Executar Testes

```bash id="m4l8q9"
pytest
```

Os testes verificam:

* Existência do modelo treinado
* Validade das previsões
* Acurácia mínima de 70%

---

## Critérios de Qualidade

* Acurácia mínima: 0.70
* Separação entre treino e teste
* Pipeline com imputação e normalização

---

## Observações

* Jogadores na posição de goleiro são ignorados
* As posições são agrupadas em categorias
* O dataset é carregado online, sendo necessária conexão com a internet

---

## Autor

Vinicius do Carmo Brito
GitHub: https://github.com/vibrito
