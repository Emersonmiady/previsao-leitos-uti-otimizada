# COVID-19: evitando um possível colapso nos sistemas de saúde
---
Este projeto foi desenvolvido durante o **Bootcamp de *Data Science* Aplicada (Alura)**, no último módulo. Neste, vimos alguns tópicos como modelos de ***Machine Learning***, ***Cross-Validation***, ***Efeitos da Aleatoriedade ao dividir o DataFrame em treino e teste***, ***Tuning de Hiperparâmetros*** ***etc***. As aulas foram feitas em cima de um *Dataset* do *Kaggle*, disponibilizado pelo **Hospital Sírio-Libanês**.

Como desafio, foi requisitado um **modelo que prevê a admissão de pacientes com COVID-19 para leitos de UTI**, ou seja, se a pessoa está em um risco maior por conta do novo coronavírus, além de realizar uma ***EDA*** dos dados.

- Análise completa em "**previsao_leitos_uti_sirio_libanes_otimizada.ipynb**".

Observações: 
- A primeira versão deste projeto está [neste link](https://github.com/Emersonmiady/previsao-leitos-uti).

# Contexto
---
A pandemia da COVID-19 atingiu o mundo inteiro, sobrecarregando assim os sistemas de saúde, os quais estavam despreparados para uma solicitação tão intensa e demorada de leitos de UTI, profissionais, equipamentos de proteção individual e recursos de saúde.

![colapse](/img/colapse.jpg)

Acima vemos a curva **cinza** sendo o ideal, pois está abaixo da capacidade que o sistema está dispondo. Obviamente que, medidas como:

- **Lavar sempre as mãos**;
- **Passar álcool em gel frequentemente e usar máscaras ao sair de casa**;
- **Se possível, ainda adotar o isolamento social**.

Ajudam muito a reduzir esses aumentos de casos diários, deixando a curva vermelha mais parecida com a situação que queremos. Entretanto, se essas não forem seguidas, teríamos que aumentar a capacidade dos sistemas de saúde, e assim, exigindo verba e tempo até adquirir mais recursos médicos.

Mas o que devemos fazer se a quantidade de leitos ficar estável e não existirem muitos equipamentos sobrando? É aí que nos deparamos com um outro problema, relacionado a **gestão de pessoas que vão para a UTI do próprio hospital:**

- Se encontrarmos uma situação em que todos os leitos estão ocupados, precisamos de **agilidade para evitar um possível superlotamento do espaço**, e além disso, teríamos que **transferir as pessoas para a UTI de outros hospitais**!
- Se determinado hospital estiver com leitos disponíveis, poderíamos contatar os outros locais de saúde para **transferir mais pessoas neste**.

Sendo assim, pensando na **necessidade de ser ágil** para a ida e vinda de pacientes, o Hospital Sírio-Libanês está tentando buscar uma solução para o caso, isto é, **prever se a pessoa precisará de leito antecipadamente**. E é disso que se trata este projeto.

**Base de dados:** [Dataset do Hospital Sírio-Libanês (*Kaggle*)](https://www.kaggle.com/S%C3%ADrio-Libanes/covid19), com todas as features já *normalizadas*.

# Ferramentas utilizadas
---
- Linguagem de programação Python;
- Pacotes: `pandas`, `numpy`, `matplotlib`, `seaborn`, `plotly`, `IPython`, `warnings`, `boruta`, `sklearn`, `statsmodels` e `picke`;
- Notebook: *Jupyter Notebook*.

# Descrição breve das etapas
---
## 1. Pré-processamento dos dados

### 1.1. Primeiro momento
- Preenchimento de valores faltantes com métodos "*backfill*" e "*fowardfill*" dado um mesmo paciente;
- Remoção de pacientes que já são declarados como `ICU` = 1 na primeira janela (`0-2`);
- Remoção dos pacientes que não foram coletadas nenhuma informação de certa variável em nenhuma das janelas;
- Seleção de uma única janela de tempo (`0-2`).

### 1.2. Segundo momento
- Remoção da maioria das variáveis que possuem `MIN`, `MAX` e `MEDIAN` no final.

### 1.3. Terceiro momento
- Seleção de variáveis com o algoritmo `BorutaPy`, indo apenas para 8 variáveis independentes e 1 dependente (uma excelente evolução)!

## 2. Análise Exploratória dos Dados
Utilizei gráficos de barras, *violinplots*, *boxplots*, matriz de correlação (mapa de calor) e gráficos de regressão. Vão alguns exemplos de visualizações abaixo:

![age_percentil](/img/age_percentil.png)

![icu_age_above](/img/icu_age_above.png)

![gender](/img/gender.png)

![diseases](/img/diseases.png)

![blood_vital_signs](/img/blood_vital_signs.png)

![correlation_map](/img/correlation_map.png)

![regression_plots_high_corr](/img/regression_plots_high_corr.png)

## 3. Machine Learning: modelos testados
Foram testados:
- Naive-Bayes;
- Regressão Logística;
- Árvore de Decisão;
- Random Forest.

## 4. Procedimento para saber qual deles é o melhor
1. Divisão das variáveis em dependente (y) e independentes (X);
2. Cross-Validation com todo o *DataFrame*, calculando também o intervalo de confiança para os resultados (métricas) dos modelos (o intervalo foi calculado pois existiam poucas linhas, podendo haver um viés aleatório na divisão de treino e teste);
3. Avaliação de todas as métricas e seleção do melhor modelo. Abaixo vão os resultados:

![metrics](/img/metrics.png)

Após algumas análises, a ***Random Forest Classifier*** foi a escolhida!

O que realmente pesou foi o melhor *Recall*, pois quanto menos falsos negativos tiver, ou seja, a pessoa não ser admitida para UTI mas que na verdade era para ser, é melhor. Um *Recall* ruim pode literalmente matar pessoas... Então basicamente optei também pelo modelo com a melhor sensibilidade.

Apesar de já escolher o melhor modelo, resolvi olhar para os *betas* da regressão logística, porque também indicam, de forma linear, as variáveis mais importantes para este modelo, e descobri que, o `PCR_MEAN` foi a mais importante, indicando que, quando é aumentado em uma unidade nesta variável, a chance da pessoa precisar de UTI é aumentada em  $𝑒^{2.96} \approx 20$ vezes! Para finalizar, fiz um paralelo com a *feature importances* do modelo da *Random Forest Classifier*, e a `PCR_MEAN` continuou sendo a mais importante!

![feature_importances_1](/img/feature_importances_1.png)

## 5. *Tuning* de Hiperparâmetros
Acabei testando no `GridSearchCV()` (função do `sklearn`) valores diferentes para `max_features` e `n_estimators`. O resultado foi:

- `max_features`: 2;
- `n_estimators`: 100.

![train_cv_validation_recall](/img/train_cv_validation_recall.png)

## 6. Visualizações extras

- *Coutor plots* (2D e 3D), realizado com o `plotly`;

- *Feature Importances* final:

![feature_importances_2](/img/feature_importances_2.png)

## 7. Validação Final

Para finalizar com o notebook, realizei a validação final com os dados separados anteriormente, é como se o modelo recebesse dados completamente novos. Na minha opinião, os resultados foram muito positivos, com um *recall* de 0.87 e *f1-score* de 0.79!

![confusion_matrix_final](/img/confusion_matrix_final.png)
