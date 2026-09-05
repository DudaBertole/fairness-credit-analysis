# Fairness em Modelos de Análise de Crédito

Projeto de iniciação científica sobre identificação, avaliação e mitigação de vieses algorítmicos em modelos de aprendizado de máquina aplicados à análise de crédito. O projeto utiliza a base **Statlog German Credit Data** e compara modelos clássicos de classificação com técnicas de fairness aplicadas em diferentes etapas do pipeline de aprendizado de máquina.

> Este repositório contém os notebooks e módulos usados nos experimentos. O [relatório final](https://dudabertole.github.io/my-files/Relatorio-final-iniciacao-cientifica-2026.pdf) apresenta o contexto teórico, a metodologia completa, os resultados e a discussão detalhada das análises.

## Sobre o projeto

Modelos de aprendizado de máquina são utilizados no setor financeiro para automatizar decisões de concessão de crédito, uma tarefa conhecida como _credit scoring_. Como esses modelos aprendem a partir de dados históricos, podem reproduzir ou amplificar desigualdades sociais preexistentes.

O projeto investiga como esse viés pode se manifestar em modelos de classificação binária e em que medida diferentes estratégias conseguem reduzi-lo. Para isso, são avaliados simultaneamente:

- desempenho preditivo, como acurácia, precisão, recall e F1-Score;
- equidade entre grupos demográficos, por meio de métricas de fairness;
- impacto das técnicas de mitigação sobre o desempenho e sobre as disparidades entre grupos.

O conjunto de dados utilizado contém 1.000 instâncias, 20 atributos preditivos e o alvo binário `good_client`, que indica se o cliente é considerado bom ou mau pagador. A análise principal utiliza `age` como atributo sensível, mantendo essa informação entre as variáveis preditoras para medir o viés presente nos dados e no modelo.

## Objetivos da pesquisa

O objetivo geral é contribuir para o estudo de vieses algorítmicos em modelos de aprendizado de máquina, tendo a concessão de crédito como contexto de aplicação.

Os objetivos específicos definidos no relatório são:

1. Estudar o estado da arte sobre técnicas computacionais de identificação, avaliação e mitigação de vieses algorítmicos, com ênfase em modelos de análise de crédito.
2. Levantar conjuntos de dados reais adequados para experimentos no contexto da pesquisa.
3. Realizar experimentos com os dados selecionados para identificar vieses nos modelos e testar diferentes abordagens de mitigação.

No código, esses objetivos são operacionalizados por meio de pipelines modulares que combinam preparação dos dados, algoritmos de classificação, técnicas de mitigação, ajuste de hiperparâmetros e avaliação comparativa.

## Conceitos importantes

### Fairness

Fairness, ou justiça algorítmica, é o campo que estuda formas de identificar e reduzir tratamentos desiguais produzidos por sistemas automatizados. Não existe uma única definição universal de justiça: cada métrica representa uma perspectiva diferente sobre a distribuição dos resultados entre grupos.

Por isso, o projeto combina métricas de desempenho e de fairness. Melhorar uma métrica pode afetar outra, e uma técnica que funciona bem para um algoritmo ou conjunto de dados pode não produzir o mesmo resultado em outro cenário.

### Atributo sensível e grupos

Um atributo sensível, também chamado de atributo protegido, é uma característica individual que pode estar associada a discriminação ou tratamento desigual, como gênero, idade ou nacionalidade.

Nos experimentos, os atributos sensíveis são derivados e codificados da seguinte forma:

```python
sensitive_attributes = {
    "sex": {
        "source": "marriage_status_sex",
        "privileged": "male",
        "unprivileged": "female",
    },
    "age": {
        "privileged": "age >= 25",
        "unprivileged": "age < 25",
    },
    "foreign_worker": {
        "privileged": "foreign worker",
        "unprivileged": "non-foreign worker",
    },
}
```

O grupo **privilegiado (P)** é o grupo que possui vantagem histórica ou contextual. O grupo **não privilegiado (NP)** é o grupo historicamente marginalizado, minoritário ou mais sujeito aos efeitos do viés. No cenário principal do projeto:

| Atributo sensível | Grupo privilegiado (P)         | Grupo não privilegiado (NP) |
| ----------------- | ------------------------------ | --------------------------- |
| `sex`             | Masculino                      | Feminino                    |
| `age`             | Idade maior ou igual a 25 anos | Idade menor que 25 anos     |
| `foreign_worker`  | Estrangeiro                    | Não estrangeiro             |

O rótulo favorável representa a aprovação do crédito, enquanto o rótulo desfavorável representa a sua negação. A definição dos grupos é uma decisão metodológica e sociotécnica: ela precisa ser justificada pelo contexto da aplicação e pelos objetivos daquele (instituto/empresa) que implementa o modelo.

### Métricas de fairness

As principais métricas discutidas no relatório são:

- **Demographic Parity Difference (DPD):** compara a taxa de seleção entre os grupos NP e P.
- **Equal Opportunity Difference (EOD):** compara a taxa de verdadeiros positivos (_true positive rate_) entre os grupos.
- **Average Odds Difference (AOD):** combina as diferenças nas taxas de verdadeiros positivos e falsos positivos.

No código, essas métricas são calculadas com `ClassificationMetric` do AIF360:

```python
from aif360.metrics import ClassificationMetric

metric = ClassificationMetric(
    dataset_true,
    dataset_pred,
    unprivileged_groups=unprivileged_groups,
    privileged_groups=privileged_groups,
)

fairness_metrics = {
    "DPD": metric.statistical_parity_difference(),
    "EOD": metric.equal_opportunity_difference(),
    "AOD": metric.average_odds_difference(),
}
```

Em geral, quanto mais próximo de zero estiver uma diferença de fairness, menor é a disparidade medida por ela. Isso não significa que um único número determine se um sistema é justo: a interpretação depende da métrica, do domínio e das consequências da decisão.

## Técnicas de mitigação

As técnicas foram organizadas de acordo com o momento em que atuam no pipeline:

### Pré-processamento

- **Reweighing (RW):** atribui pesos diferentes às instâncias para reduzir a influência de combinações sub-representadas entre grupo e rótulo, sem alterar os valores originais dos dados.
- **Disparate Impact Remover (DIR):** reduz a dependência entre o atributo sensível e os atributos selecionados, preservando tanto quanto possível as relações intragrupo.

### Em-processamento

- **Prejudice Remover (PR):** adiciona uma penalização relacionada à dependência entre as predições e o atributo sensível durante o treinamento. No AIF360, o parâmetro `eta` controla a intensidade dessa penalização.
- **Adversarial Debiasing (AD):** treina uma rede classificadora e uma rede adversária em uma dinâmica competitiva para reduzir a capacidade de inferir o atributo sensível a partir das predições.

### Pós-processamento

- **Calibrated Equalized Odds (CEO):** ajusta as predições buscando reduzir diferenças nas taxas de erro entre os grupos, mantendo a calibração das probabilidades.
- **Reject Option Classification (ROC):** modifica predições de baixa confiança, próximas ao limiar de decisão, para favorecer o grupo não privilegiado dentro de limites definidos.

Exemplo do roteamento de uma técnica de pré-processamento no projeto:

```python
from utils.bias_preprocessing import apply_bias_preprocessing

dataset_adjusted = apply_bias_preprocessing(
    method="reweighing",
    dataset=dataset_train,
    unprivileged_groups=unprivileged_groups,
    privileged_groups=privileged_groups,
)
```

## Tecnologias utilizadas

Os experimentos foram desenvolvidos em Python e Jupyter Notebook.

| Biblioteca     |   Versão | Uso principal                             |
| -------------- | -------: | ----------------------------------------- |
| `pandas`       |  `3.0.3` | Manipulação de dados                      |
| `numpy`        |  `2.2.6` | Operações numéricas                       |
| `scikit-learn` |  `1.4.2` | Pré-processamento, modelos e tuning       |
| `matplotlib`   |  `3.8.4` | Visualização                              |
| `seaborn`      | `0.13.2` | Visualização estatística                  |
| `aif360`       |  `0.6.1` | Datasets, métricas e técnicas de fairness |

O **AI Fairness 360 (AIF360)** é a principal biblioteca do projeto. Ela fornece a estrutura `BinaryLabelDataset`, as métricas de classificação entre grupos e implementações de técnicas de mitigação nas etapas de pré-processamento, em-processamento e pós-processamento.

### Instalação

Com Python instalado, crie um ambiente virtual e instale as dependências principais:

```bash
python -m venv .venv

# Windows PowerShell
.\.venv\Scripts\Activate.ps1

pip install pandas==3.0.3 numpy==2.2.6 scikit-learn==1.4.2 `
    matplotlib==3.8.4 seaborn==0.13.2 aif360==0.6.1
```

Depois, abra o projeto no Jupyter:

```bash
jupyter notebook
```

> A disponibilidade de algumas versões pode depender da versão do Python e do sistema operacional. Para reproduzir os experimentos, recomenda-se registrar o ambiente utilizado e executar os notebooks na ordem indicada na estrutura abaixo.

## Estrutura do repositório

```text
fairness-credit-analysis/
├── data/
│   ├── raw/                    # Dados originais
│   ├── processed/              # Dados após o pré-processamento
│   └── results/                # Resultados e figuras dos experimentos
├── src/
│   ├── 01-exploratory_data_analysis.ipynb
│   ├── 02-fairness_groups_analysis.ipynb
│   ├── 03_1-classic_models_pipeline.ipynb
│   ├── 03_2-fairness_models_pipeline.ipynb
│   └── utils/
│       ├── bias_postprocessing.py
│       ├── bias_preprocessing.py
│       ├── evaluation.py
│       ├── models.py
│       ├── preprocessing.py
│       └── tuning.py
└── README.md
```

### Ordem de execução

1. `01-exploratory_data_analysis.ipynb`: análise exploratória, qualidade dos dados, distribuições e relevância dos atributos.
2. `02-fairness_groups_analysis.ipynb`: transformação dos atributos sensíveis e definição dos grupos privilegiados e não privilegiados.
3. `03_1-classic_models_pipeline.ipynb`: modelos clássicos com Reweighing, DIR, CEO e ROC.
4. `03_2-fairness_models_pipeline.ipynb`: modelos AIF360 de em-processamento, incluindo Prejudice Remover e Adversarial Debiasing.

Os módulos em `src/utils/` concentram as operações reutilizáveis de preparação dos dados, construção dos modelos, tuning, mitigação e avaliação.

## Sobre a Pesquisa

O relatório final de iniciação científica contém o conteúdo mais detalhado do trabalho, incluindo:

- contextualização e revisão dos conceitos de fairness;
- justificativa da escolha e análise exploratória da base German Credit;
- definição dos atributos sensíveis e dos grupos;
- descrição dos modelos, técnicas de mitigação, parâmetros e protocolos de validação;
- resultados completos, tabelas, gráficos e discussão dos trade-offs;
- limitações do estudo.

Este projeto foi desenvolvido no contexto de uma iniciação científica da **Faculdade de Tecnologia - Universidade Estadual de Campinas (UNICAMP)**.
