import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np


# Vou importar os dados do arquivo csv e colocar na variável df
df = pd.read_csv('medical_examination.csv')


# Aqui vou calcular o IMC e criar a coluna overweight
# Primeiro calculo o IMC dividindo o peso pelo quadrado da altura em metros
# Altura tá em cm, então divido por 100 pra converter pra metros
# O resultado da comparação (IMC > 25) é um valor booleano (True/False)
# O método .astype(int) converte True para 1 e False para 0
df['overweight'] = (df['weight'] / ((df['height'] / 100) ** 2) > 25).astype(int)


# Agora vou normalizar os dados de cholesterol e gluc
# Se for 1, vou deixar como 0 (bom), se for maior que 1, vou deixar como 1 (ruim)
# O método .astype(int) converte os valores booleanos (True/False) para inteiros (1/0)
# Isso é útil porque muitas funções de visualização trabalham melhor com valores numéricos
df['cholesterol'] = (df['cholesterol'] > 1).astype(int)
df['gluc'] = (df['gluc'] > 1).astype(int)


def draw_cat_plot():
    # Vou usar o pd.melt igual a professora disse no "plantão" pra reorganizar os dados e facilitar a visualização
    # Isso vai transformar as colunas em linhas, tipo uma tabela pivô ao contrário
    # id_vars=['cardio'] - mantém a coluna 'cardio' como identificador
    # value_vars=[...] - especifica quais colunas serão transformadas em linhas
    # O resultado terá colunas: 'cardio', 'variable' (nome da coluna original) e 'value' (valor da coluna)
    df_cat = pd.melt(df, id_vars=['cardio'], 
                     value_vars=['cholesterol', 'gluc', 'smoke', 'alco', 'active', 'overweight'])

    
    # Agora vou agrupar os dados por cardio e variable (que é o nome da coluna criada pelo melt)
    # E vou contar quantos valores tem em cada grupo
    # O método .size() conta quantas linhas existem em cada grupo
    # O método .reset_index() transforma os índices em colunas, facilitando o uso dos dados
    df_cat = df_cat.groupby(['cardio', 'variable', 'value']).size().reset_index()
    # Renomeio a coluna 0 para 'total' pra funcionar com o catplot
    # O parâmetro inplace=True faz a alteração diretamente no DataFrame, sem precisar atribuir a uma nova variável
    df_cat.rename(columns={0: 'total'}, inplace=True)
    
    # Agora vou criar o gráfico usando o catplot do seaborn
    # Vou usar o kind='bar' pra criar um gráfico de barras
    # x='variable' - coloca as variáveis no eixo x
    # y='total' - usa os valores da coluna 'total' para a altura das barras
    # hue='value' - separa as barras por cores de acordo com o valor (0 ou 1)
    # col='cardio' - cria dois gráficos lado a lado, um para cardio=0 e outro para cardio=1
    # O atributo .fig no final retorna a figura do matplotlib, não o objeto do seaborn
    # Isso é necessário porque precisamos da figura para salvar o arquivo depois
    fig = sns.catplot(data=df_cat, kind='bar', x='variable', y='total', 
                      hue='value', col='cardio').fig

    # Salvar a figura no arquivo catplot.png
    fig.savefig('catplot.png')
    return fig

def draw_heat_map():

    # Vou filtrar os dados pra remover os valores incorretos
    # Primeiro filtro onde a pressão diastólica é maior que a sistólica
    # Depois filtro altura e peso pelos percentis
    # O método .quantile(0.025) retorna o valor abaixo do qual estão 2,5% dos dados
    # O método .quantile(0.975) retorna o valor abaixo do qual estão 97,5% dos dados
    # Isso remove os outliers (valores muito extremos) que podem afetar a análise
    df_heat = df[
        (df['ap_lo'] <= df['ap_hi']) & 
        (df['height'] >= df['height'].quantile(0.025)) &
        (df['height'] <= df['height'].quantile(0.975)) &
        (df['weight'] >= df['weight'].quantile(0.025)) &
        (df['weight'] <= df['weight'].quantile(0.975))
    ]


    # Calculo a matriz de correlação
    # O método .corr() calcula o coeficiente de correlação de Pearson entre todas as colunas numéricas
    # O resultado é uma matriz onde cada célula mostra a correlação entre duas variáveis
    # Valores próximos de 1 indicam correlação positiva forte
    # Valores próximos de -1 indicam correlação negativa forte
    # Valores próximos de 0 indicam pouca ou nenhuma correlação
    corr = df_heat.corr()


    # Crio uma máscara pro triângulo superior da matriz
    # Isso vai fazer com que só mostre metade do gráfico, sem repetir informação
    # A função np.triu() cria uma matriz triangular superior (upper triangle)
    # Os valores True na máscara serão escondidos no gráfico
    # Isso é útil porque a matriz de correlação é simétrica (a correlação de A com B é igual à de B com A)
    mask = np.triu(corr)


    # Configuro a figura do matplotlib
    fig, ax = plt.subplots(figsize=(13, 10))


    # Desenho o mapa de calor usando o seaborn
    # Uso a máscara pra mostrar só o triângulo inferior
    # E adiciono as anotações com os valores da correlação
    # mask=mask - aplica a máscara criada anteriormente
    # annot=True - mostra os valores numéricos dentro de cada célula
    # fmt='.1f' - formata os números com uma casa decimal
    # square=True - faz as células ficarem quadradas
    # center=0 - centraliza a escala de cores em 0
    # linewidths=.5 - adiciona linhas entre as células
    # cbar_kws={'shrink': .5} - reduz o tamanho da barra de cores para metade
    sns.heatmap(corr, mask=mask, annot=True, fmt='.1f', square=True, center=0, linewidths=.5, cbar_kws={'shrink': .5})


    # Salvar a figura no arquivo heatmap.png
    fig.savefig('heatmap.png')
    return fig
