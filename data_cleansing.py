# %%
import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
# %%

path = 'data/DENGBR24.csv'
selected_cols = columns = ['TP_NOT', 'DT_NOTIFIC', 'SG_UF_NOT', 'ID_UNIDADE' , 'DT_SIN_PRI' , 'ANO_NASC',
                           'CS_SEXO', 'CS_RACA', 'CS_ESCOL_N', 'SG_UF', 'ID_MUNICIP', 'DT_INVEST', 
                           'FEBRE' , 'MIALGIA' ,'CEFALEIA' , 'EXANTEMA' , 'VOMITO' , 'NAUSEA' , 'DOR_COSTAS' , 'CONJUNTVIT' , 'ARTRITE',
                           'ARTRALGIA' , 'PETEQUIA_N', 'LEUCOPENIA', 'LACO' , 'DOR_RETRO',
                           'HOSPITALIZ', 'DT_INTERNA', 'CLASSI_FIN', 'EVOLUCAO', 'DT_OBITO', 'DT_ENCERRA', 'ID_OCUPA_N', 'CRITERIO']

# %%
df_dengue = pd.read_csv(path, usecols=selected_cols)
df_dengue.head(100)

# %%
df_dengue.shape
# %% 

#* tratando os dados -- traduzindo as abreviacoes do sistema do sinan
df_dengue['TP_NOT'].value_counts()

# %% 
#TP_NOT: 2 = INDIVIDUAL -- 3 = SURTO 

df_dengue['TP_NOT'] = np.where(df_dengue['TP_NOT'] == 2, 'Individual', 'Surto')
df_dengue.head()
# %%

df_dengue['SG_UF_NOT'].value_counts()
# %%
#fazendo um dict para substituir os codigos das UFs pelos nomes dos estados
sg_to_name = { 
    35 : 'SP',
    31 : 'MG',
    12 : 'AC',
    27 : 'AL',
    16 : 'AP',
    13 : 'AM',
    29 : 'BA',
    23 : 'CE',
    53 : 'DF',
    52 : 'GO',
    21 : 'MA',
    51 : 'MT',
    50 : 'MS',
    15 : 'PA',
    25 : 'PB',
    41 : 'PR',
    26 : 'PE',
    22 : 'PI',
    24 : 'RN',
    43 : 'RS',
    33 : 'RJ',
    11 : 'RO',
    14 : 'RR',
    42 : 'SC',
    28 : 'SE',
    17 : 'TO',
}

df_dengue['SG_UF_NOT'] = df_dengue['SG_UF_NOT'].map(sg_to_name)
df_dengue.sort_values('SG_UF_NOT', ascending=True).tail(20)
# %% Tratamento do ano de nascimento inicial, para trabalhar com idades
df_dengue['ANO_NASC'].describe()

# %%
#dropando NAs da coluna de ano nascimento
df_dengue = df_dengue.dropna(subset=['ANO_NASC'])

# %% limitando as idades para que nao existam idade extremamente antigas ou futuras
df_dengue = df_dengue[df_dengue['ANO_NASC'] <= 2024]
df_dengue = df_dengue[df_dengue['ANO_NASC'] > 1900]

# %%
df_dengue['idade_calculada'] = 2024 - df_dengue['ANO_NASC']
df_dengue.head()
# %%
df_dengue['idade_calculada'].describe()
# %%
#* Criando um Intervalo de Confianca de 99%, para assim mostrar onde o verdadeiro parâmetro populacional provavelmente se encontra; 99% das notificacoes estao dentro dessas idades.

lim_inf = df_dengue['idade_calculada'].quantile(0.005)
lim_sup = df_dengue['idade_calculada'].quantile(0.995)

print(f"intervalo de confianca de {lim_inf} a {lim_sup}")
# %% criando um novo dataframe com 
no_outliers = df_dengue[
    (df_dengue['idade_calculada'] >= lim_inf) &
    (df_dengue['idade_calculada'] <= lim_sup)
].copy()

no_outliers.info()
# %% quantidade de outliers removidos com o intervalo de confianca de 99%
len(df_dengue) - len(no_outliers)

# %%
#todo transformar esse df no outliers para o df_dengue padrao
#todo ou a partir do primeiro tratamento, usar apenas o no_outliers (mais correto)
# %% grafico para visualizacao de notificacoes por idade
#todo remover deste arquivo
# plt.figure(figsize=(8, 5))
# sns.histplot(no_outliers['idade_calculada'], bins=30, color='lightblue', kde=True)
# plt.xlabel('Idade')
# plt.ylabel('Quantidade de Notificacoes')
# %%
#* criacao de faixas etarias para dps cruzar com variaveis socio-economicas
bins = [1, 11, 17, 59, np.inf]
labels = ['Crianca', 'Adolescente', 'Adulto', 'Idoso']

df_dengue['faixa_etaria'] = pd.cut(
    df_dengue['idade_calculada'],
    bins=bins,
    labels=labels
)

df_dengue.head()
# %% visualizando a quantia e pocentagem de notificacoes por faixa etaria
count_ages = df_dengue['faixa_etaria'].value_counts()
percent_ages = df_dengue['faixa_etaria'].value_counts(normalize=True) * 100
print(f"quantia de casos por: {count_ages}\nporcentagem de casos por: {percent_ages}")

# %% grafico para visualizacao de notificacoes por faixa etaria
#todo remover deste arquivo
# plt.figure(figsize=(10,6))
# sns.countplot(data=df_dengue, x='faixa_etaria', palette='viridis')
# plt.title('Volume de Notificações por Ciclo de Vida')
# plt.ylabel('Quantidade de Notificações')
# plt.xlabel('Faixa Etária')
# plt.show()
# %% tratando as variaveis de raca e escolaridade, para ter nocao de um aspecto mais social
df_dengue['CS_RACA'].isnull().sum()

# %% dropando as colunas 

df_dengue.dropna(subset=['CS_RACA'], inplace=True)
df_dengue['CS_RACA'].isnull().sum()

#todo ver como vou tratar a variavel ID_UNIDADE, que tem NA's mas muito poucos

# %%
n_to_race = {
    1 : 'Branca',
    2 : 'Preta',
    3 : 'Amarela',
    4 : 'Parda',
    5 : 'Indigena',
    9 : 'Ignorado',
}

df_dengue['CS_RACA'] = df_dengue['CS_RACA'].map(n_to_race)
df_dengue.head(20)

# %%
df_dengue['CS_ESCOL_N'].value_counts(dropna= False)
# %% tratando a variavel de escolaridade.
#* primeiro, mudando das variaveis do sinan para o que elas significam. *(existiam 3 variaveis diferentes, todas indicavam que o ensino fund. estava incompleto. As agrupei apenas como ens. fund. incompleto)

n_to_education = { 
    0 : 'Analfabeto',
    1 : 'Fundamental Incompleto',
    2 : 'Fundamental Incompleto',
    3 : 'Fundamental Incompleto',
    4 : 'Ensino fundamental completo',
    5 : 'Ensino medio incompleto',
    6 : 'Ensino Medio Completo',
    7 : 'Educação superior incompleta',
    8 : 'Educação superior completa ',
    9 : 'Ignorado/Branco',
    10 : 'Nao se aplica',
}

df_dengue['rotulo_escolaridade'] = df_dengue['CS_ESCOL_N'].map(n_to_education) 

# %%
#todo, dps fazer um loc, mas para identificar em qual faixa etaria estao localizados os NA's
df_dengue.loc[
    (df_dengue['idade_calculada'] < 7) & (df_dengue['rotulo_escolaridade'].isna()), 
    'rotulo_escolaridade'
].value_counts(dropna=False)
# %%
#* se a idade for menor que 7 e o nivel de escolaridade for nulo, criaremos um rotulo novo, deduzindo que nessa idade as criancas ainda nao enfrentam a escola. 
df_dengue.loc[(df_dengue['idade_calculada'] < 7) & (df_dengue['rotulo_escolaridade'].isna()), 'rotulo_escolaridade'] = 'Nao se aplica (idade)'

# %%
#* transformando o restante dos NA's em uma faixa 'informacao ausente'
df_dengue['rotulo_escolaridade'] = df_dengue['rotulo_escolaridade'].fillna('Informação Ausente')
# %%
df_dengue['rotulo_escolaridade'].value_counts(normalize=True) * 100

# %%
df_dengue['CS_SEXO'].isna().sum()

# %%
df_dengue.dropna(subset=['CS_SEXO'], inplace=True)
df_dengue['CS_SEXO'].isna().sum()

# %%
df_dengue['DT_INVEST'].value_counts(dropna=False)

# %%
#dropando os NA'S (eram 113 mil 'apenas')
df_dengue.dropna(subset=['DT_INVEST'], inplace=True)

# %%
#por mais que as datas ja estejam no padrao correto, estao como str, entao eh necessaria essa conversao
df_dengue['DT_SIN_PRI'] = pd.to_datetime(df_dengue['DT_SIN_PRI'], errors='coerce')
df_dengue['DT_INVEST'] = pd.to_datetime(df_dengue['DT_INVEST'], errors='coerce')


# %%
#criando coluna que mostrara a quantidade de dias que demoram pra comecar a investigacao
df_dengue['dias_p_investigar'] = (df_dengue['DT_INVEST'] - df_dengue['DT_SIN_PRI'])
df_dengue.head()

# %%
df_dengue.shape

# %%
df_dengue['FEBRE'].value_counts(dropna=False)
#%%
#tratamento das doencas, apenas transformando em booleanos, p facilitar a proporcao de prevalencia. dropando os nans, pq sao mto poucos (129)

symptoms = [ 
    'FEBRE' , 'MIALGIA' ,'CEFALEIA' , 'EXANTEMA' , 'VOMITO' , 'NAUSEA' , 'DOR_COSTAS' , 'CONJUNTVIT' , 'ARTRITE',
    'ARTRALGIA' , 'PETEQUIA_N', 'LEUCOPENIA', 'LACO' , 'DOR_RETRO'
]

for col in symptoms: 
    df_dengue[col] = df_dengue[col] == 1 

pd.set_option('display.max_columns', None)
df_dengue.head()

# %%
# todo tratar o restante dessas variaveis
# 'CLASSI_FIN', 'EVOLUCAO', 'DT_OBITO', 'DT_ENCERRA', 'ID_OCUPA_N', 'CRITERIO']

df_dengue['HOSPITALIZ'].value_counts(dropna=False)
# %%
n_to_hospital = {
    1 : 'Sim',
    2 : 'Nao',
    9 : 'Ignorado'
}

df_dengue['HOSPITALIZ'] = df_dengue['HOSPITALIZ'].map(n_to_hospital)
df_dengue.head()

# %%
df_dengue['HOSPITALIZ'].fillna('Não informado', inplace=True)
df_dengue.head()
# %%
df_dengue['HOSPITALIZ'].value_counts(dropna=False)
# %%
#! pelo fato de essa coluna ter mais de 6 milhoes de NA's, vou utiliza-la apenas como um filtro de qualidade. onde nao tem NA nessa coluna, irei considerar como caso grave de Dengue.
df_dengue['DT_INTERNA'].value_counts(dropna=False)

# %%
df_dengue['DT_INTERNA'] = pd.to_datetime(df_dengue['DT_INTERNA'], errors='coerce')
# %%
# criando coluna com diferenca entre inicio dos sintomas e data de internacao
df_dengue['dias_p_internar'] = (df_dengue['DT_INTERNA'] - df_dengue['DT_SIN_PRI']).dt.days
df_dengue.head()
# %%
df_severe = df_dengue[df_dengue['DT_INTERNA'].notna()]
df_severe

# %%
df_dengue['CLASSI_FIN'].value_counts(dropna=False)

# %%
