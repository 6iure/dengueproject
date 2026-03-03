# %%
import pandas as pd 
import numpy as np
# %%

path = 'data/DENGBR24.parquet'
# %%
df_dengue = pd.read_parquet(path)
df_dengue.head(100)

# %%
df_dengue.info()

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

df_dengue['SG_UF_NOT'].value_counts(dropna=False)

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
    32 : 'ES'
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
#* criacao de faixas etarias para dps cruzar com variaveis socio-economicas
bins = [1, 11, 17, 59, np.inf]
labels = ['Crianca', 'Adolescente', 'Adulto', 'Idoso']

no_outliers['faixa_etaria'] = pd.cut(
    no_outliers['idade_calculada'],
    bins=bins,
    labels=labels
)

no_outliers.head()
# %% visualizando a quantia e pocentagem de notificacoes por faixa etaria
count_ages = no_outliers['faixa_etaria'].value_counts()
percent_ages = no_outliers['faixa_etaria'].value_counts(normalize=True) * 100
print(f"quantia de casos por: {count_ages}\nporcentagem de casos por: {percent_ages}")

# %% tratando as variaveis de raca e escolaridade, para ter nocao de um aspecto mais social
no_outliers['CS_RACA'].isnull().sum()

# %% dropando as colunas 

no_outliers.dropna(subset=['CS_RACA'], inplace=True)
no_outliers['CS_RACA'].isnull().sum()

# %%
n_to_race = {
    1 : 'Branca',
    2 : 'Preta',
    3 : 'Amarela',
    4 : 'Parda',
    5 : 'Indigena',
    9 : 'Ignorado',
}

no_outliers['CS_RACA'] = no_outliers['CS_RACA'].map(n_to_race)
no_outliers.head(20)

# %%
no_outliers['CS_ESCOL_N'].value_counts(dropna= False)
# %% tratando a variavel de escolaridade.
#* primeiro, mudando das variaveis do sinan para o que elas significam. *(existiam 3 variaveis diferentes, todas indicavam que o ensino fund. estava incompleto. As agrupei apenas como ens. fund. incompleto)

n_to_education = { 
    0 : 'Analfabeto',
    1 : 'Fundamental Incompleto',
    2 : 'Fundamental Incompleto',
    3 : 'Fundamental Incompleto',
    4 : 'Fundamental Completo',
    5 : 'Ens. Médio Incompleto',
    6 : 'Ens. Médio Completo',
    7 : 'Ens. Superior Incompleto',
    8 : 'Ens. Superior Completo ',
    9 : 'Ignorado/Branco',
    10 : 'Nao se aplica',
}

no_outliers['rotulo_escolaridade'] = no_outliers['CS_ESCOL_N'].map(n_to_education) 

# %%
#todo: dps fazer um loc, mas para identificar em qual faixa etaria estao localizados os NA's
no_outliers.loc[
    (no_outliers['idade_calculada'] < 7) & (no_outliers['rotulo_escolaridade'].isna()), 
    'rotulo_escolaridade'
].value_counts(dropna=False)
# %%
#* se a idade for menor que 7 e o nivel de escolaridade for nulo, criaremos um rotulo novo, deduzindo que nessa idade as criancas ainda nao enfrentam a escola. 
no_outliers.loc[(no_outliers['idade_calculada'] < 7) & (no_outliers['rotulo_escolaridade'].isna()), 'rotulo_escolaridade'] = 'Nao se aplica (idade)'

# %%
#* transformando o restante dos NA's em uma faixa 'informacao ausente'
no_outliers['rotulo_escolaridade'] = no_outliers['rotulo_escolaridade'].fillna('Informação Ausente')
# %%
no_outliers['rotulo_escolaridade'].value_counts(normalize=True) * 100

# %%
no_outliers['CS_SEXO'].isna().sum()

# %%
no_outliers.dropna(subset=['CS_SEXO'], inplace=True)
no_outliers['CS_SEXO'].isna().sum()

# %%
#criando coluna que mostrara a quantidade de dias que demoram pra comecar a investigacao
no_outliers['dias_p_investigar'] = (no_outliers['DT_INVEST'] - no_outliers['DT_SIN_PRI'])
no_outliers.head()

# %%
pd.set_option('display.max_columns', None)
no_outliers.head(100)

# %%
no_outliers['HOSPITALIZ'].value_counts(dropna=False)
# %%
n_to_hospital = {
    1 : 'Sim',
    2 : 'Não',
    9 : 'Ignorado'
}

no_outliers['HOSPITALIZ'] = no_outliers['HOSPITALIZ'].map(n_to_hospital)
no_outliers.head()

# %%
no_outliers['HOSPITALIZ'].fillna('Não informado', inplace=True)
no_outliers.head()
# %%
no_outliers['HOSPITALIZ'].value_counts(dropna=False)
# %%
#! pelo fato de essa coluna ter mais de 6 milhoes de NA's, vou utiliza-la apenas como um filtro de qualidade. onde nao tem NA nessa coluna, irei considerar como caso grave de Dengue.
no_outliers['DT_INTERNA'].value_counts(dropna=False)

# %%
# criando coluna com diferenca entre inicio dos sintomas e data de internacao
no_outliers['dias_p_internar'] = (no_outliers['DT_INTERNA'] - no_outliers['DT_SIN_PRI']).dt.days
no_outliers.head()
# %%
df_severe = no_outliers[no_outliers['DT_INTERNA'].notna()]
df_severe

# %%
no_outliers['CLASSI_FIN'].value_counts(dropna=False)

# %%

final_classif = { 
    10 : 'Ignorado',
    8 : 'Inconclusivo',
    11 : 'Confirmado',
    12 : 'Descartado'
}

no_outliers['CLASSI_FIN'] = no_outliers['CLASSI_FIN'].map(final_classif)
no_outliers['CLASSI_FIN'].value_counts(dropna=False)
# %%
no_outliers = no_outliers.dropna(subset=['CLASSI_FIN']) 
no_outliers['CLASSI_FIN'].value_counts(dropna=False)

# %%
no_outliers['EVOLUCAO'].value_counts(dropna=False)

# %%
evol = { 
    1 : 'Cura',
    9 : 'Ignorado',
    2 : 'Óbito pelo agravo',
    3 : 'Óbito por outras causas',
    4 : 'Óbito em investigação'
}
# %%
no_outliers['EVOLUCAO'] = no_outliers['EVOLUCAO'].map(evol)

# %% para nao enviesar a taxa de letalidade, unificarei os nans e ignorados como em aberto / nao informado

no_outliers['EVOLUCAO'] = no_outliers['EVOLUCAO'].fillna('Em Aberto / Não Informado')
no_outliers.loc[no_outliers['EVOLUCAO'] == 'Ignorado', 'EVOLUCAO'] = 'Em aberto / Não Informado '
# %%
no_outliers.head()
# %%
no_outliers['DT_OBITO'].value_counts(dropna=False)

# %%
no_outliers['falecido'] = no_outliers['DT_OBITO'].notna()
no_outliers.head()

# %%
no_outliers['ID_OCUPA_N'].value_counts(dropna=False)

# %%
no_outliers.value_counts('ID_OCUPA_N').head(10)

# %%
dump_values = ['XXX', '000000', '998999']
no_outliers['ID_OCUPA_N'] = no_outliers['ID_OCUPA_N'].replace(dump_values,np.nan)

# %% 
# dando 'nome' aos ids de ocupacao de seguindo as nomenclaturas do CBO.
conditions = [
    no_outliers['ID_OCUPA_N'] == '999991',
    no_outliers['ID_OCUPA_N'] == '999992',
    no_outliers['ID_OCUPA_N'] == '999993',
    no_outliers['ID_OCUPA_N'] == '999994',
    no_outliers['ID_OCUPA_N'].str[0].isin(['0', '1', '2', '3', '4', '5', '6', '7', '8', '9'])
]

choices = [
    'Estudante',
    'Dona de casa',
    'Aposentado/Pensionista',
    'Desempregado',
    'Trabalhador Ativo'
]

no_outliers['ocupacao_sintetica'] = np.select(conditions, choices, default='Não informado')

# %% a partir de cada numero inicial do ID de ocup. é possivel pegar o seu "grande grupo", assim farei um mapeamento por grupos de ocupação, usando as nomenclaturas do CBO.

cbo_map = {
    '0': 'Militares/Policiais',
    '1': 'Diretores/Gerentes',
    '2': 'Profissionais (Nível Superior)',
    '3': 'Técnicos (Nível Médio)',
    '4': 'Administrativo',
    '5': 'Serviços/Comércio',
    '6': 'Agropecuária/Pesca',
    '7': 'Indústria/Construção',
    '8': 'Indústria/Construção',
    '9': 'Manutenção e Reparação'
}

# aplicar o detalhamento apenas onde marquei como 'trabalhador ativo'
# o que preserva os nomes 'dona de casa', 'aposentado', etc.

mascara_ativo = no_outliers['ocupacao_sintetica'] == 'Trabalhador Ativo'
no_outliers.loc[mascara_ativo, 'ocupacao_sintetica'] = no_outliers.loc[mascara_ativo, 'ID_OCUPA_N'].str[0].map(cbo_map)

# %%
no_outliers['ocupacao_sintetica'].value_counts()
# %% limpeza de aproximadamente 360 mil linhas
no_outliers.shape
# %%
no_outliers.to_parquet('data/dbr24cleansed.parquet', index=False)