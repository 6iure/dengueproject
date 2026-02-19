# %%
import pandas as pd
# %%
path = 'data/DENGBR24.csv'
selected_cols = columns = ['TP_NOT', 'DT_NOTIFIC', 'SG_UF_NOT', 'ID_UNIDADE' , 'DT_SIN_PRI' , 'ANO_NASC',
                           'CS_SEXO', 'CS_RACA', 'CS_ESCOL_N', 'SG_UF', 'ID_MUNICIP', 'DT_INVEST', 
                           'FEBRE' , 'MIALGIA' ,'CEFALEIA' , 'EXANTEMA' , 'VOMITO' , 'NAUSEA' , 'DOR_COSTAS' , 'CONJUNTVIT' , 'ARTRITE',
                           'ARTRALGIA' , 'PETEQUIA_N', 'LEUCOPENIA', 'LACO' , 'DOR_RETRO',
                           'HOSPITALIZ', 'DT_INTERNA', 'CLASSI_FIN', 'EVOLUCAO', 'DT_OBITO', 'DT_ENCERRA', 'ID_OCUPA_N']

# %%
df_dengue = pd.read_csv(path, usecols=selected_cols, low_memory=False)
df_dengue.head(100)

# %%
df_dengue.shape

# %%
df_dengue.info()

# %%
df_dengue['DT_INVEST'].value_counts(dropna=False)

# %%
df_dengue[['DT_NOTIFIC','DT_SIN_PRI', 'DT_INVEST', 'DT_INTERNA', 'DT_OBITO', 'DT_ENCERRA']] = df_dengue[['DT_NOTIFIC','DT_SIN_PRI', 'DT_INVEST', 'DT_INTERNA', 'DT_OBITO', 'DT_ENCERRA']].astype('datetime64[ns]') 
df_dengue.head(100)

# %%
#tirando as datas que estao fora de 2024
colunas_data = ['DT_NOTIFIC', 'DT_INVEST', 'DT_SIN_PRI']

for col in colunas_data:
    df_dengue = df_dengue[df_dengue[col].dt.year == 2024]

# %%
df_dengue['FEBRE'].value_counts(dropna=False)

# %%
symptoms = [ 
    'FEBRE' , 'MIALGIA' ,'CEFALEIA' , 'EXANTEMA' , 'VOMITO' , 'NAUSEA' , 'DOR_COSTAS' , 'CONJUNTVIT' , 'ARTRITE',
    'ARTRALGIA' , 'PETEQUIA_N', 'LEUCOPENIA', 'LACO' , 'DOR_RETRO'
]

# %%
df_dengue.dropna(subset=symptoms, inplace=True)

#%%
#tratamento das doencas, apenas transformando em booleanos, p facilitar a proporcao de prevalencia. dropando os nans, pq sao mto poucos (146 mil)

for col in symptoms: 
    df_dengue[col] = df_dengue[col] == 1 

pd.set_option('display.max_columns', None)
df_dengue.head()

# %%
df_dengue.shape

# %%
df_dengue.to_parquet('data/DENGBR24.parquet', index=False)

# %%
