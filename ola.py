## ideias analise descritiva: trazer um grafico de barras horizontais de sintomas por estado
## correlacionar sintomas com idade
## ideias analise inferencial:

# %%
import pandas as pd 
import matplotlib as plt
import seaborn as sns
# %%

path = 'data/DENGBR24.csv'
selected_cols = columns = ['TP_NOT', 'ID_AGRAVO', 'DT_NOTIFIC', 'SG_UF_NOT', 'ID_UNIDADE' , 'DT_SIN_PRI' , 'ANO_NASC' , 'NU_IDADE_N' , 'CS_SEXO', 'CS_RACA', 'CS_ESCOL_N', 'SG_UF', 'ID_MUNICIP', 'DT_INVEST', 
                           'FEBRE' , 'MIALGIA' ,'CEFALEIA' , 'EXANTEMA' , 'VOMITO' , 'NAUSEA' , 'DOR_COSTAS' , 'CONJUNTVIT' , 'ARTRITE' , 'ARTRALGIA' , 'PETEQUIA_N' , 'LEUCOPENIA' , 'LACO' , 'DOR_RETRO',
                           'HOSPITALIZ', 'DT_INTERNA', 'CLASSI_FIN', 'EVOLUCAO', 'DT_OBITO', 'DT_ENCERRA',  ]
# %%
df_dengue = pd.read_csv(path, usecols=selected_cols)
# %%
df_dengue.head(20)
# %%

# %%
