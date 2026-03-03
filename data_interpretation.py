# %%
#todo escolher um grafico de mapa do brasil interessante e traze-lo, alem de ajustar e padrozinar o estilo dos graficos que ja tem uma 'base'
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# %%
path = 'data/dbr24cleansed.parquet'
dengue = pd.read_parquet(path)

# %%
pd.set_option('display.max_columns', None)
dengue.head()

# %%
#filtrando apenas pelo desfecho final
df_analysis = dengue[dengue['EVOLUCAO'].isin(['Cura', 'Óbito pelo agravo'])].copy()

# Criando uma coluna numérica para facilitar o cálculo da média (1 para óbito, 0 para cura)
df_analysis['obito_binario'] = (df_analysis['EVOLUCAO'] == 'Óbito pelo agravo').astype(int)

def fatality_calc(column):
    fatality = df_analysis.groupby(column)['obito_binario'].mean() * 100
    return fatality.reset_index().sort_values('obito_binario', ascending=False)

# %%
df_analysis.shape
# %%
#! usar esse grafico de padrao para os outros! (observar apenas as paletas de cores.)
race_data = fatality_calc('CS_RACA')

plt.figure(figsize=(12,6))

ax = sns.barplot(
    data=race_data, 
    x='obito_binario', 
    y='CS_RACA', 
    palette='mako',
    edgecolor='black',
    alpha=0.8,
    )

plt.grid(axis='x', linestyle='--', alpha=0.6)
plt.title('Taxa de Letalidade por Raça/Cor - Dengue 2024', fontsize=14, pad=20, fontweight='bold')
plt.xlabel('Letalidade (%)', fontsize=12, fontweight='bold')
plt.ylabel('Raça/Cor', fontsize=12, fontweight='bold')

for i in ax.containers:
    ax.bar_label(i, fmt='%.3f%%', padding=5)

plt.tight_layout()
plt.show()
# %%
race_data
# %% 
dengue['rotulo_escolaridade'].value_counts()

# %% Grafico para ver como a instrucao escolar impactua nasobrevivencia da dengue -> analise feita em cima de 2.870.794 casos de dengue.
school_data = fatality_calc('rotulo_escolaridade')

# Removendo o 'Não Informado' para ver apenas o gradiente de instrução
school_data = school_data[school_data['rotulo_escolaridade'] != 'Ignorado/Branco']

plt.figure(figsize=(12,6))
ax = sns.pointplot(
    data=school_data, 
    x='obito_binario', 
    y='rotulo_escolaridade', 
    color="#1b89d7",
    alpha=0.8,
    )

plt.grid(axis='both', linestyle='--', alpha=0.6)
plt.title('O Impacto da Instrução na Sobrevivência', fontsize=15, pad=20, fontweight='bold')
plt.xlabel('Letalidade (%)', fontsize=12, fontweight='bold')
plt.ylabel('Nível de Escolaridade', fontsize=12, fontweight='bold')

plt.tight_layout()
plt.show()

# %%
dengue['rotulo_escolaridade'].value_counts()
# %% Visualizando o Prazo Assistencial por Estado (UF).
# apenas 147720 lin has (eh possivel tirar alguma conclusao disso? resposta: SIM!)

delay_data = dengue[(dengue['dias_p_internar'] >= 0) & (dengue['dias_p_internar'] <=20 )].copy()

delay_data.shape
# %% Ordenando os estados pela média de tempo em dias para internar
delay_order = delay_data.groupby('SG_UF_NOT')['dias_p_internar'].mean().sort_values(ascending=True).index

# %%
plt.figure(figsize=(12, 8))

ax = sns.barplot(
    data=delay_data,
    x='SG_UF_NOT',
    y='dias_p_internar',
    order=delay_order,
    palette='mako_r',
    errorbar=None,
)

ax.set_title('Tempo Médio para Internação por Estado (Sintomas → Internação)', fontsize=16, pad=20)
ax.set_ylabel('Média de Dias', fontsize=12)
ax.set_xlabel('Estado (UF)', fontsize=12)

# Adicionando uma linha da média nacional para comparação
plt.axhline(delay_data['dias_p_internar'].mean(), color='red', linestyle='--', label='Média Nacional')
plt.legend()

plt.tight_layout()
plt.show()

#todo "Embora tenhamos 6 milhões de notificações, 
#apenas 147mil dos casos apresentam registro de internação completo, 
#o que aponta para uma necessidade de melhoria no preenchimento das fichas de vigilância..."
# Isso é um insight de gestão pública fortíssimo!

# %%
delay_data

# %%
# identificando os "Sinais de Alerta". Vamos calcular o Risco Relativo (RR). Em termos simples: "Quem tem o sintoma X tem quantas vezes mais chance de morrer do que quem não tem?"

symptoms = [ 
    'FEBRE' , 'MIALGIA' ,'CEFALEIA' , 'EXANTEMA' , 'VOMITO' , 'NAUSEA' , 'DOR_COSTAS' , 'CONJUNTVIT' , 'ARTRITE',
    'ARTRALGIA' , 'PETEQUIA_N', 'LEUCOPENIA', 'LACO' , 'DOR_RETRO'
]

risks = []

for s in symptoms:
    #letalidade entre quem tem o sintoma
    w_symptom = df_analysis[df_analysis[s] == 1]['obito_binario'].mean()
    ##letalidade entre quem NAO tem o sintoma
    wo_symptom = df_analysis[df_analysis[s] != 1]['obito_binario'].mean()

    #risco relativo: se rr > 1, o sintoma aumenta a chance de obito
    rr = w_symptom / wo_symptom if wo_symptom > 0 else 0 
    risks.append({'Sintoma': s, 'Risco_Relativo': rr, 'Prevalencia': df_analysis[s].mean() * 100})

df_risk = pd.DataFrame(risks).sort_values('Risco_Relativo', ascending=False)
# %%
df_risk.head(20)
# %% GRAFICO DE BARRAS HORIZONTAIS MOSTRANDO OS SINTOMAS E COMO ELES AUMENTAM O RISCO DE MORTE DA DENGUE
#todo explicar no grafico como funciona esse risco e o que sao os riscos com valor menor de 1.0
plt.figure(figsize=(12, 8))
# linha vertical no risco neutro
plt.axvline(1, color='grey', linestyle='--', alpha=0.5)

ax = sns.barplot(
    data=df_risk, 
    x='Risco_Relativo', 
    y='Sintoma', 
    palette='mako',
    edgecolor='black',
    alpha=0.8
    )

# Adicionando uma linha vertical no 1.0 (Risco Neutro)

ax.set_title('Risco Relativo de Óbito por Sintoma - Dengue 2024', fontsize=16, pad=20)
ax.set_xlabel('Quantas vezes aumenta o risco de óbito (RR)', fontsize=12)
plt.figtext(0.5, 0.01, "*Valores acima de 1 indicam que o sintoma está associado a maior mortalidade.", ha="center", fontsize=10, style='italic')

# Adicionando os valores nas barras
for i in ax.containers:
    ax.bar_label(i, fmt='%.2f x', padding=5)

plt.show()
# %%
df_analysis['obito_binario'].value_counts(dropna=False)
# %% INDICADOR DE FREQUENCIA/PREVALENCIA DOS SINTOMAS POR IDADE DA DENGUE
# todo mudar o background para cinza

# 1. Criando Faixas Etárias para o eixo Y
bins = [0, 12, 18, 40, 60, 100]
labels = ['0-12', '13-18', '19-30', '31-60', '60+']
df_analysis['FAIXA_ETARIA'] = pd.cut(df_analysis['idade_calculada'], bins=bins, labels=labels)

# 2. Pivotando os dados: média de cada sintoma por faixa etária
heatmap_data = df_analysis.groupby('FAIXA_ETARIA')[symptoms].mean()

# 3. Plotando
plt.figure(figsize=(14, 8), facecolor="#f5f5f5")
sns.heatmap(heatmap_data, 
            annot=True,
            cmap='mako_r', 
            fmt='.1%',
            cbar_kws={'label': 'Frequência'},
            linewidths=1.0,
            annot_kws={'size': 10, 'weight': 'bold'},
            ),

plt.tick_params(left=False, bottom=False)

plt.title('Frequência dos Sintomas da Dengue por Faixa Etária (2024)', fontsize=16, pad=20, fontweight='bold')
plt.xlabel('Sintomas', fontsize=12, fontweight='bold', labelpad=15)
plt.ylabel('Faixa Etária (Anos)', fontsize=12, fontweight='bold', labelpad=15)

plt.xticks(rotation=45, ha='right')
plt.yticks(rotation=0)

plt.show()

# %%
dengue.head()
# %%
# todo ver o que fazer com este grafico! plt.figure(figsize=(10, 6))

# Criando o gráfico de violino comparando idades por desfecho
# sns.violinplot(data=df_analysis, x='falecido', y='idade_calculada', palette='Set2', inner='quartile')

# plt.title('Distribuição de Idade por Desfecho do Caso', fontsize=16, pad=20)
# plt.ylabel('Idade do Paciente', fontsize=12)
# plt.xlabel('Desfecho Final', fontsize=12)
# plt.show()

# %%
bins = [0, 12, 18, 40, 60, 100]
labels = ['0-12', '13-18', '19-30', '31-60', '60+']
if 'FAIXA_ETARIA' not in df_analysis.columns:
    df_analysis['FAIXA_ETARIA'] = pd.cut(df_analysis['idade_calculada'], bins=bins, labels=labels)

# 2. Calculando a Letalidade por Sintoma e Idade
letalidade_sintomas = {}

for s in symptoms:
    # Filtramos APENAS os pacientes que TIVERAM o sintoma (s == 1)
    df_com_sintoma = df_analysis[df_analysis[s] == 1]
    
    # Calculamos a taxa de óbito dentro desse grupo específico
    letalidade_sintomas[s] = df_com_sintoma.groupby('FAIXA_ETARIA', observed=False)['obito_binario'].mean() * 100

# 3. Criando o DataFrame para o Heatmap
df_heatmap_letalidade = pd.DataFrame(letalidade_sintomas)

# %%
plt.figure(figsize=(14, 8), facecolor='#f5f5f5')

ax = sns.heatmap(
    df_heatmap_letalidade, 
    annot=True, 
    fmt=".2f",         
    cmap='mako_r',     
    linewidths=1.0,
    cbar_kws={'label': 'Taxa de Letalidade (%)'},
    annot_kws={'size': 10, 'weight': 'bold'}
)

plt.title('O Peso da Idade e do Sintoma: Letalidade da Dengue (2024)', fontsize=16, pad=20, fontweight='bold')
plt.xlabel('Sintoma Apresentado', fontsize=12, fontweight='bold', labelpad=15 )
plt.ylabel('Faixa Etária (Anos)', fontsize=12, fontweight='bold', labelpad=15)

plt.xticks(rotation=45, ha='right')
plt.yticks(rotation=0)

plt.show()
# %%
df_states = dengue['SG_UF_NOT'].value_counts().reset_index()
df_states.columns = ['uf', 'notificacoes']
df_states
# %%
#todo fazer por estado separado e mapa de calor por municipio, usando os codigos dos municipios.

import plotly.express as px
geojson_url = "https://raw.githubusercontent.com/codeforamerica/click_that_hood/master/public/data/brazil-states.geojson"

fig = px.choropleth(
    df_states,
    geojson=geojson_url,
    locations='uf',        # Coluna do DF com as siglas (SP, MG, etc)
    featureidkey="properties.sigla", # Chave do JSON que combina com a coluna
    color='notificacoes',
    color_continuous_scale="Reds",
    scope="south america",
    title='Densidade de Notificações de Dengue por Estado - 2024'
)

fig.update_geos(fitbounds="locations", visible=False)
fig.show()
# %%
# Grafico de Curva Epidemiológica, para checar se existe alguma relacao sazonal com a doenca.

dengue['MES_NOME'] = dengue['DT_SIN_PRI'].dt.month_name()
# Criamos também uma coluna numérica para ordenar os meses corretamente
dengue['MES_NUM'] = dengue['DT_SIN_PRI'].dt.month

# 3. Agrupar os dados por mês
casos_por_mes = dengue.groupby(['MES_NUM', 'MES_NOME']).size().reset_index(name='TOTAL_CASOS')
casos_por_mes = casos_por_mes.sort_values('MES_NUM')

# Tradução dos meses para Português (Opcional, mas recomendado para o LinkedIn)
meses_pt = {
    'January': 'Jan', 'February': 'Fev', 'March': 'Mar', 'April': 'Abr',
    'May': 'Mai', 'June': 'Jun', 'July': 'Jul', 'August': 'Ago',
    'September': 'Set', 'October': 'Out', 'November': 'Nov', 'December': 'Dez'
}
casos_por_mes['MES_NOME'] = casos_por_mes['MES_NOME'].map(meses_pt)

# 4. Plotagem
plt.figure(figsize=(12, 6))

# Gráfico de linha
sns.lineplot(data=casos_por_mes, x='MES_NOME', y='TOTAL_CASOS', marker='o', color="#3078d6", linewidth=3)

# Preenchimento abaixo da linha para dar o efeito de "Área"
plt.fill_between(casos_por_mes['MES_NOME'], casos_por_mes['TOTAL_CASOS'], color="#307dd6", alpha=0.2)

get_y = lambda mes: casos_por_mes.loc[casos_por_mes['MES_NOME'] == mes, 'TOTAL_CASOS'].values[0]

arrowprops = dict(
    arrowstyle="->",
    connectionstyle="angle,angleA=0,angleB=90,rad=10")

plt.annotate('Aceleração Precoce:\nSurto iniciado antes do esperado', 
             xy=('Fev', get_y('Fev')), xytext=('Jan', get_y('Fev')*0.2),
             arrowprops=arrowprops,
             fontsize=10, style='italic', color='#636e72')

plt.annotate('Pico Histórico: Forte influência do fenômeno El Niño,\nque causou temperaturas recordes e chuvas acima da média.', 
             xy=('Mar', get_y('Mar')), xytext=('Mai', get_y('Mar')*0.9),
             arrowprops=arrowprops,
             fontsize=10, style='italic', color='#636e72')

plt.annotate('Início da Queda:\nRedução sazonal das chuvas\ne temperaturas', 
             xy=('Mai', get_y('Mai')), xytext=('Ago', get_y('Mai')*0.7),
             arrowprops=arrowprops,
             fontsize=10, style='italic', color='#636e72')

plt.annotate('Efeito da Estiagem:\nFim da janela de\ntransmissão acelerada', 
             xy=('Jul', get_y('Jul')), xytext=('Ago', get_y('Jul')*2.5),
             arrowprops=arrowprops,
             fontsize=10, style='italic', color='#636e72')

plt.title('Sazonalidade da Dengue em 2024', fontsize=14, pad=15, fontweight='bold', loc='left')
plt.xlabel('Mês de Início dos Sintomas', fontsize=11, fontweight='bold')
plt.ylabel('Número de Notificações (Milhões)', fontsize=11, fontweight='bold')

# Melhorando a formatação dos números no eixo Y (ex: 1.0M em vez de 1000000)
def format_milhoes(x, pos):
    return f'{x/1e6:.1f}M'
plt.gca().yaxis.set_major_formatter(plt.FuncFormatter(format_milhoes))

sns.despine()
plt.grid(axis='both', linestyle='--', alpha=0.6)
plt.show()
# %%
dengue.head()
# %%
import geopandas as gpd
letalidade_uf = df_analysis.groupby('SG_UF_NOT')['obito_binario'].mean() * 100
letalidade_uf = letalidade_uf.reset_index()

# 2. Carregar o mapa do Brasil (Shapefile ou GeoJSON)
# O Geopandas já tem alguns mapas, mas o do IBGE é melhor. 
# Você pode baixar o 'shx' do site do IBGE ou usar um link direto:
url_mapa = "https://raw.githubusercontent.com/codeforamerica/click_that_hood/master/public/data/brazil-states.geojson"
mapa_brasil = gpd.read_file(url_mapa)

# 3. Unir seus dados de letalidade com o desenho do mapa
# 'sigla' é o nome da coluna no GeoJSON que tem 'SP', 'MG', etc.
mapa_final = mapa_brasil.merge(letalidade_uf, left_on='sigla', right_on='SG_UF_NOT')

# 4. Plotar
fig, ax = plt.subplots(1, 1, figsize=(12, 12))

mapa_final.plot(column='obito_binario', 
                cmap='YlOrRd', 
                legend=True, 
                ax=ax,
                legend_kwds={'label': "Taxa de Letalidade (%)", 'orientation': "horizontal"})

ax.set_title('Geografia do Risco: Taxa de Letalidade por UF', fontsize=16)
ax.axis('off') # Remove as coordenadas (lat/long) para ficar limpo

plt.show()
