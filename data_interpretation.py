# %%
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import geopandas as gpd

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
#Grafico de Taxa de Letalidade por Raça.
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
plt.title('Taxa de Letalidade por Raça/Cor - Dengue 2024', fontsize=14, pad=10, fontweight='bold', loc='left')
plt.xlabel('Letalidade (%)', fontsize=12, fontweight='bold')
plt.ylabel('Raça/Cor', fontsize=12, fontweight='bold')

for i in ax.containers:
    ax.bar_label(i, fmt='%.3f%%', padding=5)

sns.despine()
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
plt.title('O Impacto da Instrução Escolar na Sobrevivência', fontsize=15, pad=15, fontweight='bold', loc='left')
plt.xlabel('Letalidade (%)', fontsize=12, fontweight='bold')
plt.ylabel('Nível de Escolaridade', fontsize=12, fontweight='bold')

sns.despine()
plt.tight_layout()
plt.show()

# %%
dengue['rotulo_escolaridade'].value_counts()
# %% Visualizando o Prazo Assistencial por Estado (UF).
# 147720 linhas (eh possivel tirar alguma conclusao disso? resposta: SIM!)

delay_data = dengue[(dengue['dias_p_internar'] >= 0) & (dengue['dias_p_internar'] <=20 )].copy()

delay_data.shape
# %% Ordenando os estados pela média de tempo em dias para internar
delay_order = delay_data.groupby('SG_UF_NOT')['dias_p_internar'].mean().sort_values(ascending=True).index

# %%
#: Grafico de barras mostrando a media de tempo de quando o paciente sentiu os primeiros sintomas e quando foi internado. (por UF) 
plt.figure(figsize=(12, 8))

ax = sns.barplot(
    data=delay_data,
    x='SG_UF_NOT',
    y='dias_p_internar',
    order=delay_order,
    palette='mako_r',
    errorbar=None,
)

ax.set_title('Tempo Médio para Internação por Estado (Sintomas → Internação)', fontsize=16, pad=15, fontweight='bold', loc='left')
ax.set_ylabel('Média de Dias', fontsize=12, fontweight='bold')
ax.set_xlabel('Estado (UF)', fontsize=12, fontweight='bold')

# Adicionando uma linha da média nacional para comparação
plt.axhline(delay_data['dias_p_internar'].mean(), color='red', linestyle='--', label='Média Nacional')
plt.legend()

sns.despine()
plt.tight_layout()
plt.show()

#Embora tenhamos 6 milhões de notificações, 
#apenas 147mil dos casos apresentam registro de internação completo, 
#o que aponta para uma necessidade de melhoria no preenchimento das fichas de vigilância..."
# porem uma amostra de 147 mil casos, ainda sim é uma amostra grande e confiavel para tirarmos conclusoes.

# %%
delay_data

# %%
# identificando os "Sinais de Alerta". Vamos calcular o Risco Relativo (RR).

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
# O risco relativo em termos simples: "Quem tem o sintoma X tem quantas vezes mais chance de morrer do que quem não tem?"
# Sintomas com multiplicador bem menores que 1 mostram que o grupo que tem esse sintoma morre menos do que o grupo que não tem.
# não significa que o sintoma "cura", mas sim que ele é um marcador de um caso típico e menos perigoso.
plt.figure(figsize=(12, 8))

ax = sns.barplot(
    data=df_risk, 
    x='Risco_Relativo', 
    y='Sintoma', 
    palette='mako',
    edgecolor='black',
    alpha=0.8
    )

ax.set_title('Risco Relativo de Óbito por Sintoma - Dengue 2024', fontsize=16, pad=15, fontweight='bold', loc='left')
ax.set_xlabel('Quantas vezes aumenta o risco de óbito (RR)', fontsize=12)
plt.figtext(0.5, 0.01, "*Valores acima de 1 indicam que o sintoma está associado a maior mortalidade.", ha="center", fontsize=10, style='italic')
plt.grid(axis='x', linestyle='--', alpha=0.6)

# Adicionando os valores nas barras
for i in ax.containers:
    ax.bar_label(i, fmt='%.2f x', padding=5)

sns.despine()
plt.show()
# %%
df_analysis['obito_binario'].value_counts(dropna=False)
# %% INDICADOR DE FREQUENCIA/PREVALENCIA DOS SINTOMAS POR IDADE DA DENGUE
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
            cbar_kws={'label': 'Frequência (%)'},
            linewidths=1.0,
            annot_kws={'size': 10, 'weight': 'bold'},
            ),

plt.tick_params(left=False, bottom=False)

plt.title('Frequência dos Sintomas da Dengue por Faixa Etária (2024)', fontsize=16, pad=15, fontweight='bold', loc='left')
plt.xlabel('Sintomas', fontsize=12, fontweight='bold', labelpad=15)
plt.ylabel('Faixa Etária (Anos)', fontsize=12, fontweight='bold', labelpad=15)

plt.xticks(rotation=45, ha='right')
plt.yticks(rotation=0)

plt.show()

# %%
dengue.head()

# %%
bins = [0, 12, 18, 40, 60, 100]
labels = ['0-12', '13-18', '19-30', '31-60', '60+']
if 'FAIXA_ETARIA' not in df_analysis.columns:
    df_analysis['FAIXA_ETARIA'] = pd.cut(df_analysis['idade_calculada'], bins=bins, labels=labels)

# 2. Calculando a Letalidade por Sintoma e Idade
symptom_letality = {}

for s in symptoms:
    # Filtramos APENAS os pacientes que TIVERAM o sintoma (s == 1)
    positive_symptom = df_analysis[df_analysis[s] == 1]
    
    # Calculamos a taxa de óbito dentro desse grupo específico
    symptom_letality[s] = positive_symptom.groupby('FAIXA_ETARIA', observed=False)['obito_binario'].mean() * 100

# 3. Criando o DataFrame para o Heatmap
letality_heatmap = pd.DataFrame(symptom_letality)

# %%
plt.figure(figsize=(14, 8), facecolor='#f5f5f5')

ax = sns.heatmap(
    letality_heatmap, 
    annot=True, 
    fmt=".2f",         
    cmap='mako_r',     
    linewidths=1.0,
    cbar_kws={'label': 'Taxa de Letalidade (%)'},
    annot_kws={'size': 10, 'weight': 'bold'}
)

plt.title('O Peso da Idade e do Sintoma: Letalidade da Dengue (2024)', fontsize=16, pad=15, fontweight='bold', loc='left')
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
# Grafico de Curva Epidemiológica, para checar se existe alguma relacao sazonal com a doenca.

dengue['MES_NOME'] = dengue['DT_SIN_PRI'].dt.month_name()
# coluna numérica para ordenar os meses corretamente
dengue['MES_NUM'] = dengue['DT_SIN_PRI'].dt.month

# agrupando os dados por mês
monthly_cases = dengue.groupby(['MES_NUM', 'MES_NOME']).size().reset_index(name='TOTAL_CASOS')
monthly_cases = monthly_cases.sort_values('MES_NUM')

# traducao dos meses para Português
months_pt = {
    'January': 'Jan', 'February': 'Fev', 'March': 'Mar', 'April': 'Abr',
    'May': 'Mai', 'June': 'Jun', 'July': 'Jul', 'August': 'Ago',
    'September': 'Set', 'October': 'Out', 'November': 'Nov', 'December': 'Dez'
}
monthly_cases['MES_NOME'] = monthly_cases['MES_NOME'].map(months_pt)

# 4. Plotagem
plt.figure(figsize=(12, 6))

# Gráfico de linha
sns.lineplot(
    data=monthly_cases, 
    x='MES_NOME', 
    y='TOTAL_CASOS', 
    marker='o', 
    color="#3078d6", 
    linewidth=3
    )

# Preenchimento abaixo da linha para dar o efeito de "Área"
plt.fill_between(monthly_cases['MES_NOME'], monthly_cases['TOTAL_CASOS'], color="#307dd6", alpha=0.2)

get_y = lambda mes: monthly_cases.loc[monthly_cases['MES_NOME'] == mes, 'TOTAL_CASOS'].values[0]

arrowprops = dict(
    arrowstyle="->",
    connectionstyle="angle,angleA=0,angleB=90,rad=10")

plt.annotate('Aceleração Precoce:\nSurto iniciado antes do esperado', 
             xy=('Jan', get_y('Jan')), xytext=('Jan', get_y('Fev')*0.1),
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
state_deaths = df_analysis.groupby('SG_UF_NOT')['obito_binario'].sum()
state_deaths
# %%
#A letalidade é o melhor indicador de risco, porem em estados com pouquíssimas notificações, a taxa pode ser 'inflada' artificialmente.
# o calculo da letalidade eh o seguinte: (total de obitos na UF/total de casos na UF) * 100 pra virar porcentagem. 

# trazendo tanto % quanto o número absoluto no mesmo lugar
uf_stats = df_analysis.groupby('SG_UF_NOT')['obito_binario'].agg(['mean', 'sum']).reset_index()
uf_stats['mean'] = uf_stats['mean'] * 100

# %%
#Grafico de 'Geografia de Risco', mostrando a % de letalidade e numero exato de morte por estado.
map_path = "https://raw.githubusercontent.com/codeforamerica/click_that_hood/master/public/data/brazil-states.geojson"
brasil_map = gpd.read_file(map_path)

final_map = brasil_map.merge(uf_stats, left_on='sigla', right_on='SG_UF_NOT')

fig, ax = plt.subplots(1, 1, figsize=(15, 12))

final_map.plot(column='mean', 
                cmap='mako_r', 
                legend=True, 
                ax=ax,
                linewidth=0.3,
                edgecolor='black',
                legend_kwds={'label': "Taxa de Letalidade (%)", 'orientation': "vertical", 'shrink': 0.5},
                missing_kwds={'color' : 'gray'}
                )

for idx, row in final_map.iterrows():
    # Pegamos o centro geográfico de cada estado para colocar o texto
    centroid = row.geometry.centroid
    coords = (centroid.x, centroid.y)
    
    # Criamos o texto: Sigla do Estado + Total de Mortes
    # Exemplo: "SP: 150"
    label = f"{row['sigla']}\n{int(row['sum'])}"
    
    ax.annotate(text=label, 
                xy=coords, 
                ha='center', 
                va='center',
                fontsize=9, 
                fontweight='bold',
                color='white' if row['mean'] > 0.2 else 'black',
                bbox=dict(boxstyle="round,pad=0.1", fc="none", ec="none", alpha=0.5))
                

ax.set_title('Geografia de Risco: Letalidade (%) e Total de óbitos por UF', fontsize=17, pad=15, fontweight='bold', loc='center')
ax.axis('off')

plt.figtext(0.5, 0.01, "*Cálculo da Taxa de Letalidade: (Total de óbitos na UF / Total de casos na UF) * 100.", ha="center", fontsize=12, style='italic')
plt.show
# %%
#
summary = df_analysis.groupby('SG_UF_NOT').agg(
    state_deaths=('obito_binario', 'sum'),
    total_notifs=('SG_UF_NOT', 'count')
).reset_index()

summary

# %%
