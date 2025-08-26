import pandas as pd
import numpy as np

# Carregar a tabela de dados do restaurante
tabela_dados = pd.read_csv('restaurante.csv', sep=';')

# Função para calcular a entropia de uma série de dados
# Essa função utiliza a equação 2 para calculo de entropia(Shannon)
def calcular_entropia(serie_dados):
    proporcoes = serie_dados.value_counts(normalize=True)
    # O 1e-9 é para evitar log de zero
    return -sum(proporcoes * np.log2(proporcoes + 1e-9))

# Função para calcular a entropia condicional de um atributo em relação à classe
def calcular_entropia_atributo(tabela, nome_atributo, nome_classe='Conclusao'):
    num_total_registros = len(tabela)
    entropia_ponderada = 0

    for valor in tabela[nome_atributo].unique():
        sub_tabela = tabela[tabela[nome_atributo] == valor]
        peso = len(sub_tabela) / num_total_registros
        entropia_ponderada += peso * calcular_entropia(sub_tabela[nome_classe])

    return entropia_ponderada


# 1º Nível: Calcular a entropia e o ganho de informação para a raiz
entropia_total = calcular_entropia(tabela_dados['Conclusao'])
print(f"Entropia Total: {entropia_total:.4f}\n")

atributos = [col for col in tabela_dados.columns if col != 'Conclusao']
resultados = {}
for nome_atributo in atributos:
    entropia_atributo = calcular_entropia_atributo(tabela_dados, nome_atributo, 'Conclusao')
    ganho = entropia_total - entropia_atributo
    resultados[nome_atributo] = {'entropia': entropia_atributo, 'ganho': ganho}

resultados_df = pd.DataFrame(resultados).T
resultados_df = resultados_df.sort_values(by='ganho', ascending=False)

print("Entropia e Ganho de Informação para cada atributo (RAIZ):")
print(resultados_df.to_string(float_format="%.4f"))

# Identificar e exibir a raiz
raiz = resultados_df.index[0]
print(f"\n>>>> A raiz da árvore de decisão é '{raiz}', pois tem o maior ganho de informação. <<<<\n")

# 2º Nível: Calcular a entropia e o ganho para os nós filhos
valores_raiz = tabela_dados[raiz].unique()
print(f"Calculando o 2º nível a partir da raiz '{raiz}':\n")

for valor_raiz in valores_raiz:
    print(f"---- Subconjunto para '{raiz}' = '{valor_raiz}' ----")
    sub_tabela = tabela_dados[tabela_dados[raiz] == valor_raiz].copy()

    # Calcular a entropia do subconjunto
    entropia_subconjunto = calcular_entropia(sub_tabela['Conclusao'])
    print(f"Entropia do subconjunto: {entropia_subconjunto:.4f}\n")
    
     # Calcular entropia e ganho para os atributos restantes
     # Utiliza da equação de ganho Eq 1.
    atributos_restantes = [col for col in sub_tabela.columns if col not in [raiz, 'Conclusao']]
    resultados_nivel_2 = {}
    for nome_atributo in atributos_restantes:
        entropia_nivel_2 = calcular_entropia_atributo(sub_tabela, nome_atributo, 'Conclusao')
        ganho_nivel_2 = entropia_subconjunto - entropia_nivel_2
        resultados_nivel_2[nome_atributo] = {'entropia': entropia_nivel_2, 'ganho': ganho_nivel_2}

    resultados_nivel_2_df = pd.DataFrame(resultados_nivel_2).T
    resultados_nivel_2_df = resultados_nivel_2_df.sort_values(by='ganho', ascending=False)
    resultados_nivel_2_df = resultados_nivel_2_df[~resultados_nivel_2_df.index.duplicated(keep='first')]

    print("Entropia e Ganho de Informação para os atributos restantes:")
    print(resultados_nivel_2_df.to_string(float_format="%.4f"))
    print("-" * 40)