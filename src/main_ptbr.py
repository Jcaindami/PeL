# ==============================================================================
# SCRIPT FINAL DE ANÁLISE DE P&L E GERAÇÃO DE RELATÓRIO ESTRATÉGICO
# ==============================================================================

# --- Importações de Bibliotecas ---
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
import matplotlib.lines as mlines
import matplotlib.dates as mdates
from scipy.stats import norm
import numpy as np
from dotenv import load_dotenv
import os
import sys
import glob
import re
from dateutil.relativedelta import relativedelta

# ==============================================================================
# --- DICIONÁRIO DE TRADUÇÃO DAS CONTAS ---
# ==============================================================================
mapa_traducao = {
    'GROSS_SALE': 'Receita Bruta',
    'TAXES': 'Impostos Sobre Venda',
    'NET_REVENUE': 'Receita Líquida',
    'COGS': 'CPV',
    'GROSS_PROFIT': 'Lucro Bruto',
    'Delivery_Services': 'Serviços de Entrega',
    'Store_Supplies': 'Suprimentos da Loja',
    'Rent_Expense': 'Despesas de Aluguel',
    'Utilities': 'Utilidades (Água,Luz,Internet)',
    'Salaries_And_Wages': 'Salários e Encargos',
    'Benefits': 'Benefícios',
    'Depreciation_And_Amortization': 'Depreciação e Amortização',
    'Marketing_And_Advertising': 'Marketing e Publicidade',
    'Maintenance_And_Repairs': 'Manutenção e Reparos',
    'Insurance': 'Seguros',
    'Other_Operating_Expenses': 'Outras Despesas Operacionais',
    'Store_Operating_Income': 'Lucro Operacional',
    'Interest_Expense': 'Despesas Financeiras',
    'Interest_Income': 'Receitas Financeiras',
    'NET_INCOME_BEFORE_TAX': 'Lucro Antes dos Impostos (LAIR)',
    'Income_Tax_Expense': 'Imposto de Renda e Contribuição Social',
    'NET_INCOME': 'Lucro Líquido'
}

# --- DEFINIÇÃO DE TIPOS DE CONTA (EM PORTUGUÊS) ---
# Lista com os nomes das contas de receita/lucro JÁ TRADUZIDOS.

CONTAS_RECEITA = [
    'Venda Bruta', 
    'Vendas de Produtos (Líquida)',
    'Vendas de Não Produtos - Brindes (Líquida)',
    'Lucro Bruto',
    'Venda Líquida',
    'Margem Bruta',
    'Resultado Operacional - SOI',
    'Lucro após Controláveis',
    'Lucro Operacional',
    'P. A. C.'
]
# As demais contas serão automaticamente tratadas como CUSTOS/DESPESAS, onde um Z-Score baixo é POSITIVO.

# ==============================================================================
# --- FUNÇÕES UTILITÁRIAS GLOBAIS ---
# ==============================================================================

def converter_para_float(valor):
    if isinstance(valor, str):
        try: return float(valor.replace('.', '').replace(',', '.'))
        except (ValueError, TypeError): return np.nan
    return valor

# ==============================================================================
# --- FUNÇÕES DE PLOTAGEM E GERAÇÃO DE TEXTO ---
# ==============================================================================

def criar_capa_pagina(pdf_pages, title, referencia, date, logo_path):
    fig, ax = plt.subplots(figsize=(8.5, 11)); ax.axis('off')

    # Ajusta as coordenadas para um layout mais espaçado
    y_logo = 0.95 # Posição vertical do logo
    y_title_group = 0.50 # Posição vertical do nome do grupo (title)
    y_report_type = 0.35 # Posição vertical do tipo de relatório
    y_reference_period = 0.30 # Posição vertical do período de referência
    y_emission_date = 0.1 # Posição vertical da data de emissão

    try:
        logo = plt.imread(logo_path)
        # Ajuste o 'zoom' e o posicionamento para centralizar e dar mais destaque
        imagebox = OffsetImage(logo, zoom=0.6)
        ab = AnnotationBbox(imagebox, (0.5, y_logo), frameon=False, box_alignment=(0.5, 0.5))
        ax.add_artist(ab)
    except FileNotFoundError:
        print(f"AVISO: Arquivo de logo não encontrado em '{logo_path}'")
        ax.text(0.5, y_logo, '[Logo não encontrado]', ha='center', va='center', fontsize=12, color='red')

    # Adiciona os textos da capa
    # Título principal (nome da franquia/empresa) - GRUPO Raízes do Brasil
    ax.text(0.5, y_title_group, title, ha='center', va='center', fontsize=26, weight='bold', color='#2F4F4F') # Cor mais escura
    
    # Relatório de Análise
    ax.text(0.5, y_report_type, 'Desempenho e Eficiência Operacional', ha='center', va='center', fontsize=16, color='#696969')
    ax.text(0.5, y_reference_period, f'Período de Referência: {referencia}', ha='center', va='center', fontsize=14, color='#696969')
    ax.text(0.5, y_emission_date, f'Data de Emissão: {date}', ha='center', va='center', fontsize=10, color='#A9A9A9')
    ax.text(0.5, 0.07, 'CONFIDENCIAL - SOMENTE PARA USO INTERNO', ha='center', va='center', fontsize=9, color='grey')

    # Salva a figura da capa no objeto PDF
    pdf_pages.savefig(fig, bbox_inches='tight')
    plt.close(fig)

def exportar_dataframe_para_pdf(df, pdf_pages, title, max_rows_per_page=20):
    num_rows = len(df)
    if num_rows == 0: return
    num_pages = (num_rows // max_rows_per_page) + (1 if num_rows % max_rows_per_page != 0 else 0)
    for page in range(num_pages):
        start_row, end_row = page * max_rows_per_page, min((page + 1) * max_rows_per_page, num_rows)
        df_page = df.iloc[start_row:end_row]
        fig, ax = plt.subplots(figsize=(12, 8)); ax.axis('off'); fig.suptitle(title, fontsize=14, weight='bold', y=0.98)
        num_cols = len(df_page.columns)
        col_widths = [0.4] + [(0.6 / (num_cols - 1))] * (num_cols - 1) if num_cols > 1 else [1.0]
        table = ax.table(cellText=df_page.values, colLabels=df_page.columns, cellLoc='center', loc='center', bbox=[0, 0, 1, 0.9], colWidths=col_widths)
        table.auto_set_font_size(False); table.set_fontsize(9); table.scale(1.0, 1.5)
        cells = table.get_celld()
        for i in range(len(df_page.index) + 1):
            cells[(i, 0)].set_text_props(ha='left')
            if i == 0:
                cells[(i, 0)].set_text_props(weight='bold', ha='left')
                for j in range(1, num_cols): cells[(i, j)].set_text_props(weight='bold')
        pdf_pages.savefig(fig, bbox_inches='tight'); plt.close(fig)

# --- FUNÇÃO ATUALIZADA ---
def medias_ponderadas_grafico_z_scores(z_scores_transposto, filiais_to_keep, pdf_pages, title, date):
    # Agora a função espera a coluna 'ZScore_Medio_Geral' que foi pré-calculada
    z_scores_sorted = z_scores_transposto.loc[filiais_to_keep, 'ZScore_Medio_Geral'].sort_values(ascending=False)
    cores = [PALETA_CORES["negativo_leve"] if x > 0 else PALETA_CORES["positivo"] for x in z_scores_sorted.values]
    fig, ax = plt.subplots(figsize=(10, 12)); sns.barplot(y=z_scores_sorted.index, x=z_scores_sorted.values, palette=cores, ax=ax)
    ax.set_title(title, fontsize=18, weight='bold', pad=20); #fig.suptitle(f'Data de Emissão: {date}', fontsize=12, y=0.92)
    ax.set_xlabel('Média dos Z-Scores (Desempenho Relativo)', fontsize=14, labelpad=10); ax.set_ylabel('Restaurantes', fontsize=14, labelpad=10)
    ax.axvline(0, color='grey', linewidth=1.5, linestyle='--')
    for index, value in enumerate(z_scores_sorted): ax.text(value, index, f' {value:.2f}', color='black', ha='left' if value >= 0 else 'right', va="center", fontsize=11)
    sns.despine(left=True, bottom=True); pdf_pages.savefig(fig, bbox_inches='tight'); plt.close(fig)

def pontuacao_individual_z_scores(z_scores_transposto, filiais_selecionadas, pdf_pages, title, date):
    for filial in filiais_selecionadas:
        if filial in z_scores_transposto.index:
            z_scores_restaurant = z_scores_transposto.loc[filial].drop(['ZScore_Medio_Geral'], errors='ignore')
            fig, ax = plt.subplots(figsize=(10, 14))
            cores = [PALETA_CORES["negativo_critico"] if z > 2 or z < -2 else PALETA_CORES["negativo_leve"] if z > 0 else PALETA_CORES["positivo"] for z in z_scores_restaurant.values]
            sns.barplot(y=z_scores_restaurant.index, x=z_scores_restaurant.values, palette=cores, ax=ax)
            ax.set_title(f'{title} - {filial}', fontsize=18, weight='bold', pad=20); #plt.suptitle(f'Data de Emissão: {date}', fontsize=12, y=0.92)
            ax.set_xlabel('Z-Score'); ax.set_ylabel('Elementos de Custo e Despesa')
            ax.tick_params(axis='y', labelsize=12 if len(z_scores_restaurant.index) < 30 else 10)
            ax.axvline(0, color='grey', linewidth=1.5, linestyle='--')
            for idx, value in enumerate(z_scores_restaurant): ax.text(value, idx, f' {value:.2f}', color='black', ha='left' if value >= 0 else 'right', va="center")
            sns.despine(left=True, bottom=True); fig.tight_layout(rect=[0, 0, 1, 0.96]); pdf_pages.savefig(fig, bbox_inches='tight'); plt.close(fig)

def identificar_outliers(z_scores, df_normalizada):
    outliers_summary = []
    # Itera sobre o DataFrame transposto para pegar por filial
    for filial in z_scores.index:
        if filial in ['mean', 'std']: continue
        for metrica in z_scores.columns:
            if metrica in ['mean', 'std', 'ZScore_Medio_Geral']: continue
            z_score = z_scores.loc[filial, metrica]
            if z_score > 2 or z_score < -2:
                outliers_summary.append({'Restaurante': filial, 'Elemento de Custo/Despesa': metrica, 'Z-Score': z_score})
    return pd.DataFrame(outliers_summary)

def gerar_pagina_introducao_outliers(pdf_pages, data_referencia):
    """
    Cria uma página de texto dedicada a explicar o que são outliers,
    exibe um gráfico de exemplo e depois explica a normalização.
    """
    # --- FUNÇÃO INTERNA PARA GERAR O GRÁFICO ---
    def gerar_grafico_distribuicao_normal(ax, media, desvio_padrao, valor_outlier):
        # Gerar pontos para a curva de distribuição normal
        x = np.linspace(media - 4*desvio_padrao, media + 4*desvio_padrao, 1000)
        pdf = norm.pdf(x, media, desvio_padrao)

        # Plotar a curva principal
        ax.plot(x, pdf, 'b-', lw=2, label='Curva de Distribuição Normal')

        # Calcular os valores de +2 e -2 desvios padrão
        limite_superior = media + 2*desvio_padrao
        limite_inferior = media - 2*desvio_padrao

        # Adicionar linhas verticais com legendas simplificadas
        ax.axvline(media, color='black', linestyle='--', label='Média do Grupo')
        ax.axvline(limite_superior, color='red', linestyle='--', label='Limite de Outlier (+/- 2 DP)')
        ax.axvline(limite_inferior, color='red', linestyle='--')

        # Destacar a área "normal"
        x_normal = np.linspace(limite_inferior, limite_superior, 500)
        ax.fill_between(x_normal, 0, norm.pdf(x_normal, media, desvio_padrao), color='lightblue', alpha=0.5, label='Valores Comuns')

        # Adicionar a linha e a anotação para o valor do outlier
        ax.axvline(valor_outlier, color='orange', linestyle='-', lw=2, label='Valor Outlier')
        ax.annotate(f'Valor Outlier:\n{valor_outlier:.2f}', xy=(valor_outlier, pdf.max()*0.1),
                    xytext=(valor_outlier, pdf.max()*0.4),
                    arrowprops=dict(facecolor='orange', shrink=0.05, width=1.5, headwidth=8),
                    ha='center', fontsize=9, bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="orange", lw=1))

        # Adicionar textos com os valores diretamente no gráfico
        posicao_y_texto = pdf.max() * 0.05
        ax.text(media, posicao_y_texto, f'{media:.2f}', ha='center', va='bottom', fontsize=9, weight='bold')
        ax.text(limite_superior, posicao_y_texto, f'{limite_superior:.2f}', ha='center', va='bottom', fontsize=9, weight='bold', color='red')
        ax.text(limite_inferior, posicao_y_texto, f'{limite_inferior:.2f}', ha='center', va='bottom', fontsize=9, weight='bold', color='red')
        
        # Configurações do gráfico
        ax.set_title('Exemplo Visual de um Outlier', fontsize=14, pad=10)
        ax.set_xlabel('Valor do Indicador de Custo', fontsize=10)
        ax.set_ylabel('Densidade de Probabilidade', fontsize=10)
        ax.grid(True, linestyle='--', alpha=0.6)
        ax.legend(fontsize=9)
        ax.tick_params(axis='both', which='major', labelsize=8)
        ax.set_yticks([])

    # --- CONFIGURAÇÃO DA PÁGINA DO PDF ---
    fig = plt.figure(figsize=(8.5, 11))
    
    # --- TÍTULOS ---
    fig.suptitle(f"Análise Detalhada de Outliers - {data_referencia}", fontsize=16, weight='bold', y=0.97)
    fig.text(0.5, 0.92, "Como Enteder está Análise", ha='center', fontsize=12, weight='bold')

    # --- TEXTO 1: Explicação de Outliers ---
    texto_definicao = (
        "1. O que são Outliers?\n"
        "Outliers (ou 'pontos fora da curva') são valores que se destacam por serem muito maiores ou menores que a "
        "média de um grupo. Nesta análise, eles nos ajudam a identificar rapidamente os custos que estão incomumente "
        "altos ou baixos em uma loja específica. Consideramos um custo como 'outlier' se ele estiver a mais de 2 "
        "desvios padrão de distância da média do grupo.\n\n"
        "O gráfico abaixo ilustra este conceito: valores que caem na área azul são considerados comuns, enquanto valores "
        "fora dela, como o 'Valor Outlier' destacado, são investigados."
    )
    fig.text(0.08, 0.88, texto_definicao, ha='left', va='top', fontsize=11, wrap=True, linespacing=1.5)

    # --- GRÁFICO ---
    # AJUSTE: A coordenada Y foi diminuída de 0.42 para 0.40 para mover o gráfico para baixo.
    chart_ax = fig.add_axes([0.1, 0.40, 0.8, 0.3])
    gerar_grafico_distribuicao_normal(ax=chart_ax, media=-1524.82, desvio_padrao=591.58, valor_outlier=-17.76)

    # --- TEXTO 2: Explicação de Normalização ---
    texto_normalizacao = (
        "2. A Importância da Normalização (Comparando Lojas de Tamanhos Diferentes)\n"
        "Para comparar as lojas de forma justa, não olhamos para o gasto absoluto (em R$), pois uma loja que fatura "
        "muito naturalmente terá custos maiores. Em vez disso, 'normalizamos' os valores, convertendo cada custo em "
        "uma proporção da receita da própria loja. Assim, quando dizemos que um custo é um outlier, estamos "
        "dizendo que o percentual da receita que aquela loja gasta com aquele item é muito diferente do "
        "percentual médio gasto pelo resto do grupo."
    )
    # AJUSTE: A coordenada Y foi diminuída de 0.35 para 0.33 para acompanhar o gráfico.
    fig.text(0.08, 0.33, texto_normalizacao, ha='left', va='top', fontsize=11, wrap=True, linespacing=1.5)

    # --- SALVA A PÁGINA ---
    pdf_pages.savefig(fig, bbox_inches='tight')
    plt.close(fig)

def gerar_analise_outliers_em_topicos(pdf_pages, outliers_df, df_normalizada):
    """
    Cria uma ou mais páginas de texto com a análise detalhada de cada outlier
    em formato de tópicos, com espaçamento ajustado para evitar sobreposição.
    """
    if outliers_df.empty:
        print("Nenhum outlier encontrado para gerar a análise em tópicos.")
        return

    outliers_agrupados = outliers_df.groupby('Restaurante')
    fig, ax = plt.subplots(figsize=(8.5, 11))
    ax.axis('off')

    fig.text(0.1, 0.95, "ANÁLISE DOS OUTLIERS IDENTIFICADOS", fontsize=12, weight='bold')

    y_pos = 0.90  # Posição inicial do texto dinâmico
    
    for restaurante, grupo in outliers_agrupados:
        # Checa se há espaço para o próximo restaurante, se não, cria nova página
        if y_pos < 0.15:
            pdf_pages.savefig(fig, bbox_inches='tight'); plt.close(fig)
            fig, ax = plt.subplots(figsize=(8.5, 11)); ax.axis('off')
            fig.text(0.1, 0.95, "ANÁLISE DOS OUTLIERS IDENTIFICADOS (continuação)", fontsize=12, weight='bold')
            y_pos = 0.90

        fig.text(0.1, y_pos, f"{restaurante}", fontsize=11, weight='bold')
        y_pos -= 0.04  # Espaçamento após o nome do restaurante

        for index, outlier in grupo.iterrows():
            metrica = outlier['Elemento de Custo/Despesa']
            z_score = outlier['Z-Score']
            
            valor_real = df_normalizada.loc[metrica, restaurante]
            valor_medio = df_normalizada.loc[metrica, 'mean']
            desvio_padrao = df_normalizada.loc[metrica, 'std']
            
            tipo_outlier = "POSITIVO" if z_score > 0 else "NEGATIVO"
            descricao_outlier = "(Um resultado maior que a média mais duas vezes o desvio padrão)" if z_score > 0 else "(Um resultado menor que a média mais duas vezes o desvio padrão)"
            posicao_relativa = "acima" if tipo_outlier == "POSITIVO" else "abaixo"
            
            texto_principal = (
                f"Para a conta '{metrica}', o indicador de custo foi de {valor_real:.2f}. "
                f"Este valor é um outlier {tipo_outlier} {descricao_outlier}, pois está muito {posicao_relativa} da média do grupo, "
                f"que foi de {valor_medio:.2f} (DP: {desvio_padrao:.2f})."
            )
            texto_conclusao = (
                f"O Z-Score de {z_score:.2f} confirma que o valor está a mais de dois desvios padrão {posicao_relativa} da média."
            )
            
            texto_final = f"- {texto_principal}\n  {texto_conclusao}"
            
            # AJUSTE: Fonte menor e espaçamento vertical maior
            fig.text(0.12, y_pos, texto_final, ha='left', va='top', fontsize=9.5, wrap=True)
            y_pos -= 0.1 # Aumentamos o decremento para dar mais espaço

    pdf_pages.savefig(fig, bbox_inches='tight')
    plt.close(fig)

def plotar_evolucao_zscore_medio(df_temporal, filiais_selecionadas, pdf_pages, titulo_grafico):
    print(f"Gerando gráfico: {titulo_grafico}..."); dados_grafico = df_temporal[(df_temporal['Metrica'] == 'ZScore_Medio_Geral') & (df_temporal['Restaurante'].isin(filiais_selecionadas))]
    if dados_grafico.empty: return
    fig, ax = plt.subplots(figsize=(14, 8)); sns.lineplot(data=dados_grafico, x='Data', y='ZScore', hue='Restaurante', marker='o', ax=ax)
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%m/%y')); ax.xaxis.set_major_locator(mdates.MonthLocator(interval=1)); fig.autofmt_xdate()
    ax.axhline(0, color='grey', linestyle='--', label='Média do Grupo'); ax.set_title(titulo_grafico, fontsize=18, weight='bold', pad=20)
    ax.set_ylabel('Z-Score Médio (Pior > 0 > Melhor)'); ax.set_xlabel('Competência')
    ax.legend(title='Restaurantes', bbox_to_anchor=(1.05, 1), loc='upper left'); fig.tight_layout(); pdf_pages.savefig(fig, bbox_inches='tight'); plt.close(fig)

def plotar_evolucao_metrica_especifica(df_temporal, metrica, filiais_selecionadas, pdf_pages, titulo_grafico):
    print(f"Gerando gráfico: {titulo_grafico} para a métrica {metrica}..."); dados_grafico = df_temporal[(df_temporal['Metrica'] == metrica) & (df_temporal['Restaurante'].isin(filiais_selecionadas))]
    if dados_grafico.empty: return
    fig, ax = plt.subplots(figsize=(14, 8)); sns.lineplot(data=dados_grafico, x='Data', y='ZScore', hue='Restaurante', marker='o', ax=ax)
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%m/%y')); ax.xaxis.set_major_locator(mdates.MonthLocator(interval=1)); fig.autofmt_xdate()
    ax.axhline(0, color='grey', linestyle='--'); ax.set_title(titulo_grafico, fontsize=18, weight='bold', pad=20)
    ax.set_ylabel('Z-Score'); ax.set_xlabel('Competência'); ax.legend(title='Restaurantes', bbox_to_anchor=(1.05, 1), loc='upper left')
    fig.tight_layout(); pdf_pages.savefig(fig, bbox_inches='tight'); plt.close(fig)

def analisar_e_plotar_comparativo_anual_valores(dados_historicos, metrica, filiais_selecionadas, data_atual, pdf_pages):
    # (Esta função permanece a mesma da versão anterior, garantindo que está completa)
    data_anterior = data_atual - relativedelta(years=1)
    if data_atual not in dados_historicos or data_anterior not in dados_historicos: return
    dados_atuais, dados_anteriores = dados_historicos[data_atual], dados_historicos[data_anterior]
    valores = []
    for filial in filiais_selecionadas:
        if filial in dados_atuais["brutos"].columns and filial in dados_anteriores["brutos"].columns and metrica in dados_atuais["brutos"].index and metrica in dados_anteriores["brutos"].index:
            valores.append([filial, data_atual.strftime('%b/%Y'), dados_atuais["brutos"].loc[metrica, filial], dados_atuais["z_scores"].loc[metrica, filial]])
            valores.append([filial, data_anterior.strftime('%b/%Y'), dados_anteriores["brutos"].loc[metrica, filial], dados_anteriores["z_scores"].loc[metrica, filial]])
    if not valores: return
    df_plot = pd.DataFrame(valores, columns=['Restaurante', 'Ano', 'Valor', 'ZScore'])
    fig, ax = plt.subplots(figsize=(12, 8)); sns.barplot(data=df_plot, y='Restaurante', x='Valor', hue='Ano', orient='h', ax=ax)
    ax.set_title(f"Comparativo Anual (YoY) de Valores - {metrica}", fontsize=16, weight='bold'); ax.set_xlabel("Valor Absoluto (R$)"); ax.set_ylabel("Restaurante")
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'R$ {int(x):,}'.replace(',', '.'))); pdf_pages.savefig(fig, bbox_inches='tight'); plt.close(fig)
    fig, ax = plt.subplots(figsize=(8.5, 11)); ax.axis('off'); fig.text(0.5, 0.95, f"Análise Comparativa de Valores (YoY) - {metrica}", ha='center', va='center', fontsize=16, weight='bold')
    y_pos = 0.90
    for filial in filiais_selecionadas:
        df_filial = df_plot[df_plot['Restaurante'] == filial]
        if len(df_filial) < 2: continue
        atual, anterior = df_filial[df_filial['Ano'] == data_atual.strftime('%b/%Y')].iloc[0], df_filial[df_filial['Ano'] == data_anterior.strftime('%b/%Y')].iloc[0]
        melhora_zscore = atual['ZScore'] < anterior['ZScore']; texto_zscore = f"O Z-Score {'melhorou' if melhora_zscore else 'piorou'} (de {anterior['ZScore']:.2f} para {atual['ZScore']:.2f})."
        aumento_valor = atual['Valor'] > anterior['Valor']; texto_valor = f"O valor absoluto {'aumentou' if aumento_valor else 'diminuiu'}, passando de R${anterior['Valor']:_.2f} para R${atual['Valor']:_.2f}.".replace('.',',').replace('_','.')
        conclusao = "ATENÇÃO: Apesar da melhora no Z-Score, o gasto/valor absoluto aumentou." if melhora_zscore and aumento_valor else ""
        fig.text(0.1, y_pos, f"{filial}", fontsize=11, weight='bold'); y_pos -= 0.03
        fig.text(0.12, y_pos, f"- {texto_zscore}\n- {texto_valor}", ha='left', va='top', fontsize=10); y_pos -= 0.05
        if conclusao: fig.text(0.12, y_pos, conclusao, ha='left', va='top', fontsize=10, color='red', weight='bold'); y_pos -= 0.04
    pdf_pages.savefig(fig, bbox_inches='tight'); plt.close(fig)

def plotar_tendencia_outlier_negativo(pdf_pages, restaurante, metrica, df_temporal):
    """
    Cria e salva no PDF um gráfico de série temporal para o Z-Score de uma métrica
    específica de um restaurante, destacando as faixas de normalidade.
    """
    print(f"Gerando gráfico de tendência para o outlier: {restaurante} - {metrica}")
    
    # Filtra os dados históricos para o restaurante e métrica específicos (últimos 12 meses)
    dados_grafico = df_temporal[(df_temporal['Restaurante'] == restaurante) & (df_temporal['Metrica'] == metrica)].tail(12)
    
    if len(dados_grafico) < 2:
        print("  -> Dados históricos insuficientes para gerar o gráfico.")
        return

    fig, ax = plt.subplots(figsize=(12, 7))
    sns.lineplot(data=dados_grafico, x='Data', y='ZScore', marker='o', ax=ax, lw=2.5, markersize=8)

    # Linhas de referência
    ax.axhline(0, color='grey', linestyle='--', label='Média do Grupo')
    ax.axhline(2, color='red', linestyle=':', lw=1.5, label='Limite Outlier Negativo (+2 DP)')
    ax.axhline(-2, color='green', linestyle=':', lw=1.5, label='Limite Outlier Positivo (-2 DP)')
    
    # Preenche a área "crítica" (acima de +2) com vermelho
    ax.fill_between(dados_grafico['Data'], 2, ax.get_ylim()[1], color='red', alpha=0.1, label='Zona de Atenção')
    
    # Formatação e Títulos
    ax.set_title(f"Evolução do Z-Score: {metrica}\nRestaurante: {restaurante}", fontsize=16, weight='bold', pad=20)
    ax.set_xlabel('Competência', fontsize=12)
    ax.set_ylabel('Z-Score (Desempenho Relativo)', fontsize=12)
    ax.legend()
    ax.grid(True, which='both', linestyle='--', linewidth=0.5)

    # Formatação do eixo X para exibir as datas corretamente
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%m/%y'))
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=1))
    fig.autofmt_xdate()

    plt.tight_layout()
    pdf_pages.savefig(fig)
    plt.close(fig)

def gerar_pagina_intro_analise_qualitativa(pdf_pages, data_referencia):
    """
    Cria uma página de texto explicando como interpretar a análise qualitativa.
    """
    fig, ax = plt.subplots(figsize=(8.5, 11))
    ax.axis('off')

    fig.suptitle(f"Guia de Leitura da Análise Qualitativa - {data_referencia}", fontsize=16, weight='bold', y=0.95)
    fig.text(0.5, 0.89, "Como Interpretar a Análise a Seguir", ha='center', fontsize=12)

    texto_geral = (
        "Nesta seção, cada \"outlier\" (ponto fora da curva) identificado no resumo anterior é analisado em detalhe. "
        "Para facilitar a compreensão, cada ponto é classificado da seguinte forma:"
    )
    fig.text(0.1, 0.82, texto_geral, ha='left', va='top', fontsize=11, wrap=True)

    # Bloco para Ponto de Atenção
    fig.text(0.1, 0.72, "• Ponto de Atenção / Atenção (Vermelho):", ha='left', va='top', fontsize=12, weight='bold', color='red')
    texto_negativo = (
        "   - Representa um resultado **negativo** que exige investigação.\n"
        "   - Pode ser um **custo significativamente maior** que a média do grupo ou, no caso de uma conta\n"
        "     de receita/lucro, um **resultado significantemente menor**.\n"
        "   - Para cada um destes pontos, um gráfico de tendência histórica é apresentado na sequência para\n"
        "     avaliar se o problema é pontual, recorrente ou uma tendência de piora."
    )
    fig.text(0.1, 0.68, texto_negativo, ha='left', va='top', fontsize=11, wrap=True, linespacing=1.5)

    # Bloco para Eficiência
    fig.text(0.1, 0.50, "• Eficiência / Ótimo Desempenho (Verde):", ha='left', va='top', fontsize=12, weight='bold', color='green')
    texto_positivo = (
        "   - Representa um resultado **positivo** e uma oportunidade de aprendizado.\n"
        "   - Pode ser um **custo significativamente menor** que a média do grupo (uma eficiência a ser\n"
        "     elogiada e, se possível, replicada) ou, no caso de uma conta de receita/lucro, um\n"
        "     **resultado significantemente maior**."
    )
    fig.text(0.1, 0.46, texto_positivo, ha='left', va='top', fontsize=11, wrap=True, linespacing=1.5)

    texto_final = (
        "Esta análise transforma os números em um diagnóstico, indicando não apenas \"o quê\" está diferente, "
        "mas qualificando o impacto (positivo ou negativo) e mostrando sua evolução no tempo."
    )
    fig.text(0.1, 0.30, texto_final, ha='left', va='top', fontsize=11, wrap=True, style='italic')

    pdf_pages.savefig(fig, bbox_inches='tight')
    plt.close(fig)

def gerar_analise_qualitativa_outliers(pdf_pages, outliers_df, df_temporal, df_normalizada, data_referencia):
    """
    Gera uma análise textual e, ao final, adiciona páginas de gráficos de tendência
    para todos os outliers negativos encontrados.
    """
    if outliers_df.empty:
        print("Nenhum outlier encontrado para a análise qualitativa.")
        return

    # Lista para armazenar os gráficos que precisam ser gerados
    graficos_para_gerar = []

    contas_receita_set = set(CONTAS_RECEITA)
    outliers_agrupados = outliers_df.groupby('Restaurante')
    
    # --- Início da Geração das PÁGINAS DE TEXTO ---
    fig, ax = plt.subplots(figsize=(8.5, 11))
    ax.axis('off')
    y_pos = 0.92
    
    fig.text(0.5, 0.97, f"Análise Qualitativa de Outliers e Tendências - {data_referencia}", ha='center', fontsize=16, weight='bold')

    for restaurante, grupo in outliers_agrupados:
        # Verifica se há espaço para o cabeçalho do restaurante
        if y_pos < 0.2:
            pdf_pages.savefig(fig, bbox_inches='tight'); plt.close(fig)
            fig, ax = plt.subplots(figsize=(8.5, 11)); ax.axis('off')
            fig.text(0.5, 0.97, f"Análise Qualitativa de Outliers (continuação) - {data_referencia}", ha='center', fontsize=16, weight='bold')
            y_pos = 0.92

        # --- CABEÇALHO DO RESTAURANTE ---
        fig.text(0.1, y_pos, f"Restaurante: {restaurante}", fontsize=13, weight='bold')
        y_pos -= 0.04 

        for _, outlier in grupo.iterrows():
            # Verifica se há espaço para a próxima análise de outlier
            if y_pos < 0.3:
                pdf_pages.savefig(fig, bbox_inches='tight'); plt.close(fig)
                fig, ax = plt.subplots(figsize=(8.5, 11)); ax.axis('off')
                fig.text(0.5, 0.97, f"Análise Qualitativa de Outliers (continuação) - {data_referencia}", ha='center', fontsize=16, weight='bold')
                y_pos = 0.92

            metrica = outlier['Elemento de Custo/Despesa']
            z_score = outlier['Z-Score']
            
            media_grupo = df_normalizada.loc[metrica, 'mean']
            std_grupo = df_normalizada.loc[metrica, 'std']

            is_negativo = False 
            classificacao, cor_classificacao, descricao, explicacao_zscore = "", "black", "", ""
            is_receita = metrica in contas_receita_set

            if is_receita:
                if z_score < -2:
                    classificacao = "Atenção"
                    cor_classificacao = "red"
                    is_negativo = True
                    descricao = "A receita/lucro foi significantemente MENOR que a média do grupo."
                    explicacao_zscore = "Este valor é um outlier, pois está muito abaixo da média do grupo"
                elif z_score > 2:
                    classificacao = "Ótimo Desempenho"
                    cor_classificacao = "green"
                    descricao = "A receita/lucro foi significantemente MAIOR que a média do grupo."
                    explicacao_zscore = "Este valor é um outlier, pois está muito acima da média do grupo"
            else: # Custo/Despesa
                if z_score < -3.5:
                    classificacao = "Ótimo Desempenho"
                    cor_classificacao = "green"
                    descricao = "O custo foi EXCEPCIONALMENTE MENOR que a média do grupo."
                    explicacao_zscore = "Este valor é um outlier, pois está muito abaixo da média do grupo"
                elif z_score < -2:
                    classificacao = "Eficiência"
                    cor_classificacao = "green"
                    descricao = "O custo foi significantemente MENOR que a média do grupo."
                    explicacao_zscore = "Este valor é um outlier, pois está muito abaixo da média do grupo"
                elif z_score > 3.5:
                    classificacao = "Atenção"
                    cor_classificacao = "red"
                    is_negativo = True # <-- CORREÇÃO ADICIONADA AQUI
                    descricao = "O custo foi EXCEPCIONALMENTE MAIOR que a média do grupo."
                    explicacao_zscore = "Este valor é um outlier, pois está muito acima da média do grupo"
                elif z_score > 2:
                    classificacao = "Ponto de Atenção"
                    cor_classificacao = "red"
                    is_negativo = True # <-- CORREÇÃO ADICIONADA AQUI
                    descricao = "O custo foi significantemente MAIOR que a média do grupo."
                    explicacao_zscore = "Este valor é um outlier, pois está muito acima da média do grupo"
            
            if is_negativo:
                graficos_para_gerar.append({'restaurante': restaurante, 'metrica': metrica})

            hist_data = df_temporal[(df_temporal['Restaurante'] == restaurante) & (df_temporal['Metrica'] == metrica)].tail(4)
            texto_tendencia = "Não há dados históricos suficientes para analisar a tendência."
            if len(hist_data) > 1:
                z_scores_recentes = hist_data['ZScore'].tolist()
                ultimo_z = z_scores_recentes[-1]
                media_anteriores = np.mean(z_scores_recentes[:-1]) if len(z_scores_recentes) > 1 else None
                
                if abs(ultimo_z) > 2 and (media_anteriores is None or abs(media_anteriores) < 1.5):
                    texto_tendencia = "Este parece ser um evento pontual, pois o desempenho nos meses anteriores estava mais próximo da média."
                elif is_receita and ultimo_z > media_anteriores:
                     texto_tendencia = "O desempenho desta conta vem melhorando em relação aos meses anteriores."
                elif not is_receita and ultimo_z < media_anteriores:
                    texto_tendencia = "O custo desta conta apresenta uma tendência de melhora (redução) em relação aos meses anteriores."
                else:
                    texto_tendencia = "O desempenho desta conta tem sido consistentemente diferente da média do grupo nos últimos meses."

            valor_real = df_normalizada.loc[metrica, restaurante]
            explicacao_zscore_completa = f"{explicacao_zscore} (Média do grupo: {media_grupo:.2f}, DP: {std_grupo:.2f})."

            fig.text(0.1, y_pos, f"• Conta: ", ha='left', va='top', fontsize=10); fig.text(0.18, y_pos, f"'{metrica}'", ha='left', va='top', fontsize=10, weight='bold'); y_pos -= 0.025
            z_score_text = f"  Z-Score: {z_score:.2f} | "; fig.text(0.1, y_pos, z_score_text, ha='left', va='top', fontsize=10); fig.text(0.1 + (len(z_score_text) * 0.009), y_pos, classificacao, ha='left', va='top', fontsize=10, weight='bold', color=cor_classificacao); y_pos -= 0.025
            fig.text(0.1, y_pos, f"  Análise: ", ha='left', va='top', fontsize=10, weight='bold'); fig.text(0.2, y_pos, descricao, ha='left', va='top', fontsize=10, style='italic'); y_pos -= 0.025
            fig.text(0.1, y_pos, f"  Detalhe Z-Score: ", ha='left', va='top', fontsize=10, weight='bold'); fig.text(0.26, y_pos, explicacao_zscore_completa, ha='left', va='top', fontsize=10, style='italic', wrap=True); y_pos -= 0.035
            fig.text(0.1, y_pos, f"  Valores: ", ha='left', va='top', fontsize=10, weight='bold'); fig.text(0.2, y_pos, f"(Seu valor normalizado: {valor_real:.2f} vs. Média do grupo: {media_grupo:.2f})", ha='left', va='top', fontsize=10, style='italic'); y_pos -= 0.025
            fig.text(0.1, y_pos, f"  Tendência: ", ha='left', va='top', fontsize=10, weight='bold'); fig.text(0.23, y_pos, texto_tendencia, ha='left', va='top', fontsize=10, style='italic'); y_pos -= 0.05
   
    pdf_pages.savefig(fig, bbox_inches='tight')
    plt.close(fig)

    for item in graficos_para_gerar:
        plotar_tendencia_outlier_negativo(pdf_pages, item['restaurante'], item['metrica'], df_temporal)

# ==============================================================================
# --- FUNÇÃO PRINCIPAL DE CARGA DE DADOS HISTÓRICOS ---
# ==============================================================================

def carregar_e_processar_dados_historicos(caminho_base):
    print(f"Lendo arquivos históricos de: {caminho_base}"); arquivos_csv = glob.glob(os.path.join(caminho_base, '*.csv')); dados_historicos_completos = {}
    for arquivo in arquivos_csv:
        nome_base = os.path.basename(arquivo)
        try:
            # Ajustado para encontrar o padrão MM.AA (ex: 01.25)
            match = re.search(r'(\d{2}\.\d{2})', nome_base)
            if not match: raise ValueError("Padrão de data não encontrado")
            
            # Ajustado para interpretar o ano com 2 dígitos (%y minúsculo)
            date_str = match.group(1); data_mes = datetime.strptime(date_str, '%m.%y')
        except Exception as e:
            print(f"AVISO: Não foi possível extrair a data do arquivo '{nome_base}'. Pulando. (Erro: {e})"); continue
        print(f"Processando {nome_base}...")
        df_mes = pd.read_csv(arquivo, delimiter=";")
        df_mes = df_mes.map(lambda x: x.strip() if isinstance(x, str) else x)
        df_mes = df_mes.drop('Grupo_Filial', axis='columns', errors='ignore')

        try: 
            colunas_para_manter = df_mes.columns[:df_mes.columns.get_loc('Store_Operating_Income') + 1]
        except KeyError: print(f"AVISO: Coluna 'Store_Operating_Income' não encontrada em '{nome_base}'. Pulando."); continue
        df_filtrada = df_mes[colunas_para_manter].copy()
        df_filtrada = df_filtrada.drop_duplicates(subset=[df_filtrada.columns[0]], keep='first')
        for col in df_filtrada.columns[1:]: df_filtrada[col] = df_filtrada[col].apply(converter_para_float)
        df_transposto = df_filtrada.set_index('Filial').transpose()
        # --- Linha que faz a tradução ---
        df_transposto.rename(index=mapa_traducao, inplace=True)
        if 'Receita Líquida' not in df_transposto.index: print(f"AVISO: 'Receita Líquida' não encontrada em '{nome_base}'. Pulando."); continue
        receita_liquida = df_transposto.loc['Receita Líquida'].replace({0: np.nan})
        if receita_liquida.isnull().all(): print(f"AVISO: 'Receita Líquida' com valores nulos em '{nome_base}'. Pulando."); continue
        df_normalizada = df_transposto.div(receita_liquida, axis=1) * 1000
        df_normalizada['mean'] = df_normalizada.mean(axis=1, skipna=True); df_normalizada['std'] = df_normalizada.std(axis=1, skipna=True)
        z_scores_mes = df_normalizada.apply(lambda row: (row - row['mean']) / row['std'] if row['std'] > 1e-6 else 0, axis=1)
        z_scores_mes.loc['ZScore_Medio_Geral'] = z_scores_mes.drop(columns=['mean', 'std']).mean()
        
        dados_historicos_completos[data_mes] = {"z_scores": z_scores_mes, "normalizados": df_normalizada, "brutos": df_transposto}
    return dados_historicos_completos

# ==============================================================================
# --- BLOCO PRINCIPAL DE EXECUÇÃO ---
# ==============================================================================

load_dotenv()
caminho_arquivo = os.getenv("caminho_arquivo"); caminho_relatorio = os.getenv("caminho_relatorio"); caminho_base = os.getenv("caminho_base")
caminho_logo = 'icon-logotipo.png'; filiais_to_keep = os.getenv('filiais_to_keep').split(','); title = os.getenv("title"); date = datetime.now().strftime('%d/%m/%Y')
PALETA_CORES = {"neutro": "#005f73", "positivo": "#ee9b00" , "negativo_leve": "#0a9396", "negativo_critico": "#ae2012"}
sns.set_theme(style="whitegrid"); sns.set_context("notebook", font_scale=1.2)

dados_historicos = carregar_e_processar_dados_historicos(caminho_base)
if not dados_historicos: dados_historicos = {}

print("\nProcessando arquivo do mês atual...")
nome_do_arquivo = next((f for f in os.listdir(caminho_arquivo) if f.lower().endswith('.csv')), None)
if not nome_do_arquivo: print(f"ERRO: Nenhum arquivo .csv encontrado em '{caminho_arquivo}'"); sys.exit()
caminho_completo = os.path.join(caminho_arquivo, nome_do_arquivo)
df_atual = pd.read_csv(caminho_completo, delimiter=";")

referencia_str = df_atual['End Date'].iloc[0]; data_objeto_atual = datetime.strptime(referencia_str, '%b/%y')
referencia_para_arquivo = data_objeto_atual.strftime('%m.%Y'); nome_arquivo_dinamico = f"Relatorio_Z_Scores_{referencia_para_arquivo}.pdf"
pdf_path_adjusted = os.path.join(caminho_relatorio, nome_arquivo_dinamico)

df_atual = df_atual.map(lambda x: x.strip() if isinstance(x, str) else x); df_atual = df_atual.drop('Grupo_Filial', axis='columns', errors='ignore')
colunas_para_manter = df_atual.columns[:df_atual.columns.get_loc('Store_Operating_Income') + 1]
df_filtrada_atual = df_atual[colunas_para_manter].copy()
df_filtrada_atual = df_filtrada_atual.drop_duplicates(subset=[df_filtrada_atual.columns[0]], keep='first')
for col in df_filtrada_atual.columns[1:]: df_filtrada_atual[col] = df_filtrada_atual[col].apply(converter_para_float)

df_transposto_atual = df_filtrada_atual.set_index('Filial').transpose()
receita_liquida_atual = df_transposto_atual.loc['NET_REVENUE'].replace({0: np.nan})
df_normalizada_atual = df_transposto_atual.div(receita_liquida_atual, axis=1) * 1000
df_normalizada_atual['mean'] = df_normalizada_atual.mean(axis=1, skipna=True); df_normalizada_atual['std'] = df_normalizada_atual.std(axis=1, skipna=True)
z_scores_atual = df_normalizada_atual.apply(lambda row: (row - row['mean']) / row['std'] if row['std'] > 1e-6 else 0, axis=1)
z_scores_atual.loc['ZScore_Medio_Geral'] = z_scores_atual.drop(columns=['mean', 'std']).mean()

# --- APLICA A TRADUÇÃO NOS DADOS DO MÊS ATUAL ---
df_transposto_atual.rename(index=mapa_traducao, inplace=True)
df_normalizada_atual.rename(index=mapa_traducao, inplace=True)
z_scores_atual.rename(index=mapa_traducao, inplace=True)

dados_historicos[data_objeto_atual] = {"z_scores": z_scores_atual, "normalizados": df_normalizada_atual, "brutos": df_transposto_atual}

print("\n[INFO] Base histórica e dados do mês atual foram unificados.")

# --- CORREÇÃO: Transpõe os dados do mês atual para o formato correto ---
z_scores_atual_transposto = z_scores_atual.drop(columns=['mean', 'std']).transpose()

# Construção do DataFrame temporal unificado para as evoluções
lista_dfs_temporais = []
for data_evento, dados_mes in dados_historicos.items():
    df_longo = dados_mes['z_scores'].stack().reset_index(); df_longo.columns = ['Metrica', 'Restaurante', 'ZScore']
    df_longo['Data'] = data_evento; lista_dfs_temporais.append(df_longo)
df_temporal_completo = pd.concat(lista_dfs_temporais, ignore_index=True)

if data_objeto_atual.month == 1:
    data_inicio_filtro = data_objeto_atual - relativedelta(months=3)
    df_temporal_filtrado = df_temporal_completo[df_temporal_completo['Data'] >= data_inicio_filtro]
else:
    df_temporal_filtrado = df_temporal_completo[df_temporal_completo['Data'].dt.year == data_objeto_atual.year]

filiais_hunger = ['ANV', 'FER', 'GTV', 'RDP', 'PCI', 'VEL']; filiais_pg = ['VSH', 'BVV', 'PCS', 'SVV', 'MAL', 'MOX']
print(f"Gerando relatório estratégico em: {pdf_path_adjusted}")

with PdfPages(pdf_path_adjusted) as pdf_pages:
    criar_capa_pagina(pdf_pages, title, referencia_para_arquivo, date, caminho_logo)

    # --- ANÁLISE DO MÊS ATUAL (Snapshot) ---
    print("Gerando análises de snapshot para o mês atual...")
    # Os gráficos já sairão com os títulos e eixos em português.
    medias_ponderadas_grafico_z_scores(z_scores_atual_transposto, filiais_to_keep, pdf_pages, f"Ranking Geral ({referencia_para_arquivo})", date)
    pontuacao_individual_z_scores(z_scores_atual_transposto, filiais_to_keep, pdf_pages, f"Análise Individual ({referencia_para_arquivo})", date)
    
    # --- ANÁLISE DE OUTLIERS ---
    # O DataFrame de outliers também já terá os nomes das contas em português.
    outliers_df = identificar_outliers(z_scores_atual_transposto, df_normalizada_atual)
    outliers_df = outliers_df.round(2)
    
    # 1. Página de introdução explicando o que são outliers
    gerar_pagina_introducao_outliers(pdf_pages, referencia_para_arquivo)
    
    # 2. Tabela de resumo com os nomes em português
    outliers_resumo = outliers_df[['Restaurante', 'Elemento de Custo/Despesa', 'Z-Score']].sort_values(by='Z-Score', key=abs, ascending=False)
    exportar_dataframe_para_pdf(outliers_resumo, pdf_pages, f"Resumo Numérico de Outliers do Mês ({referencia_para_arquivo})")

    # 3. PÁGINA INTRODUTÓRIA ANTES DA ANÁLISE QUALITATIVA
    print("Gerando página de introdução para a análise qualitativa...")
    gerar_pagina_intro_analise_qualitativa(pdf_pages, referencia_para_arquivo)


    # 4. ANÁLISE QUALITATIVA COM DADOS JÁ TRADUZIDOS
    print("Gerando a nova análise qualitativa de outliers e tendências...")
    gerar_analise_qualitativa_outliers(pdf_pages, outliers_df, df_temporal_completo, df_normalizada_atual, referencia_para_arquivo)
    
    
print("Relatório estratégico completo gerado com sucesso!")