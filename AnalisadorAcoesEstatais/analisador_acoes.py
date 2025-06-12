#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Analisador de Ações Brasileiras
Classe principal para análise de ações da B3 com regressão linear e estatísticas
"""

import yfinance as yf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
from regressao_linear import RegressaoLinearManual

class AnalisadorAcoesBrasileiras:
    """
    Classe para análise completa de ações brasileiras da B3
    """
    
    def __init__(self):
        """
        Inicializa o analisador
        """
        self.dados_acoes = {}
        self.resultados_analise = {}
        
        # Configurar matplotlib para exibir texto em português
        plt.rcParams['font.size'] = 10
        plt.rcParams['figure.figsize'] = (12, 8)
    
    def buscar_dados_historicos(self, simbolo_acao, periodo='1y'):
        """
        Busca dados históricos de uma ação específica
        
        Args:
            simbolo_acao (str): Símbolo da ação (ex: PETR4.SA)
            periodo (str): Período dos dados (ex: '1y', '6m', '3m')
        
        Returns:
            pd.DataFrame: DataFrame com dados históricos
        """
        try:
            print(f"Buscando dados históricos de {simbolo_acao}...")
            
            # Criar objeto ticker do yfinance
            ticker = yf.Ticker(simbolo_acao)
            
            # Buscar dados históricos
            dados = ticker.history(period=periodo)
            
            if dados.empty:
                raise ValueError(f"Nenhum dado encontrado para {simbolo_acao}")
            
            print(f"✓ {len(dados)} registros encontrados para {simbolo_acao}")
            return dados
            
        except Exception as erro:
            raise Exception(f"Erro ao buscar dados de {simbolo_acao}: {erro}")
    
    def calcular_estatisticas_descritivas(self, precos_fechamento):
        """
        Calcula estatísticas descritivas dos preços de fechamento
        
        Args:
            precos_fechamento (pd.Series): Série com preços de fechamento
        
        Returns:
            dict: Dicionário com estatísticas calculadas
        """
        estatisticas = {
            'minimo': float(precos_fechamento.min()),
            'maximo': float(precos_fechamento.max()),
            'media': float(precos_fechamento.mean()),
            'desvio_padrao': float(precos_fechamento.std()),
            'total_dias': len(precos_fechamento)
        }
        
        return estatisticas
    
    def analisar_acao_individual(self, simbolo_acao, periodo='1y'):
        """
        Executa análise completa de uma ação individual
        
        Args:
            simbolo_acao (str): Símbolo da ação
            periodo (str): Período de análise
        
        Returns:
            dict: Resultados da análise
        """
        # Buscar dados históricos
        dados_historicos = self.buscar_dados_historicos(simbolo_acao, periodo)
        
        # Extrair preços de fechamento
        precos_fechamento = dados_historicos['Close']
        datas = dados_historicos.index
        
        # Criar variável independente (dias desde o início)
        dias = np.arange(len(precos_fechamento))
        
        # Calcular regressão linear manual
        regressao = RegressaoLinearManual()
        coeficientes = regressao.treinar(dias, precos_fechamento.values)
        
        # Fazer previsões
        previsoes = regressao.prever(dias)
        
        # Calcular estatísticas descritivas
        estatisticas = self.calcular_estatisticas_descritivas(precos_fechamento)
        
        # Calcular métricas de qualidade da regressão
        erro_quadratico_medio = np.mean((precos_fechamento.values - previsoes) ** 2)
        coeficiente_determinacao = regressao.calcular_r_quadrado(precos_fechamento.values, previsoes)
        
        # Organizar resultados
        resultado = {
            'simbolo': simbolo_acao,
            'dados_historicos': dados_historicos,
            'precos_fechamento': precos_fechamento,
            'datas': datas,
            'dias': dias,
            'coeficientes': coeficientes,
            'previsoes': previsoes,
            'estatisticas': estatisticas,
            'erro_quadratico_medio': erro_quadratico_medio,
            'coeficiente_determinacao': coeficiente_determinacao
        }
        
        return resultado
    
    def analisar_acoes(self, lista_acoes, periodo='1y'):
        """
        Analisa múltiplas ações
        
        Args:
            lista_acoes (list): Lista com símbolos das ações
            periodo (str): Período de análise
        
        Returns:
            dict: Resultados de todas as análises
        """
        resultados = {}
        
        for acao in lista_acoes:
            try:
                print(f"\nAnalisando {acao}...")
                resultado = self.analisar_acao_individual(acao, periodo)
                resultados[acao] = resultado
                print(f"✓ Análise de {acao} concluída")
                
            except Exception as erro:
                print(f"✗ Erro na análise de {acao}: {erro}")
                continue
        
        return resultados
    
    def exibir_resultados(self, resultados):
        """
        Exibe resultados formatados no console
        
        Args:
            resultados (dict): Resultados das análises
        """
        for simbolo, dados in resultados.items():
            print(f"\n{'='*50}")
            print(f"ANÁLISE DE {simbolo}")
            print(f"{'='*50}")
            
            # Estatísticas descritivas
            print("\nESTATÍSTICAS DESCRITIVAS:")
            print(f"Preço Mínimo: R$ {dados['estatisticas']['minimo']:.2f}")
            print(f"Preço Máximo: R$ {dados['estatisticas']['maximo']:.2f}")
            print(f"Preço Médio: R$ {dados['estatisticas']['media']:.2f}")
            print(f"Desvio Padrão (Volatilidade): R$ {dados['estatisticas']['desvio_padrao']:.2f}")
            print(f"Total de Dias Analisados: {dados['estatisticas']['total_dias']}")
            
            # Coeficientes da regressão
            print("\nCOEFICIENTES DA REGRESSÃO LINEAR:")
            print(f"Intercepto (β₀): R$ {dados['coeficientes']['intercepto']:.4f}")
            print(f"Inclinação (β₁): R$ {dados['coeficientes']['inclinacao']:.6f} por dia")
            
            # Tendência
            tendencia = "ALTA" if dados['coeficientes']['inclinacao'] > 0 else "BAIXA"
            print(f"Tendência Identificada: {tendencia}")
            
            # Métricas de qualidade
            print("\nQUALIDADE DA REGRESSÃO:")
            print(f"Erro Quadrático Médio: {dados['erro_quadratico_medio']:.4f}")
            print(f"Coeficiente de Determinação (R²): {dados['coeficiente_determinacao']:.4f}")
            
            # Previsão para próximos dias
            proximos_dias = np.array([len(dados['dias']), len(dados['dias']) + 5, len(dados['dias']) + 10])
            regressao = RegressaoLinearManual()
            regressao.coeficientes = dados['coeficientes']
            previsoes_futuras = regressao.prever(proximos_dias)
            
            print("\nPREVISÕES PARA PRÓXIMOS DIAS:")
            print(f"Próximo dia útil: R$ {previsoes_futuras[0]:.2f}")
            print(f"Em 5 dias: R$ {previsoes_futuras[1]:.2f}")
            print(f"Em 10 dias: R$ {previsoes_futuras[2]:.2f}")
    
    def gerar_graficos(self, resultados):
        """
        Gera gráficos para visualização dos resultados
        
        Args:
            resultados (dict): Resultados das análises
        """
        for simbolo, dados in resultados.items():
            # Criar figura com subplots
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10))
            
            # Gráfico 1: Preços reais vs Regressão Linear
            ax1.plot(dados['datas'], dados['precos_fechamento'], 
                    label='Preços Reais', color='blue', linewidth=2, alpha=0.7)
            ax1.plot(dados['datas'], dados['previsoes'], 
                    label='Regressão Linear', color='red', linewidth=2, linestyle='--')
            
            ax1.set_title(f'Análise de Regressão Linear - {simbolo}', fontsize=14, fontweight='bold')
            ax1.set_xlabel('Data', fontsize=12)
            ax1.set_ylabel('Preço de Fechamento (R$)', fontsize=12)
            ax1.legend()
            ax1.grid(True, alpha=0.3)
            ax1.tick_params(axis='x', rotation=45)
            
            # Adicionar texto com informações da regressão
            textstr = f'R² = {dados["coeficiente_determinacao"]:.4f}\n'
            textstr += f'Inclinação = {dados["coeficientes"]["inclinacao"]:.6f}\n'
            textstr += f'Intercepto = {dados["coeficientes"]["intercepto"]:.4f}'
            props = dict(boxstyle='round', facecolor='wheat', alpha=0.8)
            ax1.text(0.02, 0.98, textstr, transform=ax1.transAxes, fontsize=10,
                    verticalalignment='top', bbox=props)
            
            # Gráfico 2: Histograma dos resíduos
            residuos = dados['precos_fechamento'].values - dados['previsoes']
            ax2.hist(residuos, bins=20, color='green', alpha=0.7, edgecolor='black')
            ax2.set_title(f'Distribuição dos Resíduos - {simbolo}', fontsize=14, fontweight='bold')
            ax2.set_xlabel('Resíduos (R$)', fontsize=12)
            ax2.set_ylabel('Frequência', fontsize=12)
            ax2.grid(True, alpha=0.3)
            
            # Adicionar linha vertical no zero
            ax2.axvline(x=0, color='red', linestyle='--', alpha=0.8)
            
            plt.tight_layout()
            
            # Salvar gráfico
            nome_arquivo = f'analise_{simbolo.replace(".", "_")}.png'
            plt.savefig(nome_arquivo, dpi=300, bbox_inches='tight')
            print(f"Gráfico salvo: {nome_arquivo}")
            
            # plt.show()  # Comentado para execução em ambiente sem display
            plt.close()
