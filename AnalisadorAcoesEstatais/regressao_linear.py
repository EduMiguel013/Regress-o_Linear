#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Implementação Manual de Regressão Linear com Gradiente Descendente
Algoritmo implementado do zero para análise de ações brasileiras
"""

import numpy as np

class RegressaoLinearManual:
    """
    Implementação manual de regressão linear usando gradiente descendente
    """
    
    def __init__(self, taxa_aprendizado=0.01, max_iteracoes=1000, tolerancia=1e-6):
        """
        Inicializa o modelo de regressão linear
        
        Args:
            taxa_aprendizado (float): Taxa de aprendizado para gradiente descendente
            max_iteracoes (int): Número máximo de iterações
            tolerancia (float): Tolerância para convergência
        """
        self.taxa_aprendizado = taxa_aprendizado
        self.max_iteracoes = max_iteracoes
        self.tolerancia = tolerancia
        self.coeficientes = {}
        self.historico_custo = []
    
    def calcular_custo(self, y_real, y_previsto):
        """
        Calcula o custo usando erro quadrático médio
        
        Args:
            y_real (np.array): Valores reais
            y_previsto (np.array): Valores previstos
        
        Returns:
            float: Custo calculado
        """
        m = len(y_real)
        custo = (1 / (2 * m)) * np.sum((y_previsto - y_real) ** 2)
        return custo
    
    def calcular_gradientes(self, X, y_real, y_previsto):
        """
        Calcula os gradientes para intercepto e inclinação
        
        Args:
            X (np.array): Variável independente
            y_real (np.array): Valores reais
            y_previsto (np.array): Valores previstos
        
        Returns:
            tuple: Gradientes para intercepto e inclinação
        """
        m = len(y_real)
        
        # Gradiente para intercepto (β₀)
        gradiente_intercepto = (1 / m) * np.sum(y_previsto - y_real)
        
        # Gradiente para inclinação (β₁)
        gradiente_inclinacao = (1 / m) * np.sum((y_previsto - y_real) * X)
        
        return gradiente_intercepto, gradiente_inclinacao
    
    def treinar(self, X, y):
        """
        Treina o modelo usando gradiente descendente
        
        Args:
            X (np.array): Variável independente (dias)
            y (np.array): Variável dependente (preços)
        
        Returns:
            dict: Coeficientes calculados
        """
        # Converter para numpy arrays
        X = np.array(X, dtype=float)
        y = np.array(y, dtype=float)
        
        # Normalizar X para melhor convergência
        X_mean = np.mean(X)
        X_std = np.std(X)
        if X_std != 0:
            X_norm = (X - X_mean) / X_std
        else:
            X_norm = X - X_mean
        
        # Inicializar coeficientes aleatoriamente
        intercepto = np.random.randn() * 0.01
        inclinacao = np.random.randn() * 0.01
        
        print(f"Iniciando treinamento com gradiente descendente...")
        print(f"Taxa de aprendizado: {self.taxa_aprendizado}")
        print(f"Máximo de iterações: {self.max_iteracoes}")
        
        # Loop principal do gradiente descendente
        for iteracao in range(self.max_iteracoes):
            # Calcular previsões
            y_previsto = intercepto + inclinacao * X_norm
            
            # Calcular custo
            custo_atual = self.calcular_custo(y, y_previsto)
            self.historico_custo.append(custo_atual)
            
            # Calcular gradientes
            grad_intercepto, grad_inclinacao = self.calcular_gradientes(X_norm, y, y_previsto)
            
            # Atualizar coeficientes
            novo_intercepto = intercepto - self.taxa_aprendizado * grad_intercepto
            nova_inclinacao = inclinacao - self.taxa_aprendizado * grad_inclinacao
            
            # Verificar convergência
            if abs(novo_intercepto - intercepto) < self.tolerancia and abs(nova_inclinacao - inclinacao) < self.tolerancia:
                print(f"Convergência alcançada na iteração {iteracao + 1}")
                break
            
            intercepto = novo_intercepto
            inclinacao = nova_inclinacao
            
            # Exibir progresso a cada 100 iterações
            if (iteracao + 1) % 100 == 0:
                print(f"Iteração {iteracao + 1}: Custo = {custo_atual:.6f}")
        
        # Ajustar coeficientes para escala original
        if X_std != 0:
            inclinacao_original = inclinacao / X_std
            intercepto_original = intercepto - (inclinacao * X_mean / X_std)
        else:
            inclinacao_original = inclinacao
            intercepto_original = intercepto - (inclinacao * X_mean)
        
        # Armazenar coeficientes finais
        self.coeficientes = {
            'intercepto': intercepto_original,
            'inclinacao': inclinacao_original,
            'X_mean': X_mean,
            'X_std': X_std
        }
        
        print(f"Treinamento concluído!")
        print(f"Custo final: {self.historico_custo[-1]:.6f}")
        print(f"Intercepto (β₀): {intercepto_original:.6f}")
        print(f"Inclinação (β₁): {inclinacao_original:.6f}")
        
        return self.coeficientes
    
    def prever(self, X):
        """
        Faz previsões usando o modelo treinado
        
        Args:
            X (np.array): Valores para predição
        
        Returns:
            np.array: Previsões calculadas
        """
        if not self.coeficientes:
            raise ValueError("Modelo não foi treinado ainda. Execute o método treinar() primeiro.")
        
        X = np.array(X, dtype=float)
        
        # Aplicar mesma transformação usada no treinamento
        # (não é necessário normalizar para predição com coeficientes ajustados)
        y_previsto = self.coeficientes['intercepto'] + self.coeficientes['inclinacao'] * X
        
        return y_previsto
    
    def calcular_r_quadrado(self, y_real, y_previsto):
        """
        Calcula o coeficiente de determinação (R²)
        
        Args:
            y_real (np.array): Valores reais
            y_previsto (np.array): Valores previstos
        
        Returns:
            float: Valor de R²
        """
        # Soma dos quadrados dos resíduos
        ss_res = np.sum((y_real - y_previsto) ** 2)
        
        # Soma total dos quadrados
        ss_tot = np.sum((y_real - np.mean(y_real)) ** 2)
        
        # Calcular R²
        if ss_tot == 0:
            return 1.0  # Caso especial onde todos os valores são iguais
        
        r_quadrado = 1 - (ss_res / ss_tot)
        return r_quadrado
    
    def obter_metricas_modelo(self, X, y_real):
        """
        Calcula métricas de avaliação do modelo
        
        Args:
            X (np.array): Variável independente
            y_real (np.array): Valores reais
        
        Returns:
            dict: Dicionário com métricas calculadas
        """
        y_previsto = self.prever(X)
        
        metricas = {
            'r_quadrado': self.calcular_r_quadrado(y_real, y_previsto),
            'erro_quadratico_medio': np.mean((y_real - y_previsto) ** 2),
            'erro_absoluto_medio': np.mean(np.abs(y_real - y_previsto)),
            'raiz_erro_quadratico_medio': np.sqrt(np.mean((y_real - y_previsto) ** 2))
        }
        
        return metricas
