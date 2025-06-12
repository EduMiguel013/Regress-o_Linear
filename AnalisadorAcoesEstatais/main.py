#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Sistema Principal de Análise de Ações Brasileiras
Executa análise completa das ações PETR4.SA, BBAS3.SA e ELET3.SA
"""

from analisador_acoes import AnalisadorAcoesBrasileiras

def main():
    """
    Função principal que executa toda a análise das ações brasileiras
    """
    print("=" * 60)
    print("SISTEMA DE ANÁLISE DE AÇÕES BRASILEIRAS")
    print("=" * 60)
    
    # Lista das ações a serem analisadas
    acoes = ['PETR4.SA', 'BBAS3.SA', 'ELET3.SA']
    
    # Período de análise (1 ano de dados históricos)
    periodo = '1y'
    
    try:
        # Inicializar o analisador
        analisador = AnalisadorAcoesBrasileiras()
        
        # Executar análise completa
        resultados = analisador.analisar_acoes(acoes, periodo)
        
        # Exibir resultados formatados
        analisador.exibir_resultados(resultados)
        
        # Gerar gráficos
        analisador.gerar_graficos(resultados)
        
        print("\n" + "=" * 60)
        print("ANÁLISE CONCLUÍDA COM SUCESSO!")
        print("Gráficos salvos como arquivos PNG na pasta atual.")
        print("=" * 60)
        
    except Exception as erro:
        print(f"\nERRO na execução da análise: {erro}")
        print("Verifique sua conexão com a internet e tente novamente.")

if __name__ == "__main__":
    main()
