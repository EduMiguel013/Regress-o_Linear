# Análise de Ações B3 com Regressão Linear

Sistema de análise de ações brasileiras da B3 utilizando regressão linear com implementação manual de gradiente descendente.

## 📊 Sobre o Projeto

Este projeto implementa um sistema completo de análise técnica de ações brasileiras negociadas na B3 (Brasil, Bolsa, Balcão), utilizando regressão linear com gradiente descendente implementado manualmente (sem bibliotecas de machine learning).

O sistema analisa três empresas estatais brasileiras:
- **PETR4.SA** - Petrobras (Petróleo Brasileiro S.A.)
- **BBAS3.SA** - Banco do Brasil S.A.
- **ELET3.SA** - Centrais Elétricas Brasileiras S.A. (Eletrobras)

## ✨ Funcionalidades

- 📈 Coleta automática de dados históricos de ações via Yahoo Finance
- 🧮 Implementação manual do algoritmo de gradiente descendente
- 📊 Cálculo de estatísticas descritivas (média, desvio padrão, mínimo, máximo)
- 📉 Análise de tendências com regressão linear
- 🎯 Cálculo do coeficiente de determinação (R²) e erro quadrático médio (EQM)
- 📸 Geração automática de gráficos de alta qualidade
- 📓 Versão em Jupyter Notebook para análise interativa
- 📄 Artigo científico em LaTeX formato ABNT

## 🛠️ Tecnologias Utilizadas

- **Python 3.11+**
- **yfinance** - Coleta de dados financeiros
- **NumPy** - Operações matemáticas e arrays
- **Pandas** - Manipulação de dados tabulares
- **Matplotlib** - Visualização de dados
- **LaTeX** - Documentação científica

## 📁 Estrutura do Projeto

```
analise-acoes-b3-regressao-linear/
│
├── main.py                      # Arquivo principal de execução
├── main.ipynb                   # Versão Jupyter Notebook
├── analisador_acoes.py          # Classe principal de análise
├── regressao_linear.py          # Implementação do gradiente descendente
├── artigo_analise_acoes.tex     # Artigo científico em LaTeX
├── README.md                    # Este arquivo
│
└── Gráficos gerados (após execução):
    ├── analise_PETR4_SA.png
    ├── analise_BBAS3_SA.png
    └── analise_ELET3_SA.png
```

## 🚀 Como Executar

### Pré-requisitos

Certifique-se de ter Python 3.11 ou superior instalado.

### Instalação

1. Clone o repositório:
```bash
git clone https://github.com/seu-usuario/analise-acoes-b3-regressao-linear.git
cd analise-acoes-b3-regressao-linear
```

2. Instale as dependências:
```bash
pip install yfinance numpy pandas matplotlib
```

### Execução

Execute o script principal:
```bash
python main.py
```

### Uso do Jupyter Notebook

Para análise interativa:
```bash
jupyter notebook main.ipynb
```

## 📊 Resultados Esperados

O sistema irá:

1. **Coletar dados** dos últimos 12 meses das três ações
2. **Calcular estatísticas descritivas** para cada ação
3. **Treinar modelo de regressão linear** usando gradiente descendente
4. **Exibir resultados no terminal** com:
   - Coeficientes da regressão (intercepto e inclinação)
   - R² (qualidade do ajuste)
   - Erro Quadrático Médio (EQM)
   - Estatísticas descritivas completas
5. **Gerar gráficos PNG** contendo:
   - Preços reais vs. linha de regressão
   - Histograma de resíduos

### Exemplo de Saída

```
====================================================================
ANÁLISE: PETR4.SA - Petrobras
====================================================================

--- COEFICIENTES DA REGRESSÃO LINEAR ---
Intercepto (θ₀): 37.8542
Inclinação (θ₁): -0.0079 (tendência diária)

--- MÉTRICAS DE AVALIAÇÃO ---
R² (Coeficiente de Determinação): 0.0861
Erro Quadrático Médio (EQM): 3.4368

--- ESTATÍSTICAS DESCRITIVAS ---
Preço Mínimo: R$ 32.15
Preço Máximo: R$ 42.68
Preço Médio: R$ 37.41
Desvio Padrão: R$ 1.98
```

## 🧮 Metodologia Matemática

### Função de Custo (Mean Squared Error)

```
J(θ₀, θ₁) = (1/2m) Σ[hθ(x⁽ⁱ⁾) - y⁽ⁱ⁾]²
```

### Atualização dos Parâmetros (Gradiente Descendente)

```
θⱼ := θⱼ - α ∂J/∂θⱼ
```

Onde:
- `α` = Taxa de aprendizado (0.01)
- `m` = Número de exemplos de treinamento
- Iterações máximas: 1000
- Tolerância de convergência: 1e-6

### Coeficiente de Determinação (R²)

```
R² = 1 - (SSres / SStot)
```

Onde:
- `SSres` = Soma dos quadrados dos resíduos
- `SStot` = Soma total dos quadrados

## 📄 Artigo Científico

O projeto inclui um artigo científico completo em LaTeX (formato ABNT) com:
- Capa e introdução
- Fundamentação teórica
- Metodologia com fórmulas matemáticas
- Descrição do dataset
- Resultados e análises gráficas
- Discussão e conclusão
- Referências bibliográficas

Para compilar o artigo:
```bash
pdflatex artigo_analise_acoes.tex
bibtex artigo_analise_acoes
pdflatex artigo_analise_acoes.tex
pdflatex artigo_analise_acoes.tex
```

## 🎯 Características Técnicas

- ✅ **Backend puro** - Sistema focado em processamento e análise
- ✅ **Código 100% em português** - Variáveis, funções e comentários
- ✅ **Implementação manual** - Gradiente descendente sem sklearn ou bibliotecas de ML
- ✅ **Normalização de features** - Para melhor convergência do algoritmo
- ✅ **Documentação completa** - Comentários detalhados em todo o código
- ✅ **Validação de modelo** - Métricas R² e EQM para avaliar qualidade
- ✅ **Visualizações profissionais** - Gráficos de alta resolução (300 DPI)

## 📖 Estrutura do Código

### `main.py`
Ponto de entrada do sistema. Orquestra o fluxo de análise.

### `analisador_acoes.py`
Contém a classe `AnalisadorAcoesBrasileiras` com métodos para:
- Busca de dados históricos
- Cálculo de estatísticas descritivas
- Análise individual de ações
- Exibição de resultados formatados
- Geração de gráficos

### `regressao_linear.py`
Implementa a classe `RegressaoLinearGradiente` com:
- Cálculo da função de custo
- Cálculo dos gradientes
- Treinamento via gradiente descendente
- Normalização de features
- Geração de previsões
- Cálculo de métricas (R², EQM)

## 🤝 Contribuições

Contribuições são bem-vindas! Sinta-se à vontade para:
- Reportar bugs
- Sugerir novas funcionalidades
- Enviar pull requests

## 📝 Licença

Este projeto é de código aberto e está disponível sob a licença MIT.

## 👨‍💻 Autor

Nome do autor: Eduardo Miguel Ribeiro Cordeiro

E-mail: edumiguelcordeiro@gmail.com

## 📚 Referências

- Yahoo Finance API para dados históricos
- Algoritmo de Gradiente Descendente aplicado à Regressão Linear
- Análise técnica de ações da B3

---

**Nota**: Este projeto foi desenvolvido para fins educacionais e de análise. Não constitui recomendação de investimento.
