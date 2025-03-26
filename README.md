"""
Projeto: Análise de Agrupamento (Clustering) com Iris e outra base de dados

Objetivos:
1. Carregar e pré-processar os dados (utilizando normalização e PCA para visualização, se necessário).
2. Aplicar dois algoritmos de clustering:
   - Um método hierárquico (por exemplo, linkage aglomerativo) 
   - Um método particional (por exemplo, Bisecting K-Means ou K-Means)
3. Selecionar os melhores parâmetros para cada método utilizando a técnica do "joelho/elbow" ou outra métrica de avaliação interna/externa.
4. Avaliar os grupos gerados e, se os rótulos reais estiverem disponíveis, utilizá-los de forma indireta.
5. Gerar gráficos que ilustrem os resultados (ex: gráficos de dispersão com PCA, curva de elbow, dendrograma, etc).

Requisitos:
- Utilize o dataset Iris (disponível no scikit-learn) e um segundo dataset escolhido (sugestões: datasets do UCI, OpenML ou Kaggle).
- O código deve omitir os rótulos reais ao aplicar os algoritmos, utilizando-os somente para avaliação posterior.
- Comente o código detalhando cada etapa e justificando a escolha dos parâmetros.
- Compare os resultados dos dois algoritmos, discutindo qual apresentou melhores resultados.

Por favor, gere um código Python bem estruturado que:
1. Carregue e normalize os dados;
2. Execute PCA para redução de dimensionalidade (para visualização);
3. Aplique o algoritmo hierárquico e o particional (ex. Bisecting K-Means);
4. Utilize a técnica do "joelho/elbow" para selecionar o número ideal de clusters;
5. Plote gráficos relevantes (como dendrograma, curva de elbow e visualização dos clusters);
6. Inclua comentários e explicações que ajudem a entender cada etapa e a justificar as escolhas dos parâmetros.
"""

# Exemplo de chamada para carregar o dataset Iris e um dataset adicional
# Utilize as bibliotecas: scikit-learn, scipy, matplotlib, seaborn, plotly (se preferir)
