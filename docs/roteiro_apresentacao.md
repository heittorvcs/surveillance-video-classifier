# Roteiro do Vídeo de Apresentação — Surveillance Video Classifier

**Candidato:** Heittor Costa · **Destinatário:** NeuIA · **Duração alvo:** 8m00s (janela do edital: 5–10 min)

**Princípio do roteiro:** todo número dito em voz alta existe num arquivo do repositório e foi obtido com o teste tocado uma única vez. A parte mais forte da apresentação não é a acurácia final — é a demonstração de que o candidato encontrou um viés metodológico no próprio trabalho, mediu o seu tamanho e o corrigiu.

---

## Pré-gravação — checklist

- [ ] `python src/build_cache.py` executado
- [ ] `python tests/test_splits.py` passando — vai rodar na tela no minuto 1:00
- [ ] `python src/evaluate.py --model ensemble` rodando de verdade na máquina de gravação
- [ ] `python src/inference.py --video sample_video.avi --label Fight` testado
- [ ] `reports/selection_bias_analysis.png` aberto num visualizador — é o slide mais importante
- [ ] Terminal em fonte 16+, 1080p, microfone testado sem eco
- [ ] Abas do editor: `README.md` (seção 5), `src/create_splits.py`, `src/model.py`, `src/select_ensemble.py`

---

## 0:00 – 0:50 · Problema e critério de sucesso

**Tela:** README aberto no topo.

> "Olá, sou o Heittor. Apresento o Surveillance Video Classifier: um classificador de vídeo para detecção de agressão física em câmeras fixas de CFTV, projetado para rodar em CPU na borda.
>
> A decisão de engenharia que orienta o projeto é a assimetria de custo entre os dois tipos de erro. Um falso positivo custa alguns segundos de atenção de um vigilante. Um falso negativo é uma agressão que ninguém viu.
>
> Vou mostrar três arquiteturas, o resultado no teste cego, e — a parte que eu considero mais importante — um erro metodológico que eu cometi na primeira versão deste projeto, quanto ele valia em pontos de acurácia, e como está corrigido."

---

## 0:50 – 2:00 · Dataset e a armadilha do vazamento

**Tela:** `src/create_splits.py`, destacando o `GroupShuffleSplit`; depois rodar os testes.

> "O dataset é o RWF-2000, de Cheng e colegas, 2020: 2.000 clipes reais de câmeras de vigilância fixas, 5 segundos a 30 FPS, balanceado em 1.000 Fight e 1.000 NonFight. Fight cobre socos, chutes, empurrões e confrontos corporais; NonFight é monitoramento normal.
>
> A armadilha é específica desse dataset: vários clipes vêm de cortes temporais da mesma câmera física — mesmo fundo, mesma iluminação, mesmo ângulo. Com um split aleatório, o modelo decora o cenário e você publica um número que não sobrevive ao primeiro cliente.
>
> Então o particionamento é por grupo: extraio o identificador do vídeo original do nome do arquivo e uso `GroupShuffleSplit` para que nenhum grupo atravesse a fronteira. São 1.600 vídeos de treino, 215 de validação e 185 de teste cego.
>
> E isso não é uma afirmação — é um teste."

*(Rodar `python tests/test_splits.py` na tela: cinco testes, cinco segundos.)*

> "Zero grupos e zero arquivos compartilhados entre os três splits. As 90 câmeras do teste são inéditas para o modelo."

---

## 2:00 – 3:00 · Pré-processamento

**Tela:** `src/dataset.py` e `src/build_cache.py`.

> "Cada clipe tem 150 quadros. Amostro 16 equidistantes, cobrindo os 5 segundos inteiros — isso corta 89% do volume de dados. A decodificação usa `grab()` para avançar o cursor e `retrieve()` só nos 16 instantes que interessam. Depois: 224 por 224, BGR para RGB e normalização com as estatísticas do ImageNet, porque o backbone é pré-treinado nelas.
>
> E uma armadilha que eu mesmo criei, que vale contar. Como o backbone é congelado, eu cacheio as features para acelerar o treino em duas ordens de grandeza. Na primeira versão, o flip horizontal era aplicado nessa extração única — ou seja, cada vídeo recebia uma perturbação fixa, idêntica em todas as épocas. Isso não é augmentation, é ruído congelado.
>
> A correção está no código: `build_cache.py` materializa dois caches, o original e o espelhado, e o treino sorteia entre eles a cada época. Sendo honesto: os números que vou mostrar foram treinados **sem** o cache espelhado, porque eu não tinha o dataset bruto no ambiente de retreinamento. Então o ganho do augmentation está implementado, mas ainda não medido — e está listado como tal nas limitações."

---

## 3:00 – 4:15 · As três arquiteturas

**Tela:** `src/model.py`, percorrendo as três classes.

> "O **Modelo 1** é o baseline: MobileNetV3-Small pré-treinado no ImageNet e congelado como extrator espacial de 576 dimensões, seguido de uma Bi-GRU. 1,17 milhão de parâmetros.
>
> Congelei o backbone de propósito. Com câmeras fixas, descongelar faz os filtros se adaptarem à textura do fundo — paredes, calçadas — em vez do movimento dos atores. E é o que permite cachear features e treinar 80 modelos em minutos, o que vai ser essencial daqui a pouco.
>
> A limitação do baseline é estrutural: ele só vê aparência estática. Duas pessoas gesticulando com energia têm embedding parecido com duas pessoas brigando. Falta a derivada.
>
> O **Modelo 2** resolve isso sem optical flow, que custaria centenas de milissegundos por quadro e mataria a proposta de borda. Eu calculo a velocidade no espaço latente: delta f igual a f_t menos f_{t-1}, sobre os vetores de 576 dimensões que já tenho. Duas Bi-GRUs em paralelo, aparência e velocidade.
>
> O efeito é nítido no recall: sobe de 71,6% para 90,9%, e os falsos negativos caem de 25 para 8. O preço são 32 falsos positivos.
>
> O **Modelo 3** adiciona a segunda derivada — aceleração — porque o que caracteriza um impacto não é a velocidade, é a mudança brusca dela. E combina três cabeças em comitê, todas partilhando o mesmo backbone: o MobileNet roda uma vez por vídeo, e as três cabeças somadas custam 6 milissegundos."

---

## 4:15 – 6:00 · O erro metodológico, e quanto ele valia

**Tela:** `reports/selection_bias_analysis.png` em tela cheia. Este é o núcleo da apresentação.

> "Agora a parte que eu quero que vocês vejam com atenção.
>
> A primeira versão deste projeto reportava **85,41% de acurácia**. Eu fui auditar como esse número tinha sido obtido, e encontrei o script de busca. Ele treinava 12 candidatos e testava todas as combinações de 3 a 7 membros, com dois esquemas de fusão e onze limiares: **70.906 configurações**. O critério de escolha era `if acc > best_ens_acc` — e esse `acc` era calculado sobre os rótulos do **teste**.
>
> Ou seja: o teste era cego para o treinamento, mas não era cego para a seleção do modelo. Os 85,41% eram o máximo de setenta mil configurações medidas em 185 vídeos.
>
> Para medir o tamanho do problema, eu treinei um pool completamente novo — 20 sementes de cada arquitetura — e comparei três protocolos sobre exatamente os mesmos modelos. É esta figura.
>
> [aponta] A distribuição cinza são os 3.800 trios possíveis avaliados no teste com limiar fixo: média de 80,99%, desvio de 1,18.
>
> A linha verde é o protocolo correto: trio e limiar escolhidos **apenas na validação**, teste avaliado uma única vez. Deu **81,62%** — meio desvio acima da média, um trio perfeitamente típico.
>
> A linha vermelha é o oráculo: deixar a busca escolher olhando o teste, como na versão original. Deu **85,41%**.
>
> E aqui está o detalhe que fecha o argumento: 85,41 é **exatamente** o mesmo número da entrega original, obtido a partir de um pool de modelos completamente diferente. Isso não é coincidência. É o teto que uma busca de centenas de milhares de configurações alcança num conjunto de 185 vídeos, independentemente de com quais modelos você começa. O 85,41% nunca foi uma propriedade daquelas sementes — era uma propriedade do procedimento.
>
> O viés é de **3,78 pontos percentuais**. E o mesmo vale para o Modelo 2 da versão anterior: os 82,16% eram exatamente o máximo da sua própria distribuição de 20 sementes.
>
> Então tudo foi refeito. Os três modelos são selecionados sob o mesmo protocolo: melhor candidato por F1 na validação, limiar fixo em 0,50, teste tocado uma vez."

**Se sobrar tempo, ou como resposta na sabatina:**

> "Duas decisões finas dentro do protocolo. Primeiro, o limiar fica fixo durante a seleção, porque escolher semente e limiar ao mesmo tempo em 215 vídeos superajusta a validação — uma varredura de 91 limiares elegeu θ igual a 0,15 para o dual-stream, que no teste rendeu 67% de precisão. Segundo, o critério é F1 e não recall sob restrição de precisão, porque a restrição de precisão 0,80 é estruturalmente inviável para o baseline: na validação ele só a atinge a θ ≥ 0,71, onde o recall desaba. As duas decisões foram tomadas olhando só a validação."

---

## 6:00 – 7:00 · Resultados e o que o AUC diz

**Tela:** tabela da seção 4 do README, matrizes de confusão, curvas ROC. Rodar `python src/evaluate.py --model ensemble`.

> "Os resultados no teste cego, com os três sob o mesmo protocolo:
>
> Baseline: 76,22% de acurácia, recall 71,6%, 25 falsos negativos.
> Dual-Stream: 78,38%, recall 90,9%, apenas 8 falsos negativos — mas 32 falsos positivos.
> Ensemble: **81,62%**, recall 85,2%, F1 de 81,5%, 13 falsos negativos e 21 falsos positivos. É o melhor equilíbrio.
>
> E o número que eu mais confio é o AUC-ROC, porque ele não depende da escolha de limiar: 85,6% no baseline, 89,3% no dual-stream, 89,6% no ensemble. É esse salto que prova que o viés indutivo cinético funciona. O que não funcionava era a forma de reportar.
>
> Na validação, o ensemble também tem desvio menor que as arquiteturas isoladas — 1,09 contra 1,59 e 1,90. É exatamente o que se espera de um comitê: menos variância."

---

## 7:00 – 7:40 · Edge AI e demonstração

**Tela:** rodar `python src/inference.py --video sample_video.avi --label Fight`, depois a tabela da seção 7.

> "Sobre borda, dois resultados que a versão anterior deste README escondia.
>
> Primeiro: eu medi os três modelos no mesmo laço e na mesma execução, separando decodificação de forward. A decodificação do vídeo custa 43,9 milissegundos — **mais que o backbone inteiro**, que custa 34. O gargalo do pipeline não é a rede, é ler o arquivo. Otimizar só o modelo tem retorno limitado, e isso é uma informação de engenharia que a medição anterior não permitia enxergar.
>
> Segundo: o ensemble custa apenas 5 milissegundos a mais que o baseline, porque as três cabeças compartilham a mesma passada do backbone.
>
> E o ONNX. Na versão anterior, o `--export_onnx` instanciava um modelo novo e exportava **sem carregar os pesos** — o grafo publicado era matematicamente ruído. Agora a exportação carrega os pesos treinados, e existe um script que verifica paridade numérica contra o PyTorch: diferença máxima da ordem de 1e-6 nos três modelos. Com ONNX Runtime, o forward do ensemble cai de 44,9 para 16,9 milissegundos — 2,7 vezes mais rápido.
>
> E a inferência no vídeo de demonstração: Fight com 69,9% de probabilidade, 19,8 pontos acima do limiar. Acertou, com margem confortável."

---

## 7:40 – 8:00 · Fechamento

> "Resumindo as decisões: split por grupo de câmera para que o número de teste signifique alguma coisa; backbone congelado por custo de borda e risco de overfitting de cenário; derivadas cinéticas no espaço latente em vez de optical flow; e seleção de modelo feita na validação, com o teste tocado uma única vez.
>
> As limitações que eu reconheço: o augmentation está implementado mas não medido; 215 vídeos de validação ainda são poucos para escolher entre 3.800 trios, e um GroupKFold seria mais estável; e o dataset é restrito a cenas diurnas e câmeras estáticas.
>
> E a conclusão que eu mais levo daqui: eu preferi entregar 81,62% que eu consigo defender do que 85,41% que não sobreviveria à primeira auditoria. Obrigado, e estou à disposição."

---

## Guia de defesa técnica

**"Por que a acurácia caiu em relação à versão anterior?"**
> "Ela não caiu — o número anterior nunca existiu como desempenho. Era o máximo de 70.906 configurações medidas no próprio teste. Quando apliquei a mesma busca-oráculo a um pool novo e independente, ela reproduziu exatamente 85,41%, o que mostra que aquele número era propriedade do procedimento, não do modelo. 81,62% é o que o sistema entrega em dados que ele nunca viu."

**"Como você garante que o 81,62% também não é sorte?"**
> "Três coisas. A seleção usou só a validação. O resultado fica a meio desvio da média da distribuição dos 3.800 trios no teste, ou seja, é um trio típico e não um outlier. E o AUC-ROC, que independe de limiar, confirma a ordenação das arquiteturas."

**"Por que não um Video Transformer?"**
> "Restrição de dados e de borda: 2.000 vídeos é pouco para atenção espaço-temporal treinada do zero, e o requisito era CPU. Não benchmarkei um Transformer nesta entrega, então não vou afirmar um resultado que não medi."

**"Por que congelar o backbone?"**
> "Custo e overfitting de cenário. Com câmeras fixas, fine-tuning faz os filtros se especializarem no fundo. Além disso, congelar permite cachear features e treinar 80 modelos em minutos — é o que viabilizou toda a análise estatística que eu mostrei."

**"O dataset é desbalanceado?"**
> "Não — é 1.000/1.000 por construção. O `fight_weight = 1.35` não é tratamento de desbalanceamento, é ponderação de custo assimétrico entre FN e FP: uma decisão de negócio codificada na loss."

**"Como isso vai para produção?"**
> "Três frentes. Calibrar o limiar por política de cliente — o `calibrate_threshold.py` faz isso na validação, com critério de recall sob precisão mínima. Quantizar INT8 partindo dos grafos ONNX já verificados. E atacar a decodificação, que hoje é o gargalo real do pipeline."

---

## Erros a evitar na gravação

1. **Não** dizer "SOTA" — o RWF-2000 tem trabalhos publicados acima deste patamar. A escolha de backbone congelado por orçamento de borda é defensável; o rótulo não é.
2. **Não** citar resultados de Transformer, "operadores perdem 95% dos eventos" ou "correlação de 98% entre quadros" — nada disso está medido no repositório.
3. **Não** apresentar `training_curves.png` como análise geral sem dizer que é a curva do **Modelo 1**.
4. **Não** rodar comando ao vivo que não foi testado na máquina de gravação minutos antes.
5. **Não** apresentar o viés de seleção com tom de desculpa. Foi encontrado, medido e corrigido — é competência demonstrada, não falha admitida.
