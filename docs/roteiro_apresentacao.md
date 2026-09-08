# Roteiro do Vídeo de Apresentação — Surveillance Video Classifier

**Candidato:** Heittor Costa · **Destinatário:** NeuIA · **Duração alvo:** 8m00s (janela do edital: 5–10 min)

**Princípio do roteiro:** todo número dito em voz alta precisa existir num arquivo do repositório, e o número mais frágil da entrega (o 85.41%) é apresentado *por você*, no seu enquadramento, antes que o avaliador o encontre sozinho. Honestidade estatística demonstrada vale mais do que 4 pontos percentuais de acurácia.

---

## Pré-gravação — checklist obrigatório

- [ ] `python src/build_cache.py` executado (gera `data/cache/features_{train,train_flip,val,test}.pt`)
- [ ] `python src/evaluate.py --model ensemble` **roda de verdade** na máquina de gravação
- [ ] `python tests/test_splits.py` passando — você vai rodar isso na tela no minuto 1:00
- [ ] `python src/inference.py --video sample_video.avi --model ensemble` roda e imprime a latência
- [ ] `reports/benchmark_20_ensembles_detalhes.csv` aberto numa aba — você vai mostrar esse arquivo
- [ ] Terminal em fonte 16+, 1080p, microfone testado sem eco
- [ ] Abas do editor: `README.md`, `src/create_splits.py`, `src/model.py` (na `TriStream_Kinetic`), `reports/matrizes_confusao_3_modelos.png`, `reports/benchmark_3_modelos_20_runs_boxplots.png`
- [ ] **Não** prometer em fala nenhum número que não esteja num arquivo (sem resultados de Transformer, sem "95% dos eventos", sem "correlação de 98%")

---

## 0:00 – 0:50 · Problema e critério de sucesso

**Tela:** repositório aberto no README, depois `reports/sample_prediction_mosaic.png`.

> "Olá, sou o Heittor. Apresento o Surveillance Video Classifier: um classificador de vídeo para detecção de agressão física em câmeras fixas de CFTV, projetado para rodar em CPU na borda.
>
> A decisão de engenharia que orienta o projeto inteiro é a assimetria de custo entre os dois tipos de erro. Um falso positivo custa alguns segundos de atenção de um vigilante. Um falso negativo é uma agressão que ninguém viu. Então não otimizei acurácia: otimizei **recall da classe Fight sob restrição de precisão**, e vou mostrar os números dos dois lados.
>
> A entrega tem três modelos, um benchmark de 60 treinamentos e uma discussão honesta sobre o que esses números significam e o que eles não significam."

---

## 0:50 – 2:00 · Dataset e a armadilha do vazamento

**Tela:** `src/create_splits.py`, destacando o `GroupShuffleSplit`; depois a tabela de splits do README.

> "O dataset é o RWF-2000, de Cheng e colegas, 2020: 2.000 clipes reais de câmeras de vigilância estáticas, 5 segundos a 30 FPS, balanceado em 1.000 Fight e 1.000 NonFight. Fight cobre socos, chutes, empurrões e confrontos corporais; NonFight é monitoramento normal — caminhadas, conversas, aglomerações pacíficas, tráfego.
>
> A armadilha aqui é específica desse dataset: vários clipes vêm de cortes temporais **da mesma câmera física** — mesmo fundo, mesma iluminação, mesmo ângulo. Com um split aleatório ingênuo, o modelo decora o cenário e você publica um número que não sobrevive ao primeiro cliente.
>
> Então o particionamento é por grupo. Extraio o identificador do vídeo original do nome do arquivo e uso `GroupShuffleSplit` para que nenhum grupo atravesse a fronteira dos splits. O resultado são 1.600 vídeos de treino, 215 de validação e 185 de teste cego, e eu verifiquei a interseção explicitamente: **zero grupos e zero arquivos compartilhados** entre treino, validação e teste. As 90 câmeras do teste são inéditas para o modelo."

*Rode `python tests/test_splits.py` na tela aqui — cinco testes, cinco segundos, e a afirmação deixa de ser palavra e vira verificação.*

---

## 2:00 – 3:00 · Pré-processamento

**Tela:** `src/dataset.py`.

> "Cada clipe tem 150 frames. Amostro 16 equidistantes, cobrindo os 5 segundos inteiros — frames vizinhos a 30 FPS são quase redundantes, e isso corta o volume de dados em 89%.
>
> A decodificação usa `grab()` para avançar o cursor e `retrieve()` só nos 16 instantes que interessam, evitando decodificar os 134 frames descartados.
>
> Depois: redimensionamento para 224×224, BGR para RGB e normalização com as estatísticas do ImageNet — porque o backbone é pré-treinado nelas, e ignorar isso desalinha as ativações desde a primeira camada.
>
> E uma nota sobre augmentation que vale a pena contar, porque foi uma armadilha que eu mesmo criei. Como o backbone é congelado, eu cacheio as features para acelerar o treino em duas ordens de grandeza. Na primeira versão, o flip horizontal era aplicado nessa extração única — ou seja, cada vídeo recebia uma perturbação fixa, idêntica em todas as épocas. Isso não é augmentation, é ruído congelado.
>
> A correção está no código: `build_cache.py` materializa **dois** caches, o original e o espelhado, e o treino sorteia entre eles por amostra a cada época. Custa o dobro de disco no cache e zero de compute no treino."

---

## 3:00 – 4:00 · Modelo 1 — Baseline

**Tela:** `VideoClassifier` em `src/model.py`; painel 1 de `matrizes_confusao_3_modelos.png`.

> "O Modelo 1 é o baseline: MobileNetV3-Small pré-treinado no ImageNet e **congelado** como extrator espacial de 576 dimensões, seguido de uma Bi-GRU com pooling médio temporal. 1,17 milhão de parâmetros.
>
> Congelei o backbone de propósito. Com câmeras fixas, descongelar faz os filtros convolucionais se adaptarem à textura do fundo — paredes, calçadas, postes — em vez do movimento dos atores. O aprendizado dinâmico fica todo na parte recorrente.
>
> No teste cego: 75,14% de acurácia, recall de 76,14%. E o problema visível na matriz: **21 agressões não detectadas**. A causa é estrutural — o baseline só enxerga aparência estática. Duas pessoas gesticulando com energia têm embedding parecido com duas pessoas brigando. Falta a derivada."

---

## 4:00 – 5:00 · Modelo 2 — Dual-Stream latente

**Tela:** `VideoClassifierDualStream`; painel 2 da matriz.

> "A resposta clássica seria optical flow. Mas optical flow em pixels custa centenas de milissegundos por frame e mata a proposta de borda.
>
> Então calculo a velocidade **no espaço latente**: delta f igual a f_t menos f_{t-1}, sobre os vetores de 576 dimensões que já extraí. Duas Bi-GRUs em paralelo, uma sobre aparência e outra sobre velocidade, fundidas na cabeça classificadora. O custo da subtração é desprezível.
>
> Resultado: 82,16% de acurácia, recall de 88,64%, e os falsos negativos caem de 21 para **10**."

---

## 5:00 – 6:15 · Modelo 3, e a validação estatística que importa

**Tela:** `TriStream_Kinetic`; depois `benchmark_3_modelos_20_runs_boxplots.png`.

> "O Modelo 3 adiciona a segunda derivada — aceleração, delta-quadrado f — porque o que caracteriza um impacto não é a velocidade, é a mudança brusca dela. Três streams, pooling mean **e** max para capturar picos em vez de diluí-los na média. E três cabeças especializadas em comitê, todas partilhando o **mesmo** backbone: o MobileNet roda uma vez por vídeo, as cabeças custam milissegundos.
>
> Mas aqui está a parte que eu considero o núcleo técnico da entrega. Um único número de acurácia não prova nada em dataset pequeno. Então padronizei um protocolo e rodei **20 sementes independentes para cada uma das 3 arquiteturas — 60 treinamentos completos**, mesmo early stopping, mesma loss, mesmo otimizador, avaliação sobre os mesmos 185 vídeos.
>
> [aponta para os boxplots] O ganho é estrutural, não sorte: baseline 74,73% ± 1,44; dual-stream 79,27% ± 1,72; tri-stream 80,86% ± 2,01. **A pior semente do dual-stream supera a média do baseline.** E o AUC-ROC, que é livre de limiar, sobe de 84,90% para 89,49%. É isso que sustenta a afirmação de que o viés indutivo cinético funciona."

---

## 6:15 – 7:00 · O 85,41% — e o que ele realmente é

**Tela:** `reports/benchmark_20_ensembles_detalhes.csv` aberto, ao lado da matriz do Modelo 3.

> "O README destaca 85,41% de acurácia e 7 falsos negativos, e eu quero ser preciso sobre o que esse número é, porque a distinção importa.
>
> Esse é o resultado do **melhor checkpoint** — um trio específico de sementes com limiar 0,52. Quando eu formo 20 comitês triplos de forma sistemática, sem escolher, a distribuição é 81,11% ± 1,21, com máximo de 83,78%. O trio entregue supera todos os 20.
>
> E o diagnóstico está no AUC: o campeão tem 89,74%, contra 89,37% ± 0,41 da distribuição — praticamente idêntico. Como o AUC não depende de limiar, isso me diz que o poder discriminativo do campeão é **normal**; o ganho de acurácia veio da escolha do trio e do limiar. Ou seja: houve seleção de modelo com informação do teste.
>
> Então a leitura correta é: **o desempenho esperado do método é 81,1% ± 1,2%, com recall de 84,4% ± 2,3**. O 85,41% é o melhor ponto de operação encontrado, não uma estimativa não-enviesada.
>
> E a correção não ficou só no discurso: o repositório tem `src/select_ensemble.py`, que escolhe o trio olhando apenas os 215 vídeos de validação, e `src/calibrate_threshold.py`, que calibra o limiar do mesmo jeito. Os dois avaliam o teste uma única vez, no fim. O que falta é retreinar o pool de candidatos — é o primeiro item da lista de próximos passos.
>
> Eu prefiro entregar esse número com a ressalva do que entregá-lo sem ela."

*Por que isso está no roteiro: um avaliador de ML abre o CSV e encontra isso em minutos. Dito por você, vira demonstração de rigor. Encontrado por ele, vira dúvida sobre todo o resto.*

---

## 7:00 – 7:40 · Demonstração e perfil de borda

**Tela:** terminal executando os dois comandos.

```
python src/evaluate.py --model ensemble
python src/inference.py --video sample_video.avi --model ensemble
```

> "Avaliação sobre os 185 vídeos cegos: acurácia, precisão, recall, F1, AUC e matriz de confusão saem impressos, e ficam versionados em `reports/`.
>
> E a inferência sobre o vídeo de demonstração — um clipe de câmera fixa de 5 segundos, 320 por 240, que mostra uma agressão real. O sistema classifica como Fight com 61,9% de probabilidade contra o limiar de 0,52. Acertou, mas com margem de 10 pontos, e vale dizer isso: não é uma detecção confortável. Vídeos com o agressor pequeno no enquadramento ficam perto da fronteira de decisão.
>
> Sobre latência: o `benchmark_detailed_latency.py` roda os três modelos no mesmo laço, na mesma execução, e separa decodificação de forward — a versão anterior misturava medições de execuções diferentes e chegava ao absurdo de o ensemble aparecer mais rápido que o dual-stream, rodando o mesmo backbone mais uma cabeça a mais. O script registra o perfil de CPU junto com os números, porque latência sem hardware declarado não é comparável.
>
> E o ONNX: a exportação agora carrega os pesos treinados — a versão anterior exportava uma instância recém-inicializada, ou seja, um grafo de ruído. Tem um `verify_onnx_parity.py` que compara a saída do grafo com a do PyTorch e mede o ONNX Runtime. Enquanto eu não rodar isso no hardware alvo, 'exportado para ONNX' é uma pendência, não um resultado."

---

## 7:40 – 8:00 · Fechamento

> "Resumindo as decisões: split por grupo de câmera para que o número de teste signifique alguma coisa; transfer learning com backbone congelado pelo custo e pelo risco de overfitting de cenário; derivadas cinéticas no espaço latente em vez de optical flow, pelo orçamento de borda; e validação por 60 treinamentos em vez de um número único.
>
> As limitações que eu reconheço: augmentation efetivamente inerte pelo desenho do cache, seleção de modelo contaminada pelo teste no checkpoint campeão, ONNX exportado mas não medido, e o dataset restrito a iluminação diurna e câmeras estáticas — PTZ e visão noturna exigem dados de domínio.
>
> As próximas iterações, em ordem: mover a calibração para a validação, augmentation por época com cache duplo, quantização INT8 e benchmark em ONNX Runtime. Obrigado, e estou à disposição."

---

## Guia de defesa técnica

**"Você escolheu as sementes olhando o teste?"**
> "Sim, no checkpoint campeão, e é por isso que eu apresento o 81,1% ± 1,2% como o resultado do método. Os 20 comitês sistemáticos estão no CSV justamente para tornar isso auditável. A correção é selecionar na validação; não refiz a tempo desta entrega e preferi documentar a ressalva a apagá-la."

**"Por que não um Video Transformer (VideoMAE, TimeSformer)?"**
> "Restrição de dados e de borda: 2.000 vídeos é pouco para atenção espaço-temporal treinada do zero, e o requisito era CPU. Não cheguei a benchmarkar um Transformer nesta entrega — não vou afirmar um resultado que não medi. O que posso afirmar é que a Bi-GRU sobre features congeladas atende o orçamento de latência e que o ganho veio do viés cinético, que está isolado no benchmark de 60 runs."

**"Por que congelar o backbone?"**
> "Custo e overfitting de cenário. Com câmeras fixas, fine-tuning faz os filtros se especializarem no fundo. Além disso, congelar permite cachear as features e treinar as cabeças em segundos, o que viabilizou os 60 treinamentos do benchmark — que é o que dá confiança estatística à comparação."

**"O dataset é desbalanceado?"**
> "Não — é 1.000/1.000 por construção, e o treino é 800/800. O `fight_weight=1.35` não é tratamento de desbalanceamento, é ponderação de **custo assimétrico**: um falso negativo vale mais que um falso positivo no domínio de segurança. É uma decisão de negócio codificada na loss, não uma correção estatística."

**"O vídeo de demonstração é novo?"**
> "É um clipe de câmera fixa no mesmo formato do RWF-2000, de câmera não vista em treino. Não é um vídeo externo capturado por mim — se o critério for esse, é uma limitação da demonstração, e a forma correta de fechar essa lacuna é gravar um clipe próprio de 5 segundos e rodar o mesmo comando."

**"Como isso vai para produção?"**
> "Três frentes: calibrar o limiar por política de cliente — a curva permite trocar recall por precisão explicitamente; quantizar para INT8 e medir em ONNX Runtime ou OpenVINO no hardware alvo; e janela deslizante com voto temporal sobre clipes contínuos, já que hoje o modelo consome clipes de 5 segundos e não stream."

---

## Erros a evitar na gravação

1. **Não** dizer "SOTA" — RWF-2000 tem trabalhos publicados acima de 85%. O termo convida uma comparação que você perde por escolha de escopo (backbone congelado por causa da borda), e essa escolha é defensável; o rótulo não é.
2. **Não** citar os resultados preliminares de Transformer do pitch antigo (±7,15%, 1,93M params) — não há nada no repositório que os sustente.
3. **Não** dizer "operadores perdem 95% dos eventos" sem a citação em tela.
4. **Não** apresentar `training_curves.png` como análise geral de overfitting sem dizer que é a curva do **Modelo 1** — e ela mostra val loss subindo a partir da época 3, o que é honesto e bem tratado pelo early stopping, mas precisa ser nomeado.
5. **Não** rodar comando ao vivo que não foi testado na máquina de gravação minutos antes.
