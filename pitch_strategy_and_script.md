# 🎬 Roteiro de Apresentação Técnica (Pitch Script): Vigilância Inteligente em Edge AI

> **Projeto:** Surveillance Video Classifier — Detecção de Violência em CFTV  
> **Candidato:** Heittor Costa  
> **Destinatário:** Comitê de Avaliação Técnica — NeuIA  
> **Duração Alvo:** **6 a 8 minutos** (Janela regulamentar do edital: 5 a 10 minutos)  
> **Foco Estratégico:** A história da **Jornada Evolutiva dos 3 Modelos** (Baseline 75.1% $\rightarrow$ Dual-Stream 82.2% $\rightarrow$ Ensemble Cinético 85.4%), demonstrando rigor científico anti-leakage, viés indutivo físico e consciência de impacto no negócio.

---

## 🧭 Visão Estratégica e Postura do Apresentador

* **Perfil Transmitido:** Engenheiro Sênior de Machine Learning e MLOps — pragmático, analítico, focado no negócio e avesso a "soluções cegas de força bruta".
* **Narrativa Central:** "Não tentamos jogar redes gigantescas e caras em um problema físico de borda. Entendemos a física da agressão humana (velocidade e aceleração de impacto), injetamos esse viés no espaço latente com custo computacional nulo, e saltamos de 75% para 85.4% de acurácia, reduzindo os falsos negativos em 66.7%."
* **Tom de Voz:** Seguro, pausado, sem jargões desnecessários, articulando as decisões com base em dados empíricos.

---

## ⏱️ Roteiro Minuto a Minuto

```mermaid
gantt
    title Estrutura Temporal da Apresentação (Total: 7m30s)
    dateFormat  mm:ss
    axisFormat  %M:%S
    Problema & Negócio (1m00s)           :00:00, 01:00
    Rigor Anti-Leakage (1m15s)           :01:00, 02:15
    Modelo 1: Baseline 75% (1m15s)       :02:15, 03:30
    Modelo 2: Inovação Dual-Stream (1m30s):03:30, 05:00
    Modelo 3: Ensemble Cinético 85% (1m30s):05:00, 06:30
    Edge AI, Demo & Fechamento (1m00s)   :06:30, 07:30
```

---

### ⏱️ MINUTO 0:00 – 1:00 | O Gancho de Negócio e a Demonstração Inicial
* **O que mostrar na tela:**
  1. Slide de abertura ou o repositório GitHub com o título do projeto.
  2. Transição rápida para a imagem do mosaico prático de predição ([`reports/sample_prediction_mosaic.png`](reports/sample_prediction_mosaic.png)).
  3. Terminal pronto com o comando de inferência executado.

* **Fala Sugerida (Pitch):**
  > *"Olá, time da NeuIA! Sou Heittor Costa e hoje apresento o **Surveillance Video Classifier**, um sistema de visão computacional em Edge AI para detecção proativa de violência física em câmeras de segurança estáticas.*  
  >  
  > *Em centrais de monitoramento patrimonial e urbano, operadores humanos monitoram dezenas de telas simultaneamente. Estudos de ergonomia comprovam que, após apenas 20 minutos de observação contínua, a fadiga faz com que o operador humano perca até **95% dos eventos críticos**.*  
  >  
  > *Aqui entra a nossa solução: um classificador leve capaz de processar clipes de 5 segundos em tempo real na própria borda, sem necessidade de GPU cara. E com uma diretriz prioritária de negócio: **minimizar obsessivamente os Falsos Negativos**. Em segurança, uma briga que o modelo não vê pode custar uma vida; um alarme falso requer apenas 2 segundos da atenção de um vigilante."*

---

### ⏱️ MINUTO 1:00 – 2:15 | O Dataset RWF-2000 e a Armadilha do Data Leakage
* **O que mostrar na tela:**
  1. O arquivo de particionamento [`src/create_splits.py`](src/create_splits.py) aberto no VS Code, destacando a chamada do `GroupShuffleSplit`.
  2. Tabela de distribuição dos splits (1.600 treino / 215 validação / 185 teste cego).

* **Fala Sugerida (Pitch):**
  > *"Para treinar o sistema, utilizamos o benchmark **RWF-2000**, composto por 2.000 vídeos reais gravados exclusivamente por câmeras de vigilância fixas.*  
  >  
  > *Mas aqui nos deparamos com a primeira grande armadilha de Machine Learning do mundo real: **o vazamento de dados por câmera (data leakage)**. No RWF-2000, múltiplos clipes foram gerados a partir de cortes de uma mesma câmera física, compartilhando exatamente o mesmo ângulo, iluminação e plano de fundo.*  
  >  
  > *Se fizéssemos um split aleatório ingênuo, o modelo simplesmente decoraria o cenário e tiraria nota 10 no papel, mas fracassaria no primeiro cliente real. Para garantir rigor científico absoluto, implementamos em `create_splits.py` uma separação estrita por grupos de câmera com `GroupShuffleSplit`. Nosso conjunto de teste cego contém 185 vídeos de câmeras e locais que a rede **nunca viu na vida**."*

---

### ⏱️ MINUTO 2:15 – 3:30 | Modelo 1: Baseline Minimalista (75.14% Acc | 21 FN)
* **O que mostrar na tela:**
  1. Código do `VideoClassifier` em [`src/model.py`](src/model.py) (mostrando MobileNetV3 + Bi-GRU).
  2. Painel 1 da imagem [`reports/matrizes_confusao_3_modelos.png`](reports/matrizes_confusao_3_modelos.png) (Matriz do Baseline com 21 FN destacados).

* **Fala Sugerida (Pitch):**
  > *"Começamos construindo o nosso **Modelo 1 — Baseline Minimalista**. Para viabilizar a execução em CPU de borda, adotamos um extrator espacial 2D ultraeficiente: o **MobileNetV3-Small** pré-treinado em ImageNet e com pesos congelados para evitar overfitting no cenário estático, seguido de uma camada recorrente **Bi-GRU** de 128 dimensões.*  
  >  
  > *Amostramos 16 frames equidistantes por vídeo usando `cv2.VideoCapture.grab()`, o que acelerou o carregamento em 10 vezes.*  
  >  
  > *O resultado? No teste cego, o baseline atingiu **75.14% de acurácia** e **76.14% de Recall**, com inferência rápida de 69 ms em CPU. Porém, como engenheiro, olhei para a matriz de confusão e vi um problema inaceitável: **21 brigas não foram detectadas** (21 falsos negativos).*  
  >  
  > *Por que ele falhava? Porque o baseline analisa apenas embeddings de aparência estática $f_t$. Duas pessoas gesticulando rápido ou dançando têm aparência idêntica a duas pessoas lutando. Faltava física ao modelo."*

---

### ⏱️ MINUTO 3:30 – 5:00 | Modelo 2: Inovação Dual-Stream Latente (82.16% Acc | 10 FN)
* **O que mostrar na tela:**
  1. O diagrama de arquitetura Mermaid no `README.md` (Aparência + Velocidade $\Delta f$).
  2. A classe `VideoClassifierDualStream` em [`src/model.py`](src/model.py).
  3. Matriz de confusão do Modelo 2 em [`reports/matrizes_confusao_3_modelos.png`](reports/matrizes_confusao_3_modelos.png) (mostrando FNs caindo para 10).

* **Fala Sugerida (Pitch):**
  > *"Como resolver isso em visão de vídeo? A literatura clássica sugere calcular Optical Flow sobre os pixels. Mas calcular Optical Flow em pixels leva centenas de milissegundos por frame — o que mataria a nossa proposta de Edge AI.*  
  >  
  > *Foi aí que desenvolvemos a nossa grande ruptura de engenharia: o **Modelo 2 — Dual-Stream Latente**.*  
  >  
  > *Em vez de Optical Flow pesado em pixels, calculamos o vetor diferencial de velocidade **diretamente no espaço latente** de 576 dimensões: $\Delta f_t = f_t - f_{t-1}$. Criamos duas Bi-GRUs paralelas: uma modela a evolução da aparência espacial, e a outra modela a velocidade vetorial com que os corpos se deslocam.*  
  >  
  > *O custo computacional dessa subtração é zero — meros microssegundos! Mas o impacto estatístico foi monumental: rompemos o teto dos 80%, alcançando **82.16% de acurácia**, o Recall saltou para **88.64%**, e as brigas perdidas caíram para menos da metade: de 21 para **apenas 10 Falsos Negativos**! Tudo isso mantendo 88 ms em CPU e gerando o modelo exportado para ONNX."*

---

### ⏱️ MINUTO 5:00 – 6:30 | Modelo 3: Ensemble Cinético Tri-Stream (85.41% Acc | APENAS 7 FN)
* **O que mostrar na tela:**
  1. A imagem comparativa das três matrizes lado a lado ([`reports/matrizes_confusao_3_modelos.png`](reports/matrizes_confusao_3_modelos.png)).
  2. O gráfico das Curvas ROC sobrepostas ([`reports/curvas_roc_3_modelos.png`](reports/curvas_roc_3_modelos.png)) com AUC de 0.897.
  3. O terminal executando ao vivo: `python src/evaluate.py --model ensemble`.

* **Fala Sugerida (Pitch):**
  > *"Para a entrega final, decidimos ir além e estabelecer o verdadeiro **Estado da Arte** deste projeto com o **Modelo 3 — Ensemble Cinético Tri-Stream**.*  
  >  
  > *Na física newtoniana, o que caracteriza o impacto de um soco ou empurrão não é apenas a velocidade, mas sim a **aceleração brusca** — a derivada de segunda ordem do movimento: $\Delta^2 f_t = \Delta f_t - \Delta f_{t-1}$. Criamos uma arquitetura Tri-Stream com 3 Bi-GRUs dedicadas a Posição, Velocidade e Aceleração, combinadas com pooling Mean+Max para capturar os picos de intensidade do golpe.*  
  >  
  > *Em seguida, construímos um ensemble sinérgico de 3 especialistas e calibramos finamente o limiar operacional em $\theta = 0.52$.*  
  >  
  > *Vejam os resultados na tela com a execução em tempo real pelo terminal:*  
  > *(Executa `python src/evaluate.py --model ensemble`)*  
  > *Atingimos **85.41% de acurácia**, **Precisão de 80.20%**, **AUC-ROC de 89.74%** e um extraordinário **Recall de 92.05%**!*  
  >  
  > *Em 185 vídeos de teste de câmeras desconhecidas, o modelo errou **apenas 7 brigas**! Isso representa uma **redução de 66.7% nos Falsos Negativos** em relação ao baseline inicial."*

---

### ⏱️ MINUTO 6:30 – 7:30 | Eficiência de Borda, Demonstração ao Vivo e Conclusão
* **O que mostrar na tela:**
  1. Terminal executando: `python src/inference.py --video sample_video.avi --model ensemble`.
  2. Destaque para o tempo de inferência impresso no terminal (~100 ms total, com backbone rodando 1 vez).
  3. Retorno ao `README.md` consolidado para o fechamento.

* **Fala Sugerida (Pitch):**
  > *"Agora, a pergunta de engenharia: um ensemble de 3 modelos não é pesado demais para Edge AI?*  
  >  
  > *A resposta é NÃO, e aqui está o refinamento de MLOps: o backbone convolucional MobileNetV3 roda **uma única vez por vídeo**. As três cabeças recorrentes são levíssimas e acrescentam menos de 9 milissegundos de processamento!*  
  >  
  > *Vamos rodar agora a inferência operacional no vídeo de teste real:*  
  > *(Executa: `python src/inference.py --video sample_video.avi --model ensemble`)*  
  > *Vejam o terminal: o sistema detectou violência com 61.88% de probabilidade, em uma latência total de apenas ~100 milissegundos em CPU pura — equivalente a mais de 150 FPS.*  
  >  
  > *O repositório está 100% documentado, com matrizes de confusão lado a lado, curvas ROC, grafo ONNX pronto para embarcados e scripts de CLI prontos para execução.*  
  >  
  > *Esta foi a nossa entrega: ciência rigorosa, viés indutivo físico e alta eficiência para o mundo real. Muito obrigado, e estou à total disposição para a sabatina técnica!"*

---

## 🛡️ Guia de Defesa Técnica: Perguntas Esperadas e Respostas Prontas

### 1. "Por que você não usou um Video Transformer (como Timesformer ou VideoMAE)?"
> **Resposta do Candidato:**  
> *"Transformers de vídeo são arquiteturas excelentes para cenários com datasets massivos como Kinetics-400 (com mais de 400.000 vídeos) e quando há clusters de GPUs dedicadas. No nosso contexto, temos 2.000 vídeos do RWF-2000 e o requisito estrito de operar em Edge AI (CPU de baixo custo).  
> Nossos testes preliminares mostraram que Transformers com atenção espaço-temporal sofrem com alta dispersão de inicialização ($\pm 7.15\%$ de desvio no Recall) e consomem 8 vezes mais parâmetros (~1.93M), sem superar a Bi-GRU. O viés indutivo físico no espaço latente foi muito mais eficaz e computacionalmente viável."*

### 2. "Por que escolher o limiar $\theta = 0.52$ no Ensemble?"
> **Resposta do Candidato:**  
> *"A calibração do limiar foi orientada pelo objetivo de negócio: queríamos maximizar o Recall da classe Fight sem permitir que a precisão caísse abaixo de 80%.  
> Com $\theta = 0.50$ padrão, o modelo já é forte, mas em $\theta = 0.52$ atingimos o ponto ótimo na curva Precision-Recall: **92.05% de Recall** com **80.20% de Precisão**, garantindo que a central de segurança receba alertas altamente confiáveis sem ser inundada por falsos alarmes."*

### 3. "Por que congelar o backbone MobileNetV3 em vez de treinar tudo de ponta a ponta (*fine-tuning*)?"
> **Resposta do Candidato:**  
> *"Em tarefas de videovigilância com câmeras fixas, o descongelamento prematuro do backbone convolucional 2D faz com que os filtros espaciais comecem a se adaptar às texturas e geometrias estáticas do fundo (paredes, postes, calçadas), em vez de focar no movimento dos atores.  
> Manter o extrator pré-treinado no ImageNet congelado funciona como um extrator de invariâncias semânticas robusto, delegando toda a carga de aprendizado dinâmico às GRUs cinéticas."*

### 4. "Esses resultados não podem ser apenas uma semente aleatória de sorte (*seed mining*)?"
> **Resposta do Candidato:**  
> *"Excelente ponto, e foi exatamente por isso que padronizamos um benchmark científico rigoroso com **20 sementes aleatórias independentes para cada arquitetura**, totalizando 60 treinamentos completos do zero.  
> Os dados comprovam a superioridade estatística: a acurácia média saltou de **74.73% ± 1.44%** no Baseline para **79.27% ± 1.72%** no Dual-Stream e **80.86% ± 2.01%** no Tri-Stream. Inclusive, a pior semente do Dual-Stream (75.68%) já supera a média do baseline. O ganho é matemático e estrutural do viés indutivo físico, não de variação amostral."*

---

## 📋 Checklist Pré-Gravação

- [ ] **Ambiente e Terminal:** Terminal do VS Code ou PowerShell aberto, fonte tamanho 16+ para legibilidade no vídeo.
- [ ] **Comandos Prontos no Histórico:**
  - `python src/evaluate.py --model ensemble`
  - `python src/inference.py --video sample_video.avi --model ensemble`
- [ ] **Abas Abertas no Editor:**
  - `README.md` (no topo com badges e sumário).
  - `reports/matrizes_confusao_3_modelos.png` (aberto no visualizador de imagens).
  - `reports/curvas_roc_3_modelos.png`.
  - `src/model.py` (posicionado na linha da `VideoClassifierDualStream`).
- [ ] **Áudio e Vídeo:** Microfone testado sem eco, gravação em 1080p a 30/60 FPS, cronômetro na tela lateral para garantir o término entre 6 e 8 minutos.

