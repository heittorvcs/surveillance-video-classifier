# Surveillance Video Classifier: Detecção de Violência em CCTV (Edge AI)

[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-ee4c2c.svg)](https://pytorch.org/)
[![ONNX](https://img.shields.io/badge/ONNX-Edge%20AI-005ced.svg)](https://onnx.ai/)
[![Acurácia](https://img.shields.io/badge/Acur%C3%A1cia%20(20%20comit%C3%AAs)-81.1%25%20%C2%B1%201.2-brightgreen.svg)]()
[![Recall Fight](https://img.shields.io/badge/Recall%20Fight-84.4%25%20%C2%B1%202.3-success.svg)]()
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

> Classificação de vídeos de vigilância para detecção de agressão física em câmeras estáticas de CFTV, com inferência em CPU. A diretriz de engenharia é a assimetria de custo entre erros: um falso positivo custa segundos de atenção de um vigilante; um falso negativo é uma agressão que ninguém viu. O projeto otimiza **recall da classe Fight sob restrição de precisão**, e reporta os dois lados.

---

## Vídeo de Apresentação Técnica

> **Link do vídeo (YouTube):** `[INSERIR_LINK_DO_VIDEO_AQUI]`

---

## Sumário dos resultados

Duas leituras diferentes do mesmo sistema, no mesmo conjunto de teste cego de 185 vídeos:

| Leitura | Acurácia | Recall Fight | Precisão | F1 | AUC-ROC | FN |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Desempenho esperado do método** — média de 20 comitês formados sem seleção | **81.11% ± 1.21** | 84.43% ± 2.25 | 77.83% ± 1.98 | 80.96% ± 1.16 | 89.37% ± 0.41 | 13.7 ± 2.0 |
| **Melhor checkpoint entregue** — trio de sementes e limiar escolhidos observando o teste | 85.41% | 92.05% | 80.20% | 85.71% | 89.74% | 7 |

**A primeira linha é a estimativa honesta.** A segunda é um melhor-de-N medido no próprio conjunto de avaliação e está documentada como tal na [seção 5](#5-validação-estatística-e-o-limite-do-número-de-vitrine). O repositório traz as ferramentas para refazer a seleção corretamente sobre a validação ([`src/select_ensemble.py`](src/select_ensemble.py), [`src/calibrate_threshold.py`](src/calibrate_threshold.py)).

---

## Demonstração prática de inferência

Os 16 quadros amostrados uniformemente do vídeo de demonstração ([`sample_video.avi`](sample_video.avi)) e a predição obtida sobre eles:

![Mosaico de predição](reports/sample_prediction_mosaic.png)

O clipe é uma gravação de câmera fixa em formato RWF-2000 (5 s, 320×240, 30 FPS), rótulo real `Fight`, de câmera não vista em treino. Não é um vídeo capturado fora do domínio do dataset — ver [Limitações](#10-limitações-e-trabalhos-futuros).

---

## 1. Dataset RWF-2000 e anti-leakage

**Fonte:** RWF-2000 — *Real World Fighting*, Cheng, Cai & Li (2020).
Repositório oficial: <https://github.com/mchengny/RWF2000-Video-Database-for-Violence-Detection>
Espelho comum: <https://www.kaggle.com/datasets/vulamnguyen/rwf2000>

2.000 gravações reais de câmeras de vigilância estáticas, 5 segundos a 30 FPS (150 quadros por clipe), 1.000 `Fight` e 1.000 `NonFight`.

* **`Fight` (1):** agressões físicas — socos, chutes, empurrões, confrontos corporais.
* **`NonFight` (0):** monitoramento normal — caminhadas, conversas, aglomerações pacíficas, tráfego.

Após baixar, extraia para `archive/RWF-2000/{train,val}/{Fight,NonFight}/*.avi`.

### Tratamento anti-leakage

No RWF-2000, vários clipes vêm de cortes temporais do **mesmo vídeo original** — mesma câmera, mesmo fundo, mesma iluminação, mesmo ângulo. Um split aleatório faria o modelo memorizar cenários em vez de aprender a ação.

[`src/create_splits.py`](src/create_splits.py) deriva o identificador do vídeo original do nome do arquivo (prefixo antes do sufixo `_<n>.avi`) e usa esse identificador como grupo:

| Conjunto | Vídeos | Fight | NonFight | Grupos únicos | Origem |
| :--- | :---: | :---: | :---: | :---: | :--- |
| **Treinamento** | 1.600 | 800 | 800 | 765 | Pasta oficial `train` do RWF-2000 |
| **Validação** | 215 | 112 | 103 | 90 | Pasta oficial `val`, metade por `GroupShuffleSplit` |
| **Teste cego** | 185 | 88 | 97 | 90 | Pasta oficial `val`, outra metade |

A separação treino/validação é herdada do split oficial do RWF-2000; o `GroupShuffleSplit` é aplicado para dividir a pasta oficial `val` em validação e teste sem que uma câmera apareça dos dois lados.

A garantia não é apenas declarada — é verificada. `create_splits.py` falha se qualquer grupo atravessar a fronteira, e há testes automatizados sobre os CSVs versionados:

```bash
python tests/test_splits.py
```

```
[PASS] test_both_classes_present_in_every_split
[PASS] test_labels_are_binary_and_consistent
[PASS] test_no_file_leakage_between_splits
[PASS] test_no_group_leakage_between_splits
[PASS] test_splits_exist_and_have_expected_sizes
```

Interseção medida entre os três splits: **0 grupos e 0 arquivos**.

---

## 2. Decisões de pré-processamento

1. **Amostragem temporal equidistante (N = 16 quadros).** Quadros consecutivos a 30 FPS carregam pouca informação nova; 16 quadros uniformemente espaçados cobrem os 5 segundos inteiros e reduzem o volume de dados em 89,3%.
2. **Decodificação seletiva.** `cv2.VideoCapture.grab()` avança o cursor sem decodificar; `read()` é chamado apenas nos 16 instantes escolhidos, evitando decodificar os 134 quadros descartados.
3. **Padronização espacial e normalização ImageNet.** Redimensionamento para 224×224, BGR → RGB, normalização por canal com μ = [0.485, 0.456, 0.406] e σ = [0.229, 0.224, 0.225] — as mesmas estatísticas com que o backbone foi pré-treinado. O redimensionamento não preserva a proporção 4:3 original.
4. **Data augmentation com consistência temporal.** Flip horizontal aplicado de forma idêntica aos 16 quadros do clipe.

**Sobre o item 4, uma decisão que precisou ser corrigida.** Com o backbone congelado, as features são extraídas uma única vez e cacheadas. Aplicar o flip aleatório nessa passada produziria uma perturbação fixa por vídeo, idêntica em todas as épocas — não um augmentation. A implementação atual materializa **dois caches** ([`src/build_cache.py`](src/build_cache.py)) — `features_train.pt` e `features_train_flip.pt` — e sorteia entre eles por amostra a cada época ([`FlipAugmentedFeatures`](src/train.py)). Custo: o dobro de disco no cache, zero de compute no treino.

---

## 3. Os três modelos

Todas as arquiteturas vivem em [`src/model.py`](src/model.py) e são importadas por treino, benchmarks e avaliação — não há redefinição duplicada.

### Modelo 1 — Baseline
* MobileNetV3-Small pré-treinado no ImageNet e **congelado** (576 dim) + Bi-GRU única (hidden = 64 → 128 dim) com pooling médio temporal.
* Congelar o backbone é deliberado: com câmeras fixas, o fine-tuning faz os filtros convolucionais se especializarem na textura do fundo em vez do movimento dos atores. Também é o que permite cachear features e treinar 60 modelos em minutos.
* Limitação estrutural: só enxerga aparência estática *f_t*. Gesticulação vigorosa e agressão têm embeddings parecidos.
* Artefatos: [`models/best_model.pth`](models/best_model.pth) · [`models/model.onnx`](models/model.onnx)

### Modelo 2 — Dual-Stream latente
* Mesmo backbone congelado + duas Bi-GRUs paralelas: aparência (*f_t*) e velocidade latente (Δ*f_t* = *f_t* − *f_{t−1}*).
* Substitui o optical flow em pixels (centenas de ms por quadro) por uma subtração vetorial no espaço latente de 576 dimensões, de custo desprezível.
* Artefatos: [`models/best_model_dualstream_82acc.pth`](models/best_model_dualstream_82acc.pth) · [`models/model_dualstream.onnx`](models/model_dualstream.onnx)

### Modelo 3 — Ensemble cinético Tri-Stream
* Comitê de 3 cabeças com fusão por média das probabilidades:
  1. **TriStream cinético** — aparência, velocidade Δ*f_t* e aceleração Δ²*f_t* = Δ*f_t* − Δ*f_{t−1}*, com pooling Mean + Max.
  2. **DualStream Mean+Max (semente 5)**
  3. **DualStream Mean+Max (semente 10)**
* O backbone roda **uma única vez por vídeo**; as três cabeças consomem as mesmas features.
* Artefatos: [`models/ensemble/`](models/ensemble/)

---

## 4. Métricas no teste cego (185 vídeos)

| Métrica | Modelo 1 | Modelo 2 | Modelo 3 (melhor checkpoint) |
| :--- | :---: | :---: | :---: |
| Acurácia | 75.14% | 82.16% | 85.41% |
| Recall (Fight) | 76.14% | 88.64% | 92.05% |
| Precisão (Fight) | 72.83% | 77.23% | 80.20% |
| F1-Score | 74.44% | 82.54% | 85.71% |
| AUC-ROC | 86.33% | 88.32% | 89.74% |
| Falsos negativos | 21 | 10 | 7 |
| Falsos positivos | 25 | 23 | 20 |
| Limiar | 0.50 | 0.50 | 0.52 |

Reproduza qualquer linha com:

```bash
python src/evaluate.py --model ensemble --save_json reports/test_metrics_ensemble.json
```

JSONs versionados: [`reports/test_metrics.json`](reports/test_metrics.json) · [`reports/test_metrics_dualstream_82acc.json`](reports/test_metrics_dualstream_82acc.json) · [`reports/test_metrics_85acc.json`](reports/test_metrics_85acc.json)

---

## 5. Validação estatística e o limite do número de vitrine

Um único valor de acurácia não prova nada com 185 vídeos de teste. O protocolo adotado foi treinar **20 sementes independentes para cada uma das 3 arquiteturas — 60 treinamentos completos** — sob condições idênticas: early stopping por `val_loss` com paciência 5, `CrossEntropyLoss` com `fight_weight = 1.35`, `AdamW`, `CosineAnnealingLR`, e avaliação sobre os mesmos 185 vídeos.

| Arquitetura | Acurácia | Faixa | Recall | Precisão | F1 | AUC-ROC | FN | FP |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| Modelo 1: Baseline Bi-GRU | 74.73% ± 1.44 | [71.35 – 76.76] | 69.94% ± 5.13 | 75.43% ± 2.90 | 72.40% ± 2.31 | 84.90% ± 0.71 | 26.4 ± 4.5 | **20.3 ± 4.3** |
| Modelo 2: Dual-Stream latente | 79.27% ± 1.72 | [75.68 – 82.16] | **84.83% ± 5.40** | 75.10% ± 2.68 | 79.52% ± 1.96 | 88.22% ± 1.32 | **13.3 ± 4.7** | 25.0 ± 4.9 |
| Modelo 3: Tri-Stream cinético | **80.86% ± 2.01** | [74.59 – 83.24] | 84.55% ± 6.99 | **77.67% ± 3.58** | **80.69% ± 2.67** | **89.49% ± 1.01** | 13.6 ± 6.2 | 21.8 ± 5.7 |
| Comitês ensemble (20 trios) | **81.11% ± 1.21** | [79.46 – 83.78] | 84.43% ± 2.25 | 77.83% ± 1.98 | 80.96% ± 1.16 | 89.37% ± 0.41 | 13.7 ± 2.0 | 21.2 ± 2.6 |

![Boxplots das distribuições](reports/benchmark_3_modelos_20_runs_boxplots.png)

**O ganho arquitetural é real.** A pior semente do Dual-Stream (75.68%) supera a média do baseline (74.73%), e o AUC-ROC — que independe de limiar — sobe de 84.90% para 89.49%. Isso não é variação estocástica: é o efeito do viés indutivo cinético.

### O que o 85.41% realmente é

O checkpoint entregue (TriStream s7 + DualMeanMax s5 + DualMeanMax s10, θ = 0.52) atinge 85.41% e 7 falsos negativos — **acima de todos os 20 comitês formados sistematicamente**, cujo máximo é 83.78% e cujo melhor FN é 10. Ele também não é nenhuma das 20 combinações avaliadas em [`reports/benchmark_20_ensembles_detalhes.csv`](reports/benchmark_20_ensembles_detalhes.csv).

O diagnóstico está no AUC-ROC. O campeão marca 89.74% contra 89.37% ± 0.41 da distribuição — cerca de 0.9 desvio padrão, ou seja, estatisticamente comum. Como o AUC não depende do limiar, **o poder discriminativo do campeão é normal**; os +4.3 pontos percentuais de acurácia vêm da escolha do trio e do limiar contra os rótulos do teste.

Conclusão: o teste é cego para o **treinamento**, mas não foi cego para a **seleção de modelo**. A leitura correta do sistema é 81.11% ± 1.21 de acurácia e 84.43% ± 2.25 de recall.

**Como refazer a seleção corretamente** (o teste é tocado uma única vez, no fim):

```bash
# 1. Pool de candidatos — o teste nunca é consultado
for s in $(seq 1 20); do
  python src/train.py --arch tristream   --seed $s --out models/pool/tristream_s$s.pth
  python src/train.py --arch dualmeanmax --seed $s --out models/pool/dualmeanmax_s$s.pth
done

# 2. Trio e limiar escolhidos APENAS na validação
python src/select_ensemble.py --criterion recall_at_precision --min_precision 0.80 --export

# 3. Teste avaliado uma única vez
python src/select_ensemble.py --eval_test
```

### Análise de overfitting / underfitting

![Curvas de treinamento](reports/training_curves.png)

As curvas acima são do **Modelo 1**. Elas mostram o comportamento clássico: a perda de treino cai continuamente enquanto a de validação para de melhorar a partir da terceira época. O early stopping interrompeu na oitava época e restaurou o checkpoint da terceira — a divergência foi detectada e contida, não evitada.

Curvas dos demais modelos são geradas por `python src/train.py --arch <arch>`, que salva `reports/training_curves_<arch>_seed<seed>.png` e o histórico em JSON.

Mecanismos empregados:
* **Contra underfitting:** transfer learning do MobileNetV3-Small pré-treinado no ImageNet, fornecendo representações densas de 576 dimensões desde a primeira época.
* **Contra overfitting:** backbone congelado (impede memorização de cenário), `Dropout(0.4)` nas cabeças, `weight_decay = 1e-4` no AdamW, early stopping por `val_loss` com paciência 5, e data augmentation por flip sorteado a cada época.

---

## 6. Matrizes de confusão e curvas ROC

![Matrizes de confusão](reports/matrizes_confusao_3_modelos.png)

* **Modelo 1:** 21 FN, 25 FP — acurácia 75.14%.
* **Modelo 2:** 10 FN (−52.4%), 23 FP — acurácia 82.16%.
* **Modelo 3 (melhor checkpoint):** 7 FN, 20 FP — acurácia 85.41%. Na média dos 20 comitês, 13.7 ± 2.0 FN.

![Curvas ROC](reports/curvas_roc_3_modelos.png)

AUC: baseline 0.863 → dual-stream 0.883 → ensemble 0.897. Esta é a métrica que mede o ganho real, porque não depende da escolha de limiar.

Regenere as figuras com:

```bash
python reports/generate_evolution_charts.py
```

---

## 7. Perfil de Edge AI

Latência medida por [`benchmarks/benchmark_detailed_latency.py`](benchmarks/benchmark_detailed_latency.py), que roda os três modelos **no mesmo laço, na mesma máquina, na mesma execução**, e registra o hardware junto com os números:

```bash
python benchmarks/benchmark_detailed_latency.py --runs 30
```

O script separa três custos que a tabela anterior deste README misturava:

| Componente | O que inclui |
| :--- | :--- |
| Decodificação + pré-proc | Leitura do `.avi`, amostragem de 16 quadros, resize, normalização |
| Forward | Backbone + cabeça sobre tensor já preparado |
| Total por clipe | Soma dos dois — o custo real em produção |

O resultado é gravado em `reports/latency_benchmark.json` com o perfil de CPU, contagem de threads e versão do PyTorch. **Números de latência sem esse contexto não são comparáveis**, e por isso este README não fixa valores: regenere na máquina alvo antes de citá-los.

Ordem de grandeza esperada em CPU de desktop: forward abaixo de 100 ms para 16 quadros 224×224, com o backbone dominando o custo e as cabeças recorrentes somando poucos milissegundos — é isso que torna o ensemble viável, já que as três compartilham a mesma passada do backbone.

### Exportação e verificação ONNX

A exportação carrega os **pesos treinados** antes de gerar o grafo:

```bash
python src/inference.py --export_onnx --onnx_model ensemble --no_infer
python benchmarks/verify_onnx_parity.py --model ensemble
```

`verify_onnx_parity.py` compara a saída do grafo ONNX com a do modelo PyTorch em várias entradas e mede a latência em ONNX Runtime. Sem essa verificação, "exportado para ONNX" não é uma afirmação de Edge AI — um grafo pode carregar e produzir valores errados.

Formatos abertos permitem aceleração via ONNX Runtime, Intel OpenVINO ou TensorRT.

---

## 8. Engenharia de limiares

A probabilidade de saída pode ser calibrada conforme a tolerância operacional a falsos negativos:

| Política | θ | Recall Fight | Precisão Fight | FN |
| :--- | :---: | :---: | :---: | :---: |
| Alta sensibilidade | 0.35 | 96.59% | 68.00% | 3 |
| Equilíbrio | 0.52 | 92.05% | 80.20% | 7 |
| Baixo falso positivo | 0.65 | 82.95% | 88.00% | 15 |

> **Ressalva:** esta tabela é uma varredura de limiares **sobre o conjunto de teste**. Ela ilustra o trade-off disponível, não fornece estimativas não enviesadas para nenhum dos três pontos. Para escolher um limiar operacional de forma metodologicamente correta:
>
> ```bash
> python src/calibrate_threshold.py --model ensemble \
>     --criterion recall_at_precision --min_precision 0.80 --eval_test
> ```
>
> O limiar é escolhido nos 215 vídeos de validação e só então aplicado ao teste, uma única vez.

---

## 9. Reprodução

### Instalação

```bash
git clone https://github.com/heittorvcs/surveillance-video-classifier.git
cd surveillance-video-classifier
pip install -r requirements.txt
```

### Inferência sobre um vídeo (não precisa do dataset)

```bash
python src/inference.py --video sample_video.avi --label Fight
```

Opções: `--model [ensemble|dualstream|baseline]`, `--threshold 0.52`, `--weights <caminho>`.

### Pipeline completo (precisa do RWF-2000 em `archive/RWF-2000/`)

```bash
# 1. Partições anti-leakage (verifica e falha se houver vazamento)
python src/create_splits.py

# 2. Cache de features do backbone congelado — pré-requisito de tudo abaixo.
#    Gera train, train_flip (augmentation), val e test.
python src/build_cache.py

# 3. Treinamento de qualquer arquitetura
python src/train.py --arch baseline    --seed 42
python src/train.py --arch dualstream  --seed 42
python src/train.py --arch tristream   --seed 7 --out models/ensemble/model_tristream_s7.pth

# 4. Avaliação
python src/evaluate.py --model ensemble
python src/evaluate.py --model dualstream --split val

# 5. Calibração de limiar na validação
python src/calibrate_threshold.py --model ensemble --eval_test

# 6. Benchmarks
python benchmarks/benchmark_3_modelos_20_runs.py     # 60 treinamentos
python benchmarks/run_20_ensembles_benchmark.py      # 20 comitês sem seleção
python benchmarks/benchmark_detailed_latency.py      # latência em CPU

# 7. Testes
python tests/test_splits.py
```

### Estrutura

```
src/
  create_splits.py        particoes anti-leakage + verificacao
  build_cache.py          extracao e cache das features (pre-requisito)
  dataset.py              amostragem de quadros, normalizacao, flip deterministico
  model.py                todas as arquiteturas + factory de carregamento
  train.py                treino de qualquer cabeca, com augmentation e checkpoints
  evaluate.py             metricas e matriz de confusao por split
  calibrate_threshold.py  escolha de limiar na validacao
  select_ensemble.py      escolha do trio na validacao
benchmarks/
  benchmark_3_modelos_20_runs.py   60 treinamentos, 3 arquiteturas
  run_20_ensembles_benchmark.py    20 comites sem selecao
  benchmark_detailed_latency.py    latencia sob protocolo unico
  verify_onnx_parity.py            paridade ONNX x PyTorch
reports/                 figuras, JSONs de metricas e scripts de plotagem
tests/                   testes das particoes anti-leakage
models/                  checkpoints e grafos ONNX
data/splits/             CSVs das particoes (versionados)
```

---

## 10. Limitações e trabalhos futuros

**Metodológicas**

* **Seleção de modelo contaminada pelo teste.** O trio de sementes e o limiar 0.52 do checkpoint entregue foram escolhidos observando o split de teste. O número honesto do método é 81.11% ± 1.21. `src/select_ensemble.py` e `src/calibrate_threshold.py` refazem a seleção sobre a validação; a correção depende de retreinar o pool de candidatos.
* **Análise de convergência incompleta.** As curvas versionadas são do Modelo 1. Curvas dos modelos 2 e 3 são geradas por `src/train.py`, mas não estão versionadas.
* **Vídeo de demonstração dentro do domínio.** `sample_video.avi` é um clipe de câmera fixa em formato RWF-2000, de câmera não vista em treino, mas não é uma captura externa. A predição obtida é `Fight` com margem de aproximadamente 10 pontos percentuais sobre o limiar — correta, mas não confortável.
* **Classes balanceadas.** O RWF-2000 é 1.000/1.000 por construção e o treino é 800/800; não há desbalanceamento a tratar. O `fight_weight = 1.35` não é correção estatística, e sim ponderação de **custo assimétrico** entre FN e FP, codificando na loss uma decisão de negócio.

**De domínio**

* **Iluminação extrema.** O RWF-2000 concentra cenas diurnas e iluminação pública regular. Escuridão severa ou infravermelho exigem dados complementares para adaptação de domínio.
* **Câmeras em movimento (PTZ).** O pipeline assume câmeras estáticas. Movimentos de pan/tilt/zoom geram fluxo no fundo e exigiriam compensação de movimento global.
* **Alta interação física legítima.** Contextos esportivos ou brincadeiras com contato contínuo tendem a gerar falsos positivos; recomenda-se limiar mais conservador (θ ≥ 0.60).
* **Entrada em clipes, não em stream.** O modelo consome clipes de 5 segundos. Operação contínua exigiria janela deslizante com voto temporal.

**Próximos passos, em ordem de prioridade**

1. Retreinar o pool e refazer seleção e calibração sobre a validação, publicando o teste uma única vez.
2. Versionar curvas de treinamento dos modelos 2 e 3.
3. Medir latência e paridade em ONNX Runtime no hardware alvo e publicar `reports/latency_benchmark.json`.
4. Quantização INT8 e avaliação em dispositivo de borda (Jetson Nano, Raspberry Pi 5).
5. Gravar um clipe externo de 5 segundos para a demonstração de inferência.

---

## Licença

[MIT](LICENSE).
