# Roteiro do Vídeo de Apresentação — Surveillance Video Classifier

**Candidato:** Heittor Costa · **Destinatário:** NeuIA · **Duração alvo:** 8m00s (janela do edital: 5–10 min)

**Princípio do roteiro:** todo número dito em voz alta existe num arquivo do repositório e foi obtido com o conjunto de teste avaliado uma única vez. O eixo da apresentação é a jornada das três arquiteturas e o rigor do protocolo que sustenta a comparação entre elas.

---

## Pré-gravação — checklist

- [ ] `python src/build_cache.py` executado
- [ ] `python tests/test_splits.py` passando — vai rodar na tela no minuto 1:00
- [ ] `python src/evaluate.py --model ensemble` rodando de verdade na máquina de gravação
- [ ] `python src/inference.py --video sample_video.avi --label Fight` testado
- [ ] Abertos num visualizador: `reports/matrizes_confusao_3_modelos.png`, `reports/curvas_roc_3_modelos.png`, `reports/selected/training_curves_modelo3_tristream.png`
- [ ] Terminal em fonte 16+, 1080p, microfone testado sem eco
- [ ] Abas do editor: `README.md`, `src/create_splits.py`, `src/model.py`, `src/select_ensemble.py`

---

## 0:00 – 0:50 · Problema e critério de sucesso

**Tela:** README aberto no topo.

> "Olá, sou o Heittor. Apresento o Surveillance Video Classifier: um classificador de vídeo para detecção de agressão física em câmeras fixas de CFTV, projetado para rodar em CPU na borda.
>
> A decisão de engenharia que orienta o projeto é a assimetria de custo entre os dois tipos de erro. Um falso positivo custa alguns segundos de atenção de um vigilante. Um falso negativo é uma agressão que ninguém viu. Então eu não otimizei acurácia isoladamente: olhei recall, precisão e o trade-off entre eles o tempo todo.
>
> Vou mostrar o dataset e o cuidado com vazamento, três arquiteturas em ordem crescente de sofisticação, o protocolo estatístico que sustenta a comparação, e o perfil de execução na borda."

---

## 0:50 – 2:00 · Dataset e a armadilha do vazamento

**Tela:** `src/create_splits.py`, destacando o `GroupShuffleSplit`; depois rodar os testes.

> "O dataset é o RWF-2000, de Cheng e colegas, 2020: 2.000 clipes reais de câmeras de vigilância fixas, 5 segundos a 30 FPS, balanceado em 1.000 Fight e 1.000 NonFight. Fight cobre socos, chutes, empurrões e confrontos corporais; NonFight é monitoramento normal — caminhadas, conversas, aglomerações pacíficas, tráfego.
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

> "Cada clipe tem 150 quadros. Amostro 16 equidistantes, cobrindo os 5 segundos inteiros — isso corta 89% do volume de dados. A decodificação usa `grab()` para avançar o cursor e `retrieve()` só nos 16 instantes que interessam, evitando decodificar os 134 quadros descartados. Depois: 224 por 224, BGR para RGB e normalização com as estatísticas do ImageNet, porque o backbone é pré-treinado nelas.
>
> E um detalhe de augmentation que vale explicar, porque é uma consequência não óbvia do backbone congelado. Como os pesos do extrator não mudam, eu cacheio as features para acelerar o treino em duas ordens de grandeza. Só que, se eu aplicasse o flip horizontal nessa extração única, cada vídeo receberia uma perturbação **fixa**, idêntica em todas as épocas. Isso não é augmentation, é ruído congelado.
>
> A solução é materializar dois caches — o original e o espelhado — e sortear entre eles a cada época. Custa o dobro de disco no cache e zero de compute no treino.
>
> Sendo transparente: os números que vou mostrar foram treinados sem o cache espelhado, porque eu não tinha o dataset bruto no ambiente de retreinamento. O mecanismo está implementado e testado, mas o ganho ainda não foi medido — e está listado assim nas limitações."

---

## 3:00 – 4:30 · As três arquiteturas

**Tela:** `src/model.py`, percorrendo as três classes.

> "O **Modelo 1** é o baseline: MobileNetV3-Small pré-treinado no ImageNet e congelado como extrator espacial de 576 dimensões, seguido de uma Bi-GRU. 1,17 milhão de parâmetros.
>
> Congelei o backbone de propósito, por dois motivos. Com câmeras fixas, descongelar faz os filtros se adaptarem à textura do fundo — paredes, calçadas — em vez do movimento dos atores. E congelar permite cachear features e treinar 80 modelos em minutos, o que viabiliza toda a análise estatística que eu mostro daqui a pouco.
>
> A limitação do baseline é estrutural: ele só vê aparência estática. Duas pessoas gesticulando com energia têm embedding parecido com duas pessoas brigando. Falta a derivada.
>
> O **Modelo 2** resolve isso sem optical flow, que custaria centenas de milissegundos por quadro e mataria a proposta de borda. Eu calculo a velocidade no espaço latente: delta f igual a f_t menos f_{t-1}, sobre os vetores de 576 dimensões que já tenho. Duas Bi-GRUs em paralelo, aparência e velocidade.
>
> O efeito é nítido no recall: sobe de 71,6% para 90,9%, e os falsos negativos caem de 25 para 8. Mas o preço são 32 falsos positivos — ele fica sensível demais.
>
> O **Modelo 3** adiciona a segunda derivada, aceleração, porque o que caracteriza um impacto não é a velocidade, é a mudança brusca dela. E combina três cabeças em comitê, todas partilhando o mesmo backbone: o MobileNet roda uma vez por vídeo, e as três cabeças somadas custam 6 milissegundos. É o que torna um ensemble viável na borda."

---

## 4:30 – 6:00 · O protocolo estatístico

**Tela:** seção 5 do README, com a tabela das distribuições na validação; depois `src/select_ensemble.py`.

> "Aqui está a parte que eu considero o núcleo técnico da entrega.
>
> Com 185 vídeos de teste, a variação entre sementes é da mesma ordem da diferença entre arquiteturas. Se eu treinar uma vez cada modelo e comparar os três números, eu não estou comparando arquiteturas — estou comparando sorteios.
>
> Então o protocolo é este. Primeiro, um pool de candidatos: 20 sementes independentes para cada arquitetura, treinadas do zero sob condições idênticas — mesmo early stopping, mesma loss ponderada, mesmo otimizador. Nessa etapa o conjunto de teste não é consultado em momento nenhum.
>
> Segundo, a seleção acontece **na validação**, nos 215 vídeos: o melhor candidato de cada arquitetura, e o melhor trio no caso do ensemble — são 3.800 combinações possíveis.
>
> Terceiro, e só então, o teste é avaliado **uma única vez**, com o que saiu da validação.
>
> [aponta para a tabela] Olhem as distribuições na validação: o baseline fica em 73,3% com desvio de 1,6; o dual-stream em 74,4% com 1,9; e o ensemble em 76,8% com desvio de **1,09**. Média mais alta e desvio menor — é exatamente o efeito que se espera de um comitê. Ele não só acerta mais, ele varia menos entre sementes.
>
> Duas decisões finas dentro do protocolo, e as duas foram tomadas olhando só a validação.
>
> A primeira: o limiar fica **fixo** em 0,50 durante a seleção. Escolher semente e limiar ao mesmo tempo em 215 vídeos superajusta a validação — numa varredura de 91 limiares, o dual-stream elegeu θ igual a 0,15, e isso rendeu 67% de precisão no teste. Com o limiar fixo, a seleção mede a arquitetura, e o ponto de operação vira uma decisão separada.
>
> A segunda: o critério é F1, não recall sob restrição de precisão. A restrição de precisão 0,80 é estruturalmente inviável para o baseline — na validação ele só a atinge a partir de θ 0,71, onde o recall desaba para 56%. Um critério que uma das arquiteturas não consegue satisfazer transforma a comparação em outra coisa."

---

## 6:00 – 7:00 · Resultados

**Tela:** tabela principal do README, matrizes de confusão, curvas ROC. Rodar `python src/evaluate.py --model ensemble`.

> "Os resultados no teste cego, com os três sob o mesmo protocolo:
>
> Baseline: 76,22% de acurácia, recall 71,6%, 25 falsos negativos.
> Dual-Stream: 78,38%, recall 90,9%, apenas 8 falsos negativos — mas 32 falsos positivos.
> Ensemble: **81,62%**, recall 85,2%, F1 de 81,5%, 13 falsos negativos e 21 falsos positivos. É o melhor equilíbrio dos três.
>
> Vale reparar que o Modelo 2 tem o menor número de falsos negativos, e num cenário de tolerância zero a agressão perdida ele seria uma escolha defensável. O Modelo 3 troca cinco falsos negativos por onze falsos positivos a menos.
>
> E o número em que eu mais confio é o AUC-ROC, porque ele não depende da escolha de limiar: 85,6% no baseline, 89,3% no dual-stream, 89,6% no ensemble. É esse salto que mostra que o viés indutivo cinético — velocidade e aceleração no espaço latente — realmente adiciona poder discriminativo, e não só desloca o ponto de operação."

*(Rodar `python src/evaluate.py --model ensemble` ao vivo.)*

---

## 7:00 – 7:40 · Edge AI e demonstração

**Tela:** rodar `python src/inference.py --video sample_video.avi --label Fight`, depois a tabela da seção 7.

> "Sobre execução na borda, dois resultados.
>
> Primeiro: eu medi os três modelos no mesmo laço e na mesma execução, separando decodificação de forward. A decodificação do vídeo custa 43,9 milissegundos — **mais que o backbone inteiro**, que custa 34. O gargalo do pipeline não é a rede, é ler o arquivo. Isso muda a prioridade de otimização: mexer só no modelo tem retorno limitado.
>
> Segundo: o ensemble custa apenas 5 milissegundos a mais que o baseline, porque as três cabeças compartilham a mesma passada do backbone.
>
> E o ONNX. Exportei os três modelos e verifiquei paridade numérica contra o PyTorch — diferença máxima da ordem de 1e-6. Isso importa porque um grafo ONNX pode carregar sem erro e ainda assim produzir valores errados; sem a verificação, dizer 'exportado para ONNX' não significa nada. Com ONNX Runtime, o forward do ensemble cai de 44,9 para 16,9 milissegundos — 2,7 vezes mais rápido, e aí a decodificação passa a dominar de vez."

*(Rodar a inferência.)*

> "Fight com 69,9% de probabilidade, quase 20 pontos acima do limiar. Acertou, com margem confortável."

---

## 7:40 – 8:00 · Fechamento

> "Resumindo as decisões técnicas: split por grupo de câmera para que o número de teste signifique alguma coisa; backbone congelado por orçamento de borda e por risco de overfitting de cenário; derivadas cinéticas no espaço latente em vez de optical flow; e seleção de modelo feita na validação, com o teste avaliado uma única vez.
>
> As limitações que eu reconheço: o augmentation está implementado mas o ganho não foi medido; 215 vídeos de validação ainda são poucos para escolher entre 3.800 trios, e um GroupKFold sobre treino mais validação daria uma estimativa mais estável; e o dataset é restrito a cenas diurnas e câmeras estáticas, então PTZ e visão noturna exigiriam dados de domínio.
>
> Os próximos passos, em ordem: medir o augmentation, trocar o split único por GroupKFold, quantizar INT8 partindo dos grafos ONNX já verificados, e atacar a decodificação, que é o gargalo real.
>
> Obrigado, e estou à disposição para as perguntas."

---

## Guia de defesa técnica

**"Como você garante que o 81,62% não é sorte de semente?"**
> "Três coisas. A seleção usou só os 215 vídeos de validação, então o teste permaneceu cego. As distribuições de 20 sementes por arquitetura estão registradas em JSON, e o ensemble tem o menor desvio dos três. E o AUC-ROC, que independe de limiar, confirma a mesma ordenação das arquiteturas."

**"Por que não um Video Transformer, tipo VideoMAE ou TimeSformer?"**
> "Restrição de dados e de borda. 2.000 vídeos é pouco para atenção espaço-temporal treinada do zero, e o requisito era CPU. Eu não benchmarkei um Transformer nesta entrega, então não vou afirmar um resultado que não medi — o que posso afirmar é que a Bi-GRU sobre features congeladas cabe no orçamento de latência e que o ganho vem do viés cinético, que está isolado na comparação entre as três arquiteturas."

**"Por que congelar o backbone em vez de fazer fine-tuning?"**
> "Custo e overfitting de cenário. Com câmeras fixas, o fine-tuning faz os filtros se especializarem no fundo em vez do movimento. Além disso, congelar permite cachear as features e treinar 80 modelos em minutos — foi o que viabilizou o protocolo estatístico."

**"O dataset é desbalanceado?"**
> "Não — é 1.000/1.000 por construção, e o treino é 800/800. O `fight_weight = 1.35` não é tratamento de desbalanceamento, é ponderação de custo assimétrico entre falso negativo e falso positivo: uma decisão de negócio codificada na função de perda."

**"O vídeo de demonstração é realmente novo?"**
> "É um clipe de câmera fixa no formato do RWF-2000, de uma câmera que o modelo nunca viu em treino. Não é uma captura externa feita por mim — se o critério for esse, é uma limitação da demonstração, e a forma de fechar é gravar um clipe próprio de 5 segundos e rodar o mesmo comando."

**"Como isso vai para produção?"**
> "Três frentes. Calibrar o limiar por política de cliente — o `calibrate_threshold.py` faz isso na validação, com critério de recall sob precisão mínima. Quantizar INT8 partindo dos grafos ONNX já verificados. E janela deslizante com voto temporal, já que hoje o modelo consome clipes de 5 segundos e não stream contínuo."

---

## Erros a evitar na gravação

1. **Não** dizer "SOTA" — o RWF-2000 tem trabalhos publicados acima deste patamar. A escolha de backbone congelado por orçamento de borda é defensável; o rótulo não é.
2. **Não** citar números que não estejam num arquivo do repositório — sem resultados de Transformer, sem estatísticas de fadiga de operador, sem correlação entre quadros.
3. **Não** apresentar as curvas de treinamento sem dizer de qual modelo e semente elas são.
4. **Não** rodar comando ao vivo que não foi testado na máquina de gravação minutos antes.
5. **Não** esconder o trade-off do Modelo 2 (menos falsos negativos, mais falsos positivos). Mostrar que você entende a troca é mais forte do que apresentar só o vencedor.
