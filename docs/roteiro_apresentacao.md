# Roteiro do Vídeo de Apresentação — Surveillance Video Classifier

**Candidato:** Heittor Costa · **Destinatário:** NeuIA · **Duração alvo:** 9m00s (janela do edital: 5–10 min)

**Princípio do roteiro:** todo número dito em voz alta existe num arquivo do repositório, e nenhuma decisão de modelagem foi tomada com o conjunto de teste. O eixo da apresentação é a jornada das três arquiteturas e o rigor do protocolo que sustenta a comparação entre elas.

---

## Pré-gravação — checklist

**Primeiro de tudo: ativar o ambiente.** O Python do sistema não tem PyTorch instalado; sem ativar o `.venv`, todo comando ao vivo falha com `ModuleNotFoundError`.

```powershell
cd "C:\Users\heitt\OneDrive\Documentos\Entrevista tecnica NeuIA\surveillance-video-classifier"
.\.venv\Scripts\Activate.ps1
```

O prompt passa a mostrar `(.venv)` na frente. Confirme com `python -c "import torch; print(torch.__version__)"`.

- [ ] Ambiente ativado, com `(.venv)` visível no prompt
- [ ] `data/cache/features_{train,val,test}.pt` presentes — não são versionados, e `evaluate.py` depende deles
- [ ] `python tests/test_splits.py` passando — vai rodar na tela no minuto 1:00
- [ ] `python src/evaluate.py --model ensemble` rodando de verdade na máquina de gravação
- [ ] `python src/inference.py --video sample_video.avi --label Fight` testado
- [ ] `python reports/error_analysis.py` testado — use no minuto 6:00 se quiser mostrar ao vivo
- [ ] `sample_scvd_violence.avi` e `sample_scvd_normal.avi` testados com `src/inference.py`
- [ ] `reports/cross_dataset_scvd.png` aberto num visualizador — é o slide do minuto 7:00
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
> E o número em que eu mais confio é o AUC-ROC, porque ele não depende da escolha de limiar: 85,6% no baseline, 89,3% no dual-stream, 89,6% no ensemble. É esse salto que mostra que o viés indutivo cinético — velocidade e aceleração no espaço latente — realmente adiciona poder discriminativo, e não só desloca o ponto de operação.
>
> E eu fui olhar *onde* o modelo erra, porque contar 34 erros não diz o que fazer com eles. Dois achados. Primeiro: dos 90 grupos de câmera do teste, **71 não produzem erro nenhum** — sete câmeras concentram 65% dos erros. Excluindo as três piores, 23 clipes de 185, a acurácia sobe de 81,6% para 86,4%. O gargalo não é capacidade média, são cenários específicos. E cinco desses sete grupos têm clipes das duas classes na mesma câmera: mesmo fundo, mesma iluminação, só o movimento muda. É o caso mais difícil possível, e é exatamente o que o split por grupo força.
>
> Segundo: os falsos negativos ficam a 10 pontos da fronteira — são agressões sutis, recuperáveis por calibração. Já os falsos positivos ficam a 30 pontos: o modelo está **confiantemente** errado neles, com probabilidade acima de 90%. Isso é falha de representação, não de limiar — e a varredura confirma que nenhum limiar melhora a acurácia. Ou seja, o próximo ganho vem de dados desses cenários, não de ajuste fino."

*(Rodar `python src/evaluate.py --model ensemble` ao vivo.)*

---

## 7:00 – 8:00 · Validação externa: outro dataset, sem retreinar

**Tela:** `reports/cross_dataset_scvd.png` em tela cheia; depois a tabela de AUCs da seção 7.

> "Todos os números até aqui vêm do RWF-2000. Um teste mais duro é pegar o modelo pronto e aplicar, **sem retreinar**, num dataset independente.
>
> Usei o SCVD — Smart-City CCTV Violence Detection — 481 vídeos de CFTV urbano em 720p. Nenhum deles participou do treino, da validação ou da seleção. Como ele tem três classes e o meu classificador é binário, fiz duas versões: uma incluindo violência armada, outra só com violência corporal.
>
> O resultado bruto parece ruim: a acurácia cai de 81,6% para 66%. Mas olhem as métricas separadas — **o recall sobe** para 92%, e a precisão é que desaba. E o AUC, que não depende do limiar, cai só 1,9 ponto: de 89,6% para 87,7%.
>
> Isso não é perda de capacidade. É perda de calibração. O gráfico da esquerda mostra o mecanismo.
>
> [aponta] A classe positiva transfere quase perfeitamente: a mediana de probabilidade da violência do SCVD é 0,880, contra 0,879 no RWF-2000. Praticamente idêntica. Quem se desloca é a classe negativa: o normal sai de mediana 0,17 para 0,46, encostando no limiar.
>
> A explicação é o próprio viés que eu injetei no modelo. O SCVD é rua urbana com trânsito e circulação constante; o não-violento do RWF-2000 é gente caminhando. O modelo aprendeu a usar **intensidade de movimento** como sinal de agressão, e isso funciona dentro de um domínio mas confunde quando o domínio muda.
>
> E dá para provar que é calibração: recalibrando **só o limiar**, sem tocar em um peso sequer, a acurácia vai de 66% para **84,6%**. Em produção isso quer dizer que implantar num local novo precisa de algumas dezenas de clipes daquele local para reposicionar o limiar — não precisa retreinar.
>
> Dois achados a mais. Primeiro: o modelo **generaliza para violência armada**, detectando 84,7% dos casos contra 91,9% da violência corporal. Queda de só 7 pontos, e coerente com a arquitetura — apontar uma arma envolve menos movimento corporal que uma briga.
>
> Segundo, e para mim o mais importante: a ordenação das três arquiteturas **se repete** no dataset novo. Baseline 76%, dual-stream 81%, ensemble 84% de AUC. Mesma ordem do RWF-2000. Ou seja, o ganho do viés cinético não era artefato do dataset de treino — ele sobrevive à troca completa de domínio."

---

## 8:00 – 8:40 · Edge AI e demonstração

**Tela:** rodar `python src/inference.py --video sample_video.avi --label Fight`, depois a tabela da seção 7.

> "Sobre execução na borda, dois resultados.
>
> Primeiro: eu medi os três modelos intercalados, no mesmo laço, e usando mediana em vez de média — em CPU compartilhada uma pausa do escalonador desloca a média em dezenas de porcento, e medir em blocos separados penaliza um modelo sozinho. A decodificação do vídeo custa 43,1 milissegundos — **mais que o backbone inteiro**, que custa 38,3. O gargalo do pipeline não é a rede, é ler o arquivo. Isso muda a prioridade de otimização.
>
> Segundo: o ensemble custa 6,8 milissegundos a mais que o baseline, porque as três cabeças compartilham a mesma passada do backbone.
>
> E o ONNX. Exportei os três modelos e verifiquei paridade numérica contra o PyTorch — diferença máxima da ordem de 1e-6. Isso importa porque um grafo ONNX pode carregar sem erro e ainda assim produzir valores errados; sem a verificação, dizer 'exportado para ONNX' não significa nada. Com ONNX Runtime, o forward do ensemble cai de 44,3 para 14,8 milissegundos — três vezes mais rápido, e aí a decodificação passa a dominar de vez."

*(Rodar a inferência nos dois clipes do SCVD.)*

> "E a demonstração eu faço com os dois clipes do outro dataset, que é o teste mais honesto: o de violência sai com 98,9% de probabilidade, o normal com 4,3%. Os dois acertam com quase 50 pontos de margem.
>
> Deixo claro que esses dois não representam o dataset inteiro — sobre os 481 vídeos o desempenho é o que eu mostrei há pouco. São dois casos escolhidos para a demonstração, e eu prefiro dizer isso do que apresentar um acerto isolado como se fosse a média."

---

## 8:40 – 9:00 · Fechamento

> "Resumindo as decisões técnicas: split por grupo de câmera para que o número de teste signifique alguma coisa; backbone congelado por orçamento de borda e por risco de overfitting de cenário; derivadas cinéticas no espaço latente em vez de optical flow; e seleção de modelo feita na validação, com o teste avaliado uma única vez.
>
> As limitações que eu reconheço: o augmentation está implementado mas o ganho não foi medido; 215 vídeos de validação ainda são poucos para escolher entre 3.800 trios; e, a mais relevante, o modelo usa intensidade de movimento como proxy de agressão — foi a validação externa que expôs isso, e é a causa raiz dos falsos positivos confiantes.
>
> Os próximos passos, em ordem: treinar com múltiplos domínios para o modelo separar movimento de agressão, medir o augmentation, quantizar INT8 partindo dos grafos ONNX já verificados, e atacar a decodificação, que é o gargalo real de latência.
>
> Obrigado, e estou à disposição para as perguntas."

---

## Guia de defesa técnica

**"Como você garante que o 81,62% não é sorte de semente?"**
> "Três coisas. A seleção usou só os 215 vídeos de validação, então o teste permaneceu cego. As distribuições de 20 sementes por arquitetura estão registradas em JSON, e o ensemble tem o menor desvio dos três. E o AUC-ROC, que independe de limiar, confirma a mesma ordenação das arquiteturas."

**"O modelo funciona fora do dataset de treino?"**
> "Testei em 481 vídeos do SCVD, um dataset independente de CFTV urbano, sem retreinar. A resposta curta é: a capacidade transfere, a calibração não. O AUC cai só 1,9 ponto, mas o limiar ótimo muda de 0,50 para 0,82 — porque o normal daquele dataset tem muito mais movimento, e o modelo usa movimento como sinal. Recalibrando só o limiar, a acurácia vai de 66% para 84,6%. Na prática: implantar num local novo precisa de um conjunto de calibração local, não de retreino."

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
