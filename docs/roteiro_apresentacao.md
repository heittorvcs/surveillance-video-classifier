# Roteiro do Vídeo de Apresentação

**Candidato:** Heittor Costa · **Destinatário:** NeuIA · **Duração alvo:** 9min39s (janela do edital: 5–10)

**Princípio:** todo número dito em voz alta existe num arquivo do repositório, e nenhuma decisão de modelagem foi tomada com o conjunto de teste.

> **Ritmo:** 1.212 palavras faladas. A 140 palavras/min, somando o tempo dos comandos rodando, dá **9min39s** — os tempos de cada bloco saem dessa conta.
>
> A margem para o limite de 10 min é de 21 segundos, o que é pouco. **Cronometre um ensaio.** Se passar de 9min45s, corte os três trechos marcados com `[CORTÁVEL]`: somam 30 s e levam a apresentação para ~9min10s.

---

## Pré-gravação

**Ative o ambiente antes de tudo** — o Python do sistema não tem PyTorch, e todo comando ao vivo falharia:

```powershell
cd "C:\Users\heitt\OneDrive\Documentos\Entrevista tecnica NeuIA\surveillance-video-classifier"
.\.venv\Scripts\Activate.ps1
```

- [ ] Prompt mostrando `(.venv)`
- [ ] `data/cache/features_{train,val,test}.pt` presentes (não são versionados)
- [ ] Testados: `tests/test_splits.py`, `src/evaluate.py --model ensemble`, `src/inference.py` nos clipes
- [ ] Abertos num visualizador: `reports/matrizes_confusao_3_modelos.png` e `reports/cross_dataset_scvd.png`
- [ ] Abas do editor: `README.md`, `src/create_splits.py`, `src/model.py`
- [ ] Terminal em fonte 16+, 1080p, microfone sem eco

---

## 0:00 – 0:31 · Problema

**Tela:** README no topo.

> "Sou o Heittor. Apresento um classificador de vídeo para detecção de agressão física em câmeras fixas de CFTV, rodando em CPU na borda.
>
> A decisão que orienta o projeto inteiro é a assimetria entre os dois erros. Um falso positivo custa alguns segundos de atenção de um vigilante. Um falso negativo é uma agressão que ninguém viu. Então eu nunca olhei acurácia sozinha — olhei recall, precisão e o trade-off entre eles."

---

## 0:31 – 1:39 · Dataset e vazamento

**Tela:** `src/create_splits.py`, destacando o `GroupShuffleSplit`.

> "O dataset é o RWF-2000: 2.000 clipes reais de câmeras de vigilância, 5 segundos cada, balanceado entre briga e não-briga.
>
> A armadilha desse dataset é que vários clipes vêm de cortes do mesmo vídeo original — mesma câmera, mesmo fundo. Num split aleatório, o modelo decora o cenário e você publica um número que não sobrevive ao primeiro cliente.
>
> Então particionei por grupo: extraio o identificador do vídeo original do nome do arquivo e garanto que nenhum grupo atravesse a fronteira. Mil e seiscentos vídeos de treino, 215 de validação, 185 de teste cego. E isso não é uma afirmação, é um teste."

*(Rodar `python tests/test_splits.py` — cinco testes, cinco segundos.)*

> "Zero grupos e zero arquivos em comum. As 90 câmeras do teste são inéditas para o modelo."

---

## 1:39 – 2:34 · Pré-processamento

**Tela:** `src/dataset.py`.

> "De cada clipe amostro 16 quadros equidistantes cobrindo os 5 segundos — corta 89% do volume sem perder a trajetória do movimento. A decodificação usa `grab` para pular quadro sem decodificar. Depois: 224 por 224 e normalização ImageNet, as estatísticas do pré-treino do backbone.
>
> E um detalhe de augmentation, porque foi uma armadilha que eu mesmo criei. Como o backbone é congelado, eu cacheio as features. Se aplicasse o flip nessa extração única, cada vídeo teria uma perturbação **fixa**, igual em todas as épocas — não é augmentation, é ruído congelado. A correção é ter dois caches, original e espelhado, e sortear a cada época.
>
> `[CORTÁVEL]` Sendo transparente: os números aqui foram treinados sem o cache espelhado. O mecanismo está implementado; o ganho não foi medido."

---

## 2:34 – 3:47 · As três arquiteturas

**Tela:** `src/model.py`.

> "O **Modelo 1** é o baseline: MobileNetV3-Small do ImageNet, congelado, mais uma Bi-GRU. Congelei de propósito — com câmera fixa, o fine-tuning especializa os filtros na textura do fundo em vez do movimento. E é o que permite cachear features e treinar 80 modelos em minutos.
>
> A limitação dele é estrutural: só vê aparência estática. Gesticulação vigorosa tem embedding parecido com briga. Falta a derivada.
>
> O **Modelo 2** resolve sem optical flow, que custaria centenas de milissegundos por quadro: calculo a velocidade no espaço latente, f de t menos f de t menos um, sobre vetores que já tenho. Duas Bi-GRUs paralelas. O recall sobe de 71 para 91 por cento — mas o preço são 32 falsos positivos.
>
> O **Modelo 3** adiciona a aceleração, porque o que caracteriza um impacto não é a velocidade, é a mudança brusca dela. E combina três cabeças em comitê sobre a mesma passada do backbone: o MobileNet roda uma vez por vídeo, e as cabeças somam 7 milissegundos."

---

## 3:47 – 5:00 · Estratégia de treinamento

**Tela:** seção 4 do README, tabela das distribuições na validação.

> "Aqui está o núcleo técnico da entrega.
>
> Com 185 vídeos de teste, a variação entre sementes é da mesma ordem da diferença entre arquiteturas. Treinar cada modelo uma vez e comparar os três números não compara arquiteturas — compara sorteios.
>
> Então: um pool de 20 sementes por arquitetura, sem consultar o teste. A seleção acontece **na validação**, nos 215 vídeos — melhor candidato de cada arquitetura, e melhor trio entre 3.800 combinações no ensemble. E só então o teste é avaliado, uma vez.
>
> [aponta] Na validação, o baseline fica em 73,3%, o dual-stream em 74,4% e o ensemble em 76,8% — com o **menor desvio** dos três. Média mais alta e menos variância, que é o efeito esperado de um comitê.
>
> Uma decisão fina: o limiar fica **fixo** durante a seleção. Escolher semente e limiar juntos em 215 vídeos superajusta a validação — numa varredura, o dual-stream elegeu 0,15, que rendeu 67% de precisão no teste. Com o limiar fixo, a seleção mede a arquitetura."

---

## 5:00 – 6:09 · Métricas e onde o modelo erra

**Tela:** tabela de resultados e matrizes de confusão. Rodar `python src/evaluate.py --model ensemble`.

> "No teste cego: baseline 76,2%; dual-stream 78,4%, com recall de 91% mas 32 alarmes falsos; ensemble **81,6%**, recall 85%, F1 de 81,5% — o melhor equilíbrio. E o número em que mais confio é o AUC, que não depende do limiar: 85,6, 89,3 e 89,6.
>
> E fui olhar **onde** ele erra. Dos 90 grupos de câmera, **71 não erram nada** — sete concentram 65% dos erros. Excluindo as três piores, a acurácia vai a 86,4%: o gargalo são cenários específicos, não capacidade média.
>
> E os dois erros são diferentes. Falso negativo fica a 10 pontos da fronteira, recuperável por calibração. Falso positivo fica a 30, com probabilidade acima de 90% — o modelo está **confiantemente** errado. Isso é falha de representação, e nenhum limiar resolve."

---

## 6:09 – 7:39 · Validação externa

**Tela:** `reports/cross_dataset_scvd.png` em tela cheia.

> "Todos esses números vêm do RWF-2000. Um teste mais duro é aplicar o modelo pronto, **sem retreinar**, num dataset independente: o SCVD, 481 vídeos de CFTV urbano em 720p, nenhum usado em treino, validação ou seleção. Como ele tem três classes, fiz duas versões — uma com violência armada, outra só corporal.
>
> O resultado bruto parece ruim: a acurácia cai para 66%. Mas o **recall sobe** para 92%, é a precisão que desaba. E o AUC cai só 1,9 ponto. Não é perda de capacidade, é perda de calibração.
>
> [aponta] A classe positiva transfere quase perfeitamente: mediana 0,880 contra 0,879 no RWF. Quem desloca é a negativa, de 0,17 para 0,46, encostando no limiar. O SCVD é rua urbana com tráfego; o não-violento do RWF é gente caminhando. O modelo usa **intensidade de movimento** como sinal de agressão.
>
> E dá para provar: recalibrando só o limiar, a acurácia vai a **84,6%**. Implantar num local novo precisa de calibração local, não de retreino.
>
> E dois achados a mais: o modelo **generaliza para violência armada**, com 84,7% de detecção contra 91,9% da corporal; `[CORTÁVEL]` e a ordenação das três arquiteturas **se repete** aqui, então o ganho do viés cinético não era artefato do dataset de treino."

---

## 7:39 – 9:05 · Edge AI e demonstração

**Tela:** tabela da seção 7; depois rodar a inferência nos dois clipes do SCVD.

> "`[CORTÁVEL]` Medi os três modelos intercalados no mesmo laço e por mediana, porque em CPU compartilhada medir em blocos separados chega a inverter a ordem.
>
> A decodificação custa 43 milissegundos, **mais que o backbone inteiro**: o gargalo é o I/O, não a rede. E o ensemble custa só 6,8 milissegundos a mais que o baseline, porque as cabeças compartilham a mesma passada. Com ONNX Runtime — exportado com paridade numérica verificada — o forward cai de 44 para 15 milissegundos."

*(Rodar a inferência nos dois clipes do SCVD.)*

> "A demonstração eu faço com os dois clipes do outro dataset: o de violência sai com 98,9%, o normal com 4,3%, os dois com quase 50 pontos de margem. E são dois casos, não a média — sobre os 481 vídeos o desempenho é o que mostrei há pouco."

---

## 9:05 – 9:39 · Fechamento

> "Resumindo: split por grupo de câmera para o número de teste significar algo; backbone congelado por orçamento de borda; derivadas cinéticas em vez de optical flow; e seleção feita na validação.
>
> As limitações que reconheço: o modelo usa intensidade de movimento como proxy de agressão — foi a validação externa que expôs isso; e o augmentation está implementado mas não medido.
>
> Próximos passos: treinar com múltiplos domínios, medir o augmentation, quantizar INT8, e atacar a decodificação. Obrigado."

---

## Guia de defesa técnica

**"Como você garante que o 81,62% não é sorte de semente?"**
> "A seleção usou só os 215 vídeos de validação, então o teste ficou cego. As distribuições de 20 sementes por arquitetura estão em JSON no repositório, e o ensemble tem o menor desvio dos três. E o AUC, que independe de limiar, confirma a mesma ordenação."

**"O modelo funciona fora do dataset de treino?"**
> "Testei em 481 vídeos do SCVD sem retreinar. A capacidade transfere, a calibração não: o AUC cai 1,9 ponto, mas o limiar ótimo vai de 0,50 para 0,82, porque o normal daquele dataset tem muito mais movimento. Recalibrando só o limiar, a acurácia vai de 66 para 84,6%."

**"Por que não um Video Transformer?"**
> "Restrição de dados e de borda: 2.000 vídeos é pouco para atenção espaço-temporal treinada do zero, e o requisito era CPU. Não benchmarkei um Transformer aqui, então não vou afirmar um resultado que não medi."

**"Por que congelar o backbone?"**
> "Custo e overfitting de cenário. Com câmera fixa, o fine-tuning especializa os filtros no fundo. E congelar permite cachear features e treinar 80 modelos em minutos — foi o que viabilizou o protocolo estatístico."

**"O dataset é desbalanceado?"**
> "Não, é mil e mil por construção. O `fight_weight` de 1,35 não é tratamento de desbalanceamento, é ponderação de custo assimétrico entre falso negativo e falso positivo — decisão de negócio codificada na loss."

**"Como isso vai para produção?"**
> "Calibrar o limiar por política de cliente, com o `calibrate_threshold.py` sobre um conjunto local. Quantizar INT8 partindo dos grafos ONNX já verificados. E janela deslizante com voto temporal, já que hoje o modelo consome clipes de 5 segundos e não stream."

---

## Erros a evitar

1. **Não** dizer "SOTA" — o RWF-2000 tem trabalhos publicados acima deste patamar. Backbone congelado por orçamento de borda é defensável; o rótulo não é.
2. **Não** citar número que não esteja num arquivo do repositório.
3. **Não** mostrar curva de treinamento sem dizer de qual modelo e semente.
4. **Não** rodar comando ao vivo que não foi testado minutos antes.
5. **Não** esconder o trade-off do Modelo 2 — menos falsos negativos, mais falsos positivos. Mostrar que entende a troca é mais forte que apresentar só o vencedor.
