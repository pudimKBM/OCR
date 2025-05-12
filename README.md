# Reconhecimento Avançado de Placas Veiculares com OpenCV e Tesseract OCR

Este projeto implementa um sistema de Reconhecimento Óptico de Caracteres (OCR) para placas veiculares utilizando Python, OpenCV para processamento de imagem e Tesseract OCR para extração de texto. O script `main.py` oferece múltiplos modos de detecção e processamento para lidar com diversas qualidades de imagem e cenários.

## Funcionalidades Principais

- **Múltiplos Modos de Detecção**:
    1.  **`crop` (MSER com Auto-Ajuste)**: Utiliza a técnica MSER (Maximally Stable Extremal Regions) para identificar regiões candidatas a caracteres, agrupa-os para formar placas e aplica OCR nos recortes. Tenta automaticamente 3 configurações de parâmetros MSER diferentes para otimizar a detecção.
    2.  **`nocrop` (OCR na Imagem Inteira)**: Aplica pré-processamento global na imagem e tenta realizar o OCR diretamente na imagem inteira, sem um recorte específico da placa.
    3.  **`sliding_window` (Janela Deslizante com Multiprocessamento)**: Varre a imagem com janelas de diferentes tamanhos e proporções, aplicando OCR em cada "patch". Utiliza multiprocessamento para acelerar a varredura.
    4.  **`all`**: Executa todos os três modos acima sequencialmente para cada imagem.
- **Pré-processamento de Imagem**:
    - Conversão para escala de cinza.
    - Aplicação de filtro Bilateral para redução de ruído preservando bordas.
    - Binarização adaptativa.
    - Melhoria de contraste com CLAHE (Contrast Limited Adaptive Histogram Equalization), especialmente no modo MSER.
- **Pós-processamento de OCR**:
    - Validação de formatos de placa (padrão antigo e Mercosul) usando expressões regulares.
    - Tentativas de correção de caracteres com base em um dicionário de substituições comuns (e.g., 'O' por '0', 'I' por '1').
    - Geração de múltiplas possibilidades de placas a partir de leituras ambíguas.
- **Saída Detalhada**:
    - Salva imagens intermediárias do processo de detecção e OCR.
    - Gera um gráfico comparativo (usando Matplotlib) mostrando a imagem original, a imagem processada para detecção, o recorte da placa (se aplicável) e o recorte binarizado para OCR.
    - Imprime o texto da placa extraída no console.
- **Configurabilidade**:
    - Caminho para o executável do Tesseract.
    - Pasta de entrada para as imagens.
    - Seleção do modo de processamento via argumentos de linha de comando.

## Estrutura do Projeto (Sugerida)

```
OCR/
├── main.py                     # O script principal do projeto
├── images/                     # Pasta para colocar as imagens de entrada
│   ├── carro1.jpg
│   └── ...
├── output_images_mser_crop_tuned/ # Saída do modo 'crop'
│   └── carro1/
│       ├── success_config1_stricter/
│       │    ├── carro1_00_fig_mser_crop.png
│       │    ├── carro1_01_original.png
│       │    └── ...
│       └── tuning_final_attempt/ (se nenhuma config teve sucesso)
├── output_images_no_crop_v2/    # Saída do modo 'nocrop'
│   └── carro1/
│       ├── carro1_00_fig_nocrop.png
│       └── ...
├── output_images_sliding_window_mp_v2/ # Saída do modo 'sliding_window'
│   └── carro1/
│       ├── carro1_00_fig_sliding_window.png
│       └── ...
└── README.md                   # Este arquivo
```

## Dependências

- Python 3.x
- OpenCV (`opencv-python`)
- Pytesseract (`pytesseract`)
- NumPy (`numpy`)
- Matplotlib (`matplotlib`)
- Tesseract OCR (instalado no sistema)

## Instalação

1.  **Instalar Tesseract OCR**:
    -   **Windows**: Baixe o instalador em Tesseract at UB Mannheim. Durante a instalação, adicione os pacotes de idioma desejados (ex: `por` para Português, `eng` para Inglês). **Importante**: Adicione o diretório de instalação do Tesseract (e.g., `C:\Program Files\Tesseract-OCR`) à variável de ambiente PATH.
    -   **Linux (Debian/Ubuntu)**: `sudo apt update && sudo apt install tesseract-ocr tesseract-ocr-por tesseract-ocr-eng`
    -   **macOS (Homebrew)**: `brew install tesseract tesseract-lang`
    Verifique a instalação com `tesseract --version`.

2.  **Configurar Caminho do Tesseract no Script**:
    No arquivo `main.py`, ajuste a linha se necessário:
    ```python
    pytesseract.pytesseract.tesseract_cmd = r'C:/Program Files/Tesseract-OCR/tesseract.exe'
    ```

3.  **Instalar Bibliotecas Python**:
    Recomenda-se o uso de um ambiente virtual.
    ```bash
    python -m venv venv
    # Windows: venv\Scripts\activate
    # Linux/macOS: source venv/bin/activate
    pip install opencv-python pytesseract numpy matplotlib
    ```

## Como Executar

1.  Crie uma pasta chamada `images` no mesmo diretório do `main.py` e coloque suas imagens de placas nela.
2.  Execute o script a partir do terminal, especificando o modo desejado:

    ```bash
    python c:\Users\anton\Documents\GitHub\OCR\main.py --mode [MODO] --images_folder [PASTA_IMAGENS]
    ```

    **Argumentos**:
    -   `--mode`: Especifica o modo de processamento. Opções:
        -   `crop` (padrão): Detecção MSER com 3 configurações de ajuste.
        -   `nocrop`: OCR na imagem inteira.
        -   `sliding_window`: OCR com janela deslizante e multiprocessamento.
        -   `all`: Executa todos os modos acima.
    -   `--images_folder`: Nome da pasta contendo as imagens de entrada (padrão: `images`).

    **Exemplos**:
    ```bash
    # Executar modo 'crop' (padrão) na pasta 'images' (padrão)
    python c:\Users\anton\Documents\GitHub\OCR\main.py

    # Executar modo 'sliding_window'
    python c:\Users\anton\Documents\GitHub\OCR\main.py --mode sliding_window

    # Executar todos os modos em imagens na pasta 'minhas_placas'
    python c:\Users\anton\Documents\GitHub\OCR\main.py --mode all --images_folder minhas_placas
    ```

3.  Os resultados (texto extraído, imagens processadas e gráficos) serão salvos em subpastas dentro de `output_images_mser_crop_tuned/`, `output_images_no_crop_v2/`, ou `output_images_sliding_window_mp_v2/`, dependendo do modo.

## Detalhamento dos Modos de Processamento

### 1. Modo `crop` (Detecção MSER com Auto-Ajuste)
   - **Orquestrador**: `detectar_placa_com_mser_crop_tuned`
   - **Lógica**:
     1.  A imagem original é carregada e convertida para escala de cinza (`processar_imagem_globalmente`).
     2.  O script itera sobre uma lista pré-definida de 3 configurações de parâmetros (`parameter_configurations`) para o MSER e funções auxiliares.
     3.  Para cada configuração:
         - `encontrar_placas_via_mser_param`:
           - Aplica CLAHE para realce de contraste na imagem em cinza.
           - Calcula dinamicamente os tamanhos mínimo e máximo de caracteres e áreas MSER com base nas dimensões da imagem e nos parâmetros da configuração atual.
           - Configura e executa o `cv2.MSER_create()`.
           - `filter_char_candidates_param`: Filtra as regiões MSER detectadas, mantendo aquelas que se assemelham a caracteres (baseado em altura e aspect ratio definidos na configuração).
           - `group_char_candidates_param`: Agrupa os caracteres candidatos em linhas horizontais que poderiam formar uma placa (baseado em alinhamento vertical, similaridade de altura, espaçamento e número de caracteres, conforme a configuração).
           - Recorta as regiões de placa candidatas da imagem original colorida e da imagem em cinza (após CLAHE e binarização adaptativa), aplicando um padding.
     4.  `aplicar_ocr_aos_recortes`:
         - Para cada recorte de placa candidato:
           - Redimensiona o recorte se a altura for muito pequena, visando uma altura de caractere ideal para OCR.
           - Tenta o OCR com Tesseract usando os idiomas Português (`por`) e Inglês (`eng`), com `psm 7` (linha única de texto).
           - Valida o texto OCRizado contra padrões de placa (antigo e Mercosul).
           - Se não houver correspondência direta, tenta correções de caracteres e gera novas possibilidades.
         - Retorna o primeiro resultado válido encontrado, ou o melhor palpite.
     5.  Se uma placa válida é encontrada com uma configuração, o processo de ajuste para aquela imagem para, e o resultado é salvo.
     6.  Se nenhuma das 3 configurações produzir uma placa válida, o resultado da última tentativa (ou uma mensagem de falha) é salvo.
   - **Saída**: Imagens originais, processadas, recortes de placa e gráfico comparativo são salvos em `output_images_mser_crop_tuned/[nome_imagem]/[success_nome_config_OU_tuning_final_attempt]/`.

### 2. Modo `nocrop` (OCR na Imagem Inteira)
   - **Orquestrador**: `detectar_placa_sem_recorte`
   - **Lógica**:
     1.  `processar_imagem_globalmente`: A imagem é convertida para escala de cinza, suavizada com filtro bilateral e binarizada com limiarização adaptativa.
     2.  `aplicar_ocr_na_imagem_inteira`: O Tesseract OCR é aplicado diretamente na imagem binarizada global.
         - Tenta com idiomas Português e Inglês, usando `psm 6` (assume um bloco uniforme de texto).
         - Valida o resultado contra padrões de placa.
   - **Saída**: Imagem original, imagem processada para OCR e gráfico são salvos em `output_images_no_crop_v2/[nome_imagem]/`.

### 3. Modo `sliding_window` (Janela Deslizante com Multiprocessamento)
   - **Orquestrador**: `detectar_placa_via_janela_deslizante`
   - **Lógica**:
     1.  `encontrar_melhor_resultado_via_janela_deslizante`:
         - Define uma série de configurações de janelas (largura, altura, passos de deslize).
         - Para cada configuração, gera "patches" (recortes) da imagem em escala de cinza.
         - Cria uma lista de tarefas, onde cada tarefa é processar um patch.
     2.  `multiprocessing.Pool`: Distribui as tarefas de processamento de patches entre múltiplos processos da CPU.
     3.  `process_patch_task` (executado por cada processo filho):
         - Para cada patch:
           - Binariza com `cv2.THRESH_OTSU`.
           - Verifica a densidade de pixels de primeiro plano para descartar patches vazios.
           - Redimensiona o patch se necessário para otimizar a altura dos caracteres para OCR.
           - Aplica Tesseract OCR (Português e Inglês, `psm 7`).
           - Valida e tenta correções.
         - Retorna o resultado do patch com um "score" de confiança (1.0 para correspondência direta, 0.8 para sugestão corrigida).
     4.  Os resultados de todos os patches são coletados, e o de maior score é selecionado.
   - **Saída**: Imagem original com retângulo na melhor região, melhor patch colorido, melhor patch binarizado e gráfico são salvos em `output_images_sliding_window_mp_v2/[nome_imagem]/`.

## Funções Chave Adicionais

- **`encontrar_placa(text_string)` e `encontrar_placa_mercosul(text_string)`**: Utilizam expressões regulares (`re`) para validar se uma string corresponde aos formatos de placa padrão antigo ([A-Z]{3}\d{4}) ou Mercosul ([A-Z]{3}[0-9][A-Z0-9][0-9]{2}).
- **`substituir_letras_por_numeros_para_recorte(ultimos_caracteres)` e `gerar_possibilidades_mercosul_para_recorte(value)`**: Funções sofisticadas para gerar variações de uma string de placa detectada, trocando caracteres visualmente similares (definidos em `LETRAS_NUMEROS`) para aumentar a chance de encontrar uma correspondência válida. São usadas principalmente no pós-processamento do OCR.
- **`exibir_e_salvar_resultado_*`**: Funções responsáveis por gerar os gráficos com `matplotlib` e salvar todas as imagens de saída para cada modo.

## Dicionário `LETRAS_NUMEROS`

Este dicionário é crucial para o pós-processamento do OCR. Ele mapeia caracteres que o Tesseract frequentemente confunde:
```python
LETRAS_NUMEROS = {
    'I': '1', 'O': '0', 'Q': '0', 'Z': '2', 'S': ['5', '9'], 'G': '6',
    'B': '8', 'A': '4', 'E': '8', 'T': '7', 'Y': '7', 'L': '1',
    # ... e vice-versa para números que podem ser letras
    '0': 'O', '1': 'I', '2': 'Z', '4': 'A', '5': 'S', '8': 'B'
}
```
Isso permite que o script tente "corrigir" leituras imperfeitas do Tesseract, especialmente para os últimos 4 caracteres da placa antiga ou para o 5º caractere da placa Mercosul.

## Limitações e Melhorias Futuras

- **Sensibilidade a Parâmetros (MSER)**: O modo MSER, mesmo com 3 configurações, ainda é sensível aos parâmetros. Imagens com iluminação muito variada, ângulos extremos, ou placas sujas/danificadas podem exigir ajuste fino adicional ou um número maior de configurações de teste. A versão anterior com geração procedural de parâmetros (se implementada) poderia ser mais robusta aqui.
- **Velocidade (Sliding Window)**: Apesar do multiprocessamento, o modo de janela deslizante pode ser lento para imagens grandes devido ao grande número de patches a serem processados.
- **Qualidade do OCR**: A precisão final depende muito da qualidade da imagem de entrada e da eficácia do pré-processamento e segmentação da placa.
- **Melhorias Possíveis**:
    - Implementar um detector de placas mais robusto baseado em aprendizado de máquina (e.g., Haar Cascades, HOG+SVM, ou modelos mais modernos como YOLO, SSD, Faster R-CNN) antes de aplicar o MSER ou OCR.
    - Adicionar correção de perspectiva para placas inclinadas.
    - Treinar um modelo Tesseract específico para as fontes usadas em placas brasileiras.
    - Desenvolver um sistema de pontuação mais sofisticado para os candidatos a placa, considerando não apenas o texto OCRizado, mas também características geométricas.
    - Refinar a lógica de geração procedural de parâmetros para o modo MSER, talvez com uma busca mais inteligente (e.g., otimização bayesiana) em vez de amostragem aleatória se o espaço de parâmetros for muito grande.

## Conclusão

O `main.py` é um script abrangente que explora diferentes abordagens para o desafio de reconhecimento de placas veiculares. Sua modularidade em modos de processamento e as tentativas de correção de OCR o tornam uma ferramenta poderosa e um excelente ponto de partida para estudos e desenvolvimentos futuros na área.
