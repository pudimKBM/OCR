# Reconhecimento de Placas Veiculares com OpenCV e Tesseract OCR

Este projeto demonstra um programa em Python que utiliza a biblioteca OpenCV para pré-processar imagens de veículos e o Tesseract OCR para extrair o texto de suas placas.

## Funcionalidades

- Carrega imagens de veículos (carros ou motos).
- Realiza pré-processamento na imagem para realçar a placa:
    - Conversão para escala de cinza.
    - Aplicação de filtro Gaussiano para suavização.
    - Binarização adaptativa para segmentação.
- Tenta detectar a região da placa utilizando análise de contornos e heurísticas de forma e tamanho.
- Recorta a região da placa detectada.
- Aplica OCR na região da placa para extrair o texto.
- Exibe o texto extraído no console.
- Salva a imagem original com a placa detectada destacada (usando o contorno aproximado) e o texto extraído sobreposto.

## Instalação e Configuração

### 1. Tesseract OCR

Você precisará instalar o Tesseract OCR em seu sistema.

-   **Windows**:
    -   Baixe o instalador em [Tesseract at UB Mannheim](https://github.com/UB-Mannheim/tesseract/wiki).
    -   Durante a instalação, certifique-se de selecionar os pacotes de idiomas desejados (ex: "Portuguese" - `por`).
    -   **Importante**: Adicione o diretório de instalação do Tesseract (e.g., `C:\Program Files\Tesseract-OCR`) à variável de ambiente PATH do sistema.
    -   Se, mesmo após adicionar ao PATH, o script não encontrar o Tesseract, você pode descomentar e ajustar a linha no início do script `main.py`:
        ```python
        # pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
        ```

-   **Linux (Debian/Ubuntu)**:
    ```bash
    sudo apt update
    sudo apt install tesseract-ocr tesseract-ocr-por  # 'por' para Português
    ```

-   **macOS (usando Homebrew)**:
    ```bash
    brew install tesseract tesseract-lang
    # Para instalar o pacote de idioma português especificamente (se não incluído por tesseract-lang):
    # brew install tesseract-data-por 
    ```

Após a instalação, verifique se o Tesseract está funcionando no terminal com o comando: `tesseract --version`.

### 2. Dependências Python

Recomenda-se criar um ambiente virtual para o projeto.

```bash
# Opcional: criar e ativar um ambiente virtual
python -m venv venv
# No Windows:
# venv\Scripts\activate
# No Linux/macOS:
# source venv/bin/activate
```

Clone este repositório (ou crie a estrutura de pastas manualmente) e instale as bibliotecas Python listadas no arquivo `requirements.txt`:

```bash
pip install -r requirements.txt
```

O conteúdo do `requirements.txt` deve ser:
```
opencv-python
pytesseract
numpy
```

Se preferir, instale manualmente:
```bash
pip install opencv-python pytesseract numpy
```

## Estrutura do Projeto

Certifique-se de que seu projeto tenha a seguinte estrutura:

```
seu_projeto_license_plate_ocr/
├── main.py                 # Script principal
├── requirements.txt        # Dependências Python
├── images/                 # Pasta para colocar as imagens de entrada (VOCÊ DEVE CRIAR ESTA PASTA)
│   ├── placa_carro1.jpg    # Exemplo de imagem
│   ├── placa_moto1.png     # Exemplo de imagem
│   └── ...                 # Suas 5 imagens de teste
├── processed_images/       # Pasta onde as imagens processadas são salvas (criada automaticamente)
│   ├── placa_carro1_processed.jpg
│   └── ...
└── README.md               # Este arquivo
```

## Como Usar

1.  **Crie uma pasta chamada `images`** no mesmo diretório do script `main.py`.
2.  Coloque as 5 imagens das placas que você deseja processar dentro da pasta `images/`.
3.  Execute o script Python a partir do diretório raiz do projeto:
    ```bash
    python main.py
    ```
4.  O programa imprimirá o texto extraído de cada placa no console.
5.  As imagens processadas, com a placa destacada e o texto sobreposto, serão salvas na pasta `processed_images/`. Os caminhos absolutos dos arquivos salvos serão exibidos no console.

## Detalhes do Processamento

O script segue as seguintes etapas principais para cada imagem:

1.  **Carregamento da Imagem**: `cv2.imread()`.
2.  **Pré-processamento da Imagem Completa**:
    -   **Conversão para Escala de Cinza**: `cv2.cvtColor()` - Reduz a complexidade.
    -   **Suavização Gaussiana**: `cv2.GaussianBlur()` - Reduz ruído.
    -   **Binarização Adaptativa**: `cv2.adaptiveThreshold()` - Cria uma imagem binária (preto e branco) lidando bem com variações de iluminação. `THRESH_BINARY_INV` é usado para obter objetos brancos em fundo preto, o que pode facilitar a detecção de contornos.
3.  **Detecção da Região da Placa**:
    -   **Encontrar Contornos**: `cv2.findContours()` na imagem binarizada.
    -   **Filtragem de Contornos**: Itera sobre os contornos e aplica filtros baseados em:
        -   **Forma Quadrilateral**: `cv2.approxPolyDP()` para verificar se o contorno se assemelha a um quadrilátero (placas são retangulares).
        -   **Proporção (Aspect Ratio)**: `largura / altura` do retângulo delimitador. Filtra para proporções típicas de placas (e.g., entre 1.0 e 5.0).
        -   **Área**: Filtra contornos muito pequenos ou muito grandes.
    -   **Seleção**: O contorno que melhor se encaixa (geralmente o maior que passa pelos filtros) é escolhido.
4.  **Recorte e Pré-processamento da ROI (Região de Interesse) da Placa para OCR**:
    -   **Recorte**: A região da placa é recortada da imagem em tons de cinza original, com uma pequena margem.
    -   **Redimensionamento (Upscaling)**: A ROI é aumentada (`cv2.resize` com `cv2.INTER_CUBIC`) para melhorar a precisão do OCR, visando uma altura mínima para os caracteres.
    -   **Binarização da ROI**: A ROI redimensionada é binarizada usando o método de Otsu (`cv2.THRESH_BINARY | cv2.THRESH_OTSU`).
    -   **Garantir Texto Preto em Fundo Branco**: O Tesseract funciona melhor com texto preto sobre fundo branco. Se a binarização resultar em texto branco sobre fundo preto (detectado pela média de intensidade de pixels), a imagem é invertida (`cv2.bitwise_not()`).
5.  **Extração de Texto com Tesseract OCR**:
    -   `pytesseract.image_to_string()` é chamado na ROI processada.
    -   **Configuração**: `custom_config = r'--oem 3 --psm 7 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789'`
        -   `--oem 3`: Motor LSTM.
        -   `--psm 7`: Trata a imagem como uma única linha de texto.
        -   `tessedit_char_whitelist`: Crucial para restringir o OCR aos caracteres válidos em placas.
    -   **Pós-processamento do Texto**: Remove caracteres não alfanuméricos e converte para maiúsculas.
6.  **Exibição e Salvamento**:
    -   O texto extraído é impresso.
    -   O contorno da placa detectada é desenhado na imagem original.
    -   O texto extraído é escrito acima da placa.
    -   A imagem resultante é salva na pasta `processed_images/`.

## Resultados Esperados (Preencha com seus testes)

Você deve testar o programa com 5 imagens diferentes e documentar os resultados aqui. Para cada imagem:

**Imagem 1: `images/NOME_DA_SUA_IMAGEM_1.ext`**
- Placa Detectada: (Sim/Não/Parcialmente)
- Texto Extraído: `TEXTO_AQUI` (ou "N/A" se não detectado/lido)
- Imagem Processada: `processed_images/NOME_DA_SUA_IMAGEM_1_processed.ext`
  <!-- Opcional: !Descrição da Imagem Processada 1 -->

**Imagem 2: `images/NOME_DA_SUA_IMAGEM_2.ext`**
- Placa Detectada: ...
- Texto Extraído: ...
- Imagem Processada: ...

**(Continue para as 5 imagens)**

## Limitações e Melhorias Futuras

-   A detecção de placas é baseada em heurísticas e pode falhar com ângulos difíceis, iluminação ruim, placas danificadas/sujas, ou múltiplos objetos retangulares.
-   A precisão do OCR depende da qualidade da segmentação e do pré-processamento da ROI.
-   **Melhorias Possíveis**:
    -   Usar detectores de objetos mais avançados (Haar Cascades, HOG+SVM, YOLO, SSD).
    -   Implementar correção de perspectiva para placas inclinadas.
    -   Ajustar dinamicamente os parâmetros de pré-processamento.
    -   Treinar um modelo Tesseract específico para fontes de placas.
    -   Validar o formato do texto extraído contra padrões de placas conhecidos.