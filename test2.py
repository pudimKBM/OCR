import cv2
import pytesseract
import os # Adicionado para manipulação de caminhos

pytesseract.pytesseract.tesseract_cmd = r'C:/Program Files/Tesseract-OCR/tesseract.exe' # Update if needed


# Função auxiliar para garantir que um diretório exista
def ensure_dir(directory_path):
    if not os.path.exists(directory_path):
        os.makedirs(directory_path)
        print(f"Diretório criado: {directory_path}")

def encontrarRoiPlaca(source_relative_path):
    script_dir = os.path.dirname(os.path.abspath(__file__))
    image_path = os.path.join(script_dir, source_relative_path)

    img = cv2.imread(image_path)

    if img is None:
        print(f"ERRO: Não foi possível carregar a imagem em '{image_path}'.")
        print("Verifique se o caminho está correto e o arquivo existe.")
        return False # Indica falha

    cv2.imshow("img", img)

    cinza = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # cv2.imshow("cinza", cinza) # Corrigido para mostrar a imagem em cinza

    _, bin = cv2.threshold(cinza, 90, 255, cv2.THRESH_BINARY)
    # cv2.imshow("binary", bin) # Corrigido para mostrar a imagem binarizada

    desfoque = cv2.GaussianBlur(bin, (5, 5), 0)
    # cv2.imshow("defoque", desfoque)

    contornos, hierarquia = cv2.findContours(desfoque, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    # cv2.drawContours(img, contornos, -1, (0, 255, 0), 1)

    # Garante que o diretório de saída exista
    output_dir_absolute = os.path.join(script_dir, "output")
    ensure_dir(output_dir_absolute)
    roi_saved_path = os.path.join(output_dir_absolute, 'roi.png')

    found_roi = False
    for c in contornos:
        perimetro = cv2.arcLength(c, True)
        if perimetro > 120:
            aprox = cv2.approxPolyDP(c, 0.03 * perimetro, True)
            if len(aprox) == 4:
                (x, y, w, h) = cv2.boundingRect(c) # Usando w (width) e h (height) para clareza
                # Adicionar filtro de aspect ratio pode ser útil aqui
                # Ex: if w / float(h) > 1.5 and w / float(h) < 5.0:
                cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
                roi = img[y:y + h, x:x + w] # Correto: y:y+altura, x:x+largura
                cv2.imwrite(roi_saved_path, roi)
                print(f"ROI salva em: {roi_saved_path}")
                found_roi = True
                # break # Descomente se quiser apenas a primeira ROI encontrada

    cv2.imshow("contornos", img)
    return found_roi # Retorna True se uma ROI foi salva


def preProcessamentoRoiPlaca():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    roi_path = os.path.join(script_dir, "output", "roi.png")
    img_roi = cv2.imread(roi_path)

    if img_roi is None:
        print(f"ERRO: Não foi possível carregar 'roi.png' de '{roi_path}'. 'encontrarRoiPlaca' pode não ter encontrado/salvo uma ROI.")
        return None

    resize_img_roi = cv2.resize(img_roi, None, fx=4, fy=4, interpolation=cv2.INTER_CUBIC)

    # Converte para escala de cinza
    img_cinza = cv2.cvtColor(resize_img_roi, cv2.COLOR_BGR2GRAY)

    # Binariza imagem
    _, img_binary = cv2.threshold(img_cinza, 70, 255, cv2.THRESH_BINARY)

    # Desfoque na Imagem
    img_desfoque = cv2.GaussianBlur(img_binary, (5, 5), 0)

    # Grava o pre-processamento para o OCR
    roi_ocr_path = os.path.join(script_dir, "output", "roi-ocr.png")
    cv2.imwrite(roi_ocr_path, img_desfoque)
    print(f"ROI pré-processada para OCR salva em: {roi_ocr_path}")

    #cv2.imshow("ROI", img_desfoque)
    return roi_ocr_path # Retorna o caminho da imagem processada


def ocrImageRoiPlaca(image_path_for_ocr):
    # Se o Tesseract não estiver no PATH, você pode precisar especificar o caminho:
    # pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe' # Exemplo Windows

    image = cv2.imread(image_path_for_ocr)
    if image is None:
        print(f"ERRO OCR: Não foi possível carregar a imagem '{image_path_for_ocr}' para OCR.")
        return "ERRO_LEITURA_IMAGEM_OCR"

    config = r'-c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789 --psm 6'
    # Considere usar 'por' (Português) se tiver o pacote de idioma instalado e as placas forem brasileiras
    saida = pytesseract.image_to_string(image, lang='eng', config=config) # ou lang='por'

    return saida.strip() # .strip() para remover espaços em branco extras


if __name__ == "__main__":
    # O caminho para a imagem de entrada, relativo ao local do script
    # Certifique-se que a pasta 'resource' existe no mesmo diretório que 'import cv2.py'
    # e que 'carro4.jpg' está dentro de 'resource'.
    input_image_relative_path = os.path.join("images", "Placa5.png")

    if encontrarRoiPlaca(input_image_relative_path):
        path_roi_processada = preProcessamentoRoiPlaca()
        if path_roi_processada:
            texto_ocr = ocrImageRoiPlaca(path_roi_processada)
            print(f"Texto Extraído da Placa: {texto_ocr}")
        else:
            print("Pré-processamento da ROI falhou. OCR não executado.")
    else:
        print("Não foi possível encontrar ou salvar a ROI da placa. Processo interrompido.")

    print("Pressione qualquer tecla para fechar as janelas de imagem...")
    cv2.waitKey(0)
    cv2.destroyAllWindows()