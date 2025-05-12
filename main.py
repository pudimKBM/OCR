import os
import cv2
import pytesseract
import matplotlib.pyplot as plt
import re
import argparse
from itertools import product
import numpy as np
import multiprocessing # For other modes, not used by MSER in this impl.
import time

# --- Configuration ---
pytesseract.pytesseract.tesseract_cmd = r'C:/Program Files/Tesseract-OCR/tesseract.exe' # Update if needed

LETRAS_NUMEROS = {
    'I': '1', 'O': '0', 'Q': '0', 'Z': '2', 'S': ['5', '9'], 'G': '6',
    'B': '8', 'A': '4', 'E': '8', 'T': '7', 'Y': '7', 'L': '1',
    'U': '0', 'D': '0', 'R': '2', 'P': '0', 'F': '0', 'J': '1',
    'K': '1', 'V': '0', 'W': '0', 'X': '0', 'N': '0', 'M': '0',
    'H': '0', 'C': '0', 'Ç': '0', 'Á': '0', 'Â': '0', 'Ã': '0', 'À': '0',
    '0': 'O', '1': 'I', '2': 'Z', '4': 'A', '5': 'S', '8': 'B'
}

# --- Helper Functions (Top Level) ---
def encontrar_placa(text_string):
    padrao = r'[A-Z]{3}\d{4}'
    placas_encontradas = re.findall(padrao, str(text_string).upper())
    return placas_encontradas[0] if placas_encontradas else None

def encontrar_placa_mercosul(text_string):
    padrao = r'[A-Z]{3}[0-9][A-Z0-9][0-9]{2}'
    placas_encontradas = re.findall(padrao, str(text_string).upper())
    return placas_encontradas[0] if placas_encontradas else None

def substituir_letras_por_numeros_para_recorte(ultimos_caracteres: str):
    todas_possibilidades = ['']
    for caractere in reversed(ultimos_caracteres):
        novas_possibilidades_para_iteracao_atual = []
        substitutions = LETRAS_NUMEROS.get(caractere.upper())
        current_chars_options = [caractere] 
        if substitutions:
            current_chars_options = substitutions if isinstance(substitutions, list) else [substitutions]
        for char_option in current_chars_options:
            for possibilidade_anterior in todas_possibilidades:
                novas_possibilidades_para_iteracao_atual.append(str(char_option) + str(possibilidade_anterior))
        todas_possibilidades = novas_possibilidades_para_iteracao_atual
    return todas_possibilidades

def _combinar_elementos_para_recorte(lista_de_listas_de_chars, prefixo=''):
    if not lista_de_listas_de_chars: return [prefixo]
    resultado = []
    elementos_atuais = lista_de_listas_de_chars[0]
    for item_char in elementos_atuais:
        novo_prefixo = prefixo + str(item_char)
        resultado.extend(_combinar_elementos_para_recorte(lista_de_listas_de_chars[1:], novo_prefixo))
    return resultado

def gerar_possibilidades_mercosul_para_recorte(value: str):
    if len(value) != 4: return [value]
    todas_as_combinacoes_finais = set()
    lista_de_opcoes_por_char = []
    for i, char_val in enumerate(value):
        char_upper = char_val.upper()
        opcoes_char_atual = [char_upper]
        substitutions = LETRAS_NUMEROS.get(char_upper)
        if i == 1: 
            if substitutions:
                opcoes_char_atual.extend(s for s in (substitutions if isinstance(substitutions, list) else [substitutions]) if s.isalnum())
            if char_upper.isalnum() and char_upper not in opcoes_char_atual:
                opcoes_char_atual.append(char_upper)
        else: 
            if substitutions:
                digit_options = [s for s in (substitutions if isinstance(substitutions, list) else [substitutions]) if s.isdigit()]
                if digit_options: opcoes_char_atual = digit_options
                elif char_upper.isdigit(): opcoes_char_atual = [char_upper]
            elif not char_upper.isdigit(): opcoes_char_atual = [char_upper]
        lista_de_opcoes_por_char.append(list(set(opcoes_char_atual)))
    if lista_de_opcoes_por_char and all(isinstance(sublist, list) for sublist in lista_de_opcoes_por_char):
         for comb in _combinar_elementos_para_recorte(lista_de_opcoes_por_char):
            if len(comb) == 4: todas_as_combinacoes_finais.add(comb)
    return sorted(list(todas_as_combinacoes_finais)) if todas_as_combinacoes_finais else [value]


def processar_imagem_globalmente(imagem_original):
    imagem_cinza = cv2.cvtColor(imagem_original, cv2.COLOR_BGR2GRAY)
    imagem_cinza_blur = cv2.bilateralFilter(imagem_cinza, 9, 75, 75) # Good for edge preservation
    # This binarized image is primarily for display or if a global threshold is needed for other steps.
    # MSER itself works on grayscale. The OCR on crops will do its own binarization.
    imagem_limiarizada_display = cv2.adaptiveThreshold(
        imagem_cinza_blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 19, 9
    )
    return imagem_limiarizada_display, imagem_cinza

# --- Parameterized MSER Helper Functions ---
def filter_char_candidates_param(bboxes, params):
    min_char_h = params.get("char_h_min_abs", 15)
    max_char_h = params.get("char_h_max_abs", 60)
    min_aspect_char = params.get("char_aspect_min", 0.1)
    max_aspect_char = params.get("char_aspect_max", 1.0)
    
    char_candidates = []
    for (x, y, w, h) in bboxes:
        if not (min_char_h <= h <= max_char_h): continue
        aspect_ratio = w / float(h) if h > 0 else 0
        if not (min_aspect_char <= aspect_ratio <= max_aspect_char): continue
        char_candidates.append({'x': x, 'y': y, 'w': w, 'h': h, 'cx': x + w/2, 'cy': y + h/2})
    return char_candidates
def group_char_candidates_param(char_candidates, params):
    if not char_candidates: return []

    max_dy_ratio = params.get("grp_dy", 0.15)
    max_dh_ratio = params.get("grp_dh", 0.2)
    max_spacing_ratio = params.get("grp_spacing", 0.6)
    min_chars = params.get("grp_min_chars", 6)
    max_chars = params.get("grp_max_chars", 8)
    min_plate_aspect = params.get("grp_plate_ar_min", 2.5)
    max_plate_aspect = params.get("grp_plate_ar_max", 5.5)
    ref_char_h_for_plate_check = params.get("char_h_min_abs", 15) # Use min char height as ref

    char_candidates.sort(key=lambda c: (c['x'], c['y']))
    potential_plates = []
    visited = [False] * len(char_candidates)

    for i in range(len(char_candidates)):
        if visited[i]: continue
        current_line = [char_candidates[i]]
        visited[i] = True; current_line_y_coords = [char_candidates[i]['cy']]; current_line_heights = [char_candidates[i]['h']]

        for j in range(i + 1, len(char_candidates)):
            if visited[j]: continue
            cand_char = char_candidates[j]; prev_char = current_line[-1]
            avg_line_y = np.mean(current_line_y_coords); avg_line_h = np.mean(current_line_heights)
            if avg_line_h == 0: continue # Should not happen if chars have height
            dy = abs(cand_char['cy'] - avg_line_y); dh = abs(cand_char['h'] - avg_line_h)
            dx_spacing = cand_char['x'] - (prev_char['x'] + prev_char['w'])
            
            if (dy < avg_line_h * max_dy_ratio and \
                dh < avg_line_h * max_dh_ratio and \
                0 <= dx_spacing < avg_line_h * max_spacing_ratio and \
                cand_char['x'] > prev_char['x'] + prev_char['w'] * 0.5): # Ensure progression
                current_line.append(cand_char)
                visited[j] = True
                current_line_y_coords.append(cand_char['cy']); current_line_heights.append(cand_char['h'])
            elif dx_spacing > avg_line_h * max_spacing_ratio * 1.5 : # If spacing is too large, new group
                 break
        
        if min_chars <= len(current_line) <= max_chars:
            line_x_coords = [c['x'] for c in current_line]; line_y_coords = [c['y'] for c in current_line]
            line_w_coords = [c['x'] + c['w'] for c in current_line]; line_h_coords = [c['y'] + c['h'] for c in current_line]
            plate_x = min(line_x_coords); plate_y = min(line_y_coords)
            plate_w = max(line_w_coords) - plate_x; plate_h = max(line_h_coords) - plate_y
            if plate_w > 0 and plate_h > 0:
                plate_aspect_ratio = plate_w / float(plate_h)
                if min_plate_aspect < plate_aspect_ratio < max_plate_aspect and \
                   plate_h > ref_char_h_for_plate_check * 0.7: # Plate height vs reference char height
                    potential_plates.append({'chars': current_line, 
                                             'bbox': (plate_x, plate_y, plate_w, plate_h),
                                             'avg_char_h': np.mean(current_line_heights)})
    return potential_plates


def encontrar_placas_via_mser_param(imagem_original_color, imagem_cinza_input, param_config):
    img_h, img_w = imagem_cinza_input.shape[:2]

    # 1. Contrast Enhancement (CLAHE)
    clahe_clip = param_config.get("clahe_clip", 2.0)
    clahe_grid_w = param_config.get("clahe_grid_w", 8)
    clahe_grid_h = param_config.get("clahe_grid_h", 8)
    clahe = cv2.createCLAHE(clipLimit=clahe_clip, tileGridSize=(clahe_grid_w, clahe_grid_h))
    imagem_cinza_enhanced = clahe.apply(imagem_cinza_input)
    # You might want to display imagem_cinza_enhanced to see its effect

    # 2. Dynamic Parameter Calculation
    # These percentages are heuristics and NEED TUNING based on your typical images
    est_char_h_min_rel = 0.020 # Min char height relative to image height
    est_char_h_max_rel = 0.080 # Max char height relative to image height    
    abs_min_char_h_px = 15    # Absolute minimum pixel height for a char
    abs_max_char_h_px = 80    # Absolute maximum pixel height for a char

    min_char_h_abs = max(param_config.get("abs_min_char_h_px", 15), 
                         int(img_h * param_config.get("char_h_min_r", 0.025)))
    max_char_h_abs = min(param_config.get("abs_max_char_h_px", 60), 
                         int(img_h * param_config.get("char_h_max_r", 0.075)))
    if min_char_h_abs >= max_char_h_abs : # Safety if image is too small or percentages are off
        min_char_h_abs = param_config.get("abs_min_char_h_px", 15)
        max_char_h_abs = param_config.get("abs_max_char_h_px", 60)


    # Estimate MSER Area based on dynamic character height and typical aspect ratio
    # Char aspect ratio (w/h) typically 0.2 (for 'I') to 1.0 (for 'W', 'M')
    mser_min_area = max(param_config.get("abs_min_mser_area_px", 30), 
                        int((min_char_h_abs**2) * param_config.get("mser_min_area_f", 0.10)))
    mser_max_area = min(int((max_char_h_abs**2) * param_config.get("mser_max_area_f", 1.5)), 
                        int(img_w * img_h * param_config.get("mser_max_area_img_f", 0.02)))
    # Cap max area to avoid huge regions, e.g., 5% of total image area as an extreme upper bound
    mser_max_area = max(mser_max_area, mser_min_area + 100) # Ensure max > min


    # --- MSER Parameters (Now more dynamic) ---
    mser = cv2.MSER_create()
    mser.setDelta(param_config.get("mser_delta", 5))
    mser.setMinArea(mser_min_area)
    mser.setMaxArea(mser_max_area) 
    mser.setMaxVariation(param_config.get("mser_max_var", 0.25))
    mser.setMinDiversity(param_config.get("mser_min_div", 0.2))
    # --- End MSER Parameters ---
    
    # Detect regions on the enhanced grayscale image
    regions, bboxes = mser.detectRegions(imagem_cinza_enhanced) # Use enhanced image
    if bboxes is None or len(bboxes) == 0:
        # print("MSER found no regions or bboxes.") # Can be verbose
        return []

    current_filter_params = {
        "char_h_min_abs": min_char_h_abs, "char_h_max_abs": max_char_h_abs,
        "char_aspect_min": param_config.get("char_aspect_min", 0.15),
        "char_aspect_max": param_config.get("char_aspect_max", 1.0)
    }
    char_candidates = filter_char_candidates_param(bboxes, current_filter_params)
    if not char_candidates: return []

    current_grouping_params = {
        "grp_dy": param_config.get("grp_dy", 0.15), "grp_dh": param_config.get("grp_dh", 0.2),
        "grp_spacing": param_config.get("grp_spacing", 0.6),
        "grp_min_chars": param_config.get("grp_min_chars", 6),
        "grp_max_chars": param_config.get("grp_max_chars", 8),
        "grp_plate_ar_min": param_config.get("grp_plate_ar_min", 2.5),
        "grp_plate_ar_max": param_config.get("grp_plate_ar_max", 5.5),
        "char_h_min_abs": min_char_h_abs # Pass ref char height for plate height check
    }
    plate_groups = group_char_candidates_param(char_candidates, current_grouping_params)
    if not plate_groups: return []

    # ... (rest of the function: cropping, binarizing crop, returning list for OCR) ...
    # The binarization of the crop might also benefit from CLAHE on plate_crop_gray before thresholding
    # Example for crop binarization:
    #   plate_crop_gray_enhanced = clahe.apply(plate_crop_gray) # Use the same CLAHE object or a new one
    #   plate_crop_bin = cv2.adaptiveThreshold(plate_crop_gray_enhanced, ...)

    possiveis_placas_para_ocr = []
    for group in plate_groups:
        x, y, w, h = group['bbox']
        avg_char_h_group = group.get('avg_char_h', min_char_h_abs) # Use avg char height of group for padding
        padding = max(3, min(10, int(avg_char_h_group * param_config.get("crop_padding_f", 0.15))))

        x1=max(0,x-padding); y1=max(0,y-padding); x2=min(img_w,x+w+padding); y2=min(img_h,y+h+padding)
        # Ensure crop is reasonably sized based on expected character dimensions
        if (x2-x1) < (param_config.get("grp_min_chars",6) * min_char_h_abs * param_config.get("char_aspect_min",0.15)) or \
           (y2-y1) < (min_char_h_abs * 0.7): continue
        
        plate_crop_color = imagem_original_color[y1:y2, x1:x2]
        plate_crop_gray = cv2.cvtColor(plate_crop_color, cv2.COLOR_BGR2GRAY)
        
        # Enhance contrast of the small crop before binarization
        plate_crop_gray_enhanced = clahe.apply(plate_crop_gray) # Applying CLAHE again to the crop
        plate_crop_bin = cv2.adaptiveThreshold(plate_crop_gray_enhanced, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                               cv2.THRESH_BINARY_INV, 
                                               param_config.get("crop_adapt_block", 11), 
                                               param_config.get("crop_adapt_c", 4))

        if plate_crop_bin.size > 0:
            possiveis_placas_para_ocr.append((plate_crop_color, plate_crop_bin))
            
    return possiveis_placas_para_ocr

def aplicar_ocr_aos_recortes(lista_possiveis_placas_recortadas):
    custom_config_base = r'-c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789 --oem 3'; psm_mode_crop = r'--psm 7'
    for tupla_placa in lista_possiveis_placas_recortadas:
        placa_recortada_original_color, placa_recortada_processada_ocr = tupla_placa
        if placa_recortada_processada_ocr is None or placa_recortada_processada_ocr.size == 0 : continue
        h_ocr, w_ocr = placa_recortada_processada_ocr.shape[:2]; target_char_height = 35 # pixels
        if h_ocr > 0 and w_ocr > 0 and h_ocr < target_char_height * 0.8: 
            scale = target_char_height / h_ocr
            new_w, new_h = int(w_ocr*scale), int(h_ocr*scale)
            if new_w > 0 and new_h > 0: 
                placa_recortada_processada_ocr = cv2.resize(placa_recortada_processada_ocr, (new_w, new_h), interpolation=cv2.INTER_LANCZOS4)
        for lang_code in ['por', 'eng']:
            custom_config = f'{custom_config_base} {psm_mode_crop} lang={lang_code}'
            try:
                texto_ocr = pytesseract.image_to_string(placa_recortada_processada_ocr, config=custom_config, timeout=5)
            except RuntimeError: texto_ocr ="" # Catch timeout
            placa_detectada_ocr = "".join(filter(str.isalnum, texto_ocr)).upper()
            if len(placa_detectada_ocr) >= 7:
                plate_m = encontrar_placa_mercosul(placa_detectada_ocr)
                if plate_m: return plate_m, placa_recortada_original_color, placa_recortada_processada_ocr
                plate_a = encontrar_placa(placa_detectada_ocr)
                if plate_a: return plate_a, placa_recortada_original_color, placa_recortada_processada_ocr
        custom_config_por = f'{custom_config_base} {psm_mode_crop} lang=por'
        try:
            texto_ocr_por = pytesseract.image_to_string(placa_recortada_processada_ocr, config=custom_config_por, timeout=5)
        except RuntimeError: texto_ocr_por = ""
        placa_base_para_correcao = "".join(filter(str.isalnum, texto_ocr_por)).upper()
        if len(placa_base_para_correcao) == 7:
            primeiros_3 = placa_base_para_correcao[:3]; quarto_char = placa_base_para_correcao[3]; ultimos_4 = placa_base_para_correcao[-4:]
            resultados_possiveis = []
            if primeiros_3.isalpha():
                poss_antiga_nums = substituir_letras_por_numeros_para_recorte(ultimos_4)
                for num_part in poss_antiga_nums:
                    if len(num_part) == 4 and all(c.isdigit() for c in num_part):
                        p_antiga = primeiros_3 + num_part
                        if encontrar_placa(p_antiga): resultados_possiveis.append(f"{p_antiga} (Old guess)")
                if quarto_char.isdigit():
                    segmento_mercosul = placa_base_para_correcao[3:7]
                    poss_mercosul_parts = gerar_possibilidades_mercosul_para_recorte(segmento_mercosul)
                    for merc_part in poss_mercosul_parts:
                        if len(merc_part) == 4 and merc_part[0].isdigit() and merc_part[1].isalnum() and merc_part[2:].isdigit():
                            p_mercosul = primeiros_3 + merc_part
                            if encontrar_placa_mercosul(p_mercosul): resultados_possiveis.append(f"{p_mercosul} (Mercosul guess)")
            if resultados_possiveis: return "\n".join(sorted(list(set(resultados_possiveis)))), placa_recortada_original_color, placa_recortada_processada_ocr
        if placa_base_para_correcao: return f"Raw OCR (MSER crop): {placa_base_para_correcao}", placa_recortada_original_color, placa_recortada_processada_ocr
    return "No plate detected (MSER crop)", None, None

# This function is used by MSER mode
def exibir_e_salvar_resultado_com_recorte(*args):
    img_orig, img_processed_display, img_crop_color, img_crop_ocr, text_plate, out_dir, base_fname = args
    os.makedirs(out_dir, exist_ok=True)
    if img_orig is not None: cv2.imwrite(os.path.join(out_dir, f"{base_fname}_01_original.png"), img_orig)
    if img_processed_display is not None: cv2.imwrite(os.path.join(out_dir, f"{base_fname}_02_processed_for_detection.png"), img_processed_display)
    if img_crop_color is not None: cv2.imwrite(os.path.join(out_dir, f"{base_fname}_03_mser_plate_color.png"), img_crop_color)
    if img_crop_ocr is not None and img_crop_ocr.size > 0: cv2.imwrite(os.path.join(out_dir, f"{base_fname}_04_mser_plate_ocr.png"), img_crop_ocr)
    
    # Ensure text_plate is a string for suptitle
    display_text_plate = str(text_plate) if text_plate is not None else "N/A"

    fig = plt.figure(figsize=(12,9)); 
    plt.suptitle(f"Plate OCR (MSER Crop Mode) for '{base_fname}':\n{display_text_plate}", fontsize=14)
    
    ax_map = {
        (0,0):("1. Original",img_orig,'rgb'), 
        (0,1):("2. Grayscale for MSER",img_processed_display,'gray'), 
        (1,0):("3. MSER Detected Plate (Color)",img_crop_color,'rgb'), 
        (1,1):("4. MSER Plate for OCR",img_crop_ocr,'gray') 
    }

    for (r,c),(title,data,cmap_type) in ax_map.items():
        ax=plt.subplot2grid((2,2),(r,c))
        if data is not None and data.size>0: 
            try:
                if cmap_type=='rgb': ax.imshow(cv2.cvtColor(data,cv2.COLOR_BGR2RGB))
                else: ax.imshow(data,cmap='gray')
            except Exception as e: 
                print(f"Error displaying {title} for {base_fname}: {e}")
                ax.text(0.5,0.5,'Display Error',ha='center',va='center',color='red')
        else: ax.text(0.5,0.5,'No Image Data',ha='center',va='center')
        ax.set_title(title); ax.axis('off')
    
    plt.tight_layout(rect=[0,0.03,1,0.93]); fig_path=os.path.join(out_dir,f"{base_fname}_00_fig_mser_crop.png"); plt.savefig(fig_path)
    print(f"MSER Crop mode outputs saved in: {out_dir}"); 
    # plt.show() # Comment out if running many images to avoid blocking
    plt.close(fig) # Close the figure to free memory

# --- Orchestrator for MSER "Crop" Mode with Tuning ---
def detectar_placa_com_mser_crop_tuned(caminho_da_imagem, output_base_dir): 
    img_orig = cv2.imread(caminho_da_imagem)
    if img_orig is None: print(f"Error reading {caminho_da_imagem}"); return
    
    base_fname = os.path.splitext(os.path.basename(caminho_da_imagem))[0]
    spec_out_dir = os.path.join(output_base_dir, base_fname)
    os.makedirs(spec_out_dir, exist_ok=True)

    _, imagem_cinza = processar_imagem_globalmente(img_orig) # Get grayscale for MSER

    # --- Define Parameter Sets for Tuning ---
    parameter_configurations = [
        # Config 1: Defaultish, slightly stricter from previous
        {"name": "config1_stricter", "mser_delta": 5, "mser_min_area_f": 0.10, "mser_max_area_f": 1.5, "mser_max_var": 0.25, "mser_min_div": 0.2,
         "char_h_min_r": 0.025, "char_h_max_r": 0.075, "char_aspect_min": 0.15, "char_aspect_max": 1.0, "abs_min_char_h_px": 15, "abs_max_char_h_px": 60,
         "grp_dy": 0.15, "grp_dh": 0.2, "grp_spacing": 0.6, "grp_min_chars": 6, "grp_max_chars": 8, "grp_plate_ar_min": 2.5, "grp_plate_ar_max": 5.5,
         "crop_padding_f": 0.15, "crop_adapt_block": 11, "crop_adapt_c": 4, "clahe_clip": 2.0, "clahe_grid_w": 8, "clahe_grid_h": 8,
         "abs_min_mser_area_px": 30, "mser_max_area_img_f": 0.02},
        # Config 2: More relaxed character filtering, tighter MSER area
        {"name": "config2_relax_charFilt", "mser_delta": 5, "mser_min_area_f": 0.08, "mser_max_area_f": 1.2, "mser_max_var": 0.3, "mser_min_div": 0.15,
         "char_h_min_r": 0.02, "char_h_max_r": 0.1, "char_aspect_min": 0.1, "char_aspect_max": 1.2, "abs_min_char_h_px": 12, "abs_max_char_h_px": 70,
         "grp_dy": 0.2, "grp_dh": 0.25, "grp_spacing": 0.8, "grp_min_chars": 5, "grp_max_chars": 8, "grp_plate_ar_min": 2.0, "grp_plate_ar_max": 6.0,
         "crop_padding_f": 0.20, "crop_adapt_block": 13, "crop_adapt_c": 5, "clahe_clip": 2.5, "clahe_grid_w": 8, "clahe_grid_h": 8,
         "abs_min_mser_area_px": 25, "mser_max_area_img_f": 0.025},
        # Config 3: Different MSER sensitivity, wider spacing for grouping
        {"name": "config3_diff_mser_sens", "mser_delta": 3, "mser_min_area_f": 0.12, "mser_max_area_f": 1.6, "mser_max_var": 0.2, "mser_min_div": 0.25,
         "char_h_min_r": 0.022, "char_h_max_r": 0.08, "char_aspect_min": 0.12, "char_aspect_max": 1.1, "abs_min_char_h_px": 14, "abs_max_char_h_px": 65,
         "grp_dy": 0.25, "grp_dh": 0.3, "grp_spacing": 1.0, "grp_min_chars": 6, "grp_max_chars": 8, "grp_plate_ar_min": 2.2, "grp_plate_ar_max": 5.8,
         "crop_padding_f": 0.10, "crop_adapt_block": 11, "crop_adapt_c": 3, "clahe_clip": 1.5, "clahe_grid_w": 6, "clahe_grid_h": 6,
         "abs_min_mser_area_px": 35, "mser_max_area_img_f": 0.015},
    ]

    found_plate_successfully = False
    best_ocr_text, best_rec_col, best_rec_ocr = "Tuning: No plate found.", None, None

    for i, current_params in enumerate(parameter_configurations):
        print(f"\nAttempt {i+1}/{len(parameter_configurations)} with MSER config: '{current_params['name']}'")
        
        lista_recortes_mser = encontrar_placas_via_mser_param(img_orig, imagem_cinza, current_params)

        if lista_recortes_mser:
            # print(f"  MSER generated {len(lista_recortes_mser)} candidates for OCR with config '{current_params['name']}'.") # Verbose
            texto_pl_ocr_result, rec_col_from_ocr, rec_ocr_from_ocr = aplicar_ocr_aos_recortes(lista_recortes_mser)
            
            best_ocr_text = texto_pl_ocr_result # Store last attempt
            best_rec_col = rec_col_from_ocr
            best_rec_ocr = rec_ocr_from_ocr

            first_line_ocr = str(texto_pl_ocr_result).split('\n')[0] if texto_pl_ocr_result else ""
            is_valid_plate = encontrar_placa(first_line_ocr) or encontrar_placa_mercosul(first_line_ocr)
            
            if is_valid_plate and not "Raw OCR" in first_line_ocr and not "No plate detected" in first_line_ocr and not "Tuning:" in first_line_ocr : 
                print(f"  SUCCESS with config '{current_params['name']}'! Plate: {first_line_ocr}")
                success_output_dir = os.path.join(spec_out_dir, f"success_{current_params['name']}")
                exibir_e_salvar_resultado_com_recorte(img_orig, imagem_cinza, rec_col_from_ocr, rec_ocr_from_ocr, 
                                                      texto_pl_ocr_result, success_output_dir, base_fname)
                found_plate_successfully = True
                break 
            else:
                print(f"  Config '{current_params['name']}' OCR: {str(texto_pl_ocr_result)[:60]}...")
        else:
            print(f"  No MSER candidates found with config '{current_params['name']}'.")
            if i == len(parameter_configurations) -1 and not found_plate_successfully: # If last attempt and still no success
                 best_ocr_text = f"No MSER candidates from any config for {base_fname}."


    if not found_plate_successfully:
        print(f"\nCould not find a valid plate for {base_fname} after trying {len(parameter_configurations)} MSER configurations.")
        final_display_text = best_ocr_text if best_ocr_text else "Tuning failed to find plate."
        exibir_e_salvar_resultado_com_recorte(img_orig, imagem_cinza, best_rec_col, best_rec_ocr, 
                                              final_display_text, os.path.join(spec_out_dir, "tuning_final_attempt"), base_fname)


# --- Mode 2: "Without Cropping" Functions ---
# ... (Unchanged: aplicar_ocr_na_imagem_inteira, exibir_e_salvar_resultado_sem_recorte, detectar_placa_sem_recorte)
def aplicar_ocr_na_imagem_inteira(imagem_processada_para_ocr):
    custom_config_base=r'-c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789 --oem 3'; psm_mode_nocrop=r'--psm 6'
    raw_ocr_texts={};
    for lang_code in ['por','eng']:
        custom_config=f'{custom_config_base} {psm_mode_nocrop} lang={lang_code}'
        texto_ocr=pytesseract.image_to_string(imagem_processada_para_ocr,config=custom_config, timeout=7)
        placa_detectada_full="".join(filter(str.isalnum,texto_ocr)).upper(); raw_ocr_texts[lang_code]=placa_detectada_full
        plate_m=encontrar_placa_mercosul(placa_detectada_full)
        if plate_m: return plate_m,imagem_processada_para_ocr
        plate_a=encontrar_placa(placa_detectada_full)
        if plate_a: return plate_a,imagem_processada_para_ocr
    best_raw_text=raw_ocr_texts.get('por','') if len(raw_ocr_texts.get('por',''))>=len(raw_ocr_texts.get('eng','')) else raw_ocr_texts.get('eng','')
    max_raw_len=60
    if best_raw_text and len(best_raw_text)<max_raw_len: return f"No exact match. Raw OCR (full): {best_raw_text}",imagem_processada_para_ocr
    elif best_raw_text: return f"No plate pattern in full OCR (output >{max_raw_len} chars).",imagem_processada_para_ocr
    return "No text detected in full image OCR.",imagem_processada_para_ocr

def exibir_e_salvar_resultado_sem_recorte(*args):
    img_orig,img_ocr_proc,text_plate,out_dir,base_fname = args
    os.makedirs(out_dir,exist_ok=True)
    if img_orig is not None: cv2.imwrite(os.path.join(out_dir,f"{base_fname}_01_original.png"),img_orig)
    if img_ocr_proc is not None: cv2.imwrite(os.path.join(out_dir,f"{base_fname}_02_processed_ocr.png"),img_ocr_proc)
    fig=plt.figure(figsize=(12,6)); plt.suptitle(f"Plate OCR (No Crop Mode) for '{base_fname}':\n{text_plate}",fontsize=14)
    ax1=plt.subplot(1,2,1);
    if img_orig is not None and img_orig.size>0: ax1.imshow(cv2.cvtColor(img_orig,cv2.COLOR_BGR2RGB))
    else: ax1.text(0.5,0.5,'No Original Image',ha='center',va='center')
    ax1.set_title("1. Original"); ax1.axis('off')
    ax2=plt.subplot(1,2,2)
    if img_ocr_proc is not None and img_ocr_proc.size>0: ax2.imshow(img_ocr_proc,cmap='gray')
    else: ax2.text(0.5,0.5,'No Processed Image',ha='center',va='center')
    ax2.set_title("2. Processed for OCR"); ax2.axis('off')
    plt.tight_layout(rect=[0,0.03,1,0.93]); fig_path=os.path.join(out_dir,f"{base_fname}_00_fig_nocrop.png"); 
    plt.savefig(fig_path); plt.close(fig) # Save and close
    print(f"No-crop mode outputs saved in: {out_dir}"); plt.show()

def detectar_placa_sem_recorte(caminho_da_imagem, output_base_dir):
    img_orig=cv2.imread(caminho_da_imagem)
    if img_orig is None: print(f"Error reading {caminho_da_imagem}"); return
    base_fname=os.path.splitext(os.path.basename(caminho_da_imagem))[0]
    spec_out_dir=os.path.join(output_base_dir,base_fname)
    imagem_binarizada_global, _ = processar_imagem_globalmente(img_orig)
    text_pl,_=aplicar_ocr_na_imagem_inteira(imagem_binarizada_global)
    exibir_e_salvar_resultado_sem_recorte(img_orig,imagem_binarizada_global,text_pl,spec_out_dir,base_fname)


# --- Mode 3: "Sliding Window" Functions (with Multiprocessing) ---
# ... (Unchanged: process_patch_task, encontrar_melhor_resultado_via_janela_deslizante, 
# exibir_e_salvar_resultado_janela_deslizante, detectar_placa_via_janela_deslizante)
def process_patch_task(args_tuple):
    x, y, win_w, win_h, full_gray_image, full_color_image, \
    target_ocr_char_height, custom_config_patch_base, psm_patch, tesseract_timeout_patch = args_tuple
    patch_cinza_roi = full_gray_image[y:y + win_h, x:x + win_w]
    color_patch_roi = full_color_image[y:y+win_h, x:x+win_w]
    _, patch_bin_ocr = cv2.threshold(patch_cinza_roi, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    if patch_bin_ocr.size > 0:
        non_zero_pixels = cv2.countNonZero(patch_bin_ocr)
        total_pixels = patch_bin_ocr.size
        foreground_ratio = non_zero_pixels / total_pixels
        if foreground_ratio < 0.015: return None 
    else: return None
    current_patch_bin_for_ocr = patch_bin_ocr
    h_ocr_p, w_ocr_p = current_patch_bin_for_ocr.shape[:2]
    if h_ocr_p == 0 or w_ocr_p == 0: return None
    if h_ocr_p < target_ocr_char_height * 0.7 and h_ocr_p > 0:
        scale = target_ocr_char_height / h_ocr_p
        max_scaled_w, max_scaled_h = 800, 200 
        scaled_w, scaled_h = int(w_ocr_p * scale), int(h_ocr_p * scale)
        if scaled_w > max_scaled_w: scale_w_cap = max_scaled_w / w_ocr_p; scaled_w = max_scaled_w; scaled_h = int(h_ocr_p * scale_w_cap)
        if scaled_h > max_scaled_h: scale_h_cap = max_scaled_h / h_ocr_p; scaled_h = max_scaled_h; scaled_w = int(w_ocr_p * scale_h_cap)
        scaled_w, scaled_h = max(1, scaled_w), max(1, scaled_h)
        if scale > 0.1: current_patch_bin_for_ocr = cv2.resize(patch_bin_ocr, (scaled_w, scaled_h), interpolation=cv2.INTER_LANCZOS4)
    found_plate_text_for_patch = None; ocr_clean_for_subst = ""
    for lang_code in ['por', 'eng']:
        config = f'{custom_config_patch_base} {psm_patch} lang={lang_code}'
        try: text_ocr = pytesseract.image_to_string(current_patch_bin_for_ocr, config=config, timeout=tesseract_timeout_patch) 
        except RuntimeError: text_ocr = ""
        except Exception: text_ocr = ""
        ocr_clean = "".join(filter(str.isalnum, text_ocr)).upper()
        if lang_code == 'por': ocr_clean_for_subst = ocr_clean
        if len(ocr_clean) >= 6:
            plate_m = encontrar_placa_mercosul(ocr_clean);
            if plate_m: found_plate_text_for_patch = plate_m; break
            plate_a = encontrar_placa(ocr_clean)
            if plate_a: found_plate_text_for_patch = plate_a; break
    if found_plate_text_for_patch: return (1.0, found_plate_text_for_patch, color_patch_roi, current_patch_bin_for_ocr, (x,y,win_w,win_h))
    if len(ocr_clean_for_subst) == 7:
        primeiros_3=ocr_clean_for_subst[:3]; quarto_char=ocr_clean_for_subst[3]; ultimos_4=ocr_clean_for_subst[-4:]
        poss_subst = []
        if primeiros_3.isalpha():
            nums_old = substituir_letras_por_numeros_para_recorte(ultimos_4)
            for num_p in nums_old:
                if len(num_p)==4 and all(c.isdigit() for c in num_p):
                    cand_plate = primeiros_3+num_p
                    if encontrar_placa(cand_plate): poss_subst.append(cand_plate)
            if quarto_char.isdigit():
                seg_merc = ocr_clean_for_subst[3:7]
                parts_merc = gerar_possibilidades_mercosul_para_recorte(seg_merc)
                for part_m in parts_merc:
                    if len(part_m)==4 and part_m[0].isdigit() and part_m[1].isalnum() and part_m[2:].isdigit():
                        cand_plate = primeiros_3+part_m
                        if encontrar_placa_mercosul(cand_plate): poss_subst.append(cand_plate)
        if poss_subst: return (0.8, "\n".join(sorted(list(set(poss_subst)))), color_patch_roi, current_patch_bin_for_ocr, (x,y,win_w,win_h))
    return None

def encontrar_melhor_resultado_via_janela_deslizante(imagem_original_color):
    start_time = time.time(); (img_h, img_w) = imagem_original_color.shape[:2]
    imagem_cinza = cv2.cvtColor(imagem_original_color, cv2.COLOR_BGR2GRAY)
    window_configs = [(120,30,0.25,0.5),(160,40,0.25,0.5),(200,50,0.20,0.4),(240,60,0.20,0.4),(280,70,0.15,0.3),(320,80,0.15,0.3),(100,35,0.25,0.5),(150,50,0.25,0.5),(180,45,0.25,0.5)] 
    target_ocr_char_height = 30; custom_config_patch_base = r'-c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789 --oem 3'
    psm_patch = r'--psm 7'; tesseract_timeout_patch = 7 
    tasks = []
    for (win_w,win_h,step_fw,step_fh) in window_configs:
        if win_w > img_w or win_h > img_h or win_w < 30 or win_h < 10: continue
        step_x=max(10,int(win_w*step_fw)); step_y=max(5,int(win_h*step_fh))
        for y_coord in range(0,img_h-win_h+1,step_y):
            for x_coord in range(0,img_w-win_w+1,step_x):
                tasks.append((x_coord,y_coord,win_w,win_h,imagem_cinza,imagem_original_color,target_ocr_char_height,custom_config_patch_base,psm_patch,tesseract_timeout_patch))
    print(f"Generated {len(tasks)} tasks for sliding window.");
    if not tasks: return "No tasks generated",None,None,None
    candidate_plates=[]; num_processes=max(1,multiprocessing.cpu_count()-1 if multiprocessing.cpu_count()>1 else 1)
    print(f"Using {num_processes} processes for sliding window...")
    with multiprocessing.Pool(processes=num_processes) as pool: results=pool.map(process_patch_task,tasks)
    for result in results:
        if result: candidate_plates.append(result)
    end_time=time.time(); print(f"Sliding window took {end_time-start_time:.2f}s, found {len(candidate_plates)} candidates.")
    if not candidate_plates: return "No plate by sliding window",None,None,None
    candidate_plates.sort(key=lambda item:item[0],reverse=True); best_cand=candidate_plates[0]
    return best_cand[1],best_cand[2],best_cand[3],best_cand[4]

def exibir_e_salvar_resultado_janela_deslizante(*args):
    img_orig,best_color_patch,best_bin_patch,best_bbox,text_plate,out_dir,base_fname=args
    os.makedirs(out_dir,exist_ok=True); img_orig_with_rect=img_orig.copy()
    if best_bbox: x,y,w,h=best_bbox; cv2.rectangle(img_orig_with_rect,(x,y),(x+w,y+h),(0,255,0),3)
    if img_orig is not None: cv2.imwrite(os.path.join(out_dir,f"{base_fname}_01_original_rect.png"),img_orig_with_rect)
    if best_color_patch is not None: cv2.imwrite(os.path.join(out_dir,f"{base_fname}_02_best_color_patch.png"),best_color_patch)
    if best_bin_patch is not None: cv2.imwrite(os.path.join(out_dir,f"{base_fname}_03_best_bin_patch_ocr.png"),best_bin_patch)
    fig=plt.figure(figsize=(15,5)); plt.suptitle(f"Plate OCR (Sliding Window) for '{base_fname}':\n{text_plate}",fontsize=14)
    ax1=plt.subplot(1,3,1);
    if img_orig_with_rect.size>0: ax1.imshow(cv2.cvtColor(img_orig_with_rect,cv2.COLOR_BGR2RGB))
    ax1.set_title("1. Original + Found Region"); ax1.axis('off')
    ax2=plt.subplot(1,3,2);
    if best_color_patch is not None and best_color_patch.size>0: ax2.imshow(cv2.cvtColor(best_color_patch,cv2.COLOR_BGR2RGB))
    else: ax2.text(0.5,0.5,'No Patch',ha='center',va='center')
    ax2.set_title("2. Best Color Patch"); ax2.axis('off')
    ax3=plt.subplot(1,3,3);
    if best_bin_patch is not None and best_bin_patch.size>0: ax3.imshow(best_bin_patch,cmap='gray')
    else: ax3.text(0.5,0.5,'No Patch',ha='center',va='center')
    ax3.set_title("3. Binarized Patch for OCR"); ax3.axis('off')
    plt.tight_layout(rect=[0,0.03,1,0.93]); fig_path=os.path.join(out_dir,f"{base_fname}_00_fig_sliding_window.png"); plt.savefig(fig_path)
    print(f"Sliding window outputs saved in: {out_dir}"); 
    plt.close(fig) # Save and close

def detectar_placa_via_janela_deslizante(caminho_da_imagem, output_base_dir):
    imagem_original=cv2.imread(caminho_da_imagem)
    if imagem_original is None: print(f"Error reading image: {caminho_da_imagem}"); return
    base_fname=os.path.splitext(os.path.basename(caminho_da_imagem))[0]
    specific_output_dir=os.path.join(output_base_dir,base_fname)
    print("Starting sliding window detection (this can be slow)...")
    texto_placa,melhor_patch_cor,melhor_patch_bin,bbox = encontrar_melhor_resultado_via_janela_deslizante(imagem_original)
    print(f"Sliding window result: {texto_placa}")
    exibir_e_salvar_resultado_janela_deslizante(imagem_original,melhor_patch_cor,melhor_patch_bin,bbox,texto_placa,specific_output_dir,base_fname)

# --- Main Execution ---
if __name__ == "__main__":
    multiprocessing.freeze_support() 
    parser = argparse.ArgumentParser(description="License Plate OCR with selectable processing mode.")
    parser.add_argument('--mode', type=str, default='crop', choices=['crop', 'nocrop', 'sliding_window', 'all'],
                        help="Processing mode. 'crop' (MSER based), 'nocrop', 'sliding_window'. Default is 'crop'.")
    parser.add_argument('--images_folder', type=str, default='images',
                        help="Folder containing input images. Default is 'images'.")
    args = parser.parse_args()

    pasta_de_imagens_entrada = args.images_folder
    current_mode = args.mode

    if not os.path.isdir(pasta_de_imagens_entrada):
        print(f"Error: Input folder '{pasta_de_imagens_entrada}' does not exist.")
    else:
        lista_de_arquivos_de_imagem = [
            f for f in os.listdir(pasta_de_imagens_entrada)
            if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff'))
        ]
        if not lista_de_arquivos_de_imagem: print(f"No image files found in '{pasta_de_imagens_entrada}'.")
        else:
            modes_to_run = []
            if current_mode == 'all':
                modes_to_run = [
                    ('crop', "output_images_mser_crop_tuned", detectar_placa_com_mser_crop_tuned),
                    ('nocrop', "output_images_no_crop_v2", detectar_placa_sem_recorte),
                    ('sliding_window', "output_images_sliding_window_mp_v2", detectar_placa_via_janela_deslizante)
                ]
            elif current_mode == 'crop': 
                modes_to_run.append(('crop', "output_images_mser_crop_tuned", detectar_placa_com_mser_crop_tuned))
            elif current_mode == 'nocrop':
                modes_to_run.append(('nocrop', "output_images_no_crop_v2", detectar_placa_sem_recorte))
            elif current_mode == 'sliding_window':
                modes_to_run.append(('sliding_window', "output_images_sliding_window_mp_v2", detectar_placa_via_janela_deslizante))

            for nome_arquivo_imagem in lista_de_arquivos_de_imagem:
                caminho_completo_imagem = os.path.join(pasta_de_imagens_entrada, nome_arquivo_imagem)
                
                for mode_name, output_dir_name, detect_func in modes_to_run:
                    print(f"\nProcessing: {caminho_completo_imagem} (Mode: {mode_name})...")
                    OUTPUT_BASE_DIR_MODE = output_dir_name
                    os.makedirs(OUTPUT_BASE_DIR_MODE, exist_ok=True)
                    print(f"Output for this mode will be saved in subdirectories under: '{OUTPUT_BASE_DIR_MODE}'")
                    
                    try:
                        start_time_img = time.time()
                        detect_func(caminho_completo_imagem, OUTPUT_BASE_DIR_MODE)
                        end_time_img = time.time()
                        print(f"Finished processing {nome_arquivo_imagem} with mode '{mode_name}' in {end_time_img - start_time_img:.2f} seconds.")
                    except Exception as e:
                        print(f"ERROR processing {nome_arquivo_imagem} with mode '{mode_name}': {e}")
                        import traceback
                        traceback.print_exc()
                    plt.close('all') # Close any lingering matplotlib figures
            
            print("\nAll processing finished.")