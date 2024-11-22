import math

import numpy as np
from scipy.fftpack import dct, idct
from PIL import Image

image_name = "river"

dct2 = lambda x: dct(dct(x.T, norm = 'ortho').T, norm = 'ortho')
idct2 = lambda x: idct(idct(x.T, norm = 'ortho').T, norm = 'ortho')

def create_blocks(array, n):
    blocks = []

    N, M = array.shape

    for i in range(0, N-n+1, n):
        for j in range(0, M-n+1, n):
            blocks.append(array[i:i+n, j:j+n])

    return blocks

def reconstruct_from_blocks(blocks, n, original_shape):
    N, M = original_shape
    
    reconstructed_array = np.zeros((N, M))
    
    block_index = 0
    
    for i in range(0, N-n+1, n):
        for j in range(0, M-n+1, n):
            reconstructed_array[i:i+n, j:j+n] = blocks[block_index]
            block_index += 1
    
    return reconstructed_array

def uniform_quantizer(data, num_levels, min_val=None, max_val=None):
    data = np.array(data)
    if min_val is None:
        min_val = np.min(data)
    
    if max_val is None:
        max_val = np.max(data)

    step_size = (max_val - min_val) / num_levels

    quantization_centers = np.linspace(min_val + step_size / 2, max_val - step_size / 2, num_levels)

    quantized_data = np.digitize(data, bins=np.linspace(min_val, max_val, num_levels + 1), right=False) - 1
    quantized_data = np.clip(quantized_data, 0, num_levels - 1)
    quantized_data = quantization_centers[quantized_data]

    return quantized_data, quantization_centers

def get_ac_terms(matrix):
    n = matrix.shape[0]
    result = []
    
    for diag in range(1, 2 * n - 1):
        if diag < n:
            row_start = diag
            col_start = 0
        else:
            row_start = n - 1
            col_start = diag - n + 1

        diagonal_elements = []
        row, col = row_start, col_start
        while row >= 0 and col < n:
            diagonal_elements.append(float(matrix[row, col]))
            row -= 1
            col += 1

        if diag % 2 == 1:
            diagonal_elements.reverse()

        result.extend(diagonal_elements)
    
    return result

def reconstruct_matrix(n, top_leftmost, zigzag_list):
    matrix = np.zeros((n, n), dtype=int)
    matrix[0, 0] = top_leftmost

    idx = 0

    for diag in range(1, 2 * n - 1):
        if diag < n:
            row_start = diag
            col_start = 0
        else:
            row_start = n - 1
            col_start = diag - n + 1

        diagonal_coords = []
        row, col = row_start, col_start
        while row >= 0 and col < n:
            diagonal_coords.append((row, col))
            row -= 1
            col += 1

        if diag % 2 == 1:
            diagonal_coords.reverse()

        for row, col in diagonal_coords:
            if idx < len(zigzag_list):
                matrix[row, col] = zigzag_list[idx]
                idx += 1
    
    return idct2(matrix)

for n in [4, 8, 16, 32, 64]:
    image = Image.open(image_name+".gif")
    image = image.convert('L')
    image_array = np.array(image)

    blocks = create_blocks(image_array, n=n)
    blocks = [dct2(block) for block in blocks]
    dc_terms = [float(block[0,0]) for block in blocks]

    quantized_dc_terms, quantization_centers_dc_terms = uniform_quantizer(dc_terms, 8)

    ac_terms = [get_ac_terms(block) for block in blocks]
    all_ac_terms = [item for block in ac_terms for item in block]
    L = min(all_ac_terms)
    H = max(all_ac_terms)

    l1 = math.floor((n*n - 1) / 10)
    first_ac_terms = [terms[:l1] for terms in ac_terms]
    second_ac_terms = [terms[l1:2*l1] for terms in ac_terms]
    third_ac_terms = [terms[2*l1:] for terms in ac_terms]

    first_ac_terms_quantized = [uniform_quantizer(terms, 4, min_val=L, max_val=H)[0] for terms in first_ac_terms]
    second_ac_terms_quantized = [uniform_quantizer(terms, 2, min_val=L, max_val=H)[0] for terms in second_ac_terms]
    third_ac_terms_quantized = [[0 for _ in range(len(terms))] for terms in third_ac_terms]

    reconstructed_ac_terms = [np.concatenate((first, second, third)) for first, second, third in zip(first_ac_terms_quantized, second_ac_terms_quantized, third_ac_terms_quantized)]

    reconstructed_blocks = [reconstruct_matrix(n, dc, ac) for dc, ac in zip(quantized_dc_terms, reconstructed_ac_terms)]

    reconstructed_array = reconstruct_from_blocks(reconstructed_blocks, n, image_array.shape)

    print("snr: ", 10 * np.log10 ( np.sum ( image_array ** 2) / np.sum (( image_array - reconstructed_array ) ** 2) ))

    normalized_array = (255 * (reconstructed_array - np.min(reconstructed_array)) / (np.max(reconstructed_array) - np.min(reconstructed_array))).astype(np.uint8)
    
    image = Image.fromarray(normalized_array, mode="L")

    image.save(f"{image_name}-{n}.png")
