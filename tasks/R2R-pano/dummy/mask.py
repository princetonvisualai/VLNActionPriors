import numpy as np
import pdb

def get_sentence_masks(reset_indices):
    max_num_sentences = 5
    sent_end_mat = np.zeros((reset_indices.shape[0], max_num_sentences+1), dtype=np.uint8)
    j_counter = np.ones(reset_indices.shape[0], dtype=np.uint8)
    rows, cols = reset_indices.nonzero()
    for r, c in zip(rows, cols):
        sent_end_mat[r, j_counter[r]] = c+1
        j_counter[r] += 1
    sent_end_mat[sent_end_mat == 0] = reset_indices.shape[1]
    sent_end_mat[:, 0] = 0
    print('Sentence End Matrix')
    print(sent_end_mat)
    sentence_masks = []
    for sent_ind in range(max_num_sentences):
        sentence_mask = np.zeros(reset_indices.shape)
        for instr_ind in range(reset_indices.shape[0]):
            sentence_mask[instr_ind, sent_end_mat[instr_ind,sent_ind]:sent_end_mat[instr_ind,sent_ind+1]] = 1
        sentence_masks.append(sentence_mask)
    return sentence_masks
 
if __name__ == "__main__":
    rand_ints = np.random.randint(5, size=(5,10))
    reset_indices = (rand_ints == 1).astype(float)
    print('Reset Indices')
    print(reset_indices)
    sentence_masks = get_sentence_masks(reset_indices)
    for i, sentence_mask in enumerate(sentence_masks):
        print('Sentence Mask {}'.format(i))
        print(sentence_mask)

