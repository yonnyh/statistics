import numpy as np


def fdr_bh(p_val_list, alpha):
	m = p_val_list.size
	sorted_idxs = np.argsort(p_val_list)
	sorted_idx_rejected_bool = p_val_list[sorted_idxs] <= (alpha * np.arange(1, m+1) / m)
	if not np.all(sorted_idx_rejected_bool):
		k = np.where(sorted_idx_rejected_bool == False)[0][0] - 1  # max idx of acception continuously
		rejected_idxs = sorted_idxs[:k+1]
	else:
		rejected_idxs = sorted_idxs
	return rejected_idxs


if __name__ == '__main__':
	pval_list = np.linspace(0, 0.1, 10)
	# pval_list = np.linspace(0, 0.06, 10)
	# pval_list = np.linspace(0, 0.05, 10)
	np.random.seed(0)
	pval_list = np.random.permutation(pval_list)
	alpha = 0.05
	idxs = fdr_bh(pval_list, alpha)
	print(pval_list[idxs])
