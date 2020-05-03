import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm


def generate_data(m=1000, m_0=500, mu=2, ro=0):
	mean = np.concatenate((np.zeros(m_0), np.ones(m - m_0) * mu))
	cov = np.ones((m, m)) * ro
	np.fill_diagonal(cov, 1)
	return np.random.multivariate_normal(mean, cov)


def histogram_and_densities(x, pi_0=0.5):
	plt.hist(x, bins=100, density=True)
	x_axis = np.linspace(-5, 5, 1000)
	f0 = norm.pdf(x_axis, 0, 1)
	f1 = norm.pdf(x_axis, 2, 1)
	plt.plot(x_axis, pi_0 * f0)
	plt.plot(x_axis, (1-pi_0) * f1)
	plt.plot(x_axis, pi_0 * f0 + (1-pi_0) * f1)
	plt.show()




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
	# pval_list = np.linspace(0, 0.1, 10)
	# # pval_list = np.linspace(0, 0.06, 10)
	# # pval_list = np.linspace(0, 0.05, 10)
	# np.random.seed(0)
	# pval_list = np.random.permutation(pval_list)
	# alpha = 0.05
	# idxs = fdr_bh(pval_list, alpha)
	# print(pval_list[idxs])
	x = generate_data()
	histogram_and_densities(x)