import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

m=1000
m_0=500
mu=2

def generate_data(rho=0):
	mean = np.concatenate((np.zeros(m_0), np.ones(m - m_0) * mu))
	cov = np.ones((m, m)) * rho
	np.fill_diagonal(cov, 1)
	# np.random.seed(0)
	return np.random.multivariate_normal(mean, cov)


def histogram_and_densities(x, pi_0=0.5):
	plt.hist(x, bins=100, density=True)
	x_axis = np.linspace(-4, 4, 1000)
	f0 = norm.pdf(x_axis, 0, 1)
	f1 = norm.pdf(x_axis, 2, 1)
	plt.plot(x_axis, pi_0 * f0, label="F0")
	plt.plot(x_axis, (1-pi_0) * f1, label="F1")
	plt.plot(x_axis, pi_0 * f0 + (1-pi_0) * f1, label=f"{pi_0}*F0 + {(1-pi_0)}*F1")
	plt.legend()
	plt.show()


def fdr(c=2):
	c_arr = np.linspace(-4, 4, 100)
	for c in c_arr:
		iters = 10
		fdrs = np.zeros(iters)
		for i in range(iters):
			x = generate_data()
			r_idx = np.where(np.logical_and(np.logical_and(x > -4, x < 4), x>=c))[0]
			r = len(r_idx)
			v = len(r_idx[r_idx < m_0])
			fdrs[i] = v / r
		print(np.mean(fdrs))


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

def bh_histograms(n=100, alpha=0.1):
	r_arr, v_arr, q_arr = np.zeros(n), np.zeros(n), np.zeros(n)
	for rho in [0, 0.95]:
		for i in range(n):
			print(i)
			x = generate_data(rho=rho)
			rejected_idx = fdr_bh(x, alpha)
			r_arr[i] = len(rejected_idx)
			v_arr[i] = len(rejected_idx[rejected_idx < m_0])
			q_arr[i] = len(rejected_idx[rejected_idx < m_0]) / len(rejected_idx) if len(rejected_idx) else np.nan
		plt.hist(r_arr, bins=100)
		plt.title(f"histogram of R with rho={rho}")
		plt.show()
		plt.hist(v_arr, bins=100)
		plt.title(f"histogram of V with rho={rho}")
		plt.show()
		plt.hist(q_arr, bins=100)
		plt.title(f"histogram of Q with rho={rho}")
		plt.show()

def bh_graph(n=10, alpha=0.1):
	rhos = np.arange(0, 1.05, 0.05)
	means = np.zeros(len(rhos))
	stds = np.zeros(len(rhos))

	for j, rho in enumerate(rhos):
		print(rho)
		q_arr = np.zeros(n)
		for i in range(n):
			x = generate_data(rho=rho)
			rejected_idx = fdr_bh(x, alpha)
			q_arr[i] = len(rejected_idx[rejected_idx < m_0]) / len(rejected_idx) if len(rejected_idx) else np.nan
		means[j] = np.mean(q_arr)
		stds[j] = np.std(q_arr)

	plt.plot(rhos, means, label="means")
	plt.plot(rhos, stds, label="standard deviations")
	plt.title("means and standard deviation as a function of rho")
	plt.legend()
	plt.show()


def FDR():
	x = generate_data()
	z = np.linspace(-4, 4, 100)[:, np.newaxis]

	fdrs_mat = (np.broadcast_to(x, (z.size, x.size)) <= z)
	r = np.sum(fdrs_mat, axis=1)
	v = np.sum(fdrs_mat[:, :m_0], axis=1)
	fdrs = np.divide(v, r, where=r > 0)

	plt.plot(z, fdrs)
	plt.show()






if __name__ == '__main__':
	# pval_list = np.linspace(0, 0.1, 10)
	# # pval_l	ist = np.linspace(0, 0.06, 10)
	# # pval_list = np.linspace(0, 0.05, 10)
	# np.random.seed(0)
	# pval_list = np.random.permutation(pval_list)
	# alpha = 0.05
	# idxs = fdr_bh(pval_list, alpha)
	# print(pval_list[idxs])

	# q5
	# x = generate_data()
	# histogram_and_densities(x)
	# fdr()
	# bh_histograms()
	bh_graph()