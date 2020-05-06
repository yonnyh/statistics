import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
import statsmodels.api as statsmodels


m=1000
m0=500
mu=2

# def generate_data(rho=0):
# 	mean = np.append(np.zeros(m0), np.ones(m - m0) * mu)
# 	cov = np.ones((m, m)) * rho
# 	np.fill_diagonal(cov, 1)
# 	# np.random.seed(0)
# 	return np.random.multivariate_normal(mean, cov)


def generate_data(rho=0):
	mus = np.append(np.zeros(m0), mu * np.ones(m - m0))
	s = np.random.normal(0, 1, m)
	return rho * s[0] + (1 - rho) * s + mus


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
			v = len(r_idx[r_idx < m0])
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


def bh_histograms(n=5000, alpha=0.1):
	r, v, q = np.zeros(n), np.zeros(n), np.zeros(n)
	for rho in [0, 0.95]:
		for i in range(n):
			x = generate_data(rho=rho)
			p_vals = norm.sf(x)
			# rejected = fdr_bh(p_vals, alpha)
			rejected, _, _, _ = statsmodels.stats.multipletests(p_vals, alpha=alpha, method='fdr_bh')
			r[i] = max(np.count_nonzero(rejected), 1)
			v[i] = np.count_nonzero(rejected[:m0])
			# r[i] = max(len(rejected), 1)
			# v[i] = len(rejected[rejected < m0])

		plt.hist(r, bins=50)
		print(np.histogram(r, bins=50))
		plt.title(f"Histogram of R with rho={rho}")
		# plt.savefig(f"R_rho{str(rho).replace('.','')}")
		plt.show()

		plt.hist(v, bins=50)
		plt.title(f"Histogram of V with rho={rho}")
		# plt.savefig(f"V_rho{str(rho).replace('.','')}")
		plt.show()

		q = v / r
		plt.hist(q, bins=50)
		plt.title(f"Histogram of Q with rho={rho}")
		# plt.savefig(f"Q_rho{str(rho).replace('.','')}")
		plt.show()


def bh_graph(n=5000, alpha=0.1):
	rhos = np.arange(0, 1.05, 0.05)
	means = np.zeros(len(rhos))
	stds = np.zeros(len(rhos))

	for j, rho in enumerate(rhos):
		print(rho)
		q = np.zeros(n)
		for i in range(n):
			x = generate_data(rho=rho)
			p_vals = norm.sf(x)
			rejected = fdr_bh(p_vals, alpha)
			q[i] = len(rejected[rejected < m0]) / max(len(rejected), 1)
		means[j] = np.mean(q)
		stds[j] = np.std(q)

	plt.errorbar(rhos, means, stds, linestyle='None', marker='^')
	plt.xticks(rhos)
	plt.title("mean and standard deviation as a function of rho")
	plt.show()


def FDR():
	x = generate_data()
	z = np.linspace(-4, 4, 100)[:, np.newaxis]

	fdrs_mat = (np.broadcast_to(x, (z.size, x.size)) <= z)
	r = np.sum(fdrs_mat, axis=1)
	v = np.sum(fdrs_mat[:, :m0], axis=1)
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
	x = generate_data()
	# histogram_and_densities(x)
	bh_histograms()
	# bh_graph()