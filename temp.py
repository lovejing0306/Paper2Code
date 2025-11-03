import numpy as np

def softmax(v):
	"""
	v shape: [b,s]
	"""
	max_value = np.max(v, axis=-1, keepdims=True)
	ss = v - max_value
	ss = np.exp(ss)
	res = ss / np.sum(ss, axis=-1, keepdims=True)
	return res
	

if __name__ == '__main__':
	a = np.random.rand(5,4)
	res = softmax(a)
	print(res)