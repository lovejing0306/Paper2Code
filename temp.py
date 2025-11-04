import numpy as np

def main(image, kernel_size):
	h, w = image.shape
	pad_h = h + kernel_size // 2 * 2
	pad_w = w + kernel_size // 2 * 2
	
	image_pad = np.zeros((pad_h, pad_w))
	image_pad[kernel_size//2:kernel_size//2+h, kernel_size//2: kernel_size//2+w] = image
	
	image_res = np.zeros_like(image)
	
	for i in range(pad_h-kernel_size+1):
		for j in range(pad_w-kernel_size+1):
			res = np.sum(image_pad[i:i+kernel_size, j:j+kernel_size]) // (kernel_size * kernel_size)
			image_res[i][j] = res
	
	return image_res


if __name__ == '__main__':
	img = np.array(
        [
            [1.0, 2.0, 3.0],
            [4.0, 5.0, 6.0],
            [7.0, 8.0, 9.0],
        ],
        dtype=float,
    )

	res = main(img, kernel_size=3)
	print(res)