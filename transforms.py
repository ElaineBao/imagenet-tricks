import numpy as np
from PIL import Image

class PCANoise(object):
    """Add PCA based noise.
    Parameters
    ----------
    alphastd : float
        Noise level
    """
    def __init__(self, alphastd):
        self.alphastd = alphastd
        self.eigval = np.array([55.46, 4.794, 1.148])
        self.eigvec = np.array([[-0.5675, 0.7192, 0.4009],
                           [-0.5808, -0.0045, -0.8140],
                           [-0.5836, -0.6948, 0.4203]])

    def __call__(self, src):
        """Augmenter body"""
        alpha = np.random.normal(0, self.alphastd, size=(3,))
        rgb = np.dot(self.eigvec * alpha, self.eigval)
        src += rgb
        pil_img = Image.fromarray(src.astype('uint8'))
        return pil_img

if __name__ == '__main__':
    test_img = 'test.jpg'
    pca_noise_param = 0.1
    transformer = PCANoise(pca_noise_param)

    test_img_array = Image.open(test_img)
    target = transformer(test_img_array)
    target.save('test-target.jpg')