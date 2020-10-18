import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

class HybridImage():
    """This class creates an hybrid image by combining a low pass filtered
        version of Image A and a high pass filtered version of Image B.

        Attributes:
            imageA (numpy array): numpy array representing the resized image A
            imageB (numpy array): numpy array representing the resized image B
            imageA_lfreq (numpy array): numpy array representing the low frequency filtered image A
            imageB_hfreq (numpy array): numpy array representing the high frequency filtered image B
            hybrid_image (numpy array): numpy array representing the hybrid image

        Parameters:
            imageA_path (str): path of the first image. The low frequency content
                of this image will be extracted
            imageB_path (str): path of the second image. The high frequency content
                of this image will be extracted
            kernel_size (int): a positive odd integer which will be gaussian
                filter size for low pass filter
    """

    def __init__(self, imageA_path, imageB_path, kernel_size):
        imageA = Image.open(imageA_path)
        imageB = Image.open(imageB_path)
        img_height = min(imageA.height, imageB.height)
        img_width = min(imageA.width, imageB.width)

        #make both images to be of equal dimensions
        self.imageA = imageA.resize((img_width, img_height))
        self.imageB = imageB.resize((img_width, img_height))
        self.imageA = np.array(self.imageA)
        self.imageB = np.array(self.imageB)

        self.imageA_lfreq = None
        self.imageB_hfreq = None
        self.filter = None
        self.hybrid_image = None
        self.__set_filter(kernel_size)
        self.__apply_filter()

    def __set_filter(self, kernel_size):
        """Funtion to compute a 2D gaussian kernel for the given kernel size. The
            kernel size can be used to tune the mixing in the hybrid image

            Args:
                kernel_size (int): a positive odd integer which will be gaussian
                 filter size for low pass filter

            Returns:
                filters (dict): dictionary containing a 2D numpy array representing
                 the low pass and high pass filters (keys: 'low', 'high')
        """

        if kernel_size % 2 == 0:
            kernel_size+=1 # make kernel size odd number
        sigma = int(kernel_size/2)/3 # keep the standard deviations of the filter such that 6*sigma = kernel size

        gaus_1D = []
        for i in np.arange(0, kernel_size, 1):
            x = (i - (kernel_size-1)/2)
            gaus_x = np.exp(-0.5*(x / sigma) ** 2)
            gaus_1D.append(gaus_x)

        gaus_1D = np.array(gaus_1D) / np.array(gaus_1D).sum() # scale to make the sum of gaus_1D = 1
        gaus_1D = gaus_1D.reshape(-1,1)
        self.filter = np.dot(gaus_1D, gaus_1D.T)

    def __apply_filter(self):
        """Function to apply the filter and update the low frequency, high frequency,
            and the hybrid image attributes of the object.

            Parameters:
                None

            Returns:
                None
        """

        self.imageA_lfreq = self.__conv_2d(self.imageA, self.filter)
        self.imageB_hfreq = self.imageB - self.__conv_2d(self.imageB, self.filter)
        self.imageB_hfreq = np.clip(self.imageB_hfreq - 150, 0, 255)
        self.hybrid_image = np.clip(self.imageA_lfreq + self.imageB_hfreq, 0, 255)

    def __conv_2d(self, image, filter):
        """Function does a 2D convolution operation on each channel of the given image
            with the passed filter. Zero padding is added across width and height.

            Parameters:
                image (numpy array): image as a numpy array of dimension H x W x C
                filter (numpy array): a 2D numpy array

            Returns:
                filtered_image (numpy array): image as a numpy array of dimension H x W x C after applying
                    the filter.
        """

        pad_len_x = filter.shape[1]//2
        pad_len_y = filter.shape[0]//2
        image_padded = np.zeros((image.shape[0]+2*pad_len_y,
                                 image.shape[1]+2*pad_len_x, 3), dtype=float)
        image_padded[pad_len_y: image.shape[0] + pad_len_y, pad_len_x: image.shape[1] + pad_len_x] = image.copy()

        filtered_image = np.zeros((image_padded.shape[0], image_padded.shape[1],
                                   image_padded.shape[2]), dtype=float)
        for y in range(pad_len_y, image.shape[0] + pad_len_y, 1):
            for x in range(pad_len_x, image.shape[1] + pad_len_x, 1):
                conv_sum = (image_padded[y-pad_len_y: y+pad_len_y+1, x-pad_len_x: x+pad_len_x+1] * filter.reshape(filter.shape[0], filter.shape[1],1)).sum(axis=1).sum(axis=0)
                filtered_image[y,x,:] = conv_sum
        filtered_image = filtered_image[pad_len_y: image.shape[0] + pad_len_y, pad_len_x: image.shape[1] + pad_len_x]
        filtered_image = np.clip(filtered_image.astype(int), 0, 255)
        return filtered_image

    def show_originals(self):
        """Function to show the orignal images side by side.

            Parameters:
                None

            Returns:
                None
        """

        fig, axes = plt.subplots(1,2)
        axes[0].imshow(self.imageA)
        axes[1].imshow(self.imageB)
        axes[0].set_title("Image A")
        axes[1].set_title("Image B")
        for ax in axes:
            ax.set_xticks([])
            ax.set_yticks([])

    def show_filtered(self):
        """Function to filtered images side by side.

            Parameters:
                None

            Returns:
                None
        """

        fig, axes = plt.subplots(1,2)
        axes[0].imshow(self.imageA_lfreq)
        axes[1].imshow(self.imageB_hfreq)
        axes[0].set_title("Image A\n(low pass filtered)")
        axes[1].set_title("Image B\n(high pass filtered)")
        for ax in axes:
            ax.set_xticks([])
            ax.set_yticks([])

    def show_hybrid(self):
        """Function to show the hybrid image.

            Parameters:
                None

            Returns:
                None
        """

        fig, axes = plt.subplots(1,1)
        axes.imshow(self.hybrid_image)
        axes.set_title("Hybrid Image")
        axes.set_xticks([])
        axes.set_yticks([])
