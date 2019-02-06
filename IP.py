from numpy import linspace, zeros, cumsum, ones, where,meshgrid, shape
from numpy.random import rand, randn
from skimage.color import rgb2gray
from skimage.io import imshow, imread
from skimage.exposure import equalize_hist as histeq
from skimage.exposure import cumulative_distribution as cdf
from matplotlib.colors import ListedColormap as colormap
#from scipy.misc import imresize
from skimage.transform import resize as imresize
import matplotlib.pyplot as plt
from  matplotlib.pyplot import  xlabel, ylabel,title
from scipy.ndimage.filters import median_filter as medfil2d
from skimage.filters.rank import maximum
from skimage.filters.rank import gradient as grad
from skimage.filters import threshold_otsu as greythres
import numpy as np
import skimage.data as Images
from scipy.signal import convolve2d as filter2D
from numpy.fft import fft, fft2, ifft, ifft2, fftshift
from skimage.transform import rotate, radon, iradon, hough_line_peaks,hough_line
from skimage.morphology import disk,square,rectangle
from skimage.restoration import wiener
from skimage.restoration import deconvolution
from skimage.morphology import binary_erosion as imerode
from skimage.morphology import binary_dilation as imdilate
from skimage.morphology import binary_opening as imopen
from skimage.morphology import binary_closing  as imclose
from skimage.morphology import erosion as gs_imerode
from skimage.morphology import dilation as gs_imdilate
from skimage.morphology import opening as gs_imopen
from skimage.morphology import closing  as gs_imclose
from skimage.morphology import black_tophat  as tophat
from skimage.transform import warp, AffineTransform,ProjectiveTransform
from skimage import data, color
from scipy.interpolate import griddata
from skimage.filters import roberts, sobel, scharr, prewitt
from skimage.feature import canny, peak_local_max
from skimage.transform import hough_ellipse
from skimage.draw import ellipse_perimeter
from scipy.io import loadmat
from pylab import imshow, plot, subplot, subplots, show

def bitget(I,Ival):
    return (I>>Ival) % 2

def hough(img_bin, theta_res=1, rho_res=1):
    """
     Computes the Hough transform of an image
    """
    nR,nC = img_bin.shape
    theta = np.linspace(-90.0, 0.0, np.ceil(90.0/theta_res) + 1.0)
    theta = np.concatenate((theta, -theta[len(theta)-2::-1]))

    D = np.sqrt((nR - 1)**2 + (nC - 1)**2)
    q = np.ceil(D/rho_res)
    nrho = 2*q + 1
    rho = np.linspace(-q*rho_res, q*rho_res, nrho)
    H = np.zeros((len(rho), len(theta)))
    for rowIdx in range(nR):
        for colIdx in range(nC):
            if img_bin[rowIdx, colIdx]:
                for thIdx in range(len(theta)):
                    rhoVal = colIdx*np.cos(theta[thIdx]*np.pi/180.0) + \
                             rowIdx*np.sin(theta[thIdx]*np.pi/180)
                    rhoIdx = np.nonzero(np.abs(rho-rhoVal) == np.min(np.abs(rho-rhoVal)))[0]
                    H[rhoIdx[0], thIdx] += 1
    return H, theta,rho



def myImshow(I0,cmp="gray"):
    imshow(I0,aspect="auto",cmap=cmp)
def im2bw(Ig,level):
    S=np.copy(Ig)
    S[Ig > level] = 1
    S[Ig <= level] = 0
    return(S)
def dtfuv(m,n):
    """
    Computes the frequency matrices, used to construct frequency domain filters

    """
    m = m  # double size to detal with wrap round effects
    n = n
    u=linspace(0,m-1,m)
    v=linspace(0,n-1,n)

    idx = where(u > m/2)
    u[idx] = u[idx]-m

    idy = where(v > n/2)
    v[idy] = v[idy]-n

    V,U = meshgrid(v,u)
    return (V,U)

def fftfilt(f,H):
    """
    This function performs fourier domain filtering. I also to zero padding to prevent wrap around effects
    output filtered imaged
    f = input image
    H =  filter coeffienets
    """
    f = f.astype("double")
    I = fft2(f,(shape(H)[0],shape(H)[1]))
    I2 = ifft2(H*I)
    return I2[:shape(f)[0],:shape(f)[1]].real

def phantom (n = 256, p_type = 'Modified Shepp-Logan', ellipses = None):
        """
         phantom (n = 256, p_type = 'Modified Shepp-Logan', ellipses = None)

        Create a Shepp-Logan or modified Shepp-Logan phantom.

        A phantom is a known object (either real or purely mathematical)
        that is used for testing image reconstruction algorithms.  The
        Shepp-Logan phantom is a popular mathematical model of a cranial
        slice, made up of a set of ellipses.  This allows rigorous
        testing of computed tomography (CT) algorithms as it can be
        analytically transformed with the radon transform (see the
        function `radon').

        Inputs
        ------
        n : The edge length of the square image to be produced.

        p_type : The type of phantom to produce. Either
          "Modified Shepp-Logan" or "Shepp-Logan".  This is overridden
          if `ellipses' is also specified.

        ellipses : Custom set of ellipses to use.  These should be in
          the form
                [[I, a, b, x0, y0, phi],
                 [I, a, b, x0, y0, phi],
                 ...]
          where each row defines an ellipse.
          I : Additive intensity of the ellipse.
          a : Length of the major axis.
          b : Length of the minor axis.
          x0 : Horizontal offset of the centre of the ellipse.
          y0 : Vertical offset of the centre of the ellipse.
          phi : Counterclockwise rotation of the ellipse in degrees,
                measured as the angle between the horizontal axis and
                the ellipse major axis.
          The image bounding box in the algorithm is [-1, -1], [1, 1],
          so the values of a, b, x0, y0 should all be specified with
          respect to this box.

        Output
        ------
        P : A phantom image.

        Usage example
        -------------
          import matplotlib.pyplot as pl
          P = phantom ()
          pl.imshow (P)

        References
        ----------
        Shepp, L. A.; Logan, B. F.; Reconstructing Interior Head Tissue
        from X-Ray Transmissions, IEEE Transactions on Nuclear Science,
        Feb. 1974, p. 232.

        Toft, P.; "The Radon Transform - Theory and Implementation",
        Ph.D. thesis, Department of Mathematical Modelling, Technical
        University of Denmark, June 1996.

        """

        if (ellipses is None):
                ellipses = _select_phantom (p_type)
        elif (np.size (ellipses, 1) != 6):
                raise AssertionError ("Wrong number of columns in user phantom")

        # Blank image
        p = np.zeros ((n, n))

        # Create the pixel grid
        ygrid, xgrid = np.mgrid[-1:1:(1j*n), -1:1:(1j*n)]

        for ellip in ellipses:
                I   = ellip [0]
                a2  = ellip [1]**2
                b2  = ellip [2]**2
                x0  = ellip [3]
                y0  = ellip [4]
                phi = ellip [5] * np.pi / 180  # Rotation angle in radians

                # Create the offset x and y values for the grid
                x = xgrid - x0
                y = ygrid - y0

                cos_p = np.cos (phi)
                sin_p = np.sin (phi)

                # Find the pixels within the ellipse
                locs = (((x * cos_p + y * sin_p)**2) / a2
              + ((y * cos_p - x * sin_p)**2) / b2) <= 1

                # Add the ellipse intensity to those pixels
                p [locs] += I

        return p


def _select_phantom (name):
        if (name.lower () == 'shepp-logan'):
                e = _shepp_logan ()
        elif (name.lower () == 'modified shepp-logan'):
                e = _mod_shepp_logan ()
        else:
                raise ValueError ("Unknown phantom type: %s" % name)

        return e


def _shepp_logan ():
        #  Standard head phantom, taken from Shepp & Logan
        return [[   2,   .69,   .92,    0,      0,   0],
                [-.98, .6624, .8740,    0, -.0184,   0],
                [-.02, .1100, .3100,  .22,      0, -18],
                [-.02, .1600, .4100, -.22,      0,  18],
                [ .01, .2100, .2500,    0,    .35,   0],
                [ .01, .0460, .0460,    0,     .1,   0],
                [ .02, .0460, .0460,    0,    -.1,   0],
                [ .01, .0460, .0230, -.08,  -.605,   0],
                [ .01, .0230, .0230,    0,  -.606,   0],
                [ .01, .0230, .0460,  .06,  -.605,   0]]

def _mod_shepp_logan ():
        #  Modified version of Shepp & Logan's head phantom,
        #  adjusted to improve contrast.  Taken from Toft.
        return [[   1,   .69,   .92,    0,      0,   0],
                [-.80, .6624, .8740,    0, -.0184,   0],
                [-.20, .1100, .3100,  .22,      0, -18],
                [-.20, .1600, .4100, -.22,      0,  18],
                [ .10, .2100, .2500,    0,    .35,   0],
                [ .10, .0460, .0460,    0,     .1,   0],
                [ .10, .0460, .0460,    0,    -.1,   0],
                [ .10, .0460, .0230, -.08,  -.605,   0],
                [ .10, .0230, .0230,    0,  -.606,   0],
                [ .10, .0230, .0460,  .06,  -.605,   0]]
