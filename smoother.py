import math
import itertools


def smooth(datain, dataout=None, window=None, padding=None, repeat=1, ftype='butter', first=0, last=-1):
    """
            Smooth the data by applying a filter twice, once forward, once backward.

            Window: 
                            Distance (in sample count) at wich the influence of a point on its
                            Neibourgh drop to less than 5%

                            Increasing the window allow to better filter out noise.
                            The compromise is that peak or sharp feature are rounded off


            Filter type:

                            There are natural trade off between 
                                    - noise removal, 
                                    - ability to follow sharp features
                                    - tendency to oscillate
                                    - overshoots

                            This function allow to choose a different set of trade off by prodividing some
                            predefined filter type. Here we approximately order filter by aggressivity
                            (better peak-following but with more oscillations)

                            0: Bessel
                            1: Butterworth (default)
                            2: Chebyshev Type II
                            3: Elliptic

                            Note: 
                            - Increasing aggressivity is mostly usefull with large window

                            - Many characteristics will changes all at once.

                            - By default filter design is tweaked to have similar output
                              for a given windows size. However it's possible that one has to find the
                              best window size for each filter type

            Repeat: 
                            Apply the filter multiple time. (Filter design is done only once)

            Padding:
                            There is some transient effect for the filter to properly "Warm up".
                            We deal with this effect by doing the warming up in a separate buffer space

                            Padding parameter control the size of that space.
                            Default value for padding is "automatic" and about 1 window size
                            Set padding to 0 to observe transient effect


            Why ?

                    * Implement the filter with desired cut-off for the user

                    * Cut-off specified in an easy unit (# of sample)

                    * Self contained, only use addition and multiplication
                      in data crunching phase. (and some divisions in design phase)

                    * Forward-Backward filtering
                            - Minimize phase related deformations
                            - Allign feature of result to feature of source
                            - Make a stronger filter (double the order)

                    * Padding is implemented to minimize "warming-up" effect
                      (filter transient response)


    """

    if last < 0:
        last = len(datain) + last

    if window is None:
        window = int(min(max(0.05 * len(datain), 5), 50))

    if dataout is None:
        dataout = datain

    filtercoeff, rtime = designFilter(window, ftype)

    if padding is None:
        padding = int(min(rtime, len(datain) / 2))

    elif padding < 3:
        padding = 0

    for k in xrange(1, repeat + 1):

        datasource = datain if k == 1 else dataout

        if padding > 0:

            #
            # Drive filterpass to do forward-backward fitlering with buffer.
            #
            # 1) Start in left buffer
            # 2) Chain in main data
            # 3) Chain in rigth buffer
            #
            # 4) Chain backward pass in filtered rigth buffer
            # 5) Chain to backward pass of main data
            #
            # buffer space is a odd extention of mirrored data
            # (slope sign is unchanged)
            #
            # buffer is allocated only once. The left buffer is discarded as soon as we have initial conditions
            # This free the space to build rigth buffer.

            # Create left bufferspace
            pivot = 2 * datasource[first]
            bufferspace = [pivot - datasource[i]
                           for i in xrange(first + padding, first - 1, -1)]

            # Process left buffer to get initial conditions
            initialz = filterpass(bufferspace, None, 0,
                                  padding, filtercoeff, None)

            # Create rigth bufferspace
            pivot = 2 * datasource[last]
            bufferspace = [pivot - datasource[i]
                           for i in xrange(last, last - padding - 1, -1)]

            # Process Forward Pass with initial conditions as calculated in left buffer
            initialz = filterpass(datasource, dataout,
                                  first, last, filtercoeff, initialz)

            #Continue in rigth-bufferspace
            initialz = filterpass(bufferspace, bufferspace,
                                  0, padding, filtercoeff, initialz)

            # Process rigth-bufferspace backward (chain as even extention)
            initialz = filterpass(
                bufferspace, None, padding, 0, filtercoeff, initialz)

            # Process Backward Pass with initial condition calculated from rigth buffer
            filterpass(dataout, dataout, last, first, filtercoeff, initialz)

        else:

            # Start in left buffer
            initialz = filterpass(datasource, dataout,
                                  first, last, filtercoeff, None)

            #Reverse in rigth-bufferspace
            filterpass(dataout, dataout, last, first, filtercoeff, initialz)


def decimate(datain, n=10):
    """
            Keep only 1 sample out of n
            Filtering is used so sample that is kept is representative of it's neibourghood
            Usefull for ploting large amount of data
    """

    tmp = [0.0] * len(datain)
    smooth(datain, tmp, 0.8 * n, ftype='bessel')
    size = int(math.floor(float(len(tmp)) / n))
    dataout = [tmp[n * (i - 1)] for i in xrange(1, size)]
    return dataout


def filterpass(datasource, dataout, first, last, filtercoeff, initialz=None):
    """

            Data Processing loop

            Direct form II transposed implementation of a second order filter.

            * Can work in place if  datasource = dataout

            * If first > last, do a backward pass

            * Can be used on chunk data
                    - Accept initial conditions
                    - Return initial conditions for next element

            Act as a selector between almost identical variants
             - 2nd order or 3rd order filter
             - optional write to output.

            This allow to handle optional once, instead of testing for each elements.

    """

    b0, b1, b2, a1, a2, w = filtercoeff
    w_ = 1 - w

    step = 1 if last > first else -1

    if initialz is None:

        # initial conditions
        x0 = datasource[first]
        x1 = datasource[first + step]

        z0 = (1 - b0) * x0
        z1 = (1 - b0 * w_) * x1 + (a1 - b1 - b0 * w) * x0
        xi = x0

    else:
        z0, z1, xi = initialz

    # Below is 3 small variantions of the same code

    if dataout is None:

        # Padding
        for i in xrange(first, last + step, step):

            # First order filter
            xi = w * xi + w_ * datasource[i]

            # Cascade with second order direct form II transposed
            yi = b0 * xi + z0
            z0 = b1 * xi + z1 - a1 * yi
            z1 = b2 * xi - a2 * yi

            # No Write

    elif w == 0:

        # Second order filter
        for i in xrange(first, last + step, step):

            # No cascaded first order
            xi = datasource[i]

            # Second order direct form II
            yi = b0 * xi + z0
            z0 = b1 * xi + z1 - a1 * yi
            z1 = b2 * xi - a2 * yi

            dataout[i] = yi

    else:

        # Third order filter
        for i in xrange(first, last + step, step):

            # First order filter
            xi = w * xi + w_ * datasource[i]

            # Cascade with second order direct form II
            yi = b0 * xi + z0
            z0 = b1 * xi + z1 - a1 * yi
            z1 = b2 * xi - a2 * yi

            dataout[i] = yi

    return (z0, z1, xi)


def bilinear(s_real, s_imag, Ts):
    """
    Bilinear Transform of a complex root


    The continous (s-domain) response is discretized using bilinear transform

                            z = (2+Ts*s)/(2-Ts*s)  

    Where s is the complex number s_real + s_imag*i

    """

    den = (s_imag * s_imag + s_real * s_real) * Ts * Ts - 4 * s_real * Ts + 4
    z_real = - (4 * Ts * s_real - 8) / den - 1
    z_imag = (4 * Ts * s_imag) / den

    return (z_real, z_imag)


def bilinearExpandPair(s_real, s_imag, Ts):
    """
    Polynominal coefficients of a transformed complex conjugate pair

    Input:  - Real and imaginary part of a complex conjugate pair (s-domain)
                    - Sampling frequency

                    ( sreal + simag*i )(sreal - simag*i )

    Output:

                    - Coefficients a1,a1 of  the z-domain polynominal:
                    z^-2 + a1*z^-1 + a2



    Equivalent to:

    1) Get bilinear transform

            z_real, z_imag = bilinear(s_real,s_imag)

    2) Find the polynominal coefficient corresponding to those roots

            #poly =  (z - z_real+z_imag*i)(z - z_real-z_imag*i)
            #poly = z^-2 + a1*z^-1 + a2

            a1 = -2*z_real
            a2 = z_real*z_real + z_imag*z_imag

            return a1,a2

    """

    den = (s_imag * s_imag + s_real * s_real) * Ts * Ts - 4 * s_real * Ts + 4
    a1 = (2 * (4 * Ts * s_real - 8)) / den + 2
    a2 = (8 * Ts * s_real) / den + 1

    return (a1, a2)


def designFilter(cutoff, ftype, mode=1):
    """

            mode = 1: Cutoff is window size

            mode = 2: Cutoff is angular frequency 
                                    (to be prewrapped for bilinear transform)

            mode = 3: Cutoff is prewrapped angular frequency


            NOTES:
            mode 2: 
            - Is the more mathematicaly correct to generate exact design frequency
            - Design frequency does not have the same meaning depending of filter type
              They all defind the cut-off edge, but different interest points:

                    * Attenuation of 3db for butterworth
                    * Attenuation of passband ripple for elliptic
                    * Attenuation of stopband ripple for cheby2

            mode 1:
            - For that reason after the filter prototype has been designed,
              They where manually tweaked to have similar result for a given window size

            mode 3:
            - Raw prewrapped frequency, to evaluate some characteristics of the filter.




            Design Procedure:

            1) Start with an analog filter prototype

            2) Scale prototype to desired (prewrapped) frequency

            3) Use bilinear transform to discretize filter
               (This include adding "nbpoles-nbzeros" zeros at -1)

            4) Estimate transient time for buffer

            5) Adjust zero for unit DC gain


    """

    fs = 1  # Samping frequency
    Ts = 1  # Sampling window = 1/fs

    if mode == 2:

        #window = 1/Wn

        # Exact frequency space design
        Wn = 2 * fs * math.tan(cutoff * math.pi / fs)

    elif mode == 3:

        Wn = cutoff

    else:

        window = cutoff

    if ftype == 'bessel' or ftype == 'safe':

        # Complex pair
        psr = -0.745640385848077
        psi = 0.711366624972835

        # Estimate
        if mode == 1:
            Wn = 2.2 * 4.76377476752 / window
            #Wn = 4.76377476752/window
            rtime = window
        else:
            rtime = -4 / (Wn * psr)

        (a1, a2) = bilinearExpandPair(Wn * psr, Wn * psi, Ts)

        # 3rd real
        pw = -0.941600026533207
        w, _ = bilinear(Wn * pw, 0, Ts)

        # Numerator
        b0 = 0.0
        b1 = 0.0
        b2 = 1.0

        #rtime = window

    elif ftype == 'cheby2' or ftype == 'aggresive':

        # Order = 3, R = -20 Db

        # Complex pole pair
        psr = - 0.275968057981812
        psi = - 0.628402822657552

        # Estimate
        if mode == 1:
            Wn = 8.76988211199 / window
            #Wn = 1.8*8.76988211199/window
            rtime = window
        else:
            rtime = -4 / (Wn * psr)

        (a1, a2) = bilinearExpandPair(Wn * psr, Wn * psi, Ts)

        # 3rd real
        pw = - 0.853447460541388
        w, _ = bilinear(Wn * pw, 0, Ts)

        # Complex zero pair
        zsr = 0.0
        zsi = 1.154700538379252

        # Polynominal coefficient (numerator)
        b0 = 1
        (b1, b2) = bilinearExpandPair(Wn * zsr, Wn * zsi, Ts)

    elif ftype == "elliptic":

        # Order = 3, Rp=-1 Db,  Rs = -20 Db

        # Complex pole pair
        psr = -0.161497094869716
        psi = 1.003344078658930

        # Estimate
        if mode == 1:
            Wn = 0.8 * 11.7583617629 / window
            #Wn = 11.7583617629/window
            rtime = window
        else:
            rtime = -2 / (Wn * psr)

        (a1, a2) = bilinearExpandPair(Wn * psr, Wn * psi, Ts)

        # 3rd pole: real
        pw = -0.643572996301072
        w, _ = bilinear(Wn * pw, 0, Ts)

        # Complex zero pair
        zsr = 0.0
        zsi = 1.439910668065716

        # Polynominal coefficient (numerator)
        b0 = 1
        (b1, b2) = bilinearExpandPair(Wn * zsr, Wn * zsi, Ts)

    else:  # default: butterworth

        # Complex pole pair
        psr = -0.5
        psi = 0.866025403784439  # math.sqrt(3)/2;

        # Result from impulse test
        #Wn = 6/Window
        # 5.9232927323236053

        # Estimate
        if mode == 1:
            Wn = 5.92191076937 / window
            rtime = window
        else:
            rtime = -4 / (Wn * psr)

        (a1, a2) = bilinearExpandPair(Wn * psr, Wn * psi, Ts)

        pw = -1.0
        w, _ = bilinear(Wn * pw, 0, Ts)

        # Numerator
        b0 = 1.0
        b1 = 2.0
        b2 = 1.0

    # Adjust the zeros so we have a unit DC gain

    dcadjust = (1 + a1 + a2) / (b0 + b1 + b2)  # = 1/dcgain

    # Multiply the zeros by adjustment factor
    b0 *= dcadjust
    b1 *= dcadjust
    b2 *= dcadjust

    # print Wn

    return ((b0, b1, b2, a1, a2, w), rtime)


def test_timeconstant(ftype):
    """
            Wn = slope/Window

            Window = 95 percent attenuation in impulse response
    """

    freq = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    #invwindow = [0.0]*len(y)
    Sfw = 0
    Sww = 0

    for i in xrange(0, len(freq)):

        f = freq[i]

        num = test_impulse(f, ftype)
        #num,graph = test_impulse(f,ftype)
        # plot(graph)
        # xlim(0,100)
        #x[i] = 1.0/num
        w = 1.0 / num

        Sfw += f * w
        Sww += w * w

    slope = Sfw / Sww
    return slope


def test_impulse(Wn, ftype):

    filtercoeff, rtime = designFilter(Wn, ftype, 3)
    b0, b1, b2, a1, a2, w = filtercoeff
    w_ = 1 - w

    # Impulse conditions
    x0 = 1
    x1 = 1

    z0 = (1 - b0) * x0
    z1 = (1 - b0 * w_) * x1 + (a1 - b1 - b0 * w) * x0
    xi = x0

    #tmp = [0.0]*500

    tresh = 0.05
    count0 = 50
    count = count0

    i = 0

    # Third order filter
    while 1:

        # First order filter
        xi = w * xi  # + w_*0.0 #Feeed with 0

        # Cascade with second order direct form II
        yi = b0 * xi + z0
        z0 = b1 * xi + z1 - a1 * yi
        z1 = b2 * xi - a2 * yi

        #tmp[i] = yi
        i = i + 1

        if abs(yi) <= tresh:
            count = count - 1

            if count == 0:
                return i - count0  # (i-count0, tmp)
        else:
            count = count0


if __name__ == '__main__':

    from pylab import *
    import random
    import numpy as np

    # print test_timeconstant('butter')
    # print test_timeconstant('cheby2')
    # print test_timeconstant('elliptic')
    # print test_timeconstant('bessel')

    # y = [50.0 for i in x]
    # y = [50 + 20*math.sin(3*i) + 20*math.cos(7*i) for i in x ]

    s0 = 20
    s1 = 40
    s2 = 100
    s3 = 300
    s4 = 600

    x = range(0, s4)
    y = [0.0] * s4

    for i in xrange(1, len(y)):
        if i < s0:

            y[i] = 5 * i

        elif i < s1:

            j = i - s0
            y[i] = 100 - 5 * j

        elif i < s2:

            y[i] = 0

        elif i < s3:

            y[i] = 100

        elif i < s4:

            y[i] = 100 + 25 * sin(i * math.pi / 50) + \
                10 * sin(i * math.pi / 20)

    #y = [i+ 5*math.sin(9*i) for i in y ]
    y_old = y
    y = [i + random.normalvariate(0, 5) for i in y]

    y_filt = np.zeros_like(y)

    f = figure()
    plt.ion
    plot(x, y, "r.")

    plot(x, y_old, "m+")

    wtest = 30

    smooth(y, y_filt, wtest, ftype='bessel')
    plot(x, y_filt, "y-")
    plt.draw()

    while not plt.waitforbuttonpress():
        pass

    smooth(y, y_filt, wtest, ftype='butter')
    plot(x, y_filt, "b-")
    plt.draw()

    while not plt.waitforbuttonpress():
        pass

    smooth(y, y_filt, wtest, ftype='cheby2')
    plot(x, y_filt, "g-")
    plt.draw()

    while not plt.waitforbuttonpress():
        pass

    smooth(y, y_filt, wtest, ftype='elliptic')
    plot(x, y_filt, "k-")
    plt.draw()

    while not plt.waitforbuttonpress():
        pass

    # xlim(-1,7)
    # smooth(y_filt,y_filt,30)
    # plot(x,y_filt,"g-")

    # smooth(y_filt,y_filt,30)
    # plot(x,y_filt,"r-")

	# plt.show()
    plt.close(f)

    f = figure()

    # plot(x,y,"b-")
    n = 8
    y_d = decimate(y, n)
    x_d = decimate(x, n)
    plot(x_d, y_d, "m.")

    print n * int(math.floor(float(len(x)) / n))
    print len(x)

    x_d2 = [x[n * (i - 1)]
            for i in xrange(1,  int(math.floor(float(len(x)) / n)))]
    y_d2 = [y[n * (i - 1)]
            for i in xrange(1,  int(math.floor(float(len(y)) / n)))]

    plot(x_d2, y_d2, "g.")

    plt.draw()
    plt.show()
