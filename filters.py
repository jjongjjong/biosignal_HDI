import scipy.signal as ss
from scipy import interpolate, optimize
from scipy.stats import stats
import keyword
import numpy as np
'''biosppy의 전처리 모듈 중 일부를 가져왔음'''
class ReturnTuple(tuple):
    def __new__(cls, values, names=None):

        return tuple.__new__(cls, tuple(values))

    def __init__(self, values, names=None):

        nargs = len(values)

        if names is None:
            # create names
            names = ['_%d' % i for i in range(nargs)]
        else:
            # check length
            if len(names) != nargs:
                raise ValueError("Number of names and values mismatch.")

            # convert to str
            names = list(map(str, names))

            # check for keywords, alphanumeric, digits, repeats
            seen = set()
            for name in names:
                if not all(c.isalnum() or (c == '_') for c in name):
                    raise ValueError("Names can only contain alphanumeric \
                                      characters and underscores: %r." % name)

                if keyword.iskeyword(name):
                    raise ValueError("Names cannot be a keyword: %r." % name)

                if name[0].isdigit():
                    raise ValueError("Names cannot start with a number: %r." %
                                     name)

                if name in seen:
                    raise ValueError("Encountered duplicate name: %r." % name)

                seen.add(name)

        self._names = names

        
        
def _norm_freq(frequency=None, sampling_rate=1000.):
    # check inputs
    if frequency is None:
        raise TypeError("Please specify a frequency to normalize.")

    # convert inputs to correct representation
    try:
        frequency = float(frequency)
    except TypeError:
        # maybe frequency is a list or array
        frequency = np.array(frequency, dtype='float')

    Fs = float(sampling_rate)

    wn = 2. * frequency / Fs

    return wn

def _filter_signal(b, a, signal, zi=None, check_phase=True, **kwargs):
    # check inputs
    if check_phase and zi is not None:
        raise ValueError(
            "Incompatible arguments: initial filter state cannot be set when \
            check_phase is True.")

    if zi is None:
        zf = None
        if check_phase:
            filtered = ss.filtfilt(b, a, signal, **kwargs)
        else:
            filtered = ss.lfilter(b, a, signal, **kwargs)
    else:
        filtered, zf = ss.lfilter(b, a, signal, zi=zi, **kwargs)

    return filtered, zf


def get_filter(ftype='FIR',
               band='lowpass',
               order=None,
               frequency=None,
               sampling_rate=1000., **kwargs):

    # check inputs
    if order is None:
        raise TypeError("Please specify the filter order.")
    if frequency is None:
        raise TypeError("Please specify the cutoff frequency.")
    if band not in ['lowpass', 'highpass', 'bandpass', 'bandstop']:
        raise ValueError(
            "Unknown filter type '%r'; choose 'lowpass', 'highpass', \
            'bandpass', or 'bandstop'."
            % band)

    # convert frequencies
    frequency = _norm_freq(frequency, sampling_rate)

    # get coeffs
    b, a = [], []
    if ftype == 'FIR':
        # FIR filter
        if order % 2 == 0:
            order += 1
        a = np.array([1])
        if band in ['lowpass', 'bandstop']:
            b = ss.firwin(numtaps=order,
                          cutoff=frequency,
                          pass_zero=True, **kwargs)
        elif band in ['highpass', 'bandpass']:
            b = ss.firwin(numtaps=order,
                          cutoff=frequency,
                          pass_zero=False, **kwargs)
    elif ftype == 'butter':
        # Butterworth filter
        b, a = ss.butter(N=order,
                         Wn=frequency,
                         btype=band,
                         analog=False,
                         output='ba', **kwargs)
    elif ftype == 'cheby1':
        # Chebyshev type I filter
        b, a = ss.cheby1(N=order,
                         Wn=frequency,
                         btype=band,
                         analog=False,
                         output='ba', **kwargs)
    elif ftype == 'cheby2':
        # chevyshev type II filter
        b, a = ss.cheby2(N=order,
                         Wn=frequency,
                         btype=band,
                         analog=False,
                         output='ba', **kwargs)
    elif ftype == 'ellip':
        # Elliptic filter
        b, a = ss.ellip(N=order,
                        Wn=frequency,
                        btype=band,
                        analog=False,
                        output='ba', **kwargs)
    elif ftype == 'bessel':
        # Bessel filter
        b, a = ss.bessel(N=order,
                         Wn=frequency,
                         btype=band,
                         analog=False,
                         output='ba', **kwargs)

    return ReturnTuple((b, a), ('b', 'a'))


def filter_signal(signal=None,
                  ftype='FIR',
                  band='lowpass',
                  order=None,
                  frequency=None,
                  sampling_rate=1000., **kwargs):


    # check inputs
    if signal is None:
        raise TypeError("Please specify a signal to filter.")

    # get filter
    b, a = get_filter(ftype=ftype,
                      order=order,
                      frequency=frequency,
                      sampling_rate=sampling_rate,
                      band=band, **kwargs)

    # filter
    filtered, _ = _filter_signal(b, a, signal, check_phase=True)
    return filtered


def filter_eeg(signal,sampling_rate=None):
      # ensure numpy
    signal = np.array(signal)

    sampling_rate = float(sampling_rate)

    # high pass filter
    b, a = get_filter(ftype='butter',
                         band='highpass',
                         order=8,
                         frequency=4,
                         sampling_rate=sampling_rate)

    aux, _ = _filter_signal(b, a, signal=signal, check_phase=True, axis=0)

    # low pass filter
    b, a = get_filter(ftype='butter',
                         band='lowpass',
                         order=16,
                         frequency=40,
                         sampling_rate=sampling_rate)

    filtered, _ = _filter_signal(b, a, signal=aux, check_phase=True, axis=0)
    return filtered