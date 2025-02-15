"""Common module"""

import os
import csiread
import numpy as np
import scipy.signal as signal
from scipy.ndimage import median_filter
from scipy.interpolate import interp1d
def conj_mult2(phase_data):

    phase_ant1 = phase_data[:,:,0,0]
    phase_ant2 = phase_data[:,:,1,0]
    phase_ant3 = phase_data[:,:,2,0]

    new_phase_ant1 = np.unwrap(phase_ant1 * np.conj(phase_ant2))
    new_phase_ant2 = np.unwrap(phase_ant2 * np.conj(phase_ant3))
    new_phase_ant3 = np.unwrap(phase_ant3 * np.conj(phase_ant1))

    conj_mult_phase = np.stack([new_phase_ant1,new_phase_ant2,new_phase_ant3],axis=2)

    return conj_mult_phase
def conj_mult(phase_data):

    phase_ant1 = phase_data[:,:,0,0]
    phase_ant2 = phase_data[:,:,1,0]
    phase_ant3 = phase_data[:,:,2,0]

    new_phase_ant1 = np.unwrap(phase_ant1 * np.conj(phase_ant2))
    new_phase_ant2 = np.unwrap(phase_ant2 * np.conj(phase_ant3))
    new_phase_ant3 = np.unwrap(phase_ant3 * np.conj(phase_ant1))

    conj_mult_phase = np.concatenate([new_phase_ant1,new_phase_ant2,new_phase_ant3],axis=-1)

    return conj_mult_phase

def standard(data,epsilon=1e-8):
    return (data-np.mean(data))/(np.std(data) + epsilon)

def MAXMINNormalization(data):
    amax, amin = np.max(data),np.min(data)
    norm_a = ( data - amin) / (amax - amin)
    return norm_a
def linear_insert(data, length):
    original_indices = np.linspace(0, 1, data.shape[0])
    # 创建目标索引
    target_indices = np.linspace(0, 1, length)
    # 创建插值函数
    interpolator = interp1d(original_indices, data, axis=0, kind='linear', fill_value='extrapolate')
    # 应用插值
    interpolated_data = interpolator(target_indices)

    return interpolated_data

def truncate_or_pad_csi(data, target_length):
    """
    对CSI数据进行截断或零填充。
    :param data: 输入数据，形状为 (n_packets, n_subcarriers)
    :param target_length: 目标时间包数
    :return: 统一长度的数据，形状为 (target_length, n_subcarriers)
    """
    n_packets, n_subcarriers,n_rx,n_tx = data.shape
    if n_packets > target_length:
        return data[:target_length]
    else:
        padded_data = np.zeros((target_length, n_subcarriers,n_rx,n_tx), dtype=data.dtype)
        padded_data[:n_packets] = data
        return padded_data

def sliding_window_2d(data, window_size, step_size):
    """
    data:数据
    window_size:窗口大小
    step_size:滑动步长
    """
    for i in range(0, len(data) - window_size + 1, step_size):
        yield data[i:i + window_size]

def sliding_window_3d(data, window_size, step_size):
    """
    滑动窗口函数，用于三维数据 (C, L, F)。
    :param data: 输入数据，形状为 (C, L, F)。
    :param window_size: 滑动窗口的大小，应用于长度维度 L。
    :param step_size: 滑动步长。
    :return: 生成器，返回形状为 (C, window_size, F) 的窗口。
    """
    C, L, F = data.shape  # 获取输入数据的形状
    for i in range(0, L - window_size + 1, step_size):
        # 提取长度维度上的窗口
        yield data[:, i:i + window_size, :]


def butterworth(data, n, wn, ptype, fs):
    """

    :param data: signal data
    :param n: N阶数，越大越陡峭，计算越复杂。
    :param wn: Wn截止频率，阻断的频率 0 < Wn < fs/2。
    :param ptype: btype低通，高通，带通，阻通。
    :param fs: fs采样频率。
    :return: signal with low pass
    """
    b, a = signal.butter(N=n, Wn=wn, btype=ptype, fs=fs)
    filtered_data = signal.filtfilt(b, a, data)
    return filtered_data



def batch_butterworth_filter(matrix, fs, cutoff, order=4, filter_type='low'):
    """
    批量对二维矩阵的所有列进行 Butterworth 滤波
    :param matrix: 输入二维矩阵，形状为 (rows, cols)
    :param fs: 采样率
    :param cutoff: 截止频率（标量或元组）
    :param order: 滤波器阶数
    :param filter_type: 滤波类型 ('low', 'high', 'bandpass', 'bandstop')
    :return: 滤波后的二维矩阵，形状与输入相同
    """
    # 归一化截止频率
    nyquist = 0.5 * fs
    normalized_cutoff = np.array(cutoff) / nyquist

    # 设计 Butterworth 滤波器
    b, a = signal.butter(order, normalized_cutoff, btype=filter_type, analog=False)

    # 使用矩阵形式批量滤波
    filtered_matrix = signal.filtfilt(b, a, matrix, axis=0)
    return filtered_matrix

def hampel_adv(data, window_size, n_sigmas=3):
    """
    :param data:  signal data
    :param window_size: 窗口大小
    :param n_sigmas: 阈值系数
    :return:signal after filter
    """
    data = np.array(data)
    filtered_data = data.copy()

    # 计算窗口的中位数和标准差（使用 scipy 的 median_filter 和标准差计算）
    median_data = median_filter(data, size=window_size)
    sigma_data = np.array(
        [np.std(data[max(0, i - window_size): min(len(data), i + window_size + 1)]) for i in range(len(data))])

    # 查找异常值并替换
    for i in range(len(data)):
        if np.abs(data[i] - median_data[i]) > n_sigmas * sigma_data[i]:
            filtered_data[i] = median_data[i]  # 用中位数替换异常值
    # print('test')
    return filtered_data
def hampel(data, window_size, n_sigmas=3):
    """
    :param data:  signal data
    :param window_size: 窗口大小
    :param n_sigmas: 阈值系数
    :return:signal after filter
    """
    data = np.array(data)
    filtered_data = data.copy()
    outline = []
    for i in range(len(data)):
        start = max(0, i - window_size)
        end = min(len(data), i + window_size + 1)

        window_data = data[start:end]
        median = np.median(window_data)
        sigma = np.std(window_data)

        if np.abs(data[i] - median) > n_sigmas * sigma:
            outline.append(i)
            filtered_data[i] = median  # Replace outlier with median
    # print('test')
    return filtered_data

def hampel_vectorized(data, window_size, n_sigmas=3):
    """
    Vectorized Hampel filter for 2D data.
    :param data: 2D signal data (e.g., 2000x180)
    :param window_size: Window size for the Hampel filter
    :param n_sigmas: Threshold factor for detecting outliers
    :return: Filtered data (same shape as input)
    """
    data = np.array(data)
    filtered_data = data.copy()

    # Precompute medians and std deviations for sliding windows
    for i in range(data.shape[0]):
        start = max(0, i - window_size)
        end = min(data.shape[0], i + window_size + 1)

        # Compute medians and standard deviations along each column
        window_data = data[start:end, :]
        median = np.median(window_data, axis=0)
        sigma = np.std(window_data, axis=0)

        # Detect and replace outliers
        mask = np.abs(data[i, :] - median) > n_sigmas * sigma
        filtered_data[i, mask] = median[mask]

    return filtered_data

def scidx(bw, ng, standard='n'):
    """subcarriers index

    Args:
        bw: bandwitdh(20, 40, 80)
        ng: grouping(1, 2, 4)
        standard: 'n' - 802.11n， 'ac' - 802.11ac.
    Ref:
        1. 802.11n-2016: IEEE Standard for Information technology—Telecommunications
        and information exchange between systems Local and metropolitan area
        networks—Specific requirements - Part 11: Wireless LAN Medium Access
        Control (MAC) and Physical Layer (PHY) Specifications, in
        IEEE Std 802.11-2016 (Revision of IEEE Std 802.11-2012), vol., no.,
        pp.1-3534, 14 Dec. 2016, doi: 10.1109/IEEESTD.2016.7786995.
        2. 802.11ac-2013 Part 11: ["IEEE Standard for Information technology--
        Telecommunications and information exchange between systemsLocal and
        metropolitan area networks-- Specific requirements--Part 11: Wireless
        LAN Medium Access Control (MAC) and Physical Layer (PHY) Specifications
        --Amendment 4: Enhancements for Very High Throughput for Operation in 
        Bands below 6 GHz.," in IEEE Std 802.11ac-2013 (Amendment to IEEE Std
        802.11-2012, as amended by IEEE Std 802.11ae-2012, IEEE Std 802.11aa-2012,
        and IEEE Std 802.11ad-2012) , vol., no., pp.1-425, 18 Dec. 2013,
        doi: 10.1109/IEEESTD.2013.6687187.](https://www.academia.edu/19690308/802_11ac_2013)
    """

    PILOT_AC = {
        20: [-21, -7, 7, 21],
        40: [-53, -25, -11, 11, 25, 53],
        80: [-103, -75, -39, -11, 11, 39, 75, 103],
        160: [-231, -203, -167, -139, -117, -89, -53, -25, 25, 53, 89, 117, 139, 167, 203, 231]
    }
    SKIP_AC_160 = {1: [-129, -128, -127, 127, 128, 129], 2: [-128, 128], 4: []}
    AB = {20: [28, 1], 40: [58, 2], 80: [122, 2], 160: [250, 6]}
    a, b = AB[bw]

    if standard == 'n':
        if bw not in [20, 40] or ng not in [1, 2, 4]:
            raise ValueError("bw should be [20, 40] and ng should be [1, 2, 4]")
        k = np.r_[-a:-b:ng, -b, b:a:ng, a]
    if standard == 'ac':
        if bw not in [20, 40, 80] or ng not in [1, 2, 4]:
            raise ValueError("bw should be [20, 40, 80] and ng should be [1, 2, 4]")

        g = np.r_[-a:-b:ng, -b]
        k = np.r_[g, -g[::-1]]

        if ng == 1:
            index = np.searchsorted(k, PILOT_AC[bw])
            k = np.delete(k, index)
        if bw == 160:
            index = np.searchsorted(k, SKIP_AC_160[ng])
            k = np.delete(k, index)
    return k


def calib(phase, k, axis=1):
    """Phase calibration

    Args:
        phase (ndarray): Unwrapped phase of CSI.
        k (ndarray): Subcarriers index
        axis (int): Axis along which is subcarrier. Default: 1

    Returns:
        ndarray: Phase calibrated

    ref:
        [Enabling Contactless Detection of Moving Humans with Dynamic Speeds Using CSI]
        (http://tns.thss.tsinghua.edu.cn/wifiradar/papers/QianKun-TECS2017.pdf)
    """
    p = np.asarray(phase)
    k = np.asarray(k)

    slice1 = [slice(None, None)] * p.ndim
    slice1[axis] = slice(-1, None)
    slice1 = tuple(slice1)
    slice2 = [slice(None, None)] * p.ndim
    slice2[axis] = slice(None, 1)
    slice2 = tuple(slice2)
    shape1 = [1] * p.ndim
    shape1[axis] = k.shape[0]
    shape1 = tuple(shape1)

    k_n, k_1 = k[-1], k[0]
    a = (p[slice1] - p[slice2]) / (k_n - k_1)
    b = p.mean(axis=axis, keepdims=True)
    k = k.reshape(shape1)

    phase_calib = p - a * k - b
    return phase_calib


def phy_ifft(x, k, axis=1):
    """PHY IFFT

    Return discrete inverse Fourier transform of real or complex sequence. It
    is based on Equation (19-25)(P2373) and Table 19-6—Timing-related constants(P2354) in
    802.11n-2016, Table 22-5—Timing-related constants in 802.11ac-2013

    Note:
        1. No ifftshift
        2. scipy.fftpack.ifft is different from Equation (19-25) and Equation (17-9)
        3. BE CAREFUL! I haven't found any code about CSI like this.

    Ref:
        1. IEEE Standard for Information technology—Telecommunications and information
        exchange between systems Local and metropolitan area networks—Specific
        requirements - Part 11: Wireless LAN Medium Access Control (MAC) and Physical
        Layer (PHY) Specifications, in IEEE Std 802.11-2016 (Revision of IEEE Std
        802.11-2012), vol., no., pp.1-3534, 14 Dec. 2016, doi: 10.1109/IEEESTD.2016.7786995.
        2. "IEEE Standard for Information technology-- Telecommunications
        and information exchange between systemsLocal and metropolitan area
        networks-- Specific requirements--Part 11: Wireless LAN Medium Access
        Control (MAC) and Physical Layer (PHY) Specifications--Amendment 4:
        Enhancements for Very High Throughput for Operation in Bands below
        6 GHz.," in IEEE Std 802.11ac-2013 (Amendment to IEEE Std 802.11-2012,
        as amended by IEEE Std 802.11ae-2012, IEEE Std 802.11aa-2012, and IEEE
        Std 802.11ad-2012) , vol., no., pp.1-425, 18 Dec. 2013,
        doi: 10.1109/IEEESTD.2013.6687187.
    """
    M = x.shape[axis]
    x = x.swapaxes(-1, axis)
    k = np.asarray(k)
    bw = np.around(k[-1] / 30) * 20

    n = 64 * (bw / 20)
    delta_f = bw * 1e6 / n
    t = np.c_[:n] / (bw * 1e6)

    g = np.exp(2.j * np.pi * k * delta_f * t) / M
    out = x @ g.T

    out = out.swapaxes(-1, axis)
    return out


def phy_fft(x, k, axis=1):
    """PHY FFT

    Return discrete Fourier transform of real or complex sequence.
    """
    x = x.swapaxes(-1, axis)
    k = np.asarray(k)
    bw = np.around(k[-1] / 30) * 20

    n = 64 * (bw / 20)
    delta_f = bw * 1e6 / n
    t = np.c_[:n] / (bw * 1e6)

    g = np.exp(-2.j * np.pi * k * delta_f * t)
    out = x @ g

    scale = k.size / n
    out = out.swapaxes(-1, axis) * scale
    return out


def infer_device(csifile):
    """Infer the CSI file format simplely

    Args:
        csifile (str): csi data file

    Returns:
        str:
            Intel: intel 5300 csi file
            Atheros: atheros csi file
            Nexmon: nexmon csi file
            AtherosPull10: `atheros pull 10 <https://github.com/xieyaxiongfly/Atheros-CSI-Tool-UserSpace-APP/pull/10/files>`_
            NexmonPull46: `nexmon_csi pull 46 <https://github.com/seemoo-lab/nexmon_csi/pull/46/files>`_
            ESP32: esp32-csi-tool csi file
            Picoscenes: Picoscenes csi fie
            Unknown: Unknown file format
    Note:
        This function cannot work for some nexmon csi file formats defined by
        projects derived from nexmon_csi
    """
    if csifile.endswith('.csv'):
        return 'ESP32'
    if csifile.endswith('.csi'):
        return 'Picoscenes'

    with open(csifile, 'rb') as f:
        buf = f.read(4)
        if buf[2] in [0xc1, 0xbb]:
            return 'Intel'
        elif buf in [b"\xa1\xb2\xc3\xd4", b"\xd4\xc3\xb2\xa1",
                     b"\xa1\xb2\x3c\x4d", b"\x4d\x3c\xb2\xa1"]:
            f.seek(20 + 16 + 42, os.SEEK_CUR)
            buf = f.read(4)
            if buf == b'\x11\x11\x11\x11':
                return 'Nexmon'
            if buf[:2] == b'\x11\x11' and buf[2:] != b'\x11\x11':
                return 'NexmonPull46'
        elif buf[0] in [0xff, 0x00]:
            return 'AtherosPull10'
        else:
            f.seek(2 + 16 - 4, os.SEEK_CUR)
            buf = f.read(1)
            if buf[0] == 56 or buf[0] == 114:
                return 'Atheros'
            else:
                return 'Unknown'


def infer_tones(csifile, device):
    """Infer the argument `tones` of Atheros and AtherosPull10

    Args:
        csifile (str): atheros csi file
        device (str): Atheros or AtherosPull10

    Returns:
        int: tones

    Examples:

        >>> tones = infer_tones(csifile, 'Atheros')
        >>> csidata = csiread.Atheros(csifile, nrxnum=3, ntxnum=3,
        >>>                           if_report=False, tones=tones)
    """
    with open(csifile, 'rb') as f:
        if device == "Atheros":
            f.seek(18, os.SEEK_CUR)
        elif device == 'AtherosPull10':
            f.seek(19, os.SEEK_CUR)
        else:
            raise Exception("It is not a Atheros file")
        tones = f.read(1)[0]
    return tones


def infer_chip_bw(csifile):
    """Infer chip, bandwidth and band of nexmon csi file.

    This function may be failed on some nexmon csi files.

    Args:
        csifile (str): nexmon csi file

    Returns:
        tuple (chip, bw, band):
            str: chip
            int: bw, MHz
            int: band, GHz

    Examples:

        >>> csifile = "../material/nexmon/dataset/example.pcap"
        >>> chip, bw, band = infer_chip_bw(csifile)
        >>> csidata = csiread.Nexmon(csifile, chip=chip, bw=bw)
        >>> csidata.read()

    References:
        1. `D11AC_IOTYPES lines 172-188 <https://github.com/seemoo-lab/nexmon/blob/master/patches/include/bcmwifi_channels.h#L172>`_
    """
    csidata = csiread.Nexmon(None, 'none', 0, False, bufsize=1)
    csidata.seek(csifile, 24, 1)
    chan_spec = csidata.chan_spec[0]
    chip_version = csidata.chip_version[0]

    CHIP_LIST = {0x0065: '43455c0', 0xdead: '4358', 0xe834: '4366c0'}
    WL_CHANSPEC_BW = chan_spec & 0x3800
    WL_CHANSPEC_BAND = chan_spec & 0xc000
    if WL_CHANSPEC_BW in [0x2800, 0x3000]:
        raise Exception("csiread.Nexmon doesn't support WL_CHANSPEC_BW_160 "
                        "and WL_CHANSPEC_BW_8080")
    if WL_CHANSPEC_BAND == 255:
        raise Exception("INVCHANSPEC")

    chip = CHIP_LIST[chip_version]
    bw = (1 << (WL_CHANSPEC_BW // 0x0800)) * 5
    band = WL_CHANSPEC_BAND // 0x4000 + 2

    return chip, bw, band


def db(x):
    return 10 * np.log10(x) + 300 - 300


def dbinv(x):
    return np.power(10, x/10)
