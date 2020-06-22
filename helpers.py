##################
# imports
import os
import csv
import re
import pathlib
import xml.etree.ElementTree as ET
import zipfile
import shutil
import librosa
import math
import numpy as np 
import pandas as pd 
import warnings
import audioread
import time
import matplotlib.pyplot as plt
import seaborn as sns
from pandas import ExcelWriter
from pydub.utils import mediainfo

##################
# Default Constants
# Directory where MP3 files have been downloaded
zip_data_dir   = '../../L5_Capstone/Audio/Quran'
# Name of ZIP file for each reciter
zip_file_name  = '000_versebyverse.zip'
# Directory where processed data & other generated data will be stored
data_dir       = '../data'
audio_data_dir = os.path.join(os.getcwd(), data_dir, "audio")
quran_meta_xml = os.path.join(os.getcwd(), data_dir, "quran-data.xml")
# Max/Min number of Sura/Aya with first as index 1
SuraIndexMIN   = 1
SuraIndexMAX   = 114
AyaIndexMIN    = 0
AyaIndexMAX    = 6236

# Suppress this warning from librosa:
# UserWarning: PySoundFile failed. Trying audioread instead.
warnings.filterwarnings('ignore')

##################
# Functions
def qsura_ayat_to_labels(suraFrom=None, suraTo=None, ayaFrom=None, ayaTo=None):
    """ Converts either Sura or Aya numbers to labels.

       :param suraFrom: An interger for Sura index to start from. Valid numbers are 1 to 114 (default: None)
       :type suraFrom: int
       :param suraTo: An interger for Sura index to end to. Valid numbers are 1 to 114 (inclusive) (default: None)
       :type suraTo: int
       :param ayaFrom: An interger for Aya index to start from (default: None)
       :type ayaFrom: int
       :param ayaTo: An interger for Aya index to end to (inclusive) (default: None)
       :type ayaTo: int
       :return: A list of labels in the form ['001001', '001002', ...], A list of aya number in quran [0, 2, ... AyaIndexMAX]
       :rtype: list, list 

    """

    # Return lists
    labels_list = list()
    ayainq = list()

    useSura = False
    useAya = False
    if suraFrom is not None and suraTo is not None:
        if suraFrom < SuraIndexMIN or suraFrom > SuraIndexMAX:
            print("ERROR: {} not between {} and {}".format('suraFrom', SuraIndexMIN,SuraIndexMAX))
            return labels_list, ayainq
        if suraTo < SuraIndexMIN or suraTo > SuraIndexMAX:
            print("ERROR: {} not between {} and {}".format('suraTo', SuraIndexMIN,SuraIndexMAX))
            return labels_list, ayainq
        useSura = True
    elif ayaFrom is not None and ayaTo is not None:
        if ayaFrom < AyaIndexMIN or ayaFrom > AyaIndexMAX:
            print("ERROR: {} not between {} and {}".format('ayaFrom', AyaIndexMIN,AyaIndexMAX))
            return labels_list, ayainq
        if ayaTo < AyaIndexMIN or ayaTo > AyaIndexMAX:
            print("ERROR: {} not between {} and {}".format('ayaTo', AyaIndexMIN,AyaIndexMAX))
            return labels_list, ayainq
        useAya = True

    ##################
    # qmeta: Quran Meta Data
    qmeta_tree = ET.parse(quran_meta_xml)
    qmeta_root = qmeta_tree.getroot()
    #print("qmeta_root :", qmeta_root)

    # As an Element, root has a tag and a dictionary of attributes:
    qmeta_root_tag = qmeta_root.tag
    qmeta_root_att = qmeta_root.attrib
    #print("qmeta_root_tag = " + qmeta_root_tag)
    #print("qmeta_root_att = ")
    #print(qmeta_root_att)

    # It also has children nodes over which we can iterate:
    for qmeta_suras in qmeta_root:
        qmeta_suras_tag = qmeta_suras.tag
        #qmeta_suras_att = qmeta_suras.attrib
        #print("qmeta_suras_tag = " + qmeta_suras_tag)
        #print("qmeta_suras_att = ")
        #print(qmeta_suras_att)
        if qmeta_suras_tag == "suras":
            for qmeta_sura in qmeta_suras:
                qmeta_sura_tag = qmeta_sura.tag
                #qmeta_sura_att = qmeta_sura.attrib
                #print("qmeta_sura_tag = " + qmeta_sura_tag)
                #print("qmeta_sura_att = ")
                #print(qmeta_sura_att)
                if qmeta_sura_tag == "sura":
                    #print("qmeta_sura :", qmeta_sura)
                    qmeta_sura_index = qmeta_sura.attrib.get('index')
                    qmeta_sura_ayas = qmeta_sura.attrib.get('ayas')
                    qmeta_sura_start = qmeta_sura.attrib.get('start')
                    #print("qmeta_sura_index :", qmeta_sura_index)
                    #print("qmeta_sura_ayas :", qmeta_sura_ayas)
                    #print("qmeta_sura_start :", qmeta_sura_start)

                    if useSura:
                        if int(qmeta_sura_index) >= suraFrom and int(qmeta_sura_index) <= suraTo:
                            #print("  MKC: qmeta_sura_index :", qmeta_sura_index)
                            #print("  MKC: qmeta_sura_ayas :", qmeta_sura_ayas)
                            #print("  MKC: qmeta_sura_start :", qmeta_sura_start)
                            for i in range(1, int(qmeta_sura_ayas)+1):
                                labels_list.append("{:03d}{:03d}".format(int(qmeta_sura_index), i))

                    if useAya:
                        # Get the current sura end ayat
                        sura_start = int(qmeta_sura_start)
                        sura_end = sura_start + (int(qmeta_sura_ayas) - 1)
                        #print("sura start -> end: {} -> {}".format(sura_start, sura_end))
                        for i in range(sura_start, sura_end+1):
                            if i >= ayaFrom and i <= ayaTo:
                                #print("  -> ",i)
                                ayainq.append(i)
                                labels_list.append("{:03d}{:03d}".format(int(qmeta_sura_index), i+1-sura_start))

    #print("labels_list :", labels_list)
    return labels_list, ayainq

def report_stats_zip_data(zip_data_dir=zip_data_dir, zip_file_name=zip_file_name):
    """ Reports statitics for the zipped data

       :param zip_data_dir: Directory containing the zipped data
       :type zip_data_dir: str
       :param zip_file_name: Name of the zip file in the directory
       :type zip_file_name: str
       :return: A list of directory names in zip_data_dir
       :rtype: list 

    """

    # Return list
    dir_names = list()

    # Directory names are also names of the reciters
    print("{:30s} {:10s} {:8s} {:9s}".format("Reciter name", "Data Size", "Files", "MP3 Files"))
    print("{:30s} {:10s} {:8s} {:9s}".format("============", "=========", "=====", "========="))
    for dd in os.listdir(zip_data_dir):
        # Each directory has one zip file called 000_versebyverse.zip
        dd_zip_file = zip_data_dir + "/" + dd + "/" + zip_file_name
        # Size
        dd_size_bytes = os.path.getsize(dd_zip_file)
        dd_size_MB = dd_size_bytes / (1024 * 1024)
        # Number of files
        archive = zipfile.ZipFile(dd_zip_file, 'r')
        num_files = len(archive.namelist())
        # Mp3 files
        mp3_cnt = 0
        for ff in archive.namelist():
            if ff.endswith('.mp3'):
                mp3_cnt += 1

        print("{:30s} {:6.0f} MB {:6d} {:12d}".format(dd, dd_size_MB, num_files, mp3_cnt))
        dir_names.append(dd)
    return dir_names

# Directory size (https://stackoverflow.com/questions/1392413/calculating-a-directorys-size-using-python)
def get_dir_size(start_path = '.'):
    """ Gets the size of the directory recursively

       :param start_path: Path to a directory whose size is needed
       :type start_path: str
       :return: Total size of the directory in bytes
       :rtype: int

    """

    total_size = 0
    for dirpath, dirnames, filenames in os.walk(start_path):
        for f in filenames:
            fp = os.path.join(dirpath, f)
            # skip if it is symbolic link
            if not os.path.islink(fp):
                total_size += os.path.getsize(fp)

    return total_size

def report_stats_audio_data(data_dir=audio_data_dir):
    """ Reports statitics for the audio data

       :param data_dir: Directory containing the audio data
       :type data_dir: str
       :param zip_file_name: Name of the zip file in the directory
       :type zip_file_name: str
       :return: DataFrame with audio directory details
       :rtype: DataFrame 

    """

    column_names = list()
    column_names.append('ReciterName')
    column_names.append('FileName')
    column_names.append('DataSizeKB')
    column_names.append('BitRate')
    column_names.append('Channels')
    column_names.append('Mono/Stereo')
    column_names.append('Duration')
    column_names.append('FileNoExt')
    column_names.append('Sura')
    column_names.append('Aya')
    column_names.append('AyaInQuran')

    sv_reciter   = list()
    sv_file_name = list()
    sv_data_size = list()
    sv_bit_rate  = list()
    sv_ch        = list()
    sv_ms        = list()
    sv_dur       = list()
    sv_filenoext = list()
    sv_sura      = list()
    sv_aya       = list()
    sv_ayainq    = list()

    # Directory names are also names of the reciters
    print("{:30s} {:10s} {:4s} {:4s} {:2s} {:6s} {:9s}".format("Reciter name/MP3 File", "Data Size", "MP3s", "kbps", "Ch", "Mono/S", "Duration(sec)"))
    print("{:30s} {:10s} {:4s} {:4s} {:2s} {:6s} {:9s}".format("==============================", "=========", "====", "====", "==", "======", "============="))
    for dd in os.listdir(data_dir):
        reciter_dir = os.path.join(data_dir, dd)
        # Size
        #dd_size_bytes = os.path.getsize(reciter_dir)
        dd_size_bytes = get_dir_size(reciter_dir)
        dd_size_KB = dd_size_bytes / (1024)
        # Number of files
        num_files = len(os.listdir(reciter_dir))
        # Mp3 files
        mp3_cnt = 0
        for ff in os.listdir(reciter_dir):
            if ff.endswith('.mp3'):
                mp3_cnt += 1

        print("{:30s} {:6.0f} KB {:5d}".format(dd, dd_size_KB, mp3_cnt))

        for ff in os.listdir(reciter_dir):
            if not ff.endswith('.mp3'):
                continue
            mp3_file = os.path.join(reciter_dir, ff)

            duration = channel_layout = bit_rate = channels = bit_rate_kbps = -1
            try:
                # Look at some audio features
                y, sr = librosa.load(mp3_file, sr=None)
                # Get the length of the audio
                duration = librosa.core.get_duration(y=y, sr=sr)
                duration = len(y) / sr

                # Sample rate
                info = mediainfo(mp3_file)
                #print(info)
                channel_layout = info['channel_layout']
                bit_rate = int(info['bit_rate'])
                channels = int(info['channels'])
                #artist = info['artist']
                artist = ""
                bit_rate_kbps = bit_rate / 1000
            
            except:
                print("Couldn't process, skipping: ", mp3_file)

            #print("{:>30s} {:6.0f} KB {:5d} {:12d} {:5.1f}".format(ff, dd_size_KB, 0, int(info['sample_rate']), duration))
            #with audioread.audio_open(mp3_file) as input_file:
            #    sr_native = input_file.samplerate
            #    n_channels = input_file.channels
            #print(sr_native, n_channels)
            #print("{:>30s} {:6.0f} KB {:5s} {:4.0f} {:2d} {:6s} {:13.1f}".format(ff, dd_size_KB, "", bit_rate_kbps, channels, channel_layout, duration))

            # Size
            dd_size_bytes = os.path.getsize(mp3_file)
            dd_size_KB = dd_size_bytes / (1024)

            sv_reciter.append(dd)
            sv_file_name.append(ff)
            sv_data_size.append(int(dd_size_KB))
            sv_bit_rate.append(int(bit_rate_kbps))
            sv_ch.append(channels)
            sv_ms.append(channel_layout)
            sv_dur.append(duration)
            filenoext, sura, aya, ayainq = get_mp3_file_info(ff)
            sv_filenoext.append(filenoext)
            sv_sura.append(sura)
            sv_aya.append(aya)
            sv_ayainq.append(ayainq)
            #break
        #break
    
    # Create a DataFrame with all the info
    df = pd.DataFrame(list(zip(sv_reciter, sv_file_name, sv_data_size, 
        sv_bit_rate, sv_ch, sv_ms, sv_dur, sv_filenoext, sv_sura, sv_aya, sv_ayainq)), 
               columns=column_names) 

    return df

def get_mp3_file_info(file_name):
    """ Get info from MP3 file name. Use lbl_aya_dict dictionary for lookup so this variable needs to be defined before.

       :param file_name: MP3 file name
       :type file_name: str
       :return: label, sura, aya, ayainquran
       :rtype: str 

    File name should be ######.mp3, e.g. 001004.mp3
        SuraAya    = 001004
        Sura       = 1
        Aya        = 4
        AyaInQuran = 3
    """

    # Return items
    suraaya   = ''
    sura      = -1
    aya       = -1
    ayainquan = -1

    suraaya = os.path.splitext(file_name)[0]
    sura = int(suraaya[:3])
    aya  = int(suraaya[3:])
    ayainquan = int(lbl_aya_dict[suraaya])

    return suraaya, sura, aya, ayainquan

def audio_data_initialize(dir_name=audio_data_dir):
    """ Initialize audio data directory

       :param dir_name: Directory name to initialize
       :type dir_name: str
       :return: dir_name
       :rtype: str 

    """

    # If directory exists, delete the directory 
    if pathlib.Path(dir_name).exists():
        print("Directory exists, deleting :", dir_name)
        #pathlib.Path(dir_name).rmdir()
        shutil.rmtree(dir_name)

    # Create the directory
    print("Creating directory :", dir_name)
    pathlib.Path(dir_name).mkdir()

    return dir_name

def populate_audio_files(zip_data_dir=zip_data_dir, zip_file_name=zip_file_name, 
        audio_data_dir=audio_data_dir, reciters=None, suraFrom=None, suraTo=None,
        ayaFrom=None, ayaTo=None):
    """ Populate audio files for the reciters in the given directory.

       :param zip_data_dir: Directory containing the zipped data
       :type zip_data_dir: str
       :param zip_file_name: Name of the zip file in the directory
       :type zip_file_name: str
       :param audio_data_dir: Name of the download directory with the reciter zip files
       :type audio_data_dir: str
       :param reciters: Name(s) of the reciters. Only their data will be processed, rest will be ignored
       :type reciters: list of strings
       :param suraFrom: Starting sura
       :type suraFrom: int
       :param suraTo: Ending sura
       :type suraTo: int
       :param ayaFrom: Starting aya
       :type ayaFrom: int
       :param ayaTo: Ending aya
       :type ayaTo: int
       :return: None
       :rtype: None

    """

    # Convert SuraAya.mp3 name to AyaInQuran 
    audio_labels, ayainq_list = qsura_ayat_to_labels(suraFrom=suraFrom, suraTo=suraTo, 
        ayaFrom=ayaFrom, ayaTo=ayaTo)
    #print("Audio labels: ", audio_labels)
    #print("Aya in Quran labels: ", ayainq_list)

    for dd in os.listdir(zip_data_dir):
        if dd not in reciters:
            continue

        print("Found reciter: ", dd)
        # Create the directory
        reciter_dir = os.path.join(audio_data_dir, dd)
        print("Creating directory :", reciter_dir)
        pathlib.Path(reciter_dir).mkdir()

        # Each directory has one zip file called 000_versebyverse.zip
        dd_zip_file = zip_data_dir + "/" + dd + "/" + zip_file_name
        # Size
        dd_size_bytes = os.path.getsize(dd_zip_file)
        dd_size_MB = dd_size_bytes / (1024 * 1024)
        # Number of files
        archive = zipfile.ZipFile(dd_zip_file, 'r')
        num_files = len(archive.namelist())
        # Mp3 files
        mp3_cnt = 0
        for ff in archive.namelist():
            if ff.endswith('.mp3'):
                mp3_cnt += 1

        print("{:30s} {:6.0f} MB {:6d} {:12d}".format(dd, dd_size_MB, num_files, mp3_cnt))

        num_files_extracted = 0
        for lbl in audio_labels:
            mp3_file = lbl + ".mp3"
            if mp3_file not in archive.namelist():
                print("ERROR: Couldn't find file: ", mp3_file)
                continue
            archive.extract(mp3_file, path=reciter_dir)
            num_files_extracted += 1
        print("{} files extracted".format(num_files_extracted))
    print()

    return None


def extract_audio_features(reciter, mp3_file, sr=22050, n_mfcc=13, n_fft=2048, hop_length=512,
    pad_duration=None, read_duration=None, features_list=['mfcc', 'zcr', 'spectral_center', 
    'spectral_rolloff', 'chroma', 'spectral_bandwidth_2', 'spectral_bandwidth_3', 
    'spectral_bandwidth_4', 'spectral_contrast'], shp_0=None, shp_1=None, normalization=True):
    """ Extract the requested audio features.

       :param reciter: Name of the reciter
       :type reciter: str
       :param mp3_file: Name of the mp3_file
       :type mp3_file: str
       :param sr: Sampling rate to apply during audio file read with librosa
       :type sr: int
       :param n_mfcc: Number of MFCC features to return by librosa
       :type n_mfcc: int
       :param n_fft: Number of Fast Frourier Transform frequeny bins to use with librosa
       :type n_fft: int
       :param hop_length: hop_length for librosa. This says how much to overlap audio frame windows during feature extraction.
       :type hop_length: int
       :param pad_duration: Pad the duration to this number if the duration of the MP3 file is shorter
       :type pad_duration: int
       :param read_duration: Read only this much duration from the audio file
       :type read_duration: int
       :param features_list: List of features to extract
       :type features_list: list
       :param shp_0: Initialize the return data NumPy array with this shape
       :type shp_0: int
       :param shp_1: Initialize the return data NumPy array with this shape
       :type shp_1: int
       :param normalization: Normalize the MFCC data. Only works for the MFCC feature
       :type normalization: bool
       :return: columns, data, feature_shapes, new_shp_0, new_shp_1
       :rtype: list, NumPy array, list, int, int

    """

    # File name is dir/reciter/mp3_file
    file_name = os.path.join(audio_data_dir, reciter, mp3_file)

    # Initilize return variables
    columns = data = feature_shapes = new_shp_0 = new_shp_1 = None

    # Few MP3 files for few reciters were corrupted. Give a message about them & bail out
    try:
        y , sr = librosa.load(file_name, sr=sr, duration=read_duration)
        orig_duration = len(y) / sr
        #print("pad_duration = ", pad_duration)
        #print("read_duration = ", read_duration)
        #print("orig_duration = ", orig_duration)
        # Pad the duration
        if pad_duration is not None:
            if pad_duration > orig_duration:
                new_len_y = pad_duration * sr
                y = librosa.util.fix_length(y, new_len_y)
            elif pad_duration <= orig_duration:
                # Nothing to be done!
                pass
        duration = len(y) / sr
        #print("FINAL: duration = ", duration)

        # Column names
        columns = list()

        # Feature shapes
        feature_shapes = list()

        #print("shp_0 :", shp_0)
        #print("shp_1 :", shp_1)
        if shp_0 is not None and shp_1 is not None:
            if 'spect' in features_list:
                #data = np.empty(
                #    (shp_0, shp_1), dtype=np.float64
                #)
                data = np.empty(
                    (shp_1, shp_0), dtype=np.float64
                )
            else:
                data = np.zeros(
                    (shp_1, shp_0), dtype=np.float64
                )
            #data = np.empty(
            #  (0, shp_0, shp_1)
            #)
            #print("data initialized:")
            #print(type(data))
            #print(data.shape)
        else:
            data = list()
            #print(type(data))

        # Start index is 0 and gets updated after feature is concatenated to "data"
        start_idx = 0
        if 'mfcc' in features_list:
            #spect = librosa.feature.melspectrogram(y=y, sr=sr,n_fft=n_fft, hop_length=hop_length)
            mfcc = librosa.feature.mfcc(
                y=y, sr=sr, hop_length=hop_length, n_mfcc=n_mfcc, n_fft=n_fft
            )
            feature_shapes.append(mfcc.shape)
            #print("mfcc:")
            #print(mfcc.shape)
            #print("mfcc.T:")
            #print(mfcc.T)
            #print("mfcc:")
            #print(mfcc)
            #print(np.amin(mfcc), np.amax(mfcc), np.mean(mfcc))
            # Normalize?
            if normalization == True:
                divby = abs(np.amin(mfcc))
                if abs(np.amax(mfcc)) > divby:
                    divby = abs(np.amax(mfcc))

                #print("divby = ", divby)
                mfcc_orig = mfcc
                x = mfcc / divby
                #print("x = ", x)
                #x = mfcc / math.abs()
                mfcc = x
            for i in range(1, mfcc.shape[0]+1):
                columns.append('mfcc{}'.format(i))
            if shp_0 is not None:
                #data = np.append(data, [mfcc.T], axis=0)
                data[:, start_idx:start_idx+mfcc.shape[0]] = mfcc.T[0:mfcc.shape[1], :]
                start_idx += mfcc.shape[0]
                #print("mfcc start_idx updated to: ", start_idx)
        if 'zcr' in features_list:
            zcr = librosa.feature.zero_crossing_rate(y)
            feature_shapes.append(zcr.shape)
            #print("zcr:")
            #print(zcr.shape)
            #print(zcr.T)
            #print(zcr.shape[1])
            columns.append('zcr')
            if shp_0 is not None:
                data[:, start_idx:start_idx+zcr.shape[0]] = zcr.T[0:zcr.shape[1], :]
                start_idx += zcr.shape[0]
                #print("zcr start_idx updated to: ", start_idx)
        if 'spectral_center' in features_list:
            spectral_center = librosa.feature.spectral_centroid(
                y=y, sr=sr, hop_length=hop_length
            )
            feature_shapes.append(spectral_center.shape)
            #print("spectral_center:")
            #print(spectral_center.shape)
            columns.append('spectral_center')
            if shp_0 is not None:
                data[:, start_idx:start_idx+spectral_center.shape[0]] = spectral_center.T[0:spectral_center.shape[1], :]
                start_idx += spectral_center.shape[0]
                #print("spectral_center start_idx updated to: ", start_idx)
        if 'spectral_rolloff' in features_list:
            #spectral_rolloff = librosa.feature.spectral_rolloff(y+0.01, sr=sr)[0]
            spectral_rolloff = librosa.feature.spectral_rolloff(y+0.01, sr=sr)
            feature_shapes.append(spectral_rolloff.shape)
            #print("spectral_rolloff:")
            #print(spectral_rolloff.shape)
            #print(spectral_rolloff)
            columns.append('spectral_rolloff')
            if shp_0 is not None:
                data[:, start_idx:start_idx+spectral_rolloff.shape[0]] = spectral_rolloff.T[0:spectral_rolloff.shape[1], :]
                start_idx += spectral_rolloff.shape[0]
                #print("spectral_rolloff start_idx updated to: ", start_idx)
        if 'chroma' in features_list:
            chroma = librosa.feature.chroma_stft(y=y, sr=sr, hop_length=hop_length)
            feature_shapes.append(chroma.shape)
            #print("chroma:")
            #print(chroma.shape)
            for i in range(1, chroma.shape[0]+1):
                columns.append('chroma{}'.format(i))
            if shp_0 is not None:
                data[:, start_idx:start_idx+chroma.shape[0]] = chroma.T[0:chroma.shape[1], :]
                start_idx += chroma.shape[0]
                #print("chroma start_idx updated to: ", start_idx)
        if 'spectral_bandwidth_2' in features_list:
            spectral_bandwidth_2 = librosa.feature.spectral_bandwidth(y+0.01, sr=sr)
            feature_shapes.append(spectral_bandwidth_2.shape)
            #print("spectral_bandwidth_2:")
            #print(spectral_bandwidth_2.shape)
            columns.append('spectral_bandwidth_2')
            if shp_0 is not None:
                data[:, start_idx:start_idx+spectral_bandwidth_2.shape[0]] = spectral_bandwidth_2.T[0:spectral_bandwidth_2.shape[1], :]
                start_idx += spectral_bandwidth_2.shape[0]
                #print("spectral_bandwidth_2 start_idx updated to: ", start_idx)
        if 'spectral_bandwidth_3' in features_list:
            spectral_bandwidth_3 = librosa.feature.spectral_bandwidth(y+0.01, sr=sr, p=3)
            feature_shapes.append(spectral_bandwidth_3.shape)
            #print("spectral_bandwidth_3:")
            #print(spectral_bandwidth_3.shape)
            columns.append('spectral_bandwidth_3')
            if shp_0 is not None:
                data[:, start_idx:start_idx+spectral_bandwidth_3.shape[0]] = spectral_bandwidth_3.T[0:spectral_bandwidth_3.shape[1], :]
                start_idx += spectral_bandwidth_3.shape[0]
                #print("spectral_bandwidth_3 start_idx updated to: ", start_idx)
        if 'spectral_bandwidth_4' in features_list:
            spectral_bandwidth_4 = librosa.feature.spectral_bandwidth(y+0.01, sr=sr, p=4)
            feature_shapes.append(spectral_bandwidth_4.shape)
            #print("spectral_bandwidth_4:")
            #print(spectral_bandwidth_4.shape)
            columns.append('spectral_bandwidth_4')
            if shp_0 is not None:
                data[:, start_idx:start_idx+spectral_bandwidth_4.shape[0]] = spectral_bandwidth_4.T[0:spectral_bandwidth_4.shape[1], :]
                start_idx += spectral_bandwidth_4.shape[0]
                #print("spectral_bandwidth_4 start_idx updated to: ", start_idx)
        if 'spectral_contrast' in features_list:
            spectral_contrast = librosa.feature.spectral_contrast(
                y=y, sr=sr, hop_length=hop_length
            )
            feature_shapes.append(spectral_contrast.shape)
            #print("spectral_contrast:")
            #print(spectral_contrast.shape)
            #print(spectral_contrast)
            #print(spectral_contrast.T)
            for i in range(1, spectral_contrast.shape[0]+1):
                columns.append('spcontr{}'.format(i))
            if shp_0 is not None:
                data[:, start_idx:start_idx+spectral_contrast.shape[0]] = spectral_contrast.T[0:spectral_contrast.shape[1], :]
                start_idx += spectral_contrast.shape[0]
                #print("spectral_contrast start_idx updated to: ", start_idx)
        if 'spect' in features_list:
            spect = librosa.feature.melspectrogram(y=y, sr=sr,n_fft=n_fft, hop_length=hop_length)
            spect = librosa.power_to_db(spect, ref=np.max)
            feature_shapes.append(spect.shape)
            #print("spect:")
            #print(spect.shape)
            for i in range(1, spect.shape[0]+1):
                columns.append('spect{}'.format(i))
            if shp_0 is not None:
                data[:, start_idx:start_idx+spect.shape[0]] = spect.T[0:spect.shape[1], :]
                start_idx += spect.shape[0]
                #print("spect start_idx updated to: ", start_idx)

        new_shp_0 = shp_0
        new_shp_1 = shp_1
        if shp_0 is None:
            #print(feature_shapes)
            new_shp_0 = 0
            for i, shp in enumerate(feature_shapes):
                if i == 0:
                    prev_shp_1 = shp[1]
                else:
                    if shp[1] != prev_shp_1:
                        print("ERROR: shape[1] are different: {} != {}".format(shp[1], prev_shp_1))
                print("shp[0] :", shp[0])
                new_shp_0 += shp[0]
            new_shp_1 = prev_shp_1
            print("new_shp_0 :", new_shp_0)
            print("new_shp_1 :", new_shp_1)

        #print("duration :", duration)

    except:
            print("Couldn't process, skipping: ", file_name)

    return columns, data, feature_shapes, new_shp_0, new_shp_1

# Suras: Update 'Set' column with test/train/val in df
def assign_set_sura(row):
    """ Mark Suras for Test/Train/Validation (uses row.Sura column). Requires Train_/Val_/Test_Suras lists to be defined prior to calling.

       :param row: DataFrame row
       :type row: DataFrame row
       :return: None
       :rtype: None

    """

    train = Train_Suras
    val   = Val_Suras
    test  = Test_Suras
    if row.Sura in val:
        return "validation"
    elif row.Sura in test:
        return "test"
    else:
        return "train"

# Ayas: Update 'Set' column with test/train/val in df
def assign_set_aya(row):
    """ Mark Ayas for Test/Train/Validation (uses row.AyaInQuran column). Requires Train_/Val_/Test_Ayas lists to be defined prior to calling.

       :param row: DataFrame row
       :type row: DataFrame row
       :return: None
       :rtype: None

    """

    train = Train_Ayas
    val   = Val_Ayas
    test  = Test_Ayas
    if row.AyaInQuran in val:
        return "validation"
    elif row.AyaInQuran in test:
        return "test"
    else:
        return "train"

# Ayas: Update 'Set' column with test/train/val in df
def assign_set_filename(row):
    """ Mark Ayas for Test/Train/Validation (uses row.FileName column). Requires Train_/Val_/Test_Ayas lists to be defined prior to calling.

       :param row: DataFrame row
       :type row: DataFrame row
       :return: None
       :rtype: None

    """

    train = Train_Ayas
    val   = Val_Ayas
    test  = Test_Ayas
    if row.FileName in val:
        return "validation"
    elif row.FileName in test:
        return "test"
    else:
        return "train"

def gen_audio_data(df, shp0, shp1, normalization=True):
    """ Extract audio features for the given df which is a Train/Val/Test subset of the main df.

       :param df: DataFrame
       :type df: DataFrame
       :param shp_0: Initialize the return data NumPy array with this shape
       :type shp_0: int
       :param shp_1: Initialize the return data NumPy array with this shape
       :type shp_1: int
       :param normalization: Normalize the MFCC data. Only works for the MFCC feature
       :type normalization: bool
       :return: X_arr, reciters_arr
       :rtype: NumPy arr, NumPy arr

    """

    print("shp0 shp1 = ", shp0, shp1)
    X_arr = np.empty((0, shp1, shp0))
    print("X_arr initialized to :", X_arr.shape)
    reciters_arr = np.empty((0, len(list(le.classes_))))
    print("reciters_arr initialized to :", reciters_arr.shape)
    print("normalization :", normalization)

    cnt = 0
    for index, row in df.iterrows():
        cnt += 1
        ReciterName = row['ReciterName']
        FileName = row['FileName']
        # Get audio features
        columns, data, feature_shapes, new_shp_0, new_shp_1 = extract_audio_features(
                reciter=ReciterName, mp3_file=FileName, sr=sr, n_mfcc=n_mfcc, n_fft=n_fft, 
                hop_length=hop_length, pad_duration=pad_duration, read_duration=read_duration, 
                features_list=features_list, shp_0=shp0, shp_1=shp1, normalization=normalization)
        if columns == None and data == None and feature_shapes == None:
            # Skips in case of errors
            continue

        X_arr = np.append(X_arr, [data], axis=0)
                            
        reciters_list = [0 for i in range(0, len(list(le.classes_)))]
        reciters_index = list(le.transform([ReciterName]))[0]
        reciters_list[reciters_index] = 1
        reciters_arr = np.append(reciters_arr, [reciters_list], axis=0)
            
        if cnt % 100 == 0:
            print("Processed ", cnt)
        #if cnt == 10:
        #    break

    return X_arr, reciters_arr

def filter_duration(row):
    """ Finds the same FileName for all the selected_reciters (uses row.FileName/row.ReciterName columns). Uses selected_reciters variable to look for the recieter, so it needs to be defined.

       :param row: DataFrame row
       :type row: DataFrame row
       :return: 'Yes' or 'NaN'
       :rtype: str

    """

    my_df = df_tmp
    FileName = row.FileName
    #print("FileName =", FileName)
    not_found = False
    for rec in selected_reciters:
        #print("  rec =", rec)
        if ((my_df['ReciterName'] == rec) & (my_df['FileName'] == FileName)).any():
            pass
        else:
            not_found = True
            #print("not_found =", not_found)
            break
    
    if not_found == True:
        return 'NaN'
    else:
        return 'Yes'

print("helpers.py LOADED!")
# End of helpers.py