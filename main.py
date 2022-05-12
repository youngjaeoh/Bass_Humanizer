import pandas
import torch
import torchaudio
import pandas as pd
import crepe
from scipy.io import wavfile
import numpy as np
import matplotlib.pyplot as plt
from pydub import AudioSegment
import os
import io
import requests
import tarfile
import boto3
from botocore import UNSIGNED
from botocore.config import Config
import matplotlib.pyplot as plt
from IPython.display import Audio, display
import sys
import librosa
import pyworld as pw
import numpy as np
from numpy.fft import fft, ifft
import matplotlib.pyplot as plt
# %matplotlib inline
import librosa
from scipy.io import wavfile
from scipy.interpolate import interp1d
import numpy as np
import matplotlib.pyplot as plt
import dtcwt.tf
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import librosa.display


def reset_wav(directory):
    waveform, sample_rate = torchaudio.load(directory)
    torchaudio.save('temp01.wav', waveform, sample_rate)

def load_wav(directory):
    waveform, sample_rate = torchaudio.load(directory)
    return waveform, sample_rate

def crepe_func(directory):
    sr, audio = wavfile.read(directory)
    # step size 10 for default
    # model capacity 'tiny', 'small', 'medium', 'large', 'full'
    time, frequency, confidence, activation = crepe.predict(audio, sr, step_size=0.1, model_capacity='full', viterbi=True)
    return time, frequency, confidence, activation

def read_csv(directory):
    data = pd.read_csv(directory)
    return data

def plt_plot(time, frequency, title, plot_dir, plotsave):
    plt.plot(time, frequency)
    plt.xlabel(time)
    plt.ylabel(frequency)
    plt.title(title)
    if plotsave == True:
        plt.savefig(plot_dir + ".png")
    plt.show()

def make_df(time, frequency, confidence):
    raw_data = {'time': time,
                'frequency': frequency,
                'confidence': confidence.flatten()}
    df = pd.DataFrame(raw_data)
    return df

def analysis(directory):
    ##############################################################################
    # directory = directory
    # time, frequency, confidence, activation = crepe_func(directory)
    # df = make_df(time, frequency, confidence)
    # df.to_pickle("use_this")
    ##############################################################################

    ##############################################################################
    df = pd.read_pickle("use_this")
    time = list()
    frequency = list()

    for i in df.index:
        time.append(df['time'][i])
        frequency.append(df['frequency'][i])
    ##############################################################################

    df_result = pd.DataFrame(index=range(0, ), columns=["time", "frequency", "original_index"])
    for i in df.index:
        if i != 0:
            if (abs(df['frequency'][i] - df['frequency'][i - 1])) > 5:
                new_data = {'time': df['time'][i], 'frequency': df['frequency'][i], 'original_index': i}
                df_result = df_result.append(new_data, ignore_index=True)

    df_result_only_front = pd.DataFrame(index=range(0, ), columns=["time", "frequency"])
    for i in df_result.index:
        number_index = df_result['original_index'][i]
        if abs(df_result['frequency'][i] - df['frequency'][number_index + 260]) < 2.6:
            new_data = {'time': df_result['time'][i], 'frequency': df_result['frequency'][i]}
            df_result_only_front = df_result_only_front.append(new_data, ignore_index=True)
            # plt.scatter(df_result['time'][i], df_result['frequency'][i], c='r', s=20)

    df_final = pd.DataFrame(index=range(0, ), columns=["time", "frequency"])
    for i in df_result_only_front.index:
        if i != 0:
            if abs(df_result_only_front['time'][i] - df_result_only_front['time'][i - 1]) > 0.1:
                new_data = {'time': df_result_only_front['time'][i], 'frequency': df_result_only_front['frequency'][i]}
                df_final = df_final.append(new_data, ignore_index=True)
                plt.scatter(df_result_only_front['time'][i], df_result_only_front['frequency'][i], c='g', s=20)
        else:
            new_data = {'time': df_result_only_front['time'][i], 'frequency': df_result_only_front['frequency'][i]}
            df_final = df_final.append(new_data, ignore_index=True)
            plt.scatter(df_result_only_front['time'][i], df_result_only_front['frequency'][i], c='g', s=20)

    plt.plot(time, frequency)
    plt.xlabel(time)
    plt.ylabel(frequency)
    plt.show()


    # save_pickle(df_final)

    return df_final

def save_pickle(df):
    df_result = df
    if os.path.exists("df_final") == False:
        df_result.to_pickle("df_final")
    else:
        os.remove("df_final")
        df_result.to_pickle("df_final")

def read_noises():
    v_50 = AudioSegment.from_file("v50.wav", format="wav")
    v_75 = AudioSegment.from_file("v75.wav", format="wav")
    v_100 = AudioSegment.from_file("v100.wav", format="wav")
    return v_50, v_75, v_100

def overlay_audio(df_result):
    v_50, v_75, v_100 = read_noises()
    # dB control
    v_50 = v_50 - 25

    for i in df_result.index:
        time = df_result["time"][i]
        time_control = (time * 1000) - 50

        if i == 0:
            print("noise added")
            print("time: ", time)
            tmp_audio = AudioSegment.from_file("temp.wav", format="wav")
            played_together = tmp_audio.overlay(v_50, position=time_control)
            name = "new_result_{}.wav".format(i)
            output_wav = played_together.export(name, format="wav")
        else:
            print("noise added")
            print("time: ", time)
            tmp_audio = AudioSegment.from_file(name, format="wav")
            played_together = tmp_audio.overlay(v_50, position=time_control)
            name = "new_result_{}.wav".format(i)
            output_wav = played_together.export(name, format="wav")

def plot_waveform(waveform, sample_rate, title="Waveform", xlim=None, ylim=None):
    waveform = waveform.numpy()

    num_channels, num_frames = waveform.shape
    time_axis = torch.arange(0, num_frames) / sample_rate

    figure, axes = plt.subplots(num_channels, 1)
    if num_channels == 1:
        axes = [axes]
    for c in range(num_channels):
        axes[c].plot(time_axis, waveform[c], linewidth=1)
        axes[c].grid(True)
        if num_channels > 1:
            axes[c].set_ylabel(f'Channel {c + 1}')
        if xlim:
            axes[c].set_xlim(xlim)
        if ylim:
            axes[c].set_ylim(ylim)
    figure.suptitle(title)
    plt.show(block=False)

def plot_specgram(waveform, sample_rate, title="Spectrogram", xlim=None):
    waveform = waveform.numpy()

    num_channels, num_frames = waveform.shape
    time_axis = torch.arange(0, num_frames) / sample_rate

    figure, axes = plt.subplots(num_channels, 1)
    if num_channels == 1:
        axes = [axes]
    for c in range(num_channels):
        axes[c].specgram(waveform[c], Fs=sample_rate)
        if num_channels > 1:
            axes[c].set_ylabel(f'Channel {c + 1}')
        if xlim:
            axes[c].set_xlim(xlim)
    figure.suptitle(title)
    plt.show(block=False)

def f0_extract_initiator(direc, ):
    directory = direc

def analysis_for_whole_dataset(directory, velo, index_num, tn):
    print("analysis for whole dataset started!!!!!!!\n")

    save_name = "{}_{}".format(tn, velo)
    if index_num < 10:
        index_num = "_0{}".format(index_num)
        save_name += index_num
    else:
        index_num = "_{}".format(index_num)
        save_name += index_num
    df_name = "{}_f0_ss01/".format(tn) + save_name

    if os.path.exists(df_name) == False:
        directory = directory
        time, frequency, confidence, activation = crepe_func(directory)
        raw_data = {'time': time,
                    'frequency': frequency,
                    'confidence': confidence}
        df = pd.DataFrame(raw_data)

        df.to_pickle(df_name)
    else:
        print("already exists!!! passing {}".format(save_name))
    print("{} dataframe to pickle saved!!!!!".format(df_name))
    print("analysis for whole dataset finished!!!")

def analysis_for_whole_dataset_initiator():
    # typeno 1 = original // 2 = glissup // 3 = slidedown // 4 = slideup
    typeno = [1, 2, 3, 4]
    for t in typeno:
        if t == 1:
            tn = "bass_original"
            velocity_list = [50, 75, 100, 125]
        elif t == 2:
            tn = "bass_glissup"
            velocity_list = [25, 50, 75, 100, 125]
        elif t == 3:
            tn = "bass_slidedown"
            velocity_list = [25, 50, 75, 100, 125]
        elif t == 4:
            tn = "bass_slideup"
            velocity_list = [25, 50, 75, 100, 125]

        for i in velocity_list:
            velocity_name = "_{}_".format(i)
            for j in range(1, 51):
                directory = "/Users/tebe/Desktop/bass_dataset/{}/{}".format(tn, tn) + velocity_name
                if j < 10:
                    index_number = "0{}".format(j)
                else:
                    index_number = "{}".format(j)
                directory = directory + index_number + ".wav"
                print(directory)
                analysis_for_whole_dataset(directory, i, j, tn)

def phase_vocode(coeffs, pitchFactor):

    mag = np.abs(coeffs)
    phase = np.angle(coeffs)
    phase = np.unwrap(phase)

    mag_t = np.linspace(0, 1, mag.shape[0])
    phase_t = np.linspace(0, 1, phase.shape[0])

    if isinstance(pitchFactor, np.ndarray):
        tp = np.linspace(0, 1, num=pitchFactor.shape[0])
        ip = interp1d(tp, pitchFactor, kind='cubic', fill_value='extrapolate')
        mag_s = mag_t * ip(mag_t)
        phase_s = phase_t * ip(phase_t)
    else:
        mag_s = mag_t * pitchFactor
        phase_s = phase_t * pitchFactor


    Ymag = np.transpose(
        np.array([interp1d(mag_s, mag[:,i], fill_value=mag[-1,i], bounds_error=False)(mag_t)
             for i in range(mag.ndim) if mag.shape[i] >= mag_t.shape[0]]))

    Yphases = np.transpose(
        np.array(
            [interp1d(phase_s, phase[:,i], fill_value=phase[-1,i],bounds_error=False)(phase_t)
             for i in range(phase.ndim) if phase.shape[i] >= phase_t.shape[0]]))

    amount = Ymag.shape[0] - Yphases.shape[0]
    if amount < 0:
        Ymag = np.pad(Ymag, ((0, abs(amount)), (0,0)))
    else:
        Yphases = np.pad(Yphases, ((0, amount), (0,0)))

    result = Ymag * np.exp(-1j * Yphases)
    return result

def pv_with_transform(waveform, pitchFactor, adjustHighpasses=True):
    transform = dtcwt.Transform1d()

    tfd = transform.forward(waveform, nlevels=1)
    new_lowpass = phase_vocode(tfd.lowpass, pitchFactor)

    # make sure length is correct.
    ld = tfd.lowpass.shape[0] - new_lowpass.shape[0]
    if ld > 0:
        tfd.lowpass = np.pad(new_lowpass, ((0, ld), (0, 0)))
    else:
        tfd.lowpass = new_lowpass[:(tfd.lowpass.shape[0])]

    if adjustHighpasses:
        new_highpasses = [phase_vocode(tfd.highpasses[i], pitchFactor)
                          for i in range(len(tfd.highpasses))]
        hds = [tfd.highpasses[i].shape[0] - new_highpasses[i].shape[0] for i in range(len(new_highpasses))]
        for i in range(len(hds)):
            if hds[i] > 0:
                new_highpasses[i] = np.pad(new_highpasses[i], ((0, hds[i]), (0, 0)))
            else:
                new_highpasses[i] = new_highpasses[i][:(tfd.highpasses[i].shape[0])]
        tfd.highpasses = tuple(new_highpasses)
    return transform.inverse(tfd)

def amplify(orig, shifted):
    maxx = np.max(np.abs(orig))
    maxy = np.max(np.abs(shifted))
    return (shifted / maxy) * maxx

def bend_helper(times, pitches, seconds=0.1, sr=44100, kind='linear'):

    pitches = [2**(-semitones/12) for semitones in pitches]
    t = np.linspace(0, 1, num=int(seconds*sr))
    y = interp1d(times, pitches, kind, fill_value='extrapolate')
    return y(t)

def pitch_bending(soundfile, times, pitches, seconds=0.1, sr=44100, kind='linear'):
    pitches = [2 ** (-semitones / 12) for semitones in pitches]
    t = np.linspace(0, 1, num=int(seconds * sr))
    y = interp1d(times, pitches, kind, fill_value='extrapolate')
    pitchFactor = y(t)
    waveform = soundfile
    transform = dtcwt.Transform1d()

    tfd = transform.forward(waveform, nlevels=1)
    new_lowpass = phase_vocode(tfd.lowpass, pitchFactor)

    ld = tfd.lowpass.shape[0] - new_lowpass.shape[0]
    if ld > 0:
        tfd.lowpass = np.pad(new_lowpass, ((0, ld), (0, 0)))
    else:
        tfd.lowpass = new_lowpass[:(tfd.lowpass.shape[0])]

    new_highpasses = [phase_vocode(tfd.highpasses[i], pitchFactor)
                      for i in range(len(tfd.highpasses))]
    hds = [tfd.highpasses[i].shape[0] - new_highpasses[i].shape[0] for i in range(len(new_highpasses))]
    for i in range(len(hds)):
        if hds[i] > 0:
            new_highpasses[i] = np.pad(new_highpasses[i], ((0, hds[i]), (0, 0)))
        else:
            new_highpasses[i] = new_highpasses[i][:(tfd.highpasses[i].shape[0])]
    tfd.highpasses = tuple(new_highpasses)

    return transform.inverse(tfd)



def pitch_bend_initiator():
    infile = "test.wav"
    outfile = "output10.wav"

    sr, orig = wavfile.read(infile)
    bent = amplify(orig, pv_with_transform(orig, bend_helper([0, 0.25, 0.5, 0.7, 1], [-2, -1, -0.3, -0.2, -0.1])))

    wavfile.write(outfile, sr, bent.astype(np.int16))

def plot_time_frequency(df):
    time = list()
    frequency = list()
    for i in df.index:
        time.append(df['time'][i])
        frequency.append(df['frequency'][i])
    plt_plot(time, frequency)

def save_pickle_for_crop(df, name):
    if os.path.exists(name) == False:
        df.to_pickle(name)
    else:
        os.remove(name)
        df.to_pickle(name)

def df_cropper(articulation):
    vel = [25, 50, 75, 100, 125]
    for i in vel:
        for j in range(1, 51):
            if j < 10:
                directory = "bass_{}_f0_ss1/bass_{}_{}_0{}".format(articulation, articulation, i, j)
            else:
                directory = "bass_{}_f0_ss1/bass_{}_{}_{}".format(articulation, articulation, i, j)
            df = pd.read_pickle(directory)
            df.drop(labels=range(200, 4031), axis=0, inplace=True)
            save_dir = "cropped_glissup/" + directory[20:]
            save_pickle_for_crop(df, save_dir)

def see_only_plots():
    vel = [25, 50, 75, 100, 125]
    for i in vel:
        for j in range(1, 51):
            if j < 10:
                directory = "cropped_glissup/bass_glissup_{}_0{}".format(i, j)
            else:
                directory = "cropped_glissup/bass_glissup_{}_{}".format(i, j)
            print(directory)
            df = pd.read_pickle(directory)

            time = list()
            frequency = list()
            for k in df.index:
                time.append(df['time'][k])
                frequency.append(df['frequency'][k])
            title = "glissup_{}_{}".format(i, j)
            plot_dir = "plots/{}".format(title)
            plt_plot(time, frequency, title, plot_dir, plotsave=True)

def translated_dataframes():
    path = "cropped_glissup/usable"
    file_names = os.listdir(path)
    # print(file_names)

    df_result = pd.read_pickle("df_temp")
    df_result.drop("frequency", axis=1, inplace=True)
    # print(df_result)

    for q in file_names:
        df = pd.read_pickle("cropped_glissup/usable/{}".format(q))
        save_list = [0, 9, 19, 29, 39, 49, 59, 69, 79, 89, 99, 109, 119, 129, 139, 149, 159, 169, 179, 189, 199]
        translation = list()
        for i in range(200):
            if i not in save_list:
                df.drop(i, axis=0, inplace=True)
        df.reset_index(drop=True, inplace=True)
        for i in df.index:
            old_value = round(df["frequency"][i], 2)
            old_min = round(df["frequency"][df["frequency"].argmin()], 2)
            old_max = round(df["frequency"][df["frequency"].argmax()], 2)
            new_min = -2
            new_max = 0
            new_value = round((((old_value - old_min) / (old_max - old_min)) * (new_max - new_min) + new_min), 2)
            translation.append(new_value)
        df_result["{}".format(q)] = translation
        # df_result.drop("frequency", axis=1, inplace=True)
    df_result.to_pickle("cropped_glissup/for_dl/df_result")

def librosa_rms_count():
    y, sr = librosa.load("/Users/tebe/Desktop/bass_dataset/bass_original/bass_original_125_28.wav")
    S, phase = librosa.magphase(librosa.stft(y, n_fft=256))
    rms = librosa.feature.rms(S=S, frame_length=256)
    times = librosa.times_like(rms)
    print(rms)
    print(rms.shape)
    print(times)
    print(times.shape)


    # fig, ax = plt.subplots(figsize=(15, 6), nrows=2, sharex=True)
    # times = librosa.times_like(rms)
    # ax[0].semilogy(times, rms[0], label='RMS Energy')
    # ax[0].set(xticks=[])
    # ax[0].legend()
    # ax[0].label_outer()
    # librosa.display.specshow(librosa.amplitude_to_db(S, ref=np.max),
    #                          y_axis='log', x_axis='time', ax=ax[1])
    # ax[1].set(title='log Power spectrogram')
    # plt.show()

def gliss_part_whole():
    ##########GLISS!!!!!##################################################################
    ##########자르는 부분 끝난 부분##################################################################
    df = pd.read_pickle("use_this")
    df.drop("confidence", axis=1, inplace=True)
    frequency = list()
    for i in df.index:
        frequency.append(df['frequency'][i])
    frequency = np.around(frequency)
    df.drop("frequency", axis=1, inplace=True)
    df['frequency'] = frequency

    time = list()
    frequency = list()

    for i in df.index:
        time.append(df['time'][i])
        frequency.append(df['frequency'][i])

    frequency = np.array(frequency)
    index_temp_frequency = np.where(frequency<120)
    temp_frequency = frequency[np.where(frequency<120)]
    index_new_frequency = np.where(frequency>115)
    index_new_frequency = index_new_frequency[0]
    new_frequency = temp_frequency[np.where(temp_frequency>115)]
    print(index_new_frequency)
    print(new_frequency)

    new_time = list()
    for i in index_new_frequency:
        new_time.append(time[i])
    print(new_time)

    """
    시작 시간 = 1.07
    끝 시간 = 1.606
    """
    ##########자르는 부분 끝난 부분##################################################################



    ####### 피치 밴드 완벽하게 다 끝난 부분 ##################################################################################
    pd.set_option('display.max_columns', None)
    pd.set_option('display.max_rows', None)

    df = pd.read_pickle("cropped_glissup/for_dl/df_result")
    df_time = df[["time"]]
    print(df_time)
    time_list = list()
    for i in df_time.index:
        old_value = round(df_time["time"][i], 2)
        old_min = round(df_time["time"][df_time["time"].argmin()], 2)
        old_max = round(df_time["time"][df_time["time"].argmax()], 2)
        new_min = 0
        new_max = 1
        new_value = round((((old_value - old_min) / (old_max - old_min)) * (new_max - new_min) + new_min), 2)
        time_list.append(new_value)
    print(time_list)

    df.drop("time", axis=1, inplace=True)
    model_df = round((df.mean(axis='columns')), 2)
    print(model_df)
    model_list = list()
    for i in model_df.index:
        model_list.append(model_df[i])

    # model_list = np.array(model_list)
    # model_list = model_list + 2

    print(model_list)
    print(time_list)

    # new_model_list = [-1.94, -1.76, -1.49, -1.05, -0.62, -0.34, -0.14, -0.06, -0.05, -0.03, -0.02]
    # new_time = np.arange(0, 1.1, 0.1)
    # # new_time = np.round_(new_time.tolist(),1)
    # # print(new_time)

    original_sound = AudioSegment.from_file("temp.wav")
    cropped_sound = original_sound[1070:1606]
    before_crop = original_sound[:1070]
    after_crop = original_sound[1606:]

    cropped_sound = cropped_sound.export("for_crop/cropped_sound.wav", format="wav")
    before_crop = before_crop.export("for_crop/before_crop.wav", format="wav")
    after_crop = after_crop.export("for_crop/after_crop.wav", format="wav")

    infile = "for_crop/cropped_sound.wav"
    outfile = "for_crop/cropped_sound_glissup.wav"
    sr, orig = wavfile.read(infile)

    # orig = np.append(orig, np.array([[0, 0]]), axis=0)
    orig = np.append(orig, np.array([0]), axis=0)

    bent = amplify(orig, pv_with_transform(orig, bend_helper(time_list, model_list)))

    wavfile.write(outfile, sr, bent.astype(np.int16))

    before_crop = AudioSegment.from_file("for_crop/before_crop.wav")
    glissed_up = AudioSegment.from_file("for_crop/cropped_sound_glissup.wav")
    after_crop = AudioSegment.from_file("for_crop/after_crop.wav")

    combined_with_crossfade = before_crop.append(glissed_up, crossfade=1)
    combined_with_crossfade_after = combined_with_crossfade.append(after_crop, crossfade=1)
    result = combined_with_crossfade_after.export("for_crop/result.wav", format="wav")
    # print(model_list)

def pitch_shift_and_put_together_whole():
    ### 메인에다가 그냥 가져다 복사해서 돌리면 된다.
    np.set_printoptions(threshold=sys.maxsize)

    pd.set_option('display.max_columns', None)
    pd.set_option('display.max_rows', None)
    df = pd.read_pickle("bass_slidedown_f0_ss1/bass_slidedown_75_42")
    # save_list = [0, 9, 19, 29, 39, 49, 59, 69, 79, 89, 99, 109, 119, 129, 139, 149, 159, 169, 179, 189, 199]
    save_list = np.arange(0, 818, 40)
    save_list = save_list.tolist()
    # print(save_list)

    df.drop(labels=range(819, 4031), axis=0, inplace=True)
    # print(df)

    for i in range(818):
        if i not in save_list:
            df.drop(i, axis=0, inplace=True)
    # print(df)
    df.reset_index(drop=True, inplace=True)

    df_time = df[["time"]]
    time_list = list()
    for i in df_time.index:
        old_value = round(df_time["time"][i], 2)
        old_min = round(df_time["time"][df_time["time"].argmin()], 2)
        old_max = round(df_time["time"][df_time["time"].argmax()], 2)
        new_min = 0
        new_max = 1
        new_value = round((((old_value - old_min) / (old_max - old_min)) * (new_max - new_min) + new_min), 2)
        time_list.append(new_value)
    df["timechange"] = time_list

    translation = list()
    for i in df.index:
        old_value = round(df["frequency"][i], 2)
        old_min = round(df["frequency"][df["frequency"].argmin()], 2)
        old_max = round(df["frequency"][df["frequency"].argmax()], 2)
        new_min = -6  # 40 세미톤이나 떨어짐...!!!!!!!!
        new_max = 0
        new_value = round((((old_value - old_min) / (old_max - old_min)) * (new_max - new_min) + new_min), 2)
        translation.append(new_value)
    df["slideup"] = translation
    print(translation)

    file_df = pd.read_pickle("use_this")
    time = list()
    frequency = list()
    for i in file_df.index:
        time.append(file_df["time"][i])
        frequency.append(file_df["frequency"][i])
    frequency = np.around(frequency)

    file_df.drop("frequency", axis=1, inplace=True)
    file_df["frequency"] = frequency

    print(file_df)

    frequency = np.array(frequency)
    index_temp_frequency = np.where(frequency < 107)
    temp_frequency = frequency[np.where(frequency < 107)]
    index_new_frequency = np.where(frequency > 99)
    index_new_frequency = index_new_frequency[0]
    new_frequency = temp_frequency[np.where(temp_frequency > 99)]
    # print(index_new_frequency)
    # print(new_frequency)

    """
    시작: 3.651초
    끝: 4.232초

    시작: 1.864
    끝: 2.148

    시작: 6.363
    끝:  6.741
    """

    original_sound = AudioSegment.from_file("temp.wav")
    cropped_sound = original_sound[6363:6741]
    before_crop = original_sound[:6363]
    after_crop = original_sound[6741:]

    cropped_sound = cropped_sound.export("for_crop/slidedown/cropped_sound.wav", format="wav")
    before_crop = before_crop.export("for_crop/slidedown/before_crop.wav", format="wav")
    after_crop = after_crop.export("for_crop/slidedown/after_crop.wav", format="wav")

    infile = "for_crop/slidedown/cropped_sound.wav"
    outfile = "for_crop/slidedown/cropped_sound_slidedown.wav"
    sr, orig = wavfile.read(infile)

    # orig = np.append(orig, np.array([[0, 0]]), axis=0)
    # orig = np.append(orig, np.array([0]), axis=0)

    new_translation = [0, 0, 0, 0, 0, 0, 0, 0, 0, -0.14, -0.36, -0.73, -1.11, -1.45, -1.76, -2.19, -2.48, -2.73, -3.0,
                       -3.46, -3.46, -4.16]

    bent = amplify(orig, pv_with_transform(orig, bend_helper(time_list, new_translation)))

    wavfile.write(outfile, sr, bent.astype(np.int16))

    before_crop = AudioSegment.from_file("for_crop/slidedown/before_crop.wav")
    slide_down = AudioSegment.from_file("for_crop/slidedown/cropped_sound_slidedown.wav")
    after_crop = AudioSegment.from_file("for_crop/slidedown/after_crop.wav")

    # slide_down_fade = slide_down.fade(to_gain=-30, end=0, duration=10)
    # slide_down_fade = slide_down_fade.export("for_crop/slidedown/after_crop_fade.wav", format="wav")

    # slide_down_final = AudioSegment.from_file("for_crop/slidedown/after_crop_fade.wav")

    combined_with_crossfade = before_crop.append(slide_down, crossfade=1)
    combined_with_crossfade_after = combined_with_crossfade.append(after_crop, crossfade=1)
    result = combined_with_crossfade_after.export("for_crop/slidedown/result.wav", format="wav")

    # print(translation)

    #################################### slide up 끝남

    ##########GLISS!!!!!##################################################################
    ##########자르는 부분 끝난 부분##################################################################
    df = pd.read_pickle("use_this")
    df.drop("confidence", axis=1, inplace=True)
    frequency = list()
    for i in df.index:
        frequency.append(df['frequency'][i])
    frequency = np.around(frequency)
    df.drop("frequency", axis=1, inplace=True)
    df['frequency'] = frequency

    time = list()
    frequency = list()

    for i in df.index:
        time.append(df['time'][i])
        frequency.append(df['frequency'][i])

    frequency = np.array(frequency)
    index_temp_frequency = np.where(frequency < 120)
    temp_frequency = frequency[np.where(frequency < 120)]
    index_new_frequency = np.where(frequency > 115)
    index_new_frequency = index_new_frequency[0]
    new_frequency = temp_frequency[np.where(temp_frequency > 115)]
    print(index_new_frequency)
    print(new_frequency)

    new_time = list()
    for i in index_new_frequency:
        new_time.append(time[i])
    print(new_time)

    """
    시작 시간 = 1.07
    끝 시간 = 1.606
    """
    ##########자르는 부분 끝난 부분##################################################################

    ####### 피치 밴드 완벽하게 다 끝난 부분 ##################################################################################
    pd.set_option('display.max_columns', None)
    pd.set_option('display.max_rows', None)

    df = pd.read_pickle("cropped_glissup/for_dl/df_result")
    df_time = df[["time"]]
    print(df_time)
    time_list = list()
    for i in df_time.index:
        old_value = round(df_time["time"][i], 2)
        old_min = round(df_time["time"][df_time["time"].argmin()], 2)
        old_max = round(df_time["time"][df_time["time"].argmax()], 2)
        new_min = 0
        new_max = 1
        new_value = round((((old_value - old_min) / (old_max - old_min)) * (new_max - new_min) + new_min), 2)
        time_list.append(new_value)
    print(time_list)

    df.drop("time", axis=1, inplace=True)
    model_df = round((df.mean(axis='columns')), 2)
    print(model_df)
    model_list = list()
    for i in model_df.index:
        model_list.append(model_df[i])

    # model_list = np.array(model_list)
    # model_list = model_list + 2

    print(model_list)
    print(time_list)

    # new_model_list = [-1.94, -1.76, -1.49, -1.05, -0.62, -0.34, -0.14, -0.06, -0.05, -0.03, -0.02]
    # new_time = np.arange(0, 1.1, 0.1)
    # # new_time = np.round_(new_time.tolist(),1)
    # # print(new_time)

    original_sound = AudioSegment.from_file("for_crop/slidedown/result.wav")
    cropped_sound = original_sound[1070:1606]
    before_crop = original_sound[:1070]
    after_crop = original_sound[1606:]

    cropped_sound = cropped_sound.export("for_crop/cropped_sound.wav", format="wav")
    before_crop = before_crop.export("for_crop/before_crop.wav", format="wav")
    after_crop = after_crop.export("for_crop/after_crop.wav", format="wav")

    infile = "for_crop/cropped_sound.wav"
    outfile = "for_crop/cropped_sound_glissup.wav"
    sr, orig = wavfile.read(infile)

    # orig = np.append(orig, np.array([[0, 0]]), axis=0)
    orig = np.append(orig, np.array([0]), axis=0)

    bent = amplify(orig, pv_with_transform(orig, bend_helper(time_list, model_list)))

    wavfile.write(outfile, sr, bent.astype(np.int16))

    before_crop = AudioSegment.from_file("for_crop/before_crop.wav")
    glissed_up = AudioSegment.from_file("for_crop/cropped_sound_glissup.wav")
    after_crop = AudioSegment.from_file("for_crop/after_crop.wav")

    combined_with_crossfade = before_crop.append(glissed_up, crossfade=1)
    combined_with_crossfade_after = combined_with_crossfade.append(after_crop, crossfade=1)
    result = combined_with_crossfade_after.export("for_crop/result.wav", format="wav")
    # print(model_list)

def main():

    # ############### 여기까지 포인트들 완벽하게 잡음!!!!!!!!######################
    # df = pd.read_pickle("use_this")
    # # print(df)
    #
    # time = list()
    # frequency = list()
    # for i in df.index:
    #     time.append(df["time"][i])
    #     frequency.append(df["frequency"][i])
    # frequency = np.around(frequency)
    # df.drop("confidence", axis=1, inplace=True)
    # df.drop("frequency", axis=1, inplace=True)
    # df['frequency'] = frequency
    # # print(df)
    #
    # df_final = pd.DataFrame(index=range(0, ), columns=["time", "frequency"])
    #
    # for i in df.index:
    #     if i > 70:
    #         if df["frequency"][i] == df["frequency"][i-70]:
    #             if df["frequency"][i] != 0:
    #                 # plt.scatter(df['time'][i], df['frequency'][i], c='g', s=20)
    #
    #                 new_data = {'time': df['time'][i],
    #                             'frequency': df['frequency'][i]}
    #                 df_final = df_final.append(new_data, ignore_index=True)
    #
    # df_final.reset_index(drop=True, inplace=True)
    # print(df_final)
    #
    # df_result = pd.DataFrame(index=range(0, ), columns=["time", "frequency"])
    #
    # for i in df_final.index:
    #     if i > 0:
    #         if df_final["frequency"][i] != df_final["frequency"][i-1]:
    #             new_data = {'time': df_final['time'][i],
    #                         'frequency': df_final['frequency'][i]}
    #             df_result = df_result.append(new_data, ignore_index=True)
    #     else:
    #         temp = df_final["frequency"][i]
    #         new_data = {'time': df_final['time'][i],
    #                     'frequency': df_final['frequency'][i]}
    #         df_result = df_result.append(new_data, ignore_index=True)
    # print(df_result)
    #
    # # for i in df_result.index:
    # #     plt.scatter(df_result['time'][i], df_result['frequency'][i], c='r', s=20)
    #
    # df_result2 = pd.DataFrame(index=range(0, ), columns=["time", "frequency"])
    # for i in df_result.index:
    #     if i > 0:
    #         if abs(df_result["frequency"][i] - df_result["frequency"][i-1]) > 3:
    #             new_data = {'time': df_result['time'][i],
    #                         'frequency': df_result['frequency'][i]}
    #             df_result2 = df_result2.append(new_data, ignore_index=True)
    #     else:
    #         temp = df_result["frequency"][i]
    #         new_data = {'time': df_result['time'][i],
    #                     'frequency': df_result['frequency'][i]}
    #         df_result2 = df_result2.append(new_data, ignore_index=True)
    # print(df_result2)
    #
    # for i in df_result2.index:
    #     plt.scatter(df_result2['time'][i], df_result2['frequency'][i], c='r', s=20)
    #
    # difference_list = list()
    # for i in df_result2.index:
    #     if i == 0:
    #         difference = 100
    #         difference_list.append(difference)
    #     else:
    #         difference = abs(df_result2["frequency"][i] - df_result2["frequency"][i-1])
    #         difference_list.append(difference)
    # df_result2["difference"] = difference_list
    # print(df_result2)
    #
    # plt.plot(time, frequency)
    # plt.xlabel(time)
    # plt.ylabel(frequency)
    # plt.show()
    #
    # ############### 여기까지 포인트들 완벽하게 잡음!!!!!!!!######################
    #
    #
    # # v_50 = AudioSegment.from_file("v50.wav", format="wav")
    # # v_75 = AudioSegment.from_file("v75.wav", format="wav")
    # # v_100 = AudioSegment.from_file("v100.wav", format="wav")
    #
    # s10_v10 = AudioSegment.from_file("noises/s10_v10.wav", format="wav")
    # s10_v50 = AudioSegment.from_file("noises/s10_v50.wav", format="wav")
    # s100 = AudioSegment.from_file("noises/s100.wav", format="wav")
    # v_75 = AudioSegment.from_file("noises/v_75.wav", format="wav")
    # v_50 = AudioSegment.from_file("noises/v_50.wav", format="wav")
    #
    # # dB control
    # s10_v10 = s10_v10 - 20 # original = 30
    # s10_v50 = s10_v50 - 20
    # s100 = s100 - 20
    # v_75 = v_75 - 20
    # v_50 = v_50 - 20
    #
    # for i in df_result2.index:
    #     print(i)
    #     time = df_result2["time"][i]
    #     time_control = (time * 1000) - 150
    #
    #     if i == 0:
    #         if df_result2['difference'][i] >= 75:
    #             print("noise added")
    #             print("time: ", time)
    #             # tmp_audio = AudioSegment.from_file("temp.wav", format="wav") # 이건 virgin용
    #             tmp_audio = AudioSegment.from_file("for_crop/result.wav", format="wav")
    #             played_together = tmp_audio.overlay(s100, position=time_control)
    #             name = "new_result_{}.wav".format(i)
    #             output_wav = played_together.export(name, format="wav")
    #         elif 50 <= df_result2['difference'][i] < 75:
    #             print("noise added")
    #             print("time: ", time)
    #             tmp_audio = AudioSegment.from_file("for_crop/result.wav", format="wav")
    #             played_together = tmp_audio.overlay(s10_v50, position=time_control)
    #             name = "new_result_{}.wav".format(i)
    #             output_wav = played_together.export(name, format="wav")
    #         elif 25 <= df_result2['difference'][i] < 50:
    #             print("noise added")
    #             print("time: ", time)
    #             tmp_audio = AudioSegment.from_file("for_crop/result.wav", format="wav")
    #             played_together = tmp_audio.overlay(v_50, position=time_control)
    #             name = "new_result_{}.wav".format(i)
    #             output_wav = played_together.export(name, format="wav")
    #         elif df_result2['difference'][i] < 25:
    #             print("noise added")
    #             print("time: ", time)
    #             tmp_audio = AudioSegment.from_file("for_crop/result.wav", format="wav")
    #             played_together = tmp_audio.overlay(s10_v10, position=time_control)
    #             name = "new_result_{}.wav".format(i)
    #             output_wav = played_together.export(name, format="wav")
    #     else:
    #         if df_result2['difference'][i] >= 75:
    #             print("noise added")
    #             print("time: ", time)
    #             tmp_audio = AudioSegment.from_file(name, format="wav")
    #             played_together = tmp_audio.overlay(s100, position=time_control)
    #             name = "new_result_{}.wav".format(i)
    #             output_wav = played_together.export(name, format="wav")
    #         elif 50 <= df_result2['difference'][i] < 75:
    #             print("noise added")
    #             print("time: ", time)
    #             tmp_audio = AudioSegment.from_file(name, format="wav")
    #             played_together = tmp_audio.overlay(s10_v50, position=time_control)
    #             name = "new_result_{}.wav".format(i)
    #             output_wav = played_together.export(name, format="wav")
    #         elif 25 <= df_result2['difference'][i] < 50:
    #             print("noise added")
    #             print("time: ", time)
    #             tmp_audio = AudioSegment.from_file(name, format="wav")
    #             played_together = tmp_audio.overlay(v_50, position=time_control)
    #             name = "new_result_{}.wav".format(i)
    #             output_wav = played_together.export(name, format="wav")
    #         elif df_result2['difference'][i] < 25:
    #             print("noise added")
    #             print("time: ", time)
    #             tmp_audio = AudioSegment.from_file(name, format="wav")
    #             played_together = tmp_audio.overlay(s10_v10, position=time_control)
    #             name = "new_result_{}.wav".format(i)
    #             output_wav = played_together.export(name, format="wav")
    #
    # slidedown_noise = AudioSegment.from_file("noises/slidedown.wav", format="wav")
    # slidedown_noise = slidedown_noise - 32
    # time = 6.363
    # time_control = (time * 1000) - 150
    # audio = AudioSegment.from_file("new_result_22.wav", format="wav")
    # played_together = audio.overlay(slidedown_noise, position=time_control)
    # name = "sound_with_slidedown_noise_boosted.wav"
    # output_wav = played_together.export(name, format="wav")
    #
    # """
    # 시작: 6.363
    # 끝:  6.741
    # """

    # print("hello world!")
    import example
    print(example.text)

if __name__ == "__main__":
    main()

    # directory = 'temp.wav'
    # df_final = analysis(directory)
    # print(df_final)
    # df_final = pd.read_pickle("df_final")
    # print(df_final)
    # overlay_audio(df_final)
    ###################################################################################################################

    # print(waveform)
    # print(waveform.shape)
    # print(sample_rate)
    # print(491520/44100)
    # print("1.070초에서의 프레임: ", 44100 * 5.026)
    # waveform = waveform.numpy()
    # frame_to = int(44100 * 1.100)
    # print("{}번째 프레임의 값: ".format(frame_to), waveform[0][frame_to])
    # waveform = waveform**2
    # print("{}번째 프레임의 제곱된 값: ".format(frame_to), waveform[0][frame_to])
    # waveform = waveform * 100000
    # print("{}번째 프레임의 제곱된 값을 크기 보정: ".format(frame_to), waveform[0][frame_to])
    # print(np.around(waveform[0][frame_to], 3))
    #
    # newlist = np.where(waveform[0] > 15000)
    # print(newlist)
