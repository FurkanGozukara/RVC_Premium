from io import BufferedWriter, BytesIO
from pathlib import Path
from typing import Dict, Tuple, Optional, Union, List
import os
import math
import wave
import signal
from multiprocessing import Process, Value, Event
from multiprocessing.shared_memory import SharedMemory

import numpy as np
from numba import jit
import av
from av.audio.resampler import AudioResampler
from av.audio.frame import AudioFrame
import scipy.io.wavfile as wavfile

video_format_dict: Dict[str, str] = {
    "m4a": "mp4",
}

audio_format_dict: Dict[str, str] = {
    "ogg": "libvorbis",
    "mp4": "aac",
}

supported_audio_rates: Dict[str, List[int]] = {
    "mp3": [8000, 11025, 12000, 16000, 22050, 24000, 32000, 44100, 48000],
}


@jit(nopython=True)
def float_to_int16(audio: np.ndarray) -> np.ndarray:
    am = int(math.ceil(float(np.abs(audio).max())) * 32768)
    am = 32767 * 32768 // am
    return np.multiply(audio, am).astype(np.int16)


def _to_samples_last(audio: np.ndarray) -> np.ndarray:
    audio = np.asarray(audio)
    if audio.ndim == 1:
        return audio
    if audio.ndim != 2:
        raise ValueError(f"Unsupported audio shape: {audio.shape}")
    # Accept either (samples, channels) or (channels, samples).
    if audio.shape[0] <= 8 and audio.shape[1] > 8:
        audio = audio.T
    return np.ascontiguousarray(audio)


def float_np_array_to_wav_buf(wav: np.ndarray, sr: int, f32=False) -> BytesIO:
    wav = _to_samples_last(wav)
    buf = BytesIO()
    if np.issubdtype(wav.dtype, np.integer):
        wavfile.write(buf, sr, wav.astype(np.int16, copy=False))
    elif f32:
        wavfile.write(buf, sr, wav.astype(np.float32, copy=False))
    else:
        wav = np.asarray(wav, dtype=np.float32)
        peak = float(np.max(np.abs(wav))) if wav.size else 0.0
        if peak > 1.0:
            wav = wav / peak
        wav = np.clip(wav, -1.0, 1.0)
        wavfile.write(buf, sr, (wav * 32767.0).astype(np.int16))
    buf.seek(0, 0)
    return buf


def _resolve_stream_rate(audio_stream) -> Optional[int]:
    for owner in (audio_stream, getattr(audio_stream, "codec_context", None)):
        if owner is None:
            continue
        for attr_name in ("sample_rate", "rate", "base_rate"):
            value = getattr(owner, attr_name, None)
            if value is None:
                continue
            try:
                value = int(value)
            except (TypeError, ValueError):
                continue
            if value > 0:
                return value
    return None


def _resolve_stream_layout(audio_stream) -> Optional[str]:
    layout = getattr(audio_stream, "layout", None)
    if layout is not None:
        layout_channels = getattr(layout, "channels", None)
        try:
            channel_count = len(layout_channels) if layout_channels is not None else None
        except TypeError:
            channel_count = None
        if channel_count == 1:
            return "mono"
        if channel_count == 2:
            return "stereo"

        layout_name = getattr(layout, "name", None)
        if layout_name in {"mono", "stereo"}:
            return str(layout_name)

    channel_count = None
    for owner in (audio_stream, getattr(audio_stream, "codec_context", None)):
        if owner is None:
            continue
        channel_count = getattr(owner, "channels", None) or getattr(
            owner, "nb_channels", None
        )
        if isinstance(channel_count, int) and channel_count > 0:
            break
    if channel_count == 1:
        return "mono"
    if channel_count == 2:
        return "stereo"
    return None


def get_supported_sample_rate_for_format(format: str, sample_rate: int) -> int:
    try:
        sample_rate = int(sample_rate)
    except (TypeError, ValueError):
        return sample_rate

    container_format = video_format_dict.get(format, format)
    codec_name = audio_format_dict.get(container_format, container_format)
    valid_rates = supported_audio_rates.get(codec_name)
    if not valid_rates or sample_rate in valid_rates:
        return sample_rate
    return min(valid_rates, key=lambda candidate: abs(candidate - sample_rate))


def save_audio(
    path: str,
    audio: np.ndarray,
    sr: int,
    f32=False,
    format="wav",
    bitrate_kbps: Optional[int] = None,
):
    buf = float_np_array_to_wav_buf(audio, sr, f32)
    if format != "wav":
        transbuf = BytesIO()
        wav2(buf, transbuf, format, bitrate_kbps=bitrate_kbps)
        buf = transbuf
    with open(path, "wb") as f:
        f.write(buf.getbuffer())


def wav2(
    i: BytesIO,
    o: BufferedWriter,
    format: str,
    bitrate_kbps: Optional[int] = None,
):
    inp = av.open(i, "r")
    format = video_format_dict.get(format, format)
    out = av.open(o, "w", format=format)
    codec_name = audio_format_dict.get(format, format)

    try:
        input_stream = next((s for s in inp.streams if s.type == "audio"), None)
        stream_rate = _resolve_stream_rate(input_stream) if input_stream else None
        output_rate = (
            get_supported_sample_rate_for_format(codec_name, stream_rate)
            if stream_rate
            else stream_rate
        )
        stream_kwargs = {"rate": output_rate} if output_rate else {}
        ostream = out.add_stream(codec_name, **stream_kwargs)
        stream_layout = _resolve_stream_layout(input_stream) if input_stream else None
        if stream_layout:
            try:
                ostream.layout = stream_layout
            except (AttributeError, ValueError):
                pass
        if bitrate_kbps is not None:
            try:
                ostream.bit_rate = int(bitrate_kbps) * 1000
            except (AttributeError, TypeError, ValueError):
                pass
        resampler = None
        if stream_rate and output_rate and stream_rate != output_rate:
            resampler = AudioResampler(
                format="fltp",
                layout=stream_layout or "stereo",
                rate=output_rate,
            )

        for frame in inp.decode(audio=0):
            frames_to_encode = resampler.resample(frame) if resampler else [frame]
            for encoded_frame in frames_to_encode:
                for p in ostream.encode(encoded_frame):
                    out.mux(p)

        for p in ostream.encode(None):
            out.mux(p)
    finally:
        out.close()
        inp.close()


def load_audio(
    file: Union[str, BytesIO, Path],
    sr: Optional[int] = None,
    format: Optional[str] = None,
    mono=True,
) -> Union[np.ndarray, Tuple[np.ndarray, int]]:
    if (isinstance(file, str) and not Path(file).exists()) or (
        isinstance(file, Path) and not file.exists()
    ):
        raise FileNotFoundError(f"File not found: {file}")
    rate = 0

    container = av.open(file, format=format)
    audio_stream = next(s for s in container.streams if s.type == "audio")
    channels = 1 if audio_stream.layout == "mono" else 2
    container.seek(0)
    resampler = (
        AudioResampler(format="fltp", layout=audio_stream.layout, rate=sr)
        if sr is not None
        else None
    )

    # Estimated maximum total number of samples to pre-allocate the array
    # AV stores length in microseconds by default
    estimated_total_samples = (
        int(container.duration * sr // 1_000_000) if sr is not None else 48000
    )
    decoded_audio = np.zeros(
        (
            estimated_total_samples + 1
            if channels == 1
            else (channels, estimated_total_samples + 1)
        ),
        dtype=np.float32,
    )

    offset = 0

    def process_packet(packet: List[AudioFrame]):
        frames_data = []
        rate = 0
        for frame in packet:
            # frame.pts = None  # 清除时间戳，避免重新采样问题
            resampled_frames = (
                resampler.resample(frame) if resampler is not None else [frame]
            )
            for resampled_frame in resampled_frames:
                frame_data = resampled_frame.to_ndarray()
                rate = resampled_frame.rate
                frames_data.append(frame_data)
        return (rate, frames_data)

    def frame_iter(container):
        for p in container.demux(container.streams.audio[0]):
            yield p.decode()

    for r, frames_data in map(process_packet, frame_iter(container)):
        if not rate:
            rate = r
        for frame_data in frames_data:
            end_index = offset + len(frame_data[0])

            # 检查 decoded_audio 是否有足够的空间，并在必要时调整大小
            if end_index > decoded_audio.shape[1]:
                decoded_audio = np.resize(
                    decoded_audio, (decoded_audio.shape[0], end_index * 4)
                )

            np.copyto(decoded_audio[..., offset:end_index], frame_data)
            offset += len(frame_data[0])

    container.close()

    # Truncate the array to the actual size
    decoded_audio = decoded_audio[..., :offset]

    if mono and decoded_audio.shape[0] > 1:
        decoded_audio = decoded_audio.mean(0)

    if sr is not None:
        return decoded_audio
    return decoded_audio, rate


def resample_audio(
    input_path: str, output_path: str, codec: str, format: str, sr: int, layout: str
) -> None:
    if not os.path.exists(input_path):
        return

    input_container = av.open(input_path)
    output_container = av.open(output_path, "w")

    # Create a stream in the output container
    input_stream = input_container.streams.audio[0]
    output_stream = output_container.add_stream(codec, rate=sr, layout=layout)

    resampler = AudioResampler(format, layout, sr)

    # Copy packets from the input file to the output file
    for packet in input_container.demux(input_stream):
        for frame in packet.decode():
            # frame.pts = None  # Clear presentation timestamp to avoid resampling issues
            out_frames = resampler.resample(frame)
            for out_frame in out_frames:
                for out_packet in output_stream.encode(out_frame):
                    output_container.mux(out_packet)

    for packet in output_stream.encode():
        output_container.mux(packet)

    # Close the containers
    input_container.close()
    output_container.close()


def get_audio_properties(input_path: str) -> Tuple[int, int]:
    def _resolve_channels(audio_stream) -> Optional[int]:
        for owner in (audio_stream, getattr(audio_stream, "codec_context", None)):
            if owner is None:
                continue
            for attr_name in ("channels", "nb_channels"):
                value = getattr(owner, attr_name, None)
                if isinstance(value, int) and value > 0:
                    return value

        layout = getattr(audio_stream, "layout", None)
        if layout is not None:
            value = getattr(layout, "nb_channels", None)
            if isinstance(value, int) and value > 0:
                return value

            layout_channels = getattr(layout, "channels", None)
            try:
                value = len(layout_channels) if layout_channels is not None else None
            except TypeError:
                value = None
            if isinstance(value, int) and value > 0:
                return value

            if getattr(layout, "name", None) == "mono":
                return 1

        return None

    def _resolve_rate(audio_stream) -> Optional[int]:
        for owner in (audio_stream, getattr(audio_stream, "codec_context", None)):
            if owner is None:
                continue
            for attr_name in ("sample_rate", "rate", "base_rate"):
                value = getattr(owner, attr_name, None)
                if value is None:
                    continue
                try:
                    value = int(value)
                except (TypeError, ValueError):
                    continue
                if value > 0:
                    return value
        return None

    container = av.open(input_path)
    try:
        audio_stream = next((s for s in container.streams if s.type == "audio"), None)
        if audio_stream is None:
            raise ValueError(f"No audio stream found in: {input_path}")

        channels = _resolve_channels(audio_stream)
        rate = _resolve_rate(audio_stream)
        if channels is None or rate is None:
            raise ValueError(
                f"Could not determine audio properties for: {input_path}"
            )
        return channels, rate
    finally:
        container.close()


class AudioIoProcess(Process):
    def __init__(
        self,
        input_device,
        output_device,
        input_audio_block_size: int,
        sample_rate: int,
        channel_num: int = 2,
        is_device_combined: bool = True,
        is_input_wasapi_exclusive: bool = False,
        is_output_wasapi_exclusive: bool = False,
    ):
        super().__init__()
        self.in_dev = input_device
        self.out_dev = output_device
        self.block_size: int = input_audio_block_size
        self.buf_size: int = self.block_size << 1  # 双缓冲
        self.sample_rate: int = sample_rate
        self.channels: int = channel_num
        self.is_device_combined: bool = is_device_combined
        self.is_input_wasapi_exclusive: bool = is_input_wasapi_exclusive
        self.is_output_wasapi_exclusive: bool = is_output_wasapi_exclusive

        self.__rec_ptr = 0
        self.in_ptr = Value("i", 0)  # 当收满一个block时由本进程设置
        self.out_ptr = Value("i", 0)  # 由主进程设置，指示下一次预期写入位置
        self.play_ptr = Value("i", 0)  # 由本进程设置，指示当前音频已经播放到哪里
        self.in_evt = Event()  # 当收满一个block时由本进程设置
        self.stop_evt = Event()  # 当主进程停止音频活动时由主进程设置

        self.latency = Value("d", 114514.1919810)

        self.buf_shape: tuple = (self.buf_size, self.channels)
        self.buf_dtype: np.dtype = np.float32
        self.buf_nbytes: int = int(
            np.prod(self.buf_shape) * np.dtype(self.buf_dtype).itemsize
        )

        self.in_mem = SharedMemory(create=True, size=self.buf_nbytes)
        self.out_mem = SharedMemory(create=True, size=self.buf_nbytes)
        self.in_mem_name: str = self.in_mem.name
        self.out_mem_name: str = self.out_mem.name

        self.in_buf = None
        self.out_buf = None

    def get_in_mem_name(self) -> str:
        return self.in_mem_name

    def get_out_mem_name(self) -> str:
        return self.out_mem_name

    def get_np_shape(self) -> tuple:
        return self.buf_shape

    def get_np_dtype(self) -> np.dtype:
        return self.buf_dtype

    def get_ptrs_and_events(self):
        return self.in_ptr, self.out_ptr, self.play_ptr, self.in_evt, self.stop_evt

    def get_latency(self) -> float:
        return self.latency.value

    def run(self):
        import sounddevice as sd

        signal.signal(signal.SIGINT, signal.SIG_IGN)

        in_mem = SharedMemory(name=self.in_mem_name)
        self.in_buf = np.ndarray(
            self.buf_shape, dtype=self.buf_dtype, buffer=in_mem.buf, order="C"
        )
        self.in_buf.fill(0.0)

        out_mem = SharedMemory(name=self.out_mem_name)
        self.out_buf = np.ndarray(
            self.buf_shape, dtype=self.buf_dtype, buffer=out_mem.buf, order="C"
        )
        self.out_buf.fill(0.0)

        exclusive_settings = sd.WasapiSettings(exclusive=True)

        sd.default.device = (self.in_dev, self.out_dev)

        def output_callback(outdata, frames, time_info, status):
            play_ptr = self.play_ptr.value
            end_ptr = play_ptr + frames

            if end_ptr <= self.buf_size:
                outdata[:] = self.out_buf[play_ptr:end_ptr]
            else:
                first = self.buf_size - play_ptr
                second = end_ptr - self.buf_size
                outdata[:first] = self.out_buf[play_ptr:]
                outdata[first:] = self.out_buf[:second]

            self.play_ptr.value = end_ptr % self.buf_size

        def input_callback(indata, frames, time_info, status):
            # 收录输入数据
            end_ptr = self.__rec_ptr + frames
            if end_ptr <= self.buf_size:  # 整块拷贝
                self.in_buf[self.__rec_ptr : end_ptr] = indata
            else:  # 处理回绕
                first = self.buf_size - self.__rec_ptr
                second = end_ptr - self.buf_size
                self.in_buf[self.__rec_ptr :] = indata[:first]
                self.in_buf[:second] = indata[first:]
            write_pos = self.__rec_ptr
            self.__rec_ptr = end_ptr % self.buf_size

            # 设置信号
            if write_pos < self.block_size and self.__rec_ptr >= self.block_size:
                self.in_ptr.value = 0
                self.in_evt.set()  # 通知主线程来取甲缓冲
            elif write_pos < self.buf_size and self.__rec_ptr < write_pos:
                self.in_ptr.value = self.block_size
                self.in_evt.set()  # 通知主线程来取乙缓冲

        def combined_callback(indata, outdata, frames, time_info, status):
            output_callback(outdata, frames, time_info, status)  # 优先出声
            input_callback(indata, frames, time_info, status)

        if self.is_device_combined:
            with sd.Stream(
                samplerate=self.sample_rate,
                channels=self.channels,
                dtype=self.buf_dtype,
                latency="low",
                extra_settings=(
                    exclusive_settings
                    if self.is_input_wasapi_exclusive
                    and self.is_output_wasapi_exclusive
                    else None
                ),
                callback=combined_callback,
            ) as s:
                self.latency.value = s.latency[-1]
                self.stop_evt.wait()
                self.out_buf.fill(0.0)
        else:
            with sd.InputStream(
                samplerate=self.sample_rate,
                channels=self.channels,
                dtype=self.buf_dtype,
                latency="low",
                extra_settings=(
                    exclusive_settings if self.is_input_wasapi_exclusive else None
                ),
                callback=input_callback,
            ) as si, sd.OutputStream(
                samplerate=self.sample_rate,
                channels=self.channels,
                dtype=self.buf_dtype,
                latency="low",
                extra_settings=(
                    exclusive_settings if self.is_output_wasapi_exclusive else None
                ),
                callback=output_callback,
            ) as so:
                self.latency.value = si.latency[-1] + so.latency[-1]
                self.stop_evt.wait()
                self.out_buf.fill(0.0)

        # 清理共享内存
        in_mem.close()
        out_mem.close()
        in_mem.unlink()
        out_mem.unlink()
