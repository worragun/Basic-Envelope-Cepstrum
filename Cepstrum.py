import streamlit as st
import pandas as pd
import numpy as np
from scipy.fft import fft, ifft
from scipy.signal import butter, filtfilt, hilbert
from scipy.signal.windows import hann
import plotly.graph_objects as go

st.set_page_config(page_title="Cepstrum & Envelope Tool", layout="wide")
st.title("📊 Multi-File Cepstrum & Envelope Analysis")

def run_fast_kurtogram(x_raw, fs):
    nyq = 0.5 * fs
    band_width = nyq / 4.0  
    
    max_kurtosis = -1.0
    best_low = 1000.0  
    best_high = 3000.0
    
    for i in range(4):
        low_cutoff = i * band_width
        high_cutoff = (i + 1) * band_width
        
        if low_cutoff < 400.0:
            continue
            
        try:
            b, a = butter(4, [low_cutoff / nyq, high_cutoff / nyq], btype='bandpass')
            x_filtered = filtfilt(b, a, x_raw)
            
            mean_sq = np.mean(x_filtered**2)
            mean_quad = np.mean(x_filtered**4)
            
            kurtosis = mean_quad / (mean_sq**2) if mean_sq > 1e-12 else 0.0
            
            if kurtosis > max_kurtosis:
                max_kurtosis = kurtosis
                best_low = low_cutoff
                best_high = high_cutoff
        except:
            continue
            
    return best_low, best_high

with st.sidebar:
    st.header("📁 Data Input")
    uploaded_files = st.file_uploader("Upload CSV files", type=["csv"], accept_multiple_files=True)
    
    st.header("🔍 Envelope Settings")
    use_envelope = st.checkbox("Enable Envelope Cepstrum", help="ใช้ Hilbert Transform เพื่อวิเคราะห์ Bearing Fault")
   
    auto_kurtogram = st.checkbox(
        "Auto-calculate using Fast Kurtogram", 
        value=False, 
        disabled=not use_envelope,
        help="ค้นหาย่านความถี่ที่มีการกระแทกชำรุดสูงที่สุดเพื่อคำนวณ Bandpass อัตโนมัติ"
    )
    
    col_bp1, col_bp2 = st.columns(2)
    with col_bp1:
        bp_low = st.number_input(
            "BP Low (Hz)", 
            value=1000, 
            disabled=(not use_envelope or auto_kurtogram),
            help="จุดเริ่มต้นย่าน Resonance (ถูกล็อกหากเลือก Fast Kurtogram)"
        )
    with col_bp2:
        bp_high = st.number_input(
            "BP High (Hz)", 
            value=5000, 
            disabled=(not use_envelope or auto_kurtogram),
            help="จุดสิ้นสุดย่าน Resonance (ถูกล็อกหากเลือก Fast Kurtogram)"
        )
    
    st.header("⚙️ General Settings")
    f_min = st.number_input("Filter Min Frequency (Hz)", value=1, disabled=use_envelope)
    f_max = st.number_input("Filter Max Frequency (Hz)", value=5000, disabled=use_envelope)
    
    st.subheader("Peak Search Range (Hz)")
    s_min = st.number_input("Search Min", value=1.1)
    s_max = st.number_input("Search Max", value=100.0)

def process_analysis(file, f_min, f_max, use_env, auto_kurt, bp_l, bp_h):
    try:
        df = pd.read_csv(file)
        t, x = df.iloc[:, 0].values, df.iloc[:, 1].values
        fs = 1 / np.mean(np.diff(t))
        N = len(x)
        
        x_raw = x - np.mean(x)
        
        if use_env:
            nyq = 0.5 * fs

            if auto_kurt:
                current_bp_l, current_bp_h = run_fast_kurtogram(x_raw, fs)
            else:
                current_bp_l, current_bp_h = bp_l, bp_h
            
            if fs < 2 * current_bp_h:
                st.warning(f"⚠️ FS ({fs:.1f} Hz) ต่ำเกินไปสำหรับ Filter {current_bp_h:.1f} Hz ในไฟล์ {file.name}")
                return None
            
            b, a = butter(4, [current_bp_l/nyq, current_bp_h/nyq], btype='bandpass')
            x_filtered = filtfilt(b, a, x_raw)

            x_final = np.abs(hilbert(x_filtered))
            x_final = x_final - np.mean(x_final)

            info_label = f"{file.name} (Env: {int(current_bp_l)}-{int(current_bp_h)} Hz)"
        else:
            x_final = x_raw
            info_label = f"{file.name}"

        x_w = x_final * hann(N)
        X_f = fft(x_w)
        mag_spec = np.abs(2/N * X_f[:N//2])
        freqs = np.fft.fftfreq(N, 1/fs)[:N//2]

        log_ps = np.log(np.abs(X_f)**2 + 1e-12)
        f_all = np.fft.fftfreq(N, 1/fs)

        if use_env:
            log_ps_filtered = (log_ps - np.mean(log_ps))
        else:
            mask = np.zeros(N)
            mask[(np.abs(f_all) >= f_min) & (np.abs(f_all) <= f_max)] = 1
            log_ps_filtered = (log_ps - np.mean(log_ps)) * mask

        c_power = np.real(ifft(log_ps_filtered))**2
        quefrency = np.arange(N) / fs
        
        return freqs, mag_spec, quefrency, c_power, info_label
    
    except Exception as e:
        st.error(f"❌ Error processing {file.name}: {e}")
        return None

if uploaded_files:
    fig_spec = go.Figure()
    fig_cep = go.Figure()
    
    for uploaded_file in uploaded_files:
        result = process_analysis(uploaded_file, f_min, f_max, use_envelope, auto_kurtogram, bp_low, bp_high)
        
        if result:
            f, mag, q, cp, label = result
            
            fig_spec.add_trace(go.Scatter(x=f, y=mag, name=label))
            
            q_min, q_max = 1/s_max, 1/s_min
            mask_q = (q >= q_min) & (q <= q_max)
            
            fig_cep.add_trace(go.Scatter(x=1/q[mask_q], y=cp[mask_q], name=label))

    fig_spec.update_layout(
        title="Spectrum Comparison", 
        template="plotly_dark", 
        xaxis_title="Frequency (Hz)",
        yaxis_title="Amplitude"
    )
    st.plotly_chart(fig_spec, use_container_width=True)

    fig_cep.update_layout(
        title="Cepstrum Comparison (Frequency Domain Analysis)", 
        template="plotly_dark", 
        xaxis_title="Equivalent Frequency (Hz)",
        yaxis_title="Cepstrum Power"
    )
    st.plotly_chart(fig_cep, use_container_width=True)
else:
    st.info("💡 กรุณาอัปโหลดไฟล์ CSV เพื่อเริ่มการวิเคราะห์ (คอลัมน์แรกเป็น Time, คอลัมน์สองเป็น Vibration)")