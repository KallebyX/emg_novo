import serial
import numpy as np
from scipy.signal import butter, filtfilt, iirnotch, welch
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm

# Configuração da porta serial
port = 'COM6'  # Substitua pela sua porta
baud_rate = 115200  # Deve corresponder ao configurado no Arduino
ser = serial.Serial(port, baudrate=baud_rate, timeout=1)

# Função para ler dados da porta serial com barra de progresso
def read_serial_data(n_samples=2000):
    data = []
    with tqdm(total=n_samples, desc="Coletando dados") as pbar:
        while len(data) < n_samples:
            if ser.in_waiting:
                line = ser.readline().decode('utf-8', errors='ignore').strip()
                try:
                    value = float(line)
                    data.append(value)
                    pbar.update(1)
                except ValueError:
                    continue
    return np.array(data)

# Funções de Filtragem
def apply_filters(data, fs=1000):
    # Filtro passa-baixa para remover alta frequência de ruído
    low_cutoff = 100  # Ajustar a frequência de corte para preservar mais do sinal EMG
    b, a = butter(6, low_cutoff / (0.5 * fs), btype='low')
    low_passed = filtfilt(b, a, data)
    
    # Filtro Notch para remover a frequência da rede elétrica (60 Hz no Brasil)
    notch_freq = 60  # frequência central do filtro notch
    quality_factor = 45  # fator de qualidade que define a largura da banda de rejeição
    b, a = iirnotch(notch_freq / (0.5 * fs), quality_factor)
    notch_filtered = filtfilt(b, a, low_passed)
    
    return notch_filtered

# Função para calcular Mean Frequency (MF) corretamente usando Welch
def mean_frequency(window, fs=1000):
    freqs, psd = welch(window, fs=fs)
    mf = np.sum(freqs * psd) / np.sum(psd)  # Cálculo correto da frequência média
    return mf

# Função para extrair características do sinal EMG
def extract_features(windows, fs=1000):
    features = []
    for window in windows:
        mav = np.mean(np.abs(window))  # Mean Absolute Value (MAV)
        rms = np.sqrt(np.mean(np.square(window)))  # Root Mean Square (RMS)
        zc = np.sum(np.diff(np.sign(window)) != 0) / len(window)  # Zero Crossing Rate (ZC)
        mf = mean_frequency(window, fs)  # Mean Frequency (MF) calculada corretamente
        features.append([mav, rms, zc, mf])
    return np.array(features)

# Função para salvar os resultados em um arquivo Excel
def save_to_excel(features):
    df = pd.DataFrame(features, columns=['MAV', 'RMS', 'ZC', 'MF'])
    df['Movimento'] = ''  # Coluna vazia para indicar manualmente o movimento
    df.to_excel('resultados_emg.xlsx', index=False)
    print("Resultados salvos no arquivo 'resultados_emg.xlsx'")

# Função para visualização
def main():
    n_samples = 2000  # 20 janelas de 100 amostras cada
    window_size = 100  # Tamanho da janela
    overlap = 0  # Sem sobreposição, uma janela começa onde a outra termina
    
    # Coleta de dados com barra de progresso
    raw_data = read_serial_data(n_samples)
    
    # Remover o DC bias (centralizar o sinal em torno de zero)
    raw_data = raw_data - np.mean(raw_data)
    
    # Aplicar filtros nos dados
    filtered_data = apply_filters(raw_data)

    # Dividir dados em 20 janelas sem sobreposição e extrair características
    windows = [filtered_data[i:i+window_size] for i in range(0, len(filtered_data), window_size)]
    features = extract_features(windows)

    # Salvar os resultados no arquivo Excel
    save_to_excel(features)

    # Visualizar dados brutos e filtrados
    plt.figure(figsize=(10, 5))
    plt.plot(raw_data, label='Dados Brutos')
    plt.plot(filtered_data, label='Dados Filtrados', color='red')
    plt.title("Sinal EMG Bruto e Filtrado")
    plt.xlabel('Número de Amostras')
    plt.ylabel('Amplitude do Sinal EMG')
    plt.legend()
    plt.show()

if __name__ == "__main__":
    main()
