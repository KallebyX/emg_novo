import serial
import numpy as np
import pandas as pd
from scipy.signal import butter, filtfilt, iirnotch, welch
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt
from tqdm import tqdm

# Função para ler dados da porta serial com barra de progresso
def read_serial_data(n_samples=2000):
    data = []
    port = 'COM6'  # Substitua pela sua porta
    baud_rate = 115200  # Deve corresponder ao configurado no Arduino
    ser = serial.Serial(port, baudrate=baud_rate, timeout=1)
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
    ser.close()
    return np.array(data)

# Funções de Filtragem
def apply_filters(data, fs=1000):
    low_cutoff = 100  # Filtro passa-baixa
    b, a = butter(6, low_cutoff / (0.5 * fs), btype='low')
    low_passed = filtfilt(b, a, data)
    
    # Filtro Notch para remover 60Hz
    notch_freq = 60
    quality_factor = 45
    b, a = iirnotch(notch_freq / (0.5 * fs), quality_factor)
    notch_filtered = filtfilt(b, a, low_passed)
    
    return notch_filtered

# Função para calcular Mean Frequency (MF) corretamente usando Welch
def mean_frequency(window, fs=1000):
    freqs, psd = welch(window, fs=fs)
    mf = np.sum(freqs * psd) / np.sum(psd)
    return mf

# Função para extrair características do sinal EMG
def extract_features(windows, fs=1000):
    features = []
    for window in windows:
        mav = np.mean(np.abs(window))  # Mean Absolute Value (MAV)
        rms = np.sqrt(np.mean(np.square(window)))  # Root Mean Square (RMS)
        zc = np.sum(np.diff(np.sign(window)) != 0) / len(window)  # Zero Crossing Rate (ZC)
        mf = mean_frequency(window, fs)  # Mean Frequency (MF)
        features.append([mav, rms, zc, mf])
    return np.array(features)

# Função para ler o CSV e verificar possíveis problemas
def check_csv_format(csv_file):
    try:
        # Tenta ler o CSV usando diferentes delimitadores
        df = pd.read_csv(csv_file, delimiter=',', encoding='ISO-8859-1')
        print("Arquivo CSV lido corretamente com vírgula como delimitador.")
        return df
    except pd.errors.ParserError:
        print("Erro ao tentar ler o CSV com vírgula. Tentando ponto e vírgula...")
        try:
            df = pd.read_csv(csv_file, delimiter=';', encoding='ISO-8859-1')
            print("Arquivo CSV lido corretamente com ponto e vírgula como delimitador.")
            return df
        except pd.errors.ParserError as e:
            print(f"Erro ao ler o arquivo CSV: {e}")
            return None

# Função para ler o CSV e treinar o modelo
def train_model(csv_file):
    # Verifica o formato do CSV
    df = check_csv_format(csv_file)
    
    if df is None:
        print("Não foi possível ler o arquivo CSV.")
        return None
    
    # Verificar se as colunas MAV, RMS, ZC, MF e Movimento existem no CSV
    expected_columns = ['MAV', 'RMS', 'ZC', 'MF', 'Movimento']
    if not all(col in df.columns for col in expected_columns):
        print(f"Erro: O arquivo CSV deve conter as colunas: {', '.join(expected_columns)}")
        return None
    
    # Separar as características e os rótulos
    X = df[['MAV', 'RMS', 'ZC', 'MF']]
    y = df['Movimento']
    
    # Dividir os dados em conjunto de treino e teste
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Treinamento do modelo Random Forest
    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    clf.fit(X_train, y_train)
    
    return clf

# Função para classificar o sinal EMG em tempo real e calcular acurácia
def classify_emg(clf, n_samples=2000, window_size=100):
    # Coleta de dados
    raw_data = read_serial_data(n_samples)
    
    # Remover o DC bias
    raw_data = raw_data - np.mean(raw_data)
    
    # Aplicar filtros nos dados
    filtered_data = apply_filters(raw_data)
    
    # Dividir em janelas
    windows = [filtered_data[i:i+window_size] for i in range(0, len(filtered_data), window_size)]
    
    # Extrair características de cada janela
    features = extract_features(windows)
    
    # Variáveis para rastrear previsões corretas e incorretas
    correct_predictions = 0
    total_predictions = 0
    
    # Classificar cada janela e perguntar ao usuário
    for i, feature in enumerate(features):
        # Converter a lista de características em um DataFrame para corresponder ao formato do modelo treinado
        feature_df = pd.DataFrame([feature], columns=['MAV', 'RMS', 'ZC', 'MF'])
        prediction = clf.predict(feature_df)[0]
        print(f'Janela {i+1}: Movimento previsto - {prediction}')
        
        # Perguntar se a previsão estava correta
        user_input = input("A previsão estava correta? (s/n): ").strip().lower()
        
        # Atualizar contagem de acertos
        if user_input == 's':
            correct_predictions += 1
        total_predictions += 1
    
    # Calcular e exibir acurácia
    accuracy = correct_predictions / total_predictions if total_predictions > 0 else 0
    print(f'Acurácia final baseada nas respostas do usuário: {accuracy * 100:.2f}%')

# Função principal
def main():
    # Treinar o modelo com o CSV existente
    csv_file = 'resultados_emg.csv'  # Substitua pelo caminho correto do CSV
    clf = train_model(csv_file)
    
    if clf is not None:
        # Classificar os dados EMG em tempo real
        classify_emg(clf)

if __name__ == "__main__":
    main()
