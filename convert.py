import os
import subprocess
import sys
import traceback
import argparse

def check_requirements():
    """Проверяет, что ffmpeg установлен и доступен"""
    try:
        result = subprocess.run(['ffmpeg', '-version'], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        if result.returncode != 0:
            print("ОШИБКА: ffmpeg не найден. Убедитесь, что ffmpeg установлен и доступен в PATH.")
            return False
        return True
    except Exception as e:
        print(f"ОШИБКА: При проверке ffmpeg произошла ошибка: {e}")
        return False

def convert_wav_to_opus(input_path, output_path, bitrate=40):
    """Конвертирует WAV в Opus с оптимизацией размера файла"""
    print(f"Конвертация: {input_path}")
    
    # Проверяем существование входного файла
    if not os.path.exists(input_path):
        print(f"ОШИБКА: Входной файл не найден: {input_path}")
        return False
    
    # Базовая команда конвертации
    command = [
        'ffmpeg',
        '-y',
        '-i', input_path,
        '-c:a', 'libopus',
        '-b:a', f'{bitrate}k',
        '-ac', '1',           # Моно звук
        '-application', 'voip',  # Оптимизация для речи
        output_path
    ]
    
    try:
        # Запускаем процесс конвертации
        result = subprocess.run(command, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        
        # Проверяем результат
        if os.path.exists(output_path):
            orig_size = os.path.getsize(input_path) / (1024 * 1024)  # в MB
            new_size = os.path.getsize(output_path) / (1024 * 1024)  # в MB
            
            print(f"  Успешно! WAV: {orig_size:.2f} MB -> Opus: {new_size:.2f} MB (Битрейт: {bitrate} kbps)")
            return True
        else:
            print(f"ОШИБКА: Выходной файл не был создан: {output_path}")
            return False
    
    except subprocess.CalledProcessError as e:
        print(f"ОШИБКА при выполнении ffmpeg: {e}")
        print(f"Stderr: {e.stderr.decode('utf-8', errors='ignore')}")
        return False
    except Exception as e:
        print(f"ОШИБКА при конвертации: {e}")
        traceback.print_exc()
        return False

def main():
    try:
        # Создаем парсер аргументов командной строки
        parser = argparse.ArgumentParser(description='Конвертация WAV файлов в Opus с настраиваемым битрейтом')
        parser.add_argument('--bitrate', '-b', type=int, default=48, 
                            help='Битрейт для Opus файла в kbps (по умолчанию: 48)')
        parser.add_argument('--folder', '-f', type=str, default='audio',
                            help='Папка с WAV файлами (по умолчанию: audio)')
        
        args = parser.parse_args()
        
        print(f"Запуск скрипта конвертации WAV -> Opus (Битрейт: {args.bitrate} kbps)")
        
        # Проверка ffmpeg
        if not check_requirements():
            return
        
        # Определяем путь к папке audio
        script_dir = os.path.dirname(os.path.abspath(__file__))
        audio_dir = os.path.join(script_dir, args.folder)
        
        print(f"Ищем WAV файлы в папке: {audio_dir}")
        
        # Проверяем существование папки
        if not os.path.exists(audio_dir):
            print(f"ОШИБКА: Папка не найдена: {audio_dir}")
            print(f"Создайте папку '{args.folder}' в той же директории, где находится скрипт")
            return
        
        # Получаем список WAV файлов
        wav_files = [f for f in os.listdir(audio_dir) if f.lower().endswith('.wav')]
        
        if not wav_files:
            print(f"ОШИБКА: WAV файлы не найдены в папке {args.folder}")
            return
        
        print(f"Найдено {len(wav_files)} WAV файлов для конвертации")
        
        # Конвертируем файлы
        successful = 0
        for wav_file in wav_files:
            input_path = os.path.join(audio_dir, wav_file)
            output_path = os.path.join(audio_dir, os.path.splitext(wav_file)[0] + ".opus")
            
            if convert_wav_to_opus(input_path, output_path, args.bitrate):
                successful += 1
        
        # Итоговая статистика
        print(f"\nГотово! Успешно конвертировано {successful} из {len(wav_files)} файлов")
        
    except Exception as e:
        print(f"КРИТИЧЕСКАЯ ОШИБКА: {e}")
        traceback.print_exc()

if __name__ == "__main__":
    main()
    
    # Оставляем консоль открытой, чтобы увидеть сообщения об ошибках
    input("\nНажмите Enter для выхода...")