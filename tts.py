import os
import re
import gc
import torch
import subprocess
from time import sleep
from tqdm import tqdm
from TTS.api import TTS
from markdown import markdown

# Конфигурация ресурсов
torch.set_num_threads(2)  # Ограничение использования CPU
device = "cuda" if torch.cuda.is_available() else "cpu"

# Инициализация модели ОДИН РАЗ - исправленная часть
print("\nИнициализация TTS модели...")
# Вместо использования gpu=True, используем .to(device)
# Добавляем игнорирование предупреждений для этой конкретной ситуации
import warnings
warnings.filterwarnings("ignore", message="The attention mask is not set and cannot be inferred from input")
tts_model = TTS("tts_models/multilingual/multi-dataset/xtts_v2").to(device)

# Остальная часть скрипта остается без изменений
class MarkdownTTSPipeline:
    def __init__(self, voice_clone_path, tts_model):
        self.voice_clone = voice_clone_path
        self.tts = tts_model
        self.stress_dict = self.load_stress_rules()
        self.regex_patterns = self.compile_regex()
        # Максимальное количество символов для одного фрагмента (гарантированно безопасное)
        self.max_chunk_size = 150  # Консервативное значение для безопасной обработки

    def md_to_clean_text(self, md_content):  # <-- md_content объявлен правильно
        html = markdown(md_content)
        
        # Удаляем все HTML-теги, но заменяем <p>, <br> и <li> на перенос строки
        text = re.sub(r'<p>|<br\s*/?>|<li>', '\n', html)
        text = re.sub(r'<[^>]+>', '', text)  # Удаляем остальные HTML-теги
        
        # Убираем лишние спецсимволы Markdown
        text = re.sub(r'[\*#_~`]', '', text)
        
        # Нормализуем кавычки (заменяем разные виды кавычек на обычные)
        text = re.sub(r'[«»<<>>""'']', '"', text)
        
        # Сохраняем оригинальные абзацы и пробельные символы
        text = re.sub(r'\n\s*\n', '\n\n', text)  # Оставляем двойные переносы строк для абзацев
        text = re.sub(r'\s+', ' ', text)  # Убираем лишние пробелы
        
        return text.strip()

    def load_stress_rules(self):
        return {
            # Простые замены
            "words": {
                 "—": "-",
                 "--": "-",
                 "облаков": "обла+ко́в",
                 "колокола": "к+о́локола",
                 "ранга": "р+анга",
                 "ярусов": "+я́русов",
                 "горного": "+г+о́рного",
                 "горных": "+г+о́рных",
                 "Пика": "П+ика",
                 "служанка": "слу+жа́нка",
                 "настоянная": "нас+то́янная",
                 "мастера": "м+а́стера",
                 "караваны": "карав+а́ны",
                 "печатей": "печ+а́тей",
                 "нефритовые": "неф+ри́товые",
                 "золотом": "з+о́лотом",
                 "нифрита": "ниф+ри́та",
                 "нефрита": "ниф+ри́та",
                 "нефритовых": "ниф+ри́товых",
                 "нефритовый": "ниф+ри́товый",
                 "нефритовое": "ниф+ри́товое",
                 "парящих": "па+ря́щих",
                 "хрусталя": "хруста+ля́",
                 "зрелище": "з+ре́лище",
                 "тона": "то+на́",
                 "договор": "дого+во́р",
                 "средства": "с+ре́дства",
                 "каталог": "ката+ло́г",
                 "красивее": "кра+си́вее",
            
                # Глаголы и причастия
                 "звонит": "зво+ни́т",
                 "включит": "вклю+чи́т",
                 "начать": "на+ча́ть",
                 "принял": "при+ня́л",
                 "сняла": "сня+ла́",
            
                # Прилагательные
                 "значимый": "з+на́чимый",
                 "ловкий": "л+о́вкий",
            
                # Наречия
                 "сверху": "св+е́рху",
                 "досуха": "д+о́суха",  
            },
            # Regex паттерны
            "patterns": {
                r"ться\b": "ца",
                r"тся\b": "ца",
                r"\bсво([ейё])\b": r"своё"
            }
        }

    def compile_regex(self):
        return {re.compile(k, flags=re.IGNORECASE): v for k, v in self.stress_dict["patterns"].items()}

    def apply_pronunciation_rules(self, text):
        # Обработка regex
        for pattern, replacement in self.regex_patterns.items():
            text = pattern.sub(replacement, text)
        
        # Простые замены слов
        for word, stressed in self.stress_dict["words"].items():
            text = text.replace(word, stressed)
        
        return text

    def force_split_text(self, text):
        """
        Разбивает текст на более мелкие части для безопасной обработки в TTS
        """
        if not text:
            return []
            
        # Разбиваем текст на предложения
        sentences = re.split(r'([.!?…]+\s+)', text)
        
        # Собираем предложения с их окончаниями
        complete_sentences = []
        for i in range(0, len(sentences)-1, 2):
            if i+1 < len(sentences):
                complete_sentences.append(sentences[i] + sentences[i+1])
            else:
                complete_sentences.append(sentences[i])
                
        # Если предложений не нашлось, рассматриваем весь текст как одно предложение
        if not complete_sentences:
            complete_sentences = [text]
            
        chunks = []
        current_chunk = ""
        
        for sentence in complete_sentences:
            # Если предложение само по себе больше максимального размера,
            # его нужно разделить
            if len(sentence) > self.max_chunk_size:
                # Если текущий чанк не пустой, добавляем его
                if current_chunk:
                    chunks.append(current_chunk)
                    current_chunk = ""
                
                # Разделяем длинное предложение на части по знакам препинания
                parts = re.split(r'([,;:—–-]\s+)', sentence)
                
                # Собираем части в чанки
                temp_chunk = ""
                for i in range(0, len(parts)):
                    part = parts[i]
                    if len(temp_chunk) + len(part) <= self.max_chunk_size:
                        temp_chunk += part
                    else:
                        # Если часть не влезает в текущий чанк, но чанк не пустой
                        if temp_chunk:
                            chunks.append(temp_chunk)
                            temp_chunk = part
                        else:
                            # Если часть сама по себе слишком большая, принудительно разделяем
                            j = 0
                            while j < len(part):
                                # Находим ближайший конец слова
                                end = min(j + self.max_chunk_size, len(part))
                                # Если можем, то ищем конец слова
                                if end < len(part):
                                    space_pos = part.rfind(' ', j, end)
                                    if space_pos > j:
                                        end = space_pos
                                
                                chunks.append(part[j:end].strip())
                                j = end
                
                # Добавляем последний фрагмент
                if temp_chunk:
                    chunks.append(temp_chunk)
            
            # Если добавление текущего предложения превысит лимит
            elif len(current_chunk) + len(sentence) > self.max_chunk_size:
                chunks.append(current_chunk)
                current_chunk = sentence
            else:
                if current_chunk:
                    current_chunk += " " + sentence
                else:
                    current_chunk = sentence
        
        # Добавляем последний чанк
        if current_chunk:
            chunks.append(current_chunk)
        
        # Убираем пустые чанки и лишние пробелы
        return [chunk.strip() for chunk in chunks if chunk.strip()]

    def convert_to_opus(self, wav_path, opus_path):
        """Конвертирует WAV в Opus с обработкой ошибок"""
        try:
            # Проверяем, существует ли исходный файл
            if not os.path.exists(wav_path):
                print(f"⚠️ Исходный файл не существует: {wav_path}")
                return False
            
            # Конвертируем в opus
            cmd = ['ffmpeg', '-y', '-i', wav_path, '-c:a', 'libopus', '-b:a', '64k', opus_path]
            result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            
            # Проверяем результат
            if result.returncode != 0:
                print(f"⚠️ Ошибка конвертации в opus: {result.stderr.decode('utf-8', errors='ignore')}")
                return False
            
            return True
            
        except Exception as e:
            print(f"⚠️ Ошибка при конвертации: {str(e)}")
            return False

    def process_folder(self, input_folder, output_folder, delay=2):
        """
        Обрабатывает все markdown-файлы в указанной директории
        """
        os.makedirs(output_folder, exist_ok=True)
        files = [f for f in os.listdir(input_folder) if f.endswith(".md")]
        
        for file in tqdm(files, desc="Обработка файлов"):
            input_path = os.path.join(input_folder, file)
            output_path = os.path.join(output_folder, f"{os.path.splitext(file)[0]}.wav")
            opus_path = os.path.join(output_folder, f"{os.path.splitext(file)[0]}.opus")
            
            print(f"\n📄 Обрабатываем файл: {file}")
            
            try:
                with open(input_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                # Очищаем текст от markdown-разметки
                clean_text = self.md_to_clean_text(content)
                
                # Применяем правила произношения
                processed_text = self.apply_pronunciation_rules(clean_text)
                
                # Разбиваем текст на безопасные фрагменты
                chunks = self.force_split_text(processed_text)
                print(f"🔄 Текст разделен на {len(chunks)} частей для безопасной обработки")
                
                # Создаем временную директорию для фрагментов
                temp_dir = os.path.join(output_folder, "temp_chunks")
                os.makedirs(temp_dir, exist_ok=True)
                
                # Обрабатываем каждый фрагмент
                temp_files = []
                
                for i, chunk in enumerate(tqdm(chunks, desc=f"Обработка частей файла {file}")):
                    temp_path = os.path.join(temp_dir, f"temp_{i:04d}.wav")
                    
                    # Пробуем озвучить фрагмент с несколькими попытками
                    success = False
                    for attempt in range(3):  # 3 попытки
                        try:
                            self.tts.tts_to_file(
                                text=chunk,
                                speaker_wav=self.voice_clone,
                                language="ru",
                                file_path=temp_path
                            )
                            success = True
                            break
                        except torch.cuda.OutOfMemoryError:
                            print(f"\n⚠️ Ошибка памяти CUDA для части {i+1}/{len(chunks)}. Попытка {attempt+1}/3")
                            torch.cuda.empty_cache()
                            gc.collect()
                            sleep(2)
                        except Exception as e:
                            print(f"\n⚠️ Ошибка при обработке части {i+1}/{len(chunks)}: {str(e)}. Попытка {attempt+1}/3")
                            sleep(1)
                    
                    # Если успешно создали временный файл, добавляем его в список
                    if success and os.path.exists(temp_path):
                        temp_files.append(temp_path)
                    else:
                        print(f"\n⚠️ Не удалось обработать часть {i+1}/{len(chunks)}")
                    
                    # Очистка памяти после каждого фрагмента
                    torch.cuda.empty_cache()
                    gc.collect()
                    sleep(0.5)
                
                # Объединяем все фрагменты в один файл, если они есть
                if temp_files:
                    try:
                        # Создаем список файлов для ffmpeg
                        concat_list = os.path.join(temp_dir, "concat_list.txt")
                        with open(concat_list, 'w', encoding='utf-8') as f:
                            for temp_file in temp_files:
                                if os.path.exists(temp_file):
                                    f.write(f"file '{os.path.abspath(temp_file)}'\n")
                        
                        # Объединяем файлы с помощью ffmpeg
                        cmd = [
                            'ffmpeg', '-y', '-f', 'concat', '-safe', '0', 
                            '-i', concat_list, '-c', 'copy', output_path
                        ]
                        subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
                        
                        print(f"✅ Аудио успешно создано: {output_path}")
                        
                        # Конвертируем WAV в Opus
                        if self.convert_to_opus(output_path, opus_path):
                            print(f"✅ Создана Opus-версия: {opus_path}")
                        
                    except Exception as e:
                        print(f"\n❌ Ошибка при объединении аудиофайлов: {str(e)}")
                        
                        # Если не удалось объединить, копируем первый файл как результат
                        if temp_files and os.path.exists(temp_files[0]):
                            import shutil
                            shutil.copy(temp_files[0], output_path)
                            print(f"\n⚠️ Сохранен только первый фрагмент как {output_path}")
                
                # Удаляем временные файлы
                for temp_file in temp_files:
                    try:
                        if os.path.exists(temp_file):
                            os.remove(temp_file)
                    except:
                        pass
                
                if os.path.exists(concat_list):
                    try:
                        os.remove(concat_list)
                    except:
                        pass
                
                try:
                    os.rmdir(temp_dir)
                except:
                    pass
                
                # Пауза между файлами
                sleep(delay)
                
            except Exception as e:
                print(f"\n❌ Ошибка в файле {file}: {str(e)}")
                continue

# Инициализация пайплайна
print("\nЗапуск пайплайна...")
pipeline = MarkdownTTSPipeline(
    voice_clone_path="Recording.wav",
    tts_model=tts_model  # Передаем предварительно инициализированную модель
)

# Запуск обработки
pipeline.process_folder("novel", "audio")
print("\n✅ Обработка завершена успешно!")