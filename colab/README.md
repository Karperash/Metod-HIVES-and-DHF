# Запуск проекта HIVES и DHF в Google Colab

Инструкция по развёртыванию проекта в [Google Colab](https://colab.research.google.com/) для воспроизведения тестов с разными комбинациями экспертов и критериев (раздел 3.7 README: 5e6c, 5e8c, 7e6c, 7e8c).

---

## 1. Открыть ноутбук в Colab

**Вариант A: из репозитория на GitHub**

1. Откройте [https://github.com/Karperash/Metod-HIVES-and-DHF](https://github.com/Karperash/Metod-HIVES-and-DHF).
2. Перейдите в папку `colab/` и откройте `HIVES_DHF_Colab.ipynb`.
3. Нажмите **Open in Colab** (если доступно) или: **Raw** → скопируйте URL, в Colab: *Файл → Открыть блокнот → Вкладка «GitHub»* — вставьте URL репозитория и выберите `colab/HIVES_DHF_Colab.ipynb`.

**Вариант B: загрузить файл**

1. Скачайте `colab/HIVES_DHF_Colab.ipynb` к себе.
2. В [Google Colab](https://colab.research.google.com/): *Файл → Загрузить блокнот* — выберите `HIVES_DHF_Colab.ipynb`.

---

## 2. Что делает ноутбук

1. **Установка**  
   Клонирует репозиторий в `/content/Metod-HIVES-and-DHF`, ставит зависимости (`pip install -r requirements.txt`), переходит в корень проекта.

2. **Быстрые проверки (по желанию)**  
   - Запуск только HIVES: `main.py hives examples/hives/input.json`  
   - Запуск только DHF: `legacy/main4.py`

3. **Подготовка к тестам 5e6c, 5e8c, 7e6c, 7e8c**  
   - Создаёт каталоги `Result/5e6c`, `Result/5e8c`, `Result/7e6c`, `Result/7e8c` и `outputs/`.
   - Для каждой комбинации (эксперты × критерии) генерирует combined JSON с блоком `rotation` и `combined_parameters` (`output_path_ga`, `output_path_hho` в `Result/...`).
   - Для `pipeline` нужны lambdas предшественника: один раз запускается HIVES по блоку `hives`, результат сохраняется в `outputs/prev_<конфиг>_<run>.json` и указывается в `rotation.predecessor_old_lambdas_path`.

4. **Запуск `pipeline`**  
   Для каждого выбранного конфига (например, 5e6c) и номера прогона (1…10) выполняется:
   ```bash
   python main.py pipeline /path/to/combined_<конфиг>_run<N>.json
   ```
   В `Result/<конфиг>/` появляются файлы `*ga.json` и `*hho.json` (например, `5e6c1ga.json`, `5e6c1hho.json`).

5. **Режимы в ноутбуке**  
   - **Демо:** один прогон для 5e6c (5 экспертов × 6 критериев), чтобы быстро проверить, что всё работает.  
   - **Расширенный:** несколько прогонов и комбинаций (5e6c, 5e8c, 7e6c, 7e8c). Имейте в виду: GA и HHO занимают время; полный сет (4 конфига × 10 прогонов) может выполняться долго.

---

## 3. Переменные в ноутбуке

- **`CONFIGS`** — список конфигов: `["5e6c"]`, `["5e6c","5e8c"]` или `["5e6c","5e8c","7e6c","7e8c"]`.
- **`N_RUNS`** — число прогонов на конфиг (1–10). Для демо достаточно `1`.
- **`BASE_SEED`** — база для `seed` при генерации (для воспроизводимости).

---

## 4. Ограничения Colab

- **Таймаут:** при долгом бездействии Colab может отключить рантайм. Для длинных прогонов периодически взаимодействуйте с ячейками или используйте Colab Pro.
- **Файлы:** всё в `/content/` (в т.ч. `Result/`) при отключении теряется. Чтобы сохранить `Result/`, в конце ноутбука можно добавить скачивание:
  ```python
  from google.colab import files
  import shutil
  shutil.make_archive("Result", "zip", "Result")
  files.download("Result.zip")
  ```

---

## 5. Соответствие README (раздел 3.7)

| Комбинация | Эксперты × Критерии | Папка в `Result/` |
|------------|----------------------|-------------------|
| 5e6c       | 5 × 6                | `Result/5e6c/`    |
| 5e8c       | 5 × 8                | `Result/5e8c/`    |
| 7e6c       | 7 × 6                | `Result/7e6c/`    |
| 7e8c       | 7 × 8                | `Result/7e8c/`    |

Имена файлов: `5e6c1ga.json`, `5e6c1hho.json`, …, `5e6c10ga.json`, `5e6c10hho.json` и по той же схеме для 5e8c, 7e6c, 7e8c.
