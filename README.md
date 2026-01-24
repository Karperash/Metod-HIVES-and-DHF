# Методы HIVES и DHF для группового принятия решений

Комбинированный подход для ранжирования альтернатив с использованием методов **HIVES** (Hierarchical Voting-based Evaluation System) и **DHF** (Dual Hesitant Fuzzy) консенсуса.

## 📋 Содержание

- [Описание проекта](#описание-проекта)
- [Структура проекта](#структура-проекта)
- [Установка](#установка)
- [Использование](#использование)
- [Формат входных данных](#формат-входных-данных)
- [Описание методов](#описание-методов)
- [Примеры](#примеры)
- [Обновление репозитория на GitHub](#обновление-репозитория-на-github)

## 📖 Описание проекта

Проект реализует два метода для группового принятия решений:

1. **HIVES** — метод иерархического голосования для ранжирования альтернатив на основе критериев
2. **DHF Consensus** — метод консенсуса с двойными нечеткими множествами для оптимизации весов экспертов

Методы могут использоваться как отдельно, так и в комбинации, где DHF оптимизирует веса экспертов, которые затем используются в HIVES для более точного ранжирования.

## 📁 Структура проекта

```
.
├── main.py                 # Главный CLI-скрипт
├── hives_dhf/              # Основной пакет с реализацией методов
│   ├── __init__.py
│   ├── hives_method.py     # Реализация метода HIVES
│   ├── dhf_consensus.py    # Реализация DHF консенсуса (GA/HHO)
│   ├── json_input.py       # Загрузка данных из JSON
│   └── models.py           # Модели данных
├── examples/               # Примеры входных данных
│   ├── hives/              # Примеры для HIVES
│   ├── dhf/                # Примеры для DHF
│   └── combined/           # Примеры для комбинированного метода
├── outputs/                # Результаты выполнения
├── docs/                   # Документация и тезисы
├── notebooks/              # Jupyter notebooks
└── legacy/                 # Старый код (для справки)
```

## 🔧 Установка

### Требования

- Python 3.8+
- NumPy

### Локальная установка

```bash
pip install numpy
```

Или используйте файл requirements.txt:

```bash
pip install -r requirements.txt
```

### 🚀 Быстрый старт в Google Colab

1. Откройте [Google Colab](https://colab.research.google.com/)
2. Загрузите ноутбук `notebooks/colab/HIVES_DHF_Colab.ipynb` или используйте прямую ссылку:
   - Скопируйте содержимое из `notebooks/colab/HIVES_DHF_Colab.ipynb`
   - Или выполните в первой ячейке:

```python
# Клонируем репозиторий
!git clone https://github.com/Karperash/Metod-HIVES-and-DHF.git
%cd Metod-HIVES-and-DHF

# Устанавливаем зависимости
!pip install -q numpy

# Запускаем пример
!python main.py hives examples/hives/input.json
```

**Альтернативный способ (через GitHub):**
1. Перейдите на https://github.com/Karperash/Metod-HIVES-and-DHF
2. Откройте файл `notebooks/colab/HIVES_DHF_Colab.ipynb`
3. Нажмите кнопку "Open in Colab" (если доступна) или скопируйте содержимое в новый Colab ноутбук

## 🚀 Использование

### 1. Запуск только HIVES

```bash
python main.py hives examples/hives/input.json
```

Сохранение результата в файл:
```bash
python main.py hives examples/hives/input.json -o outputs/result.json
```

### 2. Запуск комбинированного метода (DHF → HIVES)

```bash
python main.py combined examples/combined/combined_input_program.json
```

Этот метод:
1. Оптимизирует веса экспертов с помощью DHF (GA или HHO)
2. Использует оптимизированные веса в HIVES для ранжирования альтернатив
3. Вычисляет консенсус до и после HIVES через `compat3`

### 3. Новый порядок выполнения (HIVES → compat3 → DHF)

```bash
python main.py hives-compat-dhf examples/combined/combined_input_program.json
```

Этот метод:
1. Выполняет HIVES с равномерными весами экспертов
2. Проверяет консенсус через `compat3` с равномерными весами
3. Оптимизирует веса экспертов через DHF
4. Пересчитывает консенсус с оптимизированными весами

### 4. Эксперимент: сравнение весов экспертов

```bash
python main.py experiment examples/combined/combined_input_program.json
```

Сравнивает веса экспертов в разных сценариях:
- HIVES без DHF (равномерные веса)
- Только DHF (GA)
- HIVES + DHF без замены
- HIVES + DHF с заменой (если задан rotation)

### 5. Сравнение порядка выполнения методов

```bash
python main.py compare-order examples/combined/combined_input_program.json
```

Показывает разницу между:
- DHF → HIVES (обычный порядок)
- HIVES → DHF (обратный порядок)

### 6. Pipeline: замена эксперта (step2 → compat3 → GA+HHO → HIVES)

```bash
python main.py pipeline path/to/combined_input.json
```

Полный пайплайн с заменой эксперта (5→4):
1. Формирует **step2** — combined JSON без `predecessor_id` (4 эксперта)
2. Проверяет консенсус через **compat3** (равномерные веса)
3. Запускает **GA** и **HHO** отдельно (с опциональным seed начальных весов)
4. **HIVES** с весами от GA и от HHO + плавная замена (smooth replacement)
5. Сохраняет: `*_step2.json`, `*_HIVES_GA.json`, `*_HIVES_HHO.json`

Требуется блок `rotation` в JSON (`predecessor_id`, `new_id`, `alpha`, `predecessor_old_lambdas_path`). В `combined_parameters` можно задать `step2_output_path`, `output_path_ga`, `output_path_hho`.

### 7. Генератор данных + compat3 → HIVES → DHF → compat3

```bash
python main.py gen-compat-hives-dhf-compat --criteria 8 --experts 4 --alternatives 3 --save-input outputs/Input.json -o outputs/out.json
```

Генерирует синтетические HIVES+DHF данные, прогоняет compat3 до/после DHF, сохраняет вход (`--save-input`) и результат (`-o`).

## 📝 Формат входных данных

### Для HIVES (отдельно)

```json
{
  "alternatives": ["A1", "A2", "A3"],
  "criteria": [
    { "name": "ES", "type": "positive" },
    { "name": "SS", "type": "positive" }
  ],
  "dms": [
    {
      "id": "DM1",
      "scores": [
        [70, 60],
        [60, 75],
        [80, 55]
      ]
    }
  ],
  "experts": [
    { "id": "DM1", "weights": [25, 10] },
    { "id": "DM2", "weights": [10, 25] }
  ]
}
```

### Для комбинированного метода

```json
{
  "hives": {
    "alternatives": ["A1", "A2", "A3"],
    "criteria": [...],
    "dms": [...],
    "experts": [...]
  },
  "dhf": {
    "criteria": ["ES", "SS", "EcS", "IP", "SA", "LTC"],
    "dms": [
      {
        "id": "DM1",
        "pairwise_comparisons": {
          "ES": {
            "ES": { "membership": [0.5], "non_membership": [0.5] },
            "SS": { "membership": [0.15], "non_membership": [0.8] }
          }
        }
      }
    ],
    "parameters": {
      "desired_consensus": 0.907,
      "population_size": 20,
      "max_iterations": 500
    }
  },
  "rotation": {
    "predecessor_id": "DM1",
    "new_id": "DM3",
    "alpha": 0.5,
    "predecessor_old_lambdas_path": "outputs/prev_result.json"
  },
  "combined_parameters": {
    "dhf_method": "HHO",
    "influence_mode": "continuous",
    "influence_min": 0.01,
    "influence_max": 0.99,
    "output_path": "outputs/result.json"
  }
}
```

## 🔬 Описание методов

### HIVES (Hierarchical Voting-based Evaluation System)

Метод для ранжирования альтернатив на основе:
- Оценок альтернатив по критериям (матрица `A`)
- Весов критериев у экспертов (матрица `W`)
- Весов экспертов (`influence`)

**Основные шаги:**
1. Вычисление вкладов экспертов по критериям (λ-матрица)
2. Вычисление весов критериев (γ)
3. Вычисление итоговых оценок альтернатив: `score = Σ(оценка × вес_критерия)`
4. Ранжирование по убыванию оценок

**Особенности:**
- Поддержка "плавной замены эксперта" (smooth expert replacement)
- Непрерывное влияние экспертов (continuous influence mode)
- Социальные ограничения (clamp + normalization)

### DHF Consensus (Dual Hesitant Fuzzy)

Метод оптимизации весов экспертов для достижения консенсуса на основе:
- Парных сравнений критериев (pairwise comparisons)
- Двойных нечетких множеств (membership/non-membership)

**Алгоритмы оптимизации:**
- **GA** (Genetic Algorithm) — генетический алгоритм
- **HHO** (Harris Hawks Optimization) — алгоритм оптимизации Харриса Хоука

**Функция совместимости:**
- `compat3` — вычисляет совместимость каждого эксперта
- Консенсус = минимум из совместимостей

### Комбинированный подход

**Порядок выполнения (DHF → HIVES):**
1. DHF оптимизирует веса экспертов для достижения консенсуса
2. Оптимизированные веса передаются в HIVES
3. HIVES использует эти веса для ранжирования альтернатив
4. Пересчёт консенсуса через `compat3` с оптимизированными весами

**Результат:**
- Ранжирование альтернатив с учётом оптимизированных весов экспертов
- Сравнение консенсуса до и после HIVES
- Веса критериев (gamma_scaled)
- Lambda-матрица (вклад экспертов по критериям)

## 📊 Примеры вывода

### Результат HIVES

```
Raw criterion weights gamma:
[15.03 15.04 15.07 15.03 15.04 10.1 ]

Scaled criterion weights gamma_scaled (sum = 100):
[17.62 17.63 17.66 17.62 17.63 11.84]

Final alternative scores:
  A1: 6409.93
  A2: 6472.06
  A3: 6735.72

Ranking (1-based):
[3 2 1]
```

### Результат комбинированного метода

```
[DHF] Best consensus: 0.6454 method: HHO
[DHF] Weights: {'DM1': 0.2856, 'DM2': 0.3101, 'DM3': 0.4044}

[HIVES] Raw criterion weights gamma:
[15.03 15.04 15.07 15.03 15.04 10.1 ]

[FINAL RANKING]
  1. A3 (score: 6735.72)
  2. A2 (score: 6472.06)
  3. A1 (score: 6409.93)

[CRITERIA WEIGHTS] (gamma_scaled, sum=100%)
  ES (Environmental Sustainability): 17.62%
  SS (Social Sustainability): 17.63%
  EcS (Economic Sustainability): 17.66%
  IP (Innovation Potential): 17.61%
  SA (Strategic Alignment): 17.63%
  LTC (Long-term Competitiveness): 11.84%
```

## 🔍 Ключевые понятия

### Scores (оценки альтернатив)

Итоговые оценки альтернатив вычисляются как:
```
score_i = Σ(A_ij × gamma_scaled_j)
```
где:
- `A_ij` — оценка альтернативы `i` по критерию `j`
- `gamma_scaled_j` — вес критерия `j` (сумма = 100%)

Чем выше score, тем лучше альтернатива.

### Consensus (консенсус)

Уровень согласия между экспертами, вычисляемый через `compat3`:
- Для каждого эксперта вычисляется совместимость
- Консенсус = минимум из совместимостей
- Чем выше консенсус (ближе к 1.0), тем больше согласие

### Influence (влияние экспертов)

Веса экспертов, определяющие их вклад в итоговое решение:
- Оптимизируются через DHF для достижения консенсуса
- Используются в HIVES для взвешивания вкладов экспертов

### Lambda (λ-матрица)

Матрица вкладов экспертов по критериям:
- `λ_ij` — вклад эксперта `i` в критерий `j`
- Сумма по столбцу = 100% для каждого критерия

### Gamma (γ — веса критериев)

Итоговые веса критериев:
- Вычисляются на основе λ-матрицы и весов критериев у экспертов
- Масштабируются до суммы 100%

## 🔄 Обновление репозитория на GitHub

Как **полностью обновить** [репозиторий](https://github.com/Karperash/Metod-HIVES-and-DHF): в GitHub окажется ровно то, что в проекте (лишнее будет удалено, новое — добавлено).

### Что понадобится

- **Git** установлен, в проекте выполнен `git init` и настроен `origin`:
  ```bash
  git remote -v
  # origin  https://github.com/Karperash/Metod-HIVES-and-DHF.git (fetch)
  # origin  https://github.com/Karperash/Metod-HIVES-and-DHF.git (push)
  ```
- **Доступ к GitHub**: логин/пароль или токен, либо SSH-ключ (если репо не публичный или есть ограничения).

---

### Способ 1: скрипт `update_repo.ps1`

В **PowerShell** откройте **корень проекта** (папка с `main.py`) и выполните:

```powershell
.\update_repo.ps1
```

Скрипт:

1. Снимает с отслеживания `outputs/`, `outputsTest/`, `Resulst/`, `comparison_results.json`, `my_dhfs_data.json` (если они есть в репо).
2. Делает `git add -A` — все новые, изменённые и удалённые файлы.
3. Показывает `git status`.
4. Спрашивает **«Commit? (y/n)»** — при `y` создаёт коммит.

**Загрузка в GitHub вручную:**

```powershell
git push origin main
```

Без вопроса перед коммитом (например, в скриптах):

```powershell
.\update_repo.ps1 -Force
```

---

### Способ 2: команды вручную

В корне проекта:

```powershell
# 1) Перестать отслеживать то, что в .gitignore (если уже в репо)
git rm -r --cached outputs outputsTest Resulst 2>$null
git rm --cached comparison_results.json my_dhfs_data.json 2>$null

# 2) Взять все изменения (в т.ч. удаления)
git add -A

# 3) Посмотреть, что попадёт в коммит
git status

# 4) Создать коммит
git commit -m "Sync repo with project: ..."

# 5) Отправить на GitHub
git push origin main
```

---

### Если `git push` просит логин/пароль

- **HTTPS**: используйте [Personal Access Token](https://github.com/settings/tokens) вместо пароля.
- **SSH**: настройте ключ и замените `origin` на  
  `git@github.com:Karperash/Metod-HIVES-and-DHF.git`.

---

### Если в GitHub есть коммиты, которых нет у вас

Сначала подтяните их и только потом пушите:

```powershell
git pull origin main --rebase
git push origin main
```

Если появятся конфликты — их нужно разрешить вручную, затем `git add` и `git rebase --continue` (или `git merge --continue`).

---

### Что окажется в репозитории

- Исходный код: `main.py`, `hives_dhf/`, `examples/`, `legacy/`, `README.md`, `requirements.txt`, `.gitignore`, `update_repo.ps1`.
- **Не** попадут (из-за `.gitignore`): `outputs/`, `outputsTest/`, `Resulst/`, `comparison_results.json`, `my_dhfs_data.json`, `__pycache__/` и т.п.

## 📚 Дополнительная информация

- Подробная документация: `docs/README.md`
- Примеры использования: `examples/`
- Jupyter notebooks: `notebooks/`
- **Инструкция для Google Colab**: `notebooks/colab/README_Colab.md` или используйте готовый ноутбук `notebooks/colab/HIVES_DHF_Colab.ipynb`

## 👥 Авторы

Проект разработан на основе методов HIVES и DHF консенсуса.

## 📄 Лицензия

[Указать лицензию при необходимости]

