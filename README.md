# Методы HIVES и DHF для группового принятия решений

Проект реализует **HIVES** (иерархическое голосование) и **DHF** (Dual Hesitant Fuzzy) консенсус. Методы можно запускать по отдельности или в комбинации.

## Содержание

- [Установка](#установка)
- [1. Запуск только HIVES](#1-запуск-только-hives)
- [2. Запуск только DHF](#2-запуск-только-dhf)
- [3. Комбинации методов](#3-комбинации-методов)
- [Формат входных данных](#формат-входных-данных)
- [Структура проекта](#структура-проекта)
- [Обновление репозитория на GitHub](#обновление-репозитория-на-github)

---

## Установка

- Python 3.8+
- NumPy

```bash
pip install -r requirements.txt
```

---

## 1. Запуск только HIVES

**Через подкоманду:**
```bash
python main.py hives examples/hives/input.json
```

**Сохранение результата в JSON:**
```bash
python main.py hives examples/hives/input.json -o outputs/result.json
```

**Краткая форма** (только для HIVES, один аргумент — путь к `.json`):
```bash
python main.py examples/hives/input.json
```

---

## 2. Запуск только DHF

DHF запускается отдельным скриптом:

```bash
python legacy/main4.py
```

**Поведение по умолчанию:**
- Генерирует `my_dhfs_data.json` (6 критериев, 4 эксперта)
- Запускает GA и HHO для оптимизации весов экспертов
- Сохраняет сравнение в `comparison_results.json`

**Использование своего DHF-файла** (например `examples/dhf/input_data.json`):  
В `legacy/main4.py` в функции `main()` закомментируйте вызов `generate_and_save_dhfs_json(...)` и замените `load_input_data("my_dhfs_data.json")` на `load_input_data("examples/dhf/input_data.json")`.

Формат DHF: `criteria`, `dms` с `pairwise_comparisons` (membership / non_membership), `parameters.desired_consensus`.

---

## 3. Комбинации методов

Все команды ниже работают с **combined JSON** (блоки `hives`, `dhf`, при необходимости `rotation`, `combined_parameters`). Пример: `examples/combined/combined_input_program.json`.

### 3.1 DHF → HIVES (`combined`)

```bash
python main.py combined examples/combined/combined_input_program.json
```

DHF оптимизирует веса экспертов (GA или HHO из `combined_parameters.dhf_method`), HIVES ранжирует альтернативы с этими весами. Консенсус до/после — через `compat3`. Путь вывода: `combined_parameters.output_path`.

---

### 3.2 HIVES → compat3 → DHF (`hives-compat-dhf`)

```bash
python main.py hives-compat-dhf examples/combined/combined_input_program.json
```

Сначала HIVES с равномерными весами, затем `compat3`, затем DHF, затем пересчёт консенсуса с оптимизированными весами.

---

### 3.3 Сравнение весов (`experiment`)

```bash
python main.py experiment examples/combined/combined_input_program.json
```

Сравнивает: HIVES без DHF; только DHF (GA); HIVES+DHF без замены; HIVES+DHF с заменой (если задан `rotation`).

---

### 3.4 Сравнение порядка DHF↔HIVES (`compare-order`)

```bash
python main.py compare-order examples/combined/combined_input_program.json
```

Показывает разницу между порядками: DHF→HIVES и HIVES→DHF.

---

### 3.5 Pipeline: замена эксперта, GA и HHO отдельно (`pipeline`)

```bash
python main.py pipeline path/to/combined_input.json
```

**Требуется:**
- блок `rotation`: `predecessor_id`, `new_id`, `alpha`, `predecessor_old_lambdas_path`
- блок `combined_parameters` (опционально: `step2_output_path`, `output_path_ga`, `output_path_hho`)

**Шаги:**
1. **step2** — combined без `predecessor_id` (удаляется один эксперт, например 5→4 или 7→6)
2. **compat3** с равномерными весами
3. **GA** и **HHO** по отдельности (с опциональными начальными весами)
4. **HIVES** с весами от GA и от HHO + плавная замена (smooth replacement)
5. Сохранение: `*_step2.json`, `*_HIVES_GA.json`, `*_HIVES_HHO.json`

Значения по умолчанию: `step2_output_path=outputs/step2_combined.json`, `output_path_ga=outputs/result_hives_ga.json`, `output_path_hho=outputs/result_hives_hho.json`.

---

### 3.6 Генератор + compat3 → HIVES → DHF → compat3 (`gen-compat-hives-dhf-compat`)

```bash
python main.py gen-compat-hives-dhf-compat --criteria 8 --experts 4 --alternatives 3 --save-input outputs/Input.json -o outputs/out.json
```

Генерирует синтетические HIVES+DHF данные, прогоняет цепочку, при `--save-input` сохраняет вход, при `-o` — итог.

**Параметры:** `--criteria`, `--experts`, `--alternatives`, `--seed`, `--dhf-method` (GA | HHO), `--output` / `-o`, `--save-input`.

---

### 3.7 Тесты с разными комбинациями экспертов и критериев

Для воспроизведения прогонов, результаты которых лежат в `Result/`, использовались комбинации:

| Комбинация | Папка   | Эксперты × Критерии |
|------------|---------|----------------------|
| 5e6c       | 5e6c/   | 5 × 6                |
| 5e8c       | 5e8c/   | 5 × 8                |
| 7e6c       | 7e6c/   | 7 × 6                |
| 7e8c       | 7e8c/   | 7 × 8                |

**Подготовка:**
- Combined JSON с нужным числом экспертов и критериев (вручную или через генератор, например `gen-compat-hives-dhf-compat --save-input` с подходящими `--experts` и `--criteria`).
- Блок `rotation` с `predecessor_id`, `new_id`, `alpha`, `predecessor_old_lambdas_path` (путь к предыдущему результату с lambdas или fallback по коду).
- В `combined_parameters`:
  - `output_path_ga` = `Result/5e6c/5e6c1ga.json`
  - `output_path_hho` = `Result/5e6c/5e6c1hho.json`
  (и аналогично для 5e8c, 7e6c, 7e8c; при 10 прогонах — номера 1…10 в имени файла.)

**Запуск для одного прогона:**
```bash
python main.py pipeline path/to/combined_5e6c.json
```

Один вызов `pipeline` создаёт два файла: `*ga.json` и `*hho.json`. Для 10 прогонов (5e6c1ga … 5e6c10ga, 5e6c1hho … 5e6c10hho) нужно 10 раз задать разные `output_path_ga` и `output_path_hho` в JSON (или скриптом подставлять пути) и каждый раз запустить `pipeline`. Входной combined может быть один и тот же или разный (например, при разном `--seed` у генератора).

**Структура `Result/`:**
```
Result/
├── 5e6c/   # 5 экспертов, 6 критериев: 5e6c1ga.json, 5e6c1hho.json, … 5e6c10ga.json, 5e6c10hho.json
├── 5e8c/
├── 7e6c/
└── 7e8c/
```

---

## Формат входных данных

### HIVES (отдельно)

`alternatives`, `criteria` (массив `{name, type}`), `dms` (id, `scores`: альтернатива×критерий), `experts` (id, `weights` по критериям). Пример: `examples/hives/input.json`.

### Combined (комбинации и pipeline)

- **hives** — как для HIVES.
- **dhf** — `criteria`, `dms` с `pairwise_comparisons` (membership / non_membership), `parameters` (desired_consensus и др.).
- **rotation** (для `pipeline`, `experiment` с заменой): `predecessor_id`, `new_id`, `alpha`, `predecessor_old_lambdas_path`.
- **combined_parameters**: `dhf_method` (GA|HHO), `influence_mode`, `influence_min`, `influence_max`, `output_path`; для `pipeline` ещё `step2_output_path`, `output_path_ga`, `output_path_hho`.

Пример: `examples/combined/combined_input_program.json`.

---

## Структура проекта

```
.
├── main.py              # CLI: hives, combined, experiment, compare-order, hives-compat-dhf, pipeline, gen-compat-hives-dhf-compat
├── hives_dhf/           # HIVES, DHF, compat3, загрузка JSON
│   ├── hives_method.py
│   ├── dhf_consensus.py
│   ├── json_input.py
│   └── models.py
├── examples/
│   ├── hives/           # input.json
│   ├── dhf/             # input_data.json
│   └── combined/        # combined_input_program.json и др.
├── legacy/
│   └── main4.py         # Запуск только DHF (GA+HHO, comparison_results.json)
├── Result/              # Результаты тестов 5e6c, 5e8c, 7e6c, 7e8c (*ga.json, *hho.json)
├── requirements.txt
├── .gitignore
└── update_repo.ps1


