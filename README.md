# Методы HIVES и DHF для группового принятия решений

## Курс

**Методы принятия решений**

---

## Цель проекта

Цель работы – на основе матриц парных сравнений при замене экспертов (по одному на цикл групповой работы) сохранить и повысить уровень консенсуса в обновлённой группе.

---

## Инструкция по запуску (тесты с разными экспертами и критериями)

### Установка

- Python 3.8+
- NumPy

```bash
pip install -r requirements.txt
```

### Запуск pipeline с авторскими примерами

Команда для одного combined JSON (с блоками `hives`, `dhf`, `rotation`, `combined_parameters`):

```bash
python main.py pipeline outputsTest/combined_5e6c.json
```

**Авторские примеры** (разные комбинации экспертов × критериев):

| Файл | Эксперты × Критерии |
|------|----------------------|
| `outputsTest/combined_5e6c.json` | 5 × 6 |
| `outputsTest/combined_5experts_8criteria.json` | 5 × 8 |
| `outputsTest/combined_7experts_6criteria.json` | 7 × 6 |
| `outputsTest/combined_7experts_8criteria.json` | 7 × 8 |

Один вызов создаёт два файла: `*_HIVES_GA` и `*_HIVES_HHO` (пути из `combined_parameters.output_path_ga` и `output_path_hho`).

**Чтобы сохранять результаты в папку `Result/`** (как в репозитории), в JSON в `combined_parameters` укажите, например:

- `output_path_ga`: `Result/5e6c/5e6c1ga.json`
- `output_path_hho`: `Result/5e6c/5e6c1hho.json`

Для нескольких прогонов (1–10) меняйте номера в имени файла и запускайте `pipeline` повторно.

**Запуск в Google Colab:** ноутбук `colab/HIVES_DHF_Colab.ipynb` — клонирование репозитория, установка зависимостей, генерация combined и запуск `pipeline` для конфигов 5e6c, 5e8c, 7e6c, 7e8c.

---

## Комментарии в коде (главные этапы для тестов с разным количеством экспертов и критериев)

- **`main.py`**
  - `run_pipeline_replace_ga_hho` (≈стр. 452) — пайплайн: загрузка combined JSON, проверка `rotation`, построение step2 (удаление одного эксперта), compat3, оптимизация весов GA и HHO, HIVES с плавной заменой (smooth replacement), сохранение `*ga.json` и `*hho.json`.
  - `_build_step2_combined_input` (≈стр. 179) — формирование combined без `predecessor_id`: фильтрация `hives.experts` и `dhf.dms` по оставшимся id.
  - `_maybe_load_predecessor_old_lambdas` и fallback в `run_pipeline_replace_ga_hho` — загрузка lambdas уходящего эксперта из файла или вычисление по исходной группе.
- **`hives_dhf/dhf_consensus.py`**
  - `optimize_expert_weights` (≈стр. 435) — DHF: GA или HHO для оптимизации весов экспертов.
  - `compat3` (≈стр. 97) — совместимость экспертов, консенсус как минимум по ним.
- **`hives_dhf/hives_method.py`**
  - `hives_rank` (≈стр. 259) — HIVES: ранжирование альтернатив, поддержка `smooth_replacement` при замене эксперта.
- **`colab/HIVES_DHF_Colab.ipynb`**
  - `build_and_run_pipeline` — сборка combined под конфиг (E×C) через `_generate_hives_payload` и `_generate_dhf_payload`, предварительный HIVES для lambdas предшественника, вызов `main.py pipeline`.

---

## Участники

Основными разработчиком проекта является Дмитрий А. Терещенко, Максим Р. Есаков, студенты ИКНК СПбПУ.

Руководитель и соавтор проекта – Владимир А. Пархоменко, старший преподаватель ИКНК СПбПУ.
---
## Гарантии

Разработчики не дают никаких гарантий по поводу использования данного программного обеспечения.

## Лицензия

Эта программа открыта для использования и распространяется под лицензией MIT.

## Входные данные

https://github.com/Karperash/Metod-HIVES-and-DHF

---

## Тестовый набор данных из базовой статьи(ей)

- `examples/dhf/input_data.json`  
- `examples/hives/input.json`

---

## Авторские примеры (для запуска тестов с разными экспертами и критериями)

- `outputsTest/combined_5e6c.json`  
- `outputsTest/combined_5experts_8criteria.json`  
- `outputsTest/combined_7experts_6criteria.json`  
- `outputsTest/combined_7experts_8criteria.json`

---

## Выходные данные (ссылка в статье)

Папка **`Result/`** — все результаты прогонов по конфигурациям 5e6c, 5e8c, 7e6c, 7e8c (`*ga.json`, `*hho.json`).
