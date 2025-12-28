# Инструкция по развертыванию проекта в Google Colab

## Быстрый старт

### Вариант 1: Использование готового ноутбука

1. Откройте [Google Colab](https://colab.research.google.com/)
2. Загрузите файл `HIVES_DHF_Colab.ipynb` из папки `notebooks/colab/`
3. Запустите все ячейки последовательно

### Вариант 2: Ручная настройка

Создайте новый ноутбук в Colab и выполните следующие шаги:

#### Шаг 1: Клонирование репозитория

```python
# Клонируем репозиторий
!git clone https://github.com/Karperash/Metod-HIVES-and-DHF.git
%cd Metod-HIVES-and-DHF
```

#### Шаг 2: Установка зависимостей

```python
# Устанавливаем зависимости
!pip install -q numpy
```

#### Шаг 3: Запуск примеров

**Пример 1: Только HIVES**
```python
!python main.py hives examples/hives/input.json
```

**Пример 2: Комбинированный метод (DHF → HIVES)**
```python
!python main.py combined examples/combined/combined_input_program.json
```

**Пример 3: Эксперимент**
```python
!python main.py experiment examples/combined/combined_input_program.json
```

**Пример 4: HIVES → compat3 → DHF**
```python
!python main.py hives-compat-dhf examples/combined/combined_input_program.json
```

## Использование Python API

Если вы хотите использовать методы программно в Colab:

```python
import sys
sys.path.insert(0, '/content/Metod-HIVES-and-DHF')

from hives_dhf.json_input import load_decision_problem_from_json
from hives_dhf.hives_method import hives_rank
from hives_dhf.dhf_consensus import optimize_expert_weights
import numpy as np

# Пример использования HIVES
problem = load_decision_problem_from_json('examples/hives/input.json')
A = problem.aggregated_performance()
W = problem.weights_matrix()
expert_ids = [e.id for e in problem.experts] if problem.experts else [f"DM{i+1}" for i in range(W.shape[0])]

result = hives_rank(A=A, W=W, expert_ids=expert_ids)
print(f"Оценки альтернатив: {result['alt_scores']}")
print(f"Ранжирование: {result['ranking'] + 1}")
```

## Работа с собственными данными

1. Создайте JSON файл с вашими данными (см. формат в основном README)
2. Загрузите файл в Colab (через интерфейс или используя `files.upload()`)
3. Запустите:

```python
!python main.py hives your_data.json
```

## Полезные команды

**Просмотр структуры проекта:**
```python
!ls -la
!tree -L 2  # если установлен tree
```

**Просмотр примеров данных:**
```python
!cat examples/hives/input.json
```

**Сохранение результатов:**
```python
!python main.py hives examples/hives/input.json -o outputs/my_result.json
```

## Примечания

- Все файлы проекта будут находиться в `/content/Metod-HIVES-and-DHF/`
- Результаты можно сохранять в папку `outputs/` или скачать через интерфейс Colab
- Для работы с большими данными рекомендуется использовать GPU runtime (Runtime → Change runtime type → GPU)

## Ссылки

- Репозиторий: https://github.com/Karperash/Metod-HIVES-and-DHF
- Основной README: см. файл `README.md` в корне проекта

