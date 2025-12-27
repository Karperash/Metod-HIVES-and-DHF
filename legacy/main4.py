import json
import random
import numpy as np
from typing import List, Dict, Tuple, Any


def load_input_data(file_path): # Загрузка входных данных
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        print("Данные успешно загружены")
        print(f"Количество критериев: {len(data['criteria'])}, Количество экспертов: {len(data['dms'])}")
        return data
    except Exception as e:
        print(f"Ошибка загрузки данных: {e}")
        return None


def save_results_to_json(results, filename="results.json"):
    try:
        # Преобразуем булевы значения в строки для JSON
        json_serializable_results = {}
        for key, value in results.items():
            if isinstance(value, bool):
                json_serializable_results[key] = str(value)
            else:
                json_serializable_results[key] = value

        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(json_serializable_results, f, ensure_ascii=False, indent=2)

        print(f"Результаты успешно сохранены в файл: {filename}")
        return True

    except Exception as e:
        print(f"Ошибка сохранения результатов: {e}")
        return False


def json_to_matrix_format(data): # Преобразование входных данных в матрицы для алгоритма
    criteria = data['criteria']
    dms_data = data['dms']
    num_criteria = len(criteria)
    num_experts = len(dms_data)
    max_dhf_elements = 3  # Максимальное количество элементов DHFS

    # Инициализация матриц с динамическими размерами
    Judgments_membership = np.zeros((num_criteria, max_dhf_elements, num_criteria, num_experts))
    Judgments_nonmembership = np.zeros((num_criteria, max_dhf_elements, num_criteria, num_experts))

    # Заполнение матриц
    for dm_idx, dm in enumerate(dms_data):
        comparisons = dm['pairwise_comparisons']

        for i, crit1 in enumerate(criteria):
            for j, crit2 in enumerate(criteria):
                comp_data = comparisons[crit1][crit2]

                # Membership - всегда max_dhf_elements значений, даже если нули
                membership_vals = comp_data['membership']
                for k in range(max_dhf_elements):
                    if k < len(membership_vals):
                        Judgments_membership[i, k, j, dm_idx] = membership_vals[k]
                    else:
                        Judgments_membership[i, k, j, dm_idx] = 0  # заполняем нулями

                # Non-membership - всегда max_dhf_elements значений, даже если нули
                nonmembership_vals = comp_data['non_membership']
                for k in range(max_dhf_elements):
                    if k < len(nonmembership_vals):
                        Judgments_nonmembership[i, k, j, dm_idx] = nonmembership_vals[k]
                    else:
                        Judgments_nonmembership[i, k, j, dm_idx] = 0

    print(f"Матрицы созданы: {num_criteria} критериев × {num_experts} экспертов")
    return Judgments_membership, Judgments_nonmembership, criteria


def compat3(W, Judgments, Judgments2): # функция, которая агрегирует матрицы внутри ГА
    """
    W: массив весов экспертов [w1, w2, ..., wn]
    Judgments: матрица принадлежностей размером (num_crit, max_dhf, num_crit, num_exp)
    Judgments2: матрица непринадлежностей размером (num_crit, max_dhf, num_crit, num_exp)
    """
    num_criteria = Judgments.shape[0]
    max_dhf = Judgments.shape[1]
    num_experts = Judgments.shape[3]

    # Проверка размеров
    if len(W) != num_experts:
        raise ValueError(f"Количество весов ({len(W)}) не совпадает с количеством экспертов ({num_experts})")

    # MEMBERSHIP АГРЕГАЦИЯ
    DHFWHM = np.zeros((num_criteria, num_criteria))
    sum_count = np.zeros((num_criteria, num_criteria))

    # Динамическое агрегирование для любого количества экспертов
    # Используем рекурсивный подход или итерации
    indices = []
    for _ in range(num_experts):
        indices.append(range(max_dhf))

    # Создаем все комбинации индексов
    import itertools
    all_combinations = list(itertools.product(*[range(max_dhf) for _ in range(num_experts)]))

    for i in range(num_criteria):
        for j in range(num_criteria):
            valid_combinations = 0

            for comb in all_combinations:
                # Проверяем, все ли значения ненулевые
                all_nonzero = True
                for exp_idx in range(num_experts):
                    if Judgments[i, comb[exp_idx], j, exp_idx] == 0:
                        all_nonzero = False
                        break

                if all_nonzero:
                    # Вычисляем агрегированное значение
                    term1 = 1.0
                    term2 = 1.0

                    for exp_idx in range(num_experts):
                        val = Judgments[i, comb[exp_idx], j, exp_idx]
                        term1 *= (val ** W[exp_idx])
                        term2 *= ((1 - val) ** W[exp_idx])

                    term3 = term1 + term2

                    if term3 != 0:
                        result = term1 / term3
                        DHFWHM[i, j] += result
                        valid_combinations += 1

            sum_count[i, j] = valid_combinations

    membership = np.zeros((num_criteria, num_criteria))
    for i in range(num_criteria):
        for j in range(num_criteria):
            if sum_count[i, j] > 0:
                membership[i, j] = DHFWHM[i, j] / sum_count[i, j]
            else:
                membership[i, j] = 0.5  # значение по умолчанию

    # NON-MEMBERSHIP АГРЕГАЦИЯ
    DHFWHM2 = np.zeros((num_criteria, num_criteria))
    sum_count2 = np.zeros((num_criteria, num_criteria))

    for i in range(num_criteria):
        for j in range(num_criteria):
            valid_combinations = 0

            for comb in all_combinations:
                # Проверяем, все ли значения ненулевые
                all_nonzero = True
                for exp_idx in range(num_experts):
                    if Judgments2[i, comb[exp_idx], j, exp_idx] == 0:
                        all_nonzero = False
                        break

                if all_nonzero:
                    # Вычисляем агрегированное значение
                    term1 = 1.0
                    term2 = 1.0

                    for exp_idx in range(num_experts):
                        val = Judgments2[i, comb[exp_idx], j, exp_idx]
                        term1 *= (val ** W[exp_idx])
                        term2 *= ((1 - val) ** W[exp_idx])

                    term3 = term1 + term2

                    if term3 != 0:
                        result2 = term1 / term3
                        DHFWHM2[i, j] += result2
                        valid_combinations += 1

            sum_count2[i, j] = valid_combinations

    nonmembership = np.zeros((num_criteria, num_criteria))
    for i in range(num_criteria):
        for j in range(num_criteria):
            if sum_count2[i, j] > 0:
                nonmembership[i, j] = DHFWHM2[i, j] / sum_count2[i, j]
            else:
                nonmembership[i, j] = 0.5  # значение по умолчанию

    # ВЫЧИСЛЕНИЕ СОВМЕСТИМОСТИ
    compatibility = [0] * num_experts

    for dm_idx in range(num_experts):  # для каждого эксперта
        calculus2 = 0

        # Membership
        value = 0
        x = 0
        calculus = 0

        for k in range(num_criteria):
            for l in range(num_criteria):
                for m in range(max_dhf):
                    if Judgments[k, m, l, dm_idx] != 0:
                        result = Judgments[k, m, l, dm_idx] * membership[k, l]
                        value += result
                        x += 1

                if x > 0:
                    calculus = value / x
                    calculus2 += calculus
                    x = 0
                    value = 0
                    calculus = 0

        # Non-membership
        value = 0
        x = 0
        calculus = 0

        for k in range(num_criteria):
            for l in range(num_criteria):
                for m in range(max_dhf):
                    if Judgments2[k, m, l, dm_idx] != 0:
                        result = Judgments2[k, m, l, dm_idx] * nonmembership[k, l]
                        value += result
                        x += 1

                if x > 0:
                    calculus = value / x
                    calculus2 += calculus
                    x = 0
                    value = 0
                    calculus = 0

        # вычисление числителя
        restf1 = 0
        restf2 = 0

        t = 0
        r = 0
        rest1 = 0
        rest11 = 0
        rest2 = 0
        rest22 = 0

        for k in range(num_criteria):
            for l in range(num_criteria):
                # Для эксперта
                for m in range(max_dhf):
                    if Judgments[k, m, l, dm_idx] != 0:
                        t += 1
                        rest1 += Judgments[k, m, l, dm_idx]
                    if Judgments2[k, m, l, dm_idx] != 0:
                        r += 1
                        rest11 += Judgments2[k, m, l, dm_idx]

                # Для агрегированной матрицы
                rest2 += membership[k, l]
                rest22 += nonmembership[k, l]

        if t > 0 and r > 0:
            restf1 = 1 - (rest1 / t) - (rest11 / r)
        restf2 = 1 - rest2 - rest22

        multrest = restf1 * restf2
        upside = multrest + calculus2

        # вычисление знаменателя дроби
        pertalone11 = 0
        npertalone11 = 0
        pertalone22 = 0
        npertalone22 = 0

        pertalone1 = 0
        npertalone1 = 0
        pertalone2 = 0
        npertalone2 = 0
        t = 0
        r = 0

        for k in range(num_criteria):
            for l in range(num_criteria):
                # Для эксперта
                for m in range(max_dhf):
                    if Judgments[k, m, l, dm_idx] != 0:
                        t += 1
                        pertalone1 += Judgments[k, m, l, dm_idx] ** 2
                    if Judgments2[k, m, l, dm_idx] != 0:
                        r += 1
                        npertalone1 += Judgments2[k, m, l, dm_idx] ** 2

                # Для агрегированной матрицы
                pertalone2 += membership[k, l] ** 2
                npertalone2 += nonmembership[k, l] ** 2

        if t > 0:
            pertalone11 = pertalone1 / t
        if r > 0:
            npertalone11 = npertalone1 / r

        pertalone22 = pertalone2
        npertalone22 = npertalone2

        # Дробь, вычисление совместимости
        restff1 = restf1 ** 2
        restff2 = restf2 ** 2

        denominator = (np.sqrt(pertalone11 + npertalone11 + restff1) *
                       np.sqrt(pertalone22 + npertalone22 + restff2))

        if denominator != 0:
            compatibility[dm_idx] = upside / denominator
        else:
            compatibility[dm_idx] = 0

    return compatibility


def genetic_algorithm_exact(Judgments_m, Judgments_n, criteria,
                            desired_consensus=0.907, population_size=20, max_iterations=500):
    print("\n Запуск Генетического Алгоритма")

    num_experts = Judgments_m.shape[3]

    # Инициализация лучших значений
    best_weights = np.ones(num_experts) / num_experts  # равные веса по умолчанию
    best_fitness = -1
    best_compatibility = []

    # Инициализация популяции
    population = []
    for _ in range(population_size):
        weights = np.random.random(num_experts)
        weights = weights / np.sum(weights)  # нормализация
        population.append(weights)

    for iteration in range(max_iterations):
        fitness_scores = []

        for weights in population:
            try:
                compatibility = compat3(weights, Judgments_m, Judgments_n)
                fitness = min(compatibility)
                fitness_scores.append(fitness)

                if fitness > best_fitness:
                    best_fitness = fitness
                    best_weights = weights.copy()
                    best_compatibility = compatibility.copy()

                    print(f"Итерация {iteration}: Новый лучший результат: {fitness:.4f}")
                    print(f"  Веса: {[round(w, 4) for w in weights]}")

            except Exception as e:
                print(f" Ошибка в compat3: {e}")
                fitness_scores.append(0)

        # Проверка условия остановки
        if best_fitness >= desired_consensus:
            print(f" Консенсус достигнут на итерации {iteration}")
            break

        # Селекция (элитная)
        new_population = []
        elite_size = max(1, population_size // 5)
        elite_indices = np.argsort(fitness_scores)[-elite_size:]

        for idx in elite_indices:
            new_population.append(population[idx])

        # Скрещивание (одноточечное)
        while len(new_population) < population_size:
            parent1 = random.choice(population)
            parent2 = random.choice(population)

            crossover_point = random.randint(1, num_experts - 1)
            child = np.concatenate([parent1[:crossover_point], parent2[crossover_point:]])

            # Нормализация
            child = child / np.sum(child)

            # Мутация
            if random.random() < 0.3:
                pos = random.randint(0, num_experts - 1)
                child[pos] = max(0.01, min(0.99, random.random()))
                child = child / np.sum(child)

            new_population.append(child)

        population = new_population

        if iteration % 10 == 0:
            print(f"Итерация {iteration}: best_fitness={best_fitness:.4f}")

    print(f"\n ФИНАЛЬНЫЕ РЕЗУЛЬТАТЫ ГА:")
    print(f"Лучшие веса: {[round(w, 4) for w in best_weights]}")
    print(f"Совместимости: {[round(c, 4) for c in best_compatibility]}")
    print(f"Лучшая совместимость: {best_fitness:.4f}")

    return best_weights, best_fitness, best_compatibility


def HHO_optimization(Judgments_m, Judgments_n, criteria,
                     desired_consensus=0.907, SearchAgents_no=10, Max_iter=100):
    print("\n Запуск алгоритма Harris Hawks Optimization (HHO)")
    print("=" * 50)

    num_experts = Judgments_m.shape[3]
    dim = num_experts  # количество весов равно количеству экспертов

    lb = np.full(dim, 0.01)
    ub = np.full(dim, 0.99)

    # Лучшие решения
    best_weights = np.ones(num_experts) / num_experts
    best_fitness = -1
    best_compatibility = []

    # Инициализация популяции
    X = np.zeros((SearchAgents_no, dim))
    for i in range(SearchAgents_no):
        weights = np.random.random(dim)
        X[i] = weights / np.sum(weights)  # нормализуем

    Rabbit_Location = best_weights.copy()
    Rabbit_Energy = float("inf")

    for t in range(Max_iter):
        # Вычисление fitness для каждого агента
        fitness_values = np.zeros(SearchAgents_no)

        for i in range(SearchAgents_no):
            # Нормализуем веса
            weights_norm = X[i] / np.sum(X[i])

            # Вычисляем совместимость
            compatibility = compat3(weights_norm, Judgments_m, Judgments_n)

            fitness = 1 - min(compatibility)  # минимизируем эту величину

            fitness_values[i] = fitness

            # Проверяем
            current_compatibility = min(compatibility)
            if current_compatibility >= best_fitness:
                best_fitness = current_compatibility
                best_weights = weights_norm.copy()
                best_compatibility = compatibility.copy()

                if t % 10 == 0:
                    print(f"Итерация {t + 1}: Новая лучшая совместимость = {best_fitness:.4f}")
                    print(f"  Веса: {[round(w, 4) for w in best_weights]}")

        # Обновление лучшего решения HHO
        min_idx = np.argmin(fitness_values)
        if fitness_values[min_idx] < Rabbit_Energy:
            Rabbit_Energy = fitness_values[min_idx]
            Rabbit_Location = X[min_idx].copy()

        # Проверка условия остановки
        if best_fitness >= desired_consensus:
            print(f"\n Консенсус достигнут на итерации {t + 1}")
            break

        E1 = 2 * (1 - t / Max_iter)  # Фактор уменьшения энергии

        # Обновление позиций
        for i in range(SearchAgents_no):
            E0 = 2 * np.random.random() - 1
            Escaping_Energy = E1 * E0

            # Фаза 1: Разведка (|E| ≥ 1)
            if abs(Escaping_Energy) >= 1:
                q = np.random.random()
                if q < 0.5:
                    # Случайное исследование
                    rand_idx = np.random.randint(0, SearchAgents_no)
                    X_rand = X[rand_idx]
                    X[i] = X_rand - np.random.random() * abs(X_rand - 2 * np.random.random() * X[i])
                else:
                    # Исследование относительно средней позиции
                    X[i] = (Rabbit_Location - X.mean(axis=0)) - np.random.random() * (ub - lb) * np.random.random() + lb

            # Фаза 2: Эксплуатация (|E| < 1)
            else:
                r = np.random.random()

                if r >= 0.5 and abs(Escaping_Energy) < 0.5:
                    X[i] = Rabbit_Location - Escaping_Energy * abs(Rabbit_Location - X[i])

                elif r >= 0.5:
                    Jump = 2 * (1 - np.random.random())
                    X[i] = Rabbit_Location + Jump * (X[i] - Rabbit_Location)

                else:
                    Jump = 2 * np.random.random()

                    # Первая стратегия
                    X1 = Rabbit_Location - Escaping_Energy * abs(Jump * Rabbit_Location - X[i])

                    # Проверяем, лучше ли X1
                    weights_norm1 = X1 / np.sum(X1)
                    try:
                        comp1 = compat3(weights_norm1, Judgments_m, Judgments_n)
                        fitness1 = 1 - min(comp1)
                    except:
                        fitness1 = float('inf')

                    if fitness1 < fitness_values[i]:
                        X[i] = X1
                    else:
                        # Вторая стратегия
                        X2 = Rabbit_Location - Escaping_Energy * abs(Jump * Rabbit_Location - X[i]) + np.random.randn(
                            dim) * 0.1

                        weights_norm2 = X2 / np.sum(X2)
                        try:
                            comp2 = compat3(weights_norm2, Judgments_m, Judgments_n)
                            fitness2 = 1 - min(comp2)
                        except:
                            fitness2 = float('inf')

                        if fitness2 < fitness_values[i]:
                            X[i] = X2

            # Проверка границ
            X[i] = np.clip(X[i], lb, ub)

            # Нормализация весов
            X[i] = X[i] / np.sum(X[i])

        if (t + 1) % 10 == 0:
            print(f"Итерация {t + 1}/{Max_iter}, Best Loss: {Rabbit_Energy:.4f}")

    print(f"\n ФИНАЛЬНЫЕ РЕЗУЛЬТАТЫ HHO:")
    print(f"Лучшие веса: {[round(w, 4) for w in best_weights]}")
    print(f"Совместимости: {[round(c, 4) for c in best_compatibility]}")
    print(f"Лучшая совместимость: {best_fitness:.4f}")

    return best_weights, best_fitness, best_compatibility


def generate_and_save_dhfs_json(output_file, n_criteria=6, n_experts=3):
    """
    Генерация тестовых данных DHFS с произвольным количеством критериев и экспертов
    """
    ifn = [
        [0.05, 0.95],
        [0.15, 0.80],
        [0.30, 0.60],
        [0.50, 0.50],
        [0.70, 0.20],
        [0.85, 0.10],
        [0.95, 0.05]
    ]

    roles = ["Project Manager", "Sustainability Manager", "Investment Director",
             "Technical Expert", "Financial Analyst", "Quality Assurance"]

    # Если экспертов больше, чем ролей, используем роли по кругу
    actual_roles = []
    for i in range(n_experts):
        actual_roles.append(roles[i % len(roles)])

    # Генерация имен критериев
    criteria_names = []
    for i in range(n_criteria):
        criteria_names.append(f"Criterion_{i+1}")

    # Инициализация данных
    data = {
        "problem_description": f"Generated DHFS data with {n_criteria} criteria and {n_experts} experts",
        "criteria": criteria_names,
        "dms": [],
        "parameters": {
            "desired_consensus": 1,
            "population_size": 10,
            "max_iterations": 200
        }
    }

    # Генерация данных для каждого эксперта
    for expert_idx in range(n_experts):
        comparisons = {}

        for i, crit_i in enumerate(criteria_names):
            comparisons[crit_i] = {}

            for j, crit_j in enumerate(criteria_names):
                if i == j:
                    # Диагональные элементы
                    comparisons[crit_i][crit_j] = {
                        "membership": [0.5],
                        "non_membership": [0.5]
                    }
                else:
                    # Случайные значения для недиагональных элементов
                    idx = np.random.choice([0, 1, 2, 4, 5, 6])
                    membership_val = ifn[idx][0]
                    nonmembership_val = ifn[idx][1]

                    # Ограничение значений
                    membership_val = max(0, min(1, membership_val))
                    nonmembership_val = max(0, min(1, nonmembership_val))

                    # Обеспечение непротиворечивости
                    if membership_val + nonmembership_val > 1:
                        total = membership_val + nonmembership_val
                        membership_val /= total
                        nonmembership_val /= total

                    comparisons[crit_i][crit_j] = {
                        "membership": [membership_val],
                        "non_membership": [nonmembership_val]
                    }

        data["dms"].append({
            "id": f"DM{expert_idx + 1}",
            "role": actual_roles[expert_idx],
            "pairwise_comparisons": comparisons
        })

    # Сохраняем
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

    print(f" Файл {output_file} создан")
    print(f" Параметры: {n_criteria} критериев, {n_experts} экспертов")
    return data


def main():
    print(" Запуск системы консенсуса DHF с двумя алгоритмами оптимизации")
    print("=" * 50)

    # Генерация тестовых данных
    n_criteria = 7
    n_experts = 5

    generate_and_save_dhfs_json("my_dhfs_data.json", n_criteria=n_criteria, n_experts=n_experts)

    # Загружаем данные
    data = load_input_data("my_dhfs_data.json")

    if not data:
        print("Не удалось загрузить данные")
        return

    # Преобразуем
    Judgments_m, Judgments_n, criteria = json_to_matrix_format(data)

    # Тестируем compat3 с равными весами
    print("\n Тестирование совместимости с равными весами:")
    num_experts = len(data['dms'])
    equal_weights = np.ones(num_experts) / num_experts
    test_compatibility = compat3(equal_weights, Judgments_m, Judgments_n)
    print(f"Совместимости: {[round(c, 4) for c in test_compatibility]}")
    print(f"Минимальная совместимость: {min(test_compatibility):.4f}")

    # Параметры из файла
    desired_consensus = data['parameters']['desired_consensus']

    # Запускаем GA
    print("\n" + "=" * 50)
    ga_best_weights, ga_best_fitness, ga_best_compatibility = genetic_algorithm_exact(
        Judgments_m, Judgments_n, criteria, desired_consensus,
        population_size=15, max_iterations=200
    )

    # Запускаем HHO
    print("\n" + "=" * 50)
    hho_best_weights, hho_best_fitness, hho_best_compatibility = HHO_optimization(
        Judgments_m, Judgments_n, criteria, desired_consensus,
        SearchAgents_no=15, Max_iter=200
    )

    # Формируем результаты
    dm_ids = [dm['id'] for dm in data['dms']]
    results = {
        "algorithm_comparison": {
            "GA": {
                "final_consensus_level": round(ga_best_fitness, 4),
                "compatibility_scores": {dm_ids[i]: round(ga_best_compatibility[i], 4) for i in range(num_experts)},
                "dm_weights": {dm_ids[i]: round(ga_best_weights[i], 4) for i in range(num_experts)},
            },
            "HHO": {
                "final_consensus_level": round(hho_best_fitness, 4),
                "compatibility_scores": {dm_ids[i]: round(hho_best_compatibility[i], 4) for i in range(num_experts)},
                "dm_weights": {dm_ids[i]: round(hho_best_weights[i], 4) for i in range(num_experts)},
            }
        },
        "best_algorithm": "GA" if ga_best_fitness >= hho_best_fitness else "HHO",
        "best_overall_consensus": round(max(ga_best_fitness, hho_best_fitness), 4),
        "parameters_used": {
            "desired_consensus": desired_consensus,
            "criteria_count": len(criteria),
            "experts_count": len(dm_ids)
        }
    }

    print("\n" + "=" * 50)
    print(" СРАВНЕНИЕ АЛГОРИТМОВ:")
    print("=" * 50)
    print(f"Генетический алгоритм (GA):")
    print(f"  Уровень консенсуса: {results['algorithm_comparison']['GA']['final_consensus_level']}")
    print(f"  Веса экспертов: {results['algorithm_comparison']['GA']['dm_weights']}")
    print(f"\nАлгоритм Harris Hawks (HHO):")
    print(f"  Уровень консенсуса: {results['algorithm_comparison']['HHO']['final_consensus_level']}")
    print(f"  Веса экспертов: {results['algorithm_comparison']['HHO']['dm_weights']}")
    print(f"\nЛучший алгоритм: {results['best_algorithm']}")
    print(f"Лучший общий консенсус: {results['best_overall_consensus']}")

    save_results_to_json(results, "comparison_results.json")


if __name__ == "__main__":
    main()