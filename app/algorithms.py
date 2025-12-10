from time import perf_counter
import numpy as np
from tqdm import tqdm
import sys
import itertools
import random
from collections import defaultdict
import statistics

sys.setrecursionlimit(100000) #Aumentar el numero de recursiones en Python

# ALGORITMOS DE ORDENACION (ITERATIVOS / RECURSIVOS) 

# QuickSort Recursivo
def quicksort_recursive(arr):
    if len(arr) <= 1:
        return arr
    pivot = arr[len(arr) // 2]
    left = [x for x in arr if x < pivot]
    center = [x for x in arr if x == pivot]
    right = [x for x in arr if x > pivot]
    return quicksort_recursive(left) + center + quicksort_recursive(right)

# QuickSort Iterativo
def quicksort_iterative(arr):
    stack = [(0, len(arr) - 1)]
    while stack:
        start, end = stack.pop()
        if start >= end:
            continue
        pivot = arr[end]
        i = start
        for j in range(start, end):
            if arr[j] < pivot:
                arr[i], arr[j] = arr[j], arr[i]
                i += 1
        arr[i], arr[end] = arr[end], arr[i]
        stack.append((start, i - 1))
        stack.append((i + 1, end))
    return arr

# MergeSort Recursivo
def mergesort_recursive(arr):
    if len(arr) <= 1:
        return arr
    middle = len(arr) // 2
    left = mergesort_recursive(arr[:middle])
    right = mergesort_recursive(arr[middle:])
    return merge(left, right)

def merge(izq, der):
    result = []
    i = j = 0
    while i < len(izq) and j < len(der):
        if izq[i] < der[j]:
            result.append(izq[i])
            i += 1
        else:
            result.append(der[j])
            j += 1
    result.extend(izq[i:])
    result.extend(der[j:])
    return result

# MergeSort Iterativo
def mergesort_iterative(arr):
    width = 1
    n = len(arr)
    while width < n:
        for i in range(0, n, 2 * width):
            left = arr[i:i + width]
            right = arr[i + width:i + 2 * width]
            arr[i:i + 2 * width] = merge(left, right)
        width *= 2
    return arr

# InsertionSort Iterativo
def insertionsort_iterative(arr):
    for i in range(1, len(arr)):
        current = arr[i]
        j = i - 1
        while j >= 0 and arr[j] > current:
            arr[j + 1] = arr[j]
            j -= 1
        arr[j + 1] = current
    return arr

# BubbleSort Iterativo
def bubblesort_iterativo(arr):
    n = len(arr)
    for i in range(n):
        for j in range(0, n - i - 1):
            if arr[j] > arr[j + 1]:
                arr[j], arr[j + 1] = arr[j + 1], arr[j]
    return arr

# SelectionSort Iterativo
def selectionsort_iterative(arr):
    for i in range(len(arr)):
        min_idx = i
        for j in range(i+1, len(arr)):
            if arr[j] < arr[min_idx]:
                min_idx = j
        arr[i], arr[min_idx] = arr[min_idx], arr[i]
    return arr

#------------------------------------------------------------------------------------------------------------
# ALGORITMOS DE BUSQUEDA (ITERATIVOS / RECURSIVOS)

# BinarySearch Recursivo
def binarysearch_recursive(arr, target, low=0, high=None):
    if high is None:
        high = len(arr) - 1
    if low > high:
        return -1
    middle = (low + high) // 2
    if arr[middle] == target:
        return middle
    elif arr[middle] > target:
        return binarysearch_recursive(arr, target, low, middle - 1)
    else:
        return binarysearch_recursive(arr, target, middle + 1, high)

# BinarySearch Iterativo
def binarysearch_iterative(arr, target):
    low = 0
    high = len(arr) - 1
    while low <= high:
        middle = (low + high) // 2
        if arr[middle] == target:
            return middle
        elif arr[middle] < target:
            low = middle + 1
        else:
            high = middle - 1
    return -1

# SequentialSearch Iterativo
def sequentialsearch_iterative(arr, target):
    for i in range(len(arr)):
        if arr[i] == target:
            return i
    return -1

# SequentialSearch Recursivo
def sequentialsearch_recursive(arr, target, i=0):
    if i >= len(arr):
        return -1
    if arr[i] == target:
        return i
    return sequentialsearch_recursive(arr, target, i + 1)

# ---------------------- FUNCIONES AUXILIARES ---------------------------------

# Función para generar datos aleatorios. Devuelve una lista de datos (int o float) ordenada o desordenada
def generateData(size, dataType, sorted_):
    if dataType == "int":
        data = list(range(-size // 2, size // 2))
    else:  # float
        data = [float(i) + 0.5 for i in range(-size // 2, size // 2)]
    if not sorted_:
        np.random.shuffle(data)
    return data


# --- Función para estimar tiempo teórico ---
def predict_theoretical_time(size, model_type="nlogn", a=1e-6, b=0):
        if model_type == "nlogn":
            return a * size * np.log2(size) + b
        elif model_type == "n2":
            return a * size**2 + b
        elif model_type == "n":
            return a * size + b
        elif model_type == "logn":
            return a * np.log2(size) + b
        

# Ajustar parámetros a partir de datos reales
def fit_params(sizes, times, complexity):
    """
    sizes: array-like de tamaños (n)
    times: array-like de tiempos medidos (ms)
    complexity: "nlogn", "n2", "n", "logn"
    Devuelve a,b en time = a * f(n) + b
    """
    if complexity == "nlogn":
        x = np.array(sizes) * np.log2(sizes)
    elif complexity == "n2":
        x = np.array(sizes)**2
    elif complexity == "n":
        x = np.array(sizes)
    elif complexity == "logn":
        x = np.log2(sizes)
    else:
        raise ValueError("Complejidad no soportada")

    y = np.array(times)
    # Ajustar lineal: y = a*x + b
    a, b = np.polyfit(x, y, 1)
    return a, b


# Función para calcular parámetros (a,b) para cada algoritmo
def calculate_algorithm_params(real_data):
    """
    real_data: lista de dicts con keys: algorithm, numElements, dataType, sorted, timePredicted
    Devuelve dict: {algorithm: (complejidad, a, b)}
    Se calcula tomando solo datos con sorted=True y dataType="float" para ajuste (se puedes cambiar si se quiere)
    """
    algorithm_complexities = {
        "IterativeQuickSort": "nlogn",
        "RecursiveQuickSort": "nlogn",
        "IterativeMergeSort": "nlogn",
        "RecursiveMergeSort": "nlogn",
        "IterativeInsertionSort": "n2",
        "IterativeBubbleSort": "n2",
        "IterativeSelectionSort": "n2",
        "IterativeBinarySearch": "logn",
        "RecursiveBinarySearch": "logn",
        "IterativeSequentialSearch": "n",
        "RecursiveSequentialSearch": "n",
    }

    grouped_data = defaultdict(lambda: {"sizes": [], "times": []})

    for entry in real_data:
        algo = entry["algorithm"]
        # Para mejor ajuste, filtrar dataType float y sorted True
        if True:
            grouped_data[algo]["sizes"].append(entry["numElements"])
            grouped_data[algo]["times"].append(entry["timePredicted"])

    params = {}

    for algo, data in grouped_data.items():
        complexity = algorithm_complexities.get(algo)
        if not complexity:
            continue
        if len(data["sizes"]) < 2:
            # No hay suficientes datos para ajustar
            params[algo] = (complexity, 1e-6, 0)
            continue
        a, b = fit_params(data["sizes"], data["times"], complexity)
        params[algo] = (complexity, a, b)

    return params


# ---------------------------- Generador de predicciones teóricas para tamaños de entrada grandes ----------------------------

def generateTheoreticalPredictions(real_data = None):
    # Si  no hay tiempos reales, toma por defecto los valores de tiempos de ejecucion por defecto (estimados)
    if real_data is None:
        # Asociación de algoritmos con su complejidad y parámetros (estimados)
        algorithm_complexities = {
            "IterativeQuickSort": ("nlogn", 1.2e-6, 0.5),
            "RecursiveQuickSort": ("nlogn", 1.3e-6, 0.4),
            "MergeSortIterative": ("nlogn", 1.1e-6, 0.3),
            "RecursiveMergeSort": ("nlogn", 1.15e-6, 0.35),
            "IterativeInsertionSort": ("n2", 3e-8, 0.01),
            "IterativeBubbleSort": ("n2", 2.5e-8, 0.01),
            "IterativeSelectionSort": ("n2", 2.8e-8, 0.01),
            "IterativeBinarySearch": ("logn", 0.02, 0.01),
            "RecursiveBinarySearch": ("logn", 0.02, 0.01),
            "IterativeSequentialSearch": ("n", 5e-5, 0.1),
            "RecursiveSequentialSearch": ("n", 5e-5, 0.1),
        }
    else:
        algorithm_complexities = calculate_algorithm_params(real_data)    

    theoretical_size = sorted(set(
            list(range(2050, 5000, 5)) +
            list(range(5000, 20000, 25)) +
            list(range(20000, 100001, 50)) +
            random.sample(range(100001, 1000001, 200), 1200)  # 300 tamaños aleatorios sin repeticion
    ))
    
    dataTypes = ["int", "float"]
    sorted_options = [True, False]

    theoreticalPredictions = []

    print(f"Generando tiempos teóricos para tamaños grandes...")
    for name, (model_type, a, b) in algorithm_complexities.items():
        for theor_size, dataType, sorted_ in tqdm(itertools.product(theoretical_size, dataTypes, sorted_options)):
            time_ms = predict_theoretical_time(theor_size, model_type, a, b)
            theoreticalPredictions.append({
                "algorithm": name,
                "numElements": theor_size,
                "dataType": dataType,
                "sorted": sorted_,
                "timePredicted": round(time_ms, 4),
                "type": "theoretical"
            })

    return theoreticalPredictions


# ---------------------------- Generador de predicciones reales para tamaños de entrada pequeños ----------------------------


def generateDeterministicPredictions():

    algorithms = {
        "IterativeQuickSort": quicksort_iterative,
        "RecursiveQuickSort": quicksort_recursive,
        "IterativeMergeSort": mergesort_iterative,
        "RecursiveMergeSort": mergesort_recursive,
        "IterativeInsertionSort": insertionsort_iterative,
        "IterativeBubbleSort": bubblesort_iterativo,
        "IterativeSelectionSort": selectionsort_iterative,
        "IterativeBinarySearch": lambda arr: binarysearch_iterative(arr, arr[len(arr)//2]),
        "RecursiveBinarySearch": lambda arr: binarysearch_recursive(arr, arr[len(arr)//2]),
        "IterativeSequentialSearch": lambda arr: sequentialsearch_iterative(arr, arr[len(arr)//2]),
        "RecursiveSequentialSearch": lambda arr: sequentialsearch_recursive(arr, arr[len(arr)//2])
    }
    
        
    deterministic_size = list(range(50, 2001, 50))  # 2000 tamaños

    dataTypes = ["int", "float"]
    sorted_options = [True, False]

    deterministicPredictions = []

    total_combinations = len(algorithms) * len(deterministic_size) * len(dataTypes) * len(sorted_options) * 10
    print(f"Generando {total_combinations} ejecuciones...")

    for name, function in algorithms.items():
        print(f"Ejecutando {name}...")
        isBinarySearch = "BinarySearch" in name

        for det_size, dataType, sorted_ in tqdm(itertools.product(deterministic_size, dataTypes, sorted_options)):
            seed = hash((name, det_size, dataType, sorted_)) % (2**32)
            time = []
            for rep in range(10):
                np.random.seed(seed)
                data = generateData(det_size, dataType, sorted_)
                input_data = data.copy()

                if isBinarySearch:
                    input_data.sort()  # por seguridad

                try:
                    start = perf_counter()
                    function(input_data)
                    end = perf_counter()
                    time_ms = (end - start) * 1000
                    time.append(time_ms)
                except Exception as e:
                    print(f"Error con {name} - size={det_size}, type={dataType}, sorted={sorted_}: {e}")
                    continue

            # Si se lograron las 10 ejecuciones, eliminamos min y max    
            if len(time) >= 3:
                time = sorted(time)[1:-1]  # eliminamos el menor y el mayor
                average = round(statistics.median(time), 4)

                deterministicPredictions.append({
                    "algorithm": name,
                    "numElements": det_size,
                    "dataType": dataType,
                    "sorted": sorted_,
                    "timePredicted": average,
                    "type": "real"
                })

    return deterministicPredictions




