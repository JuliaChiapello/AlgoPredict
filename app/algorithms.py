"""
Este módulo contiene la implementación completa de los algoritmos de
ordenamiento y búsqueda utilizados por el proyecto AlgoPredict.

Su objetivo principal es:
- Ejecutar algoritmos clásicos (iterativos y recursivos)
- Medir tiempos reales de ejecución de forma reproducible
- Generar datasets reales y teóricos para entrenamiento y predicción

Este archivo NO contiene lógica de Machine Learning ni de interfaz.
Se enfoca exclusivamente en:
- algoritmos
- generación de datos
- medición de tiempos
"""

from time import perf_counter
import numpy as np
import sys
from tqdm import tqdm
import itertools
from multiprocessing import Pool, cpu_count

# Se aumenta el límite de recursión para permitir
# ejecuciones recursivas con tamaños de entrada grandes
sys.setrecursionlimit(100000)

# ==================================================
# ALGORITMOS DE ORDENAMIENTO
# ==================================================

def quicksort_recursive(arr):
    """
    Implementación recursiva de QuickSort.

    - Divide el arreglo utilizando un pivote
    - Ordena recursivamente las particiones
    - Retorna una nueva lista ordenada (no in-place)

    Complejidad:
    - Promedio: O(n log n)
    - Peor caso: O(n²)
    """
    if len(arr) <= 1:
        return arr[:]
    pivot = arr[len(arr)//2]
    left = [x for x in arr if x < pivot]
    mid = [x for x in arr if x == pivot]
    right = [x for x in arr if x > pivot]
    return quicksort_recursive(left) + mid + quicksort_recursive(right)

def quicksort_iterative(arr):
    """
    Implementación iterativa de QuickSort.

    - Utiliza una pila manual en lugar de recursión
    - Ordena el arreglo in-place
    - Evita problemas de profundidad de recursión

    Complejidad:
    - Promedio: O(n log n)
    - Peor caso: O(n²)
    """
    arr = list(arr)
    stack = [(0, len(arr)-1)]
    while stack:
        low, high = stack.pop()
        if low >= high:
            continue
        pivot = arr[high]
        i = low
        for j in range(low, high):
            if arr[j] < pivot:
                arr[i], arr[j] = arr[j], arr[i]
                i += 1
        arr[i], arr[high] = arr[high], arr[i]
        stack.append((low, i-1))
        stack.append((i+1, high))
    return arr

def mergesort_recursive(arr):
    """
    Implementación recursiva de MergeSort.

    - Divide el arreglo en mitades
    - Ordena cada mitad recursivamente
    - Combina las mitades ordenadas

    Complejidad:
    - Siempre O(n log n)
    """
    if len(arr) <= 1:
        return arr[:]
    mid = len(arr)//2
    left = mergesort_recursive(arr[:mid])
    right = mergesort_recursive(arr[mid:])
    return merge(left, right)

def merge(a, b):
    """
    Función auxiliar de MergeSort.

    - Combina dos listas ya ordenadas
    - Devuelve una nueva lista ordenada
    """
    res = []
    i = j = 0
    while i < len(a) and j < len(b):
        if a[i] <= b[j]:
            res.append(a[i]); i += 1
        else:
            res.append(b[j]); j += 1
    res.extend(a[i:])
    res.extend(b[j:])
    return res

def mergesort_iterative(arr):
    """
    Implementación iterativa de MergeSort (bottom-up).

    - Evita recursión
    - Utiliza bloques crecientes de tamaño 2^k

    Complejidad:
    - Siempre O(n log n)
    """
    width = 1
    n = len(arr)
    arr = list(arr)
    temp = [0]*n
    while width < n:
        for i in range(0, n, 2*width):
            l, r = i, min(i+width, n)
            e = min(i+2*width, n)
            temp[l:e] = merge(arr[l:r], arr[r:e])
        arr, temp = temp, arr
        width *= 2
    return arr

def insertionsort_iterative(arr):
    """
    Implementación iterativa de Insertion Sort.

    - Eficiente para listas pequeñas o casi ordenadas
    - Ordena in-place

    Complejidad:
    - Mejor caso: O(n)
    - Peor caso: O(n²)
    """
    arr = list(arr)
    for i in range(1, len(arr)):
        key = arr[i]
        j = i - 1
        while j >= 0 and arr[j] > key:
            arr[j+1] = arr[j]
            j -= 1
        arr[j+1] = key
    return arr

def bubblesort_iterativo(arr):
    """
    Implementación iterativa de Bubble Sort.

    - Algoritmo simple
    - Útil con fines educativos

    Complejidad:
    - O(n²)
    """
    arr = list(arr)
    n = len(arr)
    for i in range(n):
        for j in range(0, n-i-1):
            if arr[j] > arr[j+1]:
                arr[j], arr[j+1] = arr[j+1], arr[j]
    return arr

def selectionsort_iterative(arr):
    """
    Implementación iterativa de Selection Sort.

    - Selecciona el mínimo y lo coloca en su posición
    - Ordena in-place

    Complejidad:
    - O(n²)
    """
    arr = list(arr)
    for i in range(len(arr)):
        m = i
        for j in range(i+1, len(arr)):
            if arr[j] < arr[m]:
                m = j
        arr[i], arr[m] = arr[m], arr[i]
    return arr

# ==================================================
# ALGORITMOS DE BÚSQUEDA
# ==================================================

def binarysearch_iterative(arr, target):
    """
    Implementación iterativa de búsqueda binaria.

    Requiere:
    - arreglo previamente ordenado

    Complejidad:
    - O(log n)
    """
    l, r = 0, len(arr)-1
    while l <= r:
        m = (l+r)//2
        if arr[m] == target:
            return m
        elif arr[m] < target:
            l = m + 1
        else:
            r = m - 1
    return -1

def binarysearch_recursive(arr, target, l=0, r=None):
    """
    Implementación recursiva de búsqueda binaria.
    """
    if r is None:
        r = len(arr)-1
    if l > r:
        return -1
    m = (l+r)//2
    if arr[m] == target:
        return m
    elif arr[m] > target:
        return binarysearch_recursive(arr, target, l, m-1)
    else:
        return binarysearch_recursive(arr, target, m+1, r)

def sequentialsearch_iterative(arr, target):
    """
    Implementación iterativa de búsqueda secuencial.

    Complejidad:
    - O(n)
    """
    for i, v in enumerate(arr):
        if v == target:
            return i
    return -1

def sequentialsearch_recursive(arr, target, i=0):
    """
    Implementación recursiva de búsqueda secuencial.
    """
    if i >= len(arr):
        return -1
    if arr[i] == target:
        return i
    return sequentialsearch_recursive(arr, target, i+1)

# ==================================================
# MAPA CENTRAL DE ALGORITMOS
# ==================================================

"""
Mapa que asocia nombres de algoritmos con sus funciones reales.

Este diccionario es clave para:
- generación automática de datasets
- ejecución dinámica desde la UI
- entrenamiento del modelo
"""

ALGORITHMS_MAP = {
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
    "RecursiveSequentialSearch": lambda arr: sequentialsearch_recursive(arr, arr[len(arr)//2]),
}

# ==================================================
# GENERACIÓN DE DATOS
# ==================================================

def generateData(n, dataType, sorted_, seed):
    """
    Genera datos de entrada reproducibles para los algoritmos.

    Parámetros:
    - n: tamaño del arreglo
    - dataType: int o float
    - sorted_: indica si el arreglo está ordenado
    - seed: semilla para reproducibilidad

    Retorna:
    - lista de valores lista para ser procesada
    """
    rng = np.random.default_rng(seed)
    if dataType == "int":
        arr = rng.integers(-n, n, n).tolist()
    else:
        arr = rng.random(n).tolist()
    if sorted_:
        arr.sort()
    else:
        rng.shuffle(arr)
    return arr

# ==================================================
# MEDICIÓN DE TIEMPOS REALES (PARALELIZADA)
# ==================================================

def _measure_real(args):
    """
    Worker interno utilizado por multiprocessing.

    Ejecuta un algoritmo varias veces y devuelve
    el tiempo mediano de ejecución (en ms).
    """
    algo, n, dataType, sorted_, reps = args
    func = ALGORITHMS_MAP[algo]
    times = []
    for r in range(reps):
        data = generateData(n, dataType, sorted_, seed=(hash((algo,n,r)) & 0xffffffff))
        # BinarySearch: internally sorts if necessary
        if "BinarySearch" in algo and not sorted_:
            data = sorted(data)
        start = perf_counter()
        func(list(data))
        end = perf_counter()
        times.append((end - start) * 1000)

    return {
        "algorithm": algo,
        "numElements": n,
        "dataType": dataType,
        "sorted": sorted_,
        "timePredicted": round(float(np.median(times)), 4),
        "type": "real"
    }

def generateDeterministicPredictions(parallel_workers=None, reps=3):
    """
    Genera el dataset completo de tiempos reales.

    - Usa multiprocessing
    - Ejecuta combinaciones rectangulares de parámetros
    - Devuelve una lista lista para insertar en MongoDB
    """
    if parallel_workers is None:
        parallel_workers = max(1, cpu_count()-1)
    sizes = range(50, 2501, 50)
    combos = list(itertools.product(
        ALGORITHMS_MAP.keys(),
        sizes,
        ["int", "float"],
        [True, False]
    ))
    args = [(a,n,dt,s,reps) for a,n,dt,s in combos]
    deterministicPredictions = []
    with Pool(parallel_workers) as pool:
        for r in tqdm(pool.imap_unordered(_measure_real, args), total=len(args)):
            deterministicPredictions.append(r)
    return deterministicPredictions

# ==================================================
# GENERACIÓN DE TIEMPOS TEÓRICOS
# ==================================================

def generateTheoreticalPredictions(real_data=None):
    """
    Genera tiempos teóricos basados en complejidad algorítmica.

    - O(n)
    - O(n²)
    - O(n log n)

    Estos datos se utilizan para:
    - entrenamiento del modelo teórico
    - comparación con tiempos reales
    """
    sizes = np.unique(np.logspace(np.log10(2500), np.log10(10_000_000), 3000).astype(int))
    theoreticalPredictions = []
    for algo, n, dt, s in itertools.product(
        ALGORITHMS_MAP.keys(),
        sizes,
        ["int", "float"],
        [True, False]
    ):
        if "Search" in algo:
            t = n * 1e-5
        elif "Insertion" in algo or "Bubble" in algo or "Selection" in algo:
            t = n*n * 1e-8
        else:
            t = n*np.log2(n) * 1e-6
        theoreticalPredictions.append({
            "algorithm": algo,
            "numElements": int(n),
            "dataType": dt,
            "sorted": s,
            "timePredicted": round(float(t), 6),
            "type": "theoretical"
        })
    return theoreticalPredictions
