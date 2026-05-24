# PyTorch Lightning Backport for Python 3.8

[![Python 3.8](https://img.shields.io/badge/Python-3.8-blue.svg)](https://www.python.org/downloads/release/python-3810/)
[![PyTorch 2.13](https://img.shields.io/badge/PyTorch-2.13-orange.svg)](https://pytorch.org/)
[![Lightning 2.6.2](https://img.shields.io/badge/Lightning-2.6.2-792ee5.svg)](https://lightning.ai/)
[![Test Pass Rate](https://img.shields.io/badge/Tests-29%2F29%20(100%25)-brightgreen.svg)](./tests/test_results.json)
[![License](https://img.shields.io/badge/License-Apache%202.0-green.svg)](./LICENSE)

**[Read this in Russian / Читать на русском](#русский)**

---

## What Is This?

This is a **backported version of PyTorch Lightning 2.6.2** that runs on **Python 3.8**, specifically designed to work with a custom-compiled **PyTorch 2.13** on legacy systems (including Windows 7).

The official PyTorch Lightning 2.6.2 requires Python >= 3.10, which excludes users who are stuck on Python 3.8 due to OS limitations (e.g., Windows 7 only supports Python <= 3.8). This project bridges that gap by backporting all Python 3.10+ syntax and API usage to be compatible with Python 3.8.

## Why Do We Need This?

- **Windows 7 users** cannot upgrade beyond Python 3.8
- **Legacy environments** in production may be locked to Python 3.8
- **Custom PyTorch builds** for older hardware often target Python 3.8
- The latest PyTorch Lightning features (mixed precision, Fabric, etc.) should be accessible to all users

## What We Modified / Fixed

### PyTorch Lightning Source Code Changes

| # | Change | Files Affected | Reason |
|---|--------|---------------|--------|
| 1 | `python_requires >= 3.10` → `>= 3.8` | 3 `__setup__.py` files | Allow installation on Python 3.8 |
| 2 | `dict[X, Y]` → `Dict[X, Y]` | Hundreds of files | Python 3.8 does not support subscripting built-in types at runtime |
| 3 | `list[X]` → `List[X]` | Hundreds of files | Same as above |
| 4 | `X \| Y` → `Union[X, Y]` | Multiple files | Python 3.8 does not support `\|` union syntax outside annotations |
| 5 | `entry_points(group=...)` → `entry_points().get(group, [])` | `registry.py` (2 files) | Python 3.8 `importlib.metadata.entry_points()` does not accept `group` kwarg |
| 6 | `Iterator[T]` → `Iterator` in class bases | `combined_loader.py` | `ABCMeta` objects are not subscriptable in Python 3.8 |
| 7 | `rank_zero_experiment` import fix | `logger.py` (2 files) | Import statement was corrupted during automated replacement |
| 8 | `mode_Dict` → `mode_dict` | 36 files | Regex replacement accidentally capitalized variable names |

### PyTorch Source Code Changes (for the companion PyTorch 2.13 backport)

| # | Change | Files Affected | Reason |
|---|--------|---------------|--------|
| 1 | `cast(Sequence[int], x)` → `x` | 10 files | `collections.abc` types are not subscriptable at runtime in Python 3.8 |
| 2 | `from typing import (, cast` → `from typing import (` | 7 files | Automated `cast` import insertion created syntax errors |
| 3 | `zip(..., strict=True)` → `zip(...)` | `dataset.py` | `strict` parameter is Python 3.10+ only |
| 4 | Missing `cast` import added | Multiple files | `cast` was used without being imported |

## Key Features

All features from PyTorch Lightning 2.6.2 are preserved and working:

- **Trainer API** — Full training, validation, testing, and prediction loops
- **LightningModule** — Complete step methods, hooks, and optimizer configuration
- **LightningDataModule** — Data setup and loader management
- **Callbacks** — EarlyStopping, ModelCheckpoint, LearningRateMonitor, DeviceStatsMonitor
- **Loggers** — CSVLogger (TensorBoardLogger requires tensorboard package)
- **Mixed Precision Training** — FP16-mixed and BF16-mixed
- **Double Precision** — 64-true precision mode
- **Gradient Accumulation** — `accumulate_grad_batches` support
- **Model Checkpointing** — Save and load checkpoints
- **Lightning Fabric** — Lower-level API for custom training loops
- **Seed Reproducibility** — `seed_everything()` works correctly
- **GPU Acceleration** — CUDA support with automatic device detection

## Test Results

All 29 tests passed with a **100% pass rate** on the following environment:

- **OS:** Windows (Win7 compatible)
- **Python:** 3.8.10
- **PyTorch:** 2.13.0a0+git0607d0e (custom compiled)
- **GPU:** NVIDIA GeForce RTX 3070 Ti 8.0GB
- **CUDA:** 11.3

### Test Summary

| Category | Tests | Passed | Failed | Pass Rate |
|----------|-------|--------|--------|-----------|
| Core API | 7 | 7 | 0 | 100% |
| Training & Inference | 4 | 4 | 0 | 100% |
| Performance Metrics | 3 | 3 | 0 | 100% |
| Special Scenarios | 7 | 7 | 0 | 100% |
| Compatibility | 8 | 8 | 0 | 100% |
| **Total** | **29** | **29** | **0** | **100%** |

### Performance Highlights

| Metric | CPU | GPU | GPU/CPU Ratio |
|--------|-----|-----|---------------|
| Training Speed (samples/s) | 4,653.2 | 29,359.4 | **6.31x** |
| Epoch Time (s) | 8.60 | 1.36 | 6.31x faster |
| Peak GPU Memory | — | 21.6 MB | — |

| Special Scenario | Result |
|-----------------|--------|
| FP16 Mixed Precision | Passed (0.08s) |
| BF16 Mixed Precision | Passed (0.07s) |
| Double Precision (64-true) | Passed |
| Gradient Accumulation | Passed |
| Early Stopping | Passed (stopped at epoch 36/50) |
| Device Stats Monitor | Passed |
| Lightning Fabric | Passed |

### Known Limitations

1. **cuDNN not enabled** — May affect CNN training speed
2. **TensorBoardLogger not tested** — `tensorboard` package not installed due to disk space
3. **Distributed training not tested** — Single GPU environment only
4. **Static type checking** — Some `cast()` calls were removed, which may affect mypy/pyright inference

Full test results: [`tests/test_results.json`](./tests/test_results.json)
Full test report: [`tests/test_report.txt`](./tests/test_report.txt)

## How to Build from Source

### Prerequisites

- Python 3.8.x (tested with 3.8.10)
- A compatible PyTorch 2.13 build for Python 3.8 (see our companion repo)
- Git

### Build Steps

```bash
# 1. Clone this repository
git clone https://github.com/Lanurence666/pytorch-lightning-backport-py38.git
cd pytorch-lightning-backport-py38

# 2. Install build dependencies
pip install setuptools wheel build

# 3. Build the wheel
python -m build

# The wheel will be in the dist/ directory
```

### Install from Source (Editable Mode)

```bash
pip install -e .
```

### Install the Pre-built Wheel

Download the wheel from the [Releases page](https://github.com/Lanurence666/pytorch-lightning-backport-py38/releases) and install:

```bash
pip install pytorch_lightning-2.6.2-py3-none-any.whl
```

## Running the Tests

```bash
# Run the comprehensive test suite
python tests/comprehensive_test.py

# The results will be saved to:
#   - pl_test_results.json (machine-readable)
#   - Console output (human-readable summary)
```

## Project Structure

```
pytorch-lightning-backport-py38/
├── src/
│   ├── lightning/           # Top-level lightning package
│   │   ├── fabric/          # Lightning Fabric
│   │   ├── pytorch/         # PyTorch Lightning
│   │   ├── __setup__.py     # Package setup (python_requires >= 3.8)
│   │   └── ...
│   ├── lightning_fabric/    # Fabric standalone package
│   │   ├── __setup__.py     # (python_requires >= 3.8)
│   │   └── ...
│   └── pytorch_lightning/   # PL standalone package
│       ├── __setup__.py     # (python_requires >= 3.8)
│       └── ...
├── tests/
│   ├── comprehensive_test.py   # Full test suite
│   ├── test_results.json       # Test results (JSON)
│   └── test_report.txt         # Detailed test report
├── setup.py
├── pyproject.toml
├── README.md                    # This file
└── LICENSE
```

## Companion Project

This backport is designed to work with our custom-compiled **PyTorch 2.13 for Python 3.8**, which is available in a companion repository. The PyTorch build also required several Python 3.8 compatibility fixes.

## Acknowledgments

- [PyTorch Lightning Team](https://lightning.ai/) for the excellent framework
- This is a community backport and is not officially affiliated with or endorsed by Lightning AI

## License

This project is licensed under the Apache License 2.0 — see the [LICENSE](./LICENSE) file for details.

---

<a id="русский"></a>

# PyTorch Lightning: бэкпорт для Python 3.8

[![Python 3.8](https://img.shields.io/badge/Python-3.8-blue.svg)](https://www.python.org/downloads/release/python-3810/)
[![PyTorch 2.13](https://img.shields.io/badge/PyTorch-2.13-orange.svg)](https://pytorch.org/)
[![Lightning 2.6.2](https://img.shields.io/badge/Lightning-2.6.2-792ee5.svg)](https://lightning.ai/)
[![Тесты пройдены](https://img.shields.io/badge/Тесты-29%2F29%20(100%25)-brightgreen.svg)](./tests/test_results.json)

## Что это?

Это **бэкпорт-версия PyTorch Lightning 2.6.2**, работающая на **Python 3.8**, специально разработанная для совместимости с пользовательской сборкой **PyTorch 2.13** на устаревших системах (включая Windows 7).

Официальный PyTorch Lightning 2.6.2 требует Python >= 3.10, что исключает пользователей, ограниченных Python 3.8 из-за ограничений ОС (например, Windows 7 поддерживает только Python <= 3.8). Этот проект устраняет этот разрыв, адаптируя весь синтаксис Python 3.10+ и использование API для совместимости с Python 3.8.

## Зачем это нужно?

- **Пользователи Windows 7** не могут обновиться выше Python 3.8
- **Устаревшие среды** в продакшене могут быть привязаны к Python 3.8
- **Пользовательские сборки PyTorch** для старого оборудования часто ориентированы на Python 3.8
- Новейшие функции PyTorch Lightning (смешанная точность, Fabric и др.) должны быть доступны всем пользователям

## Что мы изменили / исправили

### Изменения в исходном коде PyTorch Lightning

| № | Изменение | Затронутые файлы | Причина |
|---|-----------|-----------------|---------|
| 1 | `python_requires >= 3.10` → `>= 3.8` | 3 файла `__setup__.py` | Разрешить установку на Python 3.8 |
| 2 | `dict[X, Y]` → `Dict[X, Y]` | Сотни файлов | Python 3.8 не поддерживает индексацию встроенных типов во время выполнения |
| 3 | `list[X]` → `List[X]` | Сотни файлов | То же, что выше |
| 4 | `X \| Y` → `Union[X, Y]` | Несколько файлов | Python 3.8 не поддерживает синтаксис объединения `\|` вне аннотаций |
| 5 | `entry_points(group=...)` → `entry_points().get(group, [])` | `registry.py` (2 файла) | Python 3.8 не принимает `group` как именованный аргумент |
| 6 | `Iterator[T]` → `Iterator` в базах классов | `combined_loader.py` | Объекты `ABCMeta` не индексируются в Python 3.8 |
| 7 | Исправление импорта `rank_zero_experiment` | `logger.py` (2 файла) | Оператор импорта был повреждён при автоматической замене |
| 8 | `mode_Dict` → `mode_dict` | 36 файлов | Регулярные выражения случайно заменили имена переменных |

### Изменения в исходном коде PyTorch (для сопутствующего бэкпорта PyTorch 2.13)

| № | Изменение | Затронутые файлы | Причина |
|---|-----------|-----------------|---------|
| 1 | `cast(Sequence[int], x)` → `x` | 10 файлов | Типы `collections.abc` не индексируются во время выполнения в Python 3.8 |
| 2 | `from typing import (, cast` → `from typing import (` | 7 файлов | Автоматическая вставка импорта `cast` создала синтаксические ошибки |
| 3 | `zip(..., strict=True)` → `zip(...)` | `dataset.py` | Параметр `strict` доступен только в Python 3.10+ |
| 4 | Добавлен отсутствующий импорт `cast` | Несколько файлов | `cast` использовался без импорта |

## Ключевые возможности

Все функции PyTorch Lightning 2.6.2 сохранены и работают:

- **Trainer API** — Полные циклы обучения, валидации, тестирования и предсказания
- **LightningModule** — Все step-методы, хуки и конфигурация оптимизатора
- **LightningDataModule** — Управление данными и загрузчиками
- **Коллбэки** — EarlyStopping, ModelCheckpoint, LearningRateMonitor, DeviceStatsMonitor
- **Логгеры** — CSVLogger (TensorBoardLogger требует установки tensorboard)
- **Обучение со смешанной точностью** — FP16-mixed и BF16-mixed
- **Двойная точность** — Режим 64-true
- **Накопление градиентов** — Поддержка `accumulate_grad_batches`
- **Контрольные точки модели** — Сохранение и загрузка чекпоинтов
- **Lightning Fabric** — API нижнего уровня для пользовательских циклов обучения
- **Воспроизводимость** — `seed_everything()` работает корректно
- **GPU-ускорение** — Поддержка CUDA с автоматическим определением устройства

## Результаты тестирования

Все 29 тестов пройдены с **процентом успешности 100%**.

### Основные показатели производительности

| Метрика | CPU | GPU | Отношение GPU/CPU |
|---------|-----|-----|-------------------|
| Скорость обучения (выборок/с) | 4 653,2 | 29 359,4 | **6,31x** |
| Время эпохи (с) | 8,60 | 1,36 | В 6,31 раза быстрее |
| Пиковая память GPU | — | 21,6 МБ | — |

| Специальный сценарий | Результат |
|---------------------|-----------|
| Смешанная точность FP16 | Пройден (0,08 с) |
| Смешанная точность BF16 | Пройден (0,07 с) |
| Двойная точность (64-true) | Пройден |
| Накопление градиентов | Пройден |
| Ранняя остановка | Пройден (остановлен на эпохе 36/50) |
| Мониторинг статистики устройства | Пройден |
| Lightning Fabric | Пройден |

### Известные ограничения

1. **cuDNN не включён** — Может повлиять на скорость обучения CNN
2. **TensorBoardLogger не протестирован** — Пакет `tensorboard` не установлен из-за нехватки дискового пространства
3. **Распределённое обучение не тестировалось** — Только однографовая среда
4. **Статическая проверка типов** — Некоторые вызовы `cast()` были удалены, что может повлиять на вывод mypy/pyright

## Как собрать из исходного кода

### Предварительные требования

- Python 3.8.x (протестировано с 3.8.10)
- Совместимая сборка PyTorch 2.13 для Python 3.8
- Git

### Шаги сборки

```bash
# 1. Клонировать репозиторий
git clone https://github.com/Lanurence666/pytorch-lightning-backport-py38.git
cd pytorch-lightning-backport-py38

# 2. Установить зависимости для сборки
pip install setuptools wheel build

# 3. Собрать wheel-пакет
python -m build

# Wheel-пакет будет в каталоге dist/
```

### Установка из исходного кода (режим редактирования)

```bash
pip install -e .
```

### Установка готового wheel-пакета

Скачайте wheel-пакет со [страницы релизов](https://github.com/Lanurence666/pytorch-lightning-backport-py38/releases) и установите:

```bash
pip install pytorch_lightning-2.6.2-py3-none-any.whl
```

## Запуск тестов

```bash
python tests/comprehensive_test.py
```

## Лицензия

Этот проект лицензирован под лицензией Apache License 2.0 — подробности см. в файле [LICENSE](./LICENSE).
