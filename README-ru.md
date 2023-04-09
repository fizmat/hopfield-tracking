# hopfield-tracking

Исследование трекинга с помощью сети Хопфилда.

## Подготовка

```shell
conda env create
conda activate hopfield-tracking
nb-clean add-filter --remove-empty-cells
```

## Структура кода
*   datasets - чтение и генерация датасетов
*   tracking - общие задачи трекинга (переход в цилиндрические координаты, визуализация событий)
*   segment - генерация массивов сегментов-кандидатов и трековых сегментов
*   hopfield - создание матрицы энергии, итерация сети Хопфилда
*   metrics - оценка результата
*   optimize.py - оптимизация параметров

## Блокноты Jupyter

`jupyter lab`

- [dataset_stats](dataset_stats.ipynb): гистограмма размеров событий (количества треков и хитов)
- [demo_datasets](demo_datasets.ipynb): визуализация одного события
- [demo_seg](demo_seg.ipynb): сравнение методов генерации сегментов-кандидатов
- [demo_track_seg](demo_track_seg.ipynb): сравнение методов генерации трековых сегментов
- [stat_seg_length](stat_seg_length.ipynb): распределения длин сегментов для оценки применимости поиска соседей
- [stat_seg_neighbors](stat_seg_neighbors.ipynb): распределение длин сегментов с помощью поиска соседей
- [profile_seg_pools](profile_seg_pools.ipynb): сравнение способов распараллеливания (для stat_seg_neighbors)
- [demo_event](demo_event.ipynb): демонстрация процесса трекинга для одного события
- [sbatch](sbatch.ipynb): запуск распределенной оптимизации с jlab-hpc

## Тестирование

`pytest` - юнит-тесты

`tox` - тестирование в разных окружениях

## Hybrilit
[Instructions](README-jhub.md) - инструкции по настройке на Гибрилите (jhub2 и jlabhpc).