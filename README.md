# AlphaZero Chess

Implementacja algorytmu AlphaZero dla szachów. Projekt wykorzystuje głębokie uczenie maszynowe i przeszukiwanie drzewa Monte Carlo (MCTS) do trenowania silnika szachowego, który uczy się grać w szachy bez wprowadzania specjalistycznej wiedzy ludzkiej.

## Informacje o projekcie

AlphaZero Chess to implementacja inspirowana algorytmem DeepMind AlphaZero, zastosowana specyficznie do gry w szachy. Projekt umożliwia:

- Trening modelu od podstaw poprzez samouczenie się (self-play)
- Granie przeciwko wytrenowanemu modelowi poprzez intuicyjny interfejs graficzny
- Ewaluację modeli w celu porównania ich siły gry
- Wizualizację i analizę rozegranych partii

## Struktura projektu

- `alpha_net.py` - Implementacja sieci neuronowej (policy i value networks)
- `MCTS_chess.py` - Implementacja przeszukiwania drzewa Monte Carlo
- `chess_board.py` - Reprezentacja szachownicy i zasad gry
- `chess_gui.py` - Interfejs graficzny do gry z modelem
- `encoder_decoder.py` - Kodowanie/dekodowanie stanów szachownicy do postaci wektorowej
- `evaluator.py` - Narzędzie do ewaluacji siły modeli
- `pipeline.py` - Główny pipeline treningowy łączący self-play i trening sieci
- `cpu_pipeline.py` - Alternatywna wersja zoptymalizowana dla procesorów CPU
- `run_alphazero.py` - Pomocniczy skrypt do uruchamiania różnych komponentów projektu
- `init_model.py` - Inicjalizacja nowego modelu
- `visualize_board.py` - Wizualizacja stanów szachownicy
- `analyze_games.py` - Narzędzie do analizy rozegranych partii

## Wymagania

- Python 3.6+
- PyTorch
- NumPy
- Matplotlib
- Pygame (dla GUI)

Pełna lista zależności znajduje się w pliku `requirements.txt`.

## Instalacja

1. Sklonuj repozytorium:
   ```
   git clone https://github.com/username/AlphaZero_Chess.git
   cd AlphaZero_Chess
   ```

2. Zainstaluj wymagane pakiety:
   ```
   pip install -r requirements.txt
   ```

3. Przygotuj środowisko:
   ```
   python run_alphazero.py setup
   ```

## Użycie

### Trening modelu

```
python run_alphazero.py train --iterations 5 --games 10 --workers 4
```

Parametry:
- `--iterations`: Liczba iteracji treningu
- `--games`: Liczba gier na proces roboczy
- `--workers`: Liczba równoległych procesów
- `--mcts_sims`: Liczba symulacji MCTS na ruch (domyślnie 800)
- `--epochs`: Liczba epok treningu na iterację (domyślnie 20)

### Gra przeciwko modelowi

```
python run_alphazero.py play --model current_net_trained_iter4.pth.tar --color white
```

Parametry:
- `--model`: Ścieżka do modelu (opcjonalnie)
- `--color`: Kolor, którym grasz ('white' lub 'black')
- `--mcts_sims`: Liczba symulacji MCTS na ruch AI (domyślnie 800)

### Ewaluacja modeli

```
python run_alphazero.py evaluate --model1 current_net_trained_iter3.pth.tar --model2 current_net_trained_iter4.pth.tar --games 100
```

Parametry:
- `--model1`: Pierwszy model do ewaluacji
- `--model2`: Drugi model do ewaluacji
- `--games`: Liczba gier do rozegrania
- `--workers`: Liczba równoległych procesów
- `--mcts_sims`: Liczba symulacji MCTS na ruch (domyślnie 800)

## Planowane ulepszenia

Lista potencjalnych ulepszeń projektu znajduje się w pliku `ToDo.md`. Zawiera ona propozycje optymalizacji:
- Architektury sieci neuronowej
- Algorytmu Monte Carlo Tree Search
- Procesu treningowego
- Zarządzania danymi
- Gier końcowych
- Integracji z serwisem Lichess
