# 📌 Lista potencjalnych ulepszeń projektu AlphaChess

## 🧠 **Optymalizacja architektury sieci neuronowej**
- Zmiana reprezentacji danych wejściowych (feature engineering), która może zwiększyć efektywność sieci nawet o 180 punktów Elo ([arxiv.org](https://arxiv.org/html/2304.14918v2)).
- Zwiększenie liczby filtrów splotowych w policy head z 3 do 32 w celu przyspieszenia treningu ([scitepress.org](https://www.scitepress.org/PublishedPapers/2021/102459/102459.pdf)).
- Testowanie warstw Squeeze-and-Excitation w blokach rezydualnych, uwzględniając bilans między wydajnością a kosztem ([arxiv.org](https://arxiv.org/pdf/2109.11602.pdf)).

### Optymalizacja algorytmu Monte Carlo Tree Search
- Możliwe jest wykorzystanie Monte Carlo Graph Search (MCGS), który poprzez wykorzystanie transpozycji pozycji zmniejsza zużycie pamięci o 30–70% ([ml-research.github.io](https://ml-research.github.io/papers/czech2021icaps_mcgs.pdf)).
- Alternatywą lub uzupełnieniem UCT jest epsilon-greedy exploration, czyli losowe wybieranie ruchów z pewnym prawdopodobieństwem w celu uniknięcia lokalnych minimów ([geeksforgeeks.org](https://www.geeksforgeeks.org/epsilon-greedy-algorithm-in-reinforcement-learning/)).
- Zastosowanie techniki „Playout Caps”, polegającej na redukcji liczby symulacji w mniej perspektywicznych ruchach, może przyspieszyć trening o około 27% bez dużych strat jakościowych ([scitepress.org](https://www.scitepress.org/PublishedPapers/2021/102459/102459.pdf)).

### Usprawnienia procesu treningowego
- Możliwe jest wykorzystanie podejścia Path Consistency (PC), które przyspiesza konwergencję modelu i osiąga lepsze rezultaty z mniejszą liczbą gier treningowych ([proceedings.mlr.press](https://proceedings.mlr.press/v162/zhao22h/zhao22h.pdf)).
- Warto rozważyć użycie Q-values generowanych w MCTS jako dodatkowego celu treningowego zamiast jedynie końcowego wyniku gry, co może zapewnić bardziej stabilne uczenie ([reddit.com](https://www.reddit.com/r/MachineLearning/comments/8ubkrl/d_improvement_to_the_alphazero_value_target/)).
- Trening populacyjny (Population-Based Training, PBT) pozwala na dynamiczną optymalizację hiperparametrów poprzez równoległe trenowanie wielu modeli i ich adaptację w trakcie uczenia ([arxiv.org](https://arxiv.org/abs/2003.06212)).

### Optymalizacja zarządzania danymi
- Rozważenie augmentacji danych treningowych z użyciem symetrii szachownicy, co zwiększa efektywność treningu bez generowania nowych partii ([scitepress.org](https://www.scitepress.org/PublishedPapers/2021/102459/102459.pdf)).
- Można również zastosować priorytetyzację danych, skupiając trening na trudnych lub nowych pozycjach.
- Optymalizacja procesu ładowania danych (asynchroniczne wczytywanie, prefetching, cache’owanie danych) może zwiększyć wykorzystanie GPU ([aaai.org](https://ojs.aaai.org/index.php/AAAI/article/view/5454/5310)).

### Poprawa efektywności gier końcowych
- Możliwe jest stworzenie specjalnej reprezentacji cech gier końcowych (np. aktywność króla, odległości między pionami i królami), co poprawia dokładność oceny tych pozycji ([arxiv.org](https://arxiv.org/html/2304.14918v2)).
- Implementacja prostego solvera pozycji końcowych umożliwia dokładne rozegranie elementarnych gier końcowych takich jak matowanie królem i hetmanem przeciwko królowi ([ml-research.github.io](https://ml-research.github.io/papers/czech2021icaps_mcgs.pdf)).
- Integracja tabel gier końcowych (np. Syzygy Tablebases) zapewnia perfekcyjną grę dla pozycji z ograniczoną liczbą figur ([arxiv.org](https://arxiv.org/pdf/2109.11602.pdf)).
- Dostosowanie funkcji wartości tak, aby uwzględniała osobno prawdopodobieństwo remisu, umożliwiając bardziej precyzyjną ocenę gier końcowych ([scitepress.org](https://www.scitepress.org/PublishedPapers/2021/102459/102459.pdf)).

### Optymalizacje wdrożeniowe
- Profilowanie wydajności modelu przy użyciu narzędzi takich jak cProfile czy tensorboard profiler pomoże w identyfikacji wąskich gardeł ([TensorFlow profiler](https://www.tensorflow.org/guide/profiler)).
- Rozważenie technik kwantyzacji lub pruning do przyspieszenia inferencji bez znaczącej utraty jakości.
- Stopniowa integracja i regularne testy poszczególnych ulepszeń, by uniknąć konfliktów między różnymi optymalizacjami.

### Integracja z serwisem Lichess
- Możliwa jest integracja bota AlphaChess z serwisem Lichess przy użyciu [API Lichess botów](https://github.com/lichess-bot-devs/lichess-bot).
- Analiza rozegranych partii na Lichess może służyć do zbierania wartościowych danych treningowych.
- Dzięki integracji możliwe będzie prowadzenie automatycznych benchmarków siły bota w rozgrywkach online na Lichess.