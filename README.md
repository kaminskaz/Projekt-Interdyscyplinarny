# NIEAKTUALNE README

# Stopień augmentacji zależny od fazy nauki
## Inspiracja podejściem Curriculum Learning

### Zespół
- [Zofia Kamińska](https://github.com/kaminskaz)
- [Karolina Dunal](https://github.com/xxkaro)
- [Natalia Choszczyk](https://github.com/nataliachoszczyk)
- [Mateusz Deptuch](https://github.com/DeptuchMateusz)

### Cel projektu
Celem jest zbadanie wpływu dynamicznego stopniowania augmentacji na proces uczenia modeli głębokich, szczególnie w kontekście klasyfikacji obrazów i problemów z niezrównoważonymi danymi.

### Pytanie badawcze
Jak dynamiczne dostosowanie intensywności augmentacji w zależności od fazy nauki wpływa na jakość i efektywność uczenia modeli głębokich?

## Harmonogram realizacji (10 tygodni)

### Etap 0: Zarys projektu i harmonogram (Tydzień 0 do 7.03.2025)
**Cel:**
- Dobór zespołu
- Ustalenie zakresu projektu
- Tworzenie harmonogramu

### Etap 1: Research i projektowanie (Tydzień 1-2 do 17.03.2025)
**Cel:**
- Przegląd literatury i opracowanie strategii augmentacji
- Analiza artykułów dotyczących curriculum learning i augmentacji
- Przegląd istniejących technik augmentacji
- Przegląd benchmarków i zbiorów danych do eksperymentów
- Wybór metryk ewaluacji i określenie struktury eksperymentu

**Efekt:**
- Dokument opisujący strategię augmentacji zależną od fazy nauki
- Wybór datasetu i modelu

### Etap 2: Implementacja strategii augmentacji (Tydzień 3-5 do 7.04.2025)
**Cel:**
- Zaimplementowanie dynamicznej augmentacji
- Implementacja augmentacji statycznej (bazowej)
- Implementacja augmentacji dynamicznej (np. rosnąca siła transformacji w kolejnych epokach)
- Dostosowanie augmentacji do klasy trudności (np. większa dla łatwych przykładów)

**Efekt:**
Zaimplementowane i gotowe do testów podejścia:
- Model bez augmentacji (baseline)
- Model ze stałą augmentacją
- Model z curriculum augmentacją

### Etap 3: Eksperymenty (Tydzień 6-8 do 28.04.2025)
**Cel:**
- Przetestowanie różnych podejść i zebranie wyników
- Uruchomienie eksperymentów na modelu bez augmentacji
- Uruchomienie eksperymentów na modelu ze stałą augmentacją
- Uruchomienie eksperymentów na modelu z dynamiczną augmentacją
- Analiza wyników

**Efekt:**
- Wyniki w formie tabel i wykresów

### Etap 4: Analiza i wnioski (Tydzień 9-10 do 12.05.2025)
**Cel:**
- Wyciągnięcie wniosków i przygotowanie podsumowania projektu
- Analiza wpływu dynamicznej augmentacji na szybkość nauki
- Analiza wpływu na overfitting i generalizację
- Porównanie wyników w problemach z niezrównoważonymi danymi
- Wnioski i potencjalne ulepszenia dla przyszłych badań

**Efekt:**
- Podsumowanie, raport i prezentacja końcowa
