# Curriculum-based Local Image Augmentation

Projekt Interdyscyplinarny — Politechnika Warszawska, Wydział MiNI  
Zespół: [Kamińska](https://github.com/kaminskaz), [Dunal](https://github.com/xxkaro), [Choszczyk](https://github.com/nataliachoszczyk), [Deptuch](https://github.com/DeptuchMateusz)

## Opis

Celem projektu było zaprojektowanie i implementacja augmentora obrazów, który umożliwia stosowanie lokalnych transformacji na nieregularnych fragmentach obrazu. Augmentor działa w oparciu o siatkę podziału i definiowalne tryby modyfikacji.

## Schemat działania augmentora

1. **Segmentacja obrazu** – siatka `n × n` dzieląca obraz na fragmenty.
2. **Losowy wybór segmentów** – wybierane są podobszary do modyfikacji.
3. **Lokalne augmentacje** – dla wybranych segmentów stosowane są transformacje (np. obrót, jasność, kontrast).
4. **Parametr p** – kontroluje intensywność lub prawdopodobieństwo transformacji (możliwa zmiana w czasie nauki).

<p align="center">
  <img src="/images_for_documentation/fig1/diagram.png" alt="Schemat działania augmentora" width="600"/>
</p>

## Struktura projektu

- `code/`
  - `Augmentor.py`, `STESAugmentor.py` – klasy implementujące lokalne augmentacje
  - `wrapper.py` – moduł nakładający augmentację na dane
  - `newtrain.py`, `visualizations.py` – pipeline treningowy i ewaluacyjny
  - `*.sl` – skrypty SLURM do uruchamiania eksperymentów na EDENie
  - `test_notebooks/` – wizualizacje i testy działania augmentorów na pojedynczych obrazach
- pozostałe pliki to wygenerowane outputy oraz pliki potrzebne do dokumentacji

## Eksperymenty

Eksperymenty porównawcze przeprowadzono na kilku wariantach augmentacji. Szczegóły, wyniki i analizy dostępne są w dokumentacji projektu:

**[Dokumentacja (PDF)](/Projekt_Interdyscyplinarny_raport.pdf)**

## Wymagania

```bash
pip install -r requirements.txt
