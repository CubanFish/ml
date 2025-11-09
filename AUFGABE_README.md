# Aufgabe 2.1 - Machine Learning mit Laptop-Preisen

## Übersicht

Diese Aufgabe ist eine umfassende Machine Learning-Übung, die auf dem **Laptop Price Dataset** von Kaggle basiert.

## Datensatz

**Kaggle-Link**: https://www.kaggle.com/datasets/muhammetvarl/laptop-price

Der Datensatz enthält verschiedene Eigenschaften von Laptops wie:
- **Company**: Hersteller (Dell, HP, Apple, etc.)
- **TypeName**: Laptop-Typ (Notebook, Gaming, Ultrabook, etc.)
- **Ram**: RAM-Größe
- **Weight**: Gewicht
- **Price**: Preis (Regression Target)
- **Cpu**: Prozessor-Information
- **Gpu**: Grafikkarten-Information
- **OpSys**: Betriebssystem
- **Screen**: Bildschirm-Spezifikationen
- **Memory**: Speicher-Informationen

## Lernziele

Diese Aufgabe deckt die folgenden Machine Learning-Konzepte ab:

### 1. Datenvorverarbeitung
- Laden von Daten mit pandas
- Datenbereinigung und -transformation
- Extraktion numerischer Werte aus String-Spalten
- Umgang mit fehlenden Werten

### 2. Explorative Datenanalyse (EDA)
- Visualisierung von Datenverteilungen
- Korrelationsanalyse
- Identifikation von Mustern und Ausreißern

### 3. Feature Engineering
- Identifikation von metrischen vs. nicht-metrischen Features
- One-Hot Encoding für kategorische Variablen
- Feature Selection (Entfernung von Features mit zu vielen Kategorien)

### 4. Regression
- **Linear Regression**: Vorhersage von Laptop-Preisen
- **Lasso Regression**: L1-Regularisierung für Feature Selection
- Evaluation mit R²-Score
- Modellstabilität durch wiederholte Train-Test-Splits

### 5. Klassifikation
- **Logistische Regression**: Vorhersage von Gewichtskategorien oder Laptop-Typen
- Evaluation mit Accuracy, Precision, Recall, F1-Score
- Confusion Matrix
- Vergleich mit Benchmark-Modellen

### 6. Best Practices
- Korrekte Train-Test-Split Durchführung
- Vermeidung von Data Leakage beim One-Hot Encoding
- Standardisierung mit MinMaxScaler
- Reproduzierbarkeit durch random_state

## Unterschiede zur Original-Aufgabe (Fahrzeug-Datensatz)

| Aspekt | Original (Fahrzeuge) | Neue Aufgabe (Laptops) |
|--------|---------------------|------------------------|
| **Datensatz** | Vehicle Dataset from Cardekho | Laptop Price Dataset |
| **Regression Target** | Price | Price |
| **Klassifikation Target** | Seating Capacity | Weight Category oder TypeName |
| **Kategorische Features** | Fuel Type, Transmission, Owner | Company, TypeName, CPU, GPU, OpSys |
| **Numerische Features** | Year, km_driven, mileage | RAM, Weight, Screen Size, Memory |
| **String-Parsing** | Mileage (z.B. "18.2 kmpl") | RAM (z.B. "8GB"), Memory (z.B. "512GB SSD") |

## Struktur der Aufgabe

Die Aufgabe ist in 25 Teilaufgaben (a-y) unterteilt:

### Teil 1: Regression (a-p)
- **a-d**: Daten laden, bereinigen, aufteilen
- **e-h**: Feature-Analyse und Strategien für fehlende Werte
- **i-m**: Train-Test-Split, Preprocessing, erstes Modell
- **n**: Modellstabilität bewerten (20 Wiederholungen)
- **o**: One-Hot Encoding implementieren
- **p**: Lasso-Regression und Vergleich

### Teil 2: Klassifikation (q-x)
- **q-t**: Neues Target definieren, Daten vorbereiten
- **u-v**: Logistische Regression trainieren
- **w**: Modellstabilität für Klassifikation
- **x**: One-Hot Encoding für Klassifikation

### Teil 3: Vertiefung (y)
- **y**: Weiterführende Fragestellung selbst definieren und beantworten

## Wichtige Hinweise für die Bearbeitung

### 1. Data Leakage vermeiden
- Immer zuerst `fit()` auf Trainingsdaten, dann `transform()` auf Train und Test
- Gilt für: Scaler, Imputer, OneHotEncoder

### 2. One-Hot Encoding Probleme
- Problem: Test-Set kann neue Kategorien enthalten, die im Training nicht vorkamen
- Lösung: `OneHotEncoder(handle_unknown='ignore')` oder `pd.get_dummies()` mit Spaltenkonsistenz

### 3. Wiederholte Experimente
- Verwenden Sie Listen zum Speichern der R²-Scores über 20 Iterationen
- Berechnen Sie Mittelwert und Standardabweichung
- Visualisieren Sie die Verteilung mit Boxplots

### 4. Dokumentation
- Kommentieren Sie Ihren Code ausreichend
- Erklären Sie Ihre Entscheidungen in Markdown-Zellen
- Interpretieren Sie die Ergebnisse

## Beispiel-Code-Struktur

```python
# Typischer Workflow für Teil n)

results = {'train_r2': [], 'test_r2': []}

for i in range(20):
    # 1. Train-Test-Split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=i)

    # 2. Preprocessing
    # ... (Imputation, Scaling)

    # 3. Training
    model = LinearRegression()
    model.fit(X_train, y_train)

    # 4. Evaluation
    train_r2 = r2_score(y_train, model.predict(X_train))
    test_r2 = r2_score(y_test, model.predict(X_test))

    results['train_r2'].append(train_r2)
    results['test_r2'].append(test_r2)

# 5. Analyse der Stabilität
print(f"Train R²: {np.mean(results['train_r2']):.3f} ± {np.std(results['train_r2']):.3f}")
print(f"Test R²: {np.mean(results['test_r2']):.3f} ± {np.std(results['test_r2']):.3f}")
```

## Erwartete Ergebnisse

### Regression
- R²-Score auf Test-Set: ~0.7 - 0.85 (mit One-Hot Encoding)
- Lasso sollte Feature Selection durchführen (einige Koeffizienten = 0)
- Mit kategorischen Features sollte die Performance besser sein

### Klassifikation
- Accuracy sollte deutlich über dem Benchmark-Modell liegen
- Bei balancierten Klassen: ~70-85% Accuracy
- Confusion Matrix zeigt, welche Klassen verwechselt werden

## Zusätzliche Ressourcen

### Sklearn-Dokumentation
- [LinearRegression](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LinearRegression.html)
- [Lasso](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.Lasso.html)
- [LogisticRegression](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html)
- [OneHotEncoder](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.OneHotEncoder.html)
- [MinMaxScaler](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.MinMaxScaler.html)
- [SimpleImputer](https://scikit-learn.org/stable/modules/generated/sklearn.impute.SimpleImputer.html)

### Pandas String-Operations
- [str.extract()](https://pandas.pydata.org/docs/reference/api/pandas.Series.str.extract.html) - Regex für Zahlen extrahieren
- [str.replace()](https://pandas.pydata.org/docs/reference/api/pandas.Series.str.replace.html) - Strings ersetzen

## Typische Herausforderungen

1. **String-Parsing**: RAM "8GB" → 8, Memory "512GB SSD + 1TB HDD" → komplexer
2. **Fehlende Werte**: Strategische Entscheidungen nötig
3. **Kategorische Features**: Viele verschiedene CPU/GPU-Modelle
4. **Overfitting**: Lasso hilft bei zu vielen Features
5. **Klassenungleichgewicht**: Bei Klassifikation möglicherweise unbalanciert

## Bewertungskriterien

- ✅ Vollständigkeit: Alle Teilaufgaben a-y bearbeitet
- ✅ Korrektheit: Technisch korrekte Implementierung
- ✅ Dokumentation: Ausreichende Erklärungen und Kommentare
- ✅ Interpretation: Ergebnisse werden interpretiert und diskutiert
- ✅ Code-Qualität: Sauberer, lesbarer Code
- ✅ Visualisierung: Aussagekräftige Plots
- ✅ Wissenschaftlichkeit: Begründete Entscheidungen, Quellenangaben

---

**Viel Erfolg bei der Bearbeitung!**
