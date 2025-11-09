# Tipps und Hilfestellungen zur Datenbereinigung und -aufbereitung

## Inhaltsverzeichnis
1. [Übersicht](#übersicht)
2. [Daten laden und inspizieren](#1-daten-laden-und-inspizieren)
3. [String-Parsing und Feature-Extraktion](#2-string-parsing-und-feature-extraktion)
4. [Umgang mit fehlenden Werten](#3-umgang-mit-fehlenden-werten)
5. [Datentyp-Konvertierung](#4-datentyp-konvertierung)
6. [Ausreißer-Behandlung](#5-ausreißer-behandlung)
7. [Best Practices](#6-best-practices)

---

## Übersicht

Datenbereinigung ist einer der wichtigsten Schritte im Machine Learning-Prozess. Schlechte Datenqualität führt zu schlechten Modellen, unabhängig davon, wie gut der Algorithmus ist.

**Faustregel**: 80% der Zeit eines Data Scientists wird für Datenbereinigung verwendet, nur 20% für Modellierung!

---

## 1. Daten laden und inspizieren

### Erste Schritte nach dem Laden

```python
import pandas as pd
import numpy as np

# Daten laden
df = pd.read_csv('laptop_data.csv')

# WICHTIG: Immer zuerst die Daten inspizieren!
print(f"Datensatz-Größe: {df.shape}")
print(f"Anzahl Zeilen: {df.shape[0]}, Anzahl Spalten: {df.shape[1]}")

# Erste Zeilen anzeigen
df.head()

# Informationen zu Datentypen und fehlenden Werten
df.info()

# Statistische Zusammenfassung
df.describe()

# Fehlende Werte pro Spalte
print("\nFehlende Werte:")
print(df.isnull().sum())

# Datentypen überprüfen
print("\nDatentypen:")
print(df.dtypes)
```

### Duplikate erkennen

```python
# Duplikate finden
duplicates = df.duplicated().sum()
print(f"Anzahl Duplikate: {duplicates}")

# Duplikate anzeigen
if duplicates > 0:
    print(df[df.duplicated(keep=False)])

# Duplikate entfernen (falls sinnvoll)
df = df.drop_duplicates()
```

---

## 2. String-Parsing und Feature-Extraktion

### Beispiel: RAM extrahieren

Das RAM-Feature enthält oft Strings wie "8GB", "16GB", etc.

```python
# Methode 1: Mit str.extract() und Regex
df['Ram_GB'] = df['Ram'].str.extract('(\d+)').astype(int)

# Methode 2: Mit str.replace()
df['Ram_GB'] = df['Ram'].str.replace('GB', '').astype(int)

# Überprüfen
print(df[['Ram', 'Ram_GB']].head())
```

### Beispiel: Memory/Speicher extrahieren

Memory kann komplex sein: "512GB SSD", "1TB HDD", "256GB SSD + 1TB HDD"

```python
import re

def extract_storage(storage_str):
    """
    Extrahiert Speichergröße in GB
    Beispiele:
    - "512GB SSD" → 512
    - "1TB HDD" → 1024
    - "256GB SSD + 1TB HDD" → 1280
    """
    if pd.isna(storage_str):
        return np.nan

    total_gb = 0

    # Finde alle Speicher-Angaben (z.B. "512GB" oder "1TB")
    # Pattern: Zahl gefolgt von GB oder TB
    pattern = r'(\d+\.?\d*)\s*(GB|TB)'
    matches = re.findall(pattern, storage_str, re.IGNORECASE)

    for size, unit in matches:
        size = float(size)
        if unit.upper() == 'TB':
            size *= 1024  # TB zu GB konvertieren
        total_gb += size

    return total_gb

# Anwenden
df['Storage_GB'] = df['Memory'].apply(extract_storage)

# Überprüfen
print(df[['Memory', 'Storage_GB']].head(10))
```

### Beispiel: SSD vs HDD unterscheiden

```python
# SSD vorhanden?
df['Has_SSD'] = df['Memory'].str.contains('SSD', case=False, na=False).astype(int)

# HDD vorhanden?
df['Has_HDD'] = df['Memory'].str.contains('HDD', case=False, na=False).astype(int)

# Nur SSD (kein HDD)
df['Only_SSD'] = ((df['Has_SSD'] == 1) & (df['Has_HDD'] == 0)).astype(int)
```

### Beispiel: Weight extrahieren

Weight könnte "1.5kg" oder "2.3kg" sein

```python
# Weight in kg als Float
df['Weight_kg'] = df['Weight'].str.extract('([\d.]+)').astype(float)

# Überprüfen
print(df[['Weight', 'Weight_kg']].describe())
```

### Beispiel: Screen Resolution extrahieren

```python
def extract_screen_info(screen_str):
    """
    Extrahiert Bildschirm-Informationen
    Beispiel: "15.6 inch Full HD 1920x1080"
    """
    if pd.isna(screen_str):
        return np.nan, np.nan, np.nan

    # Bildschirmgröße (z.B. 15.6)
    size_match = re.search(r'(\d+\.?\d*)\s*inch', screen_str, re.IGNORECASE)
    size = float(size_match.group(1)) if size_match else np.nan

    # Auflösung (z.B. 1920x1080)
    resolution_match = re.search(r'(\d+)x(\d+)', screen_str)
    if resolution_match:
        width = int(resolution_match.group(1))
        height = int(resolution_match.group(2))
        # Pixel-Dichte berechnen
        pixels = width * height
    else:
        pixels = np.nan

    # Touchscreen?
    touchscreen = 1 if 'touch' in screen_str.lower() else 0

    return size, pixels, touchscreen

# Anwenden
df[['Screen_Size', 'Screen_Pixels', 'Touchscreen']] = df['ScreenResolution'].apply(
    lambda x: pd.Series(extract_screen_info(x))
)
```

---

## 3. Umgang mit fehlenden Werten

### Schritt 1: Fehlende Werte analysieren

```python
# Absolute Anzahl
missing_count = df.isnull().sum()

# Prozentual
missing_percent = (df.isnull().sum() / len(df)) * 100

# Übersicht
missing_df = pd.DataFrame({
    'Anzahl': missing_count,
    'Prozent': missing_percent
})
missing_df = missing_df[missing_df['Anzahl'] > 0].sort_values('Anzahl', ascending=False)
print(missing_df)
```

### Schritt 2: Strategie wählen

#### Strategie 1: Zeilen/Spalten entfernen

```python
# Zeilen mit ANY fehlenden Werten entfernen
df_clean = df.dropna()

# Zeilen mit fehlenden Werten in BESTIMMTEN Spalten entfernen
df_clean = df.dropna(subset=['Price', 'Weight'])

# Spalten mit > 50% fehlenden Werten entfernen
threshold = 0.5
df_clean = df.loc[:, df.isnull().mean() < threshold]
```

**Wann verwenden?**
- Wenn < 5% der Zeilen betroffen sind
- Bei Target-Variable (y) → IMMER entfernen
- Bei Spalten mit > 50% fehlenden Werten

#### Strategie 2: Mittelwert/Median-Imputation

```python
from sklearn.impute import SimpleImputer

# Für numerische Features
imputer_mean = SimpleImputer(strategy='mean')
imputer_median = SimpleImputer(strategy='median')

# WICHTIG: Nur auf X_train fitten!
# Beispiel (nach Train-Test-Split):
imputer = SimpleImputer(strategy='median')
X_train[['Weight_kg']] = imputer.fit_transform(X_train[['Weight_kg']])
X_test[['Weight_kg']] = imputer.transform(X_test[['Weight_kg']])
```

**Wann verwenden?**
- Median: Bei Ausreißern (robust)
- Mean: Bei normalverteilten Daten ohne Ausreißer

#### Strategie 3: Most Frequent (Mode) für kategorische Features

```python
# Für kategorische Features
imputer_mode = SimpleImputer(strategy='most_frequent')

# Anwenden
df['OpSys'].fillna(df['OpSys'].mode()[0], inplace=True)
```

#### Strategie 4: Forward/Backward Fill

```python
# Forward Fill (vorherigen Wert verwenden)
df['Weight'].fillna(method='ffill', inplace=True)

# Backward Fill (nächsten Wert verwenden)
df['Weight'].fillna(method='bfill', inplace=True)
```

**Wann verwenden?**
- Bei Zeitreihen oder sortierten Daten
- NICHT empfohlen für unseren Laptop-Datensatz (keine zeitliche Ordnung)

#### Strategie 5: Konstanter Wert

```python
# Mit einem festen Wert füllen
df['Weight_kg'].fillna(0, inplace=True)

# Oder mit "Unknown" für kategorische Features
df['OpSys'].fillna('Unknown', inplace=True)
```

#### Strategie 6: KNN-Imputation (fortgeschritten)

```python
from sklearn.impute import KNNImputer

# Verwendet K-Nearest Neighbors um Werte zu schätzen
knn_imputer = KNNImputer(n_neighbors=5)

# Nur auf numerischen Daten
numeric_cols = df.select_dtypes(include=[np.number]).columns
df[numeric_cols] = knn_imputer.fit_transform(df[numeric_cols])
```

**Wann verwenden?**
- Wenn Features korreliert sind
- Bei MCAR (Missing Completely At Random)

### Vergleich der Strategien

| Strategie | Vorteile | Nachteile | Wann verwenden |
|-----------|----------|-----------|----------------|
| **Zeilen entfernen** | Einfach, keine Verzerrung | Datenverlust | < 5% fehlend |
| **Spalten entfernen** | Einfach | Feature-Verlust | > 50% fehlend |
| **Mean** | Einfach, schnell | Verringert Varianz | Normalverteilung |
| **Median** | Robust gegen Ausreißer | Verringert Varianz | Schiefe Verteilung |
| **Mode** | Für Kategorien geeignet | Verzerrt Verteilung | Kategorische Daten |
| **KNN** | Nutzt Beziehungen | Rechenintensiv | Korrelierte Features |

---

## 4. Datentyp-Konvertierung

### Kategorische Variablen identifizieren

```python
# Automatisch object-Spalten finden
categorical_cols = df.select_dtypes(include=['object']).columns
print(f"Kategorische Spalten: {categorical_cols.tolist()}")

# Numerische Spalten
numeric_cols = df.select_dtypes(include=[np.number]).columns
print(f"Numerische Spalten: {numeric_cols.tolist()}")
```

### Kategorien mit zu vielen Werten filtern

```python
# Anzahl unique Values pro kategorischer Spalte
for col in categorical_cols:
    n_unique = df[col].nunique()
    print(f"{col}: {n_unique} verschiedene Werte")

# Features mit > 50 Kategorien entfernen (wie in Aufgabe f)
threshold = 50
cols_to_drop = []

for col in categorical_cols:
    if df[col].nunique() > threshold:
        cols_to_drop.append(col)
        print(f"Entferne {col}: {df[col].nunique()} Kategorien")

df = df.drop(columns=cols_to_drop)
```

---

## 5. Ausreißer-Behandlung

### Ausreißer visualisieren

```python
import matplotlib.pyplot as plt
import seaborn as sns

# Boxplot für numerische Variablen
numeric_features = df.select_dtypes(include=[np.number]).columns

fig, axes = plt.subplots(len(numeric_features), 1, figsize=(10, 4*len(numeric_features)))

for idx, col in enumerate(numeric_features):
    sns.boxplot(data=df, x=col, ax=axes[idx])
    axes[idx].set_title(f'Boxplot: {col}')

plt.tight_layout()
plt.show()
```

### Ausreißer identifizieren: IQR-Methode

```python
def detect_outliers_iqr(df, column):
    """
    Identifiziert Ausreißer mit der IQR-Methode
    Ausreißer: Werte außerhalb von [Q1 - 1.5*IQR, Q3 + 1.5*IQR]
    """
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1

    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR

    outliers = df[(df[column] < lower_bound) | (df[column] > upper_bound)]

    print(f"{column}:")
    print(f"  Untere Grenze: {lower_bound:.2f}")
    print(f"  Obere Grenze: {upper_bound:.2f}")
    print(f"  Anzahl Ausreißer: {len(outliers)}")

    return outliers

# Anwenden
outliers_price = detect_outliers_iqr(df, 'Price')
```

### Ausreißer behandeln

```python
# Option 1: Entfernen
def remove_outliers(df, column):
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR
    return df[(df[column] >= lower) & (df[column] <= upper)]

# Option 2: Capping (Winsorization)
def cap_outliers(df, column):
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR

    df[column] = df[column].clip(lower=lower, upper=upper)
    return df

# Option 3: Log-Transformation (bei stark rechtsschiefen Daten)
df['Price_log'] = np.log1p(df['Price'])
```

**Hinweis**: Bei Preisen sind hohe Werte oft legitim (Gaming-Laptops, Workstations). Überlege gut, ob Ausreißer wirklich Fehler sind!

---

## 6. Best Practices

### 6.1 Data Leakage vermeiden

**WICHTIG**: Preprocessing-Schritte müssen auf Trainingsdaten gefittet werden!

```python
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.impute import SimpleImputer

# Schritt 1: Train-Test-Split ZUERST
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Schritt 2: Imputer auf Training fitten
imputer = SimpleImputer(strategy='median')
imputer.fit(X_train)  # NUR auf Training!

# Schritt 3: Auf beide anwenden
X_train = imputer.transform(X_train)
X_test = imputer.transform(X_test)

# Schritt 4: Scaler auf Training fitten
scaler = MinMaxScaler()
scaler.fit(X_train)  # NUR auf Training!

# Schritt 5: Auf beide anwenden
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)
```

**Warum?** Wenn du zuerst auf dem gesamten Datensatz fittet, "sieht" das Modell indirekt die Test-Daten → Overfitting!

### 6.2 Pipeline verwenden (empfohlen!)

```python
from sklearn.pipeline import Pipeline

# Alle Schritte in einer Pipeline
pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', MinMaxScaler()),
    ('model', LinearRegression())
])

# Fit und Predict in einem Schritt
pipeline.fit(X_train, y_train)
predictions = pipeline.predict(X_test)

# Vorteil: Kein Data Leakage möglich!
```

### 6.3 One-Hot Encoding richtig machen

```python
from sklearn.preprocessing import OneHotEncoder

# Problem: Neue Kategorien im Test-Set
# Lösung: handle_unknown='ignore'

encoder = OneHotEncoder(handle_unknown='ignore', sparse_output=False)

# Auf Training fitten
encoder.fit(X_train[['Company', 'TypeName']])

# Auf beide transformieren
X_train_encoded = encoder.transform(X_train[['Company', 'TypeName']])
X_test_encoded = encoder.transform(X_test[['Company', 'TypeName']])

# Alternative mit pandas: pd.get_dummies()
# ACHTUNG: Spalten müssen manuell synchronisiert werden!
X_train_dummies = pd.get_dummies(X_train, columns=['Company', 'TypeName'])
X_test_dummies = pd.get_dummies(X_test, columns=['Company', 'TypeName'])

# Fehlende Spalten hinzufügen
for col in X_train_dummies.columns:
    if col not in X_test_dummies.columns:
        X_test_dummies[col] = 0

# Überschüssige Spalten entfernen
X_test_dummies = X_test_dummies[X_train_dummies.columns]
```

### 6.4 Reproduzierbarkeit sicherstellen

```python
# IMMER random_state setzen!
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42  # Feste Zahl
)

# Bei Modellen auch
model = LinearRegression()  # Hat keinen random_state
# Aber bei anderen:
from sklearn.ensemble import RandomForestRegressor
model = RandomForestRegressor(random_state=42)
```

### 6.5 Dokumentation und Code-Qualität

```python
# SCHLECHT: Keine Erklärung
df['new_col'] = df['col1'].str.extract('(\d+)').astype(int)

# GUT: Mit Kommentar
# Extrahiere RAM in GB aus Strings wie "8GB RAM"
df['Ram_GB'] = df['Ram'].str.extract('(\d+)').astype(int)

# SEHR GUT: Mit Funktion und Docstring
def extract_ram_gb(ram_str):
    """
    Extrahiert die RAM-Größe in GB aus einem String.

    Args:
        ram_str (str): String wie "8GB" oder "16GB RAM"

    Returns:
        int: RAM-Größe in GB

    Beispiel:
        >>> extract_ram_gb("8GB")
        8
    """
    match = re.search(r'(\d+)', ram_str)
    return int(match.group(1)) if match else np.nan

df['Ram_GB'] = df['Ram'].apply(extract_ram_gb)
```

### 6.6 Validierung der Datenbereinigung

```python
# Nach jedem Schritt überprüfen!

# 1. Shape überprüfen
print(f"Shape vorher: {df.shape}")
df = df.dropna(subset=['Price'])
print(f"Shape nachher: {df.shape}")

# 2. Wertebereiche überprüfen
print(f"RAM-Bereich: {df['Ram_GB'].min()} - {df['Ram_GB'].max()}")
print(f"Speicher-Bereich: {df['Storage_GB'].min()} - {df['Storage_GB'].max()}")

# 3. Plausibilität prüfen
# Gibt es negative Preise? → Fehler!
assert (df['Price'] > 0).all(), "Negative Preise gefunden!"

# Gibt es unrealistische RAM-Werte?
assert df['Ram_GB'].max() <= 128, "Unrealistisch hoher RAM-Wert!"

# 4. Visualisieren
df[['Price', 'Ram_GB', 'Storage_GB']].hist(bins=30, figsize=(15, 5))
plt.show()
```

---

## Zusammenfassung: Workflow für Datenbereinigung

### Checkliste

1. **Daten laden**
   - [ ] CSV einlesen
   - [ ] Erste Zeilen anschauen (`head()`)
   - [ ] Info ausgeben (`info()`, `describe()`)

2. **Datenqualität prüfen**
   - [ ] Duplikate suchen und entfernen
   - [ ] Fehlende Werte zählen
   - [ ] Datentypen überprüfen

3. **String-Features parsen**
   - [ ] RAM extrahieren und in int konvertieren
   - [ ] Memory extrahieren und in GB konvertieren (TB → GB)
   - [ ] Weight extrahieren
   - [ ] Screen-Informationen extrahieren

4. **Feature Engineering**
   - [ ] Neue Features erstellen (z.B. Has_SSD, Screen_Pixels)
   - [ ] Einheiten konvertieren
   - [ ] Kategorien gruppieren (falls zu viele)

5. **Fehlende Werte behandeln**
   - [ ] Strategie wählen (entfernen, impute, etc.)
   - [ ] Target-Variable: Zeilen mit NaN entfernen
   - [ ] Features: Imputation oder Entfernung

6. **Ausreißer behandeln**
   - [ ] Visualisieren (Boxplots)
   - [ ] Identifizieren (IQR-Methode)
   - [ ] Entscheiden: Entfernen, cappen oder behalten

7. **Vorbereitung für Modellierung**
   - [ ] Features (X) und Target (y) trennen
   - [ ] Train-Test-Split
   - [ ] Kategorische Features behandeln (entfernen oder One-Hot)
   - [ ] Scaling (MinMaxScaler, StandardScaler)

8. **Validierung**
   - [ ] Shapes überprüfen
   - [ ] Wertebereiche plausibilisieren
   - [ ] Keine fehlenden Werte mehr in finalen Daten
   - [ ] Visualisierungen erstellen

---

## Häufige Fehler vermeiden

### Fehler 1: Fitting auf gesamtem Datensatz

```python
# FALSCH
scaler = MinMaxScaler()
df_scaled = scaler.fit_transform(df)  # Data Leakage!
X_train, X_test = train_test_split(df_scaled, ...)

# RICHTIG
X_train, X_test = train_test_split(df, ...)
scaler = MinMaxScaler()
scaler.fit(X_train)
X_train_scaled = scaler.transform(X_train)
X_test_scaled = scaler.transform(X_test)
```

### Fehler 2: Originale Spalten nicht entfernen

```python
# FALSCH: 'Ram' und 'Ram_GB' beide behalten
df['Ram_GB'] = df['Ram'].str.extract('(\d+)').astype(int)
# → 'Ram' ist String, kann nicht in Modell verwendet werden!

# RICHTIG
df['Ram_GB'] = df['Ram'].str.extract('(\d+)').astype(int)
df = df.drop('Ram', axis=1)
```

### Fehler 3: Fehlende Werte nicht behandeln

```python
# FALSCH: Direkt trainieren
model.fit(X_train, y_train)  # Fehler wenn NaN in X_train!

# RICHTIG: Zuerst imputieren
imputer = SimpleImputer(strategy='median')
X_train = imputer.fit_transform(X_train)
X_test = imputer.transform(X_test)
model.fit(X_train, y_train)
```

### Fehler 4: Index-Probleme nach Filtern

```python
# Problem: Nach dropna() stimmen Indices nicht mehr
X = df.drop('Price', axis=1)
y = df['Price']
df_clean = df.dropna(subset=['Weight'])  # Entfernt Zeilen
# → X und y haben jetzt unterschiedliche Indizes!

# Lösung 1: Index zurücksetzen
df_clean = df.dropna(subset=['Weight']).reset_index(drop=True)

# Lösung 2: Filtern vor dem Splitten
df = df.dropna(subset=['Weight', 'Price'])
X = df.drop('Price', axis=1)
y = df['Price']
```

---

## Zusätzliche Ressourcen

### Pandas String-Methoden
- `str.extract()`: Regex-basierte Extraktion
- `str.replace()`: Text ersetzen
- `str.contains()`: Suche nach Substring
- `str.split()`: Text aufteilen
- `str.strip()`: Whitespace entfernen

### NumPy für numerische Operationen
- `np.log()`, `np.log1p()`: Logarithmus-Transformation
- `np.sqrt()`: Quadratwurzel
- `np.nan`: Not a Number
- `np.isnan()`: NaN prüfen

### Sklearn Preprocessing
- `SimpleImputer`: Fehlende Werte füllen
- `MinMaxScaler`: Normalisierung auf [0, 1]
- `StandardScaler`: Standardisierung (Mean=0, Std=1)
- `OneHotEncoder`: Kategorische Variablen encodieren

---

**Viel Erfolg bei der Datenbereinigung! 🚀**

Bei Fragen oder Problemen, überprüfe immer:
1. Sind noch fehlende Werte vorhanden? (`df.isnull().sum()`)
2. Sind alle Datentypen korrekt? (`df.dtypes`)
3. Sind die Shapes konsistent? (`X_train.shape`, `y_train.shape`)
4. Wurde Data Leakage vermieden? (Fit nur auf Training!)
