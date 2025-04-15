
<h1 align="center">KI-gesteuerte Verkehrskreuzung</h1>

<div align="center">

[![Python version](https://img.shields.io/badge/python-3.1+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)

<h4>Eine Verkehrssimulation mit TD-Learning (Temporal Difference Learning) zur intelligenten Ampelsteuerung, basierend auf der originalen Pygame-Simulation von Mihir Gandhi.</h4>

</div>

-----------------------------------------
### Originalsimulation

Die Basis-Verkehrssimulation wurde von [Mihir Gandhi](https://github.com/mihir-m-gandhi) erstellt und ist unter [github.com/mihir-m-gandhi/Basic-Traffic-Intersection-Simulation](https://github.com/mihir-m-gandhi/Basic-Traffic-Intersection-Simulation) verfügbar. Diese Version erweitert die Simulation um KI-Funktionen.

### KI-Erweiterung - Was ist neu?

* **TD-Learning für Ampelsteuerung**: Die Ampeln werden jetzt durch ein KI-Modell gesteuert, das lernt, Verkehrsfluss zu optimieren und CO2-Emissionen zu reduzieren.
* **State/Action System**: Die KI analysiert Fahrzeugdichten in alle Richtungen, um optimale Ampelphasen zu bestimmen.
* **Reward-Mechanismus**: Die KI wird belohnt für mehr durchgekommene Fahrzeuge und weniger CO2-Ausstoß.
* **Autonomes Lernen**: Das Modell verbessert sich selbst durch wiederholtes Trainieren in der Simulation.

------------------------------------------
### KI-Ausführung

#### Training starten
```sh
# Modell mit 50 Episoden trainieren (Standard)
python td_learning.py --mode train

# Fortgeschrittene Optionen
python td_learning.py --mode train --episodes 100 --ticks 50000 --save_interval 5
```

#### Trainiertes Modell bewerten
```sh
# Modell bewerten (5 Test-Läufe)
python td_learning.py --mode evaluate --model models/td_model_final.pkl

# Mit angepassten Ticks pro Bewertungslauf
python td_learning.py --mode evaluate --model models/td_model_final.pkl --ticks 50000
```

#### Modell visuell anzeigen
```sh
# Trainiertes Modell in der visuellen Simulation ausführen
python td_learning.py --mode run --model models/td_model_final.pkl
```

------------------------------------------
### Original-Demo

<p align="center">
    <img src="./Demo.gif">
</p>

------------------------------------------
### Installation

 * Step I: Repository klonen
  * Step II: Benötigte Pakete installieren
```sh
      # Terminal öffnen im Projektverzeichnis
      $ pip install pygame numpy matplotlib
```
* Step III: Code ausführen
```sh
      # Normale Simulation starten
      $ python simulation.py
      
      # Trainieren eines neuen KI-Modells
      $ python td_learning.py --mode train -episodes 100 --ticks 100000    
      
      # Vortrainiertes Modell anzeigen
      $ python td_learning.py --mode run --model models/td_model_final.pkl
```

------------------------------------------
### Wie funktioniert's?

1. **State-Erkennung**: Die KI analysiert, wie viele Fahrzeuge in jeder Richtung warten
2. **Decision-Making**: Basierend auf dem Q-Learning-Algorithmus entscheidet die KI, welche Ampel grün sein soll
3. **Reward-System**: Die KI wird belohnt für:
   - Mehr Fahrzeuge, die die Kreuzung passieren
   - Weniger CO2-Ausstoß durch unnötiges Warten/Bremsen
4. **Lernprozess**: Mit jeder Episode verbessert die KI ihre Entscheidungen

------------------------------------------
### Credits

* Originale Verkehrssimulation: [Mihir Gandhi](https://github.com/mihir-m-gandhi)
* TD-Learning-Implementierung: Marian Rickert (rickertmar), David Wolf (davidaew), Ian Schnitzke (schniti269)

------------------------------------------
### Lizenz
Dieses Projekt steht unter der MIT-Lizenz - siehe die [LICENSE](./LICENSE) Datei für Details.
