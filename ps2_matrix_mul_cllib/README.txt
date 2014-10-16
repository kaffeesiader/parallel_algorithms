Beispiel:
Matrix Multiplikation

Kompilieren:
. make.sh

Mit Problemgröße:
N=256 ./make.sh

Kompilieren für unterschiedliche Devices und Problemgrößen:
  make
  
Ausführung und generierung der Vergleichsdatei:
  ./run.sh >> vergleich.txt
  
Erklärung:

Das Makefile erstellt Testprogramme in 3 Problemgrößen N (1024, 2048 und 4096) und 2 verschiedene Devices.
Das Script 'run.sh' führt alle Tests nacheinander aus. Die Ausführung kann allerdings relativ lange dauern!