Beispiel:
Matrix Multiplikation

Kompilieren:
. make.sh

Mit Problemgr��e:
N=256 ./make.sh

Kompilieren f�r unterschiedliche Devices und Problemgr��en:
  make
  
Ausf�hrung und generierung der Vergleichsdatei:
  ./run.sh >> vergleich.txt
  
Erkl�rung:

Das Makefile erstellt Testprogramme in 3 Problemgr��en N (1024, 2048 und 4096) und 2 verschiedene Devices.
Das Script 'run.sh' f�hrt alle Tests nacheinander aus. Die Ausf�hrung kann allerdings relativ lange dauern!