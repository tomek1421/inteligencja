from datetime import datetime
import math

# Obliczenie dni życia
def calculate_days_of_life(date_of_birth):
   current_date = datetime.now()
   difference = current_date - date_of_birth
   return difference.days

# Obliczenie biorytmów fizycznych, emocjonalnych i intelektualnych
def calculate_biorhythms(days):
   physical_cycle = days % 23
   emotional_cycle = days % 28
   intellectual_cycle = days % 33
   physical_rhythm = round(math.sin((2 * math.pi / 23) * physical_cycle), 2)
   emotional_rhythm = round(math.sin((2 * math.pi / 28) * emotional_cycle), 2)
   intellectual_rhythm = round(math.sin((2 * math.pi / 33) * intellectual_cycle), 2)
   return physical_rhythm, emotional_rhythm, intellectual_rhythm

# Sprawdzenie wyników biorytmów
def check_biorhythm(rhythm):
   if rhythm > 0.5:
       return "Wysoki poziom biorytmu. Gratulacje!"
   elif rhythm < -0.5:
       return "Niski poziom biorytmu. Pociesz się, jutro będzie lepiej!"
   else:
       return "Średni poziom biorytmu."

# Główna funkcja programu
def main():
   name = input("Podaj swoje imię: ")
   year = int(input("Podaj rok urodzenia: "))
   month = int(input("Podaj miesiąc urodzenia: "))
   day = int(input("Podaj dzień urodzenia: "))

   date_of_birth = datetime(year, month, day)
   days_of_life = calculate_days_of_life(date_of_birth)
   physical_rhythm, emotional_rhythm, intellectual_rhythm = calculate_biorhythms(days_of_life)

   print(f"Witaj {name}!")
   print(f"Dziś jest {days_of_life}-ty dzień twojego życia.")
   print("Twoje biorytmy:")
   print(f"Fizyczna fala: {physical_rhythm} - {check_biorhythm(physical_rhythm)}")
   print(f"Emocjonalna fala: {emotional_rhythm} - {check_biorhythm(emotional_rhythm)}")
   print(f"Intelektualna fala: {intellectual_rhythm} - {check_biorhythm(intellectual_rhythm)}")

if __name__ == "__main__":
   main()