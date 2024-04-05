from datetime import datetime
import math

imie = input("Podaj imie: ")
dateOfBirth = input("Podaj date urodzenia [DD-MM-YYYY]: ")
print(f"Witaj {imie} !")

current_date = datetime.now()

date = datetime(int(dateOfBirth[6:10]), int(dateOfBirth[3:5]), int(dateOfBirth[0:2]))

difference = current_date - date
days = difference.days
print(f"Przeżyłeś {days} dni")

fizyczna_fala = math.sin((2 * math.pi / 23) * days)
emocjonalna_fala = math.sin((2 * math.pi / 28) * days)
intelektualna_fala = math.sin((2 * math.pi / 33) * days)

def checkBiom(value, fala):
   if value > 0.5:
       return f"({value}) -> Wysoki poziom biorytmu. Gratulacje!!!"
   elif value < -0.5:
       if fala == "f": 
           value_jutro = math.sin((2 * math.pi / 23) * (days+1))
       elif fala == "e": 
           value_jutro = math.sin((2 * math.pi / 28) * (days+1))
       elif fala == "i": 
           value_jutro = math.sin((2 * math.pi / 33) * (days+1))
       if value_jutro > value:
           return f"({value}) -> Niski poziom biorytmu. Nie martw się. Jutro będzie lepiej!"
       else:
           return f"({value}) -> Niski poziom biorytmu. Nie martw się, każdy ma czasem gorszy dzień :)"
   else:
       return f"({value}) -> Średni poziom biorytmu"

print(f"Fala fizyczna: {checkBiom(fizyczna_fala , 'f')}")
print(f"Fala emocjonalna: {checkBiom(emocjonalna_fala , 'e')}")
print(f"Fala intelektualna: {checkBiom(intelektualna_fala , 'i')}")

# 35 min
