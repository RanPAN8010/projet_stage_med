import asyncio
from bleak import BleakScanner

async def main():
    print("Scan des appareils Bluetooth à proximité en cours, veuillez appuyer sur le bouton blanc de l'oxymètre...")
    devices = await BleakScanner.discover()
    for d in devices:
        # Recherche des appareils dont le nom contient Libelium ou MySignals
        if d.name and ("Libelium" in d.name or "MySignals" in d.name):
            print(f"Oxymètre détecté ! Nom de l'appareil : {d.name}, Adresse MAC : {d.address}")

asyncio.run(main())