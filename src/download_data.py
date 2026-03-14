import os
import urllib.request

def main():
    dossier_destination = "data/informer"
    os.makedirs(dossier_destination, exist_ok=True)

    url = "https://raw.githubusercontent.com/zhouhaoyi/ETDataset/main/ETT-small/ETTh1.csv"
    chemin_fichier = os.path.join(dossier_destination, "ETTh1.csv")
    
    try:
        urllib.request.urlretrieve(url, chemin_fichier)
    except Exception as e:
        print(f"Error : {e}")

if __name__ == "__main__":
    main()