# Predicting a community in a social network

## Short description

Node prediction based on *email-Eu-core* dataset visualised below. The network consist of 986 nodes grouped into 37 groups.
Experiments:
- Graph Attention Network (GAT)
- Graph Convolutional Network (GCN) 

In addition a `node2vec` method is used to generate an extra information about each node.


## Opis

Klasyfikację przeprowadzono na zmodyfikowanej sieci *email-Eu-core* składającej się z 986 wierzchołków, przyporządkowanych do 37 grup. Zbiór danych reprezentuje konwersacje mailowe między pracownikami pewnej firmy.

## Modele poddane eksperymentom:

- Graph Attention Network (GAT)
- Graph Convolutional Network (GCN) 

Do wygenerowania cech wierzchołków wykorzystano metodę [`node2vec`](https://github.com/aditya-grover/node2vec).

## Zawartość

- `EDA.ipynb` - przeprowadzona analiza eksploracyjna oraz modyfikacja zbioru danych,
- `NodeEmbbeddings.ipynb` - tworzenie wektorów osadzeń dla wierzchołków grafu,
- `GAT.ipynb` - implementacja i ekperymenty z modelem GAT,
- `GCN.ipynb` - implementacja i ekperymenty z modelem GCN,
- `Baseline.ipynb` - implementacja i ekperymenty z modelem referencyjnym,
- `Summary.ipynb` - podsumowanie i analiza przeprowadzonych badań,
- `src` - funkcje umożliwiające przetwarzanie i wczytywanie danych oraz przeprowadzenie treningu sieci GAT i GCN,
- `create_env.sh` - skrypt tworzący środowisko z potrzebnymi zależnościami (bazuje na virtualenv),
- `results` - pliki csv z wynikami,
- `data` - zbiór danych, przetworzony i oryginalny,
- `images` - folder z obrazkami z badań.


![Network](images/network.png)




