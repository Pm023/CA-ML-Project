# modules/visualization.py
import matplotlib.pyplot as plt

def plot_comparison(baseline, green, ylabel, title, labels=["Baseline","Green"], colors=["red","green"]):
    plt.figure(figsize=(8,5))
    plt.bar(labels, [baseline, green], color=colors)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.show()

def scatter_accuracy_vs_co2(acc_list, co2_list, labels_list=["Baseline","Green"]):
    plt.figure(figsize=(8,5))
    plt.scatter(co2_list, acc_list, s=100, c=["red","green"])
    for i, txt in enumerate(labels_list):
        plt.annotate(txt, (co2_list[i], acc_list[i]), textcoords="offset points", xytext=(5,5))
    plt.xlabel("CO2 Emission (kg)")
    plt.ylabel("R2 Accuracy")
    plt.title("Accuracy vs CO2 Emission")
    plt.show()
