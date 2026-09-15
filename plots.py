"""Raccoglie le quattro matrici di confusione in un solo file PNG."""

import matplotlib

# Agg salva il grafico senza aprire finestre, anche quando il programma
# viene eseguito da terminale su un computer privo di interfaccia grafica.
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def plot_confusion_matrices(metrics, output_dir):
    """Disegna SVM e LSTM ai livelli task e soggetto usando i conteggi reali."""
    plt.rcParams.update({"font.size": 10})
    figure, axes = plt.subplots(2, 2, figsize=(9, 8), constrained_layout=True)
    for axis, row in zip(axes.flat, metrics.itertuples(index=False)):
        # L'ordine delle classi e' sempre [0, 1]: la diagonale contiene
        # le classificazioni corrette; le altre due celle contengono gli errori.
        matrix = np.array([[row.tn, row.fp], [row.fn, row.tp]])
        axis.imshow(matrix, cmap="Blues", vmin=0, vmax=max(matrix.max(), 1))
        for (i, j), value in np.ndenumerate(matrix):
            text_color = "white" if value > matrix.max() / 2 else "#12263a"
            axis.text(j, i, str(value), ha="center", va="center", fontsize=18, color=text_color)
        level_name = "Task" if row.level == "task" else "Soggetti"
        axis.set(xticks=[0, 1], yticks=[0, 1], xticklabels=["Classe 0", "Classe 1"],
                 yticklabels=["Classe 0", "Classe 1"], xlabel="Classe predetta", ylabel="Classe di riferimento",
                 title=f"{row.model} - {level_name} (n={row.count})")
    figure.suptitle("Matrici di confusione sul test", fontsize=13)
    figure.savefig(output_dir / "confusion_matrices.png", dpi=160)
    plt.close(figure)  # Libera la figura dopo il salvataggio.
