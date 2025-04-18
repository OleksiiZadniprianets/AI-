{
  "cells": [
    {
      "cell_type": "code",
      "execution_count": 1,
      "metadata": {
        "id": "lAHj8BMEYgp2"
      },
      "outputs": [],
      "source": [
        "import numpy as np\n",
        "from sklearn.datasets import load_iris\n",
        "from sklearn.linear_model import RidgeClassifier\n",
        "from sklearn.model_selection import train_test_split"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": 2,
      "metadata": {
        "id": "WqUtPBMRYgp9"
      },
      "outputs": [],
      "source": [
        "iris = load_iris()"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": 3,
      "metadata": {
        "id": "6EZ069QXYgp-"
      },
      "outputs": [],
      "source": [
        "X, y = iris.data, iris.target"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": 4,
      "metadata": {
        "id": "0DQWGQ6AYgp_"
      },
      "outputs": [],
      "source": [
        "X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": 5,
      "metadata": {
        "id": "RvrDtn6rYgqA"
      },
      "outputs": [],
      "source": [
        "clf = RidgeClassifier(tol=1e-2, solver=\"sag\")\n",
        "clf.fit(X_train, y_train)\n",
        "y_pred = clf.predict(X_test)"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": 6,
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "kB5ShpfQYgqB",
        "outputId": "3fbcf3c9-6a3d-4f92-809d-e491e7d02d27"
      },
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "Accuracy: 0.7555555555555555\n",
            "Precision: 0.8333333333333334\n",
            "Recall: 0.7555555555555555\n",
            "F1 Score: 0.7502986857825567\n",
            "Cohen's Kappa: 0.6431146359048305\n",
            "Matthews Correlation Coefficient: 0.6830664115990899\n",
            "\t\tClassification Report:\n",
            "               precision    recall  f1-score   support\n",
            "\n",
            "           0       1.00      1.00      1.00        16\n",
            "           1       0.89      0.44      0.59        18\n",
            "           2       0.50      0.91      0.65        11\n",
            "\n",
            "    accuracy                           0.76        45\n",
            "   macro avg       0.80      0.78      0.75        45\n",
            "weighted avg       0.83      0.76      0.75        45\n",
            "\n"
          ]
        }
      ],
      "source": [
        "from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, cohen_kappa_score, matthews_corrcoef, classification_report\n",
        "\n",
        "print(\"Accuracy:\", accuracy_score(y_test, y_pred))\n",
        "print(\"Precision:\", precision_score(y_test, y_pred, average='weighted'))\n",
        "print(\"Recall:\", recall_score(y_test, y_pred, average='weighted'))\n",
        "print(\"F1 Score:\", f1_score(y_test, y_pred, average='weighted'))\n",
        "print(\"Cohen's Kappa:\", cohen_kappa_score(y_test, y_pred))\n",
        "print(\"Matthews Correlation Coefficient:\", matthews_corrcoef(y_test, y_pred))\n",
        "print(\"\\t\\tClassification Report:\\n\", classification_report(y_test, y_pred))"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": 7,
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 489
        },
        "id": "RZVKfVYnYgqE",
        "outputId": "e548c719-04ca-46cd-ebab-55bd15ce34e3"
      },
      "outputs": [
        {
          "output_type": "execute_result",
          "data": {
            "text/plain": [
              "Text(0.5, 1.0, 'Confusion Matrix')"
            ]
          },
          "metadata": {},
          "execution_count": 7
        },
        {
          "output_type": "display_data",
          "data": {
            "text/plain": [
              "<Figure size 640x480 with 1 Axes>"
            ],
            "image/png": "iVBORw0KGgoAAAANSUhEUgAAAbAAAAHHCAYAAADeaQ1TAAAAOnRFWHRTb2Z0d2FyZQBNYXRwbG90bGliIHZlcnNpb24zLjEwLjAsIGh0dHBzOi8vbWF0cGxvdGxpYi5vcmcvlHJYcgAAAAlwSFlzAAAPYQAAD2EBqD+naQAALiZJREFUeJzt3Xd0FOXixvFnQ8ImhBQgCSEKoVlAEEQCQpCIqFhAARtwlQTECogEEBtdjJUANhS9NMGDiqKI/kQSFCNBmoQmTUG8UkKRUEJJmd8fHPbeJQR2w242L/l+zsk52XdmZ57NXO/D7Lw7a7MsyxIAAIbx83UAAABKggIDABiJAgMAGIkCAwAYiQIDABiJAgMAGIkCAwAYiQIDABiJAgMAGIkCA1ywZcsW3XLLLQoLC5PNZtPcuXM9uv3t27fLZrNp6tSpHt2uyW644QbdcMMNvo6BMowCgzF+//13Pfroo6pbt64CAwMVGhqq+Ph4TZgwQceOHfPqvhMTE7V27VqNHTtWM2bMUPPmzb26v9KUlJQkm82m0NDQs/4dt2zZIpvNJpvNptdff93t7e/cuVMjR47U6tWrPZAW+C9/XwcAXDF//nzde++9stvt6tmzpxo1aqSTJ08qIyNDQ4YM0fr16/X+++97Zd/Hjh1TZmamnn/+efXr188r+4iNjdWxY8cUEBDgle2fj7+/v3JzczVv3jzdd999TstmzpypwMBAHT9+vETb3rlzp0aNGqXatWuradOmLj9vwYIFJdofyg8KDGXetm3b1K1bN8XGxio9PV01atRwLOvbt6+2bt2q+fPne23/e/fulSSFh4d7bR82m02BgYFe2/752O12xcfH6+OPPy5SYLNmzdIdd9yhOXPmlEqW3NxcVapUSRUrViyV/cFcvIWIMu/VV1/VkSNH9OGHHzqV12n169fXgAEDHI/z8/M1ZswY1atXT3a7XbVr19Zzzz2nEydOOD2vdu3a6tixozIyMtSiRQsFBgaqbt26mj59umOdkSNHKjY2VpI0ZMgQ2Ww21a5dW9Kpt95O//6/Ro4cKZvN5jT2/fffq02bNgoPD1flypV1xRVX6LnnnnMsL+4aWHp6uq6//noFBwcrPDxcd911l3777bez7m/r1q1KSkpSeHi4wsLC1KtXL+Xm5hb/hz1Djx499O233+rgwYOOseXLl2vLli3q0aNHkfUPHDigwYMHq3HjxqpcubJCQ0N12223KSsry7HODz/8oLi4OElSr169HG9Fnn6dN9xwgxo1aqSVK1eqbdu2qlSpkuPvcuY1sMTERAUGBhZ5/R06dFCVKlW0c+dOl18rLg4UGMq8efPmqW7dumrdurVL6/fp00fDhw9Xs2bNlJqaqoSEBKWkpKhbt25F1t26davuuece3XzzzXrjjTdUpUoVJSUlaf369ZKkrl27KjU1VZLUvXt3zZgxQ+PHj3cr//r169WxY0edOHFCo0eP1htvvKE777xTP//88zmft3DhQnXo0EHZ2dkaOXKkkpOTtWTJEsXHx2v79u1F1r/vvvt0+PBhpaSk6L777tPUqVM1atQol3N27dpVNptNn3/+uWNs1qxZuvLKK9WsWbMi6//xxx+aO3euOnbsqHHjxmnIkCFau3atEhISHGXSoEEDjR49WpL0yCOPaMaMGZoxY4batm3r2M7+/ft12223qWnTpho/frzatWt31nwTJkxQZGSkEhMTVVBQIEl67733tGDBAr355puKiYlx+bXiImEBZVhOTo4lybrrrrtcWn/16tWWJKtPnz5O44MHD7YkWenp6Y6x2NhYS5K1ePFix1h2drZlt9utQYMGOca2bdtmSbJee+01p20mJiZasbGxRTKMGDHC+t//tFJTUy1J1t69e4vNfXofU6ZMcYw1bdrUioqKsvbv3+8Yy8rKsvz8/KyePXsW2V/v3r2dttmlSxerWrVqxe7zf19HcHCwZVmWdc8991jt27e3LMuyCgoKrOjoaGvUqFFn/RscP37cKigoKPI67Ha7NXr0aMfY8uXLi7y20xISEixJ1qRJk866LCEhwWnsu+++syRZL774ovXHH39YlStXtjp37nze14iLE2dgKNMOHTokSQoJCXFp/W+++UaSlJyc7DQ+aNAgSSpyraxhw4a6/vrrHY8jIyN1xRVX6I8//ihx5jOdvnb25ZdfqrCw0KXn7Nq1S6tXr1ZSUpKqVq3qGL/66qt18803O17n/3rsscecHl9//fXav3+/42/oih49euiHH37Q7t27lZ6ert27d5/17UPp1HUzP79T/xdSUFCg/fv3O94eXbVqlcv7tNvt6tWrl0vr3nLLLXr00Uc1evRode3aVYGBgXrvvfdc3hcuLhQYyrTQ0FBJ0uHDh11a/88//5Sfn5/q16/vNB4dHa3w8HD9+eefTuO1atUqso0qVaron3/+KWHiou6//37Fx8erT58+ql69urp166ZPPvnknGV2OucVV1xRZFmDBg20b98+HT161Gn8zNdSpUoVSXLrtdx+++0KCQnR7NmzNXPmTMXFxRX5W55WWFio1NRUXXbZZbLb7YqIiFBkZKTWrFmjnJwcl/d5ySWXuDVh4/XXX1fVqlW1evVqTZw4UVFRUS4/FxcXCgxlWmhoqGJiYrRu3Tq3nnfmJIriVKhQ4azjlmWVeB+nr8+cFhQUpMWLF2vhwoV68MEHtWbNGt1///26+eabi6x7IS7ktZxmt9vVtWtXTZs2TV988UWxZ1+S9NJLLyk5OVlt27bVRx99pO+++07ff/+9rrrqKpfPNKVTfx93/Prrr8rOzpYkrV271q3n4uJCgaHM69ixo37//XdlZmaed93Y2FgVFhZqy5YtTuN79uzRwYMHHTMKPaFKlSpOM/ZOO/MsT5L8/PzUvn17jRs3Ths2bNDYsWOVnp6uRYsWnXXbp3Nu2rSpyLKNGzcqIiJCwcHBF/YCitGjRw/9+uuvOnz48Fknvpz22WefqV27dvrwww/VrVs33XLLLbrpppuK/E1c/ceEK44ePapevXqpYcOGeuSRR/Tqq69q+fLlHts+zEKBocx7+umnFRwcrD59+mjPnj1Flv/++++aMGGCpFNvgUkqMlNw3LhxkqQ77rjDY7nq1aunnJwcrVmzxjG2a9cuffHFF07rHThwoMhzT3+g98yp/afVqFFDTZs21bRp05wKYd26dVqwYIHjdXpDu3btNGbMGL311luKjo4udr0KFSoUObv79NNP9ffffzuNnS7as5W9u4YOHaodO3Zo2rRpGjdunGrXrq3ExMRi/464uPFBZpR59erV06xZs3T//ferQYMGTnfiWLJkiT799FMlJSVJkpo0aaLExES9//77OnjwoBISErRs2TJNmzZNnTt3LnaKdkl069ZNQ4cOVZcuXfTkk08qNzdX7777ri6//HKnSQyjR4/W4sWLdccddyg2NlbZ2dl65513dOmll6pNmzbFbv+1117TbbfdplatWumhhx7SsWPH9OabbyosLEwjR4702Os4k5+fn1544YXzrtexY0eNHj1avXr1UuvWrbV27VrNnDlTdevWdVqvXr16Cg8P16RJkxQSEqLg4GC1bNlSderU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\n"
          },
          "metadata": {}
        }
      ],
      "source": [
        "from sklearn.metrics import confusion_matrix\n",
        "import seaborn as sns\n",
        "import matplotlib.pyplot as plt\n",
        "# from io import BytesIO\n",
        "\n",
        "mat = confusion_matrix(y_test, y_pred)\n",
        "sns.heatmap(mat.T, square = True, annot = True, fmt = 'd', cbar = False)\n",
        "plt.xlabel('True label')\n",
        "plt.ylabel('Predicted label')\n",
        "plt.title('Confusion Matrix')"
      ]
    }
  ],
  "metadata": {
    "kernelspec": {
      "display_name": "base",
      "language": "python",
      "name": "python3"
    },
    "language_info": {
      "codemirror_mode": {
        "name": "ipython",
        "version": 3
      },
      "file_extension": ".py",
      "mimetype": "text/x-python",
      "name": "python",
      "nbconvert_exporter": "python",
      "pygments_lexer": "ipython3",
      "version": "3.12.7"
    },
    "colab": {
      "provenance": []
    }
  },
  "nbformat": 4,
  "nbformat_minor": 0
}