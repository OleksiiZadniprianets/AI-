{
  "cells": [
    {
      "cell_type": "markdown",
      "source": [
        "**Задніпрянець О. ІН-401**"
      ],
      "metadata": {
        "id": "L3EWOmDwLMmE"
      }
    },
    {
      "cell_type": "markdown",
      "source": [
        "Лабораторна №2: 2 частина.\n",
        "Порівняння якості класифікаторів SVM з нелінійними ядрами\n"
      ],
      "metadata": {
        "id": "o2oo0b7fLP4V"
      }
    },
    {
      "cell_type": "code",
      "execution_count": 1,
      "metadata": {
        "id": "aOvlx0j0I8TF"
      },
      "outputs": [],
      "source": [
        "import numpy as np\n",
        "import matplotlib.pyplot as plt\n",
        "from sklearn import preprocessing\n",
        "from sklearn.svm import LinearSVC, SVC\n",
        "from sklearn.multiclass import OneVsRestClassifier\n",
        "from sklearn.model_selection import train_test_split, cross_val_score"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": 2,
      "metadata": {
        "id": "KIwbBHuVI8TO"
      },
      "outputs": [],
      "source": [
        "input_file = '/content/sample_data/income_data.txt'"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": 3,
      "metadata": {
        "id": "6F3P9CQDI8TQ"
      },
      "outputs": [],
      "source": [
        "X = []\n",
        "y = []\n",
        "count_class1 = 0\n",
        "count_class2 = 0\n",
        "max_datapoints = 25000"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": 4,
      "metadata": {
        "id": "XHCZF4dnI8TS"
      },
      "outputs": [],
      "source": [
        "with open(input_file, 'r') as f:\n",
        "    for line in f.readlines():\n",
        "        if count_class1 >= max_datapoints and count_class2 >= max_datapoints:\n",
        "            break\n",
        "\n",
        "        if '?' in line:\n",
        "            continue\n",
        "\n",
        "        data = line[:-1].split(',')\n",
        "        # print(data)\n",
        "\n",
        "        if data[-1] == ' <=50K' and count_class1 < max_datapoints:\n",
        "            X.append(data)\n",
        "            count_class1 += 1\n",
        "\n",
        "        if data[-1] == ' >50K' and count_class2 < max_datapoints:\n",
        "            X.append(data)\n",
        "            count_class2 += 1"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": 5,
      "metadata": {
        "id": "oCDlBkJ0I8TU"
      },
      "outputs": [],
      "source": [
        "X = np.array(X)"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": 6,
      "metadata": {
        "id": "seGAwFrrI8TV"
      },
      "outputs": [],
      "source": [
        "label_encoder = []\n",
        "X_encoded = np.empty(X.shape)\n",
        "for i, item in enumerate(X[0]):\n",
        "    if item.isdigit():\n",
        "        X_encoded[:, i] = X[:, i]\n",
        "    else:\n",
        "        label_encoder.append(preprocessing.LabelEncoder())\n",
        "        X_encoded[:, i] = label_encoder[-1].fit_transform(X[:, i])\n",
        "\n",
        "X = X_encoded[:, :-1].astype(int)\n",
        "y = X_encoded[:, -1].astype(int)"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": 7,
      "metadata": {
        "id": "LA2jpbKQI8TX"
      },
      "outputs": [],
      "source": [
        "classifiers = {\n",
        "    'Linear SVM': LinearSVC(random_state=0),\n",
        "    'RBF SVM': SVC(kernel='rbf', random_state=0),\n",
        "    'Poly SVM (degree=3)': SVC(kernel='poly', degree=3, random_state=0),\n",
        "    'Sigmoid SVM': SVC(kernel='sigmoid', random_state=0)\n",
        "}"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": 8,
      "metadata": {
        "id": "gvArIsUlI8TY"
      },
      "outputs": [],
      "source": [
        "from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": 9,
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "VFrpUvsjI8Ta",
        "outputId": "f829a902-3b79-4435-bf68-124e12dc5ff4"
      },
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "\n",
            "Classifier: Linear SVM\n",
            "Акуратність: 78.34%\n",
            "Точність: 76.88%\n",
            "Повнота: 78.34%\n",
            "F1: 74.32%\n",
            "----------------------------------------\n",
            "\n",
            "Classifier: RBF SVM\n",
            "Акуратність: 74.64%\n",
            "Точність: 55.71%\n",
            "Повнота: 74.64%\n",
            "F1: 63.80%\n",
            "----------------------------------------\n",
            "\n",
            "Classifier: Poly SVM (degree=3)\n",
            "Акуратність: 74.64%\n",
            "Точність: 55.71%\n",
            "Повнота: 74.64%\n",
            "F1: 63.80%\n",
            "----------------------------------------\n",
            "\n",
            "Classifier: Sigmoid SVM\n",
            "Акуратність: 63.82%\n",
            "Точність: 63.56%\n",
            "Повнота: 63.82%\n",
            "F1: 63.68%\n",
            "----------------------------------------\n"
          ]
        }
      ],
      "source": [
        "for clf_name, clf_model in classifiers.items():\n",
        "    print(f'\\nClassifier: {clf_name}')\n",
        "    clf = OneVsRestClassifier(clf_model)\n",
        "    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)\n",
        "    clf.fit(X_train, y_train)\n",
        "    y_test_pred = clf.predict(X_test)\n",
        "\n",
        "    accuracy = accuracy_score(y_test, y_test_pred)\n",
        "    precision = precision_score(y_test, y_test_pred, average='weighted', zero_division=0)\n",
        "    recall = recall_score(y_test, y_test_pred, average='weighted')\n",
        "    f1 = f1_score(y_test, y_test_pred, average='weighted')\n",
        "\n",
        "    print(f'Акуратність: {accuracy*100:.2f}%')\n",
        "    print(f'Точність: {precision*100:.2f}%')\n",
        "    print(f'Повнота: {recall*100:.2f}%')\n",
        "    print(f'F1: {f1*100:.2f}%')\n",
        "    print('-' * 40)"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": 10,
      "metadata": {
        "id": "v-kPFlCzI8Tg"
      },
      "outputs": [],
      "source": [
        "input_data = ['37', ' Private', ' 215646', ' HS-grad', ' 9', ' Never-married', ' Handlers-cleaners', ' Not-in-family', ' White', ' Male', ' 0', ' 0', ' 40', ' United-States']\n"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": 11,
      "metadata": {
        "id": "i13ewfYFI8Ti"
      },
      "outputs": [],
      "source": [
        "input_data_encoded = [-1] * len(input_data)\n",
        "count = 0\n",
        "for i, item in enumerate(input_data):\n",
        "    if item.isdigit():\n",
        "        input_data_encoded[i] = int(item)\n",
        "    else:\n",
        "        input_data_encoded[i] = label_encoder[count].transform([item])[0]\n",
        "        count += 1\n",
        "\n",
        "input_data_encoded = np.array(input_data_encoded).reshape(1, -1)"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": 12,
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "JjkD4RVpI8Tk",
        "outputId": "84747425-b34f-455a-ebf8-a5b16a69de78"
      },
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "\n",
            "Classifier: LinearSVC\n",
            " <=50K\n",
            "----------------------------------------\n",
            "\n",
            "Classifier: SVC\n",
            " <=50K\n",
            "----------------------------------------\n",
            "\n",
            "Classifier: SVC\n",
            " <=50K\n",
            "----------------------------------------\n",
            "\n",
            "Classifier: SVC\n",
            " <=50K\n",
            "----------------------------------------\n"
          ]
        }
      ],
      "source": [
        "for classifier in classifiers.values():\n",
        "    print(f'\\nClassifier: {classifier.__class__.__name__}')\n",
        "    classifier.fit(X, y)\n",
        "    predicted_class = classifier.predict(input_data_encoded)\n",
        "    print(label_encoder[-1].inverse_transform(predicted_class)[0])\n",
        "    print('-' * 40)"
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