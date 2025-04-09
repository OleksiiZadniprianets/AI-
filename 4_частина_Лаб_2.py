{
  "cells": [
    {
      "cell_type": "markdown",
      "source": [
        "**Задніпрянець О. ІН-401**"
      ],
      "metadata": {
        "id": "KDQ-0_pchrPA"
      }
    },
    {
      "cell_type": "markdown",
      "source": [
        "Лабораторна №2: 2.4 частина. Порівняння якості класифікаторів для набору даних завдання 2.1\n"
      ],
      "metadata": {
        "id": "cDym0dGrhtBw"
      }
    },
    {
      "cell_type": "code",
      "execution_count": 1,
      "metadata": {
        "id": "TD65GpM2g5g-"
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
        "id": "_4h_3-irg5hJ"
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
        "id": "iCoKlwTPg5hM"
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
        "id": "e9asGSO0g5hP"
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
        "id": "cnglsFDkg5hR"
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
        "id": "hrEUIIeOg5hT"
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
        "id": "_TB8ZLdJg5hW"
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
        "id": "I2s0DJwlg5hY"
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
        "id": "b91SWX4zg5ha",
        "outputId": "f3caa511-76c0-4b42-bb56-62b9c5ec0a63"
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
        "id": "Yv_C6fMcg5hn"
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
        "id": "ko8YfEkZg5hq"
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
        "id": "_mAK8ipgg5ht",
        "outputId": "c9ffd0ff-793f-4166-d7bd-5ca7e70fe337"
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
    },
    {
      "cell_type": "code",
      "execution_count": 13,
      "metadata": {
        "id": "RmfLFtjEg5hv"
      },
      "outputs": [],
      "source": [
        "from sklearn.multiclass import OneVsRestClassifier\n",
        "from sklearn.linear_model import LogisticRegression\n",
        "from sklearn.tree import DecisionTreeClassifier\n",
        "from sklearn.neighbors import KNeighborsClassifier\n",
        "from sklearn.discriminant_analysis import LinearDiscriminantAnalysis\n",
        "from sklearn.naive_bayes import GaussianNB\n",
        "from sklearn.svm import SVC\n",
        "from sklearn.model_selection import StratifiedKFold"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": 14,
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "T6Mww28_g5hw",
        "outputId": "5d22a7ae-d0d7-4bee-bca7-4fad391a10b5"
      },
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "Logistic Regression: 0.778938 (0.008510)\n",
            "Decision Tree: 0.799245 (0.008474)\n",
            "K-Nearest Neighbors: 0.745410 (0.007315)\n",
            "Linear Discriminant Analysis: 0.780927 (0.006017)\n",
            "Gaussian Naive Bayes: 0.797132 (0.006025)\n"
          ]
        }
      ],
      "source": [
        "models = {\n",
        "    'Logistic Regression': OneVsRestClassifier(LogisticRegression(solver='liblinear')),\n",
        "    'Decision Tree': DecisionTreeClassifier(random_state=0),\n",
        "    'K-Nearest Neighbors': KNeighborsClassifier(),\n",
        "    'Linear Discriminant Analysis': LinearDiscriminantAnalysis(),\n",
        "    'Gaussian Naive Bayes': GaussianNB(),\n",
        "    # 'Support Vector Classifier': SVC(gamma='auto') дуже повільно\n",
        "}\n",
        "\n",
        "results = []\n",
        "names = []\n",
        "\n",
        "for name, model in models.items():\n",
        "    kfold = StratifiedKFold(n_splits=10, shuffle=True, random_state=1)\n",
        "    cv_results = cross_val_score(model, X_train, y_train, cv=kfold, scoring='accuracy')\n",
        "    results.append(cv_results)\n",
        "    names.append(name)\n",
        "    msg = \"%s: %f (%f)\" % (name, cv_results.mean(), cv_results.std())\n",
        "    print(msg)"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": 15,
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 452
        },
        "id": "5IDgMPLfg5hy",
        "outputId": "4c218cee-8863-4c0d-fa60-4622e3677be0"
      },
      "outputs": [
        {
          "output_type": "display_data",
          "data": {
            "text/plain": [
              "<Figure size 640x480 with 1 Axes>"
            ],
            "image/png": "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rq66hvfU0WuyZWRkBC8vL8TGxkp/96msrAyxsbGYMmXKA5fdtm0biouLH5iBW1hYALg7aP748eNYsGBBrcVORETadL10VdMvd4CXrh5Go9HUqLepa9euMkRDer+MGBoaiqCgIHTs2BHe3t5YtmwZCgoKMH78eABAYGAgmjZtioiICK3lIiMj4e/vDysrqwptbtu2DY0bN0bz5s3x+++/Y9q0afD398cLL7zwWLaJiOhp9CiXrnTFS1dUn+g92Ro5ciSuXbuGsLAwZGVloUOHDoiJiZEGzWdkZECp1L5pMjU1FYcPH8bevXsrbTMzMxOhoaHIzs5GkyZNEBgYiLlz58q+LURETzNdL10lJycjICAAmzdvhpubm87rIqov9J5sAcCUKVOqvGwYFxdXoczFxaXKwe4AEBISgpCQkNoKj4iIqqGml67c3NzYS0VPNL1PakpERET0JGOyRURERCQjJltEREREMmKyRURERCQjJltEREREMmKyRURERCQjJltEREREMmKyRURERCQjJltEREREMmKyRURERCSjOvHneoiIqG46f/488vLyZGk7OTlZ61+5mJmZwdnZWdZ1ED0Iky0iIqrU+fPn0aZNG9nXExAQIPs6zp07x4SL9IbJFhERVaq8R2vz5s1wc3Or9faLioqQnp4OJycnqNXqWm8fuNtrFhAQIFvvHFF1MNkiIqIHcnNzg6enpyxtd+3aVZZ2ieoSJltENVBYWIiUlJRq16/p2BRXV1doNBqdliEiorqFyRZRDaSkpMDLy0vn5XQdm5KYmChbjwIRET0eTLaIasDV1RWJiYnVrl/TsSmurq41CY+IiOoQJltENaDRaHTuceLYFCKipxMnNSUiIiKSEZMtIiIiIhnxMqIeyTkzM8DZmYmIiOoCJlt68rhmZgY4OzMREZE+MdnSE7lnZgY4OzMREVFdwGRLz+ScmRngHXBERET6xgHyRERERDJiskVEREQkIyZbRERERDJiskVEREQkIw6QJyKiKtmZKqC+dQ74u37+NlffOgc7U4W+w6CnHJMtIiKq0qteRnA7+CpwUN+R1Iwb7m4DkT4x2SIioiqtSSzByLANcHN11XcoNZKckoI1i8dgsL4Doacaky0iIqpSVr5AUcM2gH0HfYdSI0VZZcjKF/oOg55y9fMiPBEREVE9USeSrRUrVsDJyQnGxsbw8fFB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n3ROtj7//HOkpaXB3t4ejo6OMDEx0XpdzjFIRERERPWNzslW+eUvIiIiIno4nZOt8PBwOeIgIiIieiIp9R0AERER0ZNM554tpVL5wPm0eKciERER0f/ROdnasWOH1vPbt2/j5MmT2LhxI+bNm1drgRERERE9CXROtoYMGVKhbNiwYXjmmWcQHR2N4ODgWgmMiIiI6ElQa2O2OnfujNjY2NpqjoiIiOiJUCvJVlFRET777DM0bdq0NpojIiIiemLofBnx/j84LYRAXl4eNBoNNm/eXKvBEREREdV3OidbS5cu1Uq2lEolGjduDB8fHzRq1KhWgyMiIiKq73ROtsaNGydDGERERERPJp3HbK1fvx7btm2rUL5t2zZs3LixVoIiIiIielLonGxFRETA2tq6QrmNjQ0WLlxYK0ERERERPSl0TrYyMjLQokWLCuWOjo7IyMioURArVqyAk5MTjI2N4ePjg4SEhCrr9uzZEwqFosJjwIABUp3KXlcoFPjkk09qFB8RERFRTek8ZsvGxganT5+Gk5OTVvmpU6dgZWWlcwDR0dEIDQ3F6tWr4ePjg2XLlsHPzw+pqamwsbGpUH/79u0oKSmRnt+4cQMeHh4YPny4VJaZmam1zI8//ojg4GC89NJLOsdHRETVU1hYiJSUlGrXT05O1vpXF66urtBoNDovR6QPOidbo0ePRkhICMzMzNC9e3cAwC+//IJp06Zh1KhROgewZMkSTJo0CePHjwcArF69Grt378a6deswa9asCvUtLS21nm/duhUajUYr2bKzs9Oq8/3336NXr15o2bKlzvEREVH1pKSkwMvLS+flAgICdF4mMTERnp6eOi9HpA86J1sLFixAeno6evfuDUPDu4uXlZUhMDBQ5zFbJSUlSExMxOzZs6UypVIJX19fxMfHV6uNyMhIjBo1CiYmJpW+np2djd27dz9w8H5xcTGKi4ul57m5udXcAiIiKufq6orExMRq1y8qKkJ6ejqcnJygVqt1XhdRfaFzsmVkZITo6Gi8//77SEpKglqthru7OxwdHXVe+fXr11FaWgpbW1utcltb22p1RSckJODMmTOIjIysss7GjRthZmaGF198sco6ERER/CPaRESPSKPR6Nzb1LVrV5miIao7dE62yjk7O8PZ2bk2Y9FZZGQk3N3d4e3tXWWddevWYezYsTA2Nq6yzuzZ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\n"
          },
          "metadata": {}
        }
      ],
      "source": [
        "plt.boxplot(results, tick_labels=names)\n",
        "plt.title('Algorithm Comparison')\n",
        "plt.ylabel('Accuracy')\n",
        "plt.show()"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "id": "pZfHjGDUg5hz"
      },
      "outputs": [],
      "source": []
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