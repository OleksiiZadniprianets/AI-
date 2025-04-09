{
  "cells": [
    {
      "cell_type": "markdown",
      "source": [
        "**Задніпрянець О. ІН-401**"
      ],
      "metadata": {
        "id": "IUxywWOPUcG2"
      }
    },
    {
      "cell_type": "markdown",
      "source": [
        "Лабораторна №2: 3.1 частина.\n",
        "Порівняння якості класифікаторів на прикладі\n",
        "класифікації сортів ірисів. КРОК 1. ЗАВАНТАЖЕННЯ ТА ВИВЧЕННЯ ДАНИХ\n"
      ],
      "metadata": {
        "id": "9JJbqblkUesM"
      }
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "id": "FHj2Leu5UHex"
      },
      "outputs": [],
      "source": [
        "from sklearn.datasets import load_iris\n",
        "iris_dataset = load_iris()"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "mja8yaCFUHe2",
        "outputId": "73b3a8ca-b1e8-4912-ac60-0e3a1894ecb1"
      },
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "Ключі iris dataset: \n",
            "dict_keys(['data', 'target', 'frame', 'target_names', 'DESCR', 'feature_names', 'filename', 'data_module'])\n"
          ]
        }
      ],
      "source": [
        "print('Ключі iris dataset: \\n{}'.format(iris_dataset.keys()))"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "76KA2GpvUHe5",
        "outputId": "ef795b82-579f-4330-a794-b5053a15ba21"
      },
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            ".. _iris_dataset:\n",
            "\n",
            "Iris plants dataset\n",
            "--------------------\n",
            "\n",
            "**Data Set Characteristics:**\n",
            "\n",
            ":Number of Instances: 150 (50 in each of three classes)\n",
            ":Number of Attributes: 4 numeric, predictive \n",
            "...\n"
          ]
        }
      ],
      "source": [
        "print(iris_dataset['DESCR'][:193] + \"\\n...\")"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "fXNLbehzUHe6",
        "outputId": "4619751c-c3c7-4e7f-8b10-a5c082569600"
      },
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "Назви відповідей: ['setosa' 'versicolor' 'virginica']\n"
          ]
        }
      ],
      "source": [
        "print('Назви відповідей: {}'.format(iris_dataset['target_names']))"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "715gN1PCUHe7",
        "outputId": "9572d8bb-c84e-40cb-be1f-8ce41649a6f3"
      },
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "Назви ознак: ['sepal length (cm)', 'sepal width (cm)', 'petal length (cm)', 'petal width (cm)']\n"
          ]
        }
      ],
      "source": [
        "print('Назви ознак: {}'.format(iris_dataset['feature_names']))"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "8OpHYw5jUHe8",
        "outputId": "87d41275-ddb9-4af6-a305-68ce8fdada94"
      },
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "Тип масиву data: <class 'numpy.ndarray'>\n"
          ]
        }
      ],
      "source": [
        "print('Тип масиву data: {}'.format(type(iris_dataset['data'])))"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "lW6N6aPdUHe8",
        "outputId": "b74346f5-d171-4327-d4fe-e2023bb2a866"
      },
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "Форма масиву data: (150, 4)\n"
          ]
        }
      ],
      "source": [
        "print('Форма масиву data: {}'.format(iris_dataset['data'].shape))"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "aBKRUSizUHe9",
        "outputId": "186a0641-dc61-43c9-e8a0-4f731f616cf0"
      },
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "Перші п’ять рядків масиву data:\n",
            "[[5.1 3.5 1.4 0.2]\n",
            " [4.9 3.  1.4 0.2]\n",
            " [4.7 3.2 1.3 0.2]\n",
            " [4.6 3.1 1.5 0.2]\n",
            " [5.  3.6 1.4 0.2]]\n"
          ]
        }
      ],
      "source": [
        "print('Перші п’ять рядків масиву data:\\n{}'.format(iris_dataset['data'][:5]))"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "zCQD5jz5UHe-",
        "outputId": "9bb743a0-29a2-4b04-d09b-bacec68d1bb2"
      },
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "Тип масиву target: <class 'numpy.ndarray'>\n"
          ]
        }
      ],
      "source": [
        "print('Тип масиву target: {}'.format(type(iris_dataset['target'])))"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "G-AAcyAUUHe_",
        "outputId": "25d35543-3e3f-4282-eeba-9f20927e3e44"
      },
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "Відповіді: [0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0\n",
            " 0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1\n",
            " 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 2 2 2 2 2 2 2 2 2 2 2\n",
            " 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2\n",
            " 2 2]\n"
          ]
        }
      ],
      "source": [
        "print('Відповіді: {}'.format(iris_dataset['target']))"
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