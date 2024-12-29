import torch.nn as nn


class MNISTModelV2(nn.Module):
    """
    Конволюційна нейронна мережа (CNN) для набору даних MNIST.
    Ця модель складається з двох конволюційних блоків, після яких йде повнозв'язний шар для класифікації.
    
    Атрибути:
        conf_blok_1 (nn.Sequential): Перший конволюційний блок.
        conf_blok_2 (nn.Sequential): Другий конволюційний блок.
        classifier (nn.Sequential): Повнозв'язний шар для класифікації.
    """
    def __init__(self, input_shape, hidden_units, output_shape):
        """
        Ініціалізує MNISTModelV2.
        
        Аргументи:
            input_shape (int): Кількість вхідних каналів (наприклад, 1 для чорно-білих зображень MNIST).
            hidden_units (int): Кількість фільтрів у кожному з конволюційних шарів.
            output_shape (int): Кількість вихідних одиниць (кількість класів, наприклад, 10 для MNIST).
        """
        super().__init__()

        # Перший конволюційний блок
        self.conf_blok_1 = nn.Sequential(
            # Перший конволюційний шар
            nn.Conv2d(in_channels=input_shape, out_channels=hidden_units, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),  # Активаційна функція ReLU
            # Другий конволюційний шар
            nn.Conv2d(in_channels=hidden_units, out_channels=hidden_units, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),  # Активаційна функція ReLU
            # Пулинговий шар (зменшує розмір зображення вдвічі)
            nn.MaxPool2d(kernel_size=2)
        )

        # Другий конволюційний блок
        self.conf_blok_2 = nn.Sequential(
            # Третій конволюційний шар
            nn.Conv2d(in_channels=hidden_units, out_channels=hidden_units, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),  # Активаційна функція ReLU
            # Четвертий конволюційний шар
            nn.Conv2d(in_channels=hidden_units, out_channels=hidden_units, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),  # Активаційна функція ReLU
            # Пулинговий шар (зменшує розмір зображення ще вдвічі)
            nn.MaxPool2d(kernel_size=2)
        )

        # Класифікаційна частина (повнозв'язний шар)
        self.classifier = nn.Sequential(
            nn.Flatten(),  # Перетворює тензор з розмірності (N, C, H, W) у (N, C*H*W)
            nn.Linear(in_features=hidden_units * 7 * 7, out_features=output_shape),  # Лінійний шар для класифікації
        )
    
    def forward(self, x):
        """
        Виконує пряме проходження (forward pass) через модель.
        
        Аргументи:
            x (torch.Tensor): Вхідний тензор з розмірністю (batch_size, channels, height, width).
        
        Повертає:
            torch.Tensor: Вихід моделі (ймовірності або логіти для кожного класу).
        """
        x = self.conf_blok_1(x)  # Пропускаємо дані через перший конволюційний блок
        x = self.conf_blok_2(x)  # Пропускаємо дані через другий конволюційний блок
        x = self.classifier(x)   # Пропускаємо через класифікаційний шар
        return x
