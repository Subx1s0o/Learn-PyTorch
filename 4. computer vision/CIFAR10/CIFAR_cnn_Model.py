from torch import nn

class CIFARModel(nn.Module):
    def __init__(self, input_shape, hidden_units, output_shape): 
        super().__init__()

        # Перший конволюційний блок
        self.conv_layer_1 = nn.Sequential( 
            nn.Conv2d(in_channels=input_shape, out_channels=hidden_units, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(hidden_units),
            nn.Conv2d(in_channels=hidden_units, out_channels=hidden_units, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(hidden_units),
            nn.MaxPool2d(kernel_size=2)
        )

        # Другий конволюційний блок
        self.conv_layer_2 = nn.Sequential(
            nn.Conv2d(in_channels=hidden_units, out_channels=hidden_units * 2, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(hidden_units * 2),
            nn.Conv2d(in_channels=hidden_units * 2, out_channels=hidden_units * 2, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(hidden_units * 2),
            nn.MaxPool2d(kernel_size=2)
        )

        # Третій конволюційний блок
        self.conv_layer_3 = nn.Sequential(
            nn.Conv2d(in_channels=hidden_units * 2, out_channels=hidden_units * 2, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(hidden_units * 2),
            nn.Conv2d(in_channels=hidden_units * 2, out_channels=hidden_units * 2, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(hidden_units * 2),
            nn.MaxPool2d(kernel_size=2)
        )

        # Класифікаційний блок з глобальним пулінгом
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),  # Глобальний Average Pooling
            nn.Flatten(),
            nn.Linear(hidden_units * 2, hidden_units * 4),  # Збільшено число нейронів
            nn.ReLU(),
            nn.Dropout(0.4),  # Збільшений dropout
            nn.Linear(hidden_units * 4, output_shape)
        )

    def forward(self, x):
        x = self.conv_layer_1(x)
        x = self.conv_layer_2(x)
        x = self.conv_layer_3(x)
        x = self.classifier(x)
        return x
