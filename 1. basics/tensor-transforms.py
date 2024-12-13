import torch 

# Створюємо тензор розміром (1, 9), що містить значення від 1 до 9
x = torch.arange(1., 10.)

# Перероблюємо розмір тензора на (9, 1). Новий розмір має дорівнювати числу елементів у попередньому тензорі.
# Тобто ми перетворюємо вектор з 9 елементів в стовпчик з 9 рядків по одному елементу
x_reshaped = x.reshape(9, 1)

# Також можна перетворити на 3 на 3. Головне що б розмір дорівнював початковому. 9*1=9, 3*3 = 9
x_reshaped = x.reshape(3, 3)

# reshape() може створити нову копію з новою пам'яттю, що дозволяє змінювати форму без впливу на оригінальний тензор.
# view() працює без створення нової пам'яті і використовує те ж саме місце в пам'яті, що й оригінальний тензор, але тільки якщо тензор неперервний.
z = x.view(3,3)

# torch.stack() об'єднує кілька тензорів в один по певній осі (вимірі).
# dim=0: Об'єднує тензори вертикально (нові рядки додаються до результату).
# dim=1: Об'єднує тензори горизонтально (нові стовпці додаються поруч з іншими).
# Замість того, щоб модифікувати оригінальні тензори, ця операція створює новий тензор з об'єднаними елементами.
stacked = torch.stack([x,x,x,x],dim=1)
